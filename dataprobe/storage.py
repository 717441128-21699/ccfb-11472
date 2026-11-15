"""SQLite 元数据存储层。

画像明细以 JSON 存于 *_daily 表，检索型字段（空值率、基数等）冗余成列，
方便管理页面直接排序筛选。所有写入按 (表, 日期) 幂等，可安全重跑。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from .config import DEFAULT_THRESHOLDS, META_DB_PATH, THRESHOLD_SCHEMA

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id    TEXT UNIQUE NOT NULL,
    name        TEXT UNIQUE NOT NULL,
    kind        TEXT NOT NULL,
    path        TEXT NOT NULL,
    table_name  TEXT,
    note        TEXT DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS table_daily (
    table_id      TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    row_count     INTEGER NOT NULL,
    column_count  INTEGER NOT NULL,
    weekday       INTEGER NOT NULL,
    profile_json  TEXT NOT NULL,
    PRIMARY KEY (table_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS column_daily (
    table_id      TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    column_name   TEXT NOT NULL,
    stable_col_id TEXT,
    ordinal       INTEGER NOT NULL,
    inferred_type TEXT NOT NULL,
    null_count    INTEGER NOT NULL,
    null_rate     REAL NOT NULL,
    distinct_count INTEGER NOT NULL,
    distinct_ratio REAL NOT NULL,
    profile_json  TEXT NOT NULL,
    PRIMARY KEY (table_id, snapshot_date, column_name)
);
CREATE INDEX IF NOT EXISTS idx_col_identity ON column_daily(table_id, stable_col_id, snapshot_date);

CREATE TABLE IF NOT EXISTS column_identities (
    table_id         TEXT NOT NULL,
    stable_col_id    TEXT NOT NULL,
    first_seen_date  TEXT NOT NULL,
    original_name    TEXT NOT NULL,
    current_name     TEXT NOT NULL,
    original_ordinal INTEGER NOT NULL,
    current_ordinal  INTEGER NOT NULL,
    inferred_type    TEXT,
    fingerprint_json TEXT,
    last_seen_date   TEXT NOT NULL,
    PRIMARY KEY (table_id, stable_col_id)
);

CREATE TABLE IF NOT EXISTS events (
    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id       TEXT NOT NULL,
    stable_col_id  TEXT,
    column_name    TEXT,
    snapshot_date  TEXT NOT NULL,
    metric         TEXT NOT NULL,
    severity       TEXT NOT NULL,
    value          REAL,
    threshold      REAL,
    baseline_dates TEXT,
    message        TEXT NOT NULL,
    signature      TEXT NOT NULL,
    suppressed     INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'open',
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_date ON events(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_events_table ON events(table_id, snapshot_date);

CREATE TABLE IF NOT EXISTS feedback_signatures (
    signature  TEXT PRIMARY KEY,
    table_id   TEXT,
    metric     TEXT,
    note       TEXT DEFAULT '',
    hit_count  INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    last_hit_at TEXT
);

CREATE TABLE IF NOT EXISTS threshold_overrides (
    scope      TEXT NOT NULL,
    table_id   TEXT,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (scope, table_id, key)
);
"""


def connect(db_path: str | None = None) -> sqlite3.Connection:
    path = str(db_path or META_DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection, reset: bool = False) -> None:
    if reset:
        # 同一连接内清理，避免 WAL 残留旧表结构
        for tbl in ("events", "column_daily", "table_daily", "column_identities",
                    "feedback_signatures", "threshold_overrides", "sources"):
            conn.execute(f"DROP TABLE IF EXISTS {tbl}")
        conn.commit()
    conn.executescript(_SCHEMA)
    conn.commit()


# ---------------- 来源登记 ----------------

def upsert_source(conn: sqlite3.Connection, *, name: str, kind: str, path: str,
                  table_name: str | None, note: str = "") -> dict[str, Any]:
    kind = kind.lower()
    table_id = f"{name}__{(table_name or 'main')}"
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """INSERT INTO sources(table_id, name, kind, path, table_name, note, created_at)
           VALUES(?,?,?,?,?,?,?)
           ON CONFLICT(name) DO UPDATE SET
             table_id=excluded.table_id, kind=excluded.kind, path=excluded.path,
             table_name=excluded.table_name, note=excluded.note""",
        (table_id, name, kind, path, table_name, note, now),
    )
    conn.commit()
    return get_source(conn, name)


def list_sources(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM sources ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def get_source(conn: sqlite3.Connection, name: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM sources WHERE name=?", (name,)).fetchone()
    return dict(row) if row else None


def delete_source(conn: sqlite3.Connection, name: str) -> None:
    conn.execute("DELETE FROM sources WHERE name=?", (name,))
    conn.commit()


# ---------------- 画像写入（按 表+日期 幂等） ----------------

def replace_daily_snapshot(conn: sqlite3.Connection, *, table_id: str, snapshot_date: str,
                           row_count: int, weekday: int, table_profile: dict,
                           column_rows: list[dict]) -> None:
    with conn:
        conn.execute("DELETE FROM table_daily WHERE table_id=? AND snapshot_date=?",
                     (table_id, snapshot_date))
        conn.execute("DELETE FROM column_daily WHERE table_id=? AND snapshot_date=?",
                     (table_id, snapshot_date))
        conn.execute(
            """INSERT INTO table_daily(table_id, snapshot_date, row_count, column_count,
                                       weekday, profile_json)
               VALUES(?,?,?,?,?,?)""",
            (table_id, snapshot_date, row_count, len(column_rows), weekday,
             json.dumps(table_profile, ensure_ascii=False)),
        )
        conn.executemany(
            """INSERT INTO column_daily(table_id, snapshot_date, column_name, stable_col_id,
                    ordinal, inferred_type, null_count, null_rate, distinct_count,
                    distinct_ratio, profile_json)
               VALUES(:table_id,:snapshot_date,:column_name,:stable_col_id,:ordinal,
                      :inferred_type,:null_count,:null_rate,:distinct_count,
                      :distinct_ratio,:profile_json)""",
            [{**c, "table_id": table_id, "snapshot_date": snapshot_date,
              "profile_json": json.dumps(c["profile"], ensure_ascii=False)}
             for c in column_rows],
        )


def list_snapshot_dates(conn: sqlite3.Connection, table_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT snapshot_date FROM table_daily WHERE table_id=? ORDER BY snapshot_date",
        (table_id,)).fetchall()
    return [r["snapshot_date"] for r in rows]


def get_latest_snapshot_date(conn: sqlite3.Connection, table_id: str) -> str | None:
    row = conn.execute(
        "SELECT MAX(snapshot_date) AS d FROM table_daily WHERE table_id=?",
        (table_id,)).fetchone()
    return row["d"] if row else None


def get_table_daily(conn: sqlite3.Connection, table_id: str, date: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM table_daily WHERE table_id=? AND snapshot_date=?",
        (table_id, date)).fetchone()
    if not row:
        return None
    out = dict(row)
    out["profile"] = json.loads(out.pop("profile_json"))
    return out


def list_table_daily(conn: sqlite3.Connection, table_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT table_id,snapshot_date,row_count,column_count,weekday FROM table_daily "
        "WHERE table_id=? ORDER BY snapshot_date", (table_id,)).fetchall()
    return [dict(r) for r in rows]


def get_column_daily(conn: sqlite3.Connection, table_id: str, date: str,
                     column_name: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM column_daily WHERE table_id=? AND snapshot_date=? AND column_name=?",
        (table_id, date, column_name)).fetchone()
    if not row:
        return None
    out = dict(row)
    out["profile"] = json.loads(out.pop("profile_json"))
    return out


def get_column_history_by_identity(conn: sqlite3.Connection, table_id: str,
                                   stable_col_id: str) -> list[dict]:
    rows = conn.execute(
        """SELECT * FROM column_daily WHERE table_id=? AND stable_col_id=?
           ORDER BY snapshot_date""", (table_id, stable_col_id)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["profile"] = json.loads(d.pop("profile_json"))
        out.append(d)
    return out


def load_all_column_history(conn: sqlite3.Connection, table_id: str,
                            before_date: str | None = None
                            ) -> dict[str, list[dict]]:
    """整表一次预载各列身份的完整历史，避免逐列 N×M 次查询。"""
    sql = "SELECT * FROM column_daily WHERE table_id=?"
    params: list[Any] = [table_id]
    if before_date:
        sql += " AND snapshot_date < ?"
        params.append(before_date)
    sql += " ORDER BY snapshot_date"
    grouped: dict[str, list[dict]] = {}
    for r in conn.execute(sql, params).fetchall():
        d = dict(r)
        d["profile"] = json.loads(d.pop("profile_json"))
        grouped.setdefault(d["stable_col_id"], []).append(d)
    return grouped


def get_columns_on_date(conn: sqlite3.Connection, table_id: str, date: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM column_daily WHERE table_id=? AND snapshot_date=? ORDER BY ordinal",
        (table_id, date)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["profile"] = json.loads(d.pop("profile_json"))
        out.append(d)
    return out


def list_all_tables(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT s.table_id, s.name, s.kind,
                  t.snapshot_date AS latest_date,
                  t.row_count AS latest_rows,
                  t.column_count AS latest_columns,
                  (SELECT COUNT(*) FROM table_daily x WHERE x.table_id=s.table_id)
                      AS snapshot_count
           FROM sources s
           LEFT JOIN table_daily t
             ON t.table_id=s.table_id
            AND t.snapshot_date=(SELECT MAX(snapshot_date) FROM table_daily
                                 WHERE table_id=s.table_id)
           ORDER BY s.name""").fetchall()
    return [dict(r) for r in rows]


# ---------------- 列身份 ----------------

def list_identities(conn: sqlite3.Connection, table_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM column_identities WHERE table_id=? ORDER BY stable_col_id",
        (table_id,)).fetchall()
    return [dict(r) for r in rows]


def upsert_identity(conn: sqlite3.Connection, ident: dict) -> None:
    conn.execute(
        """INSERT INTO column_identities(table_id, stable_col_id, first_seen_date,
               original_name, current_name, original_ordinal, current_ordinal,
               inferred_type, fingerprint_json, last_seen_date)
           VALUES(:table_id,:stable_col_id,:first_seen_date,:original_name,:current_name,
                  :original_ordinal,:current_ordinal,:inferred_type,:fingerprint_json,
                  :last_seen_date)
           ON CONFLICT(table_id, stable_col_id) DO UPDATE SET
             current_name=excluded.current_name,
             current_ordinal=excluded.current_ordinal,
             inferred_type=excluded.inferred_type,
             fingerprint_json=excluded.fingerprint_json,
             last_seen_date=excluded.last_seen_date""",
        ident,
    )


# ---------------- 事件 ----------------

def event_exists(conn: sqlite3.Connection, table_id: str, date: str,
                 stable_col_id: str | None, metric: str) -> bool:
    row = conn.execute(
        """SELECT 1 FROM events WHERE table_id=? AND snapshot_date=?
           AND IFNULL(stable_col_id,'')=IFNULL(?, '') AND metric=?""",
        (table_id, date, stable_col_id, metric)).fetchone()
    return row is not None


def insert_event(conn: sqlite3.Connection, ev: dict) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        """INSERT INTO events(table_id, stable_col_id, column_name, snapshot_date, metric,
               severity, value, threshold, baseline_dates, message, signature,
               suppressed, status, created_at)
           VALUES(:table_id,:stable_col_id,:column_name,:snapshot_date,:metric,:severity,
                  :value,:threshold,:baseline_dates,:message,:signature,
                  :suppressed,:status,:created_at)""",
        {**ev, "created_at": now,
         "suppressed": ev.get("suppressed", 0),
         "status": ev.get("status", "open")})
    conn.commit()
    return cur.lastrowid


def list_events(conn: sqlite3.Connection, *, table_id: str | None = None,
                severity: str | None = None, status: str | None = None,
                metric: str | None = None, start_date: str | None = None,
                end_date: str | None = None, include_suppressed: bool = True,
                limit: int = 1000) -> list[dict]:
    sql = "SELECT * FROM events WHERE 1=1"
    params: list[Any] = []
    if table_id:
        sql += " AND table_id=?"; params.append(table_id)
    if severity:
        sql += " AND severity=?"; params.append(severity)
    if status:
        sql += " AND status=?"; params.append(status)
    if metric:
        sql += " AND metric=?"; params.append(metric)
    if start_date:
        sql += " AND snapshot_date>=?"; params.append(start_date)
    if end_date:
        sql += " AND snapshot_date<=?"; params.append(end_date)
    if not include_suppressed:
        sql += " AND suppressed=0"
    sql += " ORDER BY snapshot_date DESC, event_id DESC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def update_event_status(conn: sqlite3.Connection, event_id: int, status: str) -> None:
    conn.execute("UPDATE events SET status=? WHERE event_id=?", (status, event_id))
    conn.commit()


# ---------------- 误报反馈签名 ----------------

def list_feedback(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM feedback_signatures ORDER BY created_at DESC").fetchall()]


def add_feedback(conn: sqlite3.Connection, *, signature: str, table_id: str,
                 metric: str, note: str = "") -> None:
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """INSERT INTO feedback_signatures(signature, table_id, metric, note, hit_count,
               created_at, last_hit_at)
           VALUES(?,?,?,? ,0, ?, NULL)
           ON CONFLICT(signature) DO UPDATE SET note=excluded.note""",
        (signature, table_id, metric, note, now))
    conn.commit()


def get_feedback_set(conn: sqlite3.Connection) -> dict[str, dict]:
    return {r["signature"]: dict(r) for r in
            conn.execute("SELECT * FROM feedback_signatures").fetchall()}


def bump_feedback_hit(conn: sqlite3.Connection, signature: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "UPDATE feedback_signatures SET hit_count=hit_count+1, last_hit_at=? WHERE signature=?",
        (now, signature))


def delete_feedback(conn: sqlite3.Connection, signature: str) -> None:
    conn.execute("DELETE FROM feedback_signatures WHERE signature=?", (signature,))
    conn.commit()


# ---------------- 阈值配置 ----------------

def _coerce(key: str, raw: str) -> Any:
    default = DEFAULT_THRESHOLDS[key]
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    return raw


def get_effective_thresholds(conn: sqlite3.Connection,
                             table_id: str | None = None) -> dict[str, Any]:
    vals = dict(DEFAULT_THRESHOLDS)
    rows = conn.execute("SELECT scope, table_id, key, value FROM threshold_overrides")
    for r in rows.fetchall():
        if r["key"] not in THRESHOLD_SCHEMA:
            continue
        if r["scope"] == "global":
            vals[r["key"]] = _coerce(r["key"], r["value"])
        elif table_id and r["scope"] == "table" and r["table_id"] == table_id:
            vals[r["key"]] = _coerce(r["key"], r["value"])
    return vals


def set_override(conn: sqlite3.Connection, scope: str, key: str, value: str,
                 table_id: str | None = None) -> None:
    if key not in THRESHOLD_SCHEMA:
        raise KeyError(f"未知阈值项: {key}")
    _coerce(key, value)  # 提前校验类型
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """INSERT INTO threshold_overrides(scope, table_id, key, value, updated_at)
           VALUES(?,?,?,?,?)
           ON CONFLICT(scope, table_id, key) DO UPDATE SET value=excluded.value,
               updated_at=excluded.updated_at""",
        (scope, table_id, key, value, now))
    conn.commit()


def delete_override(conn: sqlite3.Connection, scope: str, key: str,
                    table_id: str | None = None) -> None:
    conn.execute("DELETE FROM threshold_overrides WHERE scope=? AND key=? AND "
                 "IFNULL(table_id,'')=IFNULL(?, '')", (scope, key, table_id))
    conn.commit()


def list_overrides(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM threshold_overrides ORDER BY scope, table_id, key").fetchall()]
