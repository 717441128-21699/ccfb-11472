"""列身份识别：跨快照跟踪同一列，容忍改名与位置挪动。

每个表维护 stable_col_id（c0001, c0002, ...），身份与指纹（类型+空值/基数轮廓+
数值分位数或文本签名）一起持久化。新快照到来时按名称、类型、位置、指纹构造
代价做贪心匹配：名称不变直接确认；改名/移位靠类型+数据指纹兜底。
"""
from __future__ import annotations

import json
from typing import Any

from . import storage

# ---- 匹配代价 ----
COST_NAME_EXACT = 0
COST_NAME_NORM = 1
COST_NAME_DIFF = 10
COST_TYPE_MATCH = 0
COST_TYPE_DIFF = 5
COST_ORDINAL_PER_STEP = 0.3
COST_ORDINAL_CAP = 3.0
COST_FP_MATCH = 0
COST_FP_DIFF = 10
ACCEPT_COST = 20.0          # 改名/兜底匹配的代价上限
EXACT_NAME_ACCEPT = 22.0    # 同名列即使类型抖动也优先认领，抑制身份抖动


def _norm_name(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def make_fingerprint(profile: dict) -> tuple:
    """数据指纹：类型 + 粗粒度统计特征，不依赖列名与位置。"""
    p = profile
    parts: list[Any] = [p["inferred_type"],
                        round(p["null_rate"], 1),
                        round(p["distinct_ratio"], 1)]
    num = p.get("numeric")
    if num:
        parts.append(("num", round(num["median"], 2), round(num["q25"], 2),
                      round(num["q75"], 2)))
    txt = p.get("text")
    if txt and txt.get("patterns", {}).get("signatures"):
        sigs = tuple(s["signature"] for s in txt["patterns"]["signatures"][:2])
        parts.append(("txt", sigs))
    if p.get("datetime"):
        parts.append(("dt",))
    if p.get("boolean"):
        parts.append(("bool", round(p["boolean"]["true_ratio"], 1)))
    return tuple(parts)


def _fp_compatible(a: tuple, b: tuple) -> bool:
    """指纹兼容：粗轮廓（类型/空值档/基数档）一致，且数值列的分位数位置
    或文本列的结构签名不冲突。仅靠前三位会把两个不同的同类型数值列误判为同列。"""
    if a[:3] != b[:3]:
        return False
    # 指纹第 4 段起是类型专属特征（num/txt/dt/bool），逐段核对
    for tail_a, tail_b in zip(a[3:], b[3:]):
        if tail_a[0] != tail_b[0]:
            continue
        if tail_a[0] == "num":
            # 数值列：比较中位数相对偏差，超过 50% 视为不同列
            _, med_a, _, _ = tail_a
            _, med_b, _, _ = tail_b
            scale = max(abs(med_a), abs(med_b), 1.0)
            if abs(med_a - med_b) / scale > 0.50:
                return False
        elif tail_a[0] == "txt":
            if tail_a[1] != tail_b[1]:
                return False
        elif tail_a[0] == "bool":
            if abs(tail_a[1] - tail_b[1]) > 0.20:
                return False
    return True


def assign_identities(conn, table_id: str, snapshot_date: str,
                      column_rows: list[dict]) -> list[dict]:
    """给当日各列补 stable_col_id 并更新 column_identities。"""
    previous = {r["stable_col_id"]: r for r in
                storage.list_identities(conn, table_id)}

    if not previous:
        for row in column_rows:
            cid = f"c{row['ordinal'] + 1:04d}"
            row["stable_col_id"] = cid
            _upsert(conn, table_id, cid, snapshot_date, row, is_new=True, prev=None)
        conn.commit()
        return column_rows

    cur_fp = {r["column_name"]: make_fingerprint(r["profile"]) for r in column_rows}

    def pair_cost(row: dict, cid: str, prev: dict) -> float:
        cur_name = row["column_name"]
        if cur_name == prev["current_name"]:
            c_name = COST_NAME_EXACT
        elif _norm_name(cur_name) == _norm_name(prev["current_name"]):
            c_name = COST_NAME_NORM
        else:
            c_name = COST_NAME_DIFF
        c_type = COST_TYPE_MATCH if row["inferred_type"] == prev["inferred_type"] \
            else COST_TYPE_DIFF
        c_ord = min(COST_ORDINAL_CAP,
                    COST_ORDINAL_PER_STEP * abs(row["ordinal"] - prev["current_ordinal"]))
        prev_fp = tuple(json.loads(prev["fingerprint_json"])) if prev["fingerprint_json"] else None
        c_fp = COST_FP_MATCH if prev_fp and _fp_compatible(cur_fp[cur_name], prev_fp) \
            else COST_FP_DIFF
        return c_name + c_type + c_ord + c_fp

    # 第一阶段：同名列（含归一同名）直接一对一锁定，天然唯一，优先于一切跨列匹配
    matches: dict[str, str] = {}
    used_ids: set[str] = set()
    for row in column_rows:
        for cid, prev in previous.items():
            if cid in used_ids:
                continue
            same_exact = row["column_name"] == prev["current_name"]
            same_norm = _norm_name(row["column_name"]) == _norm_name(prev["current_name"])
            if not (same_exact or same_norm):
                continue
            cost = pair_cost(row, cid, prev)
            if same_exact and cost <= EXACT_NAME_ACCEPT:
                matches[row["column_name"]] = cid
                used_ids.add(cid)
                break
            if same_norm and cost <= ACCEPT_COST:
                matches[row["column_name"]] = cid
                used_ids.add(cid)
                break

    # 第二阶段：改名/移位列，按全局代价贪心兜底
    candidates: list[tuple[float, str, str]] = []
    for row in column_rows:
        if row["column_name"] in matches:
            continue
        cur_finger = make_fingerprint(row["profile"])
        for cid, prev in previous.items():
            if cid in used_ids:
                continue
            # 名称完全不同时，数据指纹必须兼容才允许认领，
            # 防止“新增的同类列”被误认成旧列
            prev_fp = tuple(json.loads(prev["fingerprint_json"])) \
                if prev["fingerprint_json"] else None
            if not (prev_fp and _fp_compatible(cur_finger, prev_fp)):
                continue
            candidates.append((pair_cost(row, cid, prev), row["column_name"], cid))
    candidates.sort()
    used_cols: set[str] = set(matches)
    for cost, cur_name, cid in candidates:
        if cur_name in used_cols or cid in used_ids or cost > ACCEPT_COST:
            continue
        matches[cur_name] = cid
        used_cols.add(cur_name)
        used_ids.add(cid)

    next_seq = max((int(cid[1:]) for cid in previous), default=0)
    for row in column_rows:
        if row["column_name"] in matches:
            row["stable_col_id"] = matches[row["column_name"]]
        else:
            next_seq += 1
            row["stable_col_id"] = f"c{next_seq:04d}"
        _upsert(conn, table_id, row["stable_col_id"], snapshot_date, row,
                is_new=row["stable_col_id"] not in previous,
                prev=previous.get(row["stable_col_id"]))
    conn.commit()
    return column_rows


def _upsert(conn, table_id: str, cid: str, snapshot_date: str, row: dict,
            *, is_new: bool, prev: dict | None) -> None:
    storage.upsert_identity(conn, {
        "table_id": table_id, "stable_col_id": cid,
        "first_seen_date": snapshot_date if is_new else prev["first_seen_date"],
        "original_name": row["column_name"] if is_new else prev["original_name"],
        "current_name": row["column_name"],
        "original_ordinal": row["ordinal"] if is_new else prev["original_ordinal"],
        "current_ordinal": row["ordinal"],
        "inferred_type": row["inferred_type"],
        "fingerprint_json": json.dumps(make_fingerprint(row["profile"]), ensure_ascii=False),
        "last_seen_date": snapshot_date,
    })


def detect_schema_changes(conn, table_id: str, snapshot_date: str,
                          column_rows: list[dict]) -> list[dict]:
    """对照前一快照，产出新增、消失、改名、移位、类型变化结构事件。"""
    earlier = [d for d in storage.list_snapshot_dates(conn, table_id)
               if d < snapshot_date]
    if not earlier:
        return []
    prev_date = max(earlier)
    prev_by_id = {c["stable_col_id"]: c for c in
                  storage.get_columns_on_date(conn, table_id, prev_date)}
    cur_by_id = {r["stable_col_id"]: r for r in column_rows}

    events: list[dict] = []
    for cid, cur in cur_by_id.items():
        prev = prev_by_id.get(cid)
        if prev is None:
            events.append(_schema_event(table_id, snapshot_date, cur, "column_added",
                                        f"新增列「{cur['column_name']}」", "info"))
            continue
        if prev["column_name"] != cur["column_name"]:
            events.append(_schema_event(
                table_id, snapshot_date, cur, "column_renamed",
                f"列改名：「{prev['column_name']}」→「{cur['column_name']}」(身份 {cid})",
                "warning"))
        if prev["ordinal"] != cur["ordinal"]:
            events.append(_schema_event(
                table_id, snapshot_date, cur, "column_reordered",
                f"列移位：「{cur['column_name']}」位置 {prev['ordinal']} → {cur['ordinal']} (身份 {cid})",
                "info"))
        if prev["inferred_type"] != cur["inferred_type"]:
            events.append(_schema_event(
                table_id, snapshot_date, cur, "type_changed",
                f"列类型变化：「{cur['column_name']}」{prev['inferred_type']} → {cur['inferred_type']}",
                "warning"))
    for cid, prev in prev_by_id.items():
        if cid not in cur_by_id:
            events.append({
                "table_id": table_id, "stable_col_id": cid,
                "column_name": prev["column_name"], "snapshot_date": snapshot_date,
                "metric": "column_dropped", "severity": "warning",
                "value": None, "threshold": None, "baseline_dates": prev_date,
                "message": f"列消失：「{prev['column_name']}」(身份 {cid})",
                "signature": f"{table_id}|column_dropped|{cid}",
            })
    return events


def _schema_event(table_id: str, date: str, row: dict, metric: str,
                  message: str, severity: str) -> dict:
    return {
        "table_id": table_id, "stable_col_id": row["stable_col_id"],
        "column_name": row["column_name"], "snapshot_date": date,
        "metric": metric, "severity": severity,
        "value": None, "threshold": None, "baseline_dates": None,
        "message": message,
        "signature": f"{table_id}|{metric}|{row['stable_col_id']}",
    }
