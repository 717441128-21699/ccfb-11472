"""日批流水线编排：单表单日全流程，按 (表, 日期) 幂等可重跑。"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from . import drift, identity, profiler, storage
from .connectors import load_source


def run_table_day(conn, source: dict, snapshot_date: str,
                  thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    """处理一张表某一天的快照。返回统计摘要。"""
    table_id = source["table_id"]
    thresholds = thresholds or storage.get_effective_thresholds(conn, table_id)
    weekday = datetime.strptime(snapshot_date, "%Y-%m-%d").weekday()

    columns, data = load_source(source["kind"], source["path"], source.get("table_name"))

    # 重跑保护：先清掉当日旧事件，画像由 replace_daily_snapshot 整体覆盖
    conn.execute("DELETE FROM events WHERE table_id=? AND snapshot_date=?",
                 (table_id, snapshot_date))

    table_profile, column_rows = profiler.profile_table(
        snapshot_date, table_id, columns, data)
    identity.assign_identities(conn, table_id, snapshot_date, column_rows)

    storage.replace_daily_snapshot(
        conn, table_id=table_id, snapshot_date=snapshot_date,
        row_count=table_profile["row_count"], weekday=weekday,
        table_profile=table_profile, column_rows=column_rows)

    # 结构变更事件（改名/移位/新增/消失/类型变化）
    schema_events = identity.detect_schema_changes(
        conn, table_id, snapshot_date, column_rows)

    # 数值/分布类漂移（直接用内存画像行，避免存储行结构差异）
    feedback = storage.get_feedback_set(conn)
    drift_events = drift.detect_for_table(
        conn, table_id, snapshot_date, column_rows,
        weekday, thresholds, feedback)

    inserted = 0
    for ev in [*schema_events, *drift_events]:
        if storage.event_exists(conn, table_id, snapshot_date,
                                ev.get("stable_col_id"), ev["metric"]):
            continue
        storage.insert_event(conn, ev)
        inserted += 1

    return {
        "table_id": table_id, "snapshot_date": snapshot_date,
        "rows": table_profile["row_count"], "columns": len(columns),
        "events": inserted,
    }
