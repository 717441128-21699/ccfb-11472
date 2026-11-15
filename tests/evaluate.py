"""验收评估：对照 data/ground_truth.json 检查漏报与误报。

口径：
  * 召回（一个不漏）：每个真值漂移必须在漂移窗口内出现同表、同列身份、
    同指标（expected_metric）的事件。
  * 误报：把事件按 (表, 指标, 列身份) 去重成“告警情况”，凡是落在没有任何
    真值漂移的 (表, 列) 上的，记 1 条误报；同一情况持续多日只算 1 条。
    真实漂移列上衍生出的其他指标告警视为连带告警，不算误报。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataprobe import storage

OUT = Path(__file__).resolve().parent.parent / "data" / "eval_report.json"


def main() -> int:
    conn = storage.connect()
    gt_path = Path(__file__).resolve().parent.parent / "data" / "ground_truth.json"
    gt = json.loads(gt_path.read_text(encoding="utf-8"))

    sources = {s["name"]: s["table_id"] for s in storage.list_sources(conn)}
    events = storage.list_events(conn, limit=100000)

    # 名称 -> stable_col_id 映射（兼容改名：原名/现名均可）
    name_to_ids: dict[tuple[str, str], set[str]] = {}
    for ident in conn.execute("SELECT * FROM column_identities").fetchall():
        ident = dict(ident)
        for nm in (ident["original_name"], ident["current_name"]):
            name_to_ids.setdefault((ident["table_id"], nm), set()).add(
                ident["stable_col_id"])

    # 存在真值漂移的 (table_id, stable_col_id) 集合；表级漂移用 None
    drifted_cells: set[tuple[str, str | None]] = set()
    # 存在“结构性真值”（改名/移位）的表，其上的连带结构事件不算误报
    structural_tables: set[str] = set()
    structural_metrics = {"column_renamed", "column_reordered",
                          "column_added", "column_dropped", "type_changed"}
    for g in gt:
        tid = sources[g["table"]]
        if g["column"] is None:
            drifted_cells.add((tid, None))
        else:
            for cid in name_to_ids.get((tid, g["column"]), set()):
                drifted_cells.add((tid, cid))
        if g["expected_metric"] in structural_metrics:
            structural_tables.add(tid)

    # ---------------- 召回 ----------------
    missed: list[dict] = []
    matched_events: set[int] = set()
    for g in gt:
        tid = sources[g["table"]]
        want_ids = name_to_ids.get((tid, g["column"]), set()) if g["column"] else {None}
        hit = None
        for e in events:
            if e["table_id"] != tid or e["metric"] != g["expected_metric"]:
                continue
            if not (g["start_date"] <= e["snapshot_date"] <= g["end_date"]):
                continue
            if g["column"] is None:
                if e["stable_col_id"] is None:
                    hit = e; break
            elif e["stable_col_id"] in want_ids:
                hit = e; break
        if hit is None:
            missed.append(g)
        else:
            matched_events.add(hit["event_id"])

    # ---------------- 误报（按情况去重）----------------
    fp_groups: dict[tuple, dict] = {}
    tp_groups: dict[tuple, int] = {}
    for e in events:
        cell = (e["table_id"], e["stable_col_id"])
        key = (e["table_id"], e["metric"], e["stable_col_id"], e["column_name"])
        structural_side_effect = (
            e["metric"] in structural_metrics
            and e["table_id"] in structural_tables)
        if cell in drifted_cells or structural_side_effect:
            tp_groups[key] = tp_groups.get(key, 0) + 1
        else:
            fp_groups.setdefault(key, {"first": e["snapshot_date"],
                                       "last": e["snapshot_date"], "count": 0,
                                       "sample": e["message"],
                                       "severity": e["severity"]})
            fp_groups[key]["count"] += 1
            fp_groups[key]["last"] = e["snapshot_date"]

    fp_list = [
        {"table": k[0], "metric": k[1], "column": k[3],
         "first_date": v["first"], "last_date": v["last"],
         "event_days": v["count"], "severity": v["severity"],
         "message": v["sample"]}
        for k, v in sorted(fp_groups.items(), key=lambda kv: -kv[1]["count"])
    ]

    sev_counter: dict[str, int] = {}
    metric_counter: dict[str, int] = {}
    for e in events:
        sev_counter[e["severity"]] = sev_counter.get(e["severity"], 0) + 1
        metric_counter[e["metric"]] = metric_counter.get(e["metric"], 0) + 1

    report = {
        "ground_truth_total": len(gt),
        "events_total_rows": len(events),
        "true_alarm_groups": len(tp_groups),
        "missed": missed,
        "missed_count": len(missed),
        "false_positive_groups": fp_list,
        "false_positive_count": len(fp_list),
        "false_positive_event_rows": sum(v["event_days"] for v in fp_list),
        "severity_breakdown": sev_counter,
        "metric_breakdown": metric_counter,
        "pass_zero_miss": len(missed) == 0,
        "pass_fp_le_5": len(fp_list) <= 5,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 64)
    print(f"真值漂移 {len(gt)} 个 | 事件行 {len(events)} | 真实告警情况 {len(tp_groups)}")
    print(f"漏报 {len(missed)} 个（要求 0）：{'✅ 通过' if not missed else '❌ 未通过'}")
    for m in missed:
        print(f"  ✗ {m['table']}.{m['column']} {m['expected_metric']} ({m['kind']})")
    print(f"误报情况 {len(fp_list)} 条（要求 ≤5）：{'✅ 通过' if len(fp_list) <= 5 else '❌ 未通过'}")
    for fp in fp_list:
        print(f"  ! {fp['table']}.{fp['column']} {fp['metric']} "
              f"{fp['first_date']}~{fp['last_date']} x{fp['event_days']}：{fp['message']}")
    print("-" * 64)
    print("级别分布：", sev_counter)
    print("指标分布：", metric_counter)
    print("=" * 64)
    print(("🎉 全部验收通过" if report["pass_zero_miss"] and report["pass_fp_le_5"]
           else "⚠️ 需要调优，详见 data/eval_report.json"))
    return 0 if report["pass_zero_miss"] and report["pass_fp_le_5"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
