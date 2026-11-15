"""漂移检测单元测试：用临时库构造可控快照，验证检测、季节性、反馈降级。"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataprobe import storage, drift, identity
from dataprobe.config import ST_AUTO_DOWNGRADED
from dataprobe.profiler import profile_table


def _day(conn, table_id, ds, columns, data, weekday):
    _, crows = profile_table(ds, table_id, columns, data)
    identity.assign_identities(conn, table_id, ds, crows)
    storage.replace_daily_snapshot(conn, table_id=table_id, snapshot_date=ds,
                                   row_count=len(data[columns[0]]), weekday=weekday,
                                   table_profile={"x": 1}, column_rows=crows)
    return crows


def main() -> None:
    tmp = Path(tempfile.mkdtemp()) / "t.db"
    conn = storage.connect(str(tmp))
    storage.init_db(conn)
    tid = "unit__main"
    th = storage.get_effective_thresholds(conn)
    # 缩短预热，便于测试
    th["warmup_days"] = 3
    th["min_reference_days"] = 3

    from datetime import date, timedelta
    start = date(2026, 9, 1)  # 周二
    rng_seed = 7
    import random
    rng = random.Random(rng_seed)

    def clean_day(n):
        vals = [f"{rng.gauss(100, 10):.2f}" for _ in range(n)]
        cats = rng.choices(["a", "b", "c"], weights=[0.6, 0.3, 0.1], k=n)
        return ["value", "cat"], {"value": vals, "cat": cats}

    # 10 个干净快照（含周末低流量）
    history_rows = []
    for i in range(10):
        d = start + timedelta(days=i)
        wd = d.weekday()
        n = 300 if wd < 5 else 140
        cols, data = clean_day(n)
        rows = _day(conn, tid, d.isoformat(), cols, data, wd)
        history_rows.append((d, rows))

    # 场景 1：周末低流量不应告警（季节性）
    sat = start + timedelta(days=12)  # 周六
    cols, data = clean_day(140)
    rows = _day(conn, tid, sat.isoformat(), cols, data, 5)
    evs = drift.detect_for_table(conn, tid, sat.isoformat(), rows, 5, th, {})
    weekend_alarms = [e for e in evs if e["metric"] in
                      ("numeric_distribution", "category_distribution", "cardinality")]
    print(f"[1] 周末低流量分布类告警: {len(weekend_alarms)}（期望 0）",
          "OK" if not weekend_alarms else f"FAIL {[e['metric'] for e in weekend_alarms]}")

    # 场景 2：均值平移必须告警
    shift_day = start + timedelta(days=14)  # 周一
    n = 300
    vals = [f"{rng.gauss(160, 10):.2f}" for _ in range(n)]   # 均值 +60
    cats = rng.choices(["a", "b", "c"], weights=[0.6, 0.3, 0.1], k=n)
    rows = _day(conn, tid, shift_day.isoformat(),
                ["value", "cat"], {"value": vals, "cat": cats}, shift_day.weekday())
    evs = drift.detect_for_table(conn, tid, shift_day.isoformat(), rows,
                                 shift_day.weekday(), th, {})
    num_evs = [e for e in evs if e["metric"] == "numeric_distribution"]
    print(f"[2] 均值平移数值分布告警: {len(num_evs)}（期望 >=1）",
          "OK" if num_evs else "FAIL")

    # 场景 3：反馈签名生效后自动降级
    if num_evs:
        sig = num_evs[0]["signature"]
        storage.add_feedback(conn, signature=sig, table_id=tid,
                             metric="numeric_distribution", note="unit test")
        evs2 = drift.detect_for_table(conn, tid, shift_day.isoformat(), rows,
                                      shift_day.weekday(), th,
                                      storage.get_feedback_set(conn))
        fb_ev = next(e for e in evs2 if e["signature"] == sig)
        print(f"[3] 反馈后状态={fb_ev['status']} 级别={fb_ev['severity']}（期望 auto_downgraded）",
              "OK" if fb_ev["status"] == ST_AUTO_DOWNGRADED else "FAIL")

    # 场景 4：KS 对形状变化敏感（同均值不同方差）
    ks_day = start + timedelta(days=15)
    vals = [f"{rng.gauss(100, 35):.2f}" for _ in range(300)]  # 方差放大 3.5x
    cats = rng.choices(["a", "b", "c"], weights=[0.6, 0.3, 0.1], k=300)
    rows = _day(conn, tid, ks_day.isoformat(),
                ["value", "cat"], {"value": vals, "cat": cats}, ks_day.weekday())
    evs = drift.detect_for_table(conn, tid, ks_day.isoformat(), rows,
                                 ks_day.weekday(), th, {})
    print(f"[4] 方差变化告警: {[e['metric'] for e in evs]}",
          "OK" if any(e["metric"] == "numeric_distribution" for e in evs) else "FAIL")

    conn.close()


if __name__ == "__main__":
    main()
