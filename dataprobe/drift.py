"""漂移检测引擎。

比较口径：
  * 分布漂移（PSI/KS、格式 PSI、空值率、基数、最值）只看“同星期”历史基线，
    自动消化周末量少等周内季节性；同星期样本不足时回退全部基线并降低敏感度。
  * 分布类指标要求两侧样本量 >= min_sample_distribution，小样本日直接跳过，
    避免周末低流量产生伪漂移。
  * 空值率/行数使用中位数 + MAD 的稳健 z 分数，不被缓慢趋势和个别波动带偏。
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from . import storage
from .config import (
    EPS, SEV_CRITICAL, SEV_WARNING, TOP_VALUE_CAP,
    ST_AUTO_DOWNGRADED, ST_OPEN,
)
from .infer import T_BOOL, T_DATETIME, T_FLOAT, T_INT, T_TEXT

DOWNGRADE_MAP = {SEV_CRITICAL: SEV_WARNING, SEV_WARNING: "info", "info": "info"}


# ---------------- 基础统计 ----------------

def robust_z(value: float, history: list[float]) -> float:
    arr = np.asarray(history, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad
    if scale < EPS:
        # MAD 为 0（历史高度稳定）时，任何实质偏离都算异常；用均值标准差兜底
        scale = float(arr.std(ddof=0))
        if scale < EPS:
            return 0.0 if abs(value - med) < EPS else 99.0
    return float((value - med) / scale)


def psi_from_buckets(base_share: dict[Any, float], cur_share: dict[Any, float]) -> float:
    keys = set(base_share) | set(cur_share)
    total = 0.0
    for k in keys:
        p = max(base_share.get(k, 0.0), EPS)
        q = max(cur_share.get(k, 0.0), EPS)
        total += (q - p) * np.log(q / p)
    return float(total)


def ks_two_sample(a: list[float], b: list[float]) -> float:
    """两样本 KS 统计量，纯 numpy 实现。"""
    if not a or not b:
        return 0.0
    x = np.sort(np.asarray(a, dtype=float))
    y = np.sort(np.asarray(b, dtype=float))
    all_vals = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, all_vals, side="right") / len(x)
    cdf_y = np.searchsorted(y, all_vals, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


# ---------------- 基线选择（季节性） ----------------

def select_baseline(history: list[dict], current_weekday: int,
                    window: int) -> tuple[list[dict], bool]:
    """优先同星期历史快照；不足 2 个同星期样本时回退全量窗口。

    返回 (基线列表, 是否同星期口径)。
    """
    recent = history[-window:]
    same_wd = [h for h in recent if h["weekday"] == current_weekday]
    if len(same_wd) >= 2:
        return same_wd, True
    return recent, False


# ---------------- 各指标分布构造 ----------------

def _numeric_baseline(history_cols: list[dict]) -> list[float]:
    vals: list[float] = []
    for hc in history_cols:
        num = hc["profile"].get("numeric") or {}
        vals.extend(num.get("sample_numbers", []))
    return vals


def _categorical_share(profile: dict) -> dict[str, float]:
    txt = profile.get("text") or {}
    top = txt.get("top_values", [])[:TOP_VALUE_CAP]
    total = sum(t["count"] for t in top)
    non_null = profile["total_count"] - profile["null_count"]
    other = max(non_null - total, 0)
    denom = total + other
    if denom == 0:
        return {}
    share = {t["value"]: t["count"] / denom for t in top}
    if other:
        share["__OTHER__"] = other / denom
    return share


def _pattern_share(profile: dict) -> dict[str, float]:
    sigs = ((profile.get("text") or {}).get("patterns") or {}).get("signatures", [])
    total = sum(s["count"] for s in sigs)
    if total == 0:
        return {}
    return {s["signature"]: s["count"] / total for s in sigs}


def _bool_share(profile: dict) -> dict[str, float]:
    bo = profile.get("boolean") or {}
    if "true_ratio" not in bo:
        return {}
    return {"true": bo["true_ratio"], "false": 1.0 - bo["true_ratio"]}


# ---------------- 主检测流程 ----------------

def detect_for_table(conn, table_id: str, snapshot_date: str,
                     current_columns: list[dict], weekday: int,
                     thresholds: dict[str, Any],
                     feedback: dict[str, dict]) -> list[dict]:
    prior_dates = [d for d in storage.list_snapshot_dates(conn, table_id)
                   if d < snapshot_date]
    if len(prior_dates) < thresholds["min_reference_days"] or \
            len(prior_dates) < thresholds["warmup_days"]:
        return []

    # 表级行数：同星期季节性基线
    table_rows = [storage.get_table_daily(conn, table_id, d) for d in prior_dates]
    weekday_map = {t["snapshot_date"]: t["weekday"] for t in table_rows}
    base_tables, seasonal = select_baseline(table_rows, weekday,
                                            thresholds["reference_window_days"])
    events: list[dict] = []
    events.extend(_check_volume(conn, table_id, snapshot_date, current_columns,
                                base_tables, seasonal, thresholds, feedback))

    # 整表预载历史，按列身份分组（仅取当日之前的快照）
    history_by_id = storage.load_all_column_history(conn, table_id, snapshot_date)
    cur_by_id = {c["stable_col_id"]: c for c in current_columns}
    for cid, cur in cur_by_id.items():
        hist = history_by_id.get(cid, [])
        if len(hist) < thresholds["min_reference_days"]:
            continue
        for h in hist:
            h["weekday"] = weekday_map.get(h["snapshot_date"], 0)
        base_cols, seasonal_col = select_baseline(hist, weekday,
                                                  thresholds["reference_window_days"])
        events.extend(_check_column(conn, table_id, snapshot_date, cur, base_cols,
                                    seasonal_col, thresholds, feedback))
    return events


def _make_event(conn, table_id, date, col, metric, severity, value, threshold,
                baseline_dates, message, feedback, thresholds) -> dict:
    signature = f"{table_id}|{metric}|{col['stable_col_id']}"
    ev = {
        "table_id": table_id, "stable_col_id": col["stable_col_id"],
        "column_name": col["column_name"], "snapshot_date": date,
        "metric": metric, "severity": severity,
        "value": round(float(value), 6) if value is not None else None,
        "threshold": threshold,
        "baseline_dates": baseline_dates, "message": message,
        "signature": signature, "suppressed": 0, "status": ST_OPEN,
    }
    if signature in feedback:
        storage.bump_feedback_hit(conn, signature)
        if thresholds["feedback_mode"] == "suppress":
            ev["suppressed"] = 1
            ev["status"] = ST_AUTO_DOWNGRADED
        else:
            ev["severity"] = DOWNGRADE_MAP[ev["severity"]]
            ev["status"] = ST_AUTO_DOWNGRADED
    return ev


def _check_volume(conn, table_id, date, current_columns, base_tables, seasonal,
                  thresholds, feedback) -> list[dict]:
    # 行数对星期几高度敏感：没有同星期基线时（首个周末前）直接跳过，
    # 避免拿工作日体量比周末造成误报。真实骤降在基线成熟后必被捕获。
    if not base_tables or not seasonal:
        return []
    cur_rows = current_columns[0]["profile"]["total_count"] if current_columns else 0
    hist_vol = [b["row_count"] for b in base_tables]
    med = float(np.median(hist_vol))
    if med < EPS:
        return []
    z = robust_z(cur_rows, hist_vol)
    rel = abs(cur_rows - med) / med
    z_cut = thresholds["volume_robust_z_warning"]
    if abs(z) >= z_cut and rel >= thresholds["volume_min_abs_ratio"]:
        direction = "骤增" if cur_rows > med else "骤降"
        return [_make_event(conn, table_id, date,
                            {"stable_col_id": None, "column_name": "(表级)"},
                            "row_volume", SEV_WARNING, z, z_cut,
                            ",".join(b["snapshot_date"] for b in base_tables),
                            f"行数{direction}：当前 {cur_rows}，同星期基线中位数 {med:.0f}"
                            f"（稳健 z={z:.2f}，相对偏离 {rel:.1%}）",
                            feedback, thresholds)]
    return []


def _check_column(conn, table_id, date, cur, base_cols, seasonal, thresholds,
                  feedback) -> list[dict]:
    events: list[dict] = []
    p = cur["profile"]
    cur_non_null = p["total_count"] - p["null_count"]
    base_non_null = [h["profile"]["total_count"] - h["profile"]["null_count"]
                     for h in base_cols]
    base_dates = ",".join(h["snapshot_date"] for h in base_cols)
    min_n = thresholds["min_sample_distribution"]
    small_sample = cur_non_null < min_n or (base_non_null and np.median(base_non_null) < min_n)

    def add(metric, severity, value, threshold, message):
        events.append(_make_event(conn, table_id, date, cur, metric, severity, value,
                                  threshold, base_dates, message, feedback, thresholds))

    # 1) 空值率漂移（小样本下只信绝对大幅变化，不用 z）
    null_hist = [h["null_rate"] for h in base_cols]
    null_med = float(np.median(null_hist))
    null_delta = p["null_rate"] - null_med
    if abs(null_delta) >= thresholds["null_abs_delta_warning"]:
        sev = SEV_CRITICAL if abs(null_delta) >= 2 * thresholds["null_abs_delta_warning"] \
            else SEV_WARNING
        add("null_rate", sev, null_delta, thresholds["null_abs_delta_warning"],
            f"空值率变化 {null_med:.1%} → {p['null_rate']:.1%}"
            f"（绝对变化 {null_delta:+.1%}）")
    elif not small_sample:
        z = robust_z(p["null_rate"], null_hist)
        if abs(z) >= thresholds["null_robust_z_warning"] and abs(null_delta) >= 0.05:
            add("null_rate", SEV_WARNING, z,
                thresholds["null_robust_z_warning"],
                f"空值率异常：{null_med:.1%} → {p['null_rate']:.1%}（稳健 z={z:.2f}）")

    # 2) 数值分布：PSI（分桶）+ KS（经验分布）
    if p["inferred_type"] in (T_INT, T_FLOAT) and not small_sample and p.get("numeric"):
        cur_num = p["numeric"]["sample_numbers"]
        base_num = _numeric_baseline(base_cols)
        if len(base_num) >= min_n:
            ks = ks_two_sample(base_num, cur_num)
            # 用合并分位数分桶，保证两侧桶边界一致
            bins = np.unique(np.quantile(base_num, np.linspace(0, 1, 11)))
            if len(bins) >= 3:
                base_share = _bucket_share(base_num, bins)
                cur_share = _bucket_share(cur_num, bins)
                psi = psi_from_buckets(base_share, cur_share)
            else:
                psi = 0.0
            psi_w = thresholds["psi_warning"] * (1.25 if not seasonal else 1.0)
            ks_w = thresholds["ks_warning"] * (1.15 if not seasonal else 1.0)
            if psi >= psi_w or ks >= ks_w:
                sev = SEV_CRITICAL if psi >= thresholds["psi_critical"] or \
                    ks >= thresholds["ks_critical"] else SEV_WARNING
                add("numeric_distribution", sev, max(psi, ks), min(psi_w, ks_w),
                    f"数值分布漂移：PSI={psi:.3f}（阈值 {psi_w:.2f}），"
                    f"KS={ks:.3f}（阈值 {ks_w:.2f}）")

        # 3) 最值/分位数偏移：超出基线 IQR 的若干倍
        ranges = [(h["profile"].get("numeric") or {}).get("q95") for h in base_cols]
        q25s = [(h["profile"].get("numeric") or {}).get("q25") for h in base_cols]
        q75s = [(h["profile"].get("numeric") or {}).get("q75") for h in base_cols]
        if all(v is not None for v in ranges + q25s + q75s):
            iqr = float(np.median(q75s) - np.median(q25s))
            if iqr > EPS:
                cur_q05 = p["numeric"]["q05"]
                cur_q95 = p["numeric"]["q95"]
                base_q05 = float(np.median([
                    (h["profile"].get("numeric") or {}).get("q05") for h in base_cols]))
                dev = max(abs(cur_q05 - base_q05),
                          abs(cur_q95 - float(np.median(ranges)))) / iqr
                if dev >= thresholds["range_iqr_warning"]:
                    add("range_shift", SEV_WARNING, dev,
                        thresholds["range_iqr_warning"],
                        f"取值范围偏移 {dev:.1f}×IQR：当前 [{p['numeric']['min']:.3g}, "
                        f"{p['numeric']['max']:.3g}]，基线中位区间 "
                        f"[{float(np.median(q25s)):.3g}, {float(np.median(q75s)):.3g}]")

    # 4) 类别分布 PSI（跳过高基数 ID 列）
    if p["inferred_type"] == T_TEXT and not small_sample and \
            p["distinct_ratio"] < thresholds["high_cardinality_ratio"]:
        cur_share = _categorical_share(p)
        base_counter: Counter = Counter()
        base_total = 0
        for h in base_cols:
            for t in (h["profile"].get("text") or {}).get("top_values", []):
                base_counter[t["value"]] += t["count"]
                base_total += t["count"]
        if cur_share and base_total:
            base_share = {k: v / base_total for k, v in base_counter.most_common(TOP_VALUE_CAP)}
            base_share.setdefault("__OTHER__",
                                  max(1.0 - sum(base_share.values()), 0.0))
            psi = psi_from_buckets(base_share, cur_share)
            psi_w = thresholds["psi_warning"]
            if psi >= psi_w:
                sev = SEV_CRITICAL if psi >= thresholds["psi_critical"] else SEV_WARNING
                add("category_distribution", sev, psi, psi_w,
                    f"类别分布漂移：PSI={psi:.3f}（阈值 {psi_w:.2f}）")

    # 5) 文本格式模式漂移
    if p["inferred_type"] == T_TEXT and not small_sample:
        cur_pat = _pattern_share(p)
        base_pat_counter: Counter = Counter()
        for h in base_cols:
            for s in ((h["profile"].get("text") or {}).get("patterns") or {}).get("signatures", []):
                base_pat_counter[s["signature"]] += s["count"]
        base_total = sum(base_pat_counter.values())
        if cur_pat and base_total:
            base_pat = {k: v / base_total for k, v in base_pat_counter.items()}
            psi = psi_from_buckets(base_pat, cur_pat)
            if psi >= thresholds["pattern_psi_warning"]:
                add("pattern_drift", SEV_WARNING, psi,
                    thresholds["pattern_psi_warning"],
                    f"文本格式漂移：格式分布 PSI={psi:.3f}"
                    f"（阈值 {thresholds['pattern_psi_warning']:.2f}）")

    # 6) 布尔占比漂移
    if p["inferred_type"] == T_BOOL and not small_sample:
        cur_bool = _bool_share(p)
        base_true = [ (h["profile"].get("boolean") or {}).get("true_ratio")
                      for h in base_cols]
        base_true = [x for x in base_true if x is not None]
        if cur_bool and base_true:
            z = robust_z(cur_bool["true"], base_true)
            if abs(z) >= thresholds["null_robust_z_warning"] and \
                    abs(cur_bool["true"] - float(np.median(base_true))) >= 0.10:
                add("bool_ratio", SEV_WARNING, z,
                    thresholds["null_robust_z_warning"],
                    f"布尔真值占比异常：{float(np.median(base_true)):.1%} → "
                    f"{cur_bool['true']:.1%}（稳健 z={z:.2f}）")

    # 7) 基数漂移
    #    高基数 ID 列在周末样本缩小时原始基数会等比下降，必须看“去重率”；
    #    低基数枚举列（状态、类型）去重率变化不明显，改看绝对去重值数。
    #    两类指标都对样本量敏感，仅在同星期基线下启用，杜绝首个周末误报。
    if not small_sample and seasonal:
        dc_hist = [h["distinct_count"] for h in base_cols]
        dr_hist = [h["distinct_ratio"] for h in base_cols]
        med_dc = float(np.median(dc_hist))
        med_dr = float(np.median(dr_hist))
        min_count = thresholds["cardinality_min_count"]
        if med_dc < min_count:
            # 低基数枚举列：绝对取值数变化（同星期基线保证样本量可比）
            if abs(p["distinct_count"] - med_dc) >= 2:
                rel = (p["distinct_count"] - med_dc) / max(med_dc, 1.0)
                if abs(rel) >= 0.30:
                    add("cardinality", SEV_WARNING, rel,
                        thresholds["cardinality_rel_warning"],
                        f"基数变化：基线中位 {med_dc:.0f} 个取值 → 当前 "
                        f"{p['distinct_count']} 个（相对变化 {rel:+.1%}）")
        elif med_dr >= 0.20:
            # 高基数列：去重率漂移，天然免疫行数缩放
            if med_dr > EPS:
                rel_dr = abs(p["distinct_ratio"] - med_dr) / med_dr
                if rel_dr >= thresholds["cardinality_rel_warning"]:
                    add("cardinality", SEV_WARNING, rel_dr,
                        thresholds["cardinality_rel_warning"],
                        f"去重率漂移：基线中位 {med_dr:.2%} → 当前 "
                        f"{p['distinct_ratio']:.2%}（相对变化 {rel_dr:+.1%}）")
    return events


def _bucket_share(values: list[float], bins: np.ndarray) -> dict[int, float]:
    idx = np.digitize(values, bins[1:-1])
    counts = Counter(int(i) for i in idx)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}
