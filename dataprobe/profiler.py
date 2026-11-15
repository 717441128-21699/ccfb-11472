"""单日画像构建：空值率、基数、分布、最值、文本格式模式。

数值分布、KS 比较只依赖每列保留的确定性蓄水池样本（默认 300 个），
精确统计量（空值、基数、计数）则基于全量数据，保证画像不随数据量膨胀。
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Any

import numpy as np

from .config import SAMPLE_CAP, TOP_VALUE_CAP
from .infer import (
    T_BOOL, T_DATETIME, T_FLOAT, T_INT, T_TEXT,
    infer_column, parse_numeric, summarize_text_patterns, try_parse_datetime,
)

_BOOL_MAP = {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "y": 1.0,
             "n": 0.0, "t": 1.0, "f": 0.0, "是": 1.0, "否": 0.0,
             "真": 1.0, "假": 0.0}


def _stable_seed(table_id: str, column: str, date: str) -> int:
    h = hashlib.md5(f"{table_id}|{column}|{date}".encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def reservoir_sample(values: list[str], seed: int, cap: int = SAMPLE_CAP) -> list[str]:
    """确定性蓄水池采样：同一份输入永远得到同一份样本。"""
    rng = random.Random(seed)
    if len(values) <= cap:
        return list(values)
    sample = values[:cap]
    for i in range(cap, len(values)):
        j = rng.randrange(i + 1)
        if j < cap:
            sample[j] = values[i]
    return sample


def _numeric_stats(values: list[str]) -> dict[str, Any]:
    nums = [x for x in (parse_numeric(v.strip()) for v in values) if x is not None]
    if not nums:
        return {}
    arr = np.asarray(nums, dtype=float)
    qs = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
    hist, edges = np.histogram(arr, bins=10)
    return {
        "min": float(arr.min()), "max": float(arr.max()),
        "mean": round(float(arr.mean()), 6), "std": round(float(arr.std(ddof=0)), 6),
        "q05": float(qs[0]), "q25": float(qs[1]), "median": float(qs[2]),
        "q75": float(qs[3]), "q95": float(qs[4]),
        "histogram": {"counts": [int(c) for c in hist],
                      "edges": [round(float(e), 6) for e in edges]},
        "sample_numbers": [round(float(x), 6) for x in
                           np.asarray(nums[:SAMPLE_CAP], dtype=float)],
    }


def _datetime_stats(values: list[str]) -> dict[str, Any]:
    dts = [d for d in (try_parse_datetime(v.strip()) for v in values) if d is not None]
    if not dts:
        return {}
    dmin, dmax = min(dts), max(dts)
    ts = np.asarray([d.timestamp() for d in dts], dtype=float)
    hist, edges = np.histogram(ts, bins=10)
    edge_dates = [datetime_from_ts(float(e)).strftime("%Y-%m-%d %H:%M:%S") for e in edges]
    weekday_counts = Counter(d.weekday() for d in dts)
    return {
        "min": dmin.strftime("%Y-%m-%d %H:%M:%S"),
        "max": dmax.strftime("%Y-%m-%d %H:%M:%S"),
        "range_days": round((dmax - dmin).total_seconds() / 86400.0, 3),
        "weekday_dist": {str(k): weekday_counts.get(k, 0) for k in range(7)},
        "histogram": {"counts": [int(c) for c in hist], "edges": edge_dates},
    }


def datetime_from_ts(ts: float):
    from datetime import datetime
    return datetime.fromtimestamp(ts)


def _boolean_stats(values: list[str]) -> dict[str, Any]:
    flags = [_BOOL_MAP[v.strip().lower()] for v in values if v.strip().lower() in _BOOL_MAP]
    if not flags:
        return {}
    true_count = sum(1 for x in flags if x == 1.0)
    return {"true_count": true_count, "false_count": len(flags) - true_count,
            "true_ratio": round(true_count / len(flags), 6)}


def _text_stats(values: list[str]) -> dict[str, Any]:
    non_empty = [v.strip() for v in values if v.strip()]
    if not non_empty:
        return {}
    lengths = np.asarray([len(v) for v in non_empty])
    counter = Counter(non_empty)
    top = [{"value": v, "count": c,
            "ratio": round(c / len(non_empty), 6)}
           for v, c in counter.most_common(TOP_VALUE_CAP)]
    patterns = summarize_text_patterns(non_empty)
    return {
        "top_values": top,
        "min_length": int(lengths.min()), "max_length": int(lengths.max()),
        "avg_length": round(float(lengths.mean()), 3),
        "patterns": patterns,
    }


def profile_column(table_id: str, snapshot_date: str, column_name: str, ordinal: int,
                   raw_values: list[str]) -> dict[str, Any]:
    """生成单列单日画像。raw_values 为与表行数等长的原始字符串列表（空串代表空值）。"""
    n_total = len(raw_values)
    non_null = [v for v in raw_values if v.strip() != ""]
    null_count = n_total - len(non_null)
    inferred_type = infer_column(non_null)

    distinct = Counter(non_null)
    distinct_count = len(distinct)
    distinct_ratio = round(distinct_count / len(non_null), 6) if non_null else 0.0

    seed = _stable_seed(table_id, column_name, snapshot_date)
    sample = reservoir_sample(non_null, seed)

    profile: dict[str, Any] = {
        "column_name": column_name,
        "ordinal": ordinal,
        "inferred_type": inferred_type,
        "null_count": null_count,
        "null_rate": round(null_count / n_total, 6) if n_total else 0.0,
        "total_count": n_total,
        "distinct_count": distinct_count,
        "distinct_ratio": distinct_ratio,
        "sample_values": sample[:SAMPLE_CAP],
    }

    if inferred_type in (T_INT, T_FLOAT):
        profile["numeric"] = _numeric_stats(non_null)
    elif inferred_type == T_DATETIME:
        profile["datetime"] = _datetime_stats(non_null)
    elif inferred_type == T_BOOL:
        profile["boolean"] = _boolean_stats(non_null)
    else:
        profile["text"] = _text_stats(non_null)
    return profile


def profile_table(snapshot_date: str, table_id: str, columns: list[str],
                  data: dict[str, list[str]]) -> tuple[dict, list[dict]]:
    """生成整表单日画像，返回 (表级画像, 各列存储行)。"""
    row_count = len(data[columns[0]]) if columns else 0
    type_counter: Counter[str] = Counter()
    column_rows: list[dict] = []
    for ordinal, col in enumerate(columns):
        cp = profile_column(table_id, snapshot_date, col, ordinal, data[col])
        type_counter[cp["inferred_type"]] += 1
        column_rows.append({
            "column_name": col,
            "ordinal": ordinal,
            "inferred_type": cp["inferred_type"],
            "null_count": cp["null_count"],
            "null_rate": cp["null_rate"],
            "distinct_count": cp["distinct_count"],
            "distinct_ratio": cp["distinct_ratio"],
            "profile": cp,
            # stable_col_id 由 identity 模块在画像写入前补全
            "stable_col_id": None,
        })
    table_profile = {
        "table_id": table_id,
        "snapshot_date": snapshot_date,
        "row_count": row_count,
        "column_count": len(columns),
        "type_distribution": dict(type_counter),
    }
    return table_profile, column_rows
