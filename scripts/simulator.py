"""模拟 30 张表连续 60 天的数据，注入 20 个已知漂移（覆盖 14 类漂移）。

数据按天落地：
  data/sources/csv/<table>/<date>.csv
  data/sources/parquet/<table>/<date>.parquet
  data/sources/sqlite/<table>.db          （每天一个表 t_<date>）
同时输出 data/ground_truth.json 作为评估真值。

所有随机数使用固定种子，可完全复现。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
N_DAYS = 60
START = date(2026, 9, 1)          # 2026-09-01 是周二，60 天覆盖完整周末周期
DRIFT_BEGIN = 35                  # 第 36 天起注入漂移（前 35 天为干净基线）
WEEKEND_FACTOR = 0.45             # 周末量少：行数约为工作日的 45%


# ---------------- 规格定义 ----------------

@dataclass
class Drift:
    kind: str
    column: str
    start: int = DRIFT_BEGIN
    end: int = N_DAYS
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 兼容 end_day 写法
        if "end_day" in self.params:
            self.end = int(self.params.pop("end_day"))


@dataclass
class TableSpec:
    name: str
    kind: str                       # csv / parquet / sqlite
    base_rows: int
    columns: list[dict]
    drifts: list[Drift] = field(default_factory=list)
    weekend_factor: float = WEEKEND_FACTOR
    volume_drift: dict | None = None


# ---------------- 列生成器 ----------------

def gen_int(rng: np.random.Generator, n: int, p: dict) -> np.ndarray:
    return rng.integers(p.get("lo", 0), p.get("hi", 100), n)


def gen_float(rng: np.random.Generator, n: int, p: dict) -> np.ndarray:
    return rng.normal(p.get("mu", 100), p.get("sigma", 15), n).round(2)


def gen_uniform(rng: np.random.Generator, n: int, p: dict) -> np.ndarray:
    return rng.uniform(p.get("lo", 0.0), p.get("hi", 1.0), n).round(4)


def gen_categorical(rng: np.random.Generator, n: int, p: dict) -> np.ndarray:
    cats = p["categories"]
    probs = p.get("probs")
    return rng.choice(np.array(cats, dtype=object), n, p=probs)


def gen_bool(rng: np.random.Generator, n: int, p: dict) -> np.ndarray:
    return np.where(rng.random(n) < p.get("true_prob", 0.5), "true", "false")


def gen_phone(rng: np.random.Generator, n: int, p: dict) -> np.ndarray:
    prefixes = np.array(["138", "139", "136", "158", "188", "199"], dtype=object)
    pre = rng.choice(prefixes, n)
    tail = rng.integers(0, 10**8, n)
    return np.array([f"{p_}{t:08d}" for p_, t in zip(pre, tail)], dtype=object)


def gen_idcard(rng: np.random.Generator, n: int, p: dict) -> np.ndarray:
    return np.array([f"1101{rng.integers(0, 10**14):014d}" for _ in range(n)], dtype=object)


def gen_code(rng: np.random.Generator, n: int, p: dict) -> np.ndarray:
    prefix = p.get("prefix", "ORD")
    width = p.get("width", 6)
    return np.array([f"{prefix}-{rng.integers(0, 10**width):0{width}d}" for _ in range(n)],
                    dtype=object)


def gen_email(rng: np.random.Generator, n: int, p: dict) -> np.ndarray:
    domains = np.array(["example.com", "test.cn", "mail.com"], dtype=object)
    names = np.array(["user", "cust", "member", "vip"], dtype=object)
    u = rng.choice(names, n)
    d = rng.choice(domains, n)
    num = rng.integers(10, 9999, n)
    return np.array([f"{a}{b}@{c}" for a, b, c in zip(u, num, d)], dtype=object)


def gen_date(rng: np.random.Generator, n: int, p: dict) -> np.ndarray:
    start = pd.Timestamp(p.get("start", "2026-01-01"))
    span_days = p.get("span_days", 240)
    offsets = rng.integers(0, span_days, n)
    ts = start + pd.to_timedelta(offsets, unit="D")
    return ts.strftime("%Y-%m-%d").to_numpy(dtype=object)


def gen_name(rng: np.random.Generator, n: int, p: dict) -> np.ndarray:
    surnames = list("王李张刘陈杨黄赵周吴徐孙马朱胡郭何高林罗")
    given = list("伟芳娜敏静丽强磊军洋勇艳杰娟涛明超秀兰霞平刚桂英华")
    return np.array([f"{rng.choice(surnames)}{rng.choice(given)}"
                     + ("" if rng.random() < 0.5 else rng.choice(given))
                     for _ in range(n)], dtype=object)


GENERATORS = {
    "int": gen_int, "float": gen_float, "uniform": gen_uniform,
    "categorical": gen_categorical, "bool": gen_bool,
    "phone": gen_phone, "idcard": gen_idcard, "code": gen_code,
    "email": gen_email, "date": gen_date, "name": gen_name,
}


# ---------------- 漂移注入 ----------------

def apply_drift(rng: np.random.Generator, values: np.ndarray, col: dict,
                d: Drift, day_idx: int) -> np.ndarray:
    p = d.params
    out = values.copy()
    n = len(out)
    kind = d.kind

    if kind == "null_spike":
        k = int(n * p["rate"])
        idx = rng.choice(n, k, replace=False)
        out[idx] = ""
    elif kind == "mean_shift":
        mask = out != ""
        if col["gen"] in ("float",):
            out[mask] = np.round(out[mask].astype(float) + p["delta"], 2)
        else:
            out[mask] = out[mask].astype(int) + int(p["delta"])
    elif kind == "scale_change":
        mask = out != ""
        out[mask] = np.round(out[mask].astype(float) * p["factor"], 2)
    elif kind == "outlier_injection":
        k = max(int(n * p["rate"]), 1)
        idx = rng.choice(n, k, replace=False)
        out[idx] = np.round(rng.normal(p["mu"], p.get("sigma", 1.0), k), 2)
    elif kind == "category_proportion":
        out = gen_categorical(rng, n,
                              {"categories": p["categories"], "probs": p["probs"]})
    elif kind == "new_category":
        k = max(int(n * p["rate"]), 1)
        idx = rng.choice(n, k, replace=False)
        out[idx] = p["value"]
    elif kind == "bool_flip":
        out = np.where(rng.random(n) < p["true_prob"], "true", "false")
    elif kind == "pattern_change":
        k = max(int(n * p["rate"]), 1)
        idx = rng.choice(n, k, replace=False)
        if p["style"] == "phone_to_landline":
            out[idx] = np.array([f"010-{rng.integers(10**7, 10**8)}"
                                  for _ in range(k)], dtype=object)
        elif p["style"] == "email_add_quotes":
            out[idx] = np.array([f'"{v}"' for v in out[idx]], dtype=object)
        elif p["style"] == "code_prefix":
            out[idx] = np.array([f"X-{v}" for v in out[idx]], dtype=object)
    elif kind == "gradual_shift":
        # 线性渐变：每天多移一点，第 end 天达到满幅 delta
        progress = min((day_idx - d.start) / max(d.end - d.start, 1), 1.0)
        shift = p["delta"] * progress
        mask = out != ""
        if col["gen"] == "float":
            out[mask] = np.round(out[mask].astype(float) + shift, 2)
        else:
            out[mask] = out[mask].astype(int) + int(round(shift))
    elif kind == "cardinality_collapse":
        keep = p.get("keep_values", 3)
        uniq = pd.unique(out)
        if col["gen"] == "categorical" and len(uniq) > keep:
            allowed = uniq[:keep]
            out = rng.choice(allowed, n)
        else:
            # 数值列：截断到少量离散值（保留空值）
            mask = out != ""
            qs = np.quantile(out[mask].astype(float),
                              np.linspace(0, 1, keep + 1)[:-1])
            out[mask] = rng.choice(qs, int(mask.sum())).round(2)
    elif kind == "range_expand":
        k = max(int(n * p["rate"]), 1)
        idx = rng.choice(n, k, replace=False)
        out[idx] = rng.uniform(p["lo"], p["hi"], k).round(2)
    elif kind == "unit_change":
        mask = out != ""
        out[mask] = np.round(out[mask].astype(float) * p["factor"], 2)
    else:
        raise ValueError(f"未知漂移类型: {kind}")
    return out


# ---------------- 30 张表规格 ----------------

def build_specs() -> list[TableSpec]:
    specs: list[TableSpec] = []

    def csv_spec(name, rows, cols, drifts=None, **kw):
        specs.append(TableSpec(name, "csv", rows, cols, drifts or [], **kw))

    def pq_spec(name, rows, cols, drifts=None, **kw):
        specs.append(TableSpec(name, "parquet", rows, cols, drifts or [], **kw))

    def sq_spec(name, rows, cols, drifts=None, **kw):
        specs.append(TableSpec(name, "sqlite", rows, cols, drifts or [], **kw))

    # ---- 12 张 CSV ----
    csv_spec("orders", 3000, [
        {"name": "order_id", "gen": "code", "params": {"prefix": "ORD", "width": 8}, "null_rate": 0.0},
        {"name": "customer_id", "gen": "int", "params": {"lo": 10000, "hi": 99999}, "null_rate": 0.02},
        {"name": "amount", "gen": "float", "params": {"mu": 260, "sigma": 45}, "null_rate": 0.01},
        {"name": "quantity", "gen": "int", "params": {"lo": 1, "hi": 12}, "null_rate": 0.0},
        {"name": "channel", "gen": "categorical",
         "params": {"categories": ["web", "app", "mini", "offline"],
                    "probs": [0.40, 0.35, 0.15, 0.10]}, "null_rate": 0.0},
        {"name": "order_status", "gen": "categorical",
         "params": {"categories": ["paid", "pending", "refund", "closed"],
                    "probs": [0.70, 0.15, 0.10, 0.05]}, "null_rate": 0.0},
        {"name": "contact_phone", "gen": "phone", "params": {}, "null_rate": 0.03},
        {"name": "created_date", "gen": "date", "params": {"span_days": 30}, "null_rate": 0.0},
    ], drifts=[
        Drift("mean_shift", "amount", params={"delta": 130}),                 # D1
        Drift("category_proportion", "channel",
              params={"categories": ["web", "app", "mini", "offline"],
                      "probs": [0.15, 0.55, 0.20, 0.10]}),                   # D2
    ])

    csv_spec("customers", 2000, [
        {"name": "cust_id", "gen": "int", "params": {"lo": 1, "hi": 10**6}, "null_rate": 0.0},
        {"name": "id_card_no", "gen": "idcard", "params": {}, "null_rate": 0.05},
        {"name": "mobile", "gen": "phone", "params": {}, "null_rate": 0.05},
        {"name": "city_tier", "gen": "categorical",
         "params": {"categories": ["T1", "T2", "T3", "T4"],
                    "probs": [0.25, 0.35, 0.25, 0.15]}, "null_rate": 0.0},
        {"name": "age", "gen": "int", "params": {"lo": 18, "hi": 70}, "null_rate": 0.02},
        {"name": "credit_score", "gen": "float", "params": {"mu": 650, "sigma": 60}, "null_rate": 0.04},
        {"name": "email_addr", "gen": "email", "params": {}, "null_rate": 0.08},
    ], drifts=[
        Drift("null_spike", "mobile", params={"rate": 0.42}),                # D3
        Drift("pattern_change", "email_addr",
              params={"style": "email_add_quotes", "rate": 0.55}),           # D4
    ])

    csv_spec("payments", 2500, [
        {"name": "pay_id", "gen": "code", "params": {"prefix": "PAY", "width": 9}, "null_rate": 0.0},
        {"name": "order_ref", "gen": "code", "params": {"prefix": "ORD", "width": 8}, "null_rate": 0.0},
        {"name": "pay_amount", "gen": "float", "params": {"mu": 240, "sigma": 50}, "null_rate": 0.0},
        {"name": "fee", "gen": "float", "params": {"mu": 12, "sigma": 3}, "null_rate": 0.01},
        {"name": "pay_method", "gen": "categorical",
         "params": {"categories": ["card", "wechat", "alipay", "cash"],
                    "probs": [0.3, 0.3, 0.3, 0.1]}, "null_rate": 0.0},
        {"name": "is_success", "gen": "bool", "params": {"true_prob": 0.92}, "null_rate": 0.0},
        {"name": "pay_time", "gen": "date", "params": {"span_days": 15}, "null_rate": 0.0},
    ], drifts=[
        Drift("bool_flip", "is_success", params={"true_prob": 0.62}),       # D5
        Drift("new_category", "pay_method",
              params={"value": "digital_yuan", "rate": 0.22}),               # D6
    ])

    csv_spec("inventory", 1500, [
        {"name": "sku_id", "gen": "code", "params": {"prefix": "SKU", "width": 6}, "null_rate": 0.0},
        {"name": "warehouse", "gen": "categorical",
         "params": {"categories": ["WH1", "WH2", "WH3", "WH4", "WH5"],
                    "probs": [0.3, 0.25, 0.2, 0.15, 0.1]}, "null_rate": 0.0},
        {"name": "stock_qty", "gen": "int", "params": {"lo": 0, "hi": 500}, "null_rate": 0.01},
        {"name": "unit_cost", "gen": "float", "params": {"mu": 80, "sigma": 12}, "null_rate": 0.0},
        {"name": "shelf_life_days", "gen": "int", "params": {"lo": 30, "hi": 365}, "null_rate": 0.05},
        {"name": "supplier_code", "gen": "code", "params": {"prefix": "SUP", "width": 4}, "null_rate": 0.02},
    ], drifts=[
        Drift("outlier_injection", "unit_cost",
              params={"rate": 0.08, "mu": 400, "sigma": 30}),               # D7
    ])

    csv_spec("clicks", 4000, [
        {"name": "event_id", "gen": "int", "params": {"lo": 1, "hi": 10**7}, "null_rate": 0.0},
        {"name": "user_id", "gen": "int", "params": {"lo": 1, "hi": 200000}, "null_rate": 0.0},
        {"name": "page_id", "gen": "categorical",
         "params": {"categories": ["home", "search", "detail", "cart", "mine"],
                    "probs": [0.3, 0.25, 0.25, 0.1, 0.1]}, "null_rate": 0.0},
        {"name": "dwell_seconds", "gen": "float", "params": {"mu": 35, "sigma": 10}, "null_rate": 0.02},
        {"name": "device", "gen": "categorical",
         "params": {"categories": ["ios", "android", "pc"],
                    "probs": [0.35, 0.45, 0.20]}, "null_rate": 0.0},
        {"name": "click_date", "gen": "date", "params": {"span_days": 7}, "null_rate": 0.0},
    ], drifts=[
        Drift("scale_change", "dwell_seconds", params={"factor": 2.2}),      # D8
    ], volume_drift={"factor": 0.25, "desc": "埋点丢失导致行数骤降"})        # D9

    csv_spec("refunds", 900, [
        {"name": "refund_id", "gen": "code", "params": {"prefix": "RF", "width": 7}, "null_rate": 0.0},
        {"name": "order_id", "gen": "code", "params": {"prefix": "ORD", "width": 8}, "null_rate": 0.0},
        {"name": "refund_amount", "gen": "float", "params": {"mu": 150, "sigma": 40}, "null_rate": 0.0},
        {"name": "reason", "gen": "categorical",
         "params": {"categories": ["quality", "logistics", "mistake", "other"],
                    "probs": [0.4, 0.3, 0.2, 0.1]}, "null_rate": 0.03},
        {"name": "handler", "gen": "name", "params": {}, "null_rate": 0.05},
        {"name": "apply_date", "gen": "date", "params": {"span_days": 20}, "null_rate": 0.0},
    ], drifts=[
        Drift("gradual_shift", "refund_amount",
              params={"delta": 220, "end_day": 55},
              end=55),                                                       # D10
    ])

    csv_spec("logistics", 1800, [
        {"name": "shipment_id", "gen": "code", "params": {"prefix": "SF", "width": 8}, "null_rate": 0.0},
        {"name": "from_city", "gen": "categorical",
         "params": {"categories": ["BJ", "SH", "GZ", "SZ", "HZ", "CD"],
                    "probs": [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]}, "null_rate": 0.0},
        {"name": "to_city", "gen": "categorical",
         "params": {"categories": ["BJ", "SH", "GZ", "SZ", "HZ", "CD", "WH"],
                    "probs": [0.18, 0.18, 0.18, 0.14, 0.14, 0.09, 0.09]}, "null_rate": 0.0},
        {"name": "weight_kg", "gen": "float", "params": {"mu": 3.5, "sigma": 1.2}, "null_rate": 0.02},
        {"name": "duration_h", "gen": "float", "params": {"mu": 48, "sigma": 14}, "null_rate": 0.01},
        {"name": "carrier", "gen": "categorical",
         "params": {"categories": ["sf", "yto", "zto", "sto"],
                    "probs": [0.3, 0.25, 0.25, 0.2]}, "null_rate": 0.0},
        {"name": "receiver_phone", "gen": "phone", "params": {}, "null_rate": 0.10},
    ], drifts=[
        Drift("pattern_change", "receiver_phone",
              params={"style": "phone_to_landline", "rate": 0.6}),           # D11
    ])

    csv_spec("marketing", 1200, [
        {"name": "campaign_id", "gen": "code", "params": {"prefix": "CMP", "width": 5}, "null_rate": 0.0},
        {"name": "channel_name", "gen": "categorical",
         "params": {"categories": ["feed", "search", "display", "sms", "edm"],
                    "probs": [0.3, 0.25, 0.2, 0.15, 0.1]}, "null_rate": 0.0},
        {"name": "spend", "gen": "float", "params": {"mu": 5000, "sigma": 900}, "null_rate": 0.0},
        {"name": "impressions", "gen": "int", "params": {"lo": 1000, "hi": 90000}, "null_rate": 0.0},
        {"name": "ctr", "gen": "uniform", "params": {"lo": 0.005, "hi": 0.08}, "null_rate": 0.01},
        {"name": "is_active", "gen": "bool", "params": {"true_prob": 0.75}, "null_rate": 0.0},
    ], drifts=[
        Drift("range_expand", "spend",
              params={"rate": 0.1, "lo": 20000, "hi": 40000}),              # D12
    ])

    csv_spec("after_sales", 700, [
        {"name": "ticket_id", "gen": "code", "params": {"prefix": "TK", "width": 6}, "null_rate": 0.0},
        {"name": "cust_id", "gen": "int", "params": {"lo": 1, "hi": 100000}, "null_rate": 0.02},
        {"name": "satisfaction", "gen": "int", "params": {"lo": 1, "hi": 5}, "null_rate": 0.05},
        {"name": "handle_hours", "gen": "float", "params": {"mu": 8, "sigma": 2.5}, "null_rate": 0.03},
        {"name": "ticket_type", "gen": "categorical",
         "params": {"categories": ["consult", "complaint", "return", "repair"],
                    "probs": [0.4, 0.2, 0.25, 0.15]}, "null_rate": 0.0},
        {"name": "agent", "gen": "name", "params": {}, "null_rate": 0.0},
    ], drifts=[
        Drift("cardinality_collapse", "ticket_type",
              params={"keep_values": 2}),                                    # D13
    ])

    # 3 张干净对照 CSV
    csv_spec("clean_hr", 600, [
        {"name": "emp_id", "gen": "code", "params": {"prefix": "EMP", "width": 5}, "null_rate": 0.0},
        {"name": "department", "gen": "categorical",
         "params": {"categories": ["rd", "sales", "ops", "finance", "hr"],
                    "probs": [0.35, 0.25, 0.2, 0.1, 0.1]}, "null_rate": 0.0},
        {"name": "salary", "gen": "float", "params": {"mu": 15000, "sigma": 2500}, "null_rate": 0.0},
        {"name": "work_years", "gen": "int", "params": {"lo": 0, "hi": 25}, "null_rate": 0.01},
        {"name": "on_job", "gen": "bool", "params": {"true_prob": 0.95}, "null_rate": 0.0},
        {"name": "entry_date", "gen": "date", "params": {"start": "2018-01-01", "span_days": 2800}, "null_rate": 0.0},
    ])
    csv_spec("clean_finance", 800, [
        {"name": "voucher_no", "gen": "code", "params": {"prefix": "V", "width": 9}, "null_rate": 0.0},
        {"name": "subject", "gen": "categorical",
         "params": {"categories": ["cash", "bank", "ar", "ap", "tax"],
                    "probs": [0.15, 0.25, 0.25, 0.2, 0.15]}, "null_rate": 0.0},
        {"name": "debit", "gen": "float", "params": {"mu": 3200, "sigma": 700}, "null_rate": 0.0},
        {"name": "credit", "gen": "float", "params": {"mu": 3200, "sigma": 700}, "null_rate": 0.0},
        {"name": "post_date", "gen": "date", "params": {"span_days": 10}, "null_rate": 0.0},
    ])
    csv_spec("clean_geo", 1000, [
        {"name": "region_code", "gen": "code", "params": {"prefix": "R", "width": 6}, "null_rate": 0.0},
        {"name": "province", "gen": "categorical",
         "params": {"categories": ["HB", "HN", "GD", "JS", "ZJ", "SC"],
                    "probs": [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]}, "null_rate": 0.0},
        {"name": "latitude", "gen": "float", "params": {"mu": 32, "sigma": 5}, "null_rate": 0.0},
        {"name": "longitude", "gen": "float", "params": {"mu": 112, "sigma": 7}, "null_rate": 0.0},
        {"name": "population", "gen": "int", "params": {"lo": 100000, "hi": 9000000}, "null_rate": 0.02},
    ])

    # ---- 10 张 Parquet ----
    pq_spec("risk_scores", 2200, [
        {"name": "apply_id", "gen": "code", "params": {"prefix": "AP", "width": 8}, "null_rate": 0.0},
        {"name": "cust_id", "gen": "int", "params": {"lo": 1, "hi": 100000}, "null_rate": 0.0},
        {"name": "risk_score", "gen": "float", "params": {"mu": 500, "sigma": 80}, "null_rate": 0.03},
        {"name": "overdue_days", "gen": "int", "params": {"lo": 0, "hi": 30}, "null_rate": 0.10},
        {"name": "risk_level", "gen": "categorical",
         "params": {"categories": ["low", "mid", "high"],
                    "probs": [0.6, 0.3, 0.1]}, "null_rate": 0.0},
        {"name": "model_version", "gen": "categorical",
         "params": {"categories": ["v2.1", "v2.2"], "probs": [0.8, 0.2]}, "null_rate": 0.0},
        {"name": "apply_date", "gen": "date", "params": {"span_days": 25}, "null_rate": 0.0},
    ], drifts=[
        Drift("mean_shift", "risk_score", params={"delta": -160}),           # D14
    ])

    pq_spec("iot_sensors", 3500, [
        {"name": "device_id", "gen": "code", "params": {"prefix": "DEV", "width": 6}, "null_rate": 0.0},
        {"name": "temperature", "gen": "float", "params": {"mu": 24, "sigma": 1.5}, "null_rate": 0.01},
        {"name": "humidity", "gen": "float", "params": {"mu": 55, "sigma": 6}, "null_rate": 0.01},
        {"name": "pressure", "gen": "float", "params": {"mu": 101, "sigma": 2}, "null_rate": 0.02},
        {"name": "site", "gen": "categorical",
         "params": {"categories": ["north", "south", "east", "west"],
                    "probs": [0.3, 0.3, 0.2, 0.2]}, "null_rate": 0.0},
        {"name": "reading_time", "gen": "date", "params": {"span_days": 5}, "null_rate": 0.0},
    ], drifts=[
        Drift("mean_shift", "temperature", params={"delta": 6}),             # D15
    ])

    pq_spec("user_behavior", 2800, [
        {"name": "ub_id", "gen": "int", "params": {"lo": 1, "hi": 10**7}, "null_rate": 0.0},
        {"name": "uid", "gen": "int", "params": {"lo": 1, "hi": 200000}, "null_rate": 0.0},
        {"name": "event", "gen": "categorical",
         "params": {"categories": ["view", "like", "share", "buy", "collect"],
                    "probs": [0.45, 0.25, 0.1, 0.08, 0.12]}, "null_rate": 0.0},
        {"name": "duration", "gen": "float", "params": {"mu": 60, "sigma": 18}, "null_rate": 0.02},
        {"name": "network", "gen": "categorical",
         "params": {"categories": ["5g", "wifi", "4g"],
                    "probs": [0.3, 0.5, 0.2]}, "null_rate": 0.0},
    ], drifts=[
        Drift("category_proportion", "event",
              params={"categories": ["view", "like", "share", "buy", "collect"],
                      "probs": [0.2, 0.15, 0.1, 0.4, 0.15]}),               # D16
    ])

    pq_spec("transactions", 3200, [
        {"name": "txn_id", "gen": "code", "params": {"prefix": "TX", "width": 9}, "null_rate": 0.0},
        {"name": "acct_id", "gen": "int", "params": {"lo": 100000, "hi": 999999}, "null_rate": 0.0},
        {"name": "txn_amount", "gen": "float", "params": {"mu": 800, "sigma": 200}, "null_rate": 0.0},
        {"name": "currency", "gen": "categorical",
         "params": {"categories": ["CNY"], "probs": [1.0]}, "null_rate": 0.0},
        {"name": "channel", "gen": "categorical",
         "params": {"categories": ["branch", "atm", "online"],
                    "probs": [0.2, 0.2, 0.6]}, "null_rate": 0.0},
        {"name": "is_flagged", "gen": "bool", "params": {"true_prob": 0.05}, "null_rate": 0.0},
    ], drifts=[
        Drift("unit_change", "txn_amount", params={"factor": 0.01}),         # D17
    ])

    pq_spec("members", 1600, [
        {"name": "member_no", "gen": "code", "params": {"prefix": "M", "width": 7}, "null_rate": 0.0},
        {"name": "level", "gen": "categorical",
         "params": {"categories": ["bronze", "silver", "gold", "diamond"],
                    "probs": [0.4, 0.3, 0.2, 0.1]}, "null_rate": 0.0},
        {"name": "points", "gen": "int", "params": {"lo": 0, "hi": 20000}, "null_rate": 0.03},
        {"name": "discount_rate", "gen": "uniform", "params": {"lo": 0.9, "hi": 1.0}, "null_rate": 0.0},
        {"name": "register_date", "gen": "date", "params": {"start": "2020-01-01", "span_days": 2200}, "null_rate": 0.0},
        {"name": "vip_phone", "gen": "phone", "params": {}, "null_rate": 0.06},
    ], drifts=[
        Drift("pattern_change", "vip_phone",
              params={"style": "code_prefix", "rate": 0.5}),                 # D18
    ])

    pq_spec("products", 1100, [
        {"name": "product_id", "gen": "code", "params": {"prefix": "P", "width": 6}, "null_rate": 0.0},
        {"name": "category", "gen": "categorical",
         "params": {"categories": ["food", "digital", "cloth", "home", "beauty"],
                    "probs": [0.25, 0.2, 0.25, 0.15, 0.15]}, "null_rate": 0.0},
        {"name": "price", "gen": "float", "params": {"mu": 199, "sigma": 40}, "null_rate": 0.0},
        {"name": "weight_g", "gen": "int", "params": {"lo": 100, "hi": 5000}, "null_rate": 0.02},
        {"name": "rating", "gen": "uniform", "params": {"lo": 3.0, "hi": 5.0}, "null_rate": 0.04},
        {"name": "on_sale", "gen": "bool", "params": {"true_prob": 0.88}, "null_rate": 0.0},
    ])

    pq_spec("ads_attribution", 1400, [
        {"name": "attr_id", "gen": "int", "params": {"lo": 1, "hi": 10**6}, "null_rate": 0.0},
        {"name": "campaign", "gen": "categorical",
         "params": {"categories": ["c1", "c2", "c3", "c4"],
                    "probs": [0.3, 0.3, 0.25, 0.15]}, "null_rate": 0.0},
        {"name": "conversion_value", "gen": "float", "params": {"mu": 320, "sigma": 70}, "null_rate": 0.0},
        {"name": "touch_count", "gen": "int", "params": {"lo": 1, "hi": 8}, "null_rate": 0.01},
        {"name": "attr_window_h", "gen": "int", "params": {"lo": 24, "hi": 720}, "null_rate": 0.0},
    ])

    pq_spec("call_center", 950, [
        {"name": "call_id", "gen": "code", "params": {"prefix": "CC", "width": 7}, "null_rate": 0.0},
        {"name": "caller_phone", "gen": "phone", "params": {}, "null_rate": 0.12},
        {"name": "wait_seconds", "gen": "float", "params": {"mu": 25, "sigma": 8}, "null_rate": 0.0},
        {"name": "talk_minutes", "gen": "float", "params": {"mu": 6, "sigma": 2.2}, "null_rate": 0.01},
        {"name": "call_type", "gen": "categorical",
         "params": {"categories": ["in", "out", "transfer"],
                    "probs": [0.6, 0.3, 0.1]}, "null_rate": 0.0},
        {"name": "resolved", "gen": "bool", "params": {"true_prob": 0.8}, "null_rate": 0.0},
    ])

    pq_spec("supply_chain", 1300, [
        {"name": "po_no", "gen": "code", "params": {"prefix": "PO", "width": 8}, "null_rate": 0.0},
        {"name": "supplier", "gen": "categorical",
         "params": {"categories": ["s1", "s2", "s3", "s4", "s5"],
                    "probs": [0.25, 0.25, 0.2, 0.15, 0.15]}, "null_rate": 0.0},
        {"name": "lead_time_d", "gen": "int", "params": {"lo": 3, "hi": 20}, "null_rate": 0.02},
        {"name": "defect_rate", "gen": "uniform", "params": {"lo": 0.0, "hi": 0.05}, "null_rate": 0.03},
        {"name": "order_value", "gen": "float", "params": {"mu": 9000, "sigma": 1500}, "null_rate": 0.0},
    ])

    pq_spec("app_crash", 1700, [
        {"name": "crash_id", "gen": "int", "params": {"lo": 1, "hi": 10**7}, "null_rate": 0.0},
        {"name": "app_version", "gen": "categorical",
         "params": {"categories": ["8.1", "8.2", "8.3"],
                    "probs": [0.2, 0.5, 0.3]}, "null_rate": 0.0},
        {"name": "os_version", "gen": "categorical",
         "params": {"categories": ["ios16", "ios17", "ios18"],
                    "probs": [0.2, 0.5, 0.3]}, "null_rate": 0.0},
        {"name": "crash_rate", "gen": "uniform", "params": {"lo": 0.001, "hi": 0.02}, "null_rate": 0.0},
        {"name": "memory_mb", "gen": "float", "params": {"mu": 320, "sigma": 60}, "null_rate": 0.01},
    ])

    # ---- 8 张 SQLite ----
    sq_spec("bank_cards", 2000, [
        {"name": "card_id", "gen": "code", "params": {"prefix": "CARD", "width": 8}, "null_rate": 0.0},
        {"name": "cust_id", "gen": "int", "params": {"lo": 1, "hi": 100000}, "null_rate": 0.0},
        {"name": "balance", "gen": "float", "params": {"mu": 8000, "sigma": 2500}, "null_rate": 0.02},
        {"name": "card_type", "gen": "categorical",
         "params": {"categories": ["debit", "credit", "prepaid"],
                    "probs": [0.5, 0.4, 0.1]}, "null_rate": 0.0},
        {"name": "credit_limit", "gen": "int", "params": {"lo": 5000, "hi": 100000}, "null_rate": 0.08},
        {"name": "open_date", "gen": "date", "params": {"start": "2019-01-01", "span_days": 2500}, "null_rate": 0.0},
        {"name": "is_frozen", "gen": "bool", "params": {"true_prob": 0.03}, "null_rate": 0.0},
    ], drifts=[
        Drift("null_spike", "balance", params={"rate": 0.35}),                # D19
    ])

    sq_spec("loan_apps", 1500, [
        {"name": "loan_id", "gen": "code", "params": {"prefix": "LN", "width": 8}, "null_rate": 0.0},
        {"name": "applicant_id", "gen": "int", "params": {"lo": 1, "hi": 80000}, "null_rate": 0.0},
        {"name": "loan_amount", "gen": "float", "params": {"mu": 50000, "sigma": 12000}, "null_rate": 0.0},
        {"name": "annual_rate", "gen": "uniform", "params": {"lo": 0.035, "hi": 0.18}, "null_rate": 0.0},
        {"name": "term_months", "gen": "int", "params": {"lo": 3, "hi": 36}, "null_rate": 0.0},
        {"name": "loan_status", "gen": "categorical",
         "params": {"categories": ["approve", "reject", "pending"],
                    "probs": [0.55, 0.25, 0.2]}, "null_rate": 0.0},
    ], drifts=[
        Drift("category_proportion", "loan_status",
              params={"categories": ["approve", "reject", "pending"],
                      "probs": [0.2, 0.6, 0.2]}),                          # D20
    ])

    sq_spec("trades_rename", 2400, [
        {"name": "trade_no", "gen": "code", "params": {"prefix": "TR", "width": 8}, "null_rate": 0.0},
        {"name": "book_amount", "gen": "float", "params": {"mu": 700, "sigma": 120}, "null_rate": 0.0},
        {"name": "book_quantity", "gen": "int", "params": {"lo": 1, "hi": 30}, "null_rate": 0.0},
        {"name": "book_status", "gen": "categorical",
         "params": {"categories": ["open", "closed", "cancel"],
                    "probs": [0.4, 0.45, 0.15]}, "null_rate": 0.0},
        {"name": "trader", "gen": "name", "params": {}, "null_rate": 0.04},
        {"name": "trade_day", "gen": "date", "params": {"span_days": 12}, "null_rate": 0.0},
    ], drifts=[
        Drift("rename", "book_amount", params={"new_name": "book_amt"}),     # D21
    ])

    sq_spec("trades_reorder", 2400, [
        {"name": "deal_no", "gen": "code", "params": {"prefix": "DL", "width": 8}, "null_rate": 0.0},
        {"name": "deal_amount", "gen": "float", "params": {"mu": 640, "sigma": 100}, "null_rate": 0.0},
        {"name": "deal_quantity", "gen": "int", "params": {"lo": 1, "hi": 25}, "null_rate": 0.0},
        {"name": "deal_status", "gen": "categorical",
         "params": {"categories": ["open", "closed", "cancel"],
                    "probs": [0.4, 0.45, 0.15]}, "null_rate": 0.0},
        {"name": "dealer", "gen": "name", "params": {}, "null_rate": 0.04},
        {"name": "deal_day", "gen": "date", "params": {"span_days": 12}, "null_rate": 0.0},
    ], drifts=[
        Drift("reorder", "deal_status", params={"to_ordinal": 0}),           # D22
    ])

    sq_spec("fx_quotes", 2600, [
        {"name": "quote_id", "gen": "int", "params": {"lo": 1, "hi": 10**7}, "null_rate": 0.0},
        {"name": "currency_pair", "gen": "categorical",
         "params": {"categories": ["USDCNY", "EURCNY", "JPYCNY", "GBPCNY"],
                    "probs": [0.5, 0.25, 0.15, 0.1]}, "null_rate": 0.0},
        {"name": "bid", "gen": "float", "params": {"mu": 7.1, "sigma": 0.08}, "null_rate": 0.0},
        {"name": "ask", "gen": "float", "params": {"mu": 7.12, "sigma": 0.08}, "null_rate": 0.0},
        {"name": "quote_hour", "gen": "int", "params": {"lo": 0, "hi": 23}, "null_rate": 0.0},
    ])

    sq_spec("insurance", 1800, [
        {"name": "policy_no", "gen": "code", "params": {"prefix": "INS", "width": 8}, "null_rate": 0.0},
        {"name": "holder_id", "gen": "int", "params": {"lo": 1, "hi": 90000}, "null_rate": 0.02},
        {"name": "premium", "gen": "float", "params": {"mu": 4200, "sigma": 900}, "null_rate": 0.0},
        {"name": "coverage", "gen": "float", "params": {"mu": 500000, "sigma": 80000}, "null_rate": 0.01},
        {"name": "product_line", "gen": "categorical",
         "params": {"categories": ["life", "health", "auto", "property"],
                    "probs": [0.3, 0.25, 0.3, 0.15]}, "null_rate": 0.0},
        {"name": "renewed", "gen": "bool", "params": {"true_prob": 0.7}, "null_rate": 0.0},
    ])

    sq_spec("settlement", 1000, [
        {"name": "settle_id", "gen": "code", "params": {"prefix": "ST", "width": 7}, "null_rate": 0.0},
        {"name": "merchant", "gen": "categorical",
         "params": {"categories": ["m1", "m2", "m3", "m4"],
                    "probs": [0.3, 0.3, 0.25, 0.15]}, "null_rate": 0.0},
        {"name": "settle_amount", "gen": "float", "params": {"mu": 2600, "sigma": 500}, "null_rate": 0.0},
        {"name": "fee_amount", "gen": "float", "params": {"mu": 15, "sigma": 4}, "null_rate": 0.0},
        {"name": "settle_days", "gen": "int", "params": {"lo": 1, "hi": 7}, "null_rate": 0.0},
    ])

    sq_spec("audit_log", 1200, [
        {"name": "log_id", "gen": "int", "params": {"lo": 1, "hi": 10**7}, "null_rate": 0.0},
        {"name": "operator", "gen": "name", "params": {}, "null_rate": 0.0},
        {"name": "action", "gen": "categorical",
         "params": {"categories": ["login", "query", "update", "export"],
                    "probs": [0.35, 0.35, 0.2, 0.1]}, "null_rate": 0.0},
        {"name": "response_ms", "gen": "int", "params": {"lo": 10, "hi": 2000}, "null_rate": 0.01},
        {"name": "success", "gen": "bool", "params": {"true_prob": 0.97}, "null_rate": 0.0},
    ])

    assert len(specs) == 30, f"表数量应为 30，实际 {len(specs)}"
    return specs


# ---------------- 单日单表生成 ----------------

def render_day(spec: TableSpec, day_idx: int, rng: np.random.Generator) -> pd.DataFrame:
    the_date = START + timedelta(days=day_idx)
    weekday = the_date.weekday()
    factor = spec.weekend_factor if weekday >= 5 else 1.0
    # 常规 ±4% 流量噪声
    factor *= float(1.0 + rng.normal(0.0, 0.04))
    if spec.volume_drift and day_idx >= DRIFT_BEGIN:
        factor *= spec.volume_drift["factor"]
    n = max(int(spec.base_rows * factor), 50)

    data: dict[str, np.ndarray] = {}
    for col in spec.columns:
        # 统一 object 数组，使数值列也能承载空字符串（由读取层按文本推断类型）
        vals = np.asarray(GENERATORS[col["gen"]](rng, n, col.get("params", {})),
                          dtype=object)
        # 常规空值
        nr = col.get("null_rate", 0.0)
        if nr > 0:
            k = int(round(n * nr))
            if k:
                idx = rng.choice(n, k, replace=False)
                vals[idx] = ""
        data[col["name"]] = vals

    # 漂移注入
    for d in spec.drifts:
        if day_idx < d.start or day_idx > d.end:
            continue
        if d.kind == "rename":
            new_name = d.params["new_name"]
            data[new_name] = data.pop(d.column)
        elif d.kind == "reorder":
            pass  # 在列排序阶段处理
        else:
            col_spec = next(c for c in spec.columns if c["name"] == d.column)
            data[d.column] = apply_drift(rng, data[d.column], col_spec, d, day_idx)

    df = pd.DataFrame(data)
    # 全部以字符串落盘（含空串空值），类型推断由画像侧统一完成，
    # 同时规避 Parquet 对 object 混合类型的限制
    df = df.map(lambda v: "" if v is None else str(v))
    # 列顺序：reorder 漂移把目标列移到指定位置
    for d in spec.drifts:
        if d.kind == "reorder" and day_idx >= d.start and d.column in df.columns:
            cols = list(df.columns)
            cols.remove(d.column)
            to = min(d.params["to_ordinal"], len(cols))
            cols.insert(to, d.column)
            df = df[cols]
    return df


def main() -> None:
    specs = build_specs()
    (DATA / "sources").mkdir(parents=True, exist_ok=True)
    ground_truth: list[dict] = []

    for spec in specs:
        seed = int(hashlib.md5(spec.name.encode("utf-8")).hexdigest()[:12], 16)
        rng = np.random.default_rng(seed)
        out_dir = DATA / "sources" / spec.kind / spec.name
        out_dir.mkdir(parents=True, exist_ok=True)
        sqlite_path = out_dir / f"{spec.name}.db"
        if spec.kind == "sqlite" and sqlite_path.exists():
            sqlite_path.unlink()

        for day_idx in range(N_DAYS):
            the_date = START + timedelta(days=day_idx)
            ds = the_date.isoformat()
            df = render_day(spec, day_idx, rng)

            if spec.kind == "csv":
                df.to_csv(out_dir / f"{ds}.csv", index=False, encoding="utf-8")
            elif spec.kind == "parquet":
                df.to_parquet(out_dir / f"{ds}.parquet", index=False)
            else:
                conn = sqlite3.connect(sqlite_path)
                try:
                    df.to_sql(f"t_{ds.replace('-', '_')}", conn, index=False)
                finally:
                    conn.close()

        # 真值登记
        for d in spec.drifts:
            gt_metric = {
                "null_spike": "null_rate", "mean_shift": "numeric_distribution",
                "scale_change": "numeric_distribution",
                "outlier_injection": "range_shift",
                "category_proportion": "category_distribution",
                "new_category": "category_distribution",
                "bool_flip": "bool_ratio", "pattern_change": "pattern_drift",
                "gradual_shift": "numeric_distribution",
                "cardinality_collapse": "cardinality",
                "range_expand": "range_shift", "unit_change": "numeric_distribution",
                "rename": "column_renamed", "reorder": "column_reordered",
            }[d.kind]
            ground_truth.append({
                "table": spec.name, "kind": d.kind, "column": d.column,
                "expected_metric": gt_metric,
                "start_date": (START + timedelta(days=d.start)).isoformat(),
                "end_date": (START + timedelta(days=d.end)).isoformat(),
                "description": _describe(spec, d),
            })
        if spec.volume_drift:
            ground_truth.append({
                "table": spec.name, "kind": "volume_drop", "column": None,
                "expected_metric": "row_volume",
                "start_date": (START + timedelta(days=DRIFT_BEGIN)).isoformat(),
                "end_date": (START + timedelta(days=N_DAYS - 1)).isoformat(),
                "description": spec.volume_drift["desc"],
            })

    with open(DATA / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
    print(f"已生成 {len(specs)} 张表 x {N_DAYS} 天数据，"
          f"真值漂移 {len(ground_truth)} 个，输出目录: {DATA / 'sources'}")


def _describe(spec: TableSpec, d: Drift) -> str:
    desc = {
        "null_spike": f"{d.column} 空值率突增",
        "mean_shift": f"{d.column} 均值平移",
        "scale_change": f"{d.column} 尺度放大/缩小",
        "outlier_injection": f"{d.column} 注入离群值",
        "category_proportion": f"{d.column} 类别占比重排",
        "new_category": f"{d.column} 出现新类别",
        "bool_flip": f"{d.column} 布尔占比翻转",
        "pattern_change": f"{d.column} 文本格式变化",
        "gradual_shift": f"{d.column} 渐进式均值漂移",
        "cardinality_collapse": f"{d.column} 基数塌缩",
        "range_expand": f"{d.column} 取值范围扩大",
        "unit_change": f"{d.column} 单位换算导致尺度变化",
        "rename": f"{d.column} 列改名",
        "reorder": f"{d.column} 列位置挪动",
    }
    return desc.get(d.kind, d.kind)


if __name__ == "__main__":
    main()
