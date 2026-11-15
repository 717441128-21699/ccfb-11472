"""列类型推断与文本格式自动归纳。

所有数据源（CSV / Parquet / SQLite）在读取层统一转成字符串序列后交给本模块，
保证不同来源的推断口径完全一致。
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .config import BOOL_TOKENS, DATETIME_FORMATS, DATE_FORMATS, NULL_TOKENS, PATTERN_CAP

# 支持的列类型
T_BOOL = "boolean"
T_INT = "integer"
T_FLOAT = "float"
T_DATETIME = "datetime"
T_TEXT = "text"

_INFER_SAMPLE = 20000          # 推断时最多扫描的非空值数量
_STRICT_RATIO = 0.95           # 单类型占比门槛

_NUMERIC_RE = re.compile(r"^[-+]?[\d,]*\.?\d+([eE][-+]?\d+)?$")
_INT_RE = re.compile(r"^[-+]?\d+$")
_PERCENT_RE = re.compile(r"^[-+]?\d+(\.\d+)?%$")

# 语义标签：顺序即优先级
_SEMANTIC_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("email",       re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")),
    ("url",         re.compile(r"^https?://[\w.-]+(?:/[^\s]*)?$")),
    ("uuid",        re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                               r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")),
    ("phone_cn",    re.compile(r"^1[3-9]\d{9}$")),
    ("id_card_cn",  re.compile(r"^\d{17}[\dXx]$|^\d{15}$")),
    ("postal_code", re.compile(r"^\d{6}$")),
    ("code_upper",  re.compile(r"^[A-Z][A-Z0-9_-]{1,15}$")),
    ("amount_cn",   re.compile(r"^[-+]?\d+(\.\d+)?元$")),
    ("percent",     _PERCENT_RE),
    ("name_cn",     re.compile(r"^[一-龥]{2,4}$")),
    ("word_en",     re.compile(r"^[A-Za-z]+$")),
    ("digits",      re.compile(r"^\d+$")),
]


def is_null_token(v: str) -> bool:
    return v.strip().lower() in NULL_TOKENS


def parse_numeric(token: str) -> float | None:
    """把 '1,234.5'、'12.5%'、'+3' 等解析成 float；失败返回 None。"""
    s = token.strip().replace(",", "")
    if not s:
        return None
    factor = 1.0
    if s.endswith("%"):
        s = s[:-1]
        factor = 0.01
    if _NUMERIC_RE.match(s):
        try:
            return float(s) * factor
        except ValueError:
            return None
    return None


def try_parse_datetime(token: str) -> datetime | None:
    s = token.strip()
    if len(s) < 6:
        return None
    for fmt in (*DATETIME_FORMATS, *DATE_FORMATS):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _is_bool(token: str) -> bool:
    return token.strip().lower() in BOOL_TOKENS


def infer_column(values: list[str]) -> str:
    """根据非空字符串样本推断列类型。"""
    sample = values[:_INFER_SAMPLE]
    n = len(sample)
    if n == 0:
        return T_TEXT

    n_bool = n_int = n_float = n_dt = 0
    n_long_digits = 0
    for raw in sample:
        token = raw.strip()
        dt = try_parse_datetime(token)
        if dt is not None:
            n_dt += 1
            continue
        if _INT_RE.match(token.replace(",", "").lstrip("+")):
            # 长数字串（手机号 11 位、证件号 15/18 位等）按文本编码处理，
            # 否则无法对其做格式归纳与格式漂移检测
            digits = token.replace(",", "").lstrip("+")
            if len(digits) >= 11:
                n_long_digits += 1
            else:
                n_int += 1
            # 布尔只在“非纯整数语义”时统计，避免 0/1 数值列被误判布尔
            if _is_bool(token) and not digits.isdigit():
                n_bool += 1
            continue
        if parse_numeric(token) is not None:
            n_float += 1
            continue
        if _is_bool(token):
            n_bool += 1

    if n_dt / n >= _STRICT_RATIO:
        return T_DATETIME
    if n_long_digits / n >= 0.5 and (n_long_digits + n_int) / n >= _STRICT_RATIO:
        return T_TEXT
    if (n_int + n_long_digits) / n >= _STRICT_RATIO:
        return T_INT
    if (n_int + n_long_digits + n_float) / n >= _STRICT_RATIO:
        return T_FLOAT
    if n_bool / n >= _STRICT_RATIO:
        return T_BOOL
    # 兜底：取值集合本身是布尔词表（true/false/yes/no/是/否 等）
    bool_vocab = sum(1 for raw in sample if _is_bool(raw.strip()))
    if bool_vocab / n >= _STRICT_RATIO:
        return T_BOOL
    return T_TEXT


def pattern_signature(token: str) -> str:
    """把具体文本抽象成结构签名：数字->9，字母->a，汉字->中，其余字符保留。

    例：'13812345678' -> '99999999999'，'ORD-2024-0001' -> 'aaa-9999-9999'。
    超长自由文本统一归入 TEXT_LONG，避免签名爆炸。
    """
    s = token.strip()
    if len(s) > 40:
        return "TEXT_LONG"
    out: list[str] = []
    for ch in s:
        code = ord(ch)
        if ch.isdigit():
            out.append("9")
        elif ch.isascii() and ch.isalpha():
            out.append("a")
        elif 0x4E00 <= code <= 0x9FFF:
            out.append("中")
        else:
            out.append(ch)
    return "".join(out)


def semantic_label(token: str) -> str | None:
    s = token.strip()
    for label, rx in _SEMANTIC_RULES:
        if rx.match(s):
            return label
    return None


def summarize_text_patterns(values: list[str], top: int = PATTERN_CAP) -> dict[str, Any]:
    """归纳文本列的格式模式：结构签名分布 + 语义标签分布 + 示例。"""
    sig_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    total = 0
    for raw in values:
        token = raw.strip()
        if not token:
            continue
        total += 1
        sig = pattern_signature(token)
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
        examples.setdefault(sig, token[:60])
        lbl = semantic_label(token)
        if lbl:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    sig_sorted = sorted(sig_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    label_sorted = sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    dominant_label = label_sorted[0][0] if label_sorted and \
        label_sorted[0][1] / max(total, 1) >= 0.6 else None
    return {
        "total": total,
        "signatures": [
            {"signature": s, "count": c, "ratio": round(c / max(total, 1), 6),
             "example": examples.get(s, "")}
            for s, c in sig_sorted
        ],
        "labels": [{"label": l, "count": c,
                    "ratio": round(c / max(total, 1), 6)} for l, c in label_sorted],
        "dominant_label": dominant_label,
    }
