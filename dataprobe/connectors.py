"""数据源连接器：逗号分隔文本、Parquet、SQLite。

读取后统一规整成 (列名顺序, dict[列名 -> list[str]])，类型推断全部交给 infer 模块，
保证三种来源的画像口径一致。数值不使用 pandas 自动类型，而是保留原始文本。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .infer import is_null_token


def _normalize_dataframe(df: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    columns = [str(c) for c in df.columns]
    data: dict[str, list[str]] = {}
    for col in columns:
        series = df[col]
        out: list[str] = []
        is_date_like = pd.api.types.is_datetime64_any_dtype(series)
        for v in series.tolist():
            if v is None or (isinstance(v, float) and pd.isna(v)) or \
                    (not isinstance(v, (str, bytes)) and pd.isna(v)):
                out.append("")
            elif is_date_like:
                ts = pd.Timestamp(v)
                out.append(ts.strftime("%Y-%m-%d %H:%M:%S")
                           if (ts.hour or ts.minute or ts.second)
                           else ts.strftime("%Y-%m-%d"))
            elif isinstance(v, bool):
                out.append("true" if v else "false")
            elif isinstance(v, float):
                out.append(("%.10g" % v))
            else:
                out.append(str(v).strip())
        data[col] = out
    return columns, data


def read_csv_source(path: str, encoding: str = "utf-8") -> tuple[list[str], dict[str, list[str]]]:
    # dtype=str 保留原始文本，na 过滤交给统一的 NULL_TOKENS 口径
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding,
                     low_memory=False)
    return _normalize_dataframe(df)


def read_parquet_source(path: str) -> tuple[list[str], dict[str, list[str]]]:
    df = pd.read_parquet(path)
    return _normalize_dataframe(df)


def read_sqlite_source(path: str, table_name: str) -> tuple[list[str], dict[str, list[str]]]:
    conn = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    finally:
        conn.close()
    return _normalize_dataframe(df)


def load_source(kind: str, path: str, table_name: str | None = None
                ) -> tuple[list[str], dict[str, list[str]]]:
    kind = kind.lower()
    if kind in ("csv", "text"):
        return read_csv_source(path)
    if kind == "parquet":
        return read_parquet_source(path)
    if kind == "sqlite":
        if not table_name:
            raise ValueError("SQLite 来源必须指定 table_name")
        return read_sqlite_source(path, table_name)
    raise ValueError(f"不支持的来源类型: {kind}")


def count_non_null(values: list[str]) -> int:
    return sum(1 for v in values if not is_null_token(v))
