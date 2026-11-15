"""登记模拟来源并跑完 60 天画像/漂移流水线。

用法：
  py scripts/run_pipeline.py                # 全量
  py scripts/run_pipeline.py --table orders # 单表
  py scripts/run_pipeline.py --reset        # 重建元数据库
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataprobe import storage
from dataprobe.config import DATA_ROOT, META_DB_PATH
from dataprobe.pipeline import run_table_day
from scripts.simulator import N_DAYS, START, build_specs


def register_sources(conn) -> list[dict]:
    sources = []
    for spec in build_specs():
        kind_dir = DATA_ROOT / "sources" / spec.kind / spec.name
        if spec.kind == "csv":
            path = str(kind_dir / f"{START.isoformat()}.csv")
            table_name = None
        elif spec.kind == "parquet":
            path = str(kind_dir / f"{START.isoformat()}.parquet")
            table_name = None
        else:
            path = str(kind_dir / f"{spec.name}.db")
            table_name = f"t_{START.strftime('%Y_%m_%d')}"
        rec = storage.upsert_source(
            conn, name=spec.name, kind=spec.kind, path=path,
            table_name=table_name, note=f"模拟表（{spec.kind}）")
        sources.append(rec)
    return sources


def _path_for_date(source: dict, ds: str) -> tuple[str, str | None]:
    base = Path(source["path"]).parent
    if source["kind"] == "csv":
        return str(base / f"{ds}.csv"), None
    if source["kind"] == "parquet":
        return str(base / f"{ds}.parquet"), None
    return source["path"], f"t_{ds.replace('-', '_')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", help="只跑指定表名")
    ap.add_argument("--reset", action="store_true", help="删除并重建元数据库")
    args = ap.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    conn = storage.connect()
    storage.init_db(conn, reset=args.reset)

    sources = register_sources(conn)
    if args.table:
        sources = [s for s in sources if s["name"] == args.table]
        if not sources:
            raise SystemExit(f"未找到表: {args.table}")

    dates = [(START + timedelta(days=i)).isoformat() for i in range(N_DAYS)]
    t0 = time.time()
    total_events = 0
    for ds in dates:
        day_events = 0
        for src in sources:
            path, table_name = _path_for_date(src, ds)
            day_src = {**src, "path": path, "table_name": table_name}
            summary = run_table_day(conn, day_src, ds)
            day_events += summary["events"]
        total_events += day_events
        print(f"[{ds}] 处理 {len(sources)} 张表，当日事件 {day_events}")
    print(f"完成：{len(sources)} 表 x {len(dates)} 天，"
          f"事件总数 {total_events}，用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
