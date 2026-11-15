"""Web 管理页面冒烟测试：验证五个页面与关键 API 正常返回。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataprobe import storage
from dataprobe.web import create_app


def main() -> None:
    storage.init_db(storage.connect())
    app = create_app()
    client = app.test_client()
    checks = [
        ("/", "首页/表画像"),
        ("/sources", "来源登记"),
        ("/events", "漂移事件"),
        ("/thresholds", "阈值配置"),
    ]
    ok = True
    for url, name in checks:
        rc = client.get(url).status_code
        flag = "OK " if rc == 200 else "FAIL"
        ok &= rc == 200
        print(f"{flag} GET {url:14s} {name} -> {rc}")

    # 表详情与单列详情（第一张表 + 类型丰富的表，覆盖数值/文本/布尔/日期/格式模板）
    sources = storage.list_sources(storage.connect())
    rich_names = ["payments", "customers", "orders"]
    target_sources = sources[:1] + [s for s in sources if s["name"] in rich_names]
    seen = set()
    for src in target_sources:
        tid = src["table_id"]
        if tid in seen:
            continue
        seen.add(tid)
        rc = client.get(f"/tables/{tid}").status_code
        ok &= rc == 200
        print(f"{'OK ' if rc==200 else 'FAIL'} GET /tables/<id>      表详情 {src['name']} -> {rc}")

        latest = storage.get_latest_snapshot_date(storage.connect(), tid)
        if latest:
            cols = storage.get_columns_on_date(storage.connect(), tid, latest)
            # 每种推断类型测一列，覆盖全部模板分支
            seen_types = set()
            for col in cols:
                t = col["inferred_type"]
                if t in seen_types:
                    continue
                seen_types.add(t)
                cname = col["column_name"]
                rc = client.get(
                    f"/tables/{tid}/columns/{cname}?date={latest}").status_code
                ok &= rc == 200
                print(f"{'OK ' if rc==200 else 'FAIL'} GET 单列详情 {src['name']}.{cname}({t}) -> {rc}")
            if cols:
                cname = cols[0]["column_name"]
                rc = client.get(
                    f"/api/tables/{tid}/columns/{cname}/series").status_code
                ok &= rc == 200
                print(f"{'OK ' if rc==200 else 'FAIL'} GET 列趋势 API {src['name']} -> {rc}")
        rc = client.get(f"/api/tables/{tid}/series").status_code
        ok &= rc == 200
        print(f"{'OK ' if rc==200 else 'FAIL'} GET 表趋势 API {src['name']} -> {rc}")

    print("\nALL PASS" if ok else "\nHAS FAILURE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
