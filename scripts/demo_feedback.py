"""误报反馈演示：把指定事件标为误报，再重跑后续日期，验证同类事件自动降级。

用法：
  py scripts/demo_feedback.py                # 自动找一条 open 事件演示
  py scripts/demo_feedback.py <event_id>     # 指定事件
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataprobe import storage
from dataprobe.config import ST_FALSE_POSITIVE
from dataprobe.drift import DOWNGRADE_MAP


def main() -> None:
    conn = storage.connect()
    storage.init_db(conn)
    event_id = int(sys.argv[1]) if len(sys.argv) > 1 else None

    if event_id is None:
        row = conn.execute(
            "SELECT * FROM events WHERE status='open' ORDER BY event_id LIMIT 1"
        ).fetchone()
        if not row:
            print("没有 open 事件可演示")
            return
        event_id = row["event_id"]

    row = conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
    if not row:
        print(f"事件 {event_id} 不存在")
        return
    row = dict(row)
    print("目标事件：", row["event_id"], row["table_id"], row["column_name"],
          row["metric"], row["severity"], row["signature"])

    storage.add_feedback(conn, signature=row["signature"], table_id=row["table_id"],
                         metric=row["metric"], note="演示：确认误报，后续同类自动降级")
    storage.update_event_status(conn, event_id, ST_FALSE_POSITIVE)

    # 模拟同签名新事件经过反馈通道
    feedback = storage.get_feedback_set(conn)
    if row["signature"] in feedback:
        new_sev = DOWNGRADE_MAP[row["severity"]]
        print(f"反馈已生效：同签名新事件级别将由 {row['severity']} "
              f"自动降级为 {new_sev}，状态标记 auto_downgraded")
    print("当前反馈规则：")
    for fb in storage.list_feedback(conn):
        print("  -", fb["signature"], "命中", fb["hit_count"], "次")


if __name__ == "__main__":
    main()
