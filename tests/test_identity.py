"""列身份识别单元测试：改名、移位、周末类型抖动下 stable_col_id 保持稳定。"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataprobe import storage, identity
from dataprobe.profiler import profile_table


def run(conn, tid, ds, columns, data):
    _, crows = profile_table(ds, tid, columns, data)
    identity.assign_identities(conn, tid, ds, crows)
    return {r["column_name"]: r["stable_col_id"] for r in crows}


def main() -> None:
    tmp = Path(tempfile.mkdtemp()) / "id.db"
    conn = storage.connect(str(tmp))
    storage.init_db(conn)
    tid = "idt__main"
    ok = True

    # 基准快照
    base_cols = ["trade_no", "amount", "qty", "status", "trader", "day"]
    base_data = {
        "trade_no": [f"TR-{i:08d}" for i in range(1000)],
        "amount": [f"{200 + (i % 50) * 1.7:.2f}" for i in range(1000)],
        "qty": [str((i % 20) + 1) for i in range(1000)],
        "status": (["open"] * 400 + ["closed"] * 450 + ["cancel"] * 150),
        "trader": [f"业务员{i % 40}" for i in range(1000)],
        "day": ["2026-09-01"] * 1000,
    }
    m1 = run(conn, tid, "2026-09-01", base_cols, base_data)

    # 场景 1：列移位（status 移到首位），身份不变
    reorder_cols = ["status", "trade_no", "amount", "qty", "trader", "day"]
    m2 = run(conn, tid, "2026-09-02", reorder_cols, base_data)
    same = all(m1[c] == m2[c] for c in base_cols)
    print("[1] 移位后身份保持:", "OK" if same else "FAIL",
          "" if same else (m1, m2))
    ok &= same

    # 场景 2：amount 改名为 amt，身份应延续
    rename_data = dict(base_data)
    rename_data["amt"] = rename_data.pop("amount")
    rename_cols = ["status", "trade_no", "amt", "qty", "trader", "day"]
    m3 = run(conn, tid, "2026-09-03", rename_cols, rename_data)
    kept = m3["amt"] == m1["amount"]
    print("[2] 改名后身份延续:", "OK" if kept else "FAIL",
          f"({m1['amount']} -> {m3['amt']})")
    ok &= kept

    # 场景 3：周末小样本导致 qty 推断在 integer/text 间抖动，身份仍应稳定
    small = {k: v[:120] for k, v in base_data.items()}
    # 制造小整数 0/1 占多数的周末样本
    small["qty"] = (["0"] * 60 + ["1"] * 55 + ["2"] * 5)
    m4 = run(conn, tid, "2026-09-05", base_cols, small)
    stable = m4["qty"] == m1["qty"]
    print("[3] 类型抖动下身份稳定:", "OK" if stable else "FAIL",
          f"({m1['qty']} -> {m4['qty']})")
    ok &= stable

    # 场景 4：真新增列应拿到新身份，不与旧列混淆（独立表，避免场景间身份编号耦合）
    tid4 = "idt_add__main"
    m1b = run(conn, tid4, "2026-09-01", base_cols, base_data)
    add_data = dict(base_data)
    add_data["new_score"] = [f"{(i % 70) + 10.5:.2f}" for i in range(1000)]
    add_cols = base_cols + ["new_score"]
    m5 = run(conn, tid4, "2026-09-06", add_cols, add_data)
    existing_ids = set(m1b.values())
    new_id_ok = m5["new_score"] not in existing_ids
    old_kept = all(m5[c] == m1b[c] for c in base_cols)
    print("[4] 新增列拿新身份且旧列不变:",
          "OK" if new_id_ok and old_kept else "FAIL",
          f"(new_score={m5['new_score']}, 旧身份集合={sorted(existing_ids)})")
    ok &= new_id_ok and old_kept

    conn.close()
    print("\nALL PASS" if ok else "\nHAS FAILURE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
