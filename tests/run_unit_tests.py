"""一键运行全部单元测试：py tests/run_unit_tests.py"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = ["test_infer.py", "test_drift.py", "test_identity.py"]

failed = 0
for t in TESTS:
    print(f"\n{'=' * 60}\n运行 {t}\n{'=' * 60}")
    rc = subprocess.run([sys.executable, str(HERE / t)]).returncode
    if rc != 0:
        failed += 1

print(f"\n{'=' * 60}")
print(f"单元测试汇总：{len(TESTS) - failed}/{len(TESTS)} 通过")
sys.exit(1 if failed else 0)
