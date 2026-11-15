import sys
sys.path.insert(0, ".")
from dataprobe.infer import infer_column, pattern_signature, summarize_text_patterns

cases = [
    ("phone", ["13812345678", "15999998888", "18611112222"] * 30, "text"),
    ("idcard", ["11010119900307123X", "310101198506174567"] * 30, "text"),
    ("small int weekend", ["0", "1", "2", "1", "0"] * 40, "integer"),
    ("work years", [str(x) for x in [19, 4, 22, 0, 21, 10, 24, 7]] * 20, "integer"),
    ("bool text", ["true", "false", "true", "false"] * 25, "boolean"),
    ("float", ["12.5", "13.1", "100.2", "55.0"] * 25, "float"),
    ("amount", ["260.45", "310.2", "199.99"] * 30, "float"),
    ("date", ["2026-09-01", "2026-10-02"] * 30, "datetime"),
    ("category", ["web", "app", "mini", "offline"] * 25, "text"),
    ("code", ["ORD-123456", "ORD-654321", "SKU-000001"] * 20, "text"),
    ("zero one bool-ish", ["0", "1", "0", "1"] * 30, "integer"),
]
ok = True
for name, vals, expect in cases:
    got = infer_column(vals)
    flag = "OK " if got == expect else "FAIL"
    if got != expect:
        ok = False
    print(f"{flag} {name:22s} expect={expect:8s} got={got}")

print("\npattern signatures:")
for v in ["13812345678", "010-87654321", "ORD-2024-0001", '"user123@test.cn"', "张三", "X-ORD-000123"]:
    print(f"  {v:24s} -> {pattern_signature(v)}")

pat = summarize_text_patterns(
    ["13812345678"] * 60 + ["010-87654321"] * 40)
print("\nmixed phone patterns:",
      [(s["signature"], s["ratio"]) for s in pat["signatures"]],
      "label:", pat["dominant_label"])
print("\nALL PASS" if ok else "\nHAS FAILURE")
