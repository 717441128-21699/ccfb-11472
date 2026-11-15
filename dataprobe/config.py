"""全局配置与默认阈值。阈值可在管理页面覆盖，覆盖记录存 SQLite。"""
from __future__ import annotations

from pathlib import Path

# 目录约定：工作区下 data/ 存放模拟数据与元数据库
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = WORKSPACE_ROOT / "data"
META_DB_PATH = DATA_ROOT / "dataprobe_meta.db"

# 画像/漂移默认阈值（key -> 中文说明 + 默认值）
THRESHOLD_SCHEMA = {
    "warmup_days":              ("预热天数，期内只画像不告警", 7),
    "reference_window_days":    ("基线参考窗口天数", 14),
    "min_reference_days":       ("至少积累多少天基线才参与比较", 3),
    "min_sample_distribution":  ("分布类指标最小非空样本量，不足则跳过", 30),
    "psi_warning":              ("PSI 预警阈值", 0.20),
    "psi_critical":             ("PSI 严重阈值", 0.40),
    "ks_warning":               ("KS 统计量预警阈值", 0.30),
    "ks_critical":              ("KS 统计量严重阈值", 0.50),
    "null_abs_delta_warning":   ("空值率绝对变化预警门槛", 0.15),
    "null_robust_z_warning":    ("空值率稳健 z 分数预警门槛", 5.0),
    "cardinality_rel_warning":  ("基数相对变化预警门槛", 0.40),
    "cardinality_min_count":    ("基数告警最小绝对变化量，防小样本噪声", 20),
    "range_iqr_warning":        ("最值偏移超过多少倍 IQR 预警", 3.0),
    "pattern_psi_warning":      ("文本格式分布 PSI 预警阈值", 0.50),
    "volume_robust_z_warning":  ("行数稳健 z 分数预警门槛（同星期口径）", 4.0),
    "volume_min_abs_ratio":     ("行数偏离中位数的最小相对幅度", 0.35),
    "high_cardinality_ratio":   ("去重率高于该值视为高基数列，跳过类别 PSI", 0.50),
    "feedback_mode":            ("误报反馈生效方式：downgrade 降级 / suppress 抑制", "downgrade"),
}

DEFAULT_THRESHOLDS = {k: v[1] for k, v in THRESHOLD_SCHEMA.items()}

# 类型推断
NULL_TOKENS = frozenset({"", "null", "none", "nan", "n/a", "na", "nil", "#n/a"})
BOOL_TOKENS = frozenset({"true", "false", "0", "1", "yes", "no", "y", "n",
                         "t", "f", "是", "否", "真", "假"})

DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
    "%Y%m%d", "%d-%m-%Y", "%Y.%m.%d",
)
DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S.%f",
)

# 漂移事件级别
SEV_INFO = "info"
SEV_WARNING = "warning"
SEV_CRITICAL = "critical"

# 事件状态
ST_OPEN = "open"
ST_FALSE_POSITIVE = "false_positive"
ST_RESOLVED = "resolved"
ST_AUTO_DOWNGRADED = "auto_downgraded"

EPS = 1e-6
SAMPLE_CAP = 300          # 每列每日保留的蓄水池样本量（供 KS / 基线分桶）
TOP_VALUE_CAP = 50        # 类别值最多保留个数
PATTERN_CAP = 20
