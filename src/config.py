"""Central configuration: paths, seeds, schema, and column groupings.

All path handling is relative to the repository root so the code runs
identically on the local mac and on the GPU host (``maldo``).
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test_3days.csv"
TRAIN_PARQUET = DATA_DIR / "train.parquet"
TEST_PARQUET = DATA_DIR / "test_3days.parquet"
COMPLETED_CSV = DATA_DIR / "test_completed.csv"

# Google Drive file IDs (from the take-home brief)
GDRIVE_TRAIN_ID = "1ZOYuyrcBJF7fvnG6kBva1m8og5urXHjU"
GDRIVE_TEST_3DAYS_ID = "1Ni2aBrOaV1YEWspZVmBw__HV1a7M37cd"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# Expected dataset shapes (sanity checks; see brief)
EXPECTED_TRAIN_ROWS = 306_226
EXPECTED_TEST_3DAY_ROWS = 25_900   # 3 shared days
EXPECTED_TEST_FULL_ROWS = 76_255   # all 16 days (at interview)
ANCHORS_PER_DAY = 100

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
TARGET = "price"
TIME_COL = "capturedAt"

# High-cardinality categorical identifiers
ID_COLS = ["shopId", "itemId", "modelId", "cat_id", "promotionId"]
CATEGORICAL_COLS = ID_COLS + ["brand"]

# Boolean flags stored as t/f in the raw data
BOOL_COLS = [
    "is_free_shipping",
    "is_pre_order",
    "is_official_shop",
    "is_verified",
    "is_preferred_plus_seller",
]

# Numeric columns present in the raw schema
NUMERIC_COLS = [
    "priceBeforeDiscount",
    "stock",
    "normal_stock",
    "raw_discount",
    "show_discount",
    "item_price_min",
    "item_price_max",
    "review_rating",
    "total_rating_count",
    "cmt_count",
    "shop_rating",
    "shop_response_rate",
    "shop_follower_count",
]

# Full expected column set (used only for validation/reporting)
ALL_COLUMNS = (
    [TIME_COL]
    + ID_COLS
    + [TARGET]
    + NUMERIC_COLS
    + ["brand"]
    + BOOL_COLS
)
