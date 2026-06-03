"""Data acquisition and IO.

Downloads the train and test CSVs from Google Drive, coerces dtypes to a
consistent schema, and caches parquet copies for fast reloads.

Expected row counts:
  train.csv             306,226  (all prices known)
  test_3days.csv         25,900  (3 shared days, 300 anchors, 25,600 blanks)
  test_full.csv (at interview) 76,255  (16 days, 1,600 anchors, 74,655 blanks)

Usage:
    python -m src.data_io download      # fetch CSVs from Drive
    python -m src.data_io prepare       # coerce dtypes + write parquet
    python -m src.data_io info          # print shapes / sanity checks
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C


# --------------------------------------------------------------------------- #
# Friendly "missing data" helpers
# --------------------------------------------------------------------------- #
def _missing_error(path: Path, desc: str) -> None:
    msg = (
        f"\n{'='*60}\n"
        f"  ERROR: {desc} not found.\n"
        f"  Expected: {path}\n\n"
        f"  Did you run 'make download'?  The data lives on Google Drive.\n"
        f"  Run:  make download\n"
        f"{'='*60}\n"
    )
    print(msg, flush=True)
    raise FileNotFoundError(str(path))


def _assert_exists(path: Path, desc: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        _missing_error(path, desc)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download() -> None:
    """Fetch raw CSVs from Google Drive using gdown."""
    import gdown

    targets = [
        (C.GDRIVE_TRAIN_ID, C.TRAIN_CSV, "train.csv"),
        (C.GDRIVE_TEST_3DAYS_ID, C.TEST_CSV, "test_3days.csv"),
    ]
    for file_id, dest, desc in targets:
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {desc} already present ({dest.stat().st_size:,} bytes)")
            continue
        print(f"[download] {desc} (id={file_id}) -> {dest}")
        try:
            gdown.download(id=file_id, output=str(dest), quiet=False)
        except TypeError:
            gdown.download(f"https://drive.google.com/uc?id={file_id}",
                           str(dest), quiet=False)


# --------------------------------------------------------------------------- #
# Dtype coercion
# --------------------------------------------------------------------------- #
def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce raw string columns into a consistent, memory-friendly schema."""
    df = df.copy()

    if C.TIME_COL in df.columns:
        df[C.TIME_COL] = pd.to_datetime(df[C.TIME_COL], errors="coerce", utc=True)

    bool_map = {
        "t": True, "f": False, "true": True, "false": False,
        "1": True, "0": False, "yes": True, "no": False,
    }
    for col in C.BOOL_COLS:
        if col in df.columns:
            df[col] = (
                df[col].astype("string").str.strip().str.lower()
                .map(bool_map).astype("boolean")
            )

    numeric = [C.TARGET] + C.NUMERIC_COLS
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in C.ID_COLS + ["brand"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


# --------------------------------------------------------------------------- #
# Load helpers (fail loudly when data is missing)
# --------------------------------------------------------------------------- #
def load_train(prefer_parquet: bool = True) -> pd.DataFrame:
    if prefer_parquet and C.TRAIN_PARQUET.exists():
        return pd.read_parquet(C.TRAIN_PARQUET)
    _assert_exists(C.TRAIN_CSV, "train.csv")
    return _coerce(pd.read_csv(C.TRAIN_CSV, low_memory=False))


def load_test(prefer_parquet: bool = True) -> pd.DataFrame:
    if prefer_parquet and C.TEST_PARQUET.exists():
        return pd.read_parquet(C.TEST_PARQUET)
    _assert_exists(C.TEST_CSV, "test CSV file")
    return _coerce(pd.read_csv(C.TEST_CSV, low_memory=False))


def prepare() -> None:
    """Read raw CSVs, coerce, and cache parquet."""
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)

    _assert_exists(C.TRAIN_CSV, "train.csv")
    print("[prepare] reading train CSV ...")
    train = _coerce(pd.read_csv(C.TRAIN_CSV, low_memory=False))
    train.to_parquet(C.TRAIN_PARQUET, index=False)
    print(f"[prepare] wrote {C.TRAIN_PARQUET} shape={train.shape}")

    _assert_exists(C.TEST_CSV, "test CSV file")
    print("[prepare] reading test CSV ...")
    test = _coerce(pd.read_csv(C.TEST_CSV, low_memory=False))
    test.to_parquet(C.TEST_PARQUET, index=False)
    print(f"[prepare] wrote {C.TEST_PARQUET} shape={test.shape}")


# --------------------------------------------------------------------------- #
# Sanity report
# --------------------------------------------------------------------------- #
def info() -> None:
    _assert_exists(C.TRAIN_CSV, "train.csv")
    _assert_exists(C.TEST_CSV, "test CSV file")

    train = load_train()
    test = load_test()
    n_test = len(test)

    print("\n=== TRAIN ===")
    print(f"  shape: {train.shape}  (expected rows: {C.EXPECTED_TRAIN_ROWS:,})")
    if len(train) != C.EXPECTED_TRAIN_ROWS:
        print(f"  WARNING: got {len(train):,} rows, expected {C.EXPECTED_TRAIN_ROWS:,}")
    print(f"  price non-null: {int(train[C.TARGET].notna().sum()):,}")

    print("\n=== TEST ===")
    if n_test == C.EXPECTED_TEST_3DAY_ROWS:
        kind = "3 shared days"
    elif n_test == C.EXPECTED_TEST_FULL_ROWS:
        kind = "full 16-day file"
    else:
        kind = f"UNEXPECTED SIZE (expected {C.EXPECTED_TEST_3DAY_ROWS:,} or {C.EXPECTED_TEST_FULL_ROWS:,})"
    print(f"  shape: {test.shape}  ({kind})")

    if C.TIME_COL in test.columns:
        test_days = test[C.TIME_COL].dt.date
        days = sorted(test_days.dropna().unique().tolist())
        print(f"  distinct days: {len(days)}  ({days[0]} .. {days[-1]})")
        anchors = test[C.TARGET].notna()
        print(f"  anchor rows (price filled): {int(anchors.sum()):,}")
        print(f"  blank-price rows: {int((~anchors).sum()):,}")
        if int(anchors.sum()) != len(days) * C.ANCHORS_PER_DAY:
            print(f"  WARNING: expected {len(days)*C.ANCHORS_PER_DAY:,} anchors "
                  f"(100/day), got {int(anchors.sum()):,}")

        print("\n  anchors per day:")
        per_day = test.assign(day=test_days).groupby("day")[C.TARGET].apply(
            lambda s: int(s.notna().sum()))
        print(per_day.to_string())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"
    {"download": download, "prepare": prepare, "info": info}[cmd]()
