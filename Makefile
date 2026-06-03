# MrScraper Price Intelligence — reproducible pipeline
# Run with: `uv sync --locked` then `make all`.

PY ?= uv run python

# Live inference overrides:
#   make infer TEST=data/test_full.csv OUT=data/test_completed.csv STRATEGY=category
TEST ?= data/test_3days.csv
OUT ?= data/test_completed.csv
STRATEGY ?= category
VAL ?= data/val.csv
TRUTH ?= data/val_truth.csv

.PHONY: help setup download prepare info eda eda2 backtest tier1 calib tier2 infer score results all clean

help:
	@echo "Targets:"
	@echo "  setup     - uv sync --locked"
	@echo "  download  - fetch train + 3-day test CSVs from Google Drive"
	@echo "  prepare   - coerce dtypes and cache parquet"
	@echo "  info      - print train/test sanity checks"
	@echo "  eda       - exploratory data analysis report"
	@echo "  backtest  - LOCF / median baselines on the outage backtest"
	@echo "  tier1     - Tier 1 global CatBoost (CPU default; --gpu for GPU)"
	@echo "  calib     - anchor calibration study + synthetic-shift stress test"
	@echo "  tier2     - Tier 2 hierarchical entity model backtest"
	@echo "  infer     - fill blank prices -> $(OUT)"
	@echo "  score     - compare VAL=$(VAL) against TRUTH=$(TRUTH)"
	@echo "  results   - aggregate all results + plots + README tables"
	@echo "  all       - download prepare eda backtest tier1 calib tier2 results infer"

setup:
	uv sync --locked

download:
	$(PY) -m src.data_io download

prepare:
	$(PY) -m src.data_io prepare
	$(PY) -m src.data_io info

info:
	$(PY) -m src.data_io info

eda:
	$(PY) -m src.eda
	$(PY) -m src.eda2

backtest:
	$(PY) -m src.validation

tier1:
	$(PY) -m src.model_global $(TIER1_FLAGS)

calib:
	$(PY) -m src.calibration

tier2:
	$(PY) -m src.model_entity

infer:
	$(PY) -m src.infer --test $(TEST) --out $(OUT) --strategy $(STRATEGY)

score:
	$(PY) -m src.score --pred $(VAL) --truth $(TRUTH)

results:
	$(PY) -m src.results_summary

all: download prepare eda backtest tier1 calib tier2 results infer

clean:
	rm -rf reports/*.png reports/*.json reports/*.md models/*.cbm models/*.pt
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
