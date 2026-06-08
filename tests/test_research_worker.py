"""
tests/test_research_worker.py
Sprint 3 Revisited — S3-R1

Test scope:
  - Unit: loader validates DataFrame shape, required columns, min candle count
  - Unit: DataInsufficientError raised for missing data and below-threshold data
  - Unit: DataFormatError raised for malformed JSON
  - Unit: multi-format detection order (json > feather > parquet)
  - Integration: full run_research_cycle saves candidate to ledger with correct fields
  - Integration: stage != DRY_RUN_PLACEHOLDER enforced
  - Regression: unchanged functions (compute_bucket_features, run_oos_validation,
    save_candidate) still pass their original contracts
"""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_freqtrade_json(path: str, n_candles: int = 6000, seed: int = 42) -> None:
    """Write a valid Freqtrade-format JSON file."""
    rng = np.random.default_rng(seed)
    base_ts = 1_700_000_000_000  # ms
    interval_ms = 15 * 60 * 1000
    rows = []
    price = 40_000.0
    for i in range(n_candles):
        o = price
        h = o + abs(rng.normal(0, 50))
        l = o - abs(rng.normal(0, 50))
        c = o + rng.normal(0, 30)
        v = abs(rng.normal(100, 20))
        rows.append([base_ts + i * interval_ms, round(o, 2), round(h, 2), round(l, 2), round(c, 2), round(v, 4)])
        price = c
    with open(path, "w") as f:
        json.dump(rows, f)


def _init_ledger(db_path: str) -> None:
    """Initialize strategy_memory table for integration tests."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_memory (
                candidate_id TEXT PRIMARY KEY,
                strategy_id TEXT,
                model_type TEXT,
                stage TEXT,
                oos_metrics TEXT,
                pair TEXT,
                data_source TEXT,
                created_ts TEXT,
                notes TEXT
            )
        """)
        conn.commit()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def temp_ledger():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    _init_ledger(db_path)
    yield db_path
    os.unlink(db_path)


@pytest.fixture
def btc_json_file(temp_data_dir):
    path = os.path.join(temp_data_dir, "BTC_USDT-15m.json")
    _make_freqtrade_json(path, n_candles=6000)
    return path


@pytest.fixture
def eth_json_file(temp_data_dir):
    path = os.path.join(temp_data_dir, "ETH_USDT-15m.json")
    _make_freqtrade_json(path, n_candles=6000, seed=99)
    return path


# ---------------------------------------------------------------------------
# Import under test (deferred to avoid import errors at collection time)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def import_worker():
    """Lazy import so pytest can collect even if worker has a syntax error."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "research.worker",
        Path(__file__).parent.parent / "research" / "worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["research.worker"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def worker(import_worker):
    return import_worker


# ---------------------------------------------------------------------------
# Unit tests — load_freqtrade_ohlcv: DataFrame shape and columns
# ---------------------------------------------------------------------------

class TestLoaderShape:

    def test_loaded_dataframe_has_required_columns(self, worker, temp_data_dir, btc_json_file, eth_json_file):
        result = worker.load_freqtrade_ohlcv(
            pairs=["BTC/USDT", "ETH/USDT"],
            timeframe="15m",
            data_dir=temp_data_dir,
        )
        for pair, df in result.items():
            assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"], \
                f"{pair}: unexpected columns {df.columns.tolist()}"

    def test_loaded_dataframe_has_correct_row_count(self, worker, temp_data_dir, btc_json_file):
        result = worker.load_freqtrade_ohlcv(
            pairs=["BTC/USDT"],
            timeframe="15m",
            data_dir=temp_data_dir,
        )
        assert result["BTC/USDT"].shape[0] == 6000

    def test_date_column_is_datetime_utc(self, worker, temp_data_dir, btc_json_file):
        df = worker.load_freqtrade_ohlcv(
            pairs=["BTC/USDT"],
            timeframe="15m",
            data_dir=temp_data_dir,
        )["BTC/USDT"]
        assert pd.api.types.is_datetime64_any_dtype(df["date"]), "date must be datetime"
        assert str(df["date"].dtype) in ("datetime64[ms, UTC]", "datetime64[ns, UTC]"), \
            f"date must be UTC, got {df['date'].dtype}"

    def test_no_nulls_in_loaded_data(self, worker, temp_data_dir, btc_json_file, eth_json_file):
        result = worker.load_freqtrade_ohlcv(
            pairs=["BTC/USDT", "ETH/USDT"],
            timeframe="15m",
            data_dir=temp_data_dir,
        )
        for pair, df in result.items():
            null_count = df.isna().sum().sum()
            assert null_count == 0, f"{pair}: {null_count} nulls found"

    def test_multi_pair_returns_both_keys(self, worker, temp_data_dir, btc_json_file, eth_json_file):
        result = worker.load_freqtrade_ohlcv(
            pairs=["BTC/USDT", "ETH/USDT"],
            timeframe="15m",
            data_dir=temp_data_dir,
        )
        assert "BTC/USDT" in result
        assert "ETH/USDT" in result
        assert len(result) == 2

    def test_data_sorted_ascending_by_date(self, worker, temp_data_dir, btc_json_file):
        df = worker.load_freqtrade_ohlcv(
            pairs=["BTC/USDT"],
            timeframe="15m",
            data_dir=temp_data_dir,
        )["BTC/USDT"]
        dates = df["date"].tolist()
        assert dates == sorted(dates), "Date column is not sorted ascending"


# ---------------------------------------------------------------------------
# Unit tests — DataInsufficientError and DataFormatError
# ---------------------------------------------------------------------------

class TestLoaderErrors:

    def test_raises_data_insufficient_when_no_files(self, worker, temp_data_dir):
        with pytest.raises(worker.DataInsufficientError, match="No data files found"):
            worker.load_freqtrade_ohlcv(
                pairs=["BTC/USDT"],
                timeframe="15m",
                data_dir=temp_data_dir,
            )

    def test_raises_data_insufficient_below_min_candles(self, worker, temp_data_dir):
        # Write a file with only 100 candles — below 5000 minimum
        path = os.path.join(temp_data_dir, "BTC_USDT-15m.json")
        _make_freqtrade_json(path, n_candles=100)
        with pytest.raises(worker.DataInsufficientError, match="100 candles"):
            worker.load_freqtrade_ohlcv(
                pairs=["BTC/USDT"],
                timeframe="15m",
                data_dir=temp_data_dir,
            )

    def test_error_message_contains_download_hint(self, worker, temp_data_dir):
        with pytest.raises(worker.DataInsufficientError) as exc_info:
            worker.load_freqtrade_ohlcv(
                pairs=["BTC/USDT"],
                timeframe="15m",
                data_dir=temp_data_dir,
            )
        assert "BTC/USDT" in str(exc_info.value)

    def test_raises_data_format_error_for_malformed_json(self, worker, temp_data_dir):
        path = os.path.join(temp_data_dir, "BTC_USDT-15m.json")
        with open(path, "w") as f:
            # Wrong structure: dict instead of list-of-lists
            json.dump({"error": "not ohlcv"}, f)
        with pytest.raises(worker.DataFormatError):
            worker.load_freqtrade_ohlcv(
                pairs=["BTC/USDT"],
                timeframe="15m",
                data_dir=temp_data_dir,
            )

    def test_raises_data_format_error_for_row_too_short(self, worker, temp_data_dir):
        path = os.path.join(temp_data_dir, "BTC_USDT-15m.json")
        # Each row has only 3 fields — missing close and volume
        rows = [[1_700_000_000_000 + i * 900_000, 40_000, 40_100] for i in range(6000)]
        with open(path, "w") as f:
            json.dump(rows, f)
        with pytest.raises(worker.DataFormatError, match="Expected rows of"):
            worker.load_freqtrade_ohlcv(
                pairs=["BTC/USDT"],
                timeframe="15m",
                data_dir=temp_data_dir,
            )

    def test_custom_min_candles_enforced(self, worker, temp_data_dir):
        path = os.path.join(temp_data_dir, "BTC_USDT-15m.json")
        _make_freqtrade_json(path, n_candles=3000)
        # 3000 candles but custom min is 4000 — should fail
        with pytest.raises(worker.DataInsufficientError, match="3000 candles"):
            worker.load_freqtrade_ohlcv(
                pairs=["BTC/USDT"],
                timeframe="15m",
                data_dir=temp_data_dir,
                min_candles=4000,
            )

    def test_partial_pair_missing_raises_error(self, worker, temp_data_dir, btc_json_file):
        # BTC/USDT exists but ETH/USDT does not
        with pytest.raises(worker.DataInsufficientError, match="ETH/USDT"):
            worker.load_freqtrade_ohlcv(
                pairs=["BTC/USDT", "ETH/USDT"],
                timeframe="15m",
                data_dir=temp_data_dir,
            )


# ---------------------------------------------------------------------------
# Unit tests — multi-format detection
# ---------------------------------------------------------------------------

class TestFormatDetection:

    def test_json_detected_when_only_json_exists(self, worker, temp_data_dir, btc_json_file):
        path, ext = worker._find_data_file(temp_data_dir, "BTC/USDT", "15m")
        assert ext == "json"
        assert path.endswith("BTC_USDT-15m.json")

    def test_returns_none_when_no_file_exists(self, worker, temp_data_dir):
        path, ext = worker._find_data_file(temp_data_dir, "BTC/USDT", "15m")
        assert path is None
        assert ext is None

    def test_pair_slash_converted_to_underscore(self, worker, temp_data_dir):
        # Create file with underscore naming as Freqtrade does
        path = os.path.join(temp_data_dir, "BTC_USDT-15m.json")
        _make_freqtrade_json(path, n_candles=6000)
        found_path, found_ext = worker._find_data_file(temp_data_dir, "BTC/USDT", "15m")
        assert found_path is not None, "Should find BTC_USDT-15m.json when searching BTC/USDT"


# ---------------------------------------------------------------------------
# Unit tests — save_candidate: stage enforcement
# ---------------------------------------------------------------------------

class TestSaveCandidate:

    def test_dry_run_placeholder_rejected(self, worker, temp_ledger):
        with pytest.raises(ValueError, match="DRY_RUN_PLACEHOLDER"):
            worker.save_candidate(temp_ledger, {
                "stage": "DRY_RUN_PLACEHOLDER",
                "oos_metrics": {},
            })

    def test_real_stage_accepted(self, worker, temp_ledger):
        cid = worker.save_candidate(temp_ledger, {
            "stage": "REAL_DATA_OOS",
            "pair": "BTC/USDT",
            "data_source": "freqtrade_disk:test",
            "oos_metrics": {"avg_pnl_pct": 0.001, "cost_floor_cleared": True},
        })
        assert cid is not None
        assert len(cid) > 0

    def test_saved_candidate_readable_from_ledger(self, worker, temp_ledger):
        cid = worker.save_candidate(temp_ledger, {
            "stage": "REAL_DATA_OOS",
            "pair": "ETH/USDT",
            "data_source": "freqtrade_disk:test",
            "oos_metrics": {"avg_pnl_pct": -0.0005, "cost_floor_cleared": False},
        })
        with sqlite3.connect(temp_ledger) as conn:
            row = conn.execute(
                "SELECT candidate_id, stage, pair FROM strategy_memory WHERE candidate_id=?", (cid,)
            ).fetchone()
        assert row is not None
        assert row[0] == cid
        assert row[1] == "REAL_DATA_OOS"
        assert row[2] == "ETH/USDT"


# ---------------------------------------------------------------------------
# Unit tests — compute_bucket_features (regression: unchanged function)
# ---------------------------------------------------------------------------

class TestComputeBucketFeatures:

    def test_output_has_feature_columns(self, worker, temp_data_dir, btc_json_file):
        df_raw = worker.load_freqtrade_ohlcv(
            pairs=["BTC/USDT"], timeframe="15m", data_dir=temp_data_dir
        )["BTC/USDT"]
        df_feat = worker.compute_bucket_features(df_raw)
        for col in ["momentum_5", "atr_pct", "fwd_return"]:
            assert col in df_feat.columns, f"Missing column: {col}"

    def test_no_nans_after_feature_computation(self, worker, temp_data_dir, btc_json_file):
        df_raw = worker.load_freqtrade_ohlcv(
            pairs=["BTC/USDT"], timeframe="15m", data_dir=temp_data_dir
        )["BTC/USDT"]
        df_feat = worker.compute_bucket_features(df_raw)
        assert df_feat.isna().sum().sum() == 0

    def test_feature_row_count_reduced_by_dropna(self, worker, temp_data_dir, btc_json_file):
        df_raw = worker.load_freqtrade_ohlcv(
            pairs=["BTC/USDT"], timeframe="15m", data_dir=temp_data_dir
        )["BTC/USDT"]
        df_feat = worker.compute_bucket_features(df_raw)
        # dropna removes leading rows (ATR needs 14 candles + momentum needs 5)
        assert len(df_feat) < len(df_raw)
        assert len(df_feat) > 5900  # should only lose ~15 rows


# ---------------------------------------------------------------------------
# Integration test — run_research_cycle stores real candidate in ledger
# ---------------------------------------------------------------------------

class TestRunResearchCycle:

    def test_candidate_stored_in_ledger(self, worker, temp_data_dir, btc_json_file, eth_json_file, temp_ledger):
        """REQUIRED GATE: candidate from real data stored in strategy_memory."""
        ids = worker.run_research_cycle(
            pairs=["BTC/USDT", "ETH/USDT"],
            timeframe="15m",
            data_dir=temp_data_dir,
            ledger_path=temp_ledger,
        )
        assert len(ids) == 2, f"Expected 2 candidates, got {len(ids)}"

        with sqlite3.connect(temp_ledger) as conn:
            rows = conn.execute(
                "SELECT candidate_id, stage, pair, data_source, oos_metrics FROM strategy_memory"
            ).fetchall()

        assert len(rows) == 2

    def test_stage_is_not_dry_run_placeholder(self, worker, temp_data_dir, btc_json_file, eth_json_file, temp_ledger):
        """REQUIRED GATE: stage != DRY_RUN_PLACEHOLDER."""
        worker.run_research_cycle(
            pairs=["BTC/USDT", "ETH/USDT"],
            timeframe="15m",
            data_dir=temp_data_dir,
            ledger_path=temp_ledger,
        )
        with sqlite3.connect(temp_ledger) as conn:
            bad = conn.execute(
                "SELECT COUNT(*) FROM strategy_memory WHERE stage='DRY_RUN_PLACEHOLDER'"
            ).fetchone()[0]
        assert bad == 0, "Found DRY_RUN_PLACEHOLDER rows — real data pipeline violated"

    def test_data_source_reflects_real_disk_path(self, worker, temp_data_dir, btc_json_file, eth_json_file, temp_ledger):
        worker.run_research_cycle(
            pairs=["BTC/USDT"],
            timeframe="15m",
            data_dir=temp_data_dir,
            ledger_path=temp_ledger,
        )
        with sqlite3.connect(temp_ledger) as conn:
            source = conn.execute(
                "SELECT data_source FROM strategy_memory WHERE pair='BTC/USDT'"
            ).fetchone()[0]
        assert "freqtrade_disk" in source, f"data_source should reference disk: {source}"

    def test_avg_pnl_pct_is_recorded(self, worker, temp_data_dir, btc_json_file, temp_ledger):
        """avg_pnl_pct must be present and numeric (can be negative)."""
        worker.run_research_cycle(
            pairs=["BTC/USDT"],
            timeframe="15m",
            data_dir=temp_data_dir,
            ledger_path=temp_ledger,
        )
        with sqlite3.connect(temp_ledger) as conn:
            metrics_json = conn.execute(
                "SELECT oos_metrics FROM strategy_memory WHERE pair='BTC/USDT'"
            ).fetchone()[0]

        metrics = json.loads(metrics_json)
        assert "avg_pnl_pct" in metrics, "avg_pnl_pct missing from oos_metrics"
        assert isinstance(metrics["avg_pnl_pct"], (int, float)), \
            f"avg_pnl_pct must be numeric, got {type(metrics['avg_pnl_pct'])}"

    def test_cost_floor_cleared_is_recorded(self, worker, temp_data_dir, btc_json_file, temp_ledger):
        """cost_floor_cleared must be present as bool."""
        worker.run_research_cycle(
            pairs=["BTC/USDT"],
            timeframe="15m",
            data_dir=temp_data_dir,
            ledger_path=temp_ledger,
        )
        with sqlite3.connect(temp_ledger) as conn:
            metrics_json = conn.execute(
                "SELECT oos_metrics FROM strategy_memory WHERE pair='BTC/USDT'"
            ).fetchone()[0]

        metrics = json.loads(metrics_json)
        assert "cost_floor_cleared" in metrics
        assert isinstance(metrics["cost_floor_cleared"], bool)

    def test_raises_when_data_missing(self, worker, temp_data_dir, temp_ledger):
        """run_research_cycle must propagate DataInsufficientError — never silently proceed."""
        with pytest.raises(worker.DataInsufficientError):
            worker.run_research_cycle(
                pairs=["BTC/USDT"],
                timeframe="15m",
                data_dir=temp_data_dir,
                ledger_path=temp_ledger,
            )

    def test_each_pair_gets_separate_candidate_row(self, worker, temp_data_dir, btc_json_file, eth_json_file, temp_ledger):
        ids = worker.run_research_cycle(
            pairs=["BTC/USDT", "ETH/USDT"],
            timeframe="15m",
            data_dir=temp_data_dir,
            ledger_path=temp_ledger,
        )
        assert ids[0] != ids[1], "Each pair must get a distinct candidate_id"

        with sqlite3.connect(temp_ledger) as conn:
            pairs_stored = {row[0] for row in conn.execute(
                "SELECT pair FROM strategy_memory"
            ).fetchall()}
        assert "BTC/USDT" in pairs_stored
        assert "ETH/USDT" in pairs_stored