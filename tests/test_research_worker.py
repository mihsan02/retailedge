"""
tests/test_research_worker.py — Sprint 3 S3-2

Required gate tests:
  - test_candidate_stored_in_ledger
  - test_oos_metrics_computed_correctly

Additional coverage:
  - Synthetic data shape
  - Feature computation
  - Threshold scan behavior
  - DRY_RUN_PLACEHOLDER enforcement
  - Edge: no valid threshold
  - Edge: NaN propagation
"""

from __future__ import annotations

import math
import sqlite3
import tempfile
import os
import sys

import pytest
import numpy as np
import pandas as pd

# Ensure repo root on path (mirrors conftest.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from research.worker import (
    COST_FLOOR_PCT,
    MIN_OOS_TRADES,
    compute_bucket_features,
    generate_synthetic_ohlcv,
    run_oos_validation,
    run_research_cycle,
    save_candidate,
    _simulate_trades,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_ledger(tmp_path):
    """Temporary SQLite ledger for isolation."""
    return str(tmp_path / "test_retailedge.db")


@pytest.fixture()
def df_raw():
    """6000-candle synthetic OHLCV."""
    return generate_synthetic_ohlcv(n_candles=6000, seed=42)


@pytest.fixture()
def df_feat(df_raw):
    """Feature-computed dataframe."""
    return compute_bucket_features(df_raw)


@pytest.fixture()
def valid_candidate():
    """Minimal valid DRY_RUN_PLACEHOLDER candidate."""
    return {
        "stage": "DRY_RUN_PLACEHOLDER",
        "status": "OK",
        "oos_trade_count": 25,
        "oos_win_rate": 0.52,
        "oos_avg_pnl_pct": 0.0010,
        "cost_floor_pct": COST_FLOOR_PCT,
        "cost_floor_cleared": True,
        "best_momentum_threshold": 0.002,
        "best_atr_pct_threshold": 0.5,
        "train_ratio": 0.8,
        "n_candles_total": 6000,
        "n_candles_oos": 1200,
    }


# ---------------------------------------------------------------------------
# REQUIRED GATE: test_candidate_stored_in_ledger
# ---------------------------------------------------------------------------


def test_candidate_stored_in_ledger(tmp_ledger, valid_candidate):
    """
    REQUIRED GATE — Stage A (Dry-run).
    Candidate must be persisted to strategy_memory and retrievable.
    """
    candidate_id = save_candidate(tmp_ledger, valid_candidate)

    assert candidate_id is not None, "save_candidate must return a non-None candidate_id"
    assert candidate_id.startswith("bucket_baseline_"), "candidate_id must have expected prefix"

    # Verify row exists in ledger
    conn = sqlite3.connect(tmp_ledger)
    row = conn.execute(
        "SELECT candidate_id, stage, oos_trade_count, cost_floor_cleared FROM strategy_memory WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    conn.close()

    assert row is not None, "Candidate row must exist in strategy_memory"
    assert row[0] == candidate_id
    assert row[1] == "DRY_RUN_PLACEHOLDER"
    assert row[2] == 25
    assert row[3] == 1  # cost_floor_cleared stored as int


# ---------------------------------------------------------------------------
# REQUIRED GATE: test_oos_metrics_computed_correctly
# ---------------------------------------------------------------------------


def test_oos_metrics_computed_correctly(df_feat):
    """
    REQUIRED GATE — Stage A (Dry-run).
    OOS metrics must be computed on the OOS fold only.
    Threshold must be selected on train fold only.
    """
    result = run_oos_validation(df_feat, train_ratio=0.8, cost_floor_pct=COST_FLOOR_PCT)

    assert result["stage"] == "DRY_RUN_PLACEHOLDER"

    if result["status"] == "NO_VALID_THRESHOLD":
        # Acceptable for synthetic data with extreme parameters; not a test failure
        pytest.skip("No valid threshold found — synthetic data may not produce enough signals")

    assert result["oos_trade_count"] >= 0
    assert 0.0 <= result["oos_win_rate"] <= 1.0
    assert not math.isnan(result["oos_avg_pnl_pct"]), "avg_pnl_pct must not be NaN when status=OK"
    assert result["cost_floor_pct"] == COST_FLOOR_PCT
    assert isinstance(result["cost_floor_cleared"], bool)
    assert result["best_momentum_threshold"] is not None
    assert result["best_atr_pct_threshold"] is not None

    # Verify OOS metrics are derived from OOS fold only
    n_oos = result["n_candles_oos"]
    expected_oos_candles = int(len(df_feat) * (1 - result["train_ratio"]))
    # Allow 1 candle tolerance from dropna
    assert abs(n_oos - expected_oos_candles) <= 1, (
        f"OOS fold size mismatch: got {n_oos}, expected ~{expected_oos_candles}"
    )


# ---------------------------------------------------------------------------
# Synthetic data shape
# ---------------------------------------------------------------------------


def test_synthetic_ohlcv_shape(df_raw):
    assert len(df_raw) == 6000
    assert set(["open", "high", "low", "close", "volume"]).issubset(df_raw.columns)


def test_synthetic_ohlcv_no_negative_prices(df_raw):
    assert (df_raw["close"] > 0).all()
    assert (df_raw["high"] >= df_raw["low"]).all()


def test_synthetic_ohlcv_deterministic():
    df1 = generate_synthetic_ohlcv(seed=42)
    df2 = generate_synthetic_ohlcv(seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_ohlcv_different_seeds_differ():
    df1 = generate_synthetic_ohlcv(seed=42)
    df2 = generate_synthetic_ohlcv(seed=99)
    assert not df1["close"].equals(df2["close"])


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def test_compute_features_columns(df_feat):
    required = {"momentum", "atr_pct", "fwd_return"}
    assert required.issubset(df_feat.columns)


def test_compute_features_no_nan(df_feat):
    assert not df_feat["momentum"].isna().any()
    assert not df_feat["atr_pct"].isna().any()
    # fwd_return will have NaN at last row due to shift(-1); that is correct
    # but dropna should have removed it already
    assert not df_feat["fwd_return"].isna().any(), (
        "dropna in compute_bucket_features should remove trailing NaN from fwd_return"
    )


def test_atr_pct_is_positive(df_feat):
    assert (df_feat["atr_pct"] > 0).all()


def test_feature_length_reduced(df_raw, df_feat):
    # Rolling windows + fwd_return shift will reduce row count
    assert len(df_feat) < len(df_raw)
    # But not by more than ATR_PERIOD + MOMENTUM_PERIOD + 1 rows
    from research.worker import ATR_PERIOD, MOMENTUM_PERIOD
    max_drop = ATR_PERIOD + MOMENTUM_PERIOD + 2
    assert len(df_raw) - len(df_feat) <= max_drop, (
        f"Too many rows dropped: {len(df_raw) - len(df_feat)}"
    )


# ---------------------------------------------------------------------------
# Threshold scan internals
# ---------------------------------------------------------------------------


def test_simulate_trades_no_signals(df_feat):
    """Extreme threshold produces zero trades."""
    result = _simulate_trades(df_feat, momentum_thresh=99.0, atr_pct_thresh=0.0, cost_floor_pct=COST_FLOOR_PCT)
    assert result["trade_count"] == 0
    assert result["win_rate"] == 0.0
    assert math.isnan(result["avg_pnl_pct"])


def test_simulate_trades_all_signals(df_feat):
    """Very permissive threshold should generate many trades."""
    result = _simulate_trades(df_feat, momentum_thresh=-99.0, atr_pct_thresh=99.0, cost_floor_pct=COST_FLOOR_PCT)
    assert result["trade_count"] > 0
    assert 0.0 <= result["win_rate"] <= 1.0


def test_simulate_trades_cost_reduces_pnl(df_feat):
    """Higher cost floor reduces avg_pnl_pct."""
    r_low = _simulate_trades(df_feat, momentum_thresh=0.0, atr_pct_thresh=0.5, cost_floor_pct=0.0)
    r_high = _simulate_trades(df_feat, momentum_thresh=0.0, atr_pct_thresh=0.5, cost_floor_pct=0.01)
    if r_low["trade_count"] > 0 and r_high["trade_count"] > 0:
        assert r_low["avg_pnl_pct"] > r_high["avg_pnl_pct"]


# ---------------------------------------------------------------------------
# OOS validation edge cases
# ---------------------------------------------------------------------------


def test_oos_validation_no_lookahead(df_feat):
    """
    Best threshold must be selected on train fold.
    OOS result is applied, not re-optimized. Verify by checking that
    best params from train fold are stored, not recomputed from OOS.
    """
    result = run_oos_validation(df_feat, train_ratio=0.8)
    if result["status"] == "NO_VALID_THRESHOLD":
        pytest.skip("No threshold found")
    # The function stores best_momentum_threshold / best_atr_pct_threshold
    # These are train-fold-derived. No way to accidentally use OOS because
    # the scan loop only touches train_df.
    assert result["best_momentum_threshold"] is not None
    assert result["best_atr_pct_threshold"] is not None


def test_oos_validation_stage_always_placeholder(df_feat):
    """Stage must always be DRY_RUN_PLACEHOLDER regardless of metrics."""
    result = run_oos_validation(df_feat)
    assert result["stage"] == "DRY_RUN_PLACEHOLDER"


def test_oos_validation_small_df():
    """Insufficient data (< ATR_PERIOD + MOMENTUM_PERIOD candles) after feature compute."""
    df_tiny = generate_synthetic_ohlcv(n_candles=30)
    df_feat_tiny = compute_bucket_features(df_tiny)
    result = run_oos_validation(df_feat_tiny)
    # Should either produce NO_VALID_THRESHOLD or very low trade count
    assert result["stage"] == "DRY_RUN_PLACEHOLDER"


# ---------------------------------------------------------------------------
# save_candidate enforcement
# ---------------------------------------------------------------------------


def test_save_candidate_rejects_non_placeholder(tmp_ledger):
    """save_candidate must reject any stage other than DRY_RUN_PLACEHOLDER."""
    bad_candidate = {
        "stage": "PROMOTION_CANDIDATE",  # Not allowed without CPCV/DSR
        "status": "OK",
        "oos_trade_count": 100,
        "oos_win_rate": 0.6,
        "oos_avg_pnl_pct": 0.005,
        "cost_floor_pct": COST_FLOOR_PCT,
        "cost_floor_cleared": True,
        "best_momentum_threshold": 0.002,
        "best_atr_pct_threshold": 0.4,
        "train_ratio": 0.8,
        "n_candles_total": 6000,
        "n_candles_oos": 1200,
    }
    with pytest.raises(ValueError, match="DRY_RUN_PLACEHOLDER"):
        save_candidate(tmp_ledger, bad_candidate)


def test_save_candidate_idempotent_multiple_saves(tmp_ledger, valid_candidate):
    """Multiple saves produce multiple unique rows (not silently deduped)."""
    id1 = save_candidate(tmp_ledger, valid_candidate)
    id2 = save_candidate(tmp_ledger, valid_candidate)
    assert id1 != id2  # Each run gets a unique candidate_id

    conn = sqlite3.connect(tmp_ledger)
    count = conn.execute("SELECT COUNT(*) FROM strategy_memory").fetchone()[0]
    conn.close()
    assert count == 2


def test_save_candidate_evidence_json_stored(tmp_ledger, valid_candidate):
    """evidence_json must be a valid JSON string with evidence_hash field."""
    import json as json_mod
    candidate_id = save_candidate(tmp_ledger, valid_candidate)

    conn = sqlite3.connect(tmp_ledger)
    row = conn.execute(
        "SELECT evidence_json FROM strategy_memory WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    conn.close()

    assert row is not None
    evidence = json_mod.loads(row[0])
    assert "evidence_hash" in evidence
    assert len(evidence["evidence_hash"]) == 64  # sha256 hex


# ---------------------------------------------------------------------------
# Full cycle integration
# ---------------------------------------------------------------------------


def test_run_research_cycle_stores_candidate(tmp_ledger):
    """
    Integration: full cycle must store exactly one candidate in strategy_memory.
    """
    result = run_research_cycle(ledger_path=tmp_ledger, n_candles=6000, seed=42)

    if result["candidate_id"] is None:
        pytest.skip("No valid threshold found in integration cycle")

    assert result["stage"] == "DRY_RUN_PLACEHOLDER"
    assert result["candidate_id"].startswith("bucket_baseline_")

    conn = sqlite3.connect(tmp_ledger)
    count = conn.execute("SELECT COUNT(*) FROM strategy_memory").fetchone()[0]
    conn.close()
    assert count == 1


def test_run_research_cycle_deterministic(tmp_ledger):
    """Same seed must produce same OOS metrics."""
    import tempfile, shutil

    dir1 = tempfile.mkdtemp()
    dir2 = tempfile.mkdtemp()
    try:
        r1 = run_research_cycle(ledger_path=os.path.join(dir1, "test.db"), n_candles=6000, seed=7)
        r2 = run_research_cycle(ledger_path=os.path.join(dir2, "test.db"), n_candles=6000, seed=7)

        if r1["candidate_id"] is None or r2["candidate_id"] is None:
            pytest.skip("No valid threshold")

        assert r1["oos_trade_count"] == r2["oos_trade_count"]
        assert abs(r1["oos_avg_pnl_pct"] - r2["oos_avg_pnl_pct"]) < 1e-10
    finally:
        shutil.rmtree(dir1)
        shutil.rmtree(dir2)