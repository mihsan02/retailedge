"""
tests/test_atr_percentile_service.py

Test gate: Dry-run.

Required test (Sprint 2 S2-5 gate):
- test_atr_percentile_min_lookback

All tests use synthetic DataFrames — no exchange data needed.
Pandas and numpy only.
"""

import pytest
import numpy as np
import pandas as pd

from research.atr_percentile_service import (
    compute_atr_percentile_baseline,
    add_atr_pct,
    ATRPercentileService,
    MIN_LOOKBACK_CANDLES,
    HIGH_VOL_THRESHOLD,
    SPIKE_THRESHOLD,
    ATR_COLUMN,
)


# ---------------------------------------------------------------------------
# DataFrame fixtures
# ---------------------------------------------------------------------------

def make_df(n: int, atr_values=None, add_nan_prefix: int = 0) -> pd.DataFrame:
    """
    Build a minimal DataFrame with atr_pct column.

    n: total rows.
    atr_values: if None, linearly increasing from 0.001 to 0.02 (realistic range).
    add_nan_prefix: first N rows will have NaN atr_pct (simulates ATR warmup period).
    """
    if atr_values is None:
        atr_values = np.linspace(0.001, 0.020, n)

    df = pd.DataFrame({
        "open": np.ones(n) * 50000,
        "high": np.ones(n) * 50500,
        "low": np.ones(n) * 49500,
        "close": np.ones(n) * 50000,
        ATR_COLUMN: atr_values,
    })

    if add_nan_prefix > 0:
        df.loc[df.index[:add_nan_prefix], ATR_COLUMN] = np.nan

    return df


# ---------------------------------------------------------------------------
# test_atr_percentile_min_lookback  ← REQUIRED GATE
# ---------------------------------------------------------------------------

def test_atr_percentile_min_lookback():
    """
    compute_atr_percentile_baseline must:
    - Return ATR_BASELINE_INSUFFICIENT when valid candles < 5000.
    - Return OK when valid candles >= 5000.
    - atr_percentile must be None when insufficient.
    - atr_percentile must be a float in [0, 1] when OK.

    Blueprint v1.9 Section 6 Step 6:
        min_lookback_candles: 5000
        fallback_if_insufficient_history: ATR_BASELINE_INSUFFICIENT_PAUSE_FOR_MICRO_LIVE

    This test is the reason ATR percentile cannot be computed from arbitrary
    latest-dataframe quantiles: without a stable 5000-candle baseline,
    the percentile is meaningless and could trigger wrong audit intervals.
    """
    # --- Insufficient: exactly min_lookback - 1 candles
    df_short = make_df(MIN_LOOKBACK_CANDLES - 1)
    result_short = compute_atr_percentile_baseline(df_short, min_lookback=MIN_LOOKBACK_CANDLES)

    assert result_short["status"] == "ATR_BASELINE_INSUFFICIENT", (
        f"Expected ATR_BASELINE_INSUFFICIENT for {MIN_LOOKBACK_CANDLES - 1} candles, "
        f"got {result_short['status']}"
    )
    assert result_short["atr_percentile"] is None, (
        "atr_percentile must be None when baseline is insufficient"
    )
    assert result_short["baseline_n"] == MIN_LOOKBACK_CANDLES - 1
    assert result_short["required_n"] == MIN_LOOKBACK_CANDLES

    # --- Exactly at threshold: min_lookback candles
    df_exact = make_df(MIN_LOOKBACK_CANDLES)
    result_exact = compute_atr_percentile_baseline(df_exact, min_lookback=MIN_LOOKBACK_CANDLES)

    assert result_exact["status"] == "OK", (
        f"Expected OK for exactly {MIN_LOOKBACK_CANDLES} candles, got {result_exact['status']}"
    )
    assert result_exact["atr_percentile"] is not None
    assert 0.0 <= result_exact["atr_percentile"] <= 1.0, (
        f"atr_percentile must be in [0, 1], got {result_exact['atr_percentile']}"
    )

    # --- Above threshold: min_lookback + 500 candles
    df_long = make_df(MIN_LOOKBACK_CANDLES + 500)
    result_long = compute_atr_percentile_baseline(df_long, min_lookback=MIN_LOOKBACK_CANDLES)

    assert result_long["status"] == "OK"
    assert 0.0 <= result_long["atr_percentile"] <= 1.0

    # --- Zero candles
    df_empty = make_df(0, atr_values=np.array([]))
    result_empty = compute_atr_percentile_baseline(df_empty, min_lookback=MIN_LOOKBACK_CANDLES)
    assert result_empty["status"] == "ATR_BASELINE_INSUFFICIENT"
    assert result_empty["atr_percentile"] is None


# ---------------------------------------------------------------------------
# Percentile computation correctness
# ---------------------------------------------------------------------------

def test_percentile_latest_is_maximum_returns_1():
    """
    If current ATR is the maximum in the series, percentile must be 1.0.
    Linearly increasing series: last value is always max.
    """
    df = make_df(MIN_LOOKBACK_CANDLES, atr_values=np.linspace(0.001, 0.020, MIN_LOOKBACK_CANDLES))
    result = compute_atr_percentile_baseline(df)
    assert result["status"] == "OK"
    # Last value (0.020) is >= all others -> percentile == 1.0
    assert result["atr_percentile"] == pytest.approx(1.0, abs=1e-6)


def test_percentile_latest_is_minimum_returns_near_zero():
    """
    If current ATR is the minimum in the series, percentile must be near 0.
    Linearly decreasing series: last value is always min.
    """
    df = make_df(MIN_LOOKBACK_CANDLES, atr_values=np.linspace(0.020, 0.001, MIN_LOOKBACK_CANDLES))
    result = compute_atr_percentile_baseline(df)
    assert result["status"] == "OK"
    # Last value (0.001) <= all others -> 1/n fraction (only itself)
    assert result["atr_percentile"] == pytest.approx(1.0 / MIN_LOOKBACK_CANDLES, abs=1e-4)


def test_percentile_median_value_near_0_5():
    """
    If current ATR equals the median of the series, percentile must be ~0.5.
    """
    n = MIN_LOOKBACK_CANDLES
    values = np.linspace(0.001, 0.020, n)
    # Place the median value at the last position
    median_val = float(np.median(values))
    values[-1] = median_val

    df = make_df(n, atr_values=values)
    result = compute_atr_percentile_baseline(df)
    assert result["status"] == "OK"
    # Median splits the distribution 50/50
    assert 0.45 <= result["atr_percentile"] <= 0.55, (
        f"Median ATR should give percentile ~0.5, got {result['atr_percentile']}"
    )


def test_high_vol_flag_matches_threshold():
    """
    is_high_vol must be True iff atr_percentile >= HIGH_VOL_THRESHOLD (0.80).
    """
    n = MIN_LOOKBACK_CANDLES
    # Build series where last value is at exactly the 80th percentile
    values = np.linspace(0.001, 0.020, n)
    p80_val = float(np.percentile(values, 80))
    values[-1] = p80_val

    df = make_df(n, atr_values=values)
    result = compute_atr_percentile_baseline(df)
    assert result["status"] == "OK"

    # At or above 0.80: is_high_vol must be True
    if result["atr_percentile"] >= HIGH_VOL_THRESHOLD:
        assert result["is_high_vol"] is True
    else:
        assert result["is_high_vol"] is False


def test_spike_flag_at_0_90_threshold():
    """is_spike must be True iff atr_percentile >= 0.90."""
    n = MIN_LOOKBACK_CANDLES
    # Latest = max -> percentile = 1.0 -> both is_high_vol and is_spike True
    values = np.linspace(0.001, 0.020, n)
    df = make_df(n, atr_values=values)
    result = compute_atr_percentile_baseline(df)
    assert result["is_spike"] is True
    assert result["is_high_vol"] is True


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_nan_rows_excluded_from_baseline():
    """
    NaN ATR values (ATR warmup period) must not count toward min_lookback.
    A DataFrame with 5100 rows but 200 NaN prefix must behave as 4900 valid = insufficient.
    """
    total = 5100
    nan_prefix = 200
    valid_n = total - nan_prefix  # 4900 < 5000

    df = make_df(total, add_nan_prefix=nan_prefix)
    result = compute_atr_percentile_baseline(df, min_lookback=MIN_LOOKBACK_CANDLES)

    assert result["status"] == "ATR_BASELINE_INSUFFICIENT", (
        f"5100 total - 200 NaN = 4900 valid < 5000: expected INSUFFICIENT, got {result['status']}"
    )
    assert result["baseline_n"] == valid_n


def test_sufficient_valid_after_nan_prefix():
    """
    5200 total - 100 NaN = 5100 valid >= 5000: must return OK.
    """
    total = 5200
    nan_prefix = 100
    df = make_df(total, add_nan_prefix=nan_prefix)
    result = compute_atr_percentile_baseline(df, min_lookback=MIN_LOOKBACK_CANDLES)
    assert result["status"] == "OK"
    assert result["baseline_n"] == MIN_LOOKBACK_CANDLES  # capped at min_lookback


# ---------------------------------------------------------------------------
# Expanding window cap
# ---------------------------------------------------------------------------

def test_baseline_capped_at_min_lookback():
    """
    With 8000 valid candles, baseline_n must be capped at min_lookback (5000).
    The extra 3000 candles are excluded from the population.
    """
    n = 8000
    df = make_df(n)
    result = compute_atr_percentile_baseline(df, min_lookback=MIN_LOOKBACK_CANDLES)
    assert result["status"] == "OK"
    assert result["baseline_n"] == MIN_LOOKBACK_CANDLES, (
        f"baseline_n must be capped at {MIN_LOOKBACK_CANDLES}, got {result['baseline_n']}"
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_missing_atr_column_returns_error():
    """Missing atr_pct column must return ATR_BASELINE_ERROR, not raise."""
    df = pd.DataFrame({"close": [50000.0] * 100})
    result = compute_atr_percentile_baseline(df)
    assert result["status"] == "ATR_BASELINE_ERROR"
    assert "atr_pct" in result["error"]
    assert result["atr_percentile"] is None


def test_all_nan_atr_column_returns_insufficient():
    """All-NaN atr_pct column = 0 valid candles = INSUFFICIENT."""
    df = make_df(5000)
    df[ATR_COLUMN] = np.nan
    result = compute_atr_percentile_baseline(df)
    assert result["status"] == "ATR_BASELINE_INSUFFICIENT"
    assert result["baseline_n"] == 0


# ---------------------------------------------------------------------------
# add_atr_pct helper
# ---------------------------------------------------------------------------

def test_add_atr_pct_produces_valid_column():
    """add_atr_pct must produce a non-NaN atr_pct column (after warmup)."""
    n = 100
    df = pd.DataFrame({
        "open": np.ones(n) * 50000,
        "high": np.ones(n) * 50500 + np.random.rand(n) * 100,
        "low": np.ones(n) * 49500 - np.random.rand(n) * 100,
        "close": np.ones(n) * 50000 + np.random.rand(n) * 50,
    })
    result_df = add_atr_pct(df, period=14)
    assert ATR_COLUMN in result_df.columns
    # After period warmup, values should be non-NaN
    valid = result_df[ATR_COLUMN].dropna()
    assert len(valid) > 0
    assert (valid > 0).all(), "atr_pct must be positive"


def test_add_atr_pct_does_not_mutate_input():
    """add_atr_pct must return a copy, not mutate the input."""
    df = pd.DataFrame({
        "open": [50000.0] * 50,
        "high": [50500.0] * 50,
        "low": [49500.0] * 50,
        "close": [50000.0] * 50,
    })
    original_cols = set(df.columns)
    add_atr_pct(df)
    assert set(df.columns) == original_cols, "Input DataFrame must not be mutated"


# ---------------------------------------------------------------------------
# ATRPercentileService
# ---------------------------------------------------------------------------

def test_service_returns_insufficient_on_first_call_with_small_df():
    """Service must return INSUFFICIENT if provider returns < 5000 candles."""
    small_df = make_df(100)
    service = ATRPercentileService(
        df_provider=lambda: small_df,
        refresh_hours=6,
        min_lookback=MIN_LOOKBACK_CANDLES,
    )
    state = service.get_current_state()
    assert state["status"] == "ATR_BASELINE_INSUFFICIENT"
    assert state["atr_percentile"] is None


def test_service_returns_ok_with_sufficient_df():
    """Service must return OK if provider returns >= 5000 candles."""
    large_df = make_df(MIN_LOOKBACK_CANDLES + 100)
    service = ATRPercentileService(
        df_provider=lambda: large_df,
        refresh_hours=6,
        min_lookback=MIN_LOOKBACK_CANDLES,
    )
    state = service.get_current_state()
    assert state["status"] == "OK"
    assert state["atr_percentile"] is not None