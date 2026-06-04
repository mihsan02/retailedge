"""
tests/test_regime_policy.py — Sprint 3 S3-3

Required gate tests:
  - test_volatile_regime_detected       (ATR percentile >= 0.80)
  - test_ranging_regime_multiplier_lt_trending
  - test_regime_logged_to_ledger

Additional coverage:
  - DM ratio trending detection
  - Missing ATR baseline defaults to VOLATILE
  - Insufficient candles defaults to VOLATILE
  - get_position_multiplier rejects unknown regime
  - Multiplier ordering: trending > ranging > volatile
  - log_regime rejects unknown regime
  - evaluate_regime_policy integration
"""

from __future__ import annotations

import math
import os
import sqlite3
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sidecar.guardian.regime_policy import (
    DM_LOOKBACK,
    DM_RATIO_THRESHOLD,
    MIN_CANDLES,
    REGIME_RANGING,
    REGIME_TRENDING,
    REGIME_VOLATILE,
    VOLATILE_ATR_PERCENTILE,
    _compute_dm_ratio,
    detect_regime,
    evaluate_regime_policy,
    get_position_multiplier,
    log_regime,
)


# ---------------------------------------------------------------------------
# OHLCV builders
# ---------------------------------------------------------------------------


def _make_trending_df(n: int = 60, slope: float = 50.0) -> pd.DataFrame:
    """Strong uptrend: consistent directional movement, low noise."""
    rng = np.random.default_rng(1)
    close = 30000.0 + np.arange(n) * slope + rng.normal(0, 5, n)
    noise = rng.uniform(0.0005, 0.001, n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0})


def _make_ranging_df(n: int = 60) -> pd.DataFrame:
    """
    Choppy market: random walk with tiny net movement per candle, large bar range.

    DM ratio = abs(DM+ - DM-) / ATR.
    When ATR (bar range) is large relative to net directional movement, ratio is LOW.
    This is the economically correct definition of ranging: prices chop, no trend.

    Note: do NOT use white noise or a sine wave here.
    White noise gives high DM ratio (~0.58) because each bar is small relative to move.
    Sine wave gives locally directional DM within each half-period.
    """
    rng = np.random.default_rng(2)
    # Small random walk steps — low net directional movement
    steps = rng.normal(0, 2.0, n)
    close = 30000.0 + steps.cumsum()
    # Large bar range (0.4-0.8% per bar) — ATR >> net move => low DM ratio
    noise = rng.uniform(0.004, 0.008, n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0})


def _make_volatile_df(n: int = 60) -> pd.DataFrame:
    """High noise, large bars — produces high ATR pct."""
    rng = np.random.default_rng(3)
    close = 30000.0 + rng.normal(0, 200, n).cumsum()
    close = np.abs(close) + 1000  # keep positive
    noise = rng.uniform(0.015, 0.030, n)  # large bar range
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0})


@pytest.fixture()
def tmp_ledger(tmp_path):
    return str(tmp_path / "test.db")


# ---------------------------------------------------------------------------
# REQUIRED GATE: test_volatile_regime_detected
# ---------------------------------------------------------------------------


def test_volatile_regime_detected():
    """
    REQUIRED GATE.
    ATR percentile >= 0.80 must produce VOLATILE regardless of DM ratio.
    Test with a strong trend df (would be TRENDING without high ATR pct).
    """
    df = _make_trending_df(n=60)
    # Force ATR percentile above threshold
    regime, diag = detect_regime(df, atr_percentile=0.85)
    assert regime == REGIME_VOLATILE, (
        f"Expected VOLATILE for atr_percentile=0.85, got {regime}"
    )


def test_volatile_regime_at_exact_threshold():
    """Boundary: atr_percentile == 0.80 must still be VOLATILE."""
    df = _make_trending_df(n=60)
    regime, _ = detect_regime(df, atr_percentile=VOLATILE_ATR_PERCENTILE)
    assert regime == REGIME_VOLATILE


def test_volatile_regime_missing_atr_baseline():
    """
    REQUIRED GATE behavior: missing ATR baseline (None) must default to VOLATILE.
    This is the conservative fallback — never trade without a volatility baseline.
    """
    df = _make_trending_df(n=60)
    regime, _ = detect_regime(df, atr_percentile=None)
    assert regime == REGIME_VOLATILE


def test_volatile_regime_nan_atr_baseline():
    """NaN ATR percentile is treated same as None — defaults to VOLATILE."""
    df = _make_trending_df(n=60)
    regime, _ = detect_regime(df, atr_percentile=float("nan"))
    assert regime == REGIME_VOLATILE


def test_volatile_regime_insufficient_candles():
    """Fewer than MIN_CANDLES rows defaults to VOLATILE."""
    df = _make_trending_df(n=MIN_CANDLES - 1)
    regime, diag = detect_regime(df, atr_percentile=0.3)
    assert regime == REGIME_VOLATILE
    assert diag["min_candles_met"] is False


# ---------------------------------------------------------------------------
# REQUIRED GATE: test_ranging_regime_multiplier_lt_trending
# ---------------------------------------------------------------------------


def test_ranging_regime_multiplier_lt_trending():
    """
    REQUIRED GATE.
    Multiplier for RANGING must be strictly less than TRENDING.
    """
    m_trending = get_position_multiplier(REGIME_TRENDING)
    m_ranging = get_position_multiplier(REGIME_RANGING)
    assert m_ranging < m_trending, (
        f"RANGING multiplier ({m_ranging}) must be < TRENDING ({m_trending})"
    )


def test_multiplier_ordering():
    """Full ordering: trending > ranging > volatile."""
    m_t = get_position_multiplier(REGIME_TRENDING)
    m_r = get_position_multiplier(REGIME_RANGING)
    m_v = get_position_multiplier(REGIME_VOLATILE)
    assert m_t > m_r > m_v


def test_multiplier_values_are_positive():
    for regime in [REGIME_TRENDING, REGIME_RANGING, REGIME_VOLATILE]:
        assert get_position_multiplier(regime) > 0


def test_multiplier_rejects_unknown_regime():
    with pytest.raises(ValueError, match="Unknown regime"):
        get_position_multiplier("sideways")


# ---------------------------------------------------------------------------
# REQUIRED GATE: test_regime_logged_to_ledger
# ---------------------------------------------------------------------------


def test_regime_logged_to_ledger(tmp_ledger):
    """
    REQUIRED GATE.
    log_regime must write a row to regime_log retrievable by log_id.
    """
    log_id = log_regime(
        ledger_path=tmp_ledger,
        pair="BTC/USDT",
        regime=REGIME_TRENDING,
        multiplier=1.0,
        atr_pct=0.0012,
        dm_ratio=0.45,
    )

    assert log_id.startswith("regime_")

    conn = sqlite3.connect(tmp_ledger)
    row = conn.execute(
        "SELECT pair, regime, multiplier, atr_pct, dm_ratio FROM regime_log WHERE log_id = ?",
        (log_id,),
    ).fetchone()
    conn.close()

    assert row is not None, "regime_log row must exist"
    assert row[0] == "BTC/USDT"
    assert row[1] == REGIME_TRENDING
    assert abs(row[2] - 1.0) < 1e-9
    assert abs(row[3] - 0.0012) < 1e-9
    assert abs(row[4] - 0.45) < 1e-9


def test_regime_log_table_idempotent(tmp_ledger):
    """Calling log_regime twice on same DB must not error — CREATE IF NOT EXISTS."""
    for _ in range(3):
        log_regime(tmp_ledger, "ETH/USDT", REGIME_RANGING, 0.5)

    conn = sqlite3.connect(tmp_ledger)
    count = conn.execute("SELECT COUNT(*) FROM regime_log").fetchone()[0]
    conn.close()
    assert count == 3


def test_log_regime_rejects_unknown_regime(tmp_ledger):
    with pytest.raises(ValueError, match="Unknown regime"):
        log_regime(tmp_ledger, "BTC/USDT", "moon", 1.0)


# ---------------------------------------------------------------------------
# detect_regime unit tests
# ---------------------------------------------------------------------------


def test_trending_regime_detected():
    """Strong directional movement should produce TRENDING."""
    df = _make_trending_df(n=60, slope=100.0)
    dm = _compute_dm_ratio(df)
    # Verify DM ratio is above threshold before asserting regime
    # (if not, the df builder needs adjustment — this is a diagnostic check)
    assert dm > DM_RATIO_THRESHOLD, (
        f"Test df must produce dm_ratio > {DM_RATIO_THRESHOLD}, got {dm:.3f}. "
        "Adjust slope in _make_trending_df."
    )
    regime, diag = detect_regime(df, atr_percentile=0.3)
    assert regime == REGIME_TRENDING
    assert diag["dm_ratio"] is not None


def test_ranging_regime_detected():
    """Low directional movement + low ATR pct should produce RANGING."""
    df = _make_ranging_df(n=60)
    dm = _compute_dm_ratio(df)
    # Ranging df should have low DM ratio
    assert dm <= DM_RATIO_THRESHOLD, (
        f"Ranging df must produce dm_ratio <= {DM_RATIO_THRESHOLD}, got {dm:.3f}"
    )
    regime, _ = detect_regime(df, atr_percentile=0.3)
    assert regime == REGIME_RANGING


def test_detect_regime_diagnostics_populated(tmp_ledger):
    """Diagnostics dict must have dm_ratio and atr_pct populated."""
    df = _make_trending_df(n=60)
    _, diag = detect_regime(df, atr_percentile=0.5)
    assert diag["dm_ratio"] is not None
    assert not math.isnan(diag["dm_ratio"])
    assert diag["atr_pct"] is not None
    assert not math.isnan(diag["atr_pct"])


def test_detect_regime_volatile_takes_priority_over_trend():
    """Even a perfect trend must be VOLATILE if ATR pct is high."""
    df = _make_trending_df(n=60, slope=500.0)
    dm = _compute_dm_ratio(df)
    assert dm > DM_RATIO_THRESHOLD, "Pre-condition: df must be trending"

    regime, _ = detect_regime(df, atr_percentile=0.95)
    assert regime == REGIME_VOLATILE


# ---------------------------------------------------------------------------
# evaluate_regime_policy integration
# ---------------------------------------------------------------------------


def test_evaluate_regime_policy_returns_multiplier(tmp_ledger):
    """Full pipeline: detect + multiplier + log in one call."""
    df = _make_ranging_df(n=60)
    result = evaluate_regime_policy(df, pair="ETH/USDT", ledger_path=tmp_ledger, atr_percentile=0.4)

    assert "regime" in result
    assert "multiplier" in result
    assert "log_id" in result
    assert result["multiplier"] > 0
    assert result["regime"] in {REGIME_TRENDING, REGIME_RANGING, REGIME_VOLATILE}


def test_evaluate_regime_policy_logs_to_db(tmp_ledger):
    """evaluate_regime_policy must persist a row to regime_log."""
    df = _make_trending_df(n=60)
    result = evaluate_regime_policy(df, pair="BTC/USDT", ledger_path=tmp_ledger, atr_percentile=0.5)

    conn = sqlite3.connect(tmp_ledger)
    row = conn.execute(
        "SELECT regime, multiplier FROM regime_log WHERE log_id = ?",
        (result["log_id"],),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == result["regime"]
    assert abs(row[1] - result["multiplier"]) < 1e-9


def test_evaluate_regime_policy_volatile_when_no_baseline(tmp_ledger):
    """Missing ATR percentile in full pipeline must yield VOLATILE multiplier."""
    df = _make_trending_df(n=60)
    result = evaluate_regime_policy(df, pair="BTC/USDT", ledger_path=tmp_ledger, atr_percentile=None)
    assert result["regime"] == REGIME_VOLATILE
    assert result["multiplier"] == 0.25