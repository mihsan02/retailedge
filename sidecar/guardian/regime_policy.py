"""
sidecar/guardian/regime_policy.py — Sprint 3 S3-3

Adaptive Regime Policy: detect market regime from OHLCV and return position multiplier.

Regime detection logic:
  volatile  : ATR percentile >= 0.80 (checked first — safety gate)
  trending  : abs(DM+ - DM-) / ATR > DM_RATIO_THRESHOLD
  ranging   : neither volatile nor trending

Position multipliers:
  trending  : 1.0
  ranging   : 0.5
  volatile  : 0.25

DM+/DM- computed per-candle (raw, not Wilder-smoothed).
Ratio averaged over lookback window. This avoids Wilder's 14-period
lag which makes the signal too slow for regime detection on 15m bars.

ATR percentile delegated to research/atr_percentile_service.py baseline.
If baseline is unavailable, regime defaults to VOLATILE (conservative).
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGIME_TRENDING = "trending"
REGIME_RANGING = "ranging"
REGIME_VOLATILE = "volatile"

VALID_REGIMES = frozenset({REGIME_TRENDING, REGIME_RANGING, REGIME_VOLATILE})

POSITION_MULTIPLIERS: dict[str, float] = {
    REGIME_TRENDING: 1.0,
    REGIME_RANGING: 0.5,
    REGIME_VOLATILE: 0.25,
}

# ATR percentile threshold — matches v1.9 blueprint Section 6 Step 6
VOLATILE_ATR_PERCENTILE = 0.80

# DM ratio threshold: abs(DM+ - DM-) / ATR > this => trending
# Calibrated so that pure GBM (no trend) sits comfortably below threshold.
# Wilder uses 25 as the ADX threshold for trending; DM ratio ~0.30 is equivalent
# in normalized form. References: Wilder (1978), Freqtrade TA-Lib ADX docs.
DM_RATIO_THRESHOLD = 0.30

# Lookback window for averaging DM ratio (candles)
DM_LOOKBACK = 14

# Minimum candles required for a valid regime read
MIN_CANDLES = DM_LOOKBACK + 2


# ---------------------------------------------------------------------------
# DDL — idempotent, appended to schema.sql
# ---------------------------------------------------------------------------

REGIME_LOG_DDL = """
CREATE TABLE IF NOT EXISTS regime_log (
    log_id       TEXT PRIMARY KEY,
    pair         TEXT NOT NULL,
    regime       TEXT NOT NULL,
    multiplier   REAL NOT NULL,
    atr_pct      REAL,
    dm_ratio     REAL,
    created_ts   TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# 1. Feature helpers
# ---------------------------------------------------------------------------


def _compute_atr(df: pd.DataFrame, period: int = DM_LOOKBACK) -> pd.Series:
    """True Range ATR (simple rolling mean, not Wilder-smoothed)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _compute_dm_ratio(df: pd.DataFrame, lookback: int = DM_LOOKBACK) -> float:
    """
    Per-candle DM+ and DM-:
      DM+ = max(high - prev_high, 0) if (high - prev_high) > (prev_low - low) else 0
      DM- = max(prev_low - low, 0)   if (prev_low - low) > (high - prev_high) else 0

    Ratio = rolling_mean(abs(DM+ - DM-)) / rolling_mean(ATR)

    Returns the most recent ratio value (scalar float).
    Returns 0.0 if insufficient data or ATR is zero.
    """
    if len(df) < lookback + 2:
        return 0.0

    df = df.copy()
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)

    up_move = df["high"] - prev_high
    dn_move = prev_low - df["low"]

    dm_plus = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)

    diff = pd.Series(np.abs(dm_plus - dm_minus), index=df.index)
    atr = _compute_atr(df, period=lookback)

    # Use last `lookback` rows for ratio
    tail_diff = diff.iloc[-lookback:]
    tail_atr = atr.iloc[-lookback:]

    mean_atr = tail_atr.mean()
    if mean_atr == 0 or np.isnan(mean_atr):
        return 0.0

    return float(tail_diff.mean() / mean_atr)


def _compute_atr_pct(df: pd.DataFrame, lookback: int = DM_LOOKBACK) -> float:
    """Most recent ATR / close (normalized volatility scalar)."""
    if len(df) < lookback + 2:
        return float("nan")
    atr = _compute_atr(df, period=lookback)
    last_atr = atr.iloc[-1]
    last_close = df["close"].iloc[-1]
    if last_close == 0 or np.isnan(last_atr):
        return float("nan")
    return float(last_atr / last_close)


# ---------------------------------------------------------------------------
# 2. Regime detection
# ---------------------------------------------------------------------------


def detect_regime(
    df: pd.DataFrame,
    atr_percentile: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Detect regime from OHLCV dataframe.

    Args:
        df: OHLCV dataframe with columns [open, high, low, close, volume].
            Must have at least MIN_CANDLES rows.
        atr_percentile: Pre-computed ATR percentile from ATR Percentile Service.
            If None or nan, regime defaults to VOLATILE (conservative fallback).

    Returns:
        (regime_str, diagnostics_dict)

    Priority order (matches v1.9 blueprint — safety gate first):
        1. VOLATILE if atr_percentile unavailable (baseline missing)
        2. VOLATILE if atr_percentile >= VOLATILE_ATR_PERCENTILE
        3. TRENDING if dm_ratio > DM_RATIO_THRESHOLD
        4. RANGING otherwise
    """
    diagnostics: dict[str, Any] = {
        "atr_percentile": atr_percentile,
        "dm_ratio": None,
        "atr_pct": None,
        "min_candles_met": len(df) >= MIN_CANDLES,
    }

    # Gate: insufficient data -> conservative
    if len(df) < MIN_CANDLES:
        logger.warning(
            "detect_regime: insufficient candles (%d < %d), defaulting to VOLATILE",
            len(df),
            MIN_CANDLES,
        )
        return REGIME_VOLATILE, diagnostics

    # Compute diagnostics
    dm_ratio = _compute_dm_ratio(df)
    atr_pct = _compute_atr_pct(df)
    diagnostics["dm_ratio"] = dm_ratio
    diagnostics["atr_pct"] = atr_pct

    # 1. ATR baseline missing -> VOLATILE
    if atr_percentile is None or (isinstance(atr_percentile, float) and np.isnan(atr_percentile)):
        logger.warning("detect_regime: ATR percentile unavailable, defaulting to VOLATILE")
        return REGIME_VOLATILE, diagnostics

    # 2. High volatility -> VOLATILE
    if atr_percentile >= VOLATILE_ATR_PERCENTILE:
        return REGIME_VOLATILE, diagnostics

    # 3. Directional movement -> TRENDING
    if dm_ratio > DM_RATIO_THRESHOLD:
        return REGIME_TRENDING, diagnostics

    # 4. Default -> RANGING
    return REGIME_RANGING, diagnostics


# ---------------------------------------------------------------------------
# 3. Position multiplier
# ---------------------------------------------------------------------------


def get_position_multiplier(regime: str) -> float:
    """
    Return position size multiplier for regime.
    Raises ValueError for unknown regime strings.
    """
    if regime not in VALID_REGIMES:
        raise ValueError(f"Unknown regime: {regime!r}. Valid: {VALID_REGIMES}")
    return POSITION_MULTIPLIERS[regime]


# ---------------------------------------------------------------------------
# 4. Ledger persistence
# ---------------------------------------------------------------------------


def _ensure_regime_log_table(conn: sqlite3.Connection) -> None:
    """Idempotent DDL. Safe to call on existing DB."""
    conn.executescript(REGIME_LOG_DDL)
    conn.commit()


def log_regime(
    ledger_path: str,
    pair: str,
    regime: str,
    multiplier: float,
    atr_pct: float | None = None,
    dm_ratio: float | None = None,
    ts: str | None = None,
) -> str:
    """
    Write regime detection result to regime_log table.
    Returns log_id.
    """
    if regime not in VALID_REGIMES:
        raise ValueError(f"Unknown regime: {regime!r}")

    log_id = f"regime_{uuid.uuid4().hex[:12]}"
    created_ts = ts or datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(ledger_path)
    try:
        _ensure_regime_log_table(conn)
        conn.execute(
            """
            INSERT INTO regime_log (log_id, pair, regime, multiplier, atr_pct, dm_ratio, created_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (log_id, pair, regime, multiplier, atr_pct, dm_ratio, created_ts),
        )
        conn.commit()
    finally:
        conn.close()

    return log_id


# ---------------------------------------------------------------------------
# 5. Convenience: full policy evaluation
# ---------------------------------------------------------------------------


def evaluate_regime_policy(
    df: pd.DataFrame,
    pair: str,
    ledger_path: str,
    atr_percentile: float | None = None,
) -> dict[str, Any]:
    """
    Detect regime, get multiplier, log to ledger.
    Returns dict suitable for RetailEdgeStrategy._get_regime_multiplier().

    Usage in strategy:
        from sidecar.guardian.regime_policy import evaluate_regime_policy
        result = evaluate_regime_policy(dataframe, pair, ledger_path, atr_percentile)
        return result["multiplier"]
    """
    regime, diagnostics = detect_regime(df, atr_percentile=atr_percentile)
    multiplier = get_position_multiplier(regime)

    log_id = log_regime(
        ledger_path=ledger_path,
        pair=pair,
        regime=regime,
        multiplier=multiplier,
        atr_pct=diagnostics.get("atr_pct"),
        dm_ratio=diagnostics.get("dm_ratio"),
    )

    logger.info(
        "Regime: pair=%s regime=%s multiplier=%.2f dm_ratio=%.3f atr_pct=%.5f log_id=%s",
        pair,
        regime,
        multiplier,
        diagnostics.get("dm_ratio") or 0.0,
        diagnostics.get("atr_pct") or 0.0,
        log_id,
    )

    return {
        "regime": regime,
        "multiplier": multiplier,
        "log_id": log_id,
        "diagnostics": diagnostics,
    }