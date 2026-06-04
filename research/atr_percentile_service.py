"""
research/atr_percentile_service.py

ATR Percentile Baseline Service for RetailEdge.
Single responsibility: compute where the current ATR value sits within
its historical distribution, using a stable expanding-window baseline.

Hard rules (CLAUDE.md / blueprint v1.9 Section 6 Step 6):
- Minimum 5000 candles required. Below this: return ATR_BASELINE_INSUFFICIENT.
- Baseline mode: expanding_window_with_recent_cap.
  Use last min_lookback candles as the historical population.
  Current ATR = last value in the series.
  Percentile = fraction of historical ATR values <= current ATR.
- No real-time quantile from arbitrary latest-dataframe slice.
  The baseline must be stable and reproducible.
- Refresh interval: 6 hours in production (handled by caller — service is stateless).
- ATR column: "atr_pct" (ATR as percentage of price, not absolute value).
  This normalises across price regimes so percentile is meaningful over time.

Why atr_pct and not raw ATR:
Raw ATR in USDT changes as price changes (BTC at 30k vs 60k).
atr_pct = ATR / close normalises this. A percentile computed on atr_pct
is comparable across different price levels in the same series.

Dependencies: pandas, numpy only. No freqtrade, no exchange client.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Blueprint constants
MIN_LOOKBACK_CANDLES = 5000
ATR_COLUMN = "atr_pct"
HIGH_VOL_THRESHOLD = 0.80
SPIKE_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Core computation — pure function, no side effects
# ---------------------------------------------------------------------------

def compute_atr_percentile_baseline(
    df: pd.DataFrame,
    min_lookback: int = MIN_LOOKBACK_CANDLES,
    atr_column: str = ATR_COLUMN,
) -> dict[str, Any]:
    """
    Compute ATR percentile of the current (latest) candle against
    a historical expanding-window baseline.

    Args:
        df:           OHLCV DataFrame with an ATR column.
                      Must contain `atr_column` (default: "atr_pct").
                      Index can be datetime or integer.
        min_lookback: Minimum number of valid (non-NaN) candles required.
                      Default: 5000 (blueprint requirement).
        atr_column:   Column name containing ATR values.
                      Must be pre-computed by caller (freqtrade strategy or research worker).

    Returns:
        On success:
            {
                "status": "OK",
                "atr_percentile": float,   # 0.0 to 1.0
                "current_atr": float,      # latest ATR value
                "baseline_n": int,         # candles used in baseline
                "is_high_vol": bool,       # percentile >= HIGH_VOL_THRESHOLD
                "is_spike": bool,          # percentile >= SPIKE_THRESHOLD
            }
        On insufficient history:
            {
                "status": "ATR_BASELINE_INSUFFICIENT",
                "atr_percentile": None,
                "baseline_n": int,         # actual count (< min_lookback)
                "required_n": int,         # min_lookback
            }
        On error (missing column, all-NaN, etc.):
            {
                "status": "ATR_BASELINE_ERROR",
                "atr_percentile": None,
                "error": str,
            }

    Algorithm:
        1. Validate column exists and is numeric.
        2. Drop NaN values to get valid series.
        3. Check len(valid) >= min_lookback.
        4. Use last min_lookback rows as baseline population (expanding cap).
        5. current_atr = last value in valid series.
        6. percentile = fraction of baseline values <= current_atr.

    Why (series <= current_atr).mean() and not np.percentile():
    We want to know the rank of the current observation within the population.
    np.percentile(series, p) gives the value at rank p — the inverse operation.
    (series <= current_atr).mean() gives the fraction of values below current,
    which is exactly the empirical CDF evaluated at current_atr.
    """
    # --- Validate column
    if atr_column not in df.columns:
        return {
            "status": "ATR_BASELINE_ERROR",
            "atr_percentile": None,
            "error": f"Column '{atr_column}' not found in DataFrame. "
                     f"Available: {list(df.columns)}",
        }

    try:
        series = pd.to_numeric(df[atr_column], errors="coerce")
    except Exception as exc:
        return {
            "status": "ATR_BASELINE_ERROR",
            "atr_percentile": None,
            "error": f"Failed to parse column '{atr_column}' as numeric: {exc}",
        }

    # --- Drop NaN to get valid observations
    valid = series.dropna()
    valid_n = len(valid)

    # --- Minimum lookback gate (blueprint hard rule)
    if valid_n < min_lookback:
        logger.debug(
            "ATR baseline insufficient: %d valid candles, need %d",
            valid_n, min_lookback,
        )
        return {
            "status": "ATR_BASELINE_INSUFFICIENT",
            "atr_percentile": None,
            "baseline_n": valid_n,
            "required_n": min_lookback,
        }

    # --- Expanding window with recent cap
    # Use exactly the last min_lookback valid candles as the baseline population.
    # "Expanding" means we don't use a fixed rolling window —
    # the cap is at min_lookback, preventing unbounded growth in memory/compute.
    baseline = valid.iloc[-min_lookback:]
    current_atr = float(valid.iloc[-1])

    # --- Percentile: empirical CDF at current_atr
    # Edge case: all identical ATR values -> percentile = 1.0 (current equals max)
    percentile = float((baseline <= current_atr).mean())

    return {
        "status": "OK",
        "atr_percentile": percentile,
        "current_atr": current_atr,
        "baseline_n": len(baseline),
        "is_high_vol": percentile >= HIGH_VOL_THRESHOLD,
        "is_spike": percentile >= SPIKE_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# ATR computation helper — for DataFrames that don't have atr_pct yet
# ---------------------------------------------------------------------------

def add_atr_pct(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    output_col: str = ATR_COLUMN,
) -> pd.DataFrame:
    """
    Compute ATR as percentage of close and add as a new column.
    Returns a copy — does not mutate the input DataFrame.

    ATR(n) = Wilder's smoothed average of True Range over n periods.
    atr_pct = ATR / close

    Why atr_pct instead of raw ATR: see module docstring.

    This helper is provided for research worker and test fixtures.
    In production, Freqtrade strategies compute ATR via ta-lib or pandas-ta
    and store it in the dataframe before passing to this service.
    """
    df = df.copy()

    required = [high_col, low_col, close_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)
    prev_close = close.shift(1)

    # True Range = max of: high-low, |high-prev_close|, |low-prev_close|
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder smoothing: EWM with alpha = 1/period
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    # Normalise by close price
    df[output_col] = atr / close
    return df


# ---------------------------------------------------------------------------
# ATRPercentileService — stateful wrapper with refresh logic
# ---------------------------------------------------------------------------

class ATRPercentileService:
    """
    Stateful wrapper that caches the ATR state and refreshes every 6 hours.

    In production: instantiated once, called by auditor via atr_state_fn.
    Example:
        service = ATRPercentileService(df_provider=fetch_ohlcv, refresh_hours=6)
        auditor = StoplossAuditor(..., atr_state_fn=service.get_current_state)

    df_provider: callable returning a fresh OHLCV DataFrame.
                 Called on first access and every refresh_hours thereafter.
    refresh_hours: how often to recompute baseline. Default 6 (blueprint).
    min_lookback: passed to compute_atr_percentile_baseline.
    """

    def __init__(
        self,
        df_provider,
        refresh_hours: float = 6.0,
        min_lookback: int = MIN_LOOKBACK_CANDLES,
        atr_column: str = ATR_COLUMN,
    ) -> None:
        self.df_provider = df_provider
        self.refresh_interval_sec = refresh_hours * 3600
        self.min_lookback = min_lookback
        self.atr_column = atr_column
        self._cached_state: dict[str, Any] | None = None
        self._last_refresh_ts: float = 0.0

    def get_current_state(self) -> dict[str, Any]:
        """
        Return current ATR state. Refreshes if cache is stale.
        Thread-safe for read (single-threaded sidecar). Not locked for write.
        """
        import time
        now = time.monotonic()

        if self._cached_state is None or (now - self._last_refresh_ts) >= self.refresh_interval_sec:
            self._refresh()

        return self._cached_state or {
            "status": "ATR_BASELINE_INSUFFICIENT",
            "atr_percentile": None,
        }

    def _refresh(self) -> None:
        """Fetch fresh OHLCV data and recompute baseline."""
        import time
        try:
            df = self.df_provider()
            state = compute_atr_percentile_baseline(
                df, min_lookback=self.min_lookback, atr_column=self.atr_column
            )
            self._cached_state = state
            self._last_refresh_ts = time.monotonic()
            logger.info(
                "ATR baseline refreshed: status=%s percentile=%s n=%s",
                state.get("status"),
                state.get("atr_percentile"),
                state.get("baseline_n"),
            )
        except Exception as exc:
            logger.error("ATR baseline refresh failed: %s", exc, exc_info=True)
            self._cached_state = {
                "status": "ATR_BASELINE_ERROR",
                "atr_percentile": None,
                "error": str(exc),
            }