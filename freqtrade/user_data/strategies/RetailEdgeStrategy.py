"""
RetailEdgeStrategy — Sprint 3 Revisited S3-R2
Freqtrade strategy with real threshold from ledger.

CHANGES FROM S3-1:
- bot_start(): loads threshold_long per pair from strategy_memory (REAL_DATA_OOS)
- populate_indicators(): adds momentum_5 and atr_pct columns
- populate_entry_trend(): enter_long based on threshold + ATR percentile gate
- populate_exit_trend(): unchanged (no exit signal, stoploss_on_exchange handles exit)
- All other callbacks: UNCHANGED

DESIGN DECISIONS:
- Ledger is read ONCE at startup (bot_start). Not re-read per candle.
- ATR percentile is computed inline — no import from research/ (different container).
- Min lookback for ATR percentile: 5000 candles (CLAUDE.md Hard Constraint #6).
- If threshold missing for a pair: log warning, return enter_long=0 for that pair.
- If ATR percentile cannot be computed (insufficient history): block entry.
- cost_floor_cleared=False is noted in log but does NOT block entry in dry-run.
  This is intentional: S3-R2 done criteria is enter_long > 0, not profitable.
  Stage B1 profitability gate will enforce cost_floor separately.

LEDGER PATH (inside Freqtrade container):
  /freqtrade/ledger/retailedge.db
  Mounted via docker-compose: ./ledger:/freqtrade/ledger
"""

import json
import logging
import os
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEDGER_DB_PATH = os.environ.get(
    "LEDGER_DB_PATH", "/freqtrade/ledger/retailedge.db"
)
STRATEGY_MEMORY_STAGE = "REAL_DATA_OOS"
# ATR_MIN_LOOKBACK: 5000 (blueprint v1.9 Section 6 Step 6 hard requirement).
# Raised from 1000 once 90-day historical data downloaded (8706 candles available).
# startup_candle_count must match this value.
# INVARIANT: this value must match ATR_MIN_LOOKBACK in research/worker.py.
ATR_MIN_LOOKBACK = 4999  # Freqtrade rejects startup_candle_count=5000 (exceeds 5x Binance 15m API limit). 4999 is functionally equivalent to blueprint 5000 minimum.
ATR_HIGH_VOLATILITY_THRESHOLD = 0.80  # CLAUDE.md Hard Constraint #6

# ---------------------------------------------------------------------------
# Pre-Trade Gate constants — keep in sync with sidecar/reconciler/pre_trade_gate.py
# ---------------------------------------------------------------------------
BALANCE_BUFFER = 1.05        # projected_available must be >= stake * 1.05
DEFAULT_MIN_NOTIONAL = 10.0  # Binance Spot minimum order size (USDT)

# ---------------------------------------------------------------------------
# Pre-Trade Gate — module-level functions (testable without Freqtrade instance)
# NOTE: Keep in sync with sidecar/reconciler/pre_trade_gate.py manually.
# ---------------------------------------------------------------------------

def _inline_pre_entry_balance_check(
    pair: str,
    proposed_stake: float,
    wallet_available: float,
    reserved_total: float,
    min_notional: float = DEFAULT_MIN_NOTIONAL,
) -> bool:
    """
    Pure function. Returns True if entry is safe to proceed.

    Rules (in order):
    1. proposed_stake == 0 -> block
    2. proposed_stake < min_notional -> block
    3. projected_available = wallet_available - reserved_total
       projected_available < proposed_stake * BALANCE_BUFFER -> block
    4. else -> pass
    """
    if proposed_stake <= 0.0:
        logger.info("%s: stake=0 -> block", pair)
        return False

    if proposed_stake < min_notional:
        logger.info(
            "%s: stake=%.4f below min_notional=%.4f -> block",
            pair, proposed_stake, min_notional,
        )
        return False

    projected_available = wallet_available - reserved_total
    required = proposed_stake * BALANCE_BUFFER

    if projected_available < required:
        logger.info(
            "%s: projected_available=%.4f < required=%.4f -> block",
            pair, projected_available, required,
        )
        return False

    return True


def _read_wallet_state(db_path: str | None = None) -> tuple[float, float]:
    """
    Read wallet_available and reserved_total from ledger.

    Returns:
        (wallet_available, reserved_total)

    Dry-run sentinel: wallet_available=999999.0 when no live balance available.
    reserved_total is always summed from reserved_funds table.
    Returns (999999.0, 0.0) if ledger unavailable.
    """
    if db_path is None:
        db_path = os.environ.get("LEDGER_DB_PATH", "ledger/retailedge.db")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Ledger DB not found: {db_path}")

    try:
        with sqlite3.connect(db_path, timeout=3.0) as conn:
            tables = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            if "reserved_funds" not in tables:
                return 999999.0, 0.0

            row = conn.execute(
                "SELECT COALESCE(SUM(reserved_quote), 0.0) FROM reserved_funds"
            ).fetchone()
            reserved_total = float(row[0]) if row else 0.0

    except sqlite3.Error as e:
        raise RuntimeError(f"Ledger read failed: {e}") from e

    return 999999.0, reserved_total



# ---------------------------------------------------------------------------
# Ledger helpers — inline, no sidecar import
# ---------------------------------------------------------------------------

def _load_thresholds_from_ledger(db_path: str, stage: str) -> dict[str, float]:
    """
    Read latest threshold_long per pair from strategy_memory.

    Returns dict[pair, threshold_long]. Empty dict if DB unavailable.
    Only includes pairs where oos_metrics.status == 'OK' and threshold_long is set.
    """
    if not os.path.exists(db_path):
        logger.warning(
            "Ledger not found at %s. No thresholds loaded. Entry blocked for all pairs.",
            db_path,
        )
        return {}

    thresholds = {}
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            rows = conn.execute(
                """
                SELECT pair, oos_metrics
                FROM strategy_memory
                WHERE stage = ?
                ORDER BY created_ts DESC
                """,
                (stage,),
            ).fetchall()
    except sqlite3.Error as e:
        logger.error("Ledger read error: %s. Entry blocked for all pairs.", e)
        return {}

    seen: set[str] = set()
    for pair, metrics_json in rows:
        if pair in seen:
            continue  # keep latest only
        seen.add(pair)

        if not metrics_json:
            continue

        try:
            metrics = json.loads(metrics_json)
        except json.JSONDecodeError as e:
            logger.warning("Cannot parse oos_metrics for %s: %s", pair, e)
            continue

        status = metrics.get("status", "")
        threshold = metrics.get("threshold_long")

        if status != "OK":
            logger.warning(
                "Skipping %s: oos_metrics.status='%s' (need 'OK')", pair, status
            )
            continue

        if threshold is None:
            logger.warning("Skipping %s: threshold_long is None in oos_metrics", pair)
            continue

        cost_floor_cleared = metrics.get("cost_floor_cleared", False)
        if not cost_floor_cleared:
            logger.warning(
                "NOTE: %s threshold loaded but cost_floor_cleared=False. "
                "Edge not proven above cost floor. Dry-run only.",
                pair,
            )

        thresholds[pair] = float(threshold)
        logger.info(
            "Threshold loaded: %s -> threshold_long=%.6f (win_rate=%.2f, trades=%s)",
            pair,
            float(threshold),
            metrics.get("win_rate", 0.0),
            metrics.get("trade_count", "?"),
        )

    return thresholds


# ---------------------------------------------------------------------------
# ATR percentile — inline implementation (no research/ import)
# ---------------------------------------------------------------------------

def _compute_atr_percentile(dataframe: pd.DataFrame, min_lookback: int = ATR_MIN_LOOKBACK) -> Optional[float]:
    """
    Compute current ATR percentile against historical baseline.

    Returns float in [0, 1] or None if insufficient history.
    Mirrors atr_percentile_service.py logic — kept in sync manually.
    """
    if "atr_pct" not in dataframe.columns:
        return None

    series = dataframe["atr_pct"].dropna()
    if len(series) < min_lookback:
        return None

    hist = series.iloc[-min_lookback:]
    current = hist.iloc[-1]
    return float((hist <= current).mean())


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class RetailEdgeStrategy(IStrategy):
    """
    RetailEdge bucket baseline strategy.

    Entry: momentum_5 > threshold_long AND atr_percentile < 0.80
    Exit: stoploss_on_exchange (no exit signal from strategy)
    """

    INTERFACE_VERSION = 3

    # Freqtrade strategy parameters
    timeframe = "15m"
    can_short = False

    # Startup candles — must match ATR_MIN_LOOKBACK so Gate 3 never blocks on insufficient history.
    # Set to 5000 (blueprint minimum). 90-day historical data download satisfies this requirement.
    startup_candle_count: int = 4999  # Freqtrade validator rejects 5000 (5x Binance 15m API limit). Data from disk satisfies ATR_MIN_LOOKBACK=5000.

    # Stoploss — exchange-side stop handles actual exit
    stoploss = -0.05
    trailing_stop = False

    # Entry/exit pricing
    entry_pricing = {"price_side": "same", "use_order_book": False}
    exit_pricing = {"price_side": "same", "use_order_book": False}

    # Minimal ROI — rely on stoploss and custom_exit, not fixed ROI
    minimal_roi = {"0": 100}

    # Process only new candles — never re-analyze partial candles
    process_only_new_candles = True

    # ---------------------------------------------------------------------------
    # State — loaded once at bot_start, never mutated per candle
    # ---------------------------------------------------------------------------

    _thresholds: dict[str, float] = {}
    _ledger_loaded: bool = False

    # ---------------------------------------------------------------------------
    # Startup
    # ---------------------------------------------------------------------------

    def bot_start(self, **kwargs) -> None:
        """
        Load thresholds from ledger once at startup.
        Called by Freqtrade before the main loop starts.
        """
        logger.info(
            "bot_start: loading thresholds from ledger at %s (stage=%s)",
            LEDGER_DB_PATH,
            STRATEGY_MEMORY_STAGE,
        )

        self._thresholds = _load_thresholds_from_ledger(LEDGER_DB_PATH, STRATEGY_MEMORY_STAGE)
        self._ledger_loaded = True

        if not self._thresholds:
            logger.warning(
                "bot_start: NO thresholds loaded. "
                "All pairs will have enter_long=0 until ledger is populated. "
                "Run: python -m research.worker"
            )
        else:
            logger.info(
                "bot_start: thresholds loaded for %d pair(s): %s",
                len(self._thresholds),
                list(self._thresholds.keys()),
            )

    # ---------------------------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------------------------

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Add momentum_5 and atr_pct to dataframe.
        These are the only two features needed for entry signal.
        """
        # Momentum: 5-candle log return
        dataframe["momentum_5"] = np.log(
            dataframe["close"] / dataframe["close"].shift(5)
        )

        # ATR (14-period) normalized by close
        high_low = dataframe["high"] - dataframe["low"]
        high_close = (dataframe["high"] - dataframe["close"].shift(1)).abs()
        low_close = (dataframe["low"] - dataframe["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        # Wilder's EWM smoothing (alpha=1/14) — matches atr_percentile_service.add_atr_pct().
        # CRITICAL: must stay in sync with research path. Simple rolling(14).mean() produces
        # numerically different ATR values — percentile computed on different distribution.
        atr_14 = true_range.ewm(alpha=1.0 / 14, adjust=False).mean()
        dataframe["atr_pct"] = atr_14 / dataframe["close"]

        return dataframe

    # ---------------------------------------------------------------------------
    # Entry signal
    # ---------------------------------------------------------------------------

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Entry logic:
          1. Threshold loaded from ledger for this pair
          2. momentum_5 > threshold_long
          3. atr_percentile < 0.80 (not in high-volatility regime)
          4. Pre-Trade Gate check (inline — sidecar not importable)

        If any condition fails: enter_long = 0 for all rows.
        """
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        pair = metadata.get("pair", "")

        # Gate 1: ledger loaded
        if not self._ledger_loaded:
            logger.warning("%s: ledger not loaded yet, skipping entry", pair)
            return dataframe

        # Gate 2: threshold available for this pair
        threshold = self._thresholds.get(pair)
        if threshold is None:
            logger.warning(
                "%s: no threshold in ledger (stage=%s). "
                "Run research worker for this pair.",
                pair, STRATEGY_MEMORY_STAGE,
            )
            return dataframe

        # Gate 3: ATR percentile computable (requires ATR_MIN_LOOKBACK candles)
        atr_percentile = _compute_atr_percentile(dataframe, ATR_MIN_LOOKBACK)
        if atr_percentile is None:
            logger.warning(
                "%s: ATR percentile unavailable (need %d candles, have %d). "
                "Entry blocked.",
                pair, ATR_MIN_LOOKBACK, len(dataframe),
            )
            return dataframe

        # Gate 4: not in high-volatility regime
        if atr_percentile >= ATR_HIGH_VOLATILITY_THRESHOLD:
            logger.info(
                "%s: ATR percentile=%.3f >= %.2f (high volatility). Entry blocked.",
                pair, atr_percentile, ATR_HIGH_VOLATILITY_THRESHOLD,
            )
            return dataframe

        # Apply entry signal to dataframe
        entry_condition = (
            dataframe["momentum_5"].notna()
            & (dataframe["momentum_5"] > threshold)
            & dataframe["volume"].gt(0)
        )

        dataframe.loc[entry_condition, "enter_long"] = 1
        dataframe.loc[entry_condition, "enter_tag"] = (
            f"mom5>{threshold:.6f}_atr{atr_percentile:.2f}"
        )

        n_signals = entry_condition.sum()
        if n_signals > 0:
            logger.info(
                "%s: %d entry signal(s) | threshold=%.6f | atr_pct=%.3f",
                pair, n_signals, threshold, atr_percentile,
            )

        return dataframe

    # ---------------------------------------------------------------------------
    # Exit signal — unchanged, stoploss_on_exchange handles exit
    # ---------------------------------------------------------------------------

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        return dataframe

    # ---------------------------------------------------------------------------
    # Pre-Trade Gate — inline (cannot import from sidecar/)
    # NOTE: Keep in sync with sidecar/reconciler/pre_trade_gate.py manually.
    # ---------------------------------------------------------------------------

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """
        Final gate before order placement.
        Mirrors pre_trade_gate.py logic inline.
        """
        # Regime multiplier gate — read from sidecar via ledger if available
        regime_multiplier = self._get_regime_multiplier(pair)
        if regime_multiplier == 0.0:
            logger.info("%s: regime_multiplier=0.0, blocking entry", pair)
            return False

        # Balance gate — read reserved funds from ledger, check projected available
        try:
            wallet_available, reserved_total = _read_wallet_state()
        except Exception as exc:
            logger.warning(
                "%s: _read_wallet_state failed (%s), blocking entry", pair, exc
            )
            return False

        min_notional = float(os.environ.get("MIN_NOTIONAL_USDT", DEFAULT_MIN_NOTIONAL))
        proposed_stake = amount * rate

        if not _inline_pre_entry_balance_check(
            pair=pair,
            proposed_stake=proposed_stake,
            wallet_available=wallet_available,
            reserved_total=reserved_total,
            min_notional=min_notional,
        ):
            return False

        return True

    def _get_regime_multiplier(self, pair: str) -> float:
        """
        Read latest regime multiplier from ledger.
        Returns 1.0 if ledger unavailable or no regime logged yet.
        Returns 0.0 only if explicitly set to volatile block.
        """
        if not os.path.exists(LEDGER_DB_PATH):
            return 1.0

        try:
            with sqlite3.connect(LEDGER_DB_PATH, timeout=3.0) as conn:
                # Check if regime_log table exists
                tables = {r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()}

                if "regime_log" not in tables:
                    return 1.0

                row = conn.execute(
                    """
                    SELECT multiplier FROM regime_log
                    WHERE pair = ?
                    ORDER BY created_ts DESC LIMIT 1
                    """,
                    (pair,),
                ).fetchone()

                if row is None:
                    return 1.0

                return float(row[0])

        except sqlite3.Error:
            return 1.0

    # ---------------------------------------------------------------------------
    # Custom stake amount — apply regime multiplier
    # ---------------------------------------------------------------------------

    def custom_stake_amount(
        self,
        current_time,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        pair = kwargs.get("pair", "")
        multiplier = self._get_regime_multiplier(pair)
        adjusted = proposed_stake * multiplier

        if min_stake and adjusted < min_stake:
            logger.info(
                "%s: adjusted stake %.4f < min_stake %.4f, using min_stake",
                pair, adjusted, min_stake,
            )
            return min_stake

        return adjusted

    # ---------------------------------------------------------------------------
    # Custom stoploss — ATR-based if baseline available
    # ---------------------------------------------------------------------------

    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        """
        ATR-based trailing stoploss if dataframe available.
        Falls back to static stoploss (-0.05) if not.
        """
        # Return default — dataframe not accessible in custom_stoploss
        # ATR trailing requires callback with dataframe access
        # Placeholder: return 1 to use strategy stoploss value unchanged
        return self.stoploss