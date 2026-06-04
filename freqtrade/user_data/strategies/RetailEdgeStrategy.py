"""
freqtrade/user_data/strategies/RetailEdgeStrategy.py

RetailEdge strategy for Freqtrade.
Sprint 3 S3-1: callbacks wired, Pre-Trade Gate integrated, signal generation pending.

Hard rules (CLAUDE.md):
- enter_long returns 0 until Research Worker produces validated signal (Sprint 3 S3-2).
- confirm_trade_entry calls Pre-Trade Gate. Returns False if gate fails.
- custom_stake_amount applies regime multiplier (placeholder: 1.0 until S3-3).
- custom_stoploss uses ATR trailing if baseline available (placeholder: static until S3-5).
- No autonomous mutation. No challenger model in runtime.

Architecture note — why Pre-Trade Gate is inline here, not imported from sidecar/:
Freqtrade runs the strategy inside its own container/process. The sidecar/ package
is not on the Freqtrade Python path. For MVP, the gate logic is replicated as a
standalone function in this file. The canonical implementation in
sidecar/reconciler/pre_trade_gate.py remains the source of truth for the sidecar.
Both implementations must stay in sync — any change to the gate logic must be
applied to both files.

The gate reads reserved funds from a shared SQLite DB (LEDGER_DB_PATH env var).
If the DB is not reachable (dry-run without sidecar), the gate defaults to PASS
with a warning (fail-open for dry-run only). In micro-live B1, the sidecar must
be running and the DB must be present before Freqtrade starts.
"""

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from freqtrade.strategy import IStrategy, stoploss_from_open

logger = logging.getLogger(__name__)

# Gate constants — must match sidecar/reconciler/pre_trade_gate.py
BALANCE_BUFFER = 1.05
DEFAULT_MIN_NOTIONAL = 10.0   # USDT minimum order size
STATIC_STOPLOSS = -0.05       # -5% static stoploss for Sprint 3

# ATR trailing stop parameters (active when ATR baseline is available)
ATR_TRAIL_MULTIPLIER = 2.0    # stop = entry_price - ATR * multiplier
ATR_COLUMN = "atr_pct"        # must match atr_percentile_service.py

# Regime multiplier defaults (overridden by Adaptive Regime Policy in Sprint 3 S3-3)
REGIME_MULTIPLIER_DEFAULT = 1.0
REGIME_MULTIPLIER_MIN = 0.0
REGIME_MULTIPLIER_MAX = 1.5


class RetailEdgeStrategy(IStrategy):
    """
    RetailEdge strategy.

    Sprint 3 state:
    - Signal generation: disabled (enter_long = 0). Enabled in S3-2.
    - Regime multiplier: static 1.0. Dynamic in S3-3.
    - ATR trailing: static stoploss. Dynamic in S3-5.
    - Pre-Trade Gate: active. Reads from shared ledger if available.
    """

    # -----------------------------------------------------------------------
    # Freqtrade required configuration
    # -----------------------------------------------------------------------

    INTERFACE_VERSION = 3

    # Stoploss: initial static value. custom_stoploss() can tighten this.
    stoploss = STATIC_STOPLOSS

    # Use custom_stoploss — Freqtrade calls this every candle for open trades.
    use_custom_stoploss = True

    # Trailing stop disabled — we manage trailing via custom_stoploss.
    trailing_stop = False

    # Timeframe: 15m as per blueprint (ATR baseline uses 15m candles).
    timeframe = "15m"

    # Minimal ROI: disabled for Sprint 3. Exit driven by stoploss and custom_exit.
    minimal_roi = {"0": 100}  # 10000% = effectively disabled

    # Process only new candles (not tick-by-tick) for dry-run efficiency.
    process_only_new_candles = True

    # -----------------------------------------------------------------------
    # Indicators
    # -----------------------------------------------------------------------

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Compute indicators. Sprint 3: ATR only (needed for custom_stoploss).
        Signal indicators (bucket features, regime detection) added in S3-2/S3-3.
        """
        # ATR (absolute)
        high = dataframe["high"]
        low = dataframe["low"]
        close = dataframe["close"]
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()

        # ATR as percentage of close (matches atr_percentile_service.py)
        dataframe[ATR_COLUMN] = atr / close

        return dataframe

    # -----------------------------------------------------------------------
    # Entry signal — disabled until Research Worker produces candidate
    # -----------------------------------------------------------------------

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Sprint 3 S3-1: enter_long = 0 (no signal).
        Signal logic added in S3-2 after Research Worker produces first candidate.
        """
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""
        return dataframe

    # -----------------------------------------------------------------------
    # Exit signal — disabled until custom exit research lane is active
    # -----------------------------------------------------------------------

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Sprint 3: no signal-based exits. Exit via stoploss and custom_stoploss."""
        dataframe["exit_long"] = 0
        return dataframe

    # -----------------------------------------------------------------------
    # confirm_trade_entry — Pre-Trade Gate integration
    # -----------------------------------------------------------------------

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """
        Final entry gate before order is sent to exchange.

        Calls pre_entry_balance_check with:
        - proposed_stake = amount * rate (quote cost of the trade)
        - wallet_available and reserved_total from shared ledger
        - min_notional from environment config

        Returns True to proceed, False to reject.

        Fail-open in dry-run if ledger is not reachable:
        The sidecar may not be running during initial dry-run tests.
        Log a warning and allow entry so strategy can be tested without full stack.
        In micro-live B1, the sidecar MUST be running (pre_micro_live_B1 gate
        requires reserved_funds_startup_reconcile_pass).
        """
        proposed_stake = amount * rate

        # Read wallet state from shared ledger
        try:
            wallet_available, reserved_total = _read_wallet_state()
        except Exception as exc:
            logger.warning(
                "Pre-Trade Gate: could not read ledger state (%s). "
                "Fail-open for dry-run. Must be resolved before micro-live.",
                exc,
            )
            return True  # fail-open for dry-run only

        # Min notional from env (set by config_compiler)
        min_notional = float(os.getenv("MIN_NOTIONAL_USDT", str(DEFAULT_MIN_NOTIONAL)))

        passed = _inline_pre_entry_balance_check(
            pair=pair,
            proposed_stake=proposed_stake,
            wallet_available=wallet_available,
            reserved_total=reserved_total,
            min_notional=min_notional,
        )

        if not passed:
            logger.info(
                "Pre-Trade Gate BLOCK pair=%s stake=%.4f wallet=%.4f reserved=%.4f",
                pair, proposed_stake, wallet_available, reserved_total,
            )

        return passed

    # -----------------------------------------------------------------------
    # custom_stake_amount — regime multiplier
    # -----------------------------------------------------------------------

    def custom_stake_amount(
        self,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """
        Apply regime multiplier to proposed stake.

        Sprint 3 S3-1: multiplier = 1.0 (static).
        Dynamic regime multiplier from Adaptive Regime Policy added in S3-3.

        Bounds: never return less than min_stake or more than max_stake.
        """
        multiplier = _get_regime_multiplier()
        adjusted = proposed_stake * multiplier

        # Clamp to Freqtrade bounds
        if min_stake is not None:
            adjusted = max(adjusted, min_stake)
        adjusted = min(adjusted, max_stake)

        return adjusted

    # -----------------------------------------------------------------------
    # custom_stoploss — ATR trailing
    # -----------------------------------------------------------------------

    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        """
        ATR-based trailing stoploss.

        Sprint 3 S3-1: returns static STATIC_STOPLOSS until ATR baseline is available.
        Dynamic ATR trailing added in Sprint 3 S3-5 when ATR service is integrated.

        Returns stoploss as a negative fraction of current price
        (e.g. -0.05 = 5% below current price).
        Freqtrade takes the least negative value (tightest stop).
        """
        # Sprint 3: static stoploss
        # In S3-5: read ATR from dataframe, compute trailing distance,
        # return max(static_stoploss, -atr_trail) to only tighten.
        return STATIC_STOPLOSS


# ---------------------------------------------------------------------------
# Inline Pre-Trade Gate (mirrors sidecar/reconciler/pre_trade_gate.py)
# Must stay in sync with canonical implementation.
# ---------------------------------------------------------------------------

def _inline_pre_entry_balance_check(
    pair: str,
    proposed_stake: float,
    wallet_available: float,
    reserved_total: float,
    min_notional: float = DEFAULT_MIN_NOTIONAL,
) -> bool:
    """
    Inline replica of pre_entry_balance_check for use inside Freqtrade process.
    Does not post to Decision Bus (no sidecar access from strategy).
    Guardian will detect the blocked entry via order state, not bus post.
    """
    if proposed_stake <= 0:
        return False

    if min_notional > 0 and proposed_stake < min_notional:
        logger.info("Gate: stake %.4f < min_notional %.4f", proposed_stake, min_notional)
        return False

    projected_available = wallet_available - reserved_total
    required = proposed_stake * BALANCE_BUFFER

    if projected_available < required:
        logger.info(
            "Gate: projected %.4f < required %.4f (stake=%.4f buf=%.2f)",
            projected_available, required, proposed_stake, BALANCE_BUFFER,
        )
        return False

    return True


def _read_wallet_state() -> tuple[float, float]:
    """
    Read wallet_available and reserved_total from shared SQLite ledger.
    Raises on connection failure — caller handles fail-open logic.

    wallet_available: read from Freqtrade wallets table (if accessible)
                      or from ledger balance snapshot.
    reserved_total: sum of reserved_funds table.

    For Sprint 3 dry-run: if ledger has no reserved_funds rows,
    reserved_total = 0.0 (correct for fresh start).
    wallet_available defaults to a safe large value if not found
    (dry-run paper balance is effectively unlimited).
    """
    db_path = os.getenv("LEDGER_DB_PATH", "./ledger/retailedge.db")

    conn = sqlite3.connect(db_path, timeout=2.0)
    try:
        # Reserved total from ledger
        try:
            cur = conn.execute(
                "SELECT COALESCE(SUM(reserved_quote), 0.0) FROM reserved_funds"
            )
            reserved_total = float(cur.fetchone()[0])
        except sqlite3.OperationalError:
            reserved_total = 0.0  # table not yet created

        # Wallet available: use a dry-run sentinel if not tracked
        # In micro-live, this will be populated by Reconciler from exchange API.
        try:
            cur = conn.execute(
                "SELECT available_quote FROM wallet_snapshot ORDER BY snapshot_ts DESC LIMIT 1"
            )
            row = cur.fetchone()
            wallet_available = float(row[0]) if row else 999999.0
        except sqlite3.OperationalError:
            wallet_available = 999999.0  # dry-run: effectively unlimited

    finally:
        conn.close()

    return wallet_available, reserved_total


def _get_regime_multiplier() -> float:
    """
    Read current regime multiplier.
    Sprint 3 S3-1: returns 1.0 (static).
    S3-3: reads from ledger regime_policy table.
    """
    return REGIME_MULTIPLIER_DEFAULT