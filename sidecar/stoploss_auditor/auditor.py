"""
sidecar/stoploss_auditor/auditor.py

Stoploss Auditor for RetailEdge.
Single responsibility: periodically verify that every open trade has a live
stoploss order on the exchange, and post STOP_UNCONFIRMED if one is missing.

Hard rules (CLAUDE.md / blueprint v1.9 Section 6 Step 6):
- Audit interval: 60s if ATR percentile >= 0.80 OR baseline insufficient with open trades.
- Audit interval: 300s otherwise (normal conditions).
- Recent stop incidents always force 60s interval regardless of ATR state.
- If stop order not found for a trade: post STOP_UNCONFIRMED to Decision Bus.
- ATR state is INJECTABLE — auditor does not import atr_percentile_service directly.
  This keeps the auditor testable without a 5000-candle dataset.

atr_state dict contract:
    {"status": "OK", "atr_percentile": float}        # healthy baseline
    {"status": "ATR_BASELINE_INSUFFICIENT", "atr_percentile": None}  # insufficient history

Design:
- StoplossAuditor.run_once() does one audit pass: fetch open trades, check each stop.
- run_loop() wraps run_once() with the dynamic interval from next_stoploss_audit_interval().
- exchange_client must support fetch_open_orders() returning list of order dicts.
- Decision Bus is injectable for test isolation.
"""

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Normal audit interval — low volatility, no incidents
INTERVAL_NORMAL_SEC = 300

# High-risk audit interval — high ATR, insufficient baseline, or recent incidents
INTERVAL_HIGH_RISK_SEC = 60

# ATR percentile threshold above which high-risk interval applies
ATR_HIGH_VOL_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Interval policy — pure function, easily testable
# ---------------------------------------------------------------------------

def next_stoploss_audit_interval(
    open_trades: list[dict[str, Any]],
    atr_state: dict[str, Any],
    recent_stop_incidents: int = 0,
) -> int:
    """
    Compute the next stoploss audit interval in seconds.

    Rules (evaluated in priority order):
    1. No open trades → 300s (nothing to audit).
    2. recent_stop_incidents > 0 → 60s (incident response mode).
    3. ATR baseline insufficient AND open trades exist → 60s (conservative).
    4. ATR percentile >= 0.80 → 60s (high volatility).
    5. Otherwise → 300s (normal).

    Args:
        open_trades:           List of open trade dicts from Freqtrade.
        atr_state:             Dict with keys "status" and "atr_percentile".
                               status="OK" means baseline is valid.
                               status="ATR_BASELINE_INSUFFICIENT" means < 5000 candles.
        recent_stop_incidents: Count of STOP_UNCONFIRMED events in last audit cycle.
                               Guardian passes this from Decision Bus query.

    Returns:
        60 or 300 (seconds).

    Why a pure function (not a method):
    The interval logic must be testable without an exchange client or DB.
    test_stoploss_interval_uses_atr_baseline calls this directly.
    """
    # Rule 1: nothing to audit
    if not open_trades:
        return INTERVAL_NORMAL_SEC

    # Rule 2: recent incident — always aggressive interval
    if recent_stop_incidents > 0:
        return INTERVAL_HIGH_RISK_SEC

    # Rule 3: ATR baseline insufficient with open trades — conservative
    atr_status = atr_state.get("status", "ATR_BASELINE_INSUFFICIENT")
    if atr_status != "OK":
        return INTERVAL_HIGH_RISK_SEC

    # Rule 4: high volatility regime
    atr_percentile = atr_state.get("atr_percentile")
    if atr_percentile is not None and atr_percentile >= ATR_HIGH_VOL_THRESHOLD:
        return INTERVAL_HIGH_RISK_SEC

    # Rule 5: normal
    return INTERVAL_NORMAL_SEC


# ---------------------------------------------------------------------------
# StoplossAuditor
# ---------------------------------------------------------------------------

class StoplossAuditor:
    """
    Periodically verifies exchange-side stoploss orders for all open trades.

    exchange_client: object with fetch_open_orders() -> list[dict].
    ft_client: object with get_status() -> dict (open trades from Freqtrade).
    bus: DecisionBus instance for posting STOP_UNCONFIRMED.
    atr_state_fn: callable returning current atr_state dict.
                  Injectable so auditor doesn't depend on ATR service directly.
                  In production: lambda: atr_service.get_current_state()
                  In tests: lambda: {"status": "OK", "atr_percentile": 0.5}

    Why separate exchange_client and ft_client:
    - ft_client.get_status() returns Freqtrade's view of open trades (trade_id, pair, stoploss).
    - exchange_client.fetch_open_orders() returns the exchange's actual order book.
    - The gap between these two is what we're auditing.
    """

    def __init__(
        self,
        exchange_client: Any,
        ft_client: Any,
        bus: Any,
        atr_state_fn: Optional[Any] = None,
    ) -> None:
        self.exchange_client = exchange_client
        self.ft = ft_client
        self.bus = bus
        # Default atr_state_fn: assume insufficient baseline (conservative)
        self.atr_state_fn = atr_state_fn or (
            lambda: {"status": "ATR_BASELINE_INSUFFICIENT", "atr_percentile": None}
        )
        self._incident_count = 0  # reset each run_loop cycle

    def run_loop(self) -> None:
        """
        Production entry point. Polls with dynamic interval.
        Interval is recomputed after each audit based on current ATR state.
        """
        logger.info("StoplossAuditor starting")
        while True:
            try:
                result = self.run_once()
                open_trades = result.get("open_trades", [])
                atr_state = self.atr_state_fn()
                interval = next_stoploss_audit_interval(
                    open_trades, atr_state, result.get("incidents", 0)
                )
                logger.debug("Next audit in %ss (atr=%s)", interval, atr_state)
            except Exception as exc:
                logger.error("StoplossAuditor loop error: %s", exc, exc_info=True)
                interval = INTERVAL_HIGH_RISK_SEC  # conservative on error

            time.sleep(interval)

    def run_once(self) -> dict[str, Any]:
        """
        One audit pass:
        1. Fetch open trades from Freqtrade.
        2. Fetch open orders from exchange.
        3. For each open trade: check if a stoploss order exists on exchange.
        4. If missing: post STOP_UNCONFIRMED to Decision Bus.

        Returns summary dict for test inspection.
        """
        summary = {
            "open_trades": [],
            "verified": 0,
            "missing": 0,
            "incidents": 0,
            "errors": [],
        }

        # Step 1: open trades from Freqtrade
        try:
            status_resp = self.ft.get_status()
            # get_status returns list of trade dicts or {"status": [...]}
            if isinstance(status_resp, list):
                open_trades = status_resp
            elif isinstance(status_resp, dict):
                open_trades = status_resp.get("status", []) or []
            else:
                open_trades = []
        except Exception as exc:
            logger.warning("Failed to fetch Freqtrade status: %s", exc)
            summary["errors"].append(f"ft_status:{exc}")
            open_trades = []

        summary["open_trades"] = open_trades

        if not open_trades:
            return summary

        # Step 2: open orders from exchange (all order types)
        try:
            exchange_orders = self.exchange_client.fetch_open_orders()
        except Exception as exc:
            logger.warning("Failed to fetch exchange open orders: %s", exc)
            summary["errors"].append(f"exchange_orders:{exc}")
            # Cannot audit without exchange data — conservative: treat all as unconfirmed
            for trade in open_trades:
                self._post_stop_unconfirmed(trade, reason="exchange_fetch_failed")
                summary["incidents"] += 1
            summary["missing"] = len(open_trades)
            return summary

        # Step 3: build set of stoploss order identifiers from exchange
        # Exchange stoploss orders are identified by:
        # - type = "stop_limit" or "stop" (exchange-dependent)
        # - clientOrderId or a reference back to the trade
        # Freqtrade attaches trade_id or pair to stoploss orders via clientOrderId.
        exchange_stop_pairs = _extract_stop_pairs(exchange_orders)

        # Step 4: verify each open trade has a stop on exchange
        for trade in open_trades:
            trade_id = str(trade.get("trade_id") or trade.get("id") or "")
            pair = str(trade.get("pair") or trade.get("symbol") or "")

            if _has_stop_for_trade(trade_id, pair, exchange_stop_pairs, exchange_orders):
                summary["verified"] += 1
                logger.debug("Stop verified for trade %s pair=%s", trade_id, pair)
            else:
                summary["missing"] += 1
                summary["incidents"] += 1
                logger.warning("Stop MISSING for trade %s pair=%s", trade_id, pair)
                self._post_stop_unconfirmed(trade, reason="stop_not_found_on_exchange")

        return summary

    def _post_stop_unconfirmed(self, trade: dict[str, Any], reason: str) -> None:
        """Post STOP_UNCONFIRMED to Decision Bus for a trade missing its stop."""
        trade_id = str(trade.get("trade_id") or trade.get("id") or "")
        pair = str(trade.get("pair") or trade.get("symbol") or "")

        # Deduplicate: don't flood bus if stop is persistently missing
        if self.bus.has_pending_of_type("STOP_UNCONFIRMED"):
            logger.debug("STOP_UNCONFIRMED already pending, skipping duplicate post")
            return

        self.bus.post(
            "STOP_UNCONFIRMED",
            reason=reason,
            severity="HIGH",
            trade_id=trade_id,
            pair=pair,
        )


# ---------------------------------------------------------------------------
# Exchange order analysis helpers
# ---------------------------------------------------------------------------

def _extract_stop_pairs(exchange_orders: list[dict[str, Any]]) -> set[str]:
    """
    Extract set of (pair, order_type) strings from exchange open orders
    for orders that look like stoploss orders.

    Stoploss order types on Binance Spot: "stop_loss_limit", "stop_limit", "stop".
    Returns set of pair strings that have an active stop order.
    """
    stop_types = {"stop_limit", "stop_loss_limit", "stop", "stop_market"}
    pairs_with_stop = set()

    for order in exchange_orders:
        order_type = str(order.get("type") or order.get("order_type") or "").lower()
        if order_type in stop_types:
            pair = str(order.get("symbol") or order.get("pair") or "")
            if pair:
                pairs_with_stop.add(pair)

    return pairs_with_stop


def _has_stop_for_trade(
    trade_id: str,
    pair: str,
    exchange_stop_pairs: set[str],
    exchange_orders: list[dict[str, Any]],
) -> bool:
    """
    Check if a stop order exists for a given trade on the exchange.

    Matching strategy (in priority order):
    1. clientOrderId contains trade_id (Freqtrade sets this on stoploss orders).
    2. pair matches an order with stop-type.

    Pair-based matching is a fallback — it will false-positive if multiple trades
    on the same pair all have stops. Acceptable for MVP: we're checking existence,
    not uniqueness.
    """
    if not pair and not trade_id:
        return False

    stop_types = {"stop_limit", "stop_loss_limit", "stop", "stop_market"}

    for order in exchange_orders:
        order_type = str(order.get("type") or order.get("order_type") or "").lower()
        if order_type not in stop_types:
            continue

        # Priority 1: clientOrderId match
        client_id = str(order.get("clientOrderId") or order.get("client_order_id") or "")
        if trade_id and trade_id in client_id:
            return True

        # Priority 2: pair match
        order_pair = str(order.get("symbol") or order.get("pair") or "")
        if pair and order_pair == pair:
            return True

    return False