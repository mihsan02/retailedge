"""
sidecar/guardian/emergency_exit.py — Sprint 4 S4-1

Emergency Exit Cascade — v1.9 blueprint Section 6 Step 4.

Cascade sequence:
  market attempt 1
  → fail → sleep 0.5s
  → market attempt 2
  → fail → sleep 1.0s
  → aggressive limit (best_bid * 0.98)
  → fail → EMERGENCY_EXIT_FAILED posted to Decision Bus → OPERATOR_REQUIRED

Rules (from blueprint):
  - emergency_exit_enabled MUST be false until drill passes. Enforced here.
  - Bot liveness checked via ft_client.ping_alive() before any attempt.
  - DONE only on confirmed 2xx. No assumption of success.
  - If bot is down: BOT_DOWN_NO_REST_EXIT — do not attempt REST calls.
  - EMERGENCY_EXIT_FAILED is a canonical Decision Bus action type.

In production, time.sleep() calls are real. Tests must inject a no-op sleep.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGGRESSIVE_LIMIT_DISCOUNT = 0.98   # best_bid * this
MARKET_MAX_RETRIES = 2             # attempts before falling to aggressive limit
RETRY_DELAYS_SEC = [0.5, 1.0]     # delay[0] between market attempts, delay[1] before limit
LIMIT_DELAY_SEC = 1.0              # delay before aggressive limit attempt

# Return status strings — exhaustive, tested in test_no_orphan_decision_contract
STATUS_DISABLED = "DISABLED_OPERATOR_REQUIRED"
STATUS_BOT_DOWN = "BOT_DOWN_NO_REST_EXIT"
STATUS_DONE_MARKET = "DONE_MARKET"
STATUS_DONE_LIMIT = "DONE_AGGRESSIVE_LIMIT"
STATUS_FAILED = "FAILED_OPERATOR_REQUIRED"


# ---------------------------------------------------------------------------
# Protocols — injected in production, mocked in tests
# ---------------------------------------------------------------------------

class FtClient(Protocol):
    def ping_alive(self) -> bool: ...
    def forceexit(self, trade_id: str, ordertype: str, price: float | None = None) -> Any: ...


class ExchangeClient(Protocol):
    def fetch_order_book(self, pair: str, limit: int = 5) -> dict: ...


class DecisionBus(Protocol):
    def post(self, action_type: str, **kwargs) -> str: ...


# ---------------------------------------------------------------------------
# Policy dataclass
# ---------------------------------------------------------------------------

@dataclass
class EmergencyExitPolicy:
    emergency_exit_enabled: bool = False
    aggressive_limit_discount: float = AGGRESSIVE_LIMIT_DISCOUNT
    market_max_retries: int = MARKET_MAX_RETRIES
    retry_delays_sec: list[float] = field(default_factory=lambda: list(RETRY_DELAYS_SEC))


@dataclass
class Trade:
    id: str
    pair: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_2xx(response: Any) -> bool:
    """
    Check if response indicates success.
    Accepts: objects with .status_code, dicts with 'status_code', int directly.
    Returns False for None, exceptions, or any non-2xx value.
    """
    if response is None:
        return False
    if isinstance(response, int):
        return 200 <= response < 300
    if isinstance(response, dict):
        code = response.get("status_code")
        if code is not None:
            return 200 <= int(code) < 300
        # Some clients return {"status": "ok"} — not sufficient; require explicit 2xx
        return False
    code = getattr(response, "status_code", None)
    if code is not None:
        return 200 <= int(code) < 300
    return False


def _get_best_bid(exchange_client: ExchangeClient, pair: str) -> float | None:
    """Fetch best bid from order book. Returns None on any failure."""
    try:
        book = exchange_client.fetch_order_book(pair, limit=5)
        bids = book.get("bids", [])
        if not bids:
            logger.error("AGGRESSIVE_LIMIT: empty order book for %s", pair)
            return None
        return float(bids[0][0])
    except Exception as exc:
        logger.exception("AGGRESSIVE_LIMIT: order book fetch failed for %s: %s", pair, exc)
        return None


# ---------------------------------------------------------------------------
# Main cascade
# ---------------------------------------------------------------------------

def emergency_exit_cascade(
    trade: Trade,
    ft_client: FtClient,
    exchange_client: ExchangeClient,
    decision_bus: DecisionBus,
    policy: EmergencyExitPolicy,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    """
    Execute emergency exit cascade.

    Args:
        trade:           Trade to exit (id + pair).
        ft_client:       Freqtrade REST client.
        exchange_client: Exchange client for order book.
        decision_bus:    Decision Bus for posting EMERGENCY_EXIT_FAILED.
        policy:          EmergencyExitPolicy (emergency_exit_enabled must be True).
        sleep_fn:        Injected sleep — override in tests to avoid real delays.

    Returns one of: STATUS_DISABLED, STATUS_BOT_DOWN, STATUS_DONE_MARKET,
                    STATUS_DONE_LIMIT, STATUS_FAILED.

    DONE only on confirmed 2xx. All other paths post EMERGENCY_EXIT_FAILED
    and return STATUS_FAILED or a guard status.
    """

    # --- Guard: policy gate ---
    if not policy.emergency_exit_enabled:
        logger.warning(
            "EMERGENCY_EXIT: disabled by policy. trade_id=%s pair=%s. "
            "Enable only after drill test passes.",
            trade.id, trade.pair,
        )
        return STATUS_DISABLED

    # --- Guard: bot liveness ---
    if not ft_client.ping_alive():
        logger.critical(
            "EMERGENCY_EXIT: bot is down. Cannot issue REST exit. trade_id=%s pair=%s",
            trade.id, trade.pair,
        )
        decision_bus.post(
            "EMERGENCY_EXIT_FAILED",
            pair=trade.pair,
            trade_id=trade.id,
            severity="CRITICAL",
            reason="BOT_DOWN_NO_REST_EXIT",
        )
        return STATUS_BOT_DOWN

    # --- Attempt 1 and 2: market order ---
    for attempt in range(1, policy.market_max_retries + 1):
        logger.info(
            "EMERGENCY_EXIT: market attempt %d/%d trade_id=%s pair=%s",
            attempt, policy.market_max_retries, trade.id, trade.pair,
        )
        try:
            resp = ft_client.forceexit(trade_id=trade.id, ordertype="market")
        except Exception as exc:
            logger.exception("EMERGENCY_EXIT: market attempt %d raised: %s", attempt, exc)
            resp = None

        if _is_2xx(resp):
            logger.info("EMERGENCY_EXIT: DONE via market. attempt=%d trade_id=%s", attempt, trade.id)
            return STATUS_DONE_MARKET

        logger.warning(
            "EMERGENCY_EXIT: market attempt %d failed. trade_id=%s resp=%s",
            attempt, trade.id, resp,
        )

        # Sleep between market attempts (index 0 = after attempt 1)
        # After last market attempt, sleep before aggressive limit (index 1)
        delay_idx = attempt - 1  # 0 after attempt 1, 1 after attempt 2
        if delay_idx < len(policy.retry_delays_sec):
            sleep_fn(policy.retry_delays_sec[delay_idx])

    # --- Attempt 3: aggressive limit ---
    logger.info(
        "EMERGENCY_EXIT: falling back to aggressive limit. trade_id=%s pair=%s",
        trade.id, trade.pair,
    )

    best_bid = _get_best_bid(exchange_client, trade.pair)
    if best_bid is not None:
        aggressive_price = best_bid * policy.aggressive_limit_discount
        logger.info(
            "EMERGENCY_EXIT: aggressive limit price=%.6f (bid=%.6f * %.2f) trade_id=%s",
            aggressive_price, best_bid, policy.aggressive_limit_discount, trade.id,
        )
        try:
            resp_limit = ft_client.forceexit(
                trade_id=trade.id,
                ordertype="limit",
                price=aggressive_price,
            )
        except Exception as exc:
            logger.exception("EMERGENCY_EXIT: aggressive limit raised: %s", exc)
            resp_limit = None

        if _is_2xx(resp_limit):
            logger.info("EMERGENCY_EXIT: DONE via aggressive limit. trade_id=%s", trade.id)
            return STATUS_DONE_LIMIT

        logger.warning(
            "EMERGENCY_EXIT: aggressive limit failed. trade_id=%s resp=%s",
            trade.id, resp_limit,
        )
    else:
        logger.error(
            "EMERGENCY_EXIT: could not fetch order book for aggressive limit. trade_id=%s",
            trade.id,
        )

    # --- All attempts exhausted ---
    logger.critical(
        "EMERGENCY_EXIT: all attempts failed. OPERATOR_REQUIRED. trade_id=%s pair=%s",
        trade.id, trade.pair,
    )
    decision_bus.post(
        "EMERGENCY_EXIT_FAILED",
        pair=trade.pair,
        trade_id=trade.id,
        severity="CRITICAL",
        reason="market_and_aggressive_limit_failed",
    )
    return STATUS_FAILED