"""
sidecar/reconciler/pre_trade_gate.py

Pre-Trade Gate for RetailEdge.
Single responsibility: block entry if any pre-conditions fail before
an order is sent to the exchange.

Hard rules (CLAUDE.md / blueprint v1.9 Section 7.4):
- Block if projected_available < proposed_stake * 1.05 (5% safety buffer).
- Block if proposed_stake < min_notional for the venue.
- Block if exposure limit would be breached.
- Post REJECT_ENTRY to Decision Bus on every block.
- Return False on block, True on pass.

Design:
- pre_entry_balance_check() is the primary gate called from Freqtrade
  confirm_trade_entry() callback.
- All checks are pure: inputs are passed in, no global state.
- Decision Bus is injectable for test isolation.
- Regime multiplier is applied to proposed_stake by the caller (strategy);
  the gate validates the resulting stake, not the multiplier itself.

Why 1.05 buffer (5%):
Exchange fees, price slippage, and partial fills can consume more quote
than the nominal stake. A 5% buffer ensures reserved funds calculation
never underestimates actual exposure.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Safety buffer multiplier — stake must be available with this headroom
BALANCE_BUFFER = 1.05


# ---------------------------------------------------------------------------
# Core gate function
# ---------------------------------------------------------------------------

def pre_entry_balance_check(
    pair: str,
    proposed_stake: float,
    wallet_available: float,
    reserved_total: float,
    decision_bus_post_fn: Optional[Any] = None,
    min_notional: float = 0.0,
) -> bool:
    """
    Check if a proposed entry is safe given current balance and reserved funds.

    Args:
        pair:                 Trading pair (e.g. "BTC/USDT"). Used for bus post only.
        proposed_stake:       Quote amount to stake in this trade.
        wallet_available:     Current available quote balance from exchange wallet.
        reserved_total:       Total quote reserved for existing open orders.
        decision_bus_post_fn: Injectable Decision Bus post function.
                              If None, rejection is logged only (test mode).
        min_notional:         Minimum order size for this venue (from capability_matrix).
                              Default 0.0 = no min notional check.

    Returns:
        True  — all checks pass, entry is allowed.
        False — at least one check failed, entry is blocked.

    Checks (evaluated in order, all must pass):
    1. proposed_stake > 0 (sanity)
    2. proposed_stake >= min_notional (exchange minimum)
    3. projected_available >= proposed_stake * BALANCE_BUFFER

    projected_available = wallet_available - reserved_total
    This is the balance actually free to use, accounting for open order reservations.

    Why check projected_available and not wallet_available:
    wallet_available from exchange may include funds reserved for open buy orders
    that Freqtrade has not yet filled. reserved_total from the ledger accounts for
    these. Using wallet_available alone would allow over-commitment.
    """
    # --- Sanity: stake must be positive
    if proposed_stake <= 0:
        _reject(
            pair, proposed_stake,
            reason=f"proposed_stake={proposed_stake} <= 0",
            decision_bus_post_fn=decision_bus_post_fn,
        )
        return False

    # --- Check 1: min notional
    if min_notional > 0 and proposed_stake < min_notional:
        _reject(
            pair, proposed_stake,
            reason=f"proposed_stake={proposed_stake:.4f} < min_notional={min_notional:.4f}",
            decision_bus_post_fn=decision_bus_post_fn,
        )
        return False

    # --- Check 2: projected available balance with buffer
    projected_available = wallet_available - reserved_total
    required = proposed_stake * BALANCE_BUFFER

    if projected_available < required:
        _reject(
            pair, proposed_stake,
            reason=(
                f"INSUFFICIENT_PROJECTED_AVAILABLE_BALANCE "
                f"projected={projected_available:.4f} "
                f"required={required:.4f} "
                f"(stake={proposed_stake:.4f} x {BALANCE_BUFFER} buffer) "
                f"wallet={wallet_available:.4f} reserved={reserved_total:.4f}"
            ),
            decision_bus_post_fn=decision_bus_post_fn,
        )
        return False

    logger.debug(
        "Pre-trade gate PASS pair=%s stake=%.4f projected=%.4f required=%.4f",
        pair, proposed_stake, projected_available, required,
    )
    return True


def _reject(
    pair: str,
    proposed_stake: float,
    reason: str,
    decision_bus_post_fn: Optional[Any],
) -> None:
    """Log rejection and post REJECT_ENTRY to Decision Bus."""
    logger.info("Pre-trade gate BLOCK pair=%s stake=%.4f reason=%s", pair, proposed_stake, reason)

    if decision_bus_post_fn is not None:
        try:
            decision_bus_post_fn(
                action_type="REJECT_ENTRY",
                reason=reason,
                severity="INFO",
                pair=pair,
            )
        except Exception as exc:
            logger.warning("Failed to post REJECT_ENTRY to Decision Bus: %s", exc)


# ---------------------------------------------------------------------------
# Regime multiplier application (helper for strategy callbacks)
# ---------------------------------------------------------------------------

def apply_regime_multiplier(
    base_stake: float,
    regime_multiplier: float,
    min_multiplier: float = 0.0,
    max_multiplier: float = 1.5,
) -> float:
    """
    Apply regime multiplier to base stake with bounds.

    Used by RetailEdgeStrategy.custom_stake_amount() before calling
    pre_entry_balance_check().

    regime_multiplier: from Adaptive Regime Policy (e.g. 0.5 for ranging, 1.0 for trending).
    Bounds prevent extreme values from misconfigured regime policy.
    """
    multiplier = max(min_multiplier, min(max_multiplier, regime_multiplier))
    return base_stake * multiplier


# ---------------------------------------------------------------------------
# Min notional loader (helper to read from capability_matrix)
# ---------------------------------------------------------------------------

def get_min_notional(
    capability_matrix: dict[str, Any],
    exchange_key: str,
    pair: str = "",
    fallback: float = 10.0,
) -> float:
    """
    Get minimum notional for a venue from capability matrix.

    min_notional is typically pair-specific (e.g. BTC/USDT = $10, DOGE/USDT = $5).
    For MVP: use venue-level default. Pair-level refinement is post-B1 scope.

    Falls back to `fallback` (default 10 USDT) if not configured.
    """
    cap = capability_matrix.get(exchange_key, {})
    return float(cap.get("min_notional_usdt", fallback))