"""
sidecar/reconciler/startup_reconcile.py

Startup Reserved-Funds Reconciliation for RetailEdge.
Single responsibility: on every boot, recompute projected reserved funds
from exchange open orders and compare against local ledger state.

Hard constraint (CLAUDE.md #5):
Entry is BLOCKED if reconciliation finds any RESERVED_MISMATCH.

Source of truth priority:
1. Exchange open orders (fetch_open_orders) = what is actually reserved.
2. Local ledger (reserved_funds table) = what was recorded before restart.

If they differ beyond tolerance: post RESERVED_MISMATCH_ON_STARTUP to Decision Bus
and return BLOCK_ENTRY. Caller must not allow new entries until resolved.

Blueprint reference: v1.9 Section 6 Step 5.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from sidecar.reconciler.reserved_funds import ReservedFundsLedger


# ---------------------------------------------------------------------------
# Exchange client protocol
# ---------------------------------------------------------------------------
# In production: pass ccxt-based exchange client with fetch_open_orders().
# In tests: pass any object with a fetch_open_orders() method.
# No hard ccxt dependency in this module — keeps it testable without exchange creds.

class ExchangeClientProtocol:
    """Type hint only. Not enforced at runtime."""
    def fetch_open_orders(self) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Core reconciliation
# ---------------------------------------------------------------------------

def reconcile_reserved_funds_on_startup(
    exchange_client: Any,
    ledger: ReservedFundsLedger,
    decision_bus_post_fn: Optional[Callable] = None,
    tolerance_quote: float = 1.0,
) -> str:
    """
    Reconcile reserved funds on startup.

    Steps:
    1. Fetch open orders from exchange (source of truth).
    2. Load local reserved funds map from ledger.
    3. For each open buy order: compute exchange_reserved_quote = remaining * price.
    4. Compare to local ledger entry. Write audit record with MATCH or RESERVED_MISMATCH.
    5. Replace ledger reserved_funds table with exchange projection.
    6. If any RESERVED_MISMATCH found: post to Decision Bus, return BLOCK_ENTRY.
    7. If all match: return OK.

    Args:
        exchange_client:      Object with fetch_open_orders() -> list[dict].
        ledger:               ReservedFundsLedger instance.
        decision_bus_post_fn: Injectable. If None, Decision Bus step is skipped (test mode).
        tolerance_quote:      Acceptable diff in quote currency before flagging mismatch.
                              Default 1.0 USDT — absorbs rounding differences.

    Returns:
        "OK"           — all entries match, entry is allowed.
        "BLOCK_ENTRY"  — at least one RESERVED_MISMATCH found, entry is blocked.

    Why tolerance_quote=1.0 default:
    Exchange APIs return remaining quantity as a float. Multiplying by price
    introduces floating-point rounding. A 1 USDT tolerance absorbs this without
    masking genuine mismatches (which are typically order-of-magnitude larger).
    """
    run_id = _new_run_id()

    # --- Step 1: fetch exchange open orders
    try:
        open_orders = exchange_client.fetch_open_orders()
    except Exception as exc:
        # Exchange fetch failure is not a reconciliation mismatch.
        # It's an infrastructure failure — block entry conservatively.
        if decision_bus_post_fn is not None:
            decision_bus_post_fn(
                action_type="RESERVED_MISMATCH_ON_STARTUP",
                reason=f"exchange_fetch_failed: {exc}",
                severity="HIGH",
                run_id=run_id,
            )
        return "BLOCK_ENTRY"

    # --- Step 2: load local state
    local_reserved = ledger.load_reserved_funds_map()

    # --- Step 3 + 4: compute projection and write audit records
    projected: dict[str, float] = {}

    for order in open_orders:
        order_id = str(order.get("id", ""))
        pair = str(order.get("symbol") or order.get("pair") or "UNKNOWN")
        side = str(order.get("side", "")).lower()

        # Only buy orders reserve quote currency.
        # Sell orders reserve base currency, which is already held.
        if side != "buy":
            continue

        remaining = _safe_float(order.get("remaining"), 0.0)
        price = _safe_float(order.get("price"), 0.0)

        # reserved_quote = how much USDT is locked for this order
        exchange_reserved_quote = remaining * price

        local_quote = local_reserved.get(order_id, 0.0)
        diff = abs(exchange_reserved_quote - local_quote)

        if diff <= tolerance_quote:
            status = "MATCH"
        else:
            status = "RESERVED_MISMATCH"

        ledger.write_reserved_recon(
            run_id=run_id,
            order_id=order_id,
            pair=pair,
            exchange_reserved_quote=exchange_reserved_quote,
            local_reserved_quote=local_quote,
            diff_quote=diff,
            status=status,
        )

        projected[order_id] = exchange_reserved_quote

    # --- Step 5: replace ledger with exchange projection
    # Do this before checking for mismatches so the ledger is always current,
    # even if we end up blocking entry. The block is advisory; the data is truth.
    ledger.replace_reserved_funds_from_exchange_projection(projected)

    # --- Step 6: check for any mismatch
    if ledger.has_status(run_id, "RESERVED_MISMATCH"):
        if decision_bus_post_fn is not None:
            decision_bus_post_fn(
                action_type="RESERVED_MISMATCH_ON_STARTUP",
                reason="reserved_funds_mismatch_detected_on_boot",
                severity="HIGH",
                run_id=run_id,
            )
        return "BLOCK_ENTRY"

    return "OK"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_run_id() -> str:
    """
    Generate a unique run_id for this reconciliation pass.
    Format: "recon_<UTC_date>_<short_uuid>"
    Human-readable prefix makes log correlation easier than raw UUID.
    """
    date_part = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"recon_{date_part}_{short_uuid}"


def _safe_float(value: Any, default: float) -> float:
    """
    Parse a value to float without raising. Returns default on None or parse error.
    Exchange APIs can return None, "", or string floats — all must be handled.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default