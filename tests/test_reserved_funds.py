"""
tests/test_reserved_funds.py

Test gate: Dry-run.

Required tests (Sprint 1 S1-4/S1-5 gate):
- test_reserved_funds_startup_reconcile_match
- test_reserved_funds_startup_mismatch_blocks_entry

All tests use in-memory SQLite. No exchange credentials needed.
Exchange client is a simple mock object with fetch_open_orders().
"""

import pytest
from sidecar.reconciler.reserved_funds import ReservedFundsLedger
from sidecar.reconciler.startup_reconcile import reconcile_reserved_funds_on_startup


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ledger():
    """Fresh in-memory ledger for each test. No file I/O."""
    db = ReservedFundsLedger(":memory:")
    yield db
    db.close()


def make_exchange_client(open_orders: list) -> object:
    """
    Minimal mock exchange client.
    Returns the given list from fetch_open_orders().
    No ccxt dependency.
    """
    class MockExchangeClient:
        def fetch_open_orders(self):
            return open_orders
    return MockExchangeClient()


def make_open_order(
    order_id: str,
    pair: str,
    side: str,
    remaining: float,
    price: float,
) -> dict:
    """Construct a minimal exchange open order dict."""
    return {
        "id": order_id,
        "symbol": pair,
        "side": side,
        "remaining": remaining,
        "price": price,
    }


# ---------------------------------------------------------------------------
# test_reserved_funds_startup_reconcile_match
# ---------------------------------------------------------------------------

def test_reserved_funds_startup_reconcile_match(ledger):
    """
    When exchange open orders and local ledger agree (within tolerance):
    - reconcile returns "OK"
    - all reconciliation records have status "MATCH"
    - Decision Bus is NOT called (no mismatch to report)
    - ledger reserved_funds table is replaced with exchange projection

    Scenario: one open buy order. Local ledger has the same reserved amount.
    """
    order_id = "order_001"
    pair = "BTC/USDT"
    remaining = 0.01   # BTC
    price = 50000.0    # USDT
    exchange_reserved = remaining * price  # 500.0 USDT

    # Pre-populate local ledger with the same amount (no mismatch)
    ledger.upsert_reserved(order_id, pair, exchange_reserved)

    exchange_client = make_exchange_client([
        make_open_order(order_id, pair, "buy", remaining, price)
    ])

    bus_calls = []
    result = reconcile_reserved_funds_on_startup(
        exchange_client=exchange_client,
        ledger=ledger,
        decision_bus_post_fn=lambda **kw: bus_calls.append(kw),
        tolerance_quote=1.0,
    )

    # --- Core assertion: result must be OK
    assert result == "OK", f"Expected OK, got {result}"

    # --- Decision Bus must NOT have been called
    assert len(bus_calls) == 0, f"Bus called unexpectedly: {bus_calls}"

    # --- Ledger must reflect exchange projection
    reserved_map = ledger.load_reserved_funds_map()
    assert order_id in reserved_map
    assert abs(reserved_map[order_id] - exchange_reserved) < 0.01

    # --- Total reserved must equal exchange projection
    total = ledger.get_total_reserved()
    assert abs(total - exchange_reserved) < 0.01


def test_reserved_funds_startup_reconcile_match_within_tolerance(ledger):
    """
    A small floating-point diff (< 1.0 USDT) must still be MATCH, not MISMATCH.
    This guards against false positives from exchange API rounding.
    """
    order_id = "order_002"
    remaining = 0.012345
    price = 48000.0
    exchange_reserved = remaining * price  # 592.56 USDT
    local_amount = exchange_reserved + 0.50  # 0.50 USDT diff — within tolerance

    ledger.upsert_reserved(order_id, "ETH/USDT", local_amount)

    exchange_client = make_exchange_client([
        make_open_order(order_id, "ETH/USDT", "buy", remaining, price)
    ])

    result = reconcile_reserved_funds_on_startup(
        exchange_client=exchange_client,
        ledger=ledger,
        tolerance_quote=1.0,
    )
    assert result == "OK"


def test_reserved_funds_no_open_orders_returns_ok(ledger):
    """
    No open orders on exchange = nothing to reserve = OK.
    Local ledger may have stale entries; they are cleared by replace.
    """
    # Stale entry in local ledger
    ledger.upsert_reserved("stale_order", "BTC/USDT", 999.0)

    exchange_client = make_exchange_client([])  # empty

    result = reconcile_reserved_funds_on_startup(
        exchange_client=exchange_client,
        ledger=ledger,
    )
    assert result == "OK"

    # Stale entry must be cleared — exchange projection was empty
    assert ledger.get_total_reserved() == 0.0


# ---------------------------------------------------------------------------
# test_reserved_funds_startup_mismatch_blocks_entry
# ---------------------------------------------------------------------------

def test_reserved_funds_startup_mismatch_blocks_entry(ledger):
    """
    When exchange reserved amount differs from local ledger beyond tolerance:
    - reconcile returns "BLOCK_ENTRY"
    - Decision Bus is called with action_type="RESERVED_MISMATCH_ON_STARTUP"
    - ledger is still updated with exchange projection (data remains current)

    Scenario: exchange shows 500 USDT reserved, local ledger shows only 10 USDT.
    Diff = 490 USDT >> 1.0 USDT tolerance. Must trigger mismatch.

    Why this matters: a stale ledger could allow entries that exceed actual
    available balance, causing order rejection at the exchange level or
    double-spending of reserved funds.
    """
    order_id = "order_003"
    pair = "BTC/USDT"
    remaining = 0.01
    price = 50000.0
    exchange_reserved = remaining * price  # 500.0 USDT

    # Local ledger has grossly different amount — simulates crash/restart corruption
    local_stale = 10.0
    ledger.upsert_reserved(order_id, pair, local_stale)

    exchange_client = make_exchange_client([
        make_open_order(order_id, pair, "buy", remaining, price)
    ])

    bus_calls = []
    result = reconcile_reserved_funds_on_startup(
        exchange_client=exchange_client,
        ledger=ledger,
        decision_bus_post_fn=lambda **kw: bus_calls.append(kw),
        tolerance_quote=1.0,
    )

    # --- Core assertion: entry must be blocked
    assert result == "BLOCK_ENTRY", f"Expected BLOCK_ENTRY, got {result}"

    # --- Decision Bus must have been called with correct action type
    assert len(bus_calls) == 1, f"Expected 1 bus call, got {len(bus_calls)}"
    assert bus_calls[0]["action_type"] == "RESERVED_MISMATCH_ON_STARTUP"
    assert bus_calls[0]["severity"] == "HIGH"

    # --- Ledger is still updated with exchange projection (even on block)
    reserved_map = ledger.load_reserved_funds_map()
    assert order_id in reserved_map
    assert abs(reserved_map[order_id] - exchange_reserved) < 0.01


def test_reserved_funds_mismatch_exact_tolerance_boundary(ledger):
    """
    Diff exactly at tolerance boundary (== 1.0 USDT) must be MATCH (not MISMATCH).
    Diff just above tolerance (1.01 USDT) must be MISMATCH.
    Verifies boundary condition is <= not <.
    """
    order_id = "order_004"
    remaining = 0.1
    price = 10000.0
    exchange_reserved = 1000.0

    # Exactly at boundary — must be MATCH
    ledger.upsert_reserved(order_id, "BTC/USDT", exchange_reserved + 1.0)
    exchange_client = make_exchange_client([
        make_open_order(order_id, "BTC/USDT", "buy", remaining, price)
    ])
    result = reconcile_reserved_funds_on_startup(
        exchange_client=exchange_client,
        ledger=ledger,
        tolerance_quote=1.0,
    )
    assert result == "OK", "Diff == tolerance should be MATCH"

    # Just above boundary — must be MISMATCH
    ledger.upsert_reserved(order_id, "BTC/USDT", exchange_reserved + 1.01)
    result2 = reconcile_reserved_funds_on_startup(
        exchange_client=make_exchange_client([
            make_open_order(order_id, "BTC/USDT", "buy", remaining, price)
        ]),
        ledger=ledger,
        tolerance_quote=1.0,
    )
    assert result2 == "BLOCK_ENTRY", "Diff > tolerance should be MISMATCH"


def test_sell_orders_not_counted_in_reserved(ledger):
    """
    Sell orders do NOT reserve quote currency — they reserve base currency.
    Reconciler must ignore sell orders when computing reserved_quote.
    """
    exchange_client = make_exchange_client([
        make_open_order("sell_order_001", "BTC/USDT", "sell", 0.01, 50000.0)
    ])

    result = reconcile_reserved_funds_on_startup(
        exchange_client=exchange_client,
        ledger=ledger,
    )
    assert result == "OK"
    assert ledger.get_total_reserved() == 0.0


def test_exchange_fetch_failure_blocks_entry(ledger):
    """
    If exchange.fetch_open_orders() raises, reconcile must return BLOCK_ENTRY.
    Cannot verify reserved state without exchange data — conservative block.
    """
    class FailingExchangeClient:
        def fetch_open_orders(self):
            raise ConnectionError("exchange unreachable")

    bus_calls = []
    result = reconcile_reserved_funds_on_startup(
        exchange_client=FailingExchangeClient(),
        ledger=ledger,
        decision_bus_post_fn=lambda **kw: bus_calls.append(kw),
    )
    assert result == "BLOCK_ENTRY"
    assert len(bus_calls) == 1
    assert bus_calls[0]["action_type"] == "RESERVED_MISMATCH_ON_STARTUP"


# ---------------------------------------------------------------------------
# ReservedFundsLedger unit tests
# ---------------------------------------------------------------------------

def test_ledger_upsert_and_total(ledger):
    """upsert_reserved updates total correctly."""
    ledger.upsert_reserved("o1", "BTC/USDT", 100.0)
    ledger.upsert_reserved("o2", "ETH/USDT", 200.0)
    assert abs(ledger.get_total_reserved() - 300.0) < 0.001

    # Update o1 — total should change
    ledger.upsert_reserved("o1", "BTC/USDT", 150.0)
    assert abs(ledger.get_total_reserved() - 350.0) < 0.001


def test_ledger_delete_removes_entry(ledger):
    """delete_reserved removes the entry and reduces total."""
    ledger.upsert_reserved("o1", "BTC/USDT", 500.0)
    ledger.delete_reserved("o1")
    assert ledger.get_total_reserved() == 0.0
    assert "o1" not in ledger.load_reserved_funds_map()


def test_ledger_replace_clears_stale_entries(ledger):
    """replace_reserved_funds_from_exchange_projection clears all prior entries."""
    ledger.upsert_reserved("stale_1", "BTC/USDT", 999.0)
    ledger.upsert_reserved("stale_2", "ETH/USDT", 888.0)

    projection = {"new_order_1": 200.0, "new_order_2": 300.0}
    ledger.replace_reserved_funds_from_exchange_projection(projection)

    reserved_map = ledger.load_reserved_funds_map()
    assert "stale_1" not in reserved_map
    assert "stale_2" not in reserved_map
    assert "new_order_1" in reserved_map
    assert abs(reserved_map["new_order_1"] - 200.0) < 0.001