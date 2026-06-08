"""
tests/test_reconciler.py

Test gate: Dry-run.

Core done criteria:
- Fill deduplication: same (order_id, fill_id) inserted twice = only one record
- Heartbeat written every run_once() call

All tests use in-memory SQLite + mock Freqtrade client.
No running Freqtrade instance needed.
"""

import pytest
from unittest.mock import MagicMock
from sidecar.reconciler.reconciler import Reconciler, ReconcilerLedger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ledger():
    db = ReconcilerLedger(":memory:")
    yield db
    db.close()


def make_ft_client(open_orders=None, trades=None):
    """Mock FreqtradeClient returning controlled data.

    get_open_orders() maps to /api/v1/status which returns trade objects,
    each containing an 'orders' list. open_orders are wrapped into a single
    mock trade object to match live Freqtrade response shape.
    get_trades() returns {"trades": [...]} directly.
    """
    ft = MagicMock()
    if open_orders:
        pair = open_orders[0].get("symbol", "BTC/USDT")
        wrapped = {"trade_id": "mock_trade_1", "pair": pair, "orders": open_orders}
        ft.get_open_orders.return_value = {"trades": [wrapped]}
    else:
        ft.get_open_orders.return_value = {"trades": []}
    ft.get_trades.return_value = {"trades": trades or []}
    return ft


def make_order(order_id, pair="BTC/USDT", side="buy", remaining=0.01, price=50000.0, status="open"):
    return {
        "id": order_id,
        "symbol": pair,
        "side": side,
        "type": "limit",
        "status": status,
        "amount": remaining,
        "filled": 0.0,
        "remaining": remaining,
        "price": price,
        "average": None,
        "cost": 0.0,
    }


def make_fill(fill_id, amount=0.005, price=50000.0, pair="BTC/USDT", side="buy"):
    return {
        "id": fill_id,
        "symbol": pair,
        "side": side,
        "amount": amount,
        "price": price,
        "cost": amount * price,
        "datetime": "2026-06-04T10:00:00+00:00",
    }


def make_trade_with_fills(trade_id, pair, order_id, fills):
    return {
        "trade_id": trade_id,
        "pair": pair,
        "orders": [
            {
                "order_id": order_id,
                "side": "buy",
                "fills": fills,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Fill deduplication — core done criteria
# ---------------------------------------------------------------------------

def test_fill_deduplication_same_fill_id_stored_once(ledger):
    """
    The critical invariant: same (order_id, fill_id) inserted twice
    must result in exactly one record in execution_fills.

    This guards against double-counting fills caused by:
    - Reconciler restart mid-loop
    - Exchange returning duplicate fill events
    - Network retry re-delivering the same fill

    Mechanism: INSERT OR IGNORE on PRIMARY KEY (order_id, fill_id).
    """
    order_id = "order_001"
    fill = make_fill("fill_001", amount=0.005, price=50000.0)

    # Insert same fill twice
    result1 = ledger.insert_fill_if_new(order_id, fill)
    result2 = ledger.insert_fill_if_new(order_id, fill)

    # First insert: new (True)
    assert result1 is True, "First insert of a fill must return True (new)"
    # Second insert: duplicate (False)
    assert result2 is False, "Second insert of same fill must return False (duplicate)"

    # Only one record in DB
    count = ledger.count_fills(order_id)
    assert count == 1, f"Expected 1 fill record, got {count} — deduplication failed"


def test_fill_deduplication_different_fill_ids_stored_separately(ledger):
    """
    Different fill_ids for the same order must each be stored.
    A partial fill produces multiple fills per order.
    """
    order_id = "order_002"
    fill_a = make_fill("fill_002a", amount=0.003, price=50000.0)
    fill_b = make_fill("fill_002b", amount=0.002, price=50100.0)

    ledger.insert_fill_if_new(order_id, fill_a)
    ledger.insert_fill_if_new(order_id, fill_b)

    count = ledger.count_fills(order_id)
    assert count == 2, f"Expected 2 distinct fills, got {count}"


def test_fill_deduplication_same_fill_id_different_orders(ledger):
    """
    Same fill_id on different order_ids must both be stored.
    The PK is (order_id, fill_id), not just fill_id.
    """
    fill = make_fill("fill_reused_id", amount=0.01, price=48000.0)

    ledger.insert_fill_if_new("order_A", fill)
    ledger.insert_fill_if_new("order_B", fill)

    assert ledger.count_fills("order_A") == 1
    assert ledger.count_fills("order_B") == 1


def test_fill_deduplication_across_run_once_calls(ledger):
    """
    Full end-to-end deduplication through Reconciler.run_once().
    Calling run_once() twice with the same trade data must not double-count fills.

    This is the realistic failure mode: Reconciler restarts and re-processes
    the same fills from Freqtrade's trade history.
    """
    fill = make_fill("fill_003")
    trade = make_trade_with_fills("trade_1", "BTC/USDT", "order_003", [fill])
    ft = make_ft_client(trades=[trade])

    reconciler = Reconciler(ft_client=ft, ledger=ledger)

    # Run twice — simulates restart or retry
    summary1 = reconciler.run_once()
    summary2 = reconciler.run_once()

    # First run: 1 new fill
    assert summary1["fills_new"] == 1
    assert summary1["fills_duplicate"] == 0

    # Second run: same fill = duplicate, not new
    assert summary2["fills_new"] == 0
    assert summary2["fills_duplicate"] == 1

    # Total in DB: still just 1
    assert ledger.count_fills("order_003") == 1


# ---------------------------------------------------------------------------
# Heartbeat — core done criteria
# ---------------------------------------------------------------------------

def test_heartbeat_written_on_every_run_once(ledger):
    """
    Heartbeat must be written after every run_once() call.
    Guardian uses this to detect Reconciler liveness.

    Verified: timestamp exists after first call, and is updated after second call.
    """
    ft = make_ft_client()
    reconciler = Reconciler(ft_client=ft, ledger=ledger)

    # Before any run: no heartbeat
    assert ledger.get_last_heartbeat_ts() is None

    reconciler.run_once()
    ts1 = ledger.get_last_heartbeat_ts()
    assert ts1 is not None, "Heartbeat must be written after run_once()"

    reconciler.run_once()
    ts2 = ledger.get_last_heartbeat_ts()
    assert ts2 is not None

    # Timestamp must be non-decreasing (second >= first)
    assert ts2 >= ts1, f"Heartbeat timestamp went backwards: {ts1} -> {ts2}"


def test_heartbeat_written_even_if_fetch_fails(ledger):
    """
    If Freqtrade REST is unreachable, heartbeat must still be written.
    Reconciler must not silently die — it should keep ticking.

    This prevents Guardian from falsely detecting Reconciler as dead
    when the issue is actually a transient Freqtrade outage.
    """
    ft = MagicMock()
    ft.get_open_orders.side_effect = ConnectionError("freqtrade unreachable")
    ft.get_trades.side_effect = ConnectionError("freqtrade unreachable")

    reconciler = Reconciler(ft_client=ft, ledger=ledger)
    summary = reconciler.run_once()

    # Heartbeat written despite fetch failures
    ts = ledger.get_last_heartbeat_ts()
    assert ts is not None, "Heartbeat must be written even when fetch fails"

    # Errors recorded in summary
    assert len(summary["errors"]) > 0
    assert "heartbeat_ts" in summary


def test_heartbeat_loop_count_increments(ledger):
    """Loop count in heartbeat must increment on each run_once()."""
    ft = make_ft_client()
    reconciler = Reconciler(ft_client=ft, ledger=ledger)

    reconciler.run_once()
    reconciler.run_once()
    reconciler.run_once()

    cur = ledger._conn.execute(
        "SELECT loop_count FROM reconciler_heartbeat WHERE id='reconciler'"
    )
    row = cur.fetchone()
    assert row[0] == 3, f"Expected loop_count=3, got {row[0]}"


# ---------------------------------------------------------------------------
# Order upsert
# ---------------------------------------------------------------------------

def test_order_upserted_from_open_orders(ledger):
    """Orders from get_open_orders() are written to execution_orders."""
    order = make_order("order_open_1")
    ft = make_ft_client(open_orders=[order])
    reconciler = Reconciler(ft_client=ft, ledger=ledger)

    summary = reconciler.run_once()

    assert summary["orders_upserted"] == 1
    stored = ledger.get_order("order_open_1")
    assert stored is not None
    assert stored["pair"] == "BTC/USDT"
    assert stored["side"] == "buy"


def test_order_status_updated_on_subsequent_run(ledger):
    """
    If order status changes between runs (open -> closed),
    the upsert must reflect the latest state.
    """
    order_open = make_order("order_upd_1", status="open")
    order_closed = make_order("order_upd_1", status="closed")
    order_closed["filled"] = 0.01

    ft = make_ft_client(open_orders=[order_open])
    reconciler = Reconciler(ft_client=ft, ledger=ledger)
    reconciler.run_once()

    assert ledger.get_order("order_upd_1")["status"] == "open"

    # Second run: order now closed — wrap in trade object (live Freqtrade format)
    ft.get_open_orders.return_value = {"trades": [{"trade_id": "mock_trade_1", "pair": "BTC/USDT", "orders": [order_closed]}]}
    reconciler.run_once()

    updated = ledger.get_order("order_upd_1")
    assert updated["status"] == "closed"
    assert updated["filled"] == 0.01


# ---------------------------------------------------------------------------
# Reserved funds integration
# ---------------------------------------------------------------------------

def test_reserved_funds_updated_for_open_buy_orders(ledger):
    """
    open buy orders update reserved_funds in the reserved ledger.
    """
    from sidecar.reconciler.reserved_funds import ReservedFundsLedger

    reserved_ledger = ReservedFundsLedger(":memory:")
    order = make_order("order_res_1", side="buy", remaining=0.01, price=50000.0)
    ft = make_ft_client(open_orders=[order])

    reconciler = Reconciler(ft_client=ft, ledger=ledger, reserved_ledger=reserved_ledger)
    reconciler.run_once()

    total = reserved_ledger.get_total_reserved()
    assert abs(total - 500.0) < 0.01, f"Expected 500 USDT reserved, got {total}"
    reserved_ledger.close()


def test_sell_orders_not_counted_in_reserved(ledger):
    """Sell orders must not add to reserved quote currency."""
    from sidecar.reconciler.reserved_funds import ReservedFundsLedger

    reserved_ledger = ReservedFundsLedger(":memory:")
    order = make_order("order_sell_1", side="sell", remaining=0.01, price=50000.0)
    ft = make_ft_client(open_orders=[order])

    reconciler = Reconciler(ft_client=ft, ledger=ledger, reserved_ledger=reserved_ledger)
    reconciler.run_once()

    assert reserved_ledger.get_total_reserved() == 0.0
    reserved_ledger.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_fill_without_id_uses_surrogate(ledger):
    """
    Fills without an id field must still be stored using a surrogate key.
    Must not raise or silently drop the fill.
    """
    order_id = "order_no_fill_id"
    fill_no_id = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "amount": 0.005,
        "price": 50000.0,
        "cost": 250.0,
        "datetime": "2026-06-04T10:00:00+00:00",
        # no "id" field
    }
    result = ledger.insert_fill_if_new(order_id, fill_no_id)
    assert result is True
    assert ledger.count_fills(order_id) == 1


def test_empty_response_does_not_crash(ledger):
    """Empty lists from Freqtrade must not raise."""
    ft = make_ft_client(open_orders=[], trades=[])
    reconciler = Reconciler(ft_client=ft, ledger=ledger)
    summary = reconciler.run_once()
    assert summary["orders_upserted"] == 0
    assert summary["fills_new"] == 0
    assert ledger.get_last_heartbeat_ts() is not None


def test_multiple_fills_per_order_all_stored(ledger):
    """Three distinct fills on one order must all be stored."""
    order_id = "order_multi_fill"
    fills = [
        make_fill(f"fill_{i}", amount=0.001, price=50000.0 + i)
        for i in range(3)
    ]
    trade = make_trade_with_fills("trade_mf", "BTC/USDT", order_id, fills)
    ft = make_ft_client(trades=[trade])

    reconciler = Reconciler(ft_client=ft, ledger=ledger)
    summary = reconciler.run_once()

    assert summary["fills_new"] == 3
    assert ledger.count_fills(order_id) == 3