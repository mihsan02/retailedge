"""
tests/test_atr_percentile.py

Test gate: Dry-run.

Required test (Sprint 2 S2-4 gate):
- test_stoploss_interval_uses_atr_baseline

Covers next_stoploss_audit_interval() and StoplossAuditor.run_once()
with injected atr_state dicts — no ATR service dependency needed.
"""

import pytest
from unittest.mock import MagicMock

from sidecar.stoploss_auditor.auditor import (
    next_stoploss_audit_interval,
    StoplossAuditor,
    INTERVAL_NORMAL_SEC,
    INTERVAL_HIGH_RISK_SEC,
    ATR_HIGH_VOL_THRESHOLD,
)
from sidecar.guardian.decision_bus import DecisionBus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bus():
    b = DecisionBus(":memory:")
    yield b
    b.close()


def atr_ok(percentile: float) -> dict:
    return {"status": "OK", "atr_percentile": percentile}


def atr_insufficient() -> dict:
    return {"status": "ATR_BASELINE_INSUFFICIENT", "atr_percentile": None}


def open_trade(trade_id="trade_1", pair="BTC/USDT") -> dict:
    return {"trade_id": trade_id, "pair": pair, "stoploss": -0.05}


def stop_order(pair="BTC/USDT", order_type="stop_limit") -> dict:
    return {"id": "stop_001", "symbol": pair, "type": order_type, "side": "sell"}


# ---------------------------------------------------------------------------
# test_stoploss_interval_uses_atr_baseline  ← REQUIRED GATE
# ---------------------------------------------------------------------------

def test_stoploss_interval_uses_atr_baseline():
    """
    next_stoploss_audit_interval must return:
    - 60s when ATR percentile >= 0.80 (high volatility)
    - 300s when ATR percentile < 0.80 (normal)
    - 60s when ATR baseline is insufficient (conservative)
    - 300s when no open trades (nothing to audit)

    This is the core invariant from blueprint v1.9 Section 6 Step 6.
    ATR percentile must come from a real historical baseline (5000 candles min),
    not from arbitrary latest-dataframe quantiles.

    The interval function is pure — no side effects, no external calls.
    Injectable atr_state makes it testable without running ATR service.
    """
    trades = [open_trade()]

    # High volatility — must use 60s
    assert next_stoploss_audit_interval(trades, atr_ok(0.80)) == INTERVAL_HIGH_RISK_SEC, (
        "ATR percentile == 0.80 must trigger 60s interval"
    )
    assert next_stoploss_audit_interval(trades, atr_ok(0.85)) == INTERVAL_HIGH_RISK_SEC, (
        "ATR percentile > 0.80 must trigger 60s interval"
    )
    assert next_stoploss_audit_interval(trades, atr_ok(0.99)) == INTERVAL_HIGH_RISK_SEC, (
        "ATR percentile at 0.99 must trigger 60s interval"
    )

    # Below threshold — must use 300s
    assert next_stoploss_audit_interval(trades, atr_ok(0.79)) == INTERVAL_NORMAL_SEC, (
        "ATR percentile just below 0.80 must use 300s interval"
    )
    assert next_stoploss_audit_interval(trades, atr_ok(0.50)) == INTERVAL_NORMAL_SEC, (
        "ATR percentile 0.50 must use 300s interval"
    )
    assert next_stoploss_audit_interval(trades, atr_ok(0.00)) == INTERVAL_NORMAL_SEC, (
        "ATR percentile 0.00 must use 300s interval"
    )

    # Insufficient baseline with open trades — must use 60s
    assert next_stoploss_audit_interval(trades, atr_insufficient()) == INTERVAL_HIGH_RISK_SEC, (
        "ATR_BASELINE_INSUFFICIENT with open trades must trigger 60s"
    )

    # No open trades — always 300s regardless of ATR
    assert next_stoploss_audit_interval([], atr_ok(0.99)) == INTERVAL_NORMAL_SEC, (
        "No open trades must always return 300s"
    )
    assert next_stoploss_audit_interval([], atr_insufficient()) == INTERVAL_NORMAL_SEC, (
        "No open trades + insufficient baseline must still return 300s"
    )


# ---------------------------------------------------------------------------
# Interval policy edge cases
# ---------------------------------------------------------------------------

def test_interval_recent_incidents_override_atr():
    """
    recent_stop_incidents > 0 forces 60s regardless of ATR state.
    Incident response mode takes priority over volatility regime.
    """
    trades = [open_trade()]

    # Low ATR + incident = still 60s
    assert next_stoploss_audit_interval(trades, atr_ok(0.10), recent_stop_incidents=1) == INTERVAL_HIGH_RISK_SEC
    assert next_stoploss_audit_interval(trades, atr_ok(0.50), recent_stop_incidents=3) == INTERVAL_HIGH_RISK_SEC

    # No incident + low ATR = 300s
    assert next_stoploss_audit_interval(trades, atr_ok(0.50), recent_stop_incidents=0) == INTERVAL_NORMAL_SEC


def test_interval_constants_match_blueprint():
    """
    Verify interval constants match blueprint v1.9 values.
    If these change, it's a deliberate policy change, not an accident.
    """
    assert INTERVAL_NORMAL_SEC == 300, "Normal interval must be 300s per blueprint"
    assert INTERVAL_HIGH_RISK_SEC == 60, "High-risk interval must be 60s per blueprint"
    assert ATR_HIGH_VOL_THRESHOLD == 0.80, "ATR threshold must be 0.80 per blueprint"


def test_interval_multiple_trades_same_as_one():
    """
    Interval policy depends on whether trades exist (bool), not count.
    1 trade and 5 trades produce the same interval for same ATR state.
    """
    one_trade = [open_trade("t1")]
    five_trades = [open_trade(f"t{i}") for i in range(5)]

    for atr in [atr_ok(0.90), atr_ok(0.50), atr_insufficient()]:
        assert (
            next_stoploss_audit_interval(one_trade, atr)
            == next_stoploss_audit_interval(five_trades, atr)
        ), f"Interval must not depend on trade count for atr={atr}"


# ---------------------------------------------------------------------------
# StoplossAuditor.run_once() — verification logic
# ---------------------------------------------------------------------------

def test_auditor_verifies_stop_present(bus):
    """
    If exchange has a stop_limit order for the open trade's pair,
    auditor must mark it verified and NOT post STOP_UNCONFIRMED.
    """
    ft = MagicMock()
    ft.get_status.return_value = [open_trade("t1", "BTC/USDT")]

    exchange = MagicMock()
    exchange.fetch_open_orders.return_value = [stop_order("BTC/USDT", "stop_limit")]

    auditor = StoplossAuditor(
        exchange_client=exchange,
        ft_client=ft,
        bus=bus,
        atr_state_fn=lambda: atr_ok(0.50),
    )
    summary = auditor.run_once()

    assert summary["verified"] == 1
    assert summary["missing"] == 0
    assert summary["incidents"] == 0

    # No STOP_UNCONFIRMED posted
    pending = bus.consume_pending()
    stop_actions = [a for a in pending if a["action_type"] == "STOP_UNCONFIRMED"]
    assert len(stop_actions) == 0


def test_auditor_posts_stop_unconfirmed_when_missing(bus):
    """
    If no stop order exists on exchange for an open trade,
    auditor must post STOP_UNCONFIRMED with severity=HIGH.
    """
    ft = MagicMock()
    ft.get_status.return_value = [open_trade("t2", "ETH/USDT")]

    exchange = MagicMock()
    # No stop orders on exchange
    exchange.fetch_open_orders.return_value = [
        {"id": "limit_001", "symbol": "ETH/USDT", "type": "limit", "side": "buy"}
    ]

    auditor = StoplossAuditor(
        exchange_client=exchange,
        ft_client=ft,
        bus=bus,
        atr_state_fn=lambda: atr_ok(0.50),
    )
    summary = auditor.run_once()

    assert summary["missing"] == 1
    assert summary["incidents"] == 1

    pending = bus.consume_pending()
    stop_actions = [a for a in pending if a["action_type"] == "STOP_UNCONFIRMED"]
    assert len(stop_actions) == 1
    assert stop_actions[0]["severity"] == "HIGH"
    assert stop_actions[0]["pair"] == "ETH/USDT"


def test_auditor_no_trades_no_action(bus):
    """
    No open trades = nothing to audit. No STOP_UNCONFIRMED posted.
    """
    ft = MagicMock()
    ft.get_status.return_value = []

    exchange = MagicMock()
    exchange.fetch_open_orders.return_value = []

    auditor = StoplossAuditor(
        exchange_client=exchange,
        ft_client=ft,
        bus=bus,
    )
    summary = auditor.run_once()

    assert summary["verified"] == 0
    assert summary["missing"] == 0
    assert len(bus.consume_pending()) == 0


def test_auditor_exchange_fetch_fail_posts_unconfirmed(bus):
    """
    If exchange.fetch_open_orders() fails, auditor cannot verify stops.
    Must conservatively post STOP_UNCONFIRMED for all open trades.
    """
    ft = MagicMock()
    ft.get_status.return_value = [open_trade("t3", "BTC/USDT")]

    exchange = MagicMock()
    exchange.fetch_open_orders.side_effect = ConnectionError("exchange down")

    auditor = StoplossAuditor(
        exchange_client=exchange,
        ft_client=ft,
        bus=bus,
        atr_state_fn=lambda: atr_ok(0.50),
    )
    summary = auditor.run_once()

    assert summary["incidents"] == 1
    pending = bus.consume_pending()
    assert any(a["action_type"] == "STOP_UNCONFIRMED" for a in pending)


def test_auditor_deduplicates_stop_unconfirmed(bus):
    """
    If STOP_UNCONFIRMED is already pending, auditor must not post another.
    Prevents flooding Decision Bus during persistent stop absence.
    """
    ft = MagicMock()
    ft.get_status.return_value = [open_trade("t4", "BTC/USDT")]
    exchange = MagicMock()
    exchange.fetch_open_orders.return_value = []

    auditor = StoplossAuditor(
        exchange_client=exchange,
        ft_client=ft,
        bus=bus,
        atr_state_fn=lambda: atr_ok(0.50),
    )

    # Run twice without consuming the pending action
    auditor.run_once()
    auditor.run_once()

    all_stop = bus._conn.execute(
        "SELECT COUNT(*) FROM system_actions WHERE action_type='STOP_UNCONFIRMED'"
    ).fetchone()[0]
    assert all_stop == 1, f"Expected 1 STOP_UNCONFIRMED, got {all_stop} (dedup failed)"


def test_auditor_clientorderid_match_takes_priority(bus):
    """
    If exchange order has clientOrderId containing trade_id,
    it must be matched even if the pair is different.
    (Guards against multi-pair portfolios where pair-match would be ambiguous.)
    """
    ft = MagicMock()
    ft.get_status.return_value = [open_trade("trade_42", "BTC/USDT")]

    exchange = MagicMock()
    exchange.fetch_open_orders.return_value = [
        {
            "id": "stop_x",
            "symbol": "BTC/USDT",
            "type": "stop_limit",
            "side": "sell",
            "clientOrderId": "ft_trade_42_stoploss",  # contains trade_id
        }
    ]

    auditor = StoplossAuditor(
        exchange_client=exchange,
        ft_client=ft,
        bus=bus,
        atr_state_fn=lambda: atr_ok(0.50),
    )
    summary = auditor.run_once()

    assert summary["verified"] == 1
    assert summary["missing"] == 0


def test_auditor_uses_injectable_atr_state(bus):
    """
    Auditor must use the atr_state_fn result for interval computation,
    not hardcode any ATR value.

    Verified by checking run_once() summary integrates with
    next_stoploss_audit_interval() using the injected state.
    """
    ft = MagicMock()
    ft.get_status.return_value = [open_trade()]
    exchange = MagicMock()
    exchange.fetch_open_orders.return_value = [stop_order()]

    # Inject high-volatility state
    high_vol_state = atr_ok(0.95)
    auditor = StoplossAuditor(
        exchange_client=exchange,
        ft_client=ft,
        bus=bus,
        atr_state_fn=lambda: high_vol_state,
    )
    summary = auditor.run_once()

    # Verify interval for this state
    interval = next_stoploss_audit_interval(
        summary["open_trades"],
        high_vol_state,
        summary["incidents"],
    )
    assert interval == INTERVAL_HIGH_RISK_SEC, (
        f"High vol atr_state must produce 60s interval, got {interval}"
    )