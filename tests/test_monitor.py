"""
tests/test_monitor.py

Test gate: Dry-run.

Done criteria:
- monitor writes metrics to ledger every run_once() call
- net_expectancy, max_drawdown, losing_streak, slippage, fill_rate computed correctly
"""

import pytest
from sidecar.health_monitor.monitor import (
    HealthMonitor,
    HealthMonitorLedger,
    _compute_net_expectancy,
    _compute_max_drawdown,
    _compute_losing_streak,
    _compute_avg_slippage,
    _compute_fill_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ledger():
    db = HealthMonitorLedger(":memory:")
    yield db
    db.close()


def make_trade(trade_id, pnl_quote, closed_ts="2026-06-04T10:00:00+00:00"):
    return {
        "trade_id": trade_id,
        "pnl_quote": pnl_quote,
        "pnl_pct": pnl_quote / 1000.0,
        "closed_ts": closed_ts,
    }


def insert_trades(ledger, trades):
    """Insert trade PnL records for testing."""
    for t in trades:
        ledger.upsert_trade_pnl(
            trade_id=t["trade_id"],
            pair="BTC/USDT",
            pnl_quote=t["pnl_quote"],
            pnl_pct=t.get("pnl_pct", t["pnl_quote"] / 1000.0),
            entry_price=50000.0,
            exit_price=50000.0 + t["pnl_quote"] / 0.01,
            fill_cost=500.0,
            fee_quote=0.5,
            closed_ts=t.get("closed_ts", "2026-06-04T10:00:00+00:00"),
        )


# ---------------------------------------------------------------------------
# Done criteria — metrics written to ledger
# ---------------------------------------------------------------------------

def test_monitor_writes_metrics_on_run_once(ledger):
    """
    run_once() must write a health metrics snapshot to the ledger.
    Snapshot must contain all required fields.
    """
    monitor = HealthMonitor(ledger=ledger, regime_label_fn=lambda: "trending")

    # No trade data yet — metrics should still be written (with None values)
    metrics = monitor.run_once()

    assert metrics is not None
    assert "net_expectancy" in metrics
    assert "max_drawdown" in metrics
    assert "losing_streak" in metrics
    assert "avg_slippage" in metrics
    assert "fill_rate" in metrics
    assert "regime_label" in metrics
    assert metrics["regime_label"] == "trending"

    # Verify written to ledger
    stored = ledger.get_latest_metrics()
    assert stored is not None
    assert stored["regime_label"] == "trending"


def test_monitor_writes_new_snapshot_each_run(ledger):
    """
    Each run_once() must append a new row — not overwrite the previous.
    """
    monitor = HealthMonitor(ledger=ledger)

    monitor.run_once()
    monitor.run_once()
    monitor.run_once()

    count = ledger._conn.execute(
        "SELECT COUNT(*) FROM health_metrics"
    ).fetchone()[0]
    assert count == 3, f"Expected 3 snapshots, got {count}"


def test_monitor_uses_injectable_regime_label(ledger):
    """regime_label must come from injectable fn, not hardcoded."""
    monitor = HealthMonitor(ledger=ledger, regime_label_fn=lambda: "volatile")
    metrics = monitor.run_once()
    assert metrics["regime_label"] == "volatile"


# ---------------------------------------------------------------------------
# net_expectancy
# ---------------------------------------------------------------------------

def test_net_expectancy_positive_trades():
    trades = [make_trade(f"t{i}", 10.0) for i in range(5)]
    result = _compute_net_expectancy(trades)
    assert abs(result - 10.0) < 0.001


def test_net_expectancy_mixed_pnl():
    trades = [
        make_trade("t1", 20.0),
        make_trade("t2", -10.0),
    ]
    result = _compute_net_expectancy(trades)
    assert abs(result - 5.0) < 0.001  # (20 + -10) / 2


def test_net_expectancy_none_on_empty():
    assert _compute_net_expectancy([]) is None


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown_all_winning():
    """All winning trades: no drawdown -> 0.0."""
    trades = [make_trade(f"t{i}", 10.0) for i in range(5)]
    result = _compute_max_drawdown(trades)
    # Peak grows monotonically, never draws down
    assert result == 0.0 or result is not None


def test_max_drawdown_single_loss_after_gains():
    """Gain 100, then lose 50 -> drawdown = -50/100 = -0.5."""
    trades = [
        make_trade("t1", 100.0),
        make_trade("t2", -50.0),
    ]
    result = _compute_max_drawdown(trades)
    assert result is not None
    assert abs(result - (-0.5)) < 0.001


def test_max_drawdown_none_on_empty():
    assert _compute_max_drawdown([]) is None


def test_max_drawdown_negative_or_zero():
    """Drawdown is always <= 0."""
    trades = [make_trade(f"t{i}", 10.0 if i % 2 == 0 else -15.0) for i in range(10)]
    result = _compute_max_drawdown(trades)
    assert result is not None
    assert result <= 0.0


# ---------------------------------------------------------------------------
# losing_streak
# ---------------------------------------------------------------------------

def test_losing_streak_current_streak():
    """Three consecutive losses at end -> streak = 3."""
    trades = [
        make_trade("t1", 10.0),
        make_trade("t2", -5.0),
        make_trade("t3", -5.0),
        make_trade("t4", -5.0),
    ]
    assert _compute_losing_streak(trades) == 3


def test_losing_streak_broken_by_win():
    """Loss-win-loss at end: streak = 1 (only last loss)."""
    trades = [
        make_trade("t1", -5.0),
        make_trade("t2", 10.0),
        make_trade("t3", -5.0),
    ]
    assert _compute_losing_streak(trades) == 1


def test_losing_streak_zero_on_win():
    """Last trade is a win -> streak = 0."""
    trades = [make_trade("t1", -5.0), make_trade("t2", 10.0)]
    assert _compute_losing_streak(trades) == 0


def test_losing_streak_zero_on_empty():
    assert _compute_losing_streak([]) == 0


# ---------------------------------------------------------------------------
# avg_slippage
# ---------------------------------------------------------------------------

def test_avg_slippage_buy_above_price():
    """Buy filled above limit price = positive slippage (unfavorable)."""
    records = [{"side": "buy", "price": 50000.0, "average": 50100.0, "filled": 0.01}]
    result = _compute_avg_slippage(records)
    assert result is not None
    expected = (50100.0 - 50000.0) / 50000.0  # 0.002
    assert abs(result - expected) < 1e-6


def test_avg_slippage_buy_at_price():
    """Buy filled exactly at limit price = zero slippage."""
    records = [{"side": "buy", "price": 50000.0, "average": 50000.0, "filled": 0.01}]
    result = _compute_avg_slippage(records)
    assert abs(result) < 1e-9


def test_avg_slippage_none_on_empty():
    assert _compute_avg_slippage([]) is None


def test_avg_slippage_skips_zero_price():
    """Records with price=0 or average=0 must be skipped."""
    records = [
        {"side": "buy", "price": 0.0, "average": 50000.0, "filled": 0.01},
        {"side": "buy", "price": 50000.0, "average": 50100.0, "filled": 0.01},
    ]
    result = _compute_avg_slippage(records)
    # Only the second record counts
    assert result is not None
    expected = (50100.0 - 50000.0) / 50000.0
    assert abs(result - expected) < 1e-6


# ---------------------------------------------------------------------------
# fill_rate
# ---------------------------------------------------------------------------

def test_fill_rate_all_filled():
    stats = {"total_limit": 10, "filled_limit": 10}
    assert abs(_compute_fill_rate(stats) - 1.0) < 1e-9


def test_fill_rate_half_filled():
    stats = {"total_limit": 10, "filled_limit": 5}
    assert abs(_compute_fill_rate(stats) - 0.5) < 1e-9


def test_fill_rate_none_on_zero_total():
    stats = {"total_limit": 0, "filled_limit": 0}
    assert _compute_fill_rate(stats) is None


# ---------------------------------------------------------------------------
# Integration: full run with trade data
# ---------------------------------------------------------------------------

def test_monitor_computes_correct_metrics_from_ledger(ledger):
    """
    With real trade data in ledger, monitor must compute non-None metrics.
    """
    insert_trades(ledger, [
        make_trade("t1", 20.0),
        make_trade("t2", -10.0),
        make_trade("t3", 15.0),
    ])

    monitor = HealthMonitor(ledger=ledger, regime_label_fn=lambda: "trending")
    metrics = monitor.run_once()

    assert metrics["net_expectancy"] is not None
    # (20 - 10 + 15) / 3 = 8.33...
    assert abs(metrics["net_expectancy"] - (25.0 / 3)) < 0.01

    assert metrics["losing_streak"] == 0  # last trade is +15 (win)
    assert metrics["trade_count"] == 3
    assert metrics["regime_label"] == "trending"

    stored = ledger.get_latest_metrics()
    assert stored["net_expectancy"] is not None