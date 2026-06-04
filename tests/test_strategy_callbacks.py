"""
tests/test_strategy_callbacks.py

Test gate: Dry-run.

Tests the Pre-Trade Gate logic in RetailEdgeStrategy without
requiring a running Freqtrade instance.

Strategy imports freqtrade which may not be installed in the test environment.
Tests are structured to test the gate logic directly via the inline functions,
and test confirm_trade_entry via a mock strategy instance.

Done criteria:
- confirm_trade_entry returns False when gate fails (insufficient balance)
- confirm_trade_entry returns True when gate passes (sufficient balance)
"""

import pytest
import os
import sys
import types
import sqlite3
import tempfile
import importlib.util
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Bootstrap freqtrade stub + load strategy via importlib
# ---------------------------------------------------------------------------
# Strategy file lives inside freqtrade/user_data/strategies/ which is not a
# Python package in the test environment. Load it directly via importlib.util
# to avoid package resolution issues.

def _bootstrap_freqtrade_stub():
    for pkg in ["freqtrade", "freqtrade.strategy", "freqtrade.persistence"]:
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            m.__package__ = pkg
            sys.modules[pkg] = m

    class IStrategyStub:
        pass

    sys.modules["freqtrade.strategy"].IStrategy = IStrategyStub
    sys.modules["freqtrade.strategy"].stoploss_from_open = lambda *a, **kw: 0.0
    sys.modules["freqtrade"].strategy = sys.modules["freqtrade.strategy"]


def _load_strategy_module():
    _bootstrap_freqtrade_stub()
    strategy_file = os.path.join(
        os.path.dirname(__file__), "..",
        "freqtrade", "user_data", "strategies", "RetailEdgeStrategy.py"
    )
    spec = importlib.util.spec_from_file_location("RetailEdgeStrategy", strategy_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_strategy_mod = _load_strategy_module()
_inline_pre_entry_balance_check = _strategy_mod._inline_pre_entry_balance_check
_read_wallet_state = _strategy_mod._read_wallet_state
RetailEdgeStrategy = _strategy_mod.RetailEdgeStrategy
BALANCE_BUFFER = _strategy_mod.BALANCE_BUFFER
DEFAULT_MIN_NOTIONAL = _strategy_mod.DEFAULT_MIN_NOTIONAL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_ledger_db():
    """Create a temporary SQLite DB with reserved_funds table."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE reserved_funds (
            order_id TEXT PRIMARY KEY,
            pair TEXT NOT NULL,
            reserved_quote REAL NOT NULL DEFAULT 0.0,
            updated_ts TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

    yield db_path

    os.unlink(db_path)


# ---------------------------------------------------------------------------
# _inline_pre_entry_balance_check — core gate logic tests
# ---------------------------------------------------------------------------

def test_gate_blocks_when_projected_insufficient():
    """
    Core done criteria: projected_available < proposed_stake * 1.05 -> False.
    wallet=1000, reserved=900, stake=100 -> projected=100, required=105 -> BLOCK.
    """
    result = _inline_pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=100.0,
        wallet_available=1000.0,
        reserved_total=900.0,
    )
    assert result is False, (
        "Gate must block when projected_available (100) < proposed_stake * 1.05 (105)"
    )


def test_gate_passes_when_balance_sufficient():
    """
    Core done criteria: projected_available >= proposed_stake * 1.05 -> True.
    wallet=1000, reserved=800, stake=100 -> projected=200, required=105 -> PASS.
    """
    result = _inline_pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=100.0,
        wallet_available=1000.0,
        reserved_total=800.0,
    )
    assert result is True, (
        "Gate must pass when projected_available (200) >= proposed_stake * 1.05 (105)"
    )


def test_gate_boundary_exactly_at_required():
    """projected == required (exact boundary) must PASS."""
    stake = 100.0
    required = stake * BALANCE_BUFFER  # 105.0
    result = _inline_pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=stake,
        wallet_available=required,
        reserved_total=0.0,
    )
    assert result is True, f"Exactly at buffer boundary ({required}) must PASS"


def test_gate_blocks_below_boundary():
    """projected < required by 1 cent must BLOCK."""
    stake = 100.0
    required = stake * BALANCE_BUFFER  # 105.0
    result = _inline_pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=stake,
        wallet_available=required - 0.01,
        reserved_total=0.0,
    )
    assert result is False


def test_gate_blocks_min_notional():
    """Stake below min_notional must block even with sufficient balance."""
    result = _inline_pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=5.0,
        wallet_available=100000.0,
        reserved_total=0.0,
        min_notional=10.0,
    )
    assert result is False


def test_gate_passes_at_min_notional():
    result = _inline_pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=10.0,
        wallet_available=100000.0,
        reserved_total=0.0,
        min_notional=10.0,
    )
    assert result is True


def test_gate_blocks_zero_stake():
    result = _inline_pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=0.0,
        wallet_available=100000.0,
        reserved_total=0.0,
    )
    assert result is False


def test_gate_fully_reserved_blocks():
    """All balance reserved -> projected=0 -> always block."""
    result = _inline_pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=50.0,
        wallet_available=500.0,
        reserved_total=500.0,
    )
    assert result is False


# ---------------------------------------------------------------------------
# _read_wallet_state — ledger integration
# ---------------------------------------------------------------------------

def test_read_wallet_state_empty_db_returns_defaults(temp_ledger_db, monkeypatch):
    """Empty reserved_funds table -> reserved_total=0, wallet=999999 (dry-run)."""
    monkeypatch.setenv("LEDGER_DB_PATH", temp_ledger_db)
    wallet, reserved = _read_wallet_state()
    assert reserved == 0.0
    assert wallet == 999999.0  # dry-run sentinel


def test_read_wallet_state_reads_reserved_funds(temp_ledger_db, monkeypatch):
    """Reserved funds in DB must be reflected in reserved_total."""
    monkeypatch.setenv("LEDGER_DB_PATH", temp_ledger_db)

    conn = sqlite3.connect(temp_ledger_db)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO reserved_funds (order_id, pair, reserved_quote, updated_ts) VALUES (?,?,?,?)",
        ("o1", "BTC/USDT", 500.0, now)
    )
    conn.execute(
        "INSERT INTO reserved_funds (order_id, pair, reserved_quote, updated_ts) VALUES (?,?,?,?)",
        ("o2", "ETH/USDT", 200.0, now)
    )
    conn.commit()
    conn.close()

    wallet, reserved = _read_wallet_state()
    assert abs(reserved - 700.0) < 0.001


def test_read_wallet_state_raises_on_missing_db(monkeypatch):
    """Missing DB file must raise, not silently return defaults."""
    monkeypatch.setenv("LEDGER_DB_PATH", "/nonexistent/path/retailedge.db")
    with pytest.raises(Exception):
        _read_wallet_state()


# ---------------------------------------------------------------------------
# confirm_trade_entry — via mock strategy (no Freqtrade runtime needed)
# ---------------------------------------------------------------------------

def _make_mock_strategy():
    """Create strategy instance without Freqtrade runtime."""
    return object.__new__(RetailEdgeStrategy)


def test_confirm_trade_entry_blocks_on_gate_fail(temp_ledger_db, monkeypatch):
    """
    confirm_trade_entry must return False when Pre-Trade Gate fails.
    Setup: reserve almost all balance, leaving insufficient for the trade.
    """
    monkeypatch.setenv("LEDGER_DB_PATH", temp_ledger_db)
    monkeypatch.setenv("MIN_NOTIONAL_USDT", "10.0")

    with patch.object(_strategy_mod, "_read_wallet_state", return_value=(1000.0, 990.0)):
        strategy = _make_mock_strategy()
        result = strategy.confirm_trade_entry(
            pair="BTC/USDT",
            order_type="limit",
            amount=0.002,
            rate=50000.0,
            time_in_force="GTC",
            current_time=datetime.now(timezone.utc),
            entry_tag=None,
            side="long",
        )

    assert result is False, (
        "confirm_trade_entry must return False when projected balance < stake * 1.05"
    )


def test_confirm_trade_entry_passes_on_sufficient_balance(temp_ledger_db, monkeypatch):
    """
    confirm_trade_entry must return True when Pre-Trade Gate passes.
    """
    monkeypatch.setenv("LEDGER_DB_PATH", temp_ledger_db)
    monkeypatch.setenv("MIN_NOTIONAL_USDT", "10.0")

    with patch.object(_strategy_mod, "_read_wallet_state", return_value=(10000.0, 0.0)):
        strategy = _make_mock_strategy()
        result = strategy.confirm_trade_entry(
            pair="BTC/USDT",
            order_type="limit",
            amount=0.002,
            rate=50000.0,
            time_in_force="GTC",
            current_time=datetime.now(timezone.utc),
            entry_tag=None,
            side="long",
        )

    assert result is True, (
        "confirm_trade_entry must return True when balance is sufficient"
    )


def test_confirm_trade_entry_fail_open_when_ledger_unreachable(monkeypatch):
    """
    If ledger is unreachable, confirm_trade_entry must fail-open (return True).
    """
    monkeypatch.setenv("LEDGER_DB_PATH", "/nonexistent/path.db")

    strategy = _make_mock_strategy()
    result = strategy.confirm_trade_entry(
        pair="BTC/USDT",
        order_type="limit",
        amount=0.002,
        rate=50000.0,
        time_in_force="GTC",
        current_time=datetime.now(timezone.utc),
        entry_tag=None,
        side="long",
    )

    assert result is True, (
        "confirm_trade_entry must fail-open (True) when ledger is unreachable in dry-run"
    )