"""
tests/test_pre_trade_gate.py

Test gate: Dry-run.

Done criteria:
- Entry blocked if projected_available < proposed_stake * 1.05
- Entry passes if balance is sufficient
- Min notional check blocks undersized stakes
"""

import pytest
from sidecar.reconciler.pre_trade_gate import (
    pre_entry_balance_check,
    apply_regime_multiplier,
    BALANCE_BUFFER,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_bus_spy():
    """Capture Decision Bus posts without a real DB."""
    calls = []
    def post_fn(**kwargs):
        calls.append(kwargs)
    return calls, post_fn


# ---------------------------------------------------------------------------
# Core done criteria — balance buffer
# ---------------------------------------------------------------------------

def test_entry_blocked_when_projected_below_required():
    """
    projected_available < proposed_stake * 1.05 must block entry.

    This is the primary done criteria for S2-6.

    Example:
        wallet=1000, reserved=900 -> projected=100
        stake=100, required=100*1.05=105
        100 < 105 -> BLOCK
    """
    calls, post_fn = make_bus_spy()

    result = pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=100.0,
        wallet_available=1000.0,
        reserved_total=900.0,      # projected = 100.0
        decision_bus_post_fn=post_fn,
    )

    assert result is False, "Must return False when projected < stake * 1.05"
    assert len(calls) == 1, "Must post REJECT_ENTRY to Decision Bus"
    assert calls[0]["action_type"] == "REJECT_ENTRY"
    assert calls[0]["pair"] == "BTC/USDT"


def test_entry_passes_when_balance_sufficient():
    """
    projected_available >= proposed_stake * 1.05 must allow entry.

    Example:
        wallet=1000, reserved=800 -> projected=200
        stake=100, required=105
        200 >= 105 -> PASS
    """
    calls, post_fn = make_bus_spy()

    result = pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=100.0,
        wallet_available=1000.0,
        reserved_total=800.0,      # projected = 200.0
        decision_bus_post_fn=post_fn,
    )

    assert result is True, "Must return True when projected >= stake * 1.05"
    assert len(calls) == 0, "Must NOT post REJECT_ENTRY on pass"


def test_entry_blocked_at_exact_buffer_boundary():
    """
    projected_available == proposed_stake * 1.05 must PASS (>= not >).
    projected_available == proposed_stake * 1.05 - epsilon must BLOCK.
    """
    stake = 100.0
    required = stake * BALANCE_BUFFER  # 105.0

    # Exactly at boundary — must pass
    result_at = pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=stake,
        wallet_available=required,
        reserved_total=0.0,
    )
    assert result_at is True, f"projected == required ({required}) must PASS"

    # Just below boundary — must block
    result_below = pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=stake,
        wallet_available=required - 0.01,
        reserved_total=0.0,
    )
    assert result_below is False, f"projected < required must BLOCK"


def test_entry_blocked_when_fully_reserved():
    """
    If all wallet balance is reserved, projected = 0 -> always block.
    """
    calls, post_fn = make_bus_spy()
    result = pre_entry_balance_check(
        pair="ETH/USDT",
        proposed_stake=50.0,
        wallet_available=500.0,
        reserved_total=500.0,  # fully reserved
        decision_bus_post_fn=post_fn,
    )
    assert result is False
    assert calls[0]["action_type"] == "REJECT_ENTRY"


def test_no_reserved_uses_full_wallet():
    """
    With no reserved funds (fresh start), projected = wallet_available.
    """
    result = pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=100.0,
        wallet_available=200.0,
        reserved_total=0.0,
    )
    assert result is True  # 200 >= 100 * 1.05 = 105


# ---------------------------------------------------------------------------
# Min notional
# ---------------------------------------------------------------------------

def test_entry_blocked_below_min_notional():
    """
    proposed_stake < min_notional must block even if balance is sufficient.
    """
    calls, post_fn = make_bus_spy()
    result = pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=5.0,        # below min
        wallet_available=10000.0,
        reserved_total=0.0,
        decision_bus_post_fn=post_fn,
        min_notional=10.0,
    )
    assert result is False
    assert calls[0]["action_type"] == "REJECT_ENTRY"
    assert "min_notional" in calls[0]["reason"]


def test_entry_passes_at_min_notional():
    """proposed_stake == min_notional must pass the notional check."""
    result = pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=10.0,
        wallet_available=10000.0,
        reserved_total=0.0,
        min_notional=10.0,
    )
    assert result is True


def test_min_notional_zero_skips_check():
    """min_notional=0 disables the check (default)."""
    result = pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=0.01,       # tiny stake
        wallet_available=10000.0,
        reserved_total=0.0,
        min_notional=0.0,          # disabled
    )
    # Only balance check applies — 0.01 * 1.05 = 0.0105 < 10000
    assert result is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_zero_stake_blocked():
    """proposed_stake <= 0 must always block."""
    calls, post_fn = make_bus_spy()
    result = pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=0.0,
        wallet_available=10000.0,
        reserved_total=0.0,
        decision_bus_post_fn=post_fn,
    )
    assert result is False
    assert len(calls) == 1


def test_negative_stake_blocked():
    calls, post_fn = make_bus_spy()
    result = pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=-50.0,
        wallet_available=10000.0,
        reserved_total=0.0,
        decision_bus_post_fn=post_fn,
    )
    assert result is False


def test_no_bus_fn_does_not_raise():
    """If decision_bus_post_fn is None, rejection is logged only — no exception."""
    result = pre_entry_balance_check(
        pair="BTC/USDT",
        proposed_stake=500.0,
        wallet_available=100.0,    # insufficient
        reserved_total=0.0,
        decision_bus_post_fn=None, # no bus
    )
    assert result is False  # still blocks, just doesn't post


# ---------------------------------------------------------------------------
# Regime multiplier helper
# ---------------------------------------------------------------------------

def test_regime_multiplier_scales_stake():
    """apply_regime_multiplier reduces stake for ranging regime."""
    result = apply_regime_multiplier(base_stake=100.0, regime_multiplier=0.5)
    assert abs(result - 50.0) < 0.001


def test_regime_multiplier_clamped_at_max():
    """Multiplier above max_multiplier is clamped."""
    result = apply_regime_multiplier(base_stake=100.0, regime_multiplier=2.0, max_multiplier=1.5)
    assert abs(result - 150.0) < 0.001


def test_regime_multiplier_clamped_at_min():
    """Multiplier below min_multiplier is clamped."""
    result = apply_regime_multiplier(base_stake=100.0, regime_multiplier=-1.0, min_multiplier=0.0)
    assert result == 0.0