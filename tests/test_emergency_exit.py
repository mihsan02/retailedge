"""
tests/test_emergency_exit.py — Sprint 4 S4-1

Required gate tests (Micro-live B1):
  - test_emergency_exit_market_success
  - test_emergency_exit_market_fail_limit_fallback
  - test_emergency_exit_all_fail_operator_alert

Additional coverage:
  - Policy gate: disabled by default
  - Bot liveness gate
  - _is_2xx edge cases
  - Sleep delays called with correct values
  - Decision Bus receives correct action type and fields
  - Aggressive limit price = best_bid * 0.98
  - Order book failure on limit fallback
  - forceexit raises exception (not just returns non-2xx)
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sidecar.guardian.emergency_exit import (
    AGGRESSIVE_LIMIT_DISCOUNT,
    STATUS_BOT_DOWN,
    STATUS_DISABLED,
    STATUS_DONE_LIMIT,
    STATUS_DONE_MARKET,
    STATUS_FAILED,
    EmergencyExitPolicy,
    Trade,
    _is_2xx,
    emergency_exit_cascade,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _policy(enabled: bool = True) -> EmergencyExitPolicy:
    return EmergencyExitPolicy(
        emergency_exit_enabled=enabled,
        aggressive_limit_discount=AGGRESSIVE_LIMIT_DISCOUNT,
        market_max_retries=2,
        retry_delays_sec=[0.5, 1.0],
    )


def _trade(trade_id: str = "t001", pair: str = "BTC/USDT") -> Trade:
    return Trade(id=trade_id, pair=pair)


def _ft_client(alive: bool = True, forceexit_responses: list[Any] = None):
    """
    Mock FtClient.
    forceexit_responses: list of return values per call (in order).
    Each value is either an int (status_code) or an object with .status_code.
    """
    client = MagicMock()
    client.ping_alive.return_value = alive

    class FakeResp:
        def __init__(self, code): self.status_code = code

    if forceexit_responses is not None:
        client.forceexit.side_effect = [
            FakeResp(r) if isinstance(r, int) else r
            for r in forceexit_responses
        ]
    return client


def _exchange_client(best_bid: float | None = 50000.0):
    client = MagicMock()
    if best_bid is None:
        client.fetch_order_book.side_effect = Exception("order book unavailable")
    else:
        client.fetch_order_book.return_value = {"bids": [[best_bid, 1.0]], "asks": [[best_bid + 10, 1.0]]}
    return client


def _decision_bus():
    bus = MagicMock()
    bus.post.return_value = "action_id_mock"
    return bus


def _no_sleep(seconds: float) -> None:
    """No-op sleep for tests."""
    pass


# ---------------------------------------------------------------------------
# REQUIRED GATE: test_emergency_exit_market_success
# ---------------------------------------------------------------------------

def test_emergency_exit_market_success():
    """
    REQUIRED GATE — Micro-live B1.
    Market order succeeds on first attempt (2xx).
    Must return DONE_MARKET. No Decision Bus post. No limit attempt.
    """
    ft = _ft_client(forceexit_responses=[200])
    ex = _exchange_client()
    bus = _decision_bus()

    result = emergency_exit_cascade(_trade(), ft, ex, bus, _policy(), sleep_fn=_no_sleep)

    assert result == STATUS_DONE_MARKET
    # forceexit called exactly once with market
    ft.forceexit.assert_called_once_with(trade_id="t001", ordertype="market")
    # No Decision Bus post on success
    bus.post.assert_not_called()
    # Order book never queried
    ex.fetch_order_book.assert_not_called()


def test_emergency_exit_market_success_on_second_attempt():
    """Market fails first, succeeds on second attempt. DONE_MARKET, no limit fallback."""
    ft = _ft_client(forceexit_responses=[500, 200])
    ex = _exchange_client()
    bus = _decision_bus()

    result = emergency_exit_cascade(_trade(), ft, ex, bus, _policy(), sleep_fn=_no_sleep)

    assert result == STATUS_DONE_MARKET
    assert ft.forceexit.call_count == 2
    bus.post.assert_not_called()
    ex.fetch_order_book.assert_not_called()


# ---------------------------------------------------------------------------
# REQUIRED GATE: test_emergency_exit_market_fail_limit_fallback
# ---------------------------------------------------------------------------

def test_emergency_exit_market_fail_limit_fallback():
    """
    REQUIRED GATE — Micro-live B1.
    Both market attempts fail, aggressive limit succeeds (2xx).
    Must return DONE_AGGRESSIVE_LIMIT.
    Limit price must be best_bid * 0.98.
    """
    best_bid = 50000.0
    ft = _ft_client(forceexit_responses=[500, 500, 200])
    ex = _exchange_client(best_bid=best_bid)
    bus = _decision_bus()

    result = emergency_exit_cascade(_trade(), ft, ex, bus, _policy(), sleep_fn=_no_sleep)

    assert result == STATUS_DONE_LIMIT
    # Two market attempts + one limit attempt
    assert ft.forceexit.call_count == 3
    # Verify limit call used correct price
    limit_call = ft.forceexit.call_args_list[2]
    assert limit_call.kwargs["ordertype"] == "limit"
    expected_price = best_bid * AGGRESSIVE_LIMIT_DISCOUNT
    assert abs(limit_call.kwargs["price"] - expected_price) < 1e-6, (
        f"Expected price {expected_price}, got {limit_call.kwargs['price']}"
    )
    # No failure posted to Decision Bus
    bus.post.assert_not_called()


def test_aggressive_limit_price_uses_best_bid_discount():
    """Explicit price calculation check for different bid values."""
    best_bid = 38247.50
    ft = _ft_client(forceexit_responses=[503, 503, 200])
    ex = _exchange_client(best_bid=best_bid)
    bus = _decision_bus()

    emergency_exit_cascade(_trade(), ft, ex, bus, _policy(), sleep_fn=_no_sleep)

    limit_call = ft.forceexit.call_args_list[2]
    expected = best_bid * 0.98
    assert abs(limit_call.kwargs["price"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# REQUIRED GATE: test_emergency_exit_all_fail_operator_alert
# ---------------------------------------------------------------------------

def test_emergency_exit_all_fail_operator_alert():
    """
    REQUIRED GATE — Micro-live B1.
    Both market attempts and aggressive limit all fail.
    Must return FAILED_OPERATOR_REQUIRED.
    Must post EMERGENCY_EXIT_FAILED to Decision Bus with CRITICAL severity.
    """
    ft = _ft_client(forceexit_responses=[500, 500, 400])
    ex = _exchange_client(best_bid=50000.0)
    bus = _decision_bus()

    result = emergency_exit_cascade(_trade(), ft, ex, bus, _policy(), sleep_fn=_no_sleep)

    assert result == STATUS_FAILED
    # All 3 attempts made
    assert ft.forceexit.call_count == 3
    # Decision Bus must receive EMERGENCY_EXIT_FAILED
    bus.post.assert_called_once()
    call_kwargs = bus.post.call_args
    assert call_kwargs.args[0] == "EMERGENCY_EXIT_FAILED"
    assert call_kwargs.kwargs.get("severity") == "CRITICAL"
    assert call_kwargs.kwargs.get("trade_id") == "t001"
    assert call_kwargs.kwargs.get("pair") == "BTC/USDT"


def test_all_fail_operator_alert_order_book_unavailable():
    """
    All market fail + order book fetch raises exception.
    Must still reach FAILED_OPERATOR_REQUIRED and post to Decision Bus.
    """
    ft = _ft_client(forceexit_responses=[500, 500])
    ex = _exchange_client(best_bid=None)  # raises exception
    bus = _decision_bus()

    result = emergency_exit_cascade(_trade(), ft, ex, bus, _policy(), sleep_fn=_no_sleep)

    assert result == STATUS_FAILED
    assert ft.forceexit.call_count == 2  # only market attempts; limit skipped
    bus.post.assert_called_once()
    assert bus.post.call_args.args[0] == "EMERGENCY_EXIT_FAILED"


# ---------------------------------------------------------------------------
# Policy guard tests
# ---------------------------------------------------------------------------

def test_policy_disabled_by_default():
    """
    emergency_exit_enabled=False must return DISABLED immediately.
    No REST calls. No Decision Bus post.
    """
    ft = _ft_client()
    ex = _exchange_client()
    bus = _decision_bus()
    disabled_policy = _policy(enabled=False)

    result = emergency_exit_cascade(_trade(), ft, ex, bus, disabled_policy, sleep_fn=_no_sleep)

    assert result == STATUS_DISABLED
    ft.forceexit.assert_not_called()
    ft.ping_alive.assert_not_called()
    bus.post.assert_not_called()


def test_bot_down_returns_bot_down_status():
    """
    Bot not reachable (ping_alive=False).
    Must return BOT_DOWN_NO_REST_EXIT.
    Must NOT attempt forceexit.
    Must post EMERGENCY_EXIT_FAILED (bot down = operator must intervene).
    """
    ft = _ft_client(alive=False)
    ex = _exchange_client()
    bus = _decision_bus()

    result = emergency_exit_cascade(_trade(), ft, ex, bus, _policy(), sleep_fn=_no_sleep)

    assert result == STATUS_BOT_DOWN
    ft.forceexit.assert_not_called()
    bus.post.assert_called_once()
    assert bus.post.call_args.args[0] == "EMERGENCY_EXIT_FAILED"
    assert bus.post.call_args.kwargs.get("reason") == "BOT_DOWN_NO_REST_EXIT"


# ---------------------------------------------------------------------------
# Sleep delay verification
# ---------------------------------------------------------------------------

def test_sleep_delays_called_correctly():
    """
    Verify sleep is called with correct delays in cascade sequence.
    market fail → sleep(0.5) → market fail → sleep(1.0) → limit success
    """
    sleep_calls = []

    def recording_sleep(s: float) -> None:
        sleep_calls.append(s)

    ft = _ft_client(forceexit_responses=[500, 500, 200])
    ex = _exchange_client()
    bus = _decision_bus()

    emergency_exit_cascade(_trade(), ft, ex, bus, _policy(), sleep_fn=recording_sleep)

    assert sleep_calls == [0.5, 1.0], (
        f"Expected delays [0.5, 1.0], got {sleep_calls}"
    )


def test_market_success_no_sleep():
    """Market succeeds on first attempt — no sleep called."""
    sleep_calls = []
    ft = _ft_client(forceexit_responses=[200])
    ex = _exchange_client()
    bus = _decision_bus()

    emergency_exit_cascade(
        _trade(), ft, ex, bus, _policy(),
        sleep_fn=lambda s: sleep_calls.append(s),
    )

    assert sleep_calls == [], f"No sleep expected on first-attempt success, got {sleep_calls}"


def test_market_second_attempt_sleep_only_once():
    """Market fails once, succeeds second — only one sleep(0.5)."""
    sleep_calls = []
    ft = _ft_client(forceexit_responses=[500, 200])
    ex = _exchange_client()
    bus = _decision_bus()

    emergency_exit_cascade(
        _trade(), ft, ex, bus, _policy(),
        sleep_fn=lambda s: sleep_calls.append(s),
    )

    assert sleep_calls == [0.5]


# ---------------------------------------------------------------------------
# forceexit raises exception (not just non-2xx response)
# ---------------------------------------------------------------------------

def test_forceexit_raises_exception_treated_as_failure():
    """
    If forceexit raises an exception (network error), it must be caught
    and treated as a failure — not propagated.
    """
    ft = _ft_client()
    ft.forceexit.side_effect = [
        ConnectionError("connection refused"),
        ConnectionError("connection refused"),
        ConnectionError("connection refused"),
    ]
    ex = _exchange_client()
    bus = _decision_bus()

    # Must not raise — exception must be caught internally
    result = emergency_exit_cascade(_trade(), ft, ex, bus, _policy(), sleep_fn=_no_sleep)
    assert result == STATUS_FAILED
    bus.post.assert_called_once()


def test_forceexit_exception_then_success():
    """Exception on first market attempt, 2xx on second. Must return DONE_MARKET."""
    ft = _ft_client()

    class FakeResp:
        status_code = 200

    ft.forceexit.side_effect = [ConnectionError("timeout"), FakeResp()]
    ex = _exchange_client()
    bus = _decision_bus()

    result = emergency_exit_cascade(_trade(), ft, ex, bus, _policy(), sleep_fn=_no_sleep)
    assert result == STATUS_DONE_MARKET
    bus.post.assert_not_called()


# ---------------------------------------------------------------------------
# _is_2xx edge cases
# ---------------------------------------------------------------------------

def test_is_2xx_accepts_int():
    assert _is_2xx(200) is True
    assert _is_2xx(201) is True
    assert _is_2xx(299) is True
    assert _is_2xx(300) is False
    assert _is_2xx(500) is False
    assert _is_2xx(400) is False


def test_is_2xx_accepts_object_with_status_code():
    class R:
        def __init__(self, c): self.status_code = c
    assert _is_2xx(R(200)) is True
    assert _is_2xx(R(404)) is False


def test_is_2xx_rejects_none():
    assert _is_2xx(None) is False


def test_is_2xx_rejects_dict_without_explicit_2xx():
    # Dict with no status_code field — not sufficient
    assert _is_2xx({"status": "ok"}) is False


def test_is_2xx_accepts_dict_with_status_code():
    assert _is_2xx({"status_code": 200}) is True
    assert _is_2xx({"status_code": 503}) is False


# ---------------------------------------------------------------------------
# Decision Bus action type is canonical
# ---------------------------------------------------------------------------

def test_decision_bus_action_type_is_canonical():
    """
    EMERGENCY_EXIT_FAILED must match the canonical action type from CLAUDE.md.
    Any typo here would create an orphan action with no consumer.
    """
    ft = _ft_client(forceexit_responses=[500, 500, 500])
    ex = _exchange_client()
    bus = _decision_bus()

    emergency_exit_cascade(_trade(), ft, ex, bus, _policy(), sleep_fn=_no_sleep)

    posted_type = bus.post.call_args.args[0]
    assert posted_type == "EMERGENCY_EXIT_FAILED", (
        f"Action type must be exactly 'EMERGENCY_EXIT_FAILED', got {posted_type!r}"
    )