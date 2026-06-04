"""
tests/test_capability_matrix.py

Test gate: Dry-run (all three required before Stage A pass).

Covers:
- test_exchange_capability_matrix_compile_fail_post_only
- test_exchange_capability_matrix_compile_fail_stoploss_type
- test_exchange_capability_matrix_compile_fail_market_exit

Design: all tests use minimal in-memory dicts — no file I/O, no env vars.
Each test constructs the exact failing condition and asserts ValueError is raised.
Tests do NOT share state. Fixture isolation is explicit per test.

Why ValueError and not a custom exception: ValueError is the contract defined in
CLAUDE.md. Changing it would break any CI pipeline that wraps the compiler.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deployment.config_compiler import validate_full_capability


# ---------------------------------------------------------------------------
# Shared fixtures — minimal valid base that each test mutates precisely
# ---------------------------------------------------------------------------

FREQTRADE_VERSION = "2026.3_freqai"

def _base_matrix() -> dict:
    """
    Capability matrix entry that passes ALL checks.
    Individual tests flip exactly one field to force the target failure.
    """
    return {
        "binance_spot": {
            "freqtrade_min_version": "2026.1",
            "trading_mode": "spot",
            "post_only_supported": True,
            "market_order_supported": True,
            "limit_order_supported": True,
            "stoploss_on_exchange_supported": True,
            "stoploss_order_types_supported": ["stop_limit"],
            "stoploss_api_versions_supported": ["spot_stop_order_v1"],
            "conditional_order_mode_supported": ["native_spot_stop_limit"],
        }
    }


def _base_venue_cfg() -> dict:
    """
    Venue config that passes ALL checks.
    Individual tests add/mutate fields to trigger the target failure.
    """
    return {
        "exchange_key": "binance_spot",
        "maker_first": True,
        "stoploss_on_exchange_supported": True,
        "stoploss_api_version": "spot_stop_order_v1",
        "stoploss_order_type": "stop_limit",
        "conditional_order_mode": "native_spot_stop_limit",
        "emergency_exit_enabled": False,
    }


# ---------------------------------------------------------------------------
# Sanity check — valid config must NOT raise
# ---------------------------------------------------------------------------

def test_valid_config_compiles_without_error():
    """
    Guard against false positives: if a valid config raises, all three
    fail-condition tests become meaningless (they'd pass for the wrong reason).
    """
    validate_full_capability(_base_venue_cfg(), _base_matrix(), FREQTRADE_VERSION)


# ---------------------------------------------------------------------------
# test_exchange_capability_matrix_compile_fail_post_only
# ---------------------------------------------------------------------------

def test_exchange_capability_matrix_compile_fail_post_only():
    """
    Condition: venue_cfg.maker_first=True, capability.post_only_supported=False.
    Expected: ValueError — build must halt.

    Root cause this guards: if maker_first is declared but the exchange does not
    enforce post-only at the API level, limit orders may cross the spread and
    become taker fills, breaking the cost model assumption.
    """
    matrix = _base_matrix()
    # Flip post_only_supported to False — this is the exact failure condition.
    matrix["binance_spot"]["post_only_supported"] = False

    venue_cfg = _base_venue_cfg()
    # maker_first=True is already set in base — no mutation needed.

    with pytest.raises(ValueError, match="post_only_supported"):
        validate_full_capability(venue_cfg, matrix, FREQTRADE_VERSION)


# ---------------------------------------------------------------------------
# test_exchange_capability_matrix_compile_fail_stoploss_type
# ---------------------------------------------------------------------------

def test_exchange_capability_matrix_compile_fail_stoploss_type():
    """
    Condition: venue_cfg.stoploss_order_type is NOT in capability.stoploss_order_types_supported.
    Expected: ValueError — build must halt.

    Root cause this guards: an unsupported stoploss order type will be silently
    rejected by the exchange, leaving the position without an active stop.
    This is the highest-severity silent failure mode in the system.
    """
    matrix = _base_matrix()
    # Exchange supports only stop_limit. Venue requests stop_market (not in list).
    venue_cfg = _base_venue_cfg()
    venue_cfg["stoploss_order_type"] = "stop_market"

    with pytest.raises(ValueError, match="stoploss_order_type"):
        validate_full_capability(venue_cfg, matrix, FREQTRADE_VERSION)


# ---------------------------------------------------------------------------
# test_exchange_capability_matrix_compile_fail_market_exit
# ---------------------------------------------------------------------------

def test_exchange_capability_matrix_compile_fail_market_exit():
    """
    Condition: venue_cfg.emergency_exit_enabled=True, capability.market_order_supported=False.
    Expected: ValueError — build must halt.

    Root cause this guards: the emergency exit cascade sends a market order as
    its first attempt. If the exchange does not support market orders (rare but
    possible on some DEX bridges or restricted spot venues), the cascade fails
    at step 1 with no fallback, leaving the position stranded.

    Note: emergency_exit_enabled=False by default (CLAUDE.md hard constraint #4).
    This test validates the guard that fires when the operator explicitly enables it.
    """
    matrix = _base_matrix()
    matrix["binance_spot"]["market_order_supported"] = False

    venue_cfg = _base_venue_cfg()
    # Override the default — operator explicitly enables emergency exit.
    venue_cfg["emergency_exit_enabled"] = True

    with pytest.raises(ValueError, match="market_order_supported"):
        validate_full_capability(venue_cfg, matrix, FREQTRADE_VERSION)


# ---------------------------------------------------------------------------
# Edge cases — not in the required gate but prevent silent regression
# ---------------------------------------------------------------------------

def test_missing_exchange_key_raises():
    """venue_cfg without exchange_key must fail immediately."""
    venue_cfg = _base_venue_cfg()
    del venue_cfg["exchange_key"]
    with pytest.raises(ValueError, match="exchange_key"):
        validate_full_capability(venue_cfg, _base_matrix(), FREQTRADE_VERSION)


def test_unknown_exchange_key_raises():
    """venue_cfg pointing to a venue not in capability_matrix must fail."""
    venue_cfg = _base_venue_cfg()
    venue_cfg["exchange_key"] = "okx_spot"  # not in matrix
    with pytest.raises(ValueError, match="not found in capability_matrix"):
        validate_full_capability(venue_cfg, _base_matrix(), FREQTRADE_VERSION)


def test_freqtrade_version_too_old_raises():
    """Image older than freqtrade_min_version must fail before all other checks."""
    with pytest.raises(ValueError, match="Freqtrade version"):
        validate_full_capability(_base_venue_cfg(), _base_matrix(), "2025.12")


def test_unsupported_stoploss_api_version_raises():
    """stoploss_api_version not in supported list must fail."""
    venue_cfg = _base_venue_cfg()
    venue_cfg["stoploss_api_version"] = "futures_algo_order_v3"  # futures API, wrong for spot
    with pytest.raises(ValueError, match="stoploss_api_version"):
        validate_full_capability(venue_cfg, _base_matrix(), FREQTRADE_VERSION)


def test_unsupported_conditional_order_mode_raises():
    """conditional_order_mode not in supported list must fail."""
    venue_cfg = _base_venue_cfg()
    venue_cfg["conditional_order_mode"] = "oco_bracket"  # not in spot capability
    with pytest.raises(ValueError, match="conditional_order_mode"):
        validate_full_capability(venue_cfg, _base_matrix(), FREQTRADE_VERSION)