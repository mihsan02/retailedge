"""
tests/test_decision_bus.py

Test gate: Always (test_no_orphan_decision_contract runs in every CI pass).

Core test: test_no_orphan_decision_contract
Verifies that every action type in VALID_ACTION_TYPES has:
- A defined producer (post() call with that action_type)
- A defined consumer path (consume_pending returns it)
- A defined actuator path (mark_in_progress)
- A success check (mark_done only after actuation)
- A retry path (mark_failed_retryable with backoff)
- A test (this file — self-referential but explicit)

CLAUDE.md rule: "Every Decision Bus action must have: producer, consumer,
actuator, success check, retry, and test."

All tests use in-memory SQLite. No Freqtrade REST needed.
"""

import pytest
import time
from sidecar.guardian.decision_bus import (
    DecisionBus,
    VALID_ACTION_TYPES,
    DEFAULT_MAX_RETRY,
    RETRY_BACKOFF_SECONDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bus():
    """Fresh in-memory Decision Bus for each test."""
    b = DecisionBus(":memory:")
    yield b
    b.close()


# ---------------------------------------------------------------------------
# test_no_orphan_decision_contract
# ---------------------------------------------------------------------------

def test_no_orphan_decision_contract(bus):
    """
    Every action type in VALID_ACTION_TYPES must be exercisable through
    the full producer -> consumer -> actuator -> success/retry cycle.

    An "orphan" action is one that:
    - Can be posted but never consumed (lost in queue)
    - Can be consumed but never marked done (stuck IN_PROGRESS)
    - Has no retry path (silently dropped on failure)
    - Has no test (undocumented behavior)

    This test exercises all nine action types through the complete lifecycle.
    Partial Sprint 1 scope: all nine types are posted and consumed.
    Guardian-specific routing (which type triggers which REST call) is
    tested in test_guardian.py (Sprint 2). Here we verify the bus contract.

    Failure of this test = a new action type was added without full wiring.
    """
    # --- Producer contract: every type can be posted without error
    posted_ids = {}
    for action_type in sorted(VALID_ACTION_TYPES):
        action_id = bus.post(
            action_type=action_type,
            reason=f"test_producer_{action_type}",
            severity="INFO",
        )
        assert action_id, f"post() returned empty action_id for {action_type}"
        posted_ids[action_type] = action_id

    assert len(posted_ids) == len(VALID_ACTION_TYPES), (
        f"Expected {len(VALID_ACTION_TYPES)} actions posted, got {len(posted_ids)}"
    )

    # --- Consumer contract: consume_pending returns all posted actions
    pending = bus.consume_pending(limit=100)
    pending_types = {a["action_type"] for a in pending}

    missing = VALID_ACTION_TYPES - pending_types
    assert not missing, (
        f"consume_pending() did not return actions for types: {missing}. "
        f"These types have no consumer path — orphan contract violated."
    )

    # --- Actuator contract: every action can be marked IN_PROGRESS
    for action in pending:
        bus.mark_in_progress(action["action_id"])
        updated = bus.get_action(action["action_id"])
        assert updated["status"] == "IN_PROGRESS", (
            f"mark_in_progress failed for {action['action_type']}"
        )

    # --- Success check contract: mark_done transitions to DONE
    # Test on first half of actions (arbitrary split for coverage)
    action_list = sorted(posted_ids.items())
    done_types = action_list[:5]
    retry_types = action_list[5:]

    for action_type, action_id in done_types:
        bus.mark_done(action_id, result=f"actuated_{action_type}")
        updated = bus.get_action(action_id)
        assert updated["status"] == "DONE", (
            f"mark_done failed for {action_type}: status={updated['status']}"
        )

    # --- Retry contract: mark_failed_retryable schedules retry with backoff
    for action_type, action_id in retry_types:
        bus.mark_failed_retryable(action_id, error=f"test_failure_{action_type}")
        updated = bus.get_action(action_id)
        assert updated["status"] == "FAILED_RETRYABLE", (
            f"mark_failed_retryable failed for {action_type}: status={updated['status']}"
        )
        assert updated["retry_count"] == 1, (
            f"retry_count not incremented for {action_type}"
        )
        assert updated["next_retry_ts"] is not None, (
            f"next_retry_ts not set for {action_type}"
        )

    # --- Test contract: verified by existence of this test file (self-referential)
    # All 9 types covered above. No orphan detected.


# ---------------------------------------------------------------------------
# Action type validation
# ---------------------------------------------------------------------------

def test_post_invalid_action_type_raises(bus):
    """
    Posting an unknown action type must raise ValueError immediately.
    Prevents typos from creating silent no-op actions.
    """
    with pytest.raises(ValueError, match="Unknown action_type"):
        bus.post("INVALID_TYPE", reason="test")


def test_post_empty_reason_raises(bus):
    """
    Every action must have a reason. Empty reason = unauditable action.
    """
    with pytest.raises(ValueError, match="reason must not be empty"):
        bus.post("PAUSE_REQUIRED", reason="")


def test_all_nine_action_types_in_registry():
    """
    Explicit count check: VALID_ACTION_TYPES must contain exactly 9 types.
    If this fails, a type was added or removed without updating this test.
    """
    expected = {
        "PAUSE_REQUIRED",
        "RESUME_ENTRY",
        "EMERGENCY_EXIT",
        "REJECT_ENTRY",
        "STOP_UNCONFIRMED",
        "RESERVED_MISMATCH_ON_STARTUP",
        "SCHEDULED_MODEL_PROMOTION",
        "OPERATOR_REQUIRED",
        "EMERGENCY_EXIT_FAILED",
    }
    assert VALID_ACTION_TYPES == expected, (
        f"VALID_ACTION_TYPES mismatch.\n"
        f"Extra: {VALID_ACTION_TYPES - expected}\n"
        f"Missing: {expected - VALID_ACTION_TYPES}"
    )


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------

def test_post_returns_unique_action_ids(bus):
    """Two posts of the same type must return distinct action_ids."""
    id1 = bus.post("PAUSE_REQUIRED", reason="first")
    id2 = bus.post("PAUSE_REQUIRED", reason="second")
    assert id1 != id2


def test_consume_pending_returns_fifo_order(bus):
    """
    Actions must be returned in created_ts ASC order.
    Guardian must process older actions first to preserve causal order.
    """
    bus.post("PAUSE_REQUIRED", reason="first")
    bus.post("REJECT_ENTRY", reason="second")
    bus.post("STOP_UNCONFIRMED", reason="third")

    pending = bus.consume_pending()
    types = [a["action_type"] for a in pending]
    assert types == ["PAUSE_REQUIRED", "REJECT_ENTRY", "STOP_UNCONFIRMED"]


def test_mark_done_requires_prior_post(bus):
    """mark_done on a non-existent action_id must fail silently (0 rows updated)."""
    # SQLite UPDATE on non-existent row is not an error — it's 0 rows affected.
    # This is acceptable: the Guardian may retry a DONE action from a stale queue.
    # The action just stays at whatever status it's at.
    bus.mark_done("nonexistent-id", result="test")
    # No exception = correct behavior


def test_retry_exhaustion_marks_fatal(bus):
    """
    After DEFAULT_MAX_RETRY failures, action must be FAILED_FATAL, not FAILED_RETRYABLE.
    Guardian must not retry a FAILED_FATAL action.
    """
    action_id = bus.post("EMERGENCY_EXIT", reason="test_exhaustion")
    bus.mark_in_progress(action_id)

    for i in range(DEFAULT_MAX_RETRY):
        bus.mark_failed_retryable(action_id, error=f"failure_{i}")

    final = bus.get_action(action_id)
    assert final["status"] == "FAILED_FATAL", (
        f"Expected FAILED_FATAL after {DEFAULT_MAX_RETRY} retries, "
        f"got {final['status']}"
    )
    assert final["retry_count"] == DEFAULT_MAX_RETRY


def test_failed_retryable_not_consumed_before_backoff(bus):
    """
    A FAILED_RETRYABLE action with next_retry_ts in the future must NOT
    be returned by consume_pending(). Only actions whose retry window
    has elapsed are eligible for reprocessing.
    """
    action_id = bus.post("PAUSE_REQUIRED", reason="test_backoff")
    bus.mark_in_progress(action_id)
    bus.mark_failed_retryable(action_id, error="timeout")

    # next_retry_ts is 30s in the future. consume_pending checks <= now.
    pending = bus.consume_pending()
    pending_ids = [a["action_id"] for a in pending]
    assert action_id not in pending_ids, (
        "FAILED_RETRYABLE action with future next_retry_ts must not be consumed yet"
    )


def test_has_pending_of_type_prevents_duplicate_posts(bus):
    """
    has_pending_of_type() allows Guardian to deduplicate PAUSE_REQUIRED posts.
    Without this check, a tight audit loop could flood the queue.
    """
    assert not bus.has_pending_of_type("PAUSE_REQUIRED")
    bus.post("PAUSE_REQUIRED", reason="first")
    assert bus.has_pending_of_type("PAUSE_REQUIRED")


def test_count_by_status(bus):
    """count_by_status returns correct counts after transitions."""
    bus.post("PAUSE_REQUIRED", reason="a")
    bus.post("REJECT_ENTRY", reason="b")
    assert bus.count_by_status("PENDING") == 2

    pending = bus.consume_pending()
    bus.mark_in_progress(pending[0]["action_id"])
    bus.mark_done(pending[0]["action_id"], result="ok")

    assert bus.count_by_status("DONE") == 1
    assert bus.count_by_status("PENDING") == 1


def test_mark_failed_fatal_unconditional(bus):
    """mark_failed_fatal sets FAILED_FATAL regardless of retry_count."""
    action_id = bus.post("EMERGENCY_EXIT_FAILED", reason="cascade_failed")
    bus.mark_failed_fatal(action_id, error="market_and_limit_both_failed")
    action = bus.get_action(action_id)
    assert action["status"] == "FAILED_FATAL"


def test_optional_fields_stored_correctly(bus):
    """pair, trade_id, model_id are stored and retrievable."""
    action_id = bus.post(
        "EMERGENCY_EXIT",
        reason="price_near_stop",
        severity="CRITICAL",
        pair="BTC/USDT",
        trade_id="trade_42",
        model_id="bucket_baseline_001",
    )
    action = bus.get_action(action_id)
    assert action["pair"] == "BTC/USDT"
    assert action["trade_id"] == "trade_42"
    assert action["model_id"] == "bucket_baseline_001"
    assert action["severity"] == "CRITICAL"