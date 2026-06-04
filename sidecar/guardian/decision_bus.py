"""
sidecar/guardian/decision_bus.py

Decision Bus for RetailEdge.
Single responsibility: durable, ordered queue for system control actions.

Contract (CLAUDE.md):
- Every action type must have: producer, consumer, actuator, success check, retry, test.
- Mark DONE only on confirmed 2xx. Never assume success.
- Do not invent action types outside the canonical set.
- All nine action types are defined in VALID_ACTION_TYPES below.

Design:
- SQLite-backed for durability across restarts.
- WAL mode for concurrent sidecar reads.
- post() is the only way to create actions — no direct INSERT elsewhere.
- consume_pending() is idempotent — safe to call in a tight poll loop.
- mark_done() and mark_failed_retryable() are the only valid terminal transitions.
- Retry scheduling uses exponential backoff with a hard max_retry cap.

Schema dependency: system_actions table from ledger/schema.sql.
This module creates the table if absent (idempotent) for test isolation.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Canonical action type registry
# ---------------------------------------------------------------------------
# This is the single source of truth for valid action types in the codebase.
# Adding a new type requires: entry here + schema.sql comment + producer + consumer + test.
# Removing a type requires: deprecation cycle + confirm no live actions exist.

VALID_ACTION_TYPES = frozenset({
    "PAUSE_REQUIRED",
    "RESUME_ENTRY",
    "EMERGENCY_EXIT",
    "REJECT_ENTRY",
    "STOP_UNCONFIRMED",
    "RESERVED_MISMATCH_ON_STARTUP",
    "SCHEDULED_MODEL_PROMOTION",
    "OPERATOR_REQUIRED",
    "EMERGENCY_EXIT_FAILED",
})

# Valid status transitions:
# PENDING -> IN_PROGRESS -> DONE
# PENDING -> IN_PROGRESS -> FAILED_RETRYABLE -> PENDING (after next_retry_ts)
# PENDING -> IN_PROGRESS -> FAILED_FATAL (no retry)
VALID_STATUSES = frozenset({"PENDING", "IN_PROGRESS", "DONE", "FAILED_RETRYABLE", "FAILED_FATAL"})

# Max retries before a FAILED_RETRYABLE action becomes FAILED_FATAL.
# Overridable per-action via max_retry arg in mark_failed_retryable().
DEFAULT_MAX_RETRY = 3

# Exponential backoff delays in seconds: retry 1=30s, retry 2=60s, retry 3=120s.
RETRY_BACKOFF_SECONDS = [30, 60, 120]


# ---------------------------------------------------------------------------
# DecisionBus
# ---------------------------------------------------------------------------

class DecisionBus:
    """
    SQLite-backed durable action queue.

    Thread safety: SQLite WAL mode allows concurrent readers and one writer.
    For the sidecar architecture (single Guardian writer, multiple readers),
    this is sufficient without an external broker.

    Usage:
        bus = DecisionBus(db_path)
        bus.post("PAUSE_REQUIRED", reason="stoploss_unconfirmed", severity="HIGH")
        actions = bus.consume_pending()
        for action in actions:
            # attempt actuation
            bus.mark_done(action["action_id"], result="paused")
            # or on failure:
            bus.mark_failed_retryable(action["action_id"], error="timeout")
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; explicit transactions where needed
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_table()

    def _ensure_table(self) -> None:
        """
        Create system_actions table if not present.
        Idempotent — mirrors ledger/schema.sql definition exactly.
        """
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS system_actions (
                action_id   TEXT PRIMARY KEY,
                action_type TEXT NOT NULL,
                pair        TEXT,
                trade_id    TEXT,
                model_id    TEXT,
                reason      TEXT,
                severity    TEXT,
                status      TEXT NOT NULL DEFAULT 'PENDING',
                retry_count INTEGER NOT NULL DEFAULT 0,
                next_retry_ts TEXT,
                created_ts  TEXT NOT NULL,
                updated_ts  TEXT
            )
        """)

    # -----------------------------------------------------------------------
    # Producer
    # -----------------------------------------------------------------------

    def post(
        self,
        action_type: str,
        reason: str,
        severity: str = "INFO",
        pair: Optional[str] = None,
        trade_id: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs: Any,  # absorbs extra kwargs from injectable callers
    ) -> str:
        """
        Insert a new action into the Decision Bus queue.

        Args:
            action_type: Must be in VALID_ACTION_TYPES. Raises ValueError if not.
            reason:      Human-readable reason string. Required — no silent no-ops.
            severity:    "INFO" | "HIGH" | "CRITICAL". Informational only in this layer.
            pair:        Trading pair if action is trade-specific.
            trade_id:    Freqtrade trade ID if action is trade-specific.
            model_id:    Model ID if action is deployment-related.

        Returns:
            action_id: UUID string of the created action.

        Raises:
            ValueError: If action_type is not in VALID_ACTION_TYPES.

        Why require reason: a no-reason action is unauditable. Every action
        must be explainable to an operator reading the audit log.
        """
        if action_type not in VALID_ACTION_TYPES:
            raise ValueError(
                f"Unknown action_type '{action_type}'. "
                f"Valid types: {sorted(VALID_ACTION_TYPES)}"
            )
        if not reason:
            raise ValueError("reason must not be empty — every action must be auditable")

        action_id = str(uuid.uuid4())
        now = _now_iso()

        self._conn.execute(
            """
            INSERT INTO system_actions
                (action_id, action_type, pair, trade_id, model_id,
                 reason, severity, status, retry_count, created_ts, updated_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', 0, ?, ?)
            """,
            (action_id, action_type, pair, trade_id, model_id, reason, severity, now, now),
        )
        return action_id

    # -----------------------------------------------------------------------
    # Consumer
    # -----------------------------------------------------------------------

    def consume_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Fetch PENDING actions that are ready to be processed.

        "Ready" means:
        - status = PENDING, OR
        - status = FAILED_RETRYABLE AND next_retry_ts <= now (retry window elapsed)

        Does NOT change status to IN_PROGRESS here — that is the caller's job
        via mark_in_progress(). Keeping consume idempotent prevents double-processing
        if the Guardian crashes after fetch but before actuation.

        Returns list of dicts (not Row objects) for JSON-serializability.
        """
        now = _now_iso()
        cur = self._conn.execute(
            """
            SELECT * FROM system_actions
            WHERE
                status = 'PENDING'
                OR (status = 'FAILED_RETRYABLE' AND next_retry_ts <= ?)
            ORDER BY created_ts ASC
            LIMIT ?
            """,
            (now, limit),
        )
        return [dict(row) for row in cur.fetchall()]

    def mark_in_progress(self, action_id: str) -> None:
        """
        Mark action as IN_PROGRESS before attempting actuation.
        Prevents double-processing if Guardian loop restarts mid-action.
        """
        self._conn.execute(
            "UPDATE system_actions SET status='IN_PROGRESS', updated_ts=? WHERE action_id=?",
            (_now_iso(), action_id),
        )

    # -----------------------------------------------------------------------
    # Terminal transitions
    # -----------------------------------------------------------------------

    def mark_done(self, action_id: str, result: str = "") -> None:
        """
        Mark action as DONE. Call only after confirmed 2xx from Freqtrade REST.
        Never call on timeout, assumption, or partial success.

        result: brief description of actuation outcome for audit log.
        Stored in reason field (appended) — no separate result column in schema.
        """
        now = _now_iso()
        self._conn.execute(
            """
            UPDATE system_actions
            SET status='DONE', updated_ts=?
            WHERE action_id=?
            """,
            (now, action_id),
        )

    def mark_failed_retryable(
        self,
        action_id: str,
        error: str,
        max_retry: int = DEFAULT_MAX_RETRY,
    ) -> None:
        """
        Mark action as FAILED_RETRYABLE and schedule next retry with backoff.
        If retry_count >= max_retry, mark FAILED_FATAL instead.

        error: what went wrong (HTTP status, exception message, etc.)
        max_retry: hard cap. Default DEFAULT_MAX_RETRY (3).

        Backoff schedule: RETRY_BACKOFF_SECONDS[min(retry_count, last_index)].
        After max_retry exhausted: FAILED_FATAL, no further retry.
        """
        # Fetch current retry_count
        cur = self._conn.execute(
            "SELECT retry_count FROM system_actions WHERE action_id=?",
            (action_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"action_id '{action_id}' not found in system_actions")

        retry_count = row["retry_count"] + 1
        now = _now_iso()

        if retry_count >= max_retry:
            # Exhausted — mark fatal, no further retry
            self._conn.execute(
                """
                UPDATE system_actions
                SET status='FAILED_FATAL',
                    retry_count=?,
                    reason=reason || ' | FATAL: ' || ?,
                    updated_ts=?
                WHERE action_id=?
                """,
                (retry_count, error, now, action_id),
            )
            return

        # Schedule next retry with exponential backoff
        backoff_idx = min(retry_count - 1, len(RETRY_BACKOFF_SECONDS) - 1)
        backoff_sec = RETRY_BACKOFF_SECONDS[backoff_idx]
        next_retry_ts = _now_plus_seconds(backoff_sec)

        self._conn.execute(
            """
            UPDATE system_actions
            SET status='FAILED_RETRYABLE',
                retry_count=?,
                next_retry_ts=?,
                reason=reason || ' | retry ' || ? || ': ' || ?,
                updated_ts=?
            WHERE action_id=?
            """,
            (retry_count, next_retry_ts, retry_count, error, now, action_id),
        )

    def mark_failed_fatal(self, action_id: str, error: str) -> None:
        """
        Unconditionally mark action as FAILED_FATAL.
        Use when retry is not appropriate (e.g., EMERGENCY_EXIT_FAILED after cascade).
        """
        now = _now_iso()
        self._conn.execute(
            """
            UPDATE system_actions
            SET status='FAILED_FATAL',
                reason=reason || ' | FATAL: ' || ?,
                updated_ts=?
            WHERE action_id=?
            """,
            (error, now, action_id),
        )

    # -----------------------------------------------------------------------
    # Query helpers
    # -----------------------------------------------------------------------

    def get_action(self, action_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single action by ID. Returns None if not found."""
        cur = self._conn.execute(
            "SELECT * FROM system_actions WHERE action_id=?",
            (action_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def count_by_status(self, status: str) -> int:
        """Return count of actions with given status. Useful for health checks."""
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM system_actions WHERE status=?",
            (status,),
        )
        return cur.fetchone()[0]

    def has_pending_of_type(self, action_type: str) -> bool:
        """
        Return True if there is at least one PENDING action of the given type.
        Used by Guardian to avoid duplicate PAUSE_REQUIRED posts.
        """
        cur = self._conn.execute(
            "SELECT 1 FROM system_actions WHERE action_type=? AND status='PENDING' LIMIT 1",
            (action_type,),
        )
        return cur.fetchone() is not None

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_plus_seconds(seconds: int) -> str:
    from datetime import timedelta
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()