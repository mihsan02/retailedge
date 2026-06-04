"""
sidecar/reconciler/reserved_funds.py

Reserved Funds Ledger interface for RetailEdge.
Single responsibility: read/write reserved quote amounts per open order,
and write startup reconciliation audit records.

Design constraint (CLAUDE.md hard constraint #5):
Entry is blocked if reserved funds reconciliation fails on startup.

This module owns the ledger layer only.
Decision Bus posting lives in startup_reconcile.py — not here.
No business logic. Pure data access.

Schema dependency:
- reserved_funds (order_id PK, reserved_quote REAL, updated_ts TEXT)
- reserved_funds_reconciliation (run_id + order_id PK, full audit record)

Both tables must exist before any method is called.
Call ledger.init_schema() at startup.
"""

import sqlite3
from datetime import datetime, timezone
from typing import Optional


class ReservedFundsLedger:
    """
    Thin wrapper over SQLite for reserved funds operations.

    Uses the same DB connection pattern as ledger/ledger.py:
    WAL mode, check_same_thread=False for sidecar container use.

    Why not inherit from Ledger: reserved funds is a distinct operational
    domain. Tight coupling to the base Ledger class creates a god object.
    This class takes a db_path and manages its own connection.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; explicit transactions where needed
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """
        Create tables if they don't exist.
        Idempotent — safe to call on every startup.
        Does not replace ledger/schema.sql as the canonical schema source;
        this is a safety net for test isolation.
        """
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS reserved_funds (
                order_id TEXT PRIMARY KEY,
                pair TEXT NOT NULL,
                reserved_quote REAL NOT NULL DEFAULT 0.0,
                updated_ts TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reserved_funds_reconciliation (
                run_id TEXT NOT NULL,
                order_id TEXT NOT NULL,
                pair TEXT NOT NULL,
                exchange_reserved_quote REAL,
                local_reserved_quote REAL,
                diff_quote REAL,
                status TEXT NOT NULL,
                created_ts TEXT NOT NULL,
                PRIMARY KEY(run_id, order_id)
            );
        """)

    # -----------------------------------------------------------------------
    # Read operations
    # -----------------------------------------------------------------------

    def load_reserved_funds_map(self) -> dict[str, float]:
        """
        Return {order_id: reserved_quote} for all current reserved entries.

        This is the local ledger state — may differ from exchange reality
        after a crash or restart. startup_reconcile.py uses this as the
        'local' side of the diff.
        """
        cur = self._conn.execute(
            "SELECT order_id, reserved_quote FROM reserved_funds"
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    def get_total_reserved(self) -> float:
        """
        Sum of all reserved_quote entries.
        Used by pre_trade_gate to compute projected_available balance.
        """
        cur = self._conn.execute(
            "SELECT COALESCE(SUM(reserved_quote), 0.0) FROM reserved_funds"
        )
        return float(cur.fetchone()[0])

    def has_status(self, run_id: str, status: str) -> bool:
        """
        Return True if any reconciliation record for run_id has the given status.
        Used by startup_reconcile to detect RESERVED_MISMATCH before allowing entry.
        """
        cur = self._conn.execute(
            "SELECT 1 FROM reserved_funds_reconciliation "
            "WHERE run_id = ? AND status = ? LIMIT 1",
            (run_id, status),
        )
        return cur.fetchone() is not None

    # -----------------------------------------------------------------------
    # Write operations
    # -----------------------------------------------------------------------

    def upsert_reserved(self, order_id: str, pair: str, reserved_quote: float) -> None:
        """
        Insert or replace a reserved funds entry for an order.
        Called by reconciler after a new buy order is placed or partially filled.
        """
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO reserved_funds (order_id, pair, reserved_quote, updated_ts)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(order_id) DO UPDATE SET
                reserved_quote = excluded.reserved_quote,
                updated_ts = excluded.updated_ts
            """,
            (order_id, pair, reserved_quote, now),
        )

    def delete_reserved(self, order_id: str) -> None:
        """
        Remove reserved funds entry when an order is fully filled, cancelled, or expired.
        """
        self._conn.execute(
            "DELETE FROM reserved_funds WHERE order_id = ?",
            (order_id,),
        )

    def write_reserved_recon(
        self,
        run_id: str,
        order_id: str,
        pair: str,
        exchange_reserved_quote: float,
        local_reserved_quote: float,
        diff_quote: float,
        status: str,
    ) -> None:
        """
        Write one reconciliation audit record.
        status: "MATCH" or "RESERVED_MISMATCH".

        Called once per open order during startup reconciliation.
        Records are immutable after write — never updated, only inserted.
        """
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO reserved_funds_reconciliation
                (run_id, order_id, pair, exchange_reserved_quote,
                 local_reserved_quote, diff_quote, status, created_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, order_id, pair,
                exchange_reserved_quote, local_reserved_quote,
                diff_quote, status, now,
            ),
        )

    def replace_reserved_funds_from_exchange_projection(
        self, projected: dict[str, float]
    ) -> None:
        """
        Replace the entire reserved_funds table with exchange-derived projections.

        Called at the end of a successful startup reconciliation.
        Exchange open orders are the source of truth on startup —
        local ledger state from before the restart is discarded.

        Uses a single transaction: either the full replace succeeds or
        the old state is preserved. No partial state allowed.

        projected: {order_id: reserved_quote}
        Note: pair is not available in the projection dict at this point.
        Pair is set to "UNKNOWN" for entries that come only from projection.
        The reconciler will update pair on the next order sync cycle.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:  # transaction context manager
            self._conn.execute("DELETE FROM reserved_funds")
            self._conn.executemany(
                """
                INSERT INTO reserved_funds (order_id, pair, reserved_quote, updated_ts)
                VALUES (?, 'UNKNOWN', ?, ?)
                """,
                [(oid, quote, now) for oid, quote in projected.items() if quote > 0],
            )

    def close(self) -> None:
        self._conn.close()