"""
ledger.py — RetailEdge Ledger read/write interface.

Semua akses ke SQLite database melewati kelas ini.
Tidak ada komponen lain yang boleh import sqlite3 langsung.

Usage:
    from ledger.ledger import Ledger
    db = Ledger("/ledger/retailedge.db")
    db.init_schema()
"""

import sqlite3
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


class Ledger:
    def __init__(self, db_path: str):
        self.db_path = db_path

        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def init_schema(self) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        sql = schema_path.read_text()
        self._conn.executescript(sql)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Decision Bus
    # ------------------------------------------------------------------

    def post_action(
        self,
        action_type: str,
        reason: str,
        severity: str,
        pair: Optional[str] = None,
        trade_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> str:
        action_id = _new_id()
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
        self._conn.commit()
        return action_id

    def consume_pending_actions(self) -> list:
        cursor = self._conn.execute(
            """
            SELECT * FROM system_actions
            WHERE status = 'PENDING'
            ORDER BY created_ts ASC
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def mark_action_done(self, action_id: str, result: str) -> None:
        self._conn.execute(
            """
            UPDATE system_actions
            SET status = 'DONE', reason = reason || ' | result: ' || ?,
                updated_ts = ?
            WHERE action_id = ?
            """,
            (result, _now_iso(), action_id),
        )
        self._conn.commit()

    def mark_action_failed_retryable(self, action_id: str, error: str) -> None:
        self._conn.execute(
            """
            UPDATE system_actions
            SET status = 'FAILED_RETRYABLE',
                retry_count = retry_count + 1,
                reason = reason || ' | error: ' || ?,
                updated_ts = ?
            WHERE action_id = ?
            """,
            (error, _now_iso(), action_id),
        )
        self._conn.commit()

    def mark_action_fatal(self, action_id: str, error: str) -> None:
        self._conn.execute(
            """
            UPDATE system_actions
            SET status = 'FATAL',
                reason = reason || ' | fatal: ' || ?,
                updated_ts = ?
            WHERE action_id = ?
            """,
            (error, _now_iso(), action_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Reserved Funds
    # ------------------------------------------------------------------

    def load_reserved_funds_map(self) -> dict:
        cursor = self._conn.execute(
            "SELECT order_id, reserved_quote FROM reserved_funds WHERE status = 'ACTIVE'"
        )
        return {row["order_id"]: row["reserved_quote"] for row in cursor.fetchall()}

    def write_reserved_recon(
        self,
        run_id: str,
        order_id: str,
        pair: Optional[str],
        exchange_reserved_quote: float,
        local_reserved_quote: float,
        diff_quote: float,
        status: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO reserved_funds_reconciliation
                (run_id, order_id, pair, exchange_reserved_quote,
                 local_reserved_quote, diff_quote, status, created_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, order_id, pair, exchange_reserved_quote,
                local_reserved_quote, diff_quote, status, _now_iso(),
            ),
        )
        self._conn.commit()

    def replace_reserved_funds_from_exchange_projection(self, projected: dict) -> None:
        now = _now_iso()
        self._conn.execute("UPDATE reserved_funds SET status = 'SUPERSEDED', updated_ts = ?", (now,))
        for order_id, reserved_quote in projected.items():
            self._conn.execute(
                """
                INSERT OR REPLACE INTO reserved_funds
                    (order_id, pair, reserved_quote, status, created_ts, updated_ts)
                VALUES (?, 'UNKNOWN', ?, 'ACTIVE', ?, ?)
                """,
                (order_id, reserved_quote, now, now),
            )
        self._conn.commit()

    def has_recon_status(self, run_id: str, status: str) -> bool:
        cursor = self._conn.execute(
            "SELECT 1 FROM reserved_funds_reconciliation WHERE run_id = ? AND status = ? LIMIT 1",
            (run_id, status),
        )
        return cursor.fetchone() is not None

    def get_total_reserved_quote(self) -> float:
        cursor = self._conn.execute(
            "SELECT COALESCE(SUM(reserved_quote), 0.0) FROM reserved_funds WHERE status = 'ACTIVE'"
        )
        return cursor.fetchone()[0]

    # ------------------------------------------------------------------
    # Execution Orders
    # ------------------------------------------------------------------

    def upsert_order(
        self,
        order_id: str,
        trade_id: str,
        pair: str,
        side: str,
        order_type: str,
        price: Optional[float],
        amount: Optional[float],
        status: str,
    ) -> None:
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO execution_orders
                (order_id, trade_id, pair, side, order_type, price, amount, status, created_ts, updated_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(order_id) DO UPDATE SET
                status = excluded.status,
                price = excluded.price,
                amount = excluded.amount,
                updated_ts = excluded.updated_ts
            """,
            (order_id, trade_id, pair, side, order_type, price, amount, status, now, now),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Execution Fills
    # ------------------------------------------------------------------

    def insert_fill_if_not_exists(
        self,
        fill_id: str,
        order_id: str,
        trade_id: str,
        pair: str,
        side: str,
        fill_price: float,
        fill_amount: float,
        fee_quote: Optional[float],
    ) -> bool:
        try:
            self._conn.execute(
                """
                INSERT INTO execution_fills
                    (fill_id, order_id, trade_id, pair, side,
                     fill_price, fill_amount, fee_quote, created_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (fill_id, order_id, trade_id, pair, side,
                 fill_price, fill_amount, fee_quote, _now_iso()),
            )
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # duplicate — ignored by design

    # ------------------------------------------------------------------
    # Strategy Memory
    # ------------------------------------------------------------------

    def write_candidate(
        self,
        strategy_id: str,
        model_type: str,
        model_hash: Optional[str] = None,
        feature_hash: Optional[str] = None,
        oos_trades: Optional[int] = None,
        oos_expectancy: Optional[float] = None,
        oos_sharpe: Optional[float] = None,
        pbo: Optional[float] = None,
        dsr: Optional[float] = None,
        cost_floor_pct: Optional[float] = None,
    ) -> str:
        candidate_id = _new_id()
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO strategy_memory
                (candidate_id, strategy_id, model_type, model_hash, feature_hash,
                 oos_trades, oos_expectancy, oos_sharpe, pbo, dsr, cost_floor_pct,
                 stage_gate, created_ts, updated_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'CANDIDATE', ?, ?)
            """,
            (
                candidate_id, strategy_id, model_type, model_hash, feature_hash,
                oos_trades, oos_expectancy, oos_sharpe, pbo, dsr, cost_floor_pct,
                now, now,
            ),
        )
        self._conn.commit()
        return candidate_id

    def load_candidate(self, candidate_id: str) -> Optional[dict]:
        cursor = self._conn.execute(
            "SELECT * FROM strategy_memory WHERE candidate_id = ?", (candidate_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Deployment Audit
    # ------------------------------------------------------------------

    def write_deployment_audit(
        self,
        model_id: str,
        model_type: str,
        model_hash: str,
        approved_by: str,
        approved_at: str,
        promotion_source: str,
        rollback_model_id: Optional[str] = None,
        config_compile_pass: bool = False,
        dry_replay_pass: bool = False,
        notes: Optional[str] = None,
    ) -> str:
        audit_id = _new_id()
        self._conn.execute(
            """
            INSERT INTO deployment_audit
                (audit_id, model_id, model_type, model_hash, rollback_model_id,
                 approved_by, approved_at, promotion_source,
                 config_compile_pass, dry_replay_pass, notes, created_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id, model_id, model_type, model_hash, rollback_model_id,
                approved_by, approved_at, promotion_source,
                int(config_compile_pass), int(dry_replay_pass), notes, _now_iso(),
            ),
        )
        self._conn.commit()
        return audit_id

    # ------------------------------------------------------------------
    # Trade State Flags
    # ------------------------------------------------------------------

    def set_stop_confirmed(self, trade_id: str, pair: str, confirmed: bool) -> None:
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO trade_state_flags
                (trade_id, pair, stop_confirmed, created_ts, updated_ts)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(trade_id) DO UPDATE SET
                stop_confirmed = excluded.stop_confirmed,
                updated_ts = excluded.updated_ts
            """,
            (trade_id, pair, int(confirmed), now, now),
        )
        self._conn.commit()

    def increment_stop_unconfirmed(self, trade_id: str, pair: str) -> None:
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO trade_state_flags
                (trade_id, pair, stop_unconfirmed_count, created_ts, updated_ts)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(trade_id) DO UPDATE SET
                stop_unconfirmed_count = stop_unconfirmed_count + 1,
                updated_ts = excluded.updated_ts
            """,
            (trade_id, pair, now, now),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Health / Heartbeat (dipakai Reconciler)
    # ------------------------------------------------------------------

    def write_heartbeat(self, component: str) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO trade_state_flags
                (trade_id, pair, created_ts, updated_ts)
            VALUES (?, ?, ?, ?)
            """,
            (f"heartbeat_{component}", component, _now_iso(), _now_iso()),
        )
        self._conn.commit()

    def get_last_heartbeat(self, component: str) -> Optional[str]:
        cursor = self._conn.execute(
            "SELECT updated_ts FROM trade_state_flags WHERE trade_id = ?",
            (f"heartbeat_{component}",),
        )
        row = cursor.fetchone()
        return row["updated_ts"] if row else None