"""
sidecar/reconciler/reconciler.py

Reconciler Worker for RetailEdge.
Single responsibility: sync order/fill state from Freqtrade REST into the ledger,
deduplicate fill events, and write a heartbeat timestamp every loop.

Hard rules (CLAUDE.md):
- Deduplicate fill events by (order_id, fill_id). Never double-count a fill.
- Write heartbeat timestamp every loop iteration.
- Guardian monitors heartbeat: if stale > 120s, Guardian posts PAUSE_REQUIRED.
- Never assume Freqtrade state — always fetch from REST, then reconcile.

Design:
- Reconciler is stateless between iterations. All state lives in SQLite.
- One loop: fetch openorders + trades -> upsert orders -> process fills -> write heartbeat.
- Fill deduplication uses INSERT OR IGNORE on (order_id, fill_id) PK.
- Heartbeat is a single-row upsert in a dedicated table.
- reserved_funds are updated post-fill: remaining * price for open buy orders.

Schema dependency:
- execution_orders (order_id PK)
- execution_fills (order_id + fill_id composite PK)
- reconciler_heartbeat (singleton row, id='reconciler')
- reserved_funds (order_id PK) — via ReservedFundsLedger

All tables created idempotently if missing (safety net for test isolation).
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

import sqlite3


class ReconcilerLedger:
    """
    Ledger interface for Reconciler: orders, fills, heartbeat.
    Separate from ReservedFundsLedger to maintain single-responsibility.
    Both classes share the same DB file in production.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS execution_orders (
                order_id        TEXT PRIMARY KEY,
                trade_id        TEXT,
                pair            TEXT NOT NULL,
                side            TEXT NOT NULL,
                order_type      TEXT,
                status          TEXT NOT NULL,
                amount          REAL,
                filled          REAL DEFAULT 0.0,
                remaining       REAL,
                price           REAL,
                average         REAL,
                cost            REAL DEFAULT 0.0,
                fee_quote       REAL DEFAULT 0.0,
                created_ts      TEXT NOT NULL,
                updated_ts      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS execution_fills (
                order_id        TEXT NOT NULL,
                fill_id         TEXT NOT NULL,
                pair            TEXT NOT NULL,
                side            TEXT NOT NULL,
                fill_amount     REAL NOT NULL,
                fill_price      REAL NOT NULL,
                fill_cost       REAL NOT NULL,
                fee_quote       REAL DEFAULT 0.0,
                fill_ts         TEXT NOT NULL,
                created_ts      TEXT NOT NULL,
                PRIMARY KEY (order_id, fill_id)
            );

            CREATE TABLE IF NOT EXISTS reconciler_heartbeat (
                id              TEXT PRIMARY KEY DEFAULT 'reconciler',
                last_ts         TEXT NOT NULL,
                loop_count      INTEGER NOT NULL DEFAULT 0
            );
        """)

    # -----------------------------------------------------------------------
    # Orders
    # -----------------------------------------------------------------------

    def upsert_order(self, order: dict[str, Any]) -> None:
        """
        Insert or update an order record.
        Uses INSERT OR REPLACE — last-write-wins for order state.
        This is safe because order state is always fetched fresh from exchange.
        """
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO execution_orders
                (order_id, trade_id, pair, side, order_type, status,
                 amount, filled, remaining, price, average, cost, fee_quote,
                 created_ts, updated_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(order_id) DO UPDATE SET
                status      = excluded.status,
                filled      = excluded.filled,
                remaining   = excluded.remaining,
                average     = excluded.average,
                cost        = excluded.cost,
                fee_quote   = excluded.fee_quote,
                updated_ts  = excluded.updated_ts
            """,
            (
                _s(order.get("id")),
                _s(order.get("trade_id")),
                _s(order.get("symbol") or order.get("pair") or "UNKNOWN"),
                _s(order.get("side", "")),
                _s(order.get("type") or order.get("order_type")),
                _s(order.get("status", "open")),
                _f(order.get("amount")),
                _f(order.get("filled"), 0.0),
                _f(order.get("remaining")),
                _f(order.get("price")),
                _f(order.get("average")),
                _f(order.get("cost"), 0.0),
                _f(order.get("fee", {}).get("cost") if isinstance(order.get("fee"), dict) else order.get("fee_quote"), 0.0),
                now,
                now,
            ),
        )

    def get_order(self, order_id: str) -> Optional[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM execution_orders WHERE order_id=?", (order_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # -----------------------------------------------------------------------
    # Fills — deduplication is the critical invariant
    # -----------------------------------------------------------------------

    def insert_fill_if_new(self, order_id: str, fill: dict[str, Any]) -> bool:
        """
        Insert a fill record only if (order_id, fill_id) does not exist.
        Uses INSERT OR IGNORE — idempotent on duplicate (order_id, fill_id).

        Returns True if fill was new (inserted), False if duplicate (ignored).

        Why INSERT OR IGNORE and not INSERT OR REPLACE:
        A fill is an immutable financial event. If it already exists, the stored
        record is correct. Replacing it would mask a data integrity bug where
        two different fills share the same fill_id.
        """
        fill_id = _s(fill.get("id") or fill.get("fill_id"))
        if not fill_id:
            # Exchange returned a fill without an ID — generate a surrogate
            # from order_id + amount + timestamp to avoid total loss.
            fill_id = f"surrogate_{order_id}_{fill.get('amount', 0)}_{fill.get('timestamp', '')}"
            logger.warning("Fill missing id for order %s, using surrogate %s", order_id, fill_id)

        now = _now_iso()
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO execution_fills
                (order_id, fill_id, pair, side, fill_amount, fill_price,
                 fill_cost, fee_quote, fill_ts, created_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                fill_id,
                _s(fill.get("symbol") or fill.get("pair") or "UNKNOWN"),
                _s(fill.get("side", "")),
                _f(fill.get("amount"), 0.0),
                _f(fill.get("price"), 0.0),
                _f(fill.get("cost"), 0.0),
                _f(fill.get("fee", {}).get("cost") if isinstance(fill.get("fee"), dict) else fill.get("fee_quote"), 0.0),
                _s(fill.get("datetime") or fill.get("fill_ts") or now),
                now,
            ),
        )
        return cur.rowcount > 0

    def count_fills(self, order_id: str) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM execution_fills WHERE order_id=?", (order_id,)
        )
        return cur.fetchone()[0]

    def get_fills(self, order_id: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM execution_fills WHERE order_id=? ORDER BY fill_ts ASC",
            (order_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    # -----------------------------------------------------------------------
    # Heartbeat
    # -----------------------------------------------------------------------

    def write_heartbeat(self, loop_count: int = 0) -> str:
        """
        Upsert the reconciler heartbeat timestamp.
        Guardian reads this to verify Reconciler is alive.
        Returns the written timestamp.
        """
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO reconciler_heartbeat (id, last_ts, loop_count)
            VALUES ('reconciler', ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                last_ts    = excluded.last_ts,
                loop_count = excluded.loop_count
            """,
            (now, loop_count),
        )
        return now

    def get_last_heartbeat_ts(self) -> Optional[str]:
        """
        Return last heartbeat timestamp or None if never written.
        This is the method injected into GuardianPolicy.heartbeat_store.
        """
        cur = self._conn.execute(
            "SELECT last_ts FROM reconciler_heartbeat WHERE id='reconciler'"
        )
        row = cur.fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------

class Reconciler:
    """
    Order state machine and fill deduplication engine.

    One loop iteration:
    1. Fetch open orders from Freqtrade REST.
    2. Fetch recent trades from Freqtrade REST.
    3. Upsert all orders into execution_orders.
    4. Process fills from trades: insert_fill_if_new per fill.
    5. Update reserved_funds for open buy orders.
    6. Write heartbeat.

    ft_client: FreqtradeClient instance (or mock).
    ledger: ReconcilerLedger instance.
    reserved_ledger: Optional ReservedFundsLedger for post-fill reserved update.
      If None, reserved funds update is skipped (test mode for reconciler unit tests).
    """

    def __init__(
        self,
        ft_client: Any,
        ledger: "ReconcilerLedger",
        reserved_ledger: Optional[Any] = None,
        poll_interval_sec: float = 30.0,
    ) -> None:
        self.ft = ft_client
        self.ledger = ledger
        self.reserved_ledger = reserved_ledger
        self.poll_interval_sec = poll_interval_sec
        self._loop_count = 0

    def run_loop(self) -> None:
        """Production entry point. Polls indefinitely."""
        logger.info("Reconciler starting poll loop, interval=%ss", self.poll_interval_sec)
        while True:
            try:
                self.run_once()
            except Exception as exc:
                logger.error("Reconciler loop error: %s", exc, exc_info=True)
            time.sleep(self.poll_interval_sec)

    def run_once(self) -> dict[str, Any]:
        """
        Process one reconciliation cycle.
        Returns summary dict for test inspection.
        Always writes heartbeat, even if fetch fails.
        """
        self._loop_count += 1
        summary = {
            "loop": self._loop_count,
            "orders_upserted": 0,
            "fills_new": 0,
            "fills_duplicate": 0,
            "errors": [],
        }

        try:
            # Step 1: open orders
            try:
                open_orders_resp = self.ft.get_open_orders()
                open_orders = self._extract_list(open_orders_resp, "orders")
            except Exception as exc:
                logger.warning("Failed to fetch open orders: %s", exc)
                summary["errors"].append(f"open_orders:{exc}")
                open_orders = []

            # Step 2: trades (closed + open, source of fill events)
            try:
                trades_resp = self.ft.get_trades()
                trades = self._extract_list(trades_resp, "trades")
            except Exception as exc:
                logger.warning("Failed to fetch trades: %s", exc)
                summary["errors"].append(f"trades:{exc}")
                trades = []

            # Step 3: upsert orders
            for order in open_orders:
                try:
                    self.ledger.upsert_order(order)
                    summary["orders_upserted"] += 1
                except Exception as exc:
                    logger.warning("Failed to upsert order %s: %s", order.get("id"), exc)
                    summary["errors"].append(f"upsert_order:{exc}")

            # Step 4: process fills from trades
            # Freqtrade /trades returns trade objects which may contain
            # an 'orders' list with individual order/fill records.
            for trade in trades:
                trade_orders = trade.get("orders", [])
                for order in trade_orders:
                    order_id = _s(order.get("order_id") or order.get("id"))
                    if not order_id:
                        continue

                    # Upsert the order record from trade context
                    order_for_upsert = dict(order)
                    order_for_upsert.setdefault("trade_id", _s(trade.get("trade_id") or trade.get("id")))
                    order_for_upsert.setdefault("symbol", trade.get("pair"))
                    try:
                        self.ledger.upsert_order(order_for_upsert)
                    except Exception:
                        pass

                    # Process fills for this order
                    fills = order.get("fills", [])
                    for fill in fills:
                        fill_with_context = dict(fill)
                        fill_with_context.setdefault("symbol", trade.get("pair"))
                        fill_with_context.setdefault("side", order.get("side"))
                        try:
                            is_new = self.ledger.insert_fill_if_new(order_id, fill_with_context)
                            if is_new:
                                summary["fills_new"] += 1
                                logger.debug("New fill recorded: order=%s fill=%s",
                                             order_id[:8], fill_with_context.get("id"))
                            else:
                                summary["fills_duplicate"] += 1
                        except Exception as exc:
                            logger.warning("Failed to insert fill for order %s: %s", order_id, exc)
                            summary["errors"].append(f"insert_fill:{exc}")

            # Step 5: update reserved funds for open buy orders
            if self.reserved_ledger is not None:
                self._update_reserved_funds(open_orders)

        finally:
            # Step 6: heartbeat — always written, even if steps above failed
            # Guardian depends on this to know Reconciler is alive.
            hb_ts = self.ledger.write_heartbeat(self._loop_count)
            summary["heartbeat_ts"] = hb_ts

        return summary

    def _update_reserved_funds(self, open_orders: list[dict[str, Any]]) -> None:
        """
        Update reserved funds ledger from current open buy orders.
        Only buy orders reserve quote currency.
        """
        for order in open_orders:
            order_id = _s(order.get("id"))
            side = _s(order.get("side", "")).lower()
            if side != "buy" or not order_id:
                continue
            remaining = _f(order.get("remaining"), 0.0)
            price = _f(order.get("price"), 0.0)
            reserved_quote = remaining * price
            pair = _s(order.get("symbol") or order.get("pair") or "UNKNOWN")
            try:
                self.reserved_ledger.upsert_reserved(order_id, pair, reserved_quote)
            except Exception as exc:
                logger.warning("Failed to update reserved funds for order %s: %s", order_id, exc)

    @staticmethod
    def _extract_list(response: Any, key: str) -> list:
        """
        Extract a list from a Freqtrade REST response dict.
        Freqtrade wraps lists in {"trades": [...]} or {"orders": [...]}.
        Falls back to treating the response as a list directly.
        """
        if isinstance(response, list):
            return response
        if isinstance(response, dict):
            return response.get(key, []) or []
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _s(val: Any) -> str:
    """Safely convert to string. None -> empty string."""
    return str(val) if val is not None else ""


def _f(val: Any, default: float = 0.0) -> float:
    """Safely convert to float. None/bad input -> default."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default