"""
sidecar/health_monitor/monitor.py

Health Monitor for RetailEdge.
Single responsibility: compute and write health metrics to ledger every hour.

Metrics tracked (blueprint v1.9 R11):
- net_expectancy:   average PnL per trade (fill-adjusted, after fees)
- max_drawdown:     maximum peak-to-trough drawdown in rolling window
- losing_streak:    current consecutive losing trades
- slippage:         average (actual_fill_price - expected_price) / expected_price
- fill_rate:        limit orders filled / limit orders placed
- regime_label:     current detected regime (trending / ranging / volatile)

Design:
- HealthMonitor.run_once() computes all metrics from ledger data and writes snapshot.
- run_loop() wraps run_once() with a 1-hour sleep (configurable).
- All computation is done from execution_fills and execution_orders tables.
- No external calls — monitor reads only from the local ledger.
- Metrics are append-only: each run writes a new row with timestamp.
  Never overwrites historical metrics (audit trail).

Schema: health_metrics table created idempotently if absent.
"""

import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SEC = 3600  # 1 hour
DRAWDOWN_WINDOW = 100             # rolling window for drawdown computation


class HealthMonitorLedger:
    """
    Read interface for execution data + write interface for health metrics.
    Reads from execution_fills and execution_orders (written by Reconciler).
    Writes to health_metrics (append-only, never updated).
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
            CREATE TABLE IF NOT EXISTS health_metrics (
                metric_id       TEXT PRIMARY KEY,
                snapshot_ts     TEXT NOT NULL,
                net_expectancy  REAL,
                max_drawdown    REAL,
                losing_streak   INTEGER,
                avg_slippage    REAL,
                fill_rate       REAL,
                regime_label    TEXT,
                trade_count     INTEGER,
                window_size     INTEGER,
                notes           TEXT
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

            CREATE TABLE IF NOT EXISTS trade_pnl (
                trade_id        TEXT PRIMARY KEY,
                pair            TEXT,
                pnl_quote       REAL,
                pnl_pct         REAL,
                entry_price     REAL,
                exit_price      REAL,
                fill_cost       REAL,
                fee_quote       REAL,
                closed_ts       TEXT,
                created_ts      TEXT NOT NULL
            );
        """)

    def get_closed_trade_pnl(self, limit: int = DRAWDOWN_WINDOW) -> list[dict[str, Any]]:
        """
        Return recent closed trade PnL records for metric computation.
        Ordered by closed_ts ASC (oldest first for drawdown calculation).
        """
        cur = self._conn.execute(
            """
            SELECT trade_id, pnl_quote, pnl_pct, closed_ts
            FROM trade_pnl
            WHERE closed_ts IS NOT NULL
            ORDER BY closed_ts ASC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_fill_rate_stats(self) -> dict[str, int]:
        """
        Count limit orders placed vs filled for fill rate computation.
        A limit order is 'filled' when status = 'closed' and filled > 0.
        """
        cur = self._conn.execute(
            """
            SELECT
                COUNT(*) as total_limit,
                SUM(CASE WHEN status='closed' AND filled > 0 THEN 1 ELSE 0 END) as filled_limit
            FROM execution_orders
            WHERE order_type = 'limit'
            """
        )
        row = cur.fetchone()
        return {
            "total_limit": row["total_limit"] or 0,
            "filled_limit": row["filled_limit"] or 0,
        }

    def get_slippage_records(self, limit: int = DRAWDOWN_WINDOW) -> list[dict[str, Any]]:
        """
        Return records where price (expected) and average (actual fill) both exist.
        Slippage = (average - price) / price for buys; inverted for sells.
        """
        cur = self._conn.execute(
            """
            SELECT order_id, side, price, average, filled
            FROM execution_orders
            WHERE price IS NOT NULL AND average IS NOT NULL
              AND filled > 0 AND price > 0 AND average > 0
            ORDER BY updated_ts DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    def write_metrics(self, metrics: dict[str, Any]) -> str:
        """
        Append a health metrics snapshot. Returns metric_id.
        Never overwrites existing rows — append-only audit trail.
        """
        import uuid
        metric_id = str(uuid.uuid4())
        now = _now_iso()

        self._conn.execute(
            """
            INSERT INTO health_metrics
                (metric_id, snapshot_ts, net_expectancy, max_drawdown, losing_streak,
                 avg_slippage, fill_rate, regime_label, trade_count, window_size, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metric_id, now,
                metrics.get("net_expectancy"),
                metrics.get("max_drawdown"),
                metrics.get("losing_streak"),
                metrics.get("avg_slippage"),
                metrics.get("fill_rate"),
                metrics.get("regime_label", "unknown"),
                metrics.get("trade_count", 0),
                metrics.get("window_size", DRAWDOWN_WINDOW),
                metrics.get("notes"),
            ),
        )
        return metric_id

    def get_latest_metrics(self) -> Optional[dict[str, Any]]:
        """Return most recent health metrics snapshot."""
        cur = self._conn.execute(
            "SELECT * FROM health_metrics ORDER BY snapshot_ts DESC LIMIT 1"
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def upsert_trade_pnl(
        self,
        trade_id: str,
        pair: str,
        pnl_quote: float,
        pnl_pct: float,
        entry_price: float,
        exit_price: float,
        fill_cost: float,
        fee_quote: float,
        closed_ts: str,
    ) -> None:
        """Insert or update a closed trade PnL record (for test data setup)."""
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO trade_pnl
                (trade_id, pair, pnl_quote, pnl_pct, entry_price, exit_price,
                 fill_cost, fee_quote, closed_ts, created_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trade_id) DO UPDATE SET
                pnl_quote = excluded.pnl_quote,
                pnl_pct   = excluded.pnl_pct,
                closed_ts = excluded.closed_ts
            """,
            (trade_id, pair, pnl_quote, pnl_pct, entry_price, exit_price,
             fill_cost, fee_quote, closed_ts, now),
        )

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# HealthMonitor
# ---------------------------------------------------------------------------

class HealthMonitor:
    """
    Computes and writes health metrics from ledger data.

    ledger: HealthMonitorLedger instance.
    regime_label_fn: callable returning current regime string.
                     Injectable so monitor doesn't depend on regime policy directly.
                     Default: lambda returning "unknown".
    poll_interval_sec: seconds between metric snapshots. Default 3600 (1 hour).
    """

    def __init__(
        self,
        ledger: HealthMonitorLedger,
        regime_label_fn=None,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
    ) -> None:
        self.ledger = ledger
        self.regime_label_fn = regime_label_fn or (lambda: "unknown")
        self.poll_interval_sec = poll_interval_sec

    def run_loop(self) -> None:
        """Production entry point. Writes metrics every poll_interval_sec."""
        logger.info("HealthMonitor starting, interval=%ss", self.poll_interval_sec)
        while True:
            try:
                self.run_once()
            except Exception as exc:
                logger.error("HealthMonitor error: %s", exc, exc_info=True)
            time.sleep(self.poll_interval_sec)

    def run_once(self) -> dict[str, Any]:
        """
        Compute all metrics and write one snapshot to ledger.
        Returns the computed metrics dict.
        """
        trades = self.ledger.get_closed_trade_pnl(limit=DRAWDOWN_WINDOW)
        fill_stats = self.ledger.get_fill_rate_stats()
        slippage_records = self.ledger.get_slippage_records(limit=DRAWDOWN_WINDOW)

        metrics = {
            "net_expectancy":  _compute_net_expectancy(trades),
            "max_drawdown":    _compute_max_drawdown(trades),
            "losing_streak":   _compute_losing_streak(trades),
            "avg_slippage":    _compute_avg_slippage(slippage_records),
            "fill_rate":       _compute_fill_rate(fill_stats),
            "regime_label":    self.regime_label_fn(),
            "trade_count":     len(trades),
            "window_size":     DRAWDOWN_WINDOW,
        }

        metric_id = self.ledger.write_metrics(metrics)
        logger.info(
            "Health snapshot written id=%s net_exp=%.4f drawdown=%.4f streak=%d "
            "slippage=%.4f fill_rate=%.4f regime=%s",
            metric_id[:8],
            metrics["net_expectancy"] or 0,
            metrics["max_drawdown"] or 0,
            metrics["losing_streak"] or 0,
            metrics["avg_slippage"] or 0,
            metrics["fill_rate"] or 0,
            metrics["regime_label"],
        )

        return metrics


# ---------------------------------------------------------------------------
# Metric computation functions — pure, testable independently
# ---------------------------------------------------------------------------

def _compute_net_expectancy(trades: list[dict[str, Any]]) -> Optional[float]:
    """
    Average PnL per trade in quote currency.
    Returns None if no trades.

    net_expectancy > 0 = profitable on average.
    net_expectancy <= 0 = losing or breakeven — alert territory.
    """
    if not trades:
        return None
    pnls = [t["pnl_quote"] for t in trades if t.get("pnl_quote") is not None]
    if not pnls:
        return None
    return sum(pnls) / len(pnls)


def _compute_max_drawdown(trades: list[dict[str, Any]]) -> Optional[float]:
    """
    Maximum peak-to-trough drawdown as a fraction of peak equity.
    Uses cumulative PnL curve from trade history.

    Returns negative float (e.g. -0.15 = 15% drawdown) or None if no trades.

    Why cumulative PnL and not balance:
    We don't have access to full balance history. Cumulative trade PnL
    approximates the drawdown from the trading strategy alone.
    """
    if not trades:
        return None

    pnls = [t.get("pnl_quote", 0.0) or 0.0 for t in trades]
    if not pnls:
        return None

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0

    for pnl in pnls:
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        if peak > 0:
            dd = (cumulative - peak) / peak
            if dd < max_dd:
                max_dd = dd

    return max_dd  # negative or zero


def _compute_losing_streak(trades: list[dict[str, Any]]) -> int:
    """
    Current consecutive losing trades (counted from most recent backwards).
    A trade is losing if pnl_quote < 0.
    Returns 0 if last trade is winning or no trades.
    """
    if not trades:
        return 0

    streak = 0
    # trades are ordered ASC — iterate from end (most recent)
    for trade in reversed(trades):
        pnl = trade.get("pnl_quote")
        if pnl is None:
            break
        if pnl < 0:
            streak += 1
        else:
            break  # streak broken

    return streak


def _compute_avg_slippage(records: list[dict[str, Any]]) -> Optional[float]:
    """
    Average slippage as fraction of expected price.
    slippage_i = (average_fill - expected_price) / expected_price for buys.
                 (expected_price - average_fill) / expected_price for sells.

    Positive slippage = paid more than expected (unfavorable).
    Returns None if no records with both price and average.
    """
    if not records:
        return None

    slippages = []
    for r in records:
        price = r.get("price", 0.0) or 0.0
        average = r.get("average", 0.0) or 0.0
        side = str(r.get("side", "")).lower()

        if price <= 0 or average <= 0:
            continue

        if side == "buy":
            # Paid more than limit price = positive (bad)
            slippage = (average - price) / price
        else:
            # Sold for less than limit price = positive (bad)
            slippage = (price - average) / price

        slippages.append(slippage)

    if not slippages:
        return None
    return sum(slippages) / len(slippages)


def _compute_fill_rate(stats: dict[str, int]) -> Optional[float]:
    """
    Fraction of limit orders that were filled.
    Returns None if no limit orders placed yet.
    Returns float in [0.0, 1.0].
    """
    total = stats.get("total_limit", 0)
    filled = stats.get("filled_limit", 0)
    if total == 0:
        return None
    return filled / total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()