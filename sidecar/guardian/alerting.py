"""
sidecar/guardian/alerting.py

Standalone alert consumer for the Decision Bus.
Does NOT import guardian.py. Zero coupling to Freqtrade REST.

Design contract:
- Polls system_actions for un-alerted HIGH/CRITICAL actions.
- send_alert() tries Telegram first.
- On Telegram failure: writes log file as backup evidence, returns False.
- Returns True only on confirmed Telegram 2xx.
- Caller (alert_consumer_loop) retries on False because alerted_at stays NULL.
- alerted_at column tracks state; does NOT touch the existing status column.

Schema migration required (run once before deploying):
    ALTER TABLE system_actions ADD COLUMN alerted_at TEXT;
"""

import logging
import os
import sqlite3
import time
from datetime import datetime, timezone

try:
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

ALERTABLE_ACTION_TYPES = frozenset({
    "OPERATOR_REQUIRED",
    "EMERGENCY_EXIT_FAILED",
    "STOP_UNCONFIRMED",
})

ALERTABLE_SEVERITIES = frozenset({"HIGH", "CRITICAL"})

TELEGRAM_API_BASE = "https://api.telegram.org"
TELEGRAM_TIMEOUT_SEC = 5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def send_alert(
    message: str,
    severity: str,
    pair: str | None = None,
    *,
    token: str | None = None,
    chat_id: str | None = None,
    log_path: str | None = None,
    http_client=None,
) -> bool:
    """
    Send an operator alert.

    Returns True only if Telegram confirmed 2xx.
    On Telegram failure: writes to log file and returns False so the caller retries.

    Parameters
    ----------
    message   : Human-readable alert body.
    severity  : INFO / HIGH / CRITICAL.
    pair      : Optional trading pair for context.
    token     : Telegram bot token. Defaults to env TELEGRAM_BOT_TOKEN.
    chat_id   : Telegram chat id. Defaults to env TELEGRAM_CHAT_ID.
    log_path  : Log file path. Defaults to env LOG_PATH or ./logs/operator_alerts.log.
    http_client : Injectable HTTP client (for tests). Defaults to requests module.
    """
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    log_path = log_path or os.environ.get("LOG_PATH", "./logs/operator_alerts.log")

    text = _format_alert(message, severity, pair)

    if token and chat_id:
        success = _send_telegram(token, chat_id, text, http_client=http_client)
        if success:
            return True
        # Telegram failed — write backup evidence then report failure so caller retries.
        _write_log_file(text, log_path)
        logger.warning("Telegram failed; alert written to %s. Will retry.", log_path)
        return False

    # No token configured — log file is the primary channel.
    written = _write_log_file(text, log_path)
    if not written:
        logger.error("Alert lost: Telegram not configured and log write failed.")
    return False


def alert_consumer_loop(
    ledger_path: str,
    interval_sec: float = 5.0,
    *,
    token: str | None = None,
    chat_id: str | None = None,
    log_path: str | None = None,
    http_client=None,
    _stop_event=None,
) -> None:
    """
    Standalone consumer loop. Run in its own thread or process.

    Polls system_actions for unalerted HIGH/CRITICAL actions of the three
    alertable types. On send success, stamps alerted_at. On failure, leaves
    alerted_at NULL so the next iteration retries.

    _stop_event: threading.Event — if set, loop exits cleanly (used in tests).
    """
    logger.info("Alert consumer loop started. DB: %s, interval: %ss", ledger_path, interval_sec)

    while True:
        if _stop_event is not None and _stop_event.is_set():
            break

        try:
            pending = _fetch_alertable_actions(ledger_path)
            for action in pending:
                msg = _build_message_from_action(action)
                severity = action.get("severity", "HIGH")
                pair = action.get("pair")
                sent = send_alert(
                    msg,
                    severity,
                    pair,
                    token=token,
                    chat_id=chat_id,
                    log_path=log_path,
                    http_client=http_client,
                )
                if sent:
                    _mark_alerted(ledger_path, action["action_id"])
        except Exception:
            logger.exception("Alert consumer loop error — continuing.")

        time.sleep(interval_sec)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_alert(message: str, severity: str, pair: str | None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pair_str = f" [{pair}]" if pair else ""
    prefix = {
        "CRITICAL": "CRITICAL ALERT",
        "HIGH": "HIGH ALERT",
        "INFO": "INFO",
    }.get(severity.upper(), "ALERT")
    return f"[RetailEdge] {prefix}{pair_str} {ts}\n{message}"


def _send_telegram(
    token: str,
    chat_id: str,
    text: str,
    *,
    http_client=None,
) -> bool:
    """
    POST to Telegram sendMessage. Returns True only on HTTP 2xx.
    Never raises — all exceptions are caught and logged.
    """
    client = http_client if http_client is not None else _requests
    if client is None:
        logger.error("requests library not available — cannot send Telegram alert.")
        return False

    url = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    try:
        resp = client.post(url, json=payload, timeout=TELEGRAM_TIMEOUT_SEC)
        if 200 <= resp.status_code < 300:
            logger.info("Telegram alert sent. status=%s", resp.status_code)
            return True
        logger.warning("Telegram non-2xx: %s %s", resp.status_code, resp.text[:200])
        return False
    except Exception as exc:
        logger.warning("Telegram request failed: %s", exc)
        return False


def _write_log_file(text: str, log_path: str) -> bool:
    """
    Append alert text to log file. Returns True on success, False on any error.
    Creates parent directories if they do not exist.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n---\n")
        logger.info("Alert written to %s", log_path)
        return True
    except Exception as exc:
        logger.error("Failed to write alert log: %s", exc)
        return False


def _fetch_alertable_actions(ledger_path: str) -> list[dict]:
    """
    Return system_actions rows where:
    - alerted_at IS NULL
    - action_type IN ALERTABLE_ACTION_TYPES
    - severity IN ALERTABLE_SEVERITIES
    """
    placeholders_type = ",".join("?" * len(ALERTABLE_ACTION_TYPES))
    placeholders_sev = ",".join("?" * len(ALERTABLE_SEVERITIES))
    sql = f"""
        SELECT action_id, action_type, pair, trade_id, reason, severity, created_ts
        FROM system_actions
        WHERE alerted_at IS NULL
          AND action_type IN ({placeholders_type})
          AND severity IN ({placeholders_sev})
        ORDER BY created_ts ASC
    """
    params = list(ALERTABLE_ACTION_TYPES) + list(ALERTABLE_SEVERITIES)

    with sqlite3.connect(ledger_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def _mark_alerted(ledger_path: str, action_id: str) -> None:
    """Stamp alerted_at. Does not touch status — Guardian owns that column."""
    ts = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(ledger_path) as conn:
        conn.execute(
            "UPDATE system_actions SET alerted_at = ? WHERE action_id = ?",
            (ts, action_id),
        )
        conn.commit()
    logger.debug("Marked alerted: %s at %s", action_id, ts)


def _build_message_from_action(action: dict) -> str:
    parts = [f"action_type: {action.get('action_type', 'UNKNOWN')}"]
    if action.get("trade_id"):
        parts.append(f"trade_id: {action['trade_id']}")
    if action.get("reason"):
        parts.append(f"reason: {action['reason']}")
    if action.get("created_ts"):
        parts.append(f"created: {action['created_ts']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Schema migration helper (idempotent — safe to run on existing DB)
# ---------------------------------------------------------------------------


def migrate_add_alerted_at(ledger_path: str) -> None:
    """
    Adds alerted_at column to system_actions if it does not exist.
    Call once at startup before alert_consumer_loop.
    """
    with sqlite3.connect(ledger_path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(system_actions)").fetchall()]
        if "alerted_at" not in cols:
            conn.execute("ALTER TABLE system_actions ADD COLUMN alerted_at TEXT")
            conn.commit()
            logger.info("Migrated: added alerted_at to system_actions.")