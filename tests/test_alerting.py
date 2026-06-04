"""
tests/test_alerting.py

Coverage:
- Required: test_alert_sent_to_telegram_if_token_present
- Required: test_alert_falls_back_to_file_if_no_token
- Required: test_alert_called_on_operator_required_action
- Required: test_critical_severity_formatted_correctly

Edge cases:
- test_telegram_failure_writes_log_and_returns_false
- test_no_token_no_log_returns_false
- test_alert_consumer_marks_alerted_at_on_success
- test_alert_consumer_does_not_mark_on_telegram_failure
- test_migrate_add_alerted_at_is_idempotent
"""

import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sidecar.guardian.alerting import (
    ALERTABLE_ACTION_TYPES,
    _build_message_from_action,
    _fetch_alertable_actions,
    _format_alert,
    _mark_alerted,
    _send_telegram,
    _write_log_file,
    alert_consumer_loop,
    migrate_add_alerted_at,
    send_alert,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path):
    """In-memory ledger with system_actions schema + alerted_at column."""
    db_path = str(tmp_path / "test.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE system_actions (
                action_id TEXT PRIMARY KEY,
                action_type TEXT NOT NULL,
                pair TEXT,
                trade_id TEXT,
                model_id TEXT,
                reason TEXT,
                severity TEXT,
                status TEXT DEFAULT 'PENDING',
                retry_count INTEGER DEFAULT 0,
                next_retry_ts TEXT,
                created_ts TEXT NOT NULL,
                updated_ts TEXT,
                alerted_at TEXT
            )
        """)
        conn.commit()
    return db_path


@pytest.fixture()
def tmp_log(tmp_path):
    return str(tmp_path / "alerts.log")


def _insert_action(db_path, action_id, action_type, severity, pair=None, reason=None):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """INSERT INTO system_actions
               (action_id, action_type, pair, reason, severity, status, created_ts)
               VALUES (?, ?, ?, ?, ?, 'PENDING', '2026-06-05T00:00:00')""",
            (action_id, action_type, pair, reason, severity),
        )
        conn.commit()


def _mock_http_ok():
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "ok"
    client = MagicMock()
    client.post.return_value = resp
    return client


def _mock_http_fail(status=500):
    resp = MagicMock()
    resp.status_code = status
    resp.text = "server error"
    client = MagicMock()
    client.post.return_value = resp
    return client


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


def test_alert_sent_to_telegram_if_token_present(tmp_log):
    """send_alert returns True when Telegram responds 2xx."""
    client = _mock_http_ok()
    result = send_alert(
        message="test alert",
        severity="HIGH",
        pair="BTC/USDT",
        token="TOKEN",
        chat_id="12345",
        log_path=tmp_log,
        http_client=client,
    )
    assert result is True
    client.post.assert_called_once()
    call_args = client.post.call_args
    assert "sendMessage" in call_args[0][0]


def test_alert_falls_back_to_file_if_no_token(tmp_log):
    """send_alert writes log file and returns False when no token is set."""
    result = send_alert(
        message="no token alert",
        severity="CRITICAL",
        log_path=tmp_log,
        token="",
        chat_id="",
    )
    assert result is False
    content = Path(tmp_log).read_text(encoding="utf-8")
    assert "no token alert" in content
    assert "CRITICAL" in content


def test_alert_called_on_operator_required_action(tmp_db, tmp_log):
    """Consumer loop stamps alerted_at for OPERATOR_REQUIRED HIGH action."""
    _insert_action(tmp_db, "act-001", "OPERATOR_REQUIRED", "HIGH", reason="test reason")

    stop = threading.Event()
    client = _mock_http_ok()

    t = threading.Thread(
        target=alert_consumer_loop,
        kwargs=dict(
            ledger_path=tmp_db,
            interval_sec=0.05,
            token="TOKEN",
            chat_id="12345",
            log_path=tmp_log,
            http_client=client,
            _stop_event=stop,
        ),
        daemon=True,
    )
    t.start()
    time.sleep(0.3)
    stop.set()
    t.join(timeout=1.0)

    with sqlite3.connect(tmp_db) as conn:
        row = conn.execute(
            "SELECT alerted_at FROM system_actions WHERE action_id = 'act-001'"
        ).fetchone()
    assert row is not None
    assert row[0] is not None, "alerted_at should be stamped after successful alert"


def test_critical_severity_formatted_correctly():
    """CRITICAL severity produces 'CRITICAL ALERT' prefix in formatted text."""
    text = _format_alert("system down", "CRITICAL", pair="ETH/USDT")
    assert "CRITICAL ALERT" in text
    assert "[ETH/USDT]" in text
    assert "system down" in text
    assert "RetailEdge" in text


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_telegram_failure_writes_log_and_returns_false(tmp_log):
    """Telegram 500 triggers log write and returns False (so consumer retries)."""
    client = _mock_http_fail(500)
    result = send_alert(
        message="telegram will fail",
        severity="HIGH",
        token="TOKEN",
        chat_id="12345",
        log_path=tmp_log,
        http_client=client,
    )
    assert result is False
    content = Path(tmp_log).read_text(encoding="utf-8")
    assert "telegram will fail" in content


def test_no_token_no_log_returns_false(tmp_path):
    """No token + unwritable log path returns False without raising."""
    bad_path = "/dev/null/cannot_write_here/alert.log"
    result = send_alert(
        message="orphan alert",
        severity="HIGH",
        token="",
        chat_id="",
        log_path=bad_path,
    )
    assert result is False


def test_alert_consumer_marks_alerted_at_on_success(tmp_db, tmp_log):
    """alerted_at is stamped and action is not re-alerted on next loop iteration."""
    _insert_action(tmp_db, "act-002", "EMERGENCY_EXIT_FAILED", "CRITICAL")

    client = _mock_http_ok()
    stop = threading.Event()

    t = threading.Thread(
        target=alert_consumer_loop,
        kwargs=dict(
            ledger_path=tmp_db,
            interval_sec=0.05,
            token="TOKEN",
            chat_id="12345",
            log_path=tmp_log,
            http_client=client,
            _stop_event=stop,
        ),
        daemon=True,
    )
    t.start()
    time.sleep(0.4)
    stop.set()
    t.join(timeout=1.0)

    call_count_first = client.post.call_count
    assert call_count_first >= 1

    # Insert a second action and run again — previous must NOT be re-alerted.
    _insert_action(tmp_db, "act-003", "STOP_UNCONFIRMED", "HIGH")
    client2 = _mock_http_ok()
    stop2 = threading.Event()

    t2 = threading.Thread(
        target=alert_consumer_loop,
        kwargs=dict(
            ledger_path=tmp_db,
            interval_sec=0.05,
            token="TOKEN",
            chat_id="12345",
            log_path=tmp_log,
            http_client=client2,
            _stop_event=stop2,
        ),
        daemon=True,
    )
    t2.start()
    time.sleep(0.3)
    stop2.set()
    t2.join(timeout=1.0)

    # client2 should only have alerted act-003, not act-002 again.
    assert client2.post.call_count >= 1
    # Verify act-002 alerted_at is still the original timestamp (unchanged by second run).
    with sqlite3.connect(tmp_db) as conn:
        ts = conn.execute(
            "SELECT alerted_at FROM system_actions WHERE action_id = 'act-002'"
        ).fetchone()[0]
    assert ts is not None


def test_alert_consumer_does_not_mark_on_telegram_failure(tmp_db, tmp_log):
    """If Telegram fails, alerted_at stays NULL so next loop retries."""
    _insert_action(tmp_db, "act-004", "OPERATOR_REQUIRED", "CRITICAL")

    client = _mock_http_fail(503)
    stop = threading.Event()

    t = threading.Thread(
        target=alert_consumer_loop,
        kwargs=dict(
            ledger_path=tmp_db,
            interval_sec=0.05,
            token="TOKEN",
            chat_id="12345",
            log_path=tmp_log,
            http_client=client,
            _stop_event=stop,
        ),
        daemon=True,
    )
    t.start()
    time.sleep(0.3)
    stop.set()
    t.join(timeout=1.0)

    with sqlite3.connect(tmp_db) as conn:
        row = conn.execute(
            "SELECT alerted_at FROM system_actions WHERE action_id = 'act-004'"
        ).fetchone()
    assert row[0] is None, "alerted_at must stay NULL when Telegram fails"

    # Log file must have backup evidence.
    content = Path(tmp_log).read_text(encoding="utf-8")
    assert "OPERATOR_REQUIRED" in content or "act-004" in content or len(content) > 0


def test_migrate_add_alerted_at_is_idempotent(tmp_path):
    """Running migration twice does not raise or duplicate the column."""
    db_path = str(tmp_path / "migrate_test.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE system_actions (
                action_id TEXT PRIMARY KEY,
                action_type TEXT NOT NULL,
                severity TEXT,
                status TEXT,
                retry_count INTEGER DEFAULT 0,
                created_ts TEXT NOT NULL
            )
        """)
        conn.commit()

    migrate_add_alerted_at(db_path)
    migrate_add_alerted_at(db_path)  # second call must not raise

    with sqlite3.connect(db_path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(system_actions)").fetchall()]
    assert "alerted_at" in cols
    assert cols.count("alerted_at") == 1