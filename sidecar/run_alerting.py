#!/usr/bin/env python3
"""
sidecar/run_alerting.py

Docker entrypoint for Alerting service.
Does NOT modify alerting.py.

Function signatures (confirmed):
  migrate_add_alerted_at(ledger_path) — idempotent
  alert_consumer_loop(ledger_path, interval_sec, token, chat_id, log_path)
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ALERTING] %(levelname)s %(message)s",
)
log = logging.getLogger("alerting.runner")


def main() -> None:
    ledger_path = os.environ.get("LEDGER_DB_PATH", "/workdir/ledger/retailedge.db")
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    log_path = os.environ.get("ALERT_LOG_PATH", "/workdir/logs/operator_alerts.log")
    interval_sec = int(os.environ.get("ALERT_POLL_INTERVAL_SEC", "5"))

    log.info("Alerting runner starting")
    log.info("LEDGER_DB_PATH=%s", ledger_path)
    log.info("Telegram token present: %s", bool(token))
    log.info("Alert log path: %s", log_path)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    from sidecar.guardian.alerting import migrate_add_alerted_at, alert_consumer_loop

    try:
        migrate_add_alerted_at(ledger_path)
        log.info("migrate_add_alerted_at: OK")
    except Exception as exc:
        log.error("Migration failed: %s", exc)
        sys.exit(1)

    log.info("Starting alert_consumer_loop (interval=%ds)", interval_sec)
    alert_consumer_loop(
        ledger_path=ledger_path,
        interval_sec=interval_sec,
        token=token,
        chat_id=chat_id,
        log_path=log_path,
    )


if __name__ == "__main__":
    main()