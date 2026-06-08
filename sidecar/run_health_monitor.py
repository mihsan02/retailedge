"""
sidecar/run_health_monitor.py

Docker entrypoint for Health Monitor service.
No __main__ block exists in monitor.py — this wrapper provides it.

Pattern: identical to run_reconciler.py and run_alerting.py.
Reads LEDGER_DB_PATH from env. Runs HealthMonitor.run_loop() indefinitely.
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [HEALTH_MONITOR] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def main() -> None:
    db_path = os.environ.get("LEDGER_DB_PATH", "/ledger/retailedge.db")
    poll_interval = float(os.environ.get("HEALTH_MONITOR_INTERVAL_SEC", "3600"))

    logger.info("Health Monitor runner starting")
    logger.info("db_path=%s", db_path)
    logger.info("poll_interval=%ss", poll_interval)

    from health_monitor.monitor import HealthMonitor, HealthMonitorLedger

    ledger = HealthMonitorLedger(db_path)
    monitor = HealthMonitor(ledger=ledger, poll_interval_sec=poll_interval)

    logger.info("Health Monitor starting poll loop, interval=%ss", poll_interval)
    monitor.run_loop()


if __name__ == "__main__":
    main()
