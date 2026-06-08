#!/usr/bin/env python3
"""
sidecar/run_reconciler.py

Docker entrypoint for Reconciler service.
Does NOT modify reconciler.py.

Constructor signatures (confirmed):
  FreqtradeClient(base_url, username, password) — all optional, reads from env.
  ReconcilerLedger(db_path)
  Reconciler(ft_client, ledger, reserved_ledger=None, poll_interval_sec=30.0)
"""

import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RECONCILER] %(levelname)s %(message)s",
)
log = logging.getLogger("reconciler.runner")


def main() -> None:
    ledger_path = os.environ.get("LEDGER_DB_PATH", "/workdir/ledger/retailedge.db")
    poll_interval = float(os.environ.get("RECONCILER_POLL_INTERVAL_SEC", "30"))

    log.info("Reconciler runner starting")
    log.info("LEDGER_DB_PATH=%s", ledger_path)
    log.info("FREQTRADE_API_URL=%s", os.environ.get("FREQTRADE_API_URL", "http://freqtrade:8080"))
    log.info("poll_interval=%.0fs", poll_interval)

    from sidecar.guardian.ft_client import FreqtradeClient
    from sidecar.reconciler.reconciler import Reconciler, ReconcilerLedger

    ft_client = FreqtradeClient()          # reads FREQTRADE_API_URL/USER/PASS from env
    ledger = ReconcilerLedger(ledger_path)

    reconciler = Reconciler(
        ft_client=ft_client,
        ledger=ledger,
        reserved_ledger=None,
        poll_interval_sec=poll_interval,
    )
    reconciler.run_loop()


if __name__ == "__main__":
    main()