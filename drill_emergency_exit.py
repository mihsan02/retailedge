"""
S4-2: Emergency Exit Drill Script
Jalankan: python drill_emergency_exit.py

Script ini:
1. Menjalankan 3 scenario drill dengan mock (tidak ada real exchange call)
2. Mencatat hasil ke deployment_audit di ledger
3. Print status setiap scenario
4. Exit code 0 jika semua pass, 1 jika ada yang fail

PENTING: Script ini memakai mock ft_client dan exchange_client.
emergency_exit_enabled=True HANYA di dalam script ini — venue_costs.json tidak diubah.
"""

import os
import sys
import sqlite3
import uuid
import datetime
from unittest.mock import MagicMock

# Pastikan repo root ada di path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from sidecar.guardian.emergency_exit import (
    emergency_exit_cascade,
    EmergencyExitPolicy,
    Trade,
    STATUS_DONE_MARKET,
    STATUS_DONE_LIMIT,
    STATUS_FAILED,
)

LEDGER_PATH = os.environ.get("LEDGER_DB_PATH", "./ledger/retailedge.db")
DRILL_PAIR = "BTC/USDT"


# ---------------------------------------------------------------------------
# Ledger helpers
# ---------------------------------------------------------------------------

def _init_audit_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS deployment_audit (
            audit_id    TEXT PRIMARY KEY,
            event_type  TEXT NOT NULL,
            scenario    TEXT,
            result      TEXT,
            notes       TEXT,
            created_ts  TEXT NOT NULL
        )
    """)
    conn.commit()


def record_drill(scenario: str, result: str, notes: str = ""):
    conn = sqlite3.connect(LEDGER_PATH)
    _init_audit_table(conn)
    conn.execute(
        "INSERT INTO deployment_audit VALUES (?,?,?,?,?,?)",
        (
            str(uuid.uuid4()),
            "EMERGENCY_EXIT_DRILL",
            scenario,
            result,
            notes,
            datetime.datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Mock builders
# ---------------------------------------------------------------------------

class FakeResp:
    def __init__(self, code):
        self.status_code = code
    def __repr__(self):
        return f"FakeResp({self.status_code})"


def _make_policy():
    # emergency_exit_enabled=True ONLY for drill — never set this in venue_costs.json
    # until all 3 scenarios pass and are recorded in deployment_audit
    return EmergencyExitPolicy(
        emergency_exit_enabled=True,
        market_max_retries=2,
        retry_delays_sec=[0.0, 0.0],   # no real sleep during drill
    )


def _make_exchange():
    ex = MagicMock()
    ex.fetch_order_book.return_value = {"bids": [[50000.0, 1.0]]}
    return ex


def _make_bus():
    bus = MagicMock()
    bus.post.return_value = "drill_action_id"
    return bus


# ---------------------------------------------------------------------------
# Drill scenarios
# ---------------------------------------------------------------------------

def scenario_A():
    """Market order succeeds on first attempt."""
    print("\n[A] Scenario: market order SUCCESS on first attempt")
    ft = MagicMock()
    ft.ping_alive.return_value = True
    ft.forceexit.return_value = FakeResp(200)

    result = emergency_exit_cascade(
        Trade("drill_A", DRILL_PAIR),
        ft, _make_exchange(), _make_bus(), _make_policy(),
        sleep_fn=lambda s: None,
    )

    print(f"    Result  : {result}")
    print(f"    Expected: {STATUS_DONE_MARKET}")
    ok = result == STATUS_DONE_MARKET
    print(f"    Status  : {'PASS' if ok else 'FAIL'}")
    return ok, result


def scenario_B():
    """Both market attempts fail, aggressive limit succeeds."""
    print("\n[B] Scenario: market FAIL x2 → aggressive limit SUCCESS")
    ft = MagicMock()
    ft.ping_alive.return_value = True
    ft.forceexit.side_effect = [
        FakeResp(500),   # market attempt 1
        FakeResp(500),   # market attempt 2
        FakeResp(200),   # aggressive limit
    ]

    result = emergency_exit_cascade(
        Trade("drill_B", DRILL_PAIR),
        ft, _make_exchange(), _make_bus(), _make_policy(),
        sleep_fn=lambda s: None,
    )

    print(f"    Result  : {result}")
    print(f"    Expected: {STATUS_DONE_LIMIT}")
    ok = result == STATUS_DONE_LIMIT
    print(f"    Status  : {'PASS' if ok else 'FAIL'}")
    return ok, result


def scenario_C():
    """All attempts fail — OPERATOR_REQUIRED, Decision Bus alerted."""
    print("\n[C] Scenario: market FAIL x2 + limit FAIL → OPERATOR_REQUIRED")
    ft = MagicMock()
    ft.ping_alive.return_value = True
    ft.forceexit.side_effect = [
        FakeResp(500),   # market attempt 1
        FakeResp(500),   # market attempt 2
        FakeResp(500),   # aggressive limit
    ]
    bus = _make_bus()

    result = emergency_exit_cascade(
        Trade("drill_C", DRILL_PAIR),
        ft, _make_exchange(), bus, _make_policy(),
        sleep_fn=lambda s: None,
    )

    bus_called = bus.post.called
    bus_action = bus.post.call_args.args[0] if bus_called else None

    print(f"    Result         : {result}")
    print(f"    Expected       : {STATUS_FAILED}")
    print(f"    Bus action     : {bus_action}")
    print(f"    Expected action: EMERGENCY_EXIT_FAILED")
    ok = (
        result == STATUS_FAILED
        and bus_called
        and bus_action == "EMERGENCY_EXIT_FAILED"
    )
    print(f"    Status         : {'PASS' if ok else 'FAIL'}")
    return ok, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("RetailEdge — Emergency Exit Cascade Drill (S4-2)")
    print(f"Ledger: {LEDGER_PATH}")
    print("=" * 60)

    # Confirm ledger dir exists
    ledger_dir = os.path.dirname(LEDGER_PATH)
    if ledger_dir and not os.path.exists(ledger_dir):
        print(f"ERROR: Ledger directory not found: {ledger_dir}")
        print("Jalankan dari root repo dan pastikan ledger/ sudah ada.")
        sys.exit(1)

    results = {}

    ok_A, r_A = scenario_A()
    record_drill("A_market_success", r_A, "market order 2xx on first attempt")
    results["A"] = ok_A

    ok_B, r_B = scenario_B()
    record_drill("B_market_fail_limit_success", r_B, "2x market 500, limit 200")
    results["B"] = ok_B

    ok_C, r_C = scenario_C()
    record_drill("C_all_fail_operator_required", r_C, "2x market 500, limit 500, bus alerted")
    results["C"] = ok_C

    # Summary
    print("\n" + "=" * 60)
    print("DRILL SUMMARY")
    print("=" * 60)
    all_pass = all(results.values())
    for k, v in results.items():
        print(f"  Scenario {k}: {'PASS' if v else 'FAIL'}")

    print()
    if all_pass:
        print("DRILL COMPLETE — semua 3 scenario PASS")
        print("Hasil tercatat di deployment_audit.")
        print()
        print("Verifikasi:")
        print(f"  sqlite3 {LEDGER_PATH} \"SELECT scenario, result, created_ts FROM deployment_audit WHERE event_type='EMERGENCY_EXIT_DRILL';\"")
        print()
        print("Next: emergency_exit_cascade_drill_pass = true")
        print("Next: lanjut ke S4-3 (champion rollback drill)")
    else:
        print("DRILL FAILED — ada scenario yang tidak pass.")
        print("Jangan lanjut ke S4-3 sampai semua scenario pass.")
        sys.exit(1)


if __name__ == "__main__":
    main()
