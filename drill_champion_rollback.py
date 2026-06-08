"""
drill_champion_rollback.py

Champion Promotion + Rollback Drill — Sprint 4 S4-3 (Session 14 reimplementation).

Scenarios:
  A_promote  — promote drill_new_model, verify manifest + rollback slot
  B_rollback — promote original back, verify manifest reverted

Done criteria:
  - 2 rows in deployment_audit with valid audit_id and model_id in notes
  - manifest active_model = INITIAL after scenario B
  - 213 tests still passing

Run:
  python drill_champion_rollback.py

Evidence:
  sqlite3 ledger/retailedge.db \
    "SELECT audit_id, event_type, scenario, result, notes, created_ts
     FROM deployment_audit
     WHERE event_type='CHAMPION_ROLLBACK_DRILL'
     ORDER BY created_ts DESC LIMIT 2;"
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from deployment.champion_controller import (
    CandidateRecord,
    OperatorApproval,
    build_candidate_from_dict,
    promote_champion,
)

LEDGER_DB_PATH = os.environ.get(
    "LEDGER_DB_PATH", "/workspaces/retailedge/ledger/retailedge.db"
)
MANIFEST_PATH = os.environ.get(
    "ACTIVE_MODEL_MANIFEST_PATH",
    "/workspaces/retailedge/deployment/active_model_manifest.json",
)

DRILL_EVENT_TYPE = "CHAMPION_ROLLBACK_DRILL"

# The legitimate B1 active model — drill must restore this.
INITIAL_MANIFEST_MODEL_ID = "bucket_baseline_dry_run_001"

# Drill-only model — never live, only used to test promote/rollback path.
DRILL_MODEL_ID = "drill_new_model_20260605_001"

CANDIDATE_NEW = {
    "model_id": DRILL_MODEL_ID,
    "model_type": "bucket_baseline",
    "strategy_id": "trend_pullback_v1",
    "model_hash": "sha256:DRY_RUN_PLACEHOLDER",
    "feature_hash": "sha256:DRY_RUN_PLACEHOLDER",
    "cost_model_hash": "sha256:DRY_RUN_PLACEHOLDER",
    "regime_policy_hash": "sha256:DRY_RUN_PLACEHOLDER",
    "stage_gate_passed": True,
    "rollback_model_id": None,  # populated from manifest at runtime
    "venue_key": "binance_spot_usdt",
}

APPROVAL = OperatorApproval(
    operator_id="operator",
    approved_at=datetime.now(timezone.utc).isoformat(),
    promotion_source="manual",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_manifest(manifest_path: str) -> dict:
    return json.loads(Path(manifest_path).read_text())


def _record_drill(
    db_path: str,
    scenario: str,
    result: str,
    model_id: str = "",
) -> None:
    audit_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    notes = f"model_id={model_id}" if model_id else ""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """INSERT INTO deployment_audit
               (audit_id, event_type, scenario, result, notes, created_ts)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (audit_id, DRILL_EVENT_TYPE, scenario, result, notes, ts),
        )
        conn.commit()
    print(f"  [audit] {scenario}: {result} | model={model_id} | id={audit_id[:8]}...")


# ---------------------------------------------------------------------------
# Pre-flight check
# ---------------------------------------------------------------------------

def _assert_manifest_clean(manifest_path: str) -> None:
    """
    Verify manifest is in expected pre-drill state.
    Raises if active model is not INITIAL_MANIFEST_MODEL_ID.
    This prevents drill from running against a corrupted or unknown manifest state.
    """
    m = _read_manifest(manifest_path)
    active = m.get("model_id")
    if active != INITIAL_MANIFEST_MODEL_ID:
        raise RuntimeError(
            f"Pre-drill manifest check FAILED.\n"
            f"  Expected active model: {INITIAL_MANIFEST_MODEL_ID}\n"
            f"  Found: {active}\n"
            f"  Fix: restore manifest manually before running drill."
        )
    print(f"  [preflight] manifest OK — active={active}")


# ---------------------------------------------------------------------------
# Scenario A
# ---------------------------------------------------------------------------

def scenario_a_promote(manifest_path: str, db_path: str) -> str:
    print("\n[Scenario A] Promoting drill model...")

    pre = _read_manifest(manifest_path)
    pre_model_id = pre.get("model_id")

    candidate = build_candidate_from_dict(CANDIDATE_NEW)
    promote_champion(
        candidate=candidate,
        approval=APPROVAL,
        manifest_path=manifest_path,
        decision_bus_post_fn=None,
    )

    post = _read_manifest(manifest_path)

    assert post["model_id"] == DRILL_MODEL_ID, (
        f"model_id mismatch: expected {DRILL_MODEL_ID}, got {post['model_id']}"
    )
    assert post.get("challenger_live") is False, (
        f"challenger_live must be False, got {post.get('challenger_live')}"
    )
    assert post.get("rollback_model_id") == pre_model_id, (
        f"rollback_model_id must be '{pre_model_id}', got '{post.get('rollback_model_id')}'"
    )

    _record_drill(db_path, "A_promote", "PASS", model_id=DRILL_MODEL_ID)
    print(f"  promoted: {pre_model_id} -> {DRILL_MODEL_ID}")
    print(f"  rollback_slot: {post.get('rollback_model_id')}")
    return DRILL_MODEL_ID


# ---------------------------------------------------------------------------
# Scenario B
# ---------------------------------------------------------------------------

def scenario_b_rollback(manifest_path: str, db_path: str) -> str:
    print("\n[Scenario B] Rolling back to original champion...")

    current = _read_manifest(manifest_path)
    rollback_id = current.get("rollback_model_id")

    if not rollback_id:
        raise RuntimeError(
            "No rollback_model_id in manifest — run scenario A first."
        )
    if rollback_id != INITIAL_MANIFEST_MODEL_ID:
        raise RuntimeError(
            f"rollback_model_id is '{rollback_id}', expected '{INITIAL_MANIFEST_MODEL_ID}'.\n"
            f"Manifest state is unexpected — aborting rollback."
        )

    rollback_candidate = build_candidate_from_dict({
        "model_id": rollback_id,
        "model_type": "bucket_baseline",
        "strategy_id": "trend_pullback_v1",
        "model_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "feature_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "cost_model_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "regime_policy_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "stage_gate_passed": True,
        "rollback_model_id": None,  # let promote_champion read from manifest
        "venue_key": "binance_spot_usdt",
    })

    promote_champion(
        candidate=rollback_candidate,
        approval=APPROVAL,
        manifest_path=manifest_path,
        decision_bus_post_fn=None,
    )

    post = _read_manifest(manifest_path)

    assert post["model_id"] == INITIAL_MANIFEST_MODEL_ID, (
        f"After rollback, model_id must be '{INITIAL_MANIFEST_MODEL_ID}', "
        f"got '{post['model_id']}'"
    )
    assert post.get("challenger_live") is False, (
        f"challenger_live must be False, got {post.get('challenger_live')}"
    )
    assert post.get("rollback_model_id") == DRILL_MODEL_ID, (
        f"rollback_slot after B must be '{DRILL_MODEL_ID}', "
        f"got '{post.get('rollback_model_id')}'"
    )

    _record_drill(db_path, "B_rollback", "PASS", model_id=INITIAL_MANIFEST_MODEL_ID)
    print(f"  restored: {INITIAL_MANIFEST_MODEL_ID}")
    print(f"  rollback_slot: {post.get('rollback_model_id')}")
    return INITIAL_MANIFEST_MODEL_ID


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_drill(
    manifest_path: str = MANIFEST_PATH,
    db_path: str = LEDGER_DB_PATH,
) -> None:
    print("=" * 60)
    print("Champion Rollback Drill — Sprint 4 S4-3 (Session 14)")
    print("=" * 60)

    _assert_manifest_clean(manifest_path)

    results = {}

    try:
        results["A"] = scenario_a_promote(manifest_path, db_path)
    except Exception as exc:
        _record_drill(db_path, "A_promote", f"FAIL: {exc}")
        raise

    try:
        results["B"] = scenario_b_rollback(manifest_path, db_path)
    except Exception as exc:
        _record_drill(db_path, "B_rollback", f"FAIL: {exc}")
        raise

    print("\n" + "=" * 60)
    print("Drill COMPLETE — 2 scenarios PASS")
    print(f"  A promoted to:  {results['A']}")
    print(f"  B restored to:  {results['B']}")
    print("=" * 60)
    print("\nEvidence query:")
    print(
        f'  sqlite3 {db_path} \\\n'
        f'    "SELECT audit_id, scenario, result, notes, created_ts\n'
        f'     FROM deployment_audit\n'
        f"     WHERE event_type='{DRILL_EVENT_TYPE}'\n"
        f'     ORDER BY created_ts DESC LIMIT 2;"'
    )


if __name__ == "__main__":
    run_drill()
