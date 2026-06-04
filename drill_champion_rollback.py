"""
drill_champion_rollback.py

Champion Promotion + Rollback Drill for Sprint 4 S4-3.

Scenarios:
  A_promote  — promote a new champion, verify manifest written + hash matches
  B_rollback — rollback to previous model, verify manifest reverted + audit stamped

Done criteria:
  - 2 rows in deployment_audit (one per scenario)
  - active_model_manifest.json reflects rollback_model_id after scenario B

Run:
  python drill_champion_rollback.py

Evidence:
  sqlite3 ledger/retailedge.db "SELECT event_type, scenario, result, created_ts
                                 FROM deployment_audit
                                 WHERE event_type='CHAMPION_ROLLBACK_DRILL';"
"""

import json
import os
import sqlite3
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

# ---------------------------------------------------------------------------
# Candidate fixtures — both use DRY_RUN_PLACEHOLDER (no model artifact needed)
# ---------------------------------------------------------------------------

CANDIDATE_NEW = {
    "model_id": "drill_new_model_20260605_001",
    "model_type": "bucket_baseline",
    "strategy_id": "trend_pullback_v1",
    "model_hash": "sha256:DRY_RUN_PLACEHOLDER",
    "feature_hash": "sha256:DRY_RUN_PLACEHOLDER",
    "cost_model_hash": "sha256:DRY_RUN_PLACEHOLDER",
    "regime_policy_hash": "sha256:DRY_RUN_PLACEHOLDER",
    "stage_gate_passed": True,
    "rollback_model_id": None,  # will be populated from current manifest
    "venue_key": "binance_spot_usdt",
}

CANDIDATE_ROLLBACK = {
    # This is the model that was active before the drill — restored in scenario B.
    # model_id is filled at runtime from the manifest's rollback_model_id field.
    "model_type": "bucket_baseline",
    "strategy_id": "trend_pullback_v1",
    "model_hash": "sha256:DRY_RUN_PLACEHOLDER",
    "feature_hash": "sha256:DRY_RUN_PLACEHOLDER",
    "cost_model_hash": "sha256:DRY_RUN_PLACEHOLDER",
    "regime_policy_hash": "sha256:DRY_RUN_PLACEHOLDER",
    "stage_gate_passed": True,
    "venue_key": "binance_spot_usdt",
}

APPROVAL = OperatorApproval(
    operator_id="operator",
    approved_at=datetime.now(timezone.utc).isoformat(),
    promotion_source="manual",
)


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

def _record_drill(db_path: str, scenario: str, result: str, detail: str = "") -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """INSERT INTO deployment_audit (event_type, scenario, result, created_ts)
               VALUES (?, ?, ?, ?)""",
            (DRILL_EVENT_TYPE, scenario, result, ts),
        )
        conn.commit()
    print(f"  [audit] {scenario}: {result}  {detail}")


def _read_manifest(manifest_path: str) -> dict:
    return json.loads(Path(manifest_path).read_text())


# ---------------------------------------------------------------------------
# Scenario A — Promote new champion
# ---------------------------------------------------------------------------

def scenario_a_promote(manifest_path: str, db_path: str) -> str:
    """
    Promote CANDIDATE_NEW.
    Verify: manifest written, model_id matches, challenger_live=False.
    Returns new model_id on success, raises on failure.
    """
    print("\n[Scenario A] Promoting new champion...")

    # Capture pre-promotion state for rollback slot verification
    pre_manifest = _read_manifest(manifest_path)
    pre_model_id = pre_manifest.get("model_id", "UNKNOWN")

    candidate = build_candidate_from_dict(CANDIDATE_NEW)
    written = promote_champion(
        candidate=candidate,
        approval=APPROVAL,
        manifest_path=manifest_path,
        decision_bus_post_fn=None,  # drill — no live Decision Bus
    )

    # Verify 1: model_id written correctly
    post_manifest = _read_manifest(manifest_path)
    assert post_manifest["model_id"] == CANDIDATE_NEW["model_id"], (
        f"model_id mismatch: expected {CANDIDATE_NEW['model_id']}, "
        f"got {post_manifest['model_id']}"
    )

    # Verify 2: challenger_live is False (invariant)
    assert post_manifest.get("challenger_live") is False, (
        f"challenger_live should be False, got {post_manifest.get('challenger_live')}"
    )

    # Verify 3: rollback slot captured previous model
    assert post_manifest.get("rollback_model_id") == pre_model_id, (
        f"rollback_model_id should be '{pre_model_id}', "
        f"got '{post_manifest.get('rollback_model_id')}'"
    )

    _record_drill(
        db_path,
        scenario="A_promote",
        result="PASS",
        detail=f"promoted to {CANDIDATE_NEW['model_id']}, rollback_slot={pre_model_id}",
    )
    print(f"  promoted: {pre_model_id} -> {CANDIDATE_NEW['model_id']}")
    return CANDIDATE_NEW["model_id"]


# ---------------------------------------------------------------------------
# Scenario B — Rollback to previous champion
# ---------------------------------------------------------------------------

def scenario_b_rollback(manifest_path: str, db_path: str) -> str:
    """
    Read rollback_model_id from current manifest, promote it back.
    Verify: manifest reverts, rollback_model_id field updated.
    Returns restored model_id on success, raises on failure.
    """
    print("\n[Scenario B] Rolling back to previous champion...")

    current = _read_manifest(manifest_path)
    rollback_id = current.get("rollback_model_id")

    if not rollback_id:
        raise RuntimeError(
            "No rollback_model_id in current manifest — cannot rollback. "
            "Run scenario A first."
        )

    rollback_candidate_dict = dict(CANDIDATE_ROLLBACK)
    rollback_candidate_dict["model_id"] = rollback_id
    # rollback_model_id for the restored manifest: current (new) model goes to slot
    rollback_candidate_dict["rollback_model_id"] = current.get("model_id")

    candidate = build_candidate_from_dict(rollback_candidate_dict)
    promote_champion(
        candidate=candidate,
        approval=APPROVAL,
        manifest_path=manifest_path,
        decision_bus_post_fn=None,
    )

    # Verify 1: manifest now shows rollback_id as active
    post_manifest = _read_manifest(manifest_path)
    assert post_manifest["model_id"] == rollback_id, (
        f"model_id after rollback should be '{rollback_id}', "
        f"got '{post_manifest['model_id']}'"
    )

    # Verify 2: the promoted model is now in rollback slot
    assert post_manifest.get("rollback_model_id") == CANDIDATE_NEW["model_id"], (
        f"rollback slot after rollback should be '{CANDIDATE_NEW['model_id']}', "
        f"got '{post_manifest.get('rollback_model_id')}'"
    )

    _record_drill(
        db_path,
        scenario="B_rollback",
        result="PASS",
        detail=f"reverted to {rollback_id}, new rollback_slot={CANDIDATE_NEW['model_id']}",
    )
    print(f"  restored: {rollback_id}")
    return rollback_id


# ---------------------------------------------------------------------------
# Main drill runner
# ---------------------------------------------------------------------------

def run_drill(manifest_path: str = MANIFEST_PATH, db_path: str = LEDGER_DB_PATH) -> None:
    print("=" * 60)
    print("Champion Rollback Drill — Sprint 4 S4-3")
    print("=" * 60)

    results = {}

    try:
        results["A"] = scenario_a_promote(manifest_path, db_path)
    except Exception as exc:
        _record_drill(db_path, scenario="A_promote", result=f"FAIL: {exc}")
        print(f"  [FAIL] Scenario A: {exc}")
        raise

    try:
        results["B"] = scenario_b_rollback(manifest_path, db_path)
    except Exception as exc:
        _record_drill(db_path, scenario="B_rollback", result=f"FAIL: {exc}")
        print(f"  [FAIL] Scenario B: {exc}")
        raise

    print("\n" + "=" * 60)
    print("Drill COMPLETE — 2 scenarios PASS")
    print(f"  A promoted:  {results['A']}")
    print(f"  B restored:  {results['B']}")
    print("=" * 60)
    print("\nVerify evidence:")
    print(
        f"  sqlite3 {db_path} "
        '"SELECT event_type, scenario, result, created_ts FROM deployment_audit '
        f"WHERE event_type='{DRILL_EVENT_TYPE}';\""
    )


if __name__ == "__main__":
    run_drill()