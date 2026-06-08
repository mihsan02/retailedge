"""
tests/test_champion_rollback.py

Required gates (S4-3):
- test_rollback_recorded_in_deployment_audit
- test_manifest_reverts_to_initial_model_id
- test_promote_sets_rollback_slot_correctly
- test_rollback_fails_if_no_rollback_model_id
- test_challenger_live_false_invariant
- test_audit_rows_have_valid_uuid
- test_audit_notes_contain_model_id
- test_preflight_fails_on_wrong_active_model
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from deployment.champion_controller import (
    OperatorApproval,
    build_candidate_from_dict,
    promote_champion,
)
from drill_champion_rollback import (
    DRILL_EVENT_TYPE,
    DRILL_MODEL_ID,
    INITIAL_MANIFEST_MODEL_ID,
    _assert_manifest_clean,
    run_drill,
    scenario_a_promote,
    scenario_b_rollback,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_manifest(tmp_path):
    manifest = {
        "strategy_id": "trend_pullback_v1",
        "model_id": INITIAL_MANIFEST_MODEL_ID,
        "model_type": "bucket_baseline",
        "model_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "feature_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "cost_model_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "regime_policy_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "venue_key": "binance_spot_usdt",
        "approved_by": "operator",
        "approved_at": "2026-06-01T00:00:00+00:00",
        "promotion_source": "manual",
        "rollback_model_id": None,
        "challenger_live": False,
    }
    p = tmp_path / "active_model_manifest.json"
    p.write_text(json.dumps(manifest, indent=2))
    return str(p)


@pytest.fixture()
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE deployment_audit (
                audit_id   TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                scenario   TEXT,
                result     TEXT,
                notes      TEXT,
                created_ts TEXT NOT NULL
            )
        """)
        conn.commit()
    return db_path


def _read_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _audit_rows(db_path: str) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM deployment_audit WHERE event_type = ? ORDER BY created_ts",
            (DRILL_EVENT_TYPE,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------

def test_rollback_recorded_in_deployment_audit(tmp_manifest, tmp_db):
    run_drill(manifest_path=tmp_manifest, db_path=tmp_db)
    rows = _audit_rows(tmp_db)
    scenarios = {r["scenario"] for r in rows}
    assert len(rows) == 2, f"Expected 2 audit rows, got {len(rows)}"
    assert "A_promote" in scenarios
    assert "B_rollback" in scenarios
    assert all(r["result"] == "PASS" for r in rows), (
        f"Not all rows PASS: {[r['result'] for r in rows]}"
    )


def test_manifest_reverts_to_initial_model_id(tmp_manifest, tmp_db):
    run_drill(manifest_path=tmp_manifest, db_path=tmp_db)
    final = _read_manifest(tmp_manifest)
    assert final["model_id"] == INITIAL_MANIFEST_MODEL_ID, (
        f"Expected '{INITIAL_MANIFEST_MODEL_ID}', got '{final['model_id']}'"
    )


def test_promote_sets_rollback_slot_correctly(tmp_manifest, tmp_db):
    scenario_a_promote(tmp_manifest, tmp_db)
    post = _read_manifest(tmp_manifest)
    assert post["rollback_model_id"] == INITIAL_MANIFEST_MODEL_ID
    assert post["model_id"] == DRILL_MODEL_ID


def test_rollback_fails_if_no_rollback_model_id(tmp_path, tmp_db):
    manifest = {
        "model_id": DRILL_MODEL_ID,
        "model_type": "bucket_baseline",
        "model_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "feature_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "cost_model_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "regime_policy_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "stage_gate_passed": True,
        "challenger_live": False,
        "rollback_model_id": None,
        "strategy_id": "trend_pullback_v1",
        "venue_key": "binance_spot_usdt",
    }
    p = tmp_path / "manifest_no_rollback.json"
    p.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="No rollback_model_id"):
        scenario_b_rollback(str(p), tmp_db)


def test_challenger_live_false_invariant(tmp_manifest, tmp_db):
    scenario_a_promote(tmp_manifest, tmp_db)
    post = _read_manifest(tmp_manifest)
    assert post.get("challenger_live") is False


def test_audit_rows_have_valid_uuid(tmp_manifest, tmp_db):
    run_drill(manifest_path=tmp_manifest, db_path=tmp_db)
    rows = _audit_rows(tmp_db)
    for r in rows:
        audit_id = r["audit_id"]
        assert audit_id is not None, "audit_id is NULL"
        assert len(audit_id) == 36, f"audit_id not UUID format: {audit_id}"
        uuid.UUID(audit_id)  # raises ValueError if invalid


def test_audit_notes_contain_model_id(tmp_manifest, tmp_db):
    run_drill(manifest_path=tmp_manifest, db_path=tmp_db)
    rows = _audit_rows(tmp_db)
    for r in rows:
        notes = r.get("notes", "")
        assert "model_id=" in notes, (
            f"notes should contain 'model_id=', got: '{notes}'"
        )


def test_preflight_fails_on_wrong_active_model(tmp_path):
    manifest = {
        "model_id": "some_unexpected_model",
        "model_type": "bucket_baseline",
        "model_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "feature_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "cost_model_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "regime_policy_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "stage_gate_passed": True,
        "challenger_live": False,
        "rollback_model_id": None,
        "strategy_id": "trend_pullback_v1",
        "venue_key": "binance_spot_usdt",
    }
    p = tmp_path / "manifest_wrong.json"
    p.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Pre-drill manifest check FAILED"):
        _assert_manifest_clean(str(p))
