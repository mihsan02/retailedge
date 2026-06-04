"""
tests/test_champion_rollback.py

Required gates:
- test_rollback_recorded_in_deployment_audit
- test_manifest_reverts_to_rollback_model_id

Edge cases:
- test_promote_sets_rollback_slot_from_current_manifest
- test_rollback_fails_if_no_rollback_model_id_in_manifest
- test_challenger_live_false_after_promote
"""

import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from deployment.champion_controller import (
    OperatorApproval,
    build_candidate_from_dict,
    promote_champion,
)
from drill_champion_rollback import (
    CANDIDATE_NEW,
    CANDIDATE_ROLLBACK,
    DRILL_EVENT_TYPE,
    run_drill,
    scenario_a_promote,
    scenario_b_rollback,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

APPROVAL = OperatorApproval(
    operator_id="operator",
    approved_at=datetime.now(timezone.utc).isoformat(),
    promotion_source="manual",
)

INITIAL_MODEL_ID = "original_model_20260601_001"


@pytest.fixture()
def tmp_manifest(tmp_path):
    """Write a valid initial manifest with known model_id."""
    manifest = {
        "strategy_id": "trend_pullback_v1",
        "model_id": INITIAL_MODEL_ID,
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
    """SQLite DB with deployment_audit table."""
    db_path = str(tmp_path / "test.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE deployment_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                scenario TEXT,
                result TEXT,
                created_ts TEXT NOT NULL
            )
        """)
        conn.commit()
    return db_path


def _read_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _audit_rows(db_path: str, event_type: str) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM deployment_audit WHERE event_type = ?", (event_type,)
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


def test_rollback_recorded_in_deployment_audit(tmp_manifest, tmp_db):
    """Both scenarios must write a row to deployment_audit."""
    run_drill(manifest_path=tmp_manifest, db_path=tmp_db)

    rows = _audit_rows(tmp_db, DRILL_EVENT_TYPE)
    scenarios = {r["scenario"] for r in rows}

    assert len(rows) == 2, f"Expected 2 audit rows, got {len(rows)}"
    assert "A_promote" in scenarios, "A_promote not in deployment_audit"
    assert "B_rollback" in scenarios, "B_rollback not in deployment_audit"
    assert all(r["result"] == "PASS" for r in rows), (
        f"Not all rows PASS: {[r['result'] for r in rows]}"
    )


def test_manifest_reverts_to_rollback_model_id(tmp_manifest, tmp_db):
    """After full drill, manifest must show original model_id as active."""
    run_drill(manifest_path=tmp_manifest, db_path=tmp_db)

    final = _read_manifest(tmp_manifest)
    assert final["model_id"] == INITIAL_MODEL_ID, (
        f"Expected model_id '{INITIAL_MODEL_ID}' after rollback, "
        f"got '{final['model_id']}'"
    )


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_promote_sets_rollback_slot_from_current_manifest(tmp_manifest, tmp_db):
    """After scenario A, rollback_model_id must be the pre-promotion model_id."""
    scenario_a_promote(tmp_manifest, tmp_db)

    post = _read_manifest(tmp_manifest)
    assert post["rollback_model_id"] == INITIAL_MODEL_ID, (
        f"rollback_model_id should be '{INITIAL_MODEL_ID}', "
        f"got '{post.get('rollback_model_id')}'"
    )
    assert post["model_id"] == CANDIDATE_NEW["model_id"]


def test_rollback_fails_if_no_rollback_model_id_in_manifest(tmp_path, tmp_db):
    """scenario_b_rollback raises RuntimeError if rollback_model_id is absent."""
    manifest = {
        "model_id": "some_model",
        "model_type": "bucket_baseline",
        "model_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "feature_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "cost_model_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "regime_policy_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "stage_gate_passed": True,
        "challenger_live": False,
        "rollback_model_id": None,  # no rollback slot
    }
    p = tmp_path / "manifest_no_rollback.json"
    p.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="No rollback_model_id"):
        scenario_b_rollback(str(p), tmp_db)


def test_challenger_live_false_after_promote(tmp_manifest, tmp_db):
    """promote_champion must always write challenger_live=False."""
    scenario_a_promote(tmp_manifest, tmp_db)
    post = _read_manifest(tmp_manifest)
    assert post.get("challenger_live") is False, (
        f"challenger_live should be False, got {post.get('challenger_live')}"
    )


def test_full_drill_leaves_two_audit_rows_with_correct_event_type(tmp_manifest, tmp_db):
    """Audit rows must carry the canonical DRILL_EVENT_TYPE string."""
    run_drill(manifest_path=tmp_manifest, db_path=tmp_db)
    rows = _audit_rows(tmp_db, DRILL_EVENT_TYPE)
    assert len(rows) == 2
    for r in rows:
        assert r["event_type"] == DRILL_EVENT_TYPE
        assert r["result"] == "PASS"
        assert r["created_ts"]  # non-empty timestamp