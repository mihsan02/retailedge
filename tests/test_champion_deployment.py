"""
tests/test_champion_deployment.py

Test gate: Dry-run.

Required tests (Sprint 1 S1-3 gate):
- test_champion_only_one_active_model
- test_challenger_cannot_auto_promote
- test_scheduled_model_promotion_manifest_hash

Each test is fully isolated: uses tmp_path (pytest fixture) for file I/O,
never touches deployment/active_model_manifest.json in the repo.
"""

import json
import hashlib
import pytest
from pathlib import Path

from deployment.champion_controller import (
    load_active_model,
    promote_champion,
    build_candidate_from_dict,
    CandidateRecord,
    OperatorApproval,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _valid_candidate() -> dict:
    """Baseline candidate that passes all gates."""
    return {
        "model_id": "bucket_baseline_20260604_001",
        "model_type": "bucket_baseline",
        "strategy_id": "retail_edge_v1",
        "model_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "feature_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "cost_model_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "regime_policy_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "stage_gate_passed": True,
        "rollback_model_id": None,
        "venue_key": "binance_spot_usdt",
    }


def _valid_approval() -> OperatorApproval:
    return OperatorApproval(
        operator_id="operator",
        approved_at="2026-06-04T07:00:00Z",
        promotion_source="scheduled_review",
    )


def _write_manifest(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# test_champion_only_one_active_model
# ---------------------------------------------------------------------------

def test_champion_only_one_active_model(tmp_path):
    """
    load_active_model must succeed when manifest has exactly one model_id.
    It must fail (RuntimeError) when model_id is missing or empty.

    This test covers the single-model invariant from CLAUDE.md hard constraint #1:
    'Runtime loads exactly ONE active champion model at all times.'

    Two sub-cases:
    A) Valid manifest with one model_id -> loads without error.
    B) Manifest with empty model_id -> RuntimeError immediately.
    """
    manifest_path = tmp_path / "active_model_manifest.json"

    # Sub-case A: valid — exactly one model_id
    valid_manifest = {
        "model_id": "bucket_baseline_20260604_001",
        "model_type": "bucket_baseline",
        "model_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "challenger_live": False,
    }
    _write_manifest(manifest_path, valid_manifest)
    result = load_active_model(str(manifest_path))
    assert result["model_id"] == "bucket_baseline_20260604_001"

    # Sub-case B: model_id empty string -> must fail
    invalid_manifest = dict(valid_manifest)
    invalid_manifest["model_id"] = ""
    _write_manifest(manifest_path, invalid_manifest)
    with pytest.raises(RuntimeError, match="model_id"):
        load_active_model(str(manifest_path))

    # Sub-case C: model_id absent -> must fail
    no_id_manifest = {
        "model_type": "bucket_baseline",
        "model_hash": "sha256:DRY_RUN_PLACEHOLDER",
    }
    _write_manifest(manifest_path, no_id_manifest)
    with pytest.raises(RuntimeError, match="model_id"):
        load_active_model(str(manifest_path))


# ---------------------------------------------------------------------------
# test_challenger_cannot_auto_promote
# ---------------------------------------------------------------------------

def test_challenger_cannot_auto_promote(tmp_path):
    """
    Two distinct invariants tested:

    1. load_active_model must raise RuntimeError if challenger_live=True in manifest.
       (CLAUDE.md hard constraint #2: challenger cannot touch live config.)

    2. promote_champion must raise ValueError if called with a freqai_challenger
       candidate that has no operator approval record (stage_gate_passed=False).

    Why two invariants in one test: both guard the same threat — a challenger
    model reaching the live execution path without human review.
    """
    manifest_path = tmp_path / "active_model_manifest.json"

    # Invariant 1: challenger_live=True in existing manifest -> load must fail
    challenger_manifest = {
        "model_id": "freqai_challenger_20260604_001",
        "model_type": "freqai_challenger",
        "model_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "challenger_live": True,  # the forbidden flag
    }
    _write_manifest(manifest_path, challenger_manifest)
    with pytest.raises(RuntimeError, match="challenger_live"):
        load_active_model(str(manifest_path))

    # Invariant 2: promote_champion blocks if stage_gate_passed=False
    # This simulates a challenger that hasn't cleared OOS evidence gate.
    candidate_dict = _valid_candidate()
    candidate_dict["model_type"] = "freqai_challenger"
    candidate_dict["stage_gate_passed"] = False  # gate not passed

    candidate = build_candidate_from_dict(candidate_dict)
    approval = _valid_approval()

    with pytest.raises(ValueError, match="stage gate"):
        promote_champion(
            candidate=candidate,
            approval=approval,
            manifest_path=str(manifest_path),
        )


# ---------------------------------------------------------------------------
# test_scheduled_model_promotion_manifest_hash
# ---------------------------------------------------------------------------

def test_scheduled_model_promotion_manifest_hash(tmp_path):
    """
    promote_champion must write a manifest where:
    - model_id matches the candidate.
    - model_hash matches the candidate's declared hash.
    - challenger_live is explicitly False.
    - approved_by and promotion_source are recorded from the approval object.
    - rollback_model_id captures the previous champion's model_id.

    This guards against silent manifest corruption during promotion:
    if any field is wrong at write time, the next load_active_model call
    will either load the wrong model or fail the hash check.

    Decision Bus post is verified via injectable spy (no real DB needed).
    """
    manifest_path = tmp_path / "active_model_manifest.json"

    # Write a "previous champion" manifest to test rollback slot capture
    previous_manifest = {
        "model_id": "bucket_baseline_20260521_002",
        "model_type": "bucket_baseline",
        "model_hash": "sha256:DRY_RUN_PLACEHOLDER",
        "challenger_live": False,
    }
    _write_manifest(manifest_path, previous_manifest)

    # Build new candidate for promotion
    candidate = build_candidate_from_dict(_valid_candidate())
    approval = _valid_approval()

    # Inject Decision Bus spy — captures what would be posted
    bus_calls = []
    def mock_bus_post(**kwargs):
        bus_calls.append(kwargs)

    result = promote_champion(
        candidate=candidate,
        approval=approval,
        manifest_path=str(manifest_path),
        decision_bus_post_fn=mock_bus_post,
    )

    # --- Verify manifest fields
    assert result["model_id"] == candidate.model_id
    assert result["model_hash"] == candidate.model_hash
    assert result["challenger_live"] is False
    assert result["approved_by"] == approval.operator_id
    assert result["promotion_source"] == approval.promotion_source

    # --- Verify rollback slot captured previous champion
    assert result["rollback_model_id"] == "bucket_baseline_20260521_002"

    # --- Verify manifest was actually written to disk (not just returned)
    on_disk = json.loads(manifest_path.read_text())
    assert on_disk["model_id"] == candidate.model_id
    assert on_disk["challenger_live"] is False

    # --- Verify Decision Bus was called with correct action type
    assert len(bus_calls) == 1
    assert bus_calls[0]["action_type"] == "SCHEDULED_MODEL_PROMOTION"
    assert bus_calls[0]["model_id"] == candidate.model_id

    # --- Verify load_active_model accepts the promoted manifest
    loaded = load_active_model(str(manifest_path))
    assert loaded["model_id"] == candidate.model_id


# ---------------------------------------------------------------------------
# Edge cases — regression guards
# ---------------------------------------------------------------------------

def test_manifest_not_found_raises(tmp_path):
    """load_active_model on missing file must raise RuntimeError, not FileNotFoundError."""
    with pytest.raises(RuntimeError, match="not found"):
        load_active_model(str(tmp_path / "nonexistent.json"))


def test_promote_requires_operator_id(tmp_path):
    """OperatorApproval with empty operator_id must be rejected before manifest write."""
    candidate = build_candidate_from_dict(_valid_candidate())
    bad_approval = OperatorApproval(
        operator_id="",  # empty
        approved_at="2026-06-04T07:00:00Z",
        promotion_source="scheduled_review",
    )
    with pytest.raises(ValueError, match="operator_id"):
        promote_champion(
            candidate=candidate,
            approval=bad_approval,
            manifest_path=str(tmp_path / "manifest.json"),
        )


def test_promote_invalid_promotion_source_raises(tmp_path):
    """promotion_source outside allowed values must be rejected."""
    candidate = build_candidate_from_dict(_valid_candidate())
    bad_approval = OperatorApproval(
        operator_id="operator",
        approved_at="2026-06-04T07:00:00Z",
        promotion_source="auto_trigger",  # not allowed
    )
    with pytest.raises(ValueError, match="promotion_source"):
        promote_champion(
            candidate=candidate,
            approval=bad_approval,
            manifest_path=str(tmp_path / "manifest.json"),
        )


def test_promote_writes_challenger_live_false(tmp_path):
    """
    challenger_live must be False in written manifest regardless of input.
    Ensures the invariant is enforced at write time, not just at load time.
    """
    candidate = build_candidate_from_dict(_valid_candidate())
    approval = _valid_approval()
    manifest_path = tmp_path / "manifest.json"

    result = promote_champion(
        candidate=candidate,
        approval=approval,
        manifest_path=str(manifest_path),
    )
    assert result["challenger_live"] is False
    on_disk = json.loads(manifest_path.read_text())
    assert on_disk["challenger_live"] is False


def test_freqai_challenger_can_promote_with_approval(tmp_path):
    """
    FreqAI challenger CAN be promoted — but only with valid operator approval
    and stage_gate_passed=True. This verifies the gate is about process, not type.
    """
    candidate_dict = _valid_candidate()
    candidate_dict["model_type"] = "freqai_challenger"
    candidate_dict["model_id"] = "freqai_challenger_20260604_001"
    candidate_dict["stage_gate_passed"] = True  # gate passed

    candidate = build_candidate_from_dict(candidate_dict)
    approval = OperatorApproval(
        operator_id="operator",
        approved_at="2026-06-04T08:00:00Z",
        promotion_source="manual",
    )

    result = promote_champion(
        candidate=candidate,
        approval=approval,
        manifest_path=str(tmp_path / "manifest.json"),
    )
    assert result["model_id"] == "freqai_challenger_20260604_001"
    assert result["challenger_live"] is False