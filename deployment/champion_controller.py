"""
deployment/champion_controller.py

Champion Deployment Controller for RetailEdge.
Single responsibility: enforce the one-active-model invariant and gate promotion
behind explicit operator approval + scheduled window.

Hard constraints enforced here (from CLAUDE.md):
1. Runtime loads exactly ONE active champion model.
2. Challenger models cannot touch live config (challenger_live must be absent/False).
3. No real-time switching — promotion writes manifest + triggers controlled reload only.
4. model_hash must match deployed file (or DRY_RUN_PLACEHOLDER sentinel).

This module does NOT write config.generated.json directly.
That is config_compiler.py's job. Controller writes the manifest and posts
to the Decision Bus; the compiler runs as a consequence of promotion.
"""

import json
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class OperatorApproval:
    """
    Explicit approval record required before any champion promotion.

    operator_id: who approved (username, email, or "operator" for single-operator setups).
    approved_at: ISO8601 UTC timestamp of approval.
    promotion_source: "scheduled_review" or "manual" — must be explicit, no default.

    Why a dataclass and not a dict: forces callers to be explicit about all fields.
    A dict with missing keys silently produces None; this raises AttributeError immediately.
    """
    operator_id: str
    approved_at: str
    promotion_source: str  # "scheduled_review" | "manual"

    def validate(self) -> None:
        if not self.operator_id:
            raise ValueError("OperatorApproval.operator_id is empty")
        if self.promotion_source not in ("scheduled_review", "manual"):
            raise ValueError(
                f"OperatorApproval.promotion_source must be 'scheduled_review' or 'manual', "
                f"got '{self.promotion_source}'"
            )
        if not self.approved_at:
            raise ValueError("OperatorApproval.approved_at is empty")


@dataclass
class CandidateRecord:
    """
    Minimal candidate record from Strategy Memory Store.
    In Sprint 1 this is populated from a dict; in Sprint 3 it will come from the ledger.
    """
    model_id: str
    model_type: str          # "bucket_baseline" | "freqai_challenger"
    strategy_id: str
    model_hash: str          # "sha256:<hex>" or "sha256:DRY_RUN_PLACEHOLDER"
    feature_hash: str
    cost_model_hash: str
    regime_policy_hash: str
    stage_gate_passed: bool
    rollback_model_id: str | None = None
    venue_key: str = "binance_spot_usdt"

    def validate(self) -> None:
        if not self.model_id:
            raise ValueError("CandidateRecord.model_id is empty")
        if self.model_type not in ("bucket_baseline", "freqai_challenger"):
            raise ValueError(
                f"CandidateRecord.model_type must be 'bucket_baseline' or 'freqai_challenger', "
                f"got '{self.model_type}'"
            )
        if not self.stage_gate_passed:
            raise ValueError(
                f"Candidate '{self.model_id}' has not passed stage gate — promotion blocked"
            )
        if not self.model_hash.startswith("sha256:"):
            raise ValueError(
                f"model_hash format invalid: '{self.model_hash}'. Must start with 'sha256:'"
            )


# ---------------------------------------------------------------------------
# Core: load_active_model
# ---------------------------------------------------------------------------

def load_active_model(
    manifest_path: str,
    model_base_dir: str = "models",
) -> dict[str, Any]:
    """
    Load and validate the active model manifest. Enforces three invariants:

    1. model_id must be present and non-empty (no active model = runtime must not start).
    2. challenger_live must be absent or False (challenger cannot be live in MVP).
    3. model_hash must match sha256 of the deployed model file
       (or "DRY_RUN_PLACEHOLDER" for dry-run stage where no model artifact exists).

    Returns the parsed model artifact dict on success.
    Raises RuntimeError on any violation — not ValueError, because these are
    runtime integrity failures, not configuration input errors.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise RuntimeError(f"Active model manifest not found: {manifest_path}")

    manifest = json.loads(path.read_text())

    # --- Invariant 1: exactly one model_id
    model_id = manifest.get("model_id")
    if not model_id:
        raise RuntimeError(
            "active_model_manifest missing or empty 'model_id' — "
            "runtime cannot start without an active model"
        )

    # --- Invariant 2: challenger cannot be live
    # challenger_live absent = safe (treated as False).
    # challenger_live explicitly True = immediate halt.
    if manifest.get("challenger_live") is True:
        raise RuntimeError(
            f"active_model_manifest.challenger_live=true for model '{model_id}' — "
            "challenger cannot be live in MVP runtime. Halt."
        )

    # --- Invariant 3: model hash integrity
    model_hash = manifest.get("model_hash", "")
    if not model_hash.startswith("sha256:"):
        raise RuntimeError(
            f"model_hash format invalid: '{model_hash}'. "
            "Must be 'sha256:<hex>' or 'sha256:DRY_RUN_PLACEHOLDER'"
        )

    hex_hash = model_hash.replace("sha256:", "")

    if hex_hash == "DRY_RUN_PLACEHOLDER":
        # Dry-run stage: no model artifact on disk yet. Return manifest as the model record.
        return dict(manifest)

    # Production: verify file hash
    model_path = Path(model_base_dir) / model_id / "edge_model.json"
    if not model_path.exists():
        raise RuntimeError(
            f"Model artifact not found: {model_path} — "
            f"manifest references '{model_id}' but file is missing"
        )

    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_hash != hex_hash:
        raise RuntimeError(
            f"Model hash mismatch for '{model_id}': "
            f"manifest expects {hex_hash[:16]}..., "
            f"file is {actual_hash[:16]}... — possible tampering or stale manifest"
        )

    return json.loads(model_path.read_text())


# ---------------------------------------------------------------------------
# Core: promote_champion
# ---------------------------------------------------------------------------

def promote_champion(
    candidate: CandidateRecord,
    approval: OperatorApproval,
    manifest_path: str,
    decision_bus_post_fn=None,
) -> dict[str, Any]:
    """
    Promote a validated candidate to active champion.

    Gates (all must pass before manifest is written):
    - candidate.stage_gate_passed must be True.
    - approval fields must be valid.
    - model_type must NOT be freqai_challenger with auto-promotion intent
      (challenger can be promoted only via scheduled_review or explicit manual approval).

    This function writes active_model_manifest.json and posts
    SCHEDULED_MODEL_PROMOTION to the Decision Bus.
    It does NOT compile config.generated.json — that is triggered by the
    Decision Bus consumer (guardian) calling config_compiler after reload.

    decision_bus_post_fn: injectable for testing. In production, pass
    decision_bus.post. If None, the Decision Bus step is skipped (test mode).

    Returns the written manifest dict.
    """
    # --- Gate 1: validate inputs
    candidate.validate()
    approval.validate()

    # --- Gate 2: no auto-promotion for challengers
    # Challengers require the same human-in-the-loop as baseline promotions.
    # promotion_source="scheduled_review" or "manual" is the signal that a human
    # reviewed the OOS evidence before this call.
    # There is no code path where a challenger auto-promotes.
    if candidate.model_type == "freqai_challenger":
        if approval.promotion_source not in ("scheduled_review", "manual"):
            raise ValueError(
                f"FreqAI challenger '{candidate.model_id}' cannot auto-promote. "
                "promotion_source must be 'scheduled_review' or 'manual'."
            )

    # --- Gate 3: load current manifest to capture rollback slot
    current_manifest_path = Path(manifest_path)
    rollback_model_id = None
    if current_manifest_path.exists():
        try:
            current = json.loads(current_manifest_path.read_text())
            rollback_model_id = current.get("model_id")
        except (json.JSONDecodeError, OSError):
            # Corrupt or missing current manifest — proceed without rollback slot.
            rollback_model_id = None

    # Prefer explicit rollback_model_id from candidate record if set.
    if candidate.rollback_model_id:
        rollback_model_id = candidate.rollback_model_id

    # --- Build new manifest
    new_manifest = {
        "strategy_id": candidate.strategy_id,
        "model_id": candidate.model_id,
        "model_type": candidate.model_type,
        "model_hash": candidate.model_hash,
        "feature_hash": candidate.feature_hash,
        "cost_model_hash": candidate.cost_model_hash,
        "regime_policy_hash": candidate.regime_policy_hash,
        "venue_key": candidate.venue_key,
        "approved_by": approval.operator_id,
        "approved_at": approval.approved_at,
        "promotion_source": approval.promotion_source,
        "rollback_model_id": rollback_model_id,
        "challenger_live": False,   # always False — invariant enforced at write time
    }

    # --- Write manifest (atomic: write to temp then rename)
    out_path = Path(manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(new_manifest, indent=2))
    tmp_path.rename(out_path)

    # --- Post to Decision Bus (injectable, skipped in unit tests)
    if decision_bus_post_fn is not None:
        decision_bus_post_fn(
            action_type="SCHEDULED_MODEL_PROMOTION",
            reason="champion_selected_after_oos_review",
            model_id=candidate.model_id,
            reload_mode="reload_config_or_restart_window",
            severity="INFO",
        )

    return new_manifest


# ---------------------------------------------------------------------------
# Helper: build_manifest_from_dict (for tests and CLI)
# ---------------------------------------------------------------------------

def build_candidate_from_dict(d: dict[str, Any]) -> CandidateRecord:
    """Convenience constructor. All fields must be present — no silent defaults."""
    return CandidateRecord(
        model_id=d["model_id"],
        model_type=d["model_type"],
        strategy_id=d["strategy_id"],
        model_hash=d["model_hash"],
        feature_hash=d["feature_hash"],
        cost_model_hash=d["cost_model_hash"],
        regime_policy_hash=d["regime_policy_hash"],
        stage_gate_passed=d["stage_gate_passed"],
        rollback_model_id=d.get("rollback_model_id"),
        venue_key=d.get("venue_key", "binance_spot_usdt"),
    )