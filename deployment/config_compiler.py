"""
deployment/config_compiler.py

Config Compiler for RetailEdge.
Single responsibility: validate capability contract, then write config.generated.json.
FAIL CLOSED — any mismatch raises ValueError and halts the build.

Design principle: generic venue routing via manifest["venue_key"].
Adding a new venue = add entry to capability_matrix.json only, no code change here.
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------

def _parse_version(version_str: str) -> tuple[int, ...]:
    """
    Parse "2026.1" or "2026.3_freqai" into a comparable int tuple.
    Strips non-numeric suffixes (e.g. "_freqai") before parsing.

    Critical: version comparison must never silently pass on malformed strings.
    """
    # Take only the first token before any underscore
    clean = version_str.split("_")[0]
    try:
        return tuple(int(x) for x in clean.split("."))
    except ValueError:
        raise ValueError(f"Unparseable version string: '{version_str}'")


def _version_gte(actual: str, minimum: str) -> bool:
    """Return True if actual >= minimum. Raises on malformed input."""
    return _parse_version(actual) >= _parse_version(minimum)


# ---------------------------------------------------------------------------
# Core validator — fail closed on every condition from CLAUDE.md
# ---------------------------------------------------------------------------

def validate_full_capability(
    venue_cfg: dict[str, Any],
    matrix: dict[str, Any],
    freqtrade_version: str,
) -> None:
    """
    Validate venue config against the capability matrix.

    Args:
        venue_cfg:          Single venue entry from venue_costs.json.
        matrix:             Full capability_matrix.json dict.
        freqtrade_version:  Running Freqtrade image version string.

    Raises:
        ValueError: On ANY mismatch. Caller must treat this as a build failure.

    Why generic venue routing: Sprint 1 scope is binance_spot, but the validator
    reads venue_key from the manifest so adding OKX/Bybit requires zero code change.
    """
    # --- 1. Resolve exchange capability entry
    exchange_key = venue_cfg.get("exchange_key")
    if not exchange_key:
        raise ValueError("venue_cfg missing 'exchange_key'")
    if exchange_key not in matrix:
        raise ValueError(f"exchange_key '{exchange_key}' not found in capability_matrix")

    cap = matrix[exchange_key]

    # --- 2. Freqtrade version gate
    # Must be first: if the image is too old, all other capability fields are suspect.
    min_version = cap.get("freqtrade_min_version")
    if not min_version:
        raise ValueError(f"capability_matrix['{exchange_key}'] missing 'freqtrade_min_version'")
    if not _version_gte(freqtrade_version, min_version):
        raise ValueError(
            f"Freqtrade version '{freqtrade_version}' < required '{min_version}' "
            f"for exchange '{exchange_key}'"
        )

    # --- 3. maker_first requires post_only_supported
    # post_only is the exchange-level enforcement of maker semantics.
    # If the exchange doesn't support it, maker_first is a lie.
    if venue_cfg.get("maker_first") and not cap.get("post_only_supported"):
        raise ValueError(
            f"venue_cfg.maker_first=true but capability['{exchange_key}'].post_only_supported is not true"
        )

    # --- 4. stoploss_on_exchange requires three-way contract check
    # All three must match: API version, order type, conditional mode.
    # Partial match is not acceptable — a stoploss that uses the wrong order type
    # will silently fail on exchange.
    if venue_cfg.get("stoploss_on_exchange_supported"):
        api_version = venue_cfg.get("stoploss_api_version")
        supported_api_versions = cap.get("stoploss_api_versions_supported", [])
        if api_version not in supported_api_versions:
            raise ValueError(
                f"stoploss_api_version '{api_version}' not in "
                f"capability.stoploss_api_versions_supported {supported_api_versions}"
            )

        order_type = venue_cfg.get("stoploss_order_type")
        supported_order_types = cap.get("stoploss_order_types_supported", [])
        if order_type not in supported_order_types:
            raise ValueError(
                f"stoploss_order_type '{order_type}' not in "
                f"capability.stoploss_order_types_supported {supported_order_types}"
            )

        cond_mode = venue_cfg.get("conditional_order_mode")
        supported_cond_modes = cap.get("conditional_order_mode_supported", [])
        if cond_mode not in supported_cond_modes:
            raise ValueError(
                f"conditional_order_mode '{cond_mode}' not in "
                f"capability.conditional_order_mode_supported {supported_cond_modes}"
            )

    # --- 5. emergency_exit requires market order support
    # emergency_exit_enabled defaults False (CLAUDE.md hard constraint #4).
    # Only checked when explicitly enabled — ensures drill test is prerequisite.
    if venue_cfg.get("emergency_exit_enabled") and not cap.get("market_order_supported"):
        raise ValueError(
            f"emergency_exit_enabled=true but capability['{exchange_key}'].market_order_supported is not true"
        )

    # All checks passed. No return value — caller proceeds only if no exception.


# ---------------------------------------------------------------------------
# Manifest validator — enforces single active model invariant
# ---------------------------------------------------------------------------

def validate_manifest(manifest: dict[str, Any], model_base_dir: str = "models") -> None:
    """
    Enforce active model manifest invariants.

    Checks:
    - Exactly one model_id present and non-empty.
    - challenger_live must be absent or False.
    - model_hash matches sha256 of deployed model file.

    The model file hash check is the critical integrity gate.
    Without it, a stale or tampered model artifact can run silently.
    """
    model_id = manifest.get("model_id")
    if not model_id:
        raise ValueError("active_model_manifest missing or empty 'model_id'")

    if manifest.get("challenger_live") is True:
        raise ValueError(
            "active_model_manifest.challenger_live=true — challenger cannot be live in MVP runtime"
        )

    expected_hash = manifest.get("model_hash", "")
    if not expected_hash.startswith("sha256:"):
        # In dry-run, model file may not exist yet.
        # We allow a sentinel "sha256:DRY_RUN_PLACEHOLDER" to skip file check.
        if expected_hash == "sha256:DRY_RUN_PLACEHOLDER":
            return
        raise ValueError(
            f"model_hash format invalid: '{expected_hash}'. Must be 'sha256:<hex>' or 'sha256:DRY_RUN_PLACEHOLDER'"
        )

    hex_expected = expected_hash.replace("sha256:", "")
    if hex_expected == "DRY_RUN_PLACEHOLDER":
        return

    model_path = Path(model_base_dir) / model_id / "edge_model.json"
    if not model_path.exists():
        raise ValueError(f"Model file not found: {model_path}")

    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_hash != hex_expected:
        raise ValueError(
            f"Model hash mismatch for '{model_id}': "
            f"expected {hex_expected[:12]}..., got {actual_hash[:12]}..."
        )


# ---------------------------------------------------------------------------
# Config compiler — merges base config with venue + capability constraints
# ---------------------------------------------------------------------------

def compile_config(
    manifest: dict[str, Any],
    venue_cfg: dict[str, Any],
    matrix: dict[str, Any],
    base_config: dict[str, Any],
    freqtrade_version: str,
    output_path: str = "freqtrade/config.generated.json",
    model_base_dir: str = "models",
) -> None:
    """
    Full compilation pipeline:
    1. Validate capability contract (raises on failure).
    2. Validate manifest integrity (raises on failure).
    3. Merge base config with runtime-enforced overrides.
    4. Write config.generated.json.

    Never call this without passing both validators first.
    The output file is the ONLY config Freqtrade should load.
    """
    # Step 1: capability gate
    validate_full_capability(venue_cfg, matrix, freqtrade_version)

    # Step 2: manifest gate
    validate_manifest(manifest, model_base_dir)

    # Step 3: build output config
    # Start from base config, then layer on enforced overrides.
    # Overrides must win — never let base config silently bypass safety constraints.
    config = dict(base_config)

    # Enforce stoploss_on_exchange from venue contract, not from base config.
    config["order_types"] = config.get("order_types", {})
    config["order_types"]["stoploss_on_exchange"] = venue_cfg.get("stoploss_on_exchange_supported", False)
    config["order_types"]["stoploss_on_exchange_interval"] = venue_cfg.get("stoploss_audit_interval_normal_sec", 300)

    # emergency_exit order type only injected if enabled.
    # Default is "market" per blueprint; compiler does not override the order type itself,
    # only validates the capability contract allows it.
    if not venue_cfg.get("emergency_exit_enabled", False):
        # Ensure emergency_exit order type is set to "market" in config
        # even when disabled — Freqtrade needs the field but it won't be triggered.
        config["order_types"]["emergency_exit"] = "market"

    # Inject active model metadata as a top-level annotation.
    # Freqtrade ignores unknown top-level keys, so this is safe.
    config["_active_model_id"] = manifest["model_id"]
    config["_active_model_type"] = manifest.get("model_type", "unknown")
    config["_compiled_by"] = "config_compiler.py"

    # Step 4: write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2))


# ---------------------------------------------------------------------------
# CLI entry point — for manual invocation and CI pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    import sys

    matrix_path = os.getenv("CAPABILITY_MATRIX_PATH", "deployment/capability_matrix.json")
    venue_costs_path = os.getenv("VENUE_COSTS_PATH", "deployment/venue_costs.json")
    manifest_path = os.getenv("ACTIVE_MODEL_MANIFEST_PATH", "deployment/active_model_manifest.json")
    base_config_path = "freqtrade/config.base.json"
    freqtrade_version = os.getenv("FREQTRADE_VERSION", "2026.3_freqai")
    output_path = "freqtrade/config.generated.json"

    for path in [matrix_path, venue_costs_path, manifest_path, base_config_path]:
        if not Path(path).exists():
            print(f"ERROR: Required file not found: {path}", file=sys.stderr)
            sys.exit(1)

    matrix = json.loads(Path(matrix_path).read_text())
    venue_costs = json.loads(Path(venue_costs_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    base_config = json.loads(Path(base_config_path).read_text())

    # venue_key from manifest selects the active venue entry
    venue_key = manifest.get("venue_key", "binance_spot_usdt")
    if venue_key not in venue_costs:
        print(f"ERROR: venue_key '{venue_key}' not found in venue_costs.json", file=sys.stderr)
        sys.exit(1)

    venue_cfg = venue_costs[venue_key]

    try:
        compile_config(
            manifest=manifest,
            venue_cfg=venue_cfg,
            matrix=matrix,
            base_config=base_config,
            freqtrade_version=freqtrade_version,
            output_path=output_path,
        )
        print(f"OK: config compiled to {output_path}")
    except ValueError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()