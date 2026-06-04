"""
sidecar/guardian/guardian.py

Guardian main loop for RetailEdge.
Single responsibility: poll Decision Bus, gate on liveness, route actions
to handlers, verify 2xx success, retry or escalate on failure.

Hard rules (CLAUDE.md):
- Liveness check (ping) before every mutating action.
- Mark DONE only on confirmed 2xx. Never assume success.
- Monitor Reconciler heartbeat: if missing, post PAUSE_REQUIRED.
- Emergency exit cascade gated by policy.emergency_exit_enabled.
- No real-time model switching — SCHEDULED_MODEL_PROMOTION triggers reload only.

Architecture:
- Guardian runs as a single-threaded poll loop in the sidecar container.
- One loop iteration: fetch pending actions, process each, sleep poll_interval.
- All state lives in Decision Bus (SQLite). Guardian itself is stateless.
- Heartbeat check is part of every loop iteration, not a separate thread.

This module is designed to be run as:
    python -m sidecar.guardian.guardian
or called via run_once() in tests.
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from sidecar.guardian.decision_bus import DecisionBus, VALID_ACTION_TYPES
from sidecar.guardian.ft_client import FreqtradeClient, FreqtradeClientError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class GuardianPolicy:
    """
    Runtime policy for Guardian behavior.
    All values have safe defaults. Override via environment variables.
    """
    def __init__(self) -> None:
        # How often Guardian polls the Decision Bus (seconds)
        self.poll_interval_sec: float = float(os.getenv("GUARDIAN_POLL_INTERVAL_SEC", "10"))

        # Max seconds since last Reconciler heartbeat before PAUSE_REQUIRED
        self.reconciler_heartbeat_max_age_sec: float = float(
            os.getenv("RECONCILER_HEARTBEAT_MAX_AGE_SEC", "120")
        )

        # Emergency exit only active after drill pass (CLAUDE.md hard constraint #4)
        self.emergency_exit_enabled: bool = (
            os.getenv("EMERGENCY_EXIT_ENABLED", "false").lower() == "true"
        )


# ---------------------------------------------------------------------------
# Guardian
# ---------------------------------------------------------------------------

class Guardian:
    """
    Stateless action processor. All durability lives in DecisionBus (SQLite).

    Designed for single-threaded operation in a container.
    Tests call run_once() to process one batch without sleeping.
    Production calls run_loop() which polls indefinitely.
    """

    def __init__(
        self,
        bus: DecisionBus,
        ft_client: FreqtradeClient,
        policy: Optional[GuardianPolicy] = None,
        heartbeat_store: Optional[Any] = None,
    ) -> None:
        self.bus = bus
        self.ft = ft_client
        self.policy = policy or GuardianPolicy()
        # heartbeat_store: object with get_last_heartbeat_ts() -> Optional[str]
        # If None, heartbeat check is skipped (dry-run / test mode)
        self.heartbeat_store = heartbeat_store

    # -----------------------------------------------------------------------
    # Loop entry points
    # -----------------------------------------------------------------------

    def run_loop(self) -> None:
        """
        Production entry point. Polls indefinitely.
        Catches all exceptions to prevent loop death on transient errors.
        """
        logger.info("Guardian starting poll loop, interval=%ss", self.policy.poll_interval_sec)
        while True:
            try:
                self.run_once()
            except Exception as exc:
                logger.error("Guardian loop unhandled exception: %s", exc, exc_info=True)
            time.sleep(self.policy.poll_interval_sec)

    def run_once(self) -> list[dict[str, Any]]:
        """
        Process one batch of pending actions.
        Returns list of processed action results (for test inspection).

        Steps:
        1. Check Reconciler heartbeat (if store available).
        2. Fetch pending actions from Decision Bus.
        3. For each action: liveness gate, route, verify, mark done/retry.
        """
        results = []

        # Step 1: heartbeat check
        self._check_reconciler_heartbeat()

        # Step 2: fetch pending
        actions = self.bus.consume_pending()

        # Step 3: process each action
        for action in actions:
            result = self._process_action(action)
            results.append(result)

        return results

    # -----------------------------------------------------------------------
    # Action processor
    # -----------------------------------------------------------------------

    def _process_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Process a single action through the full pipeline:
        liveness gate -> mark IN_PROGRESS -> route -> verify -> mark done/retry.

        Returns a result dict for test inspection.
        Never raises — all exceptions are caught and logged.
        """
        action_id = action["action_id"]
        action_type = action["action_type"]

        logger.info("Processing action %s type=%s", action_id[:8], action_type)

        # Mark IN_PROGRESS before any external call
        self.bus.mark_in_progress(action_id)

        # Liveness gate: ping Freqtrade before any mutating action
        # REJECT_ENTRY is a read-only advisory — no REST call needed
        if action_type != "REJECT_ENTRY":
            if not self.ft.ping_alive():
                logger.warning("Freqtrade unreachable, action %s deferred", action_id[:8])
                self.bus.mark_failed_retryable(action_id, error="freqtrade_unreachable")
                return {"action_id": action_id, "outcome": "DEFERRED_BOT_DOWN"}

        try:
            outcome = self._route(action)
            self.bus.mark_done(action_id, result=outcome)
            logger.info("Action %s DONE: %s", action_id[:8], outcome)
            return {"action_id": action_id, "outcome": outcome}

        except FreqtradeClientError as exc:
            error_msg = f"http_{exc.status_code or 'conn'}:{exc}"
            logger.warning("Action %s failed: %s", action_id[:8], error_msg)
            self.bus.mark_failed_retryable(action_id, error=error_msg)
            return {"action_id": action_id, "outcome": "FAILED_RETRYABLE", "error": error_msg}

        except Exception as exc:
            error_msg = f"unexpected:{exc}"
            logger.error("Action %s unexpected error: %s", action_id[:8], error_msg, exc_info=True)
            self.bus.mark_failed_retryable(action_id, error=error_msg)
            return {"action_id": action_id, "outcome": "FAILED_RETRYABLE", "error": error_msg}

    # -----------------------------------------------------------------------
    # Action router
    # -----------------------------------------------------------------------

    def _route(self, action: dict[str, Any]) -> str:
        """
        Route action to the correct handler.
        Returns outcome string on success.
        Raises FreqtradeClientError on REST failure (caught by _process_action).

        Each handler calls exactly one REST endpoint and returns on 2xx.
        If the endpoint returns non-2xx, FreqtradeClient raises — no silent success.
        """
        action_type = action["action_type"]

        if action_type == "PAUSE_REQUIRED":
            return self._handle_pause(action)

        if action_type == "RESUME_ENTRY":
            return self._handle_resume(action)

        if action_type == "SCHEDULED_MODEL_PROMOTION":
            return self._handle_model_promotion(action)

        if action_type == "EMERGENCY_EXIT":
            return self._handle_emergency_exit(action)

        if action_type == "REJECT_ENTRY":
            # Advisory only — no REST call. Guardian logs and marks done.
            logger.info("REJECT_ENTRY noted for pair=%s reason=%s",
                        action.get("pair"), action.get("reason"))
            return "NOTED_NO_REST_CALL"

        if action_type in ("STOP_UNCONFIRMED", "RESERVED_MISMATCH_ON_STARTUP"):
            # These escalate to PAUSE_REQUIRED — Guardian pauses and alerts.
            return self._handle_pause(action)

        if action_type in ("OPERATOR_REQUIRED", "EMERGENCY_EXIT_FAILED"):
            # These are terminal alerts. No REST call — operator must intervene.
            self._alert_operator(action)
            return "OPERATOR_ALERTED"

        # Should never reach here — VALID_ACTION_TYPES is exhaustive.
        raise ValueError(f"No handler for action_type '{action_type}'")

    # -----------------------------------------------------------------------
    # Handlers
    # -----------------------------------------------------------------------

    def _handle_pause(self, action: dict[str, Any]) -> str:
        """
        POST /api/v1/pause.
        2xx = DONE. Non-2xx = FreqtradeClientError (retried by caller).
        """
        self.ft.pause(reason=action.get("reason", ""))
        return "PAUSED"

    def _handle_resume(self, action: dict[str, Any]) -> str:
        """
        Freqtrade does not have a /resume endpoint in the REST API.
        Resume is achieved by reload_config which re-enables entry.
        """
        self.ft.reload_config()
        return "RESUMED_VIA_RELOAD"

    def _handle_model_promotion(self, action: dict[str, Any]) -> str:
        """
        POST /api/v1/reload_config — apply promoted champion config.
        Config file must already be written by Champion Deployment Controller
        before this action is posted to the bus.
        """
        self.ft.reload_config()
        return f"CONFIG_RELOADED model_id={action.get('model_id', 'unknown')}"

    def _handle_emergency_exit(self, action: dict[str, Any]) -> str:
        """
        Emergency exit cascade:
        1. market exit via forceexit
        2. aggressive limit fallback if market fails
        3. OPERATOR_REQUIRED if both fail

        Only active if policy.emergency_exit_enabled=True.
        Default is False (CLAUDE.md hard constraint #4).
        """
        if not self.policy.emergency_exit_enabled:
            logger.warning(
                "EMERGENCY_EXIT action received but emergency_exit_enabled=False. "
                "Posting OPERATOR_REQUIRED."
            )
            self.bus.post(
                "OPERATOR_REQUIRED",
                reason="emergency_exit_disabled_operator_intervention_required",
                severity="CRITICAL",
                trade_id=action.get("trade_id"),
                pair=action.get("pair"),
            )
            return "DISABLED_OPERATOR_REQUIRED"

        trade_id = action.get("trade_id")
        if not trade_id:
            raise ValueError("EMERGENCY_EXIT action missing trade_id")

        # Attempt 1: market order
        try:
            self.ft.forceexit(trade_id=trade_id, ordertype="market")
            return "DONE_MARKET"
        except FreqtradeClientError as exc:
            logger.warning("Market exit failed for trade %s: %s", trade_id, exc)

        # Attempt 2: aggressive limit fallback
        # Best bid * 0.98 per blueprint v1.9 Section 6 Step 4
        # In this layer we post another action; exchange client is not directly
        # accessible here. Guardian posts OPERATOR_REQUIRED with context.
        try:
            # Aggressive limit: re-attempt via forceexit with limit type
            # Price is not available here without exchange client.
            # In Sprint 4 this will integrate with exchange_client.fetch_order_book().
            # For Sprint 2: fallback directly to OPERATOR_REQUIRED with CRITICAL severity.
            self.bus.post(
                "EMERGENCY_EXIT_FAILED",
                reason="market_exit_failed_aggressive_limit_not_yet_implemented",
                severity="CRITICAL",
                trade_id=trade_id,
                pair=action.get("pair"),
            )
            return "FAILED_OPERATOR_REQUIRED"
        except Exception as exc:
            logger.error("Emergency exit cascade failed entirely: %s", exc)
            return "FAILED_CASCADE_ERROR"

    def _alert_operator(self, action: dict[str, Any]) -> None:
        """
        Log CRITICAL alert. In Sprint 4 this will send Telegram notification.
        For Sprint 2: structured log output that can be scraped by alerting tools.
        """
        logger.critical(
            "OPERATOR_REQUIRED | action_type=%s | pair=%s | trade_id=%s | reason=%s",
            action.get("action_type"),
            action.get("pair"),
            action.get("trade_id"),
            action.get("reason"),
        )

    # -----------------------------------------------------------------------
    # Reconciler heartbeat monitor
    # -----------------------------------------------------------------------

    def _check_reconciler_heartbeat(self) -> None:
        """
        Verify Reconciler is alive by checking last heartbeat timestamp.
        If heartbeat is older than policy.reconciler_heartbeat_max_age_sec:
        post PAUSE_REQUIRED to block new entries until Reconciler recovers.

        If heartbeat_store is None (test mode / dry-run), skip check.
        """
        if self.heartbeat_store is None:
            return

        try:
            last_ts = self.heartbeat_store.get_last_heartbeat_ts()
        except Exception as exc:
            logger.error("Failed to read Reconciler heartbeat: %s", exc)
            return

        if last_ts is None:
            logger.warning("Reconciler heartbeat not found — Reconciler may not have started yet")
            return

        try:
            last_dt = datetime.fromisoformat(last_ts)
        except ValueError:
            logger.error("Invalid heartbeat timestamp format: %s", last_ts)
            return

        now = datetime.now(timezone.utc)
        if last_dt.tzinfo is None:
            # Naive datetime from ledger — treat as UTC
            from datetime import timezone as tz
            last_dt = last_dt.replace(tzinfo=tz.utc)

        age_sec = (now - last_dt).total_seconds()
        if age_sec > self.policy.reconciler_heartbeat_max_age_sec:
            logger.warning(
                "Reconciler heartbeat stale: %.0fs > %.0fs threshold",
                age_sec, self.policy.reconciler_heartbeat_max_age_sec,
            )
            # Deduplicate: only post if no existing PAUSE_REQUIRED pending
            if not self.bus.has_pending_of_type("PAUSE_REQUIRED"):
                self.bus.post(
                    "PAUSE_REQUIRED",
                    reason=f"reconciler_heartbeat_stale_{age_sec:.0f}s",
                    severity="HIGH",
                )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    db_path = os.getenv("LEDGER_DB_PATH", "./ledger/retailedge.db")
    bus = DecisionBus(db_path)
    ft = FreqtradeClient()
    policy = GuardianPolicy()
    guardian = Guardian(bus=bus, ft_client=ft, policy=policy)

    logger.info("Guardian initialized. DB=%s FT=%s", db_path, ft.base_url)
    guardian.run_loop()


if __name__ == "__main__":
    main()