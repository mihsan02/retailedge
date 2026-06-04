"""
sidecar/guardian/ft_client.py

Freqtrade REST client for RetailEdge Guardian.
Single responsibility: wrap Freqtrade REST API calls with strict 2xx checking.

Hard rules (CLAUDE.md):
- 2xx = success. Anything else = failure. Never assume success.
- ping() must be called before any mutating action.
- No silent swallowing of non-2xx responses.

Design:
- No retry logic here — retry lives in Guardian/Decision Bus layer.
- Raises FreqtradeClientError on non-2xx so Guardian can route to mark_failed_retryable.
- All methods return the raw response dict on success.
- ping_alive() is the only method that returns bool (not dict) — used for liveness gating.

Dependencies: requests only. No ccxt, no Freqtrade SDK.
"""

import os
from typing import Any, Optional
import requests
from requests.auth import HTTPBasicAuth
from requests.exceptions import ConnectionError, Timeout, RequestException


class FreqtradeClientError(Exception):
    """
    Raised when Freqtrade REST returns non-2xx or is unreachable.
    Carries status_code (None if connection failed entirely).
    Guardian catches this and routes to mark_failed_retryable or OPERATOR_REQUIRED.
    """
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class FreqtradeClient:
    """
    Thin HTTP wrapper over Freqtrade REST API.

    All mutating calls (pause, reload_config, forceexit) must be preceded
    by ping_alive() in the Guardian layer. This client does not enforce that
    order — Guardian owns the liveness gate.

    Instantiation reads from environment variables by default.
    Pass explicit args for test injection.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout_sec: float = 5.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("FREQTRADE_API_URL", "http://localhost:8080")).rstrip("/")
        self.auth = HTTPBasicAuth(
            username or os.getenv("FREQTRADE_API_USER", "retailedge"),
            password or os.getenv("FREQTRADE_API_PASS", ""),
        )
        self.timeout = timeout_sec

    # -----------------------------------------------------------------------
    # Liveness
    # -----------------------------------------------------------------------

    def ping_alive(self) -> bool:
        """
        Return True if Freqtrade REST is reachable and responding.
        Returns False on any error — never raises.

        Guardian calls this before every mutating action.
        A False result means: do not attempt actuation, post OPERATOR_REQUIRED.
        """
        try:
            resp = self._get("/api/v1/ping")
            return True
        except (FreqtradeClientError, RequestException):
            return False

    # -----------------------------------------------------------------------
    # Read endpoints
    # -----------------------------------------------------------------------

    def get_trades(self) -> dict[str, Any]:
        """GET /api/v1/trades — reconciler trade state sync."""
        return self._get("/api/v1/trades")

    def get_open_orders(self) -> dict[str, Any]:
        """GET /api/v1/openorders — reconciler open order sync."""
        return self._get("/api/v1/openorders")

    def get_status(self) -> dict[str, Any]:
        """GET /api/v1/status — current open trade status."""
        return self._get("/api/v1/status")

    # -----------------------------------------------------------------------
    # Mutating endpoints
    # -----------------------------------------------------------------------

    def pause(self, reason: str = "") -> dict[str, Any]:
        """
        POST /api/v1/pause — pause entry.
        Raises FreqtradeClientError on non-2xx.
        Guardian marks DONE only after this succeeds.
        """
        return self._post("/api/v1/pause", payload={"reason": reason})

    def reload_config(self) -> dict[str, Any]:
        """
        POST /api/v1/reload_config — apply new champion config.
        Used by Guardian after SCHEDULED_MODEL_PROMOTION.
        Raises FreqtradeClientError on non-2xx.
        """
        return self._post("/api/v1/reload_config")

    def forceexit(
        self,
        trade_id: str,
        ordertype: str = "market",
        price: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        POST /api/v1/forceexit — emergency exit.
        ordertype: "market" (first attempt) or "limit" (aggressive fallback).
        price: required for limit orders.
        Raises FreqtradeClientError on non-2xx.
        """
        payload: dict[str, Any] = {
            "tradeid": trade_id,
            "ordertype": ordertype,
        }
        if price is not None:
            payload["price"] = price
        return self._post("/api/v1/forceexit", payload=payload)

    # -----------------------------------------------------------------------
    # HTTP primitives
    # -----------------------------------------------------------------------

    def _get(self, path: str) -> dict[str, Any]:
        """
        Execute GET request. Raises FreqtradeClientError on non-2xx or connection failure.
        Returns parsed JSON dict on 2xx.
        """
        url = self.base_url + path
        try:
            resp = requests.get(url, auth=self.auth, timeout=self.timeout)
        except (ConnectionError, Timeout) as exc:
            raise FreqtradeClientError(
                f"GET {path} connection failed: {exc}", status_code=None
            )
        except RequestException as exc:
            raise FreqtradeClientError(
                f"GET {path} request error: {exc}", status_code=None
            )

        if not (200 <= resp.status_code < 300):
            raise FreqtradeClientError(
                f"GET {path} returned {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
            )
        return resp.json()

    def _post(self, path: str, payload: Optional[dict] = None) -> dict[str, Any]:
        """
        Execute POST request. Raises FreqtradeClientError on non-2xx or connection failure.
        Returns parsed JSON dict on 2xx.
        """
        url = self.base_url + path
        try:
            resp = requests.post(
                url,
                json=payload or {},
                auth=self.auth,
                timeout=self.timeout,
            )
        except (ConnectionError, Timeout) as exc:
            raise FreqtradeClientError(
                f"POST {path} connection failed: {exc}", status_code=None
            )
        except RequestException as exc:
            raise FreqtradeClientError(
                f"POST {path} request error: {exc}", status_code=None
            )

        if not (200 <= resp.status_code < 300):
            raise FreqtradeClientError(
                f"POST {path} returned {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
            )
        return resp.json()