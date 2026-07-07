"""HTTP client for classifier-svc. Never raises; None means fall back."""

from __future__ import annotations

import time

import httpx

from session_router import config
from session_router.config import rate_limited
from session_router.state_machine import ClassifyResult


class ClassifierClient:
    def __init__(self, base_url: str | None = None, timeout_s: float | None = None):
        self._base_url = (base_url or config.ROUTER_CLASSIFIER_URL).rstrip("/")
        self._timeout_s = timeout_s or config.ROUTER_CLASSIFIER_TIMEOUT_S
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout_s, connect=min(0.3, self._timeout_s))
            )
        return self._client

    async def classify(
        self, first_message: str, system_summary: str, repo_hints: dict
    ) -> ClassifyResult | None:
        payload = {
            "first_message": first_message[:4000],
            "system_summary": system_summary[:1000],
            "repo_hints": repo_hints,
        }
        t0 = time.perf_counter()
        try:
            resp = await self._get_client().post(f"{self._base_url}/classify", json=payload)
            if resp.status_code != 200:
                rate_limited.warning(
                    "classifier_status", "classifier returned %s", resp.status_code
                )
                return None
            body = resp.json()
            policy_name = body.get("policy_name")
            tier = config.policies_holder.get().tier_by_name(policy_name)
            if tier is None:
                rate_limited.warning(
                    "classifier_policy", "classifier returned unknown policy %r", policy_name
                )
                return None
            return ClassifyResult(
                policy_name=policy_name,
                model=tier.model,
                confidence=float(body.get("confidence", 0.0)),
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
        except Exception:
            rate_limited.warning("classifier_error", "classifier call failed", exc_info=True)
            return None

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
