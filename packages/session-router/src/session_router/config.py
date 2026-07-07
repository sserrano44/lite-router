"""Environment configuration for the session router hook.

Everything is read from the environment; the hook must be droppable into an
existing LiteLLM container with env vars only. Policies are re-checked by
mtime at most once per 60s so a PR-merged policies.yaml lands without a
restart.
"""

from __future__ import annotations

import logging
import os
import time

from router_common.policies import PoliciesConfig, load_policies

logger = logging.getLogger("ripio_router")


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


ROUTER_ENABLED = _env_bool("ROUTER_ENABLED", True)
SHADOW_MODE = _env_bool("SHADOW_MODE", True)
SUBAGENT_ROUTING_ENABLED = _env_bool("SUBAGENT_ROUTING_ENABLED", False)
CAPTURE_FIRST_MESSAGES = _env_bool("ROUTER_CAPTURE_FIRST_MESSAGES", True)

ROUTER_VIRTUAL_MODEL = os.environ.get("ROUTER_VIRTUAL_MODEL", "ripio-auto")
ROUTER_REDIS_URL = os.environ.get("ROUTER_REDIS_URL", "redis://127.0.0.1:6379/0")
ROUTER_CLASSIFIER_URL = os.environ.get("ROUTER_CLASSIFIER_URL", "http://127.0.0.1:8891")
ROUTER_DATABASE_URL = os.environ.get("ROUTER_DATABASE_URL", "")
ROUTER_POLICIES_PATH = os.environ.get("ROUTER_POLICIES_PATH", "/app/policies.yaml")
ROUTER_CLASSIFIER_TIMEOUT_S = float(os.environ.get("ROUTER_CLASSIFIER_TIMEOUT_S", "1.0"))
ROUTER_REDIS_TIMEOUT_S = float(os.environ.get("ROUTER_REDIS_TIMEOUT_S", "0.25"))
ROUTER_TIMING_LOG = _env_bool("ROUTER_TIMING_LOG", False)

_POLICIES_RECHECK_S = 60.0


class PoliciesHolder:
    """Loads policies.yaml once and re-checks mtime at most every 60s."""

    def __init__(self, path: str):
        self._path = path
        self._policies: PoliciesConfig | None = None
        self._mtime: float = 0.0
        self._next_check: float = 0.0

    def get(self) -> PoliciesConfig:
        now = time.monotonic()
        if self._policies is not None and now < self._next_check:
            return self._policies
        self._next_check = now + _POLICIES_RECHECK_S
        try:
            mtime = os.stat(self._path).st_mtime
            if self._policies is None or mtime != self._mtime:
                self._policies = load_policies(self._path)
                self._mtime = mtime
                logger.info("loaded policies from %s (mtime=%s)", self._path, mtime)
        except Exception:
            if self._policies is None:
                raise
            logger.warning("failed to reload policies, keeping previous", exc_info=True)
        return self._policies


policies_holder = PoliciesHolder(ROUTER_POLICIES_PATH)


class RateLimitedLogger:
    """Logs a given key at most once per interval to avoid log storms."""

    def __init__(self, interval_s: float = 30.0):
        self._interval = interval_s
        self._last: dict[str, float] = {}

    def warning(self, key: str, msg: str, *args, **kwargs) -> None:
        now = time.monotonic()
        if now - self._last.get(key, 0.0) >= self._interval:
            self._last[key] = now
            logger.warning(msg, *args, **kwargs)


rate_limited = RateLimitedLogger()
