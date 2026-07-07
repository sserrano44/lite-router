"""Redis-backed session records. Every method fails open — never raises."""

from __future__ import annotations

import json

import redis.asyncio as aioredis
from redis.exceptions import RedisError, ResponseError

from session_router import config
from session_router.config import rate_limited

CLAIM_TTL_S = 30


class SessionStore:
    def __init__(self, redis_url: str | None = None, ttl_seconds: int | None = None,
                 key_prefix: str | None = None):
        self._redis_url = redis_url or config.ROUTER_REDIS_URL
        self._ttl = ttl_seconds
        self._prefix = key_prefix
        self._client: aioredis.Redis | None = None
        self._getex_supported = True

    def _params(self) -> tuple[int, str]:
        pol = config.policies_holder.get().session
        return (self._ttl or pol.ttl_seconds, self._prefix or pol.redis_key_prefix)

    def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(
                self._redis_url,
                socket_timeout=config.ROUTER_REDIS_TIMEOUT_S,
                socket_connect_timeout=config.ROUTER_REDIS_TIMEOUT_S,
                decode_responses=True,
            )
        return self._client

    async def get_and_refresh(self, key: str) -> dict | None:
        """Fetch the record and slide its TTL. None on miss or Redis trouble."""
        ttl, prefix = self._params()
        rkey = prefix + key
        try:
            client = self._get_client()
            if self._getex_supported:
                try:
                    raw = await client.getex(rkey, ex=ttl)
                except ResponseError:
                    # Redis < 6.2: no GETEX
                    self._getex_supported = False
                    raw = None
            if not self._getex_supported:
                pipe = client.pipeline(transaction=False)
                pipe.get(rkey)
                pipe.expire(rkey, ttl)
                raw = (await pipe.execute())[0]
            return json.loads(raw) if raw else None
        except (RedisError, OSError, json.JSONDecodeError):
            rate_limited.warning("redis_get", "redis get_and_refresh failed, failing open",
                                 exc_info=True)
            return None

    async def claim_for_classification(self, key: str, placeholder: dict) -> bool:
        """SET NX a short-lived placeholder; True means we won the race."""
        _, prefix = self._params()
        try:
            client = self._get_client()
            return bool(
                await client.set(prefix + key, json.dumps(placeholder), nx=True, ex=CLAIM_TTL_S)
            )
        except (RedisError, OSError):
            rate_limited.warning("redis_claim", "redis claim failed, failing open", exc_info=True)
            return False

    async def write_pin(self, key: str, record: dict) -> None:
        ttl, prefix = self._params()
        try:
            await self._get_client().set(prefix + key, json.dumps(record), ex=ttl)
        except (RedisError, OSError):
            rate_limited.warning("redis_pin", "redis write_pin failed, failing open",
                                 exc_info=True)

    async def update(self, key: str, record: dict) -> None:
        """Escalation / scan-state update; keeps the sliding TTL untouched."""
        ttl, prefix = self._params()
        try:
            client = self._get_client()
            try:
                await client.set(prefix + key, json.dumps(record), keepttl=True)
            except ResponseError:
                # Redis < 6.0: no KEEPTTL — reset the TTL instead.
                await client.set(prefix + key, json.dumps(record), ex=ttl)
        except (RedisError, OSError):
            rate_limited.warning("redis_update", "redis update failed, failing open",
                                 exc_info=True)

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
