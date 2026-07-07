"""Async decision logging to Postgres.

Hot-path cost is a single Queue.put_nowait; a background task batches inserts.
Postgres being down never touches the request path — events queue up to the
cap, then drop with a counter.
"""

from __future__ import annotations

import asyncio
import json
import logging

from router_common.events import DecisionEvent

from session_router import config
from session_router.config import rate_limited

logger = logging.getLogger("ripio_router")

QUEUE_MAX = 10_000
BATCH_MAX = 200
FLUSH_INTERVAL_S = 2.0
BACKOFF_MAX_S = 30.0

_DECISIONS_SQL = """
INSERT INTO router_decisions
  (session_key, event_type, policy_name, model, confidence,
   first_message_hash, api_key_alias, latency_ms, shadow, agent_id, detail)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
"""

_FIRST_MESSAGES_SQL = """
INSERT INTO router_first_messages (first_message_hash, session_key, raw_message, system_excerpt)
VALUES ($1, $2, $3, $4)
ON CONFLICT (first_message_hash) DO NOTHING
"""


class DecisionLog:
    def __init__(self, database_url: str | None = None):
        self._database_url = database_url if database_url is not None else config.ROUTER_DATABASE_URL
        self._queue: asyncio.Queue[DecisionEvent] | None = None
        self._task: asyncio.Task | None = None
        self._pool = None
        self.dropped = 0

    def emit(self, event: DecisionEvent) -> None:
        """The only hot-path entry point. Never raises, never blocks."""
        if not self._database_url:
            return
        try:
            if self._queue is None:
                self._queue = asyncio.Queue(maxsize=QUEUE_MAX)
                self._task = asyncio.get_running_loop().create_task(self._flusher())
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1
            rate_limited.warning(
                "log_queue_full", "decision log queue full, dropped=%d", self.dropped
            )
        except RuntimeError:
            # No running loop (import-time call) — drop silently.
            self.dropped += 1

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self._database_url, min_size=0, max_size=2, command_timeout=10
            )
        return self._pool

    async def _flusher(self) -> None:
        backoff = 1.0
        while True:
            batch = await self._collect_batch()
            if not batch:
                continue
            try:
                await self._write_batch(batch)
                backoff = 1.0
            except Exception:
                rate_limited.warning("log_pg", "decision log write failed", exc_info=True)
                self._pool = None
                # Requeue what fits; the rest is dropped.
                for ev in batch:
                    try:
                        self._queue.put_nowait(ev)  # type: ignore[union-attr]
                    except asyncio.QueueFull:
                        self.dropped += 1
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX_S)

    async def _collect_batch(self) -> list[DecisionEvent]:
        assert self._queue is not None
        batch: list[DecisionEvent] = [await self._queue.get()]
        deadline = asyncio.get_running_loop().time() + FLUSH_INTERVAL_S
        while len(batch) < BATCH_MAX:
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout))
            except asyncio.TimeoutError:
                break
        return batch

    async def _write_batch(self, batch: list[DecisionEvent]) -> None:
        pool = await self._get_pool()
        decision_rows = [
            (
                ev.session_key,
                str(ev.event_type),
                ev.policy_name,
                ev.model,
                ev.confidence,
                ev.first_message_hash,
                ev.api_key_alias,
                ev.latency_ms,
                ev.shadow,
                ev.agent_id,
                json.dumps(ev.detail or {}),
            )
            for ev in batch
        ]
        message_rows = [
            (ev.first_message_hash, ev.session_key, ev.raw_first_message, ev.system_excerpt)
            for ev in batch
            if ev.raw_first_message and ev.first_message_hash
        ]
        async with pool.acquire() as conn:
            await conn.executemany(_DECISIONS_SQL, decision_rows)
            if message_rows:
                await conn.executemany(_FIRST_MESSAGES_SQL, message_rows)

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception:
                pass
            self._pool = None
