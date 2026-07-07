#!/usr/bin/env python3
"""Delete raw first messages older than the retention window (R20: 90 days).

Run daily via cron/systemd timer:
    ROUTER_DATABASE_URL=postgresql://... uv run scripts/retention_purge.py
"""

from __future__ import annotations

import asyncio
import os

import asyncpg

RETENTION_DAYS = int(os.environ.get("ROUTER_MESSAGE_RETENTION_DAYS", "90"))


async def main() -> None:
    url = os.environ.get(
        "ROUTER_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:5432/postgres"
    )
    conn = await asyncpg.connect(url)
    try:
        result = await conn.execute(
            "DELETE FROM router_first_messages WHERE ts < now() - make_interval(days => $1)",
            RETENTION_DAYS,
        )
        print(f"purged rows older than {RETENTION_DAYS}d: {result}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
