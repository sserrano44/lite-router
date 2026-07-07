#!/usr/bin/env python3
"""Apply migrations/*.sql in filename order, tracked in schema_migrations.

Usage: ROUTER_DATABASE_URL=postgresql://... uv run scripts/migrate.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import asyncpg

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


async def main() -> int:
    url = os.environ.get(
        "ROUTER_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:5432/postgres"
    )
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " filename text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        applied = {
            r["filename"] for r in await conn.fetch("SELECT filename FROM schema_migrations")
        }
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in applied:
                print(f"skip  {path.name}")
                continue
            sql = path.read_text(encoding="utf-8")
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES ($1)", path.name
                )
            print(f"apply {path.name}")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
