import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Single source of truth: tests run against the shipped example policies.yaml.
POLICIES = REPO_ROOT / "policies.yaml"

# Must be set before session_router.config is imported anywhere.
os.environ.setdefault("ROUTER_POLICIES_PATH", str(POLICIES))
os.environ.setdefault("ROUTER_DATABASE_URL", "")

import pytest  # noqa: E402

from router_common.policies import load_policies  # noqa: E402


@pytest.fixture(scope="session")
def policies():
    return load_policies(POLICIES)
