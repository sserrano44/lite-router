import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The unit suite runs against a frozen 3-tier fixture, decoupled from the
# shipped example (repo-root policies.yaml) so the example can evolve freely.
FIXTURE_POLICIES = REPO_ROOT / "tests" / "fixtures" / "policies.yaml"

# Must be set before session_router.config is imported anywhere.
os.environ.setdefault("ROUTER_POLICIES_PATH", str(FIXTURE_POLICIES))
os.environ.setdefault("ROUTER_DATABASE_URL", "")

import pytest  # noqa: E402

from router_common.policies import load_policies  # noqa: E402


@pytest.fixture(scope="session")
def policies():
    return load_policies(FIXTURE_POLICIES)
