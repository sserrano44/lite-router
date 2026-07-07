import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Must be set before session_router.config is imported anywhere.
os.environ.setdefault("ROUTER_POLICIES_PATH", str(REPO_ROOT / "policies.yaml"))
os.environ.setdefault("ROUTER_DATABASE_URL", "")

import pytest  # noqa: E402

from router_common.policies import load_policies  # noqa: E402


@pytest.fixture(scope="session")
def policies():
    return load_policies(REPO_ROOT / "policies.yaml")
