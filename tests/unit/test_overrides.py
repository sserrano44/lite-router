from session_router import overrides


SYSTEM = """You are Claude Code.
<env>
Working directory: /home/dev/capyfi-rewards
Is a git repository: true
</env>
Contents of CLAUDE.md (project instructions):
# CLAUDE.md
This repo holds the rewards engine.
"""


class TestExtractRepoHints:
    def test_cwd(self):
        hints = overrides.extract_repo_hints(SYSTEM)
        assert hints.cwd == "/home/dev/capyfi-rewards"

    def test_claude_md_excerpt(self):
        hints = overrides.extract_repo_hints(SYSTEM)
        assert hints.claude_md_excerpt.startswith("CLAUDE.md")
        assert "rewards engine" in hints.claude_md_excerpt

    def test_empty_system(self):
        hints = overrides.extract_repo_hints("")
        assert hints.cwd == "" and hints.claude_md_excerpt == ""


class TestMatchPathOverride:
    def test_cwd_pattern_match(self, policies):
        hints = overrides.RepoHints(cwd="/home/dev/capyfi-rewards")
        assert overrides.match_path_override(hints, {}, policies) == "capyfi"

    def test_case_insensitive(self, policies):
        hints = overrides.RepoHints(cwd="/home/dev/Solidity-Playground")
        assert overrides.match_path_override(hints, {}, policies) == "solidity"

    def test_claude_md_excerpt_match(self, policies):
        hints = overrides.RepoHints(claude_md_excerpt="CLAUDE.md — the custody signer service")
        assert overrides.match_path_override(hints, {}, policies) in ("custody", "signer")

    def test_no_match(self, policies):
        hints = overrides.RepoHints(cwd="/home/dev/website")
        assert overrides.match_path_override(hints, {}, policies) is None

    def test_empty_hints_no_match(self, policies):
        assert overrides.match_path_override(overrides.RepoHints(), {}, policies) is None

    def test_tier_header_forces(self, policies):
        assert (
            overrides.match_path_override(
                overrides.RepoHints(), {"x-lite-tier": "hard_dev"}, policies
            )
            == "header:x-lite-tier"
        )

    def test_tier_header_other_value_ignored(self, policies):
        assert (
            overrides.match_path_override(
                overrides.RepoHints(), {"x-lite-tier": "standard_dev"}, policies
            )
            is None
        )
