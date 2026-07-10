from session_router import client


class TestDetectClient:
    def test_claude_code_by_header(self):
        headers = {"x-claude-code-session-id": "s", "user-agent": "opencode/1.0"}
        # Claude Code headers are authoritative even if UA says otherwise.
        assert client.detect_client(headers) == client.CLIENT_CLAUDE_CODE

    def test_claude_code_any_prefixed_header(self):
        assert client.detect_client({"x-claude-code-agent-id": "a"}) == client.CLIENT_CLAUDE_CODE

    def test_opencode_by_user_agent(self):
        assert client.detect_client({"user-agent": "opencode/0.3.1"}) == client.CLIENT_OPENCODE

    def test_opencode_user_agent_case_insensitive(self):
        assert client.detect_client({"user-agent": "OpenCode-CLI"}) == client.CLIENT_OPENCODE

    def test_generic_when_unknown(self):
        assert client.detect_client({"user-agent": "curl/8.0"}) == client.CLIENT_GENERIC

    def test_generic_when_empty(self):
        assert client.detect_client({}) == client.CLIENT_GENERIC

    def test_call_type_is_not_decisive(self):
        assert client.detect_client({}, call_type="acompletion") == client.CLIENT_GENERIC
