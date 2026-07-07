from session_router import session_key as sk


def _data(headers=None, where="proxy_server_request", messages=None, system=None):
    data = {}
    if headers is not None:
        if where == "proxy_server_request":
            data["proxy_server_request"] = {"headers": headers}
        else:
            data[where] = {"headers": headers}
    if messages is not None:
        data["messages"] = messages
    if system is not None:
        data["system"] = system
    return data


class TestExtractHeaders:
    def test_proxy_server_request_location(self):
        data = _data({"X-Claude-Code-Session-Id": "abc"})
        assert sk.extract_headers(data)["x-claude-code-session-id"] == "abc"

    def test_metadata_location(self):
        data = _data({"x-claude-code-session-id": "abc"}, where="metadata")
        assert sk.extract_headers(data)["x-claude-code-session-id"] == "abc"

    def test_litellm_metadata_location(self):
        data = _data({"X-CLAUDE-CODE-SESSION-ID": "abc"}, where="litellm_metadata")
        assert sk.extract_headers(data)["x-claude-code-session-id"] == "abc"

    def test_merges_all_locations(self):
        data = {
            "proxy_server_request": {"headers": {"a": "1"}},
            "metadata": {"headers": {"b": "2"}},
            "litellm_metadata": {"headers": {"C": "3"}},
        }
        merged = sk.extract_headers(data)
        assert merged == {"a": "1", "b": "2", "c": "3"}

    def test_missing_and_malformed(self):
        assert sk.extract_headers({}) == {}
        assert sk.extract_headers({"metadata": {"headers": "nope"}}) == {}
        assert sk.extract_headers({"proxy_server_request": {"headers": {1: "x", "k": 2}}}) == {}


class TestDeriveSessionKey:
    def test_header_wins(self):
        data = _data({"x-claude-code-session-id": "sess-123"})
        key, source = sk.derive_session_key(data, sk.extract_headers(data), "alias")
        assert (key, source) == ("sess-123", "header")

    def test_fallback_deterministic(self):
        data = _data({}, messages=[{"role": "user", "content": "hello"}], system="sys")
        k1, s1 = sk.derive_session_key(data, {}, "alias")
        k2, _ = sk.derive_session_key(data, {}, "alias")
        assert s1 == "derived"
        assert k1 == k2
        assert len(k1) == 16
        int(k1, 16)  # hex

    def test_fallback_varies_by_inputs(self):
        base = _data({}, messages=[{"role": "user", "content": "hello"}], system="sys")
        other = _data({}, messages=[{"role": "user", "content": "different"}], system="sys")
        k1, _ = sk.derive_session_key(base, {}, "alias")
        k2, _ = sk.derive_session_key(other, {}, "alias")
        k3, _ = sk.derive_session_key(base, {}, "other-alias")
        assert len({k1, k2, k3}) == 3


class TestFlattenContent:
    def test_string(self):
        assert sk.flatten_content("hi") == "hi"

    def test_text_blocks(self):
        content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert sk.flatten_content(content) == "ab"

    def test_nested_tool_result_content(self):
        content = [{"type": "tool_result", "content": [{"type": "text", "text": "out"}]}]
        assert sk.flatten_content(content) == "out"

    def test_image_blocks_ignored(self):
        content = [{"type": "image", "source": {"data": "AAAA"}}, {"type": "text", "text": "x"}]
        assert sk.flatten_content(content) == "x"

    def test_limit(self):
        assert len(sk.flatten_content("x" * 9000, limit=4000)) == 4000
        blocks = [{"type": "text", "text": "y" * 3000}, {"type": "text", "text": "z" * 3000}]
        assert len(sk.flatten_content(blocks, limit=4000)) == 4000


class TestAgentIds:
    def test_extract(self):
        headers = {
            "x-claude-code-agent-id": "agent-1",
            "x-claude-code-parent-agent-id": "parent-1",
        }
        assert sk.extract_agent_ids(headers) == ("agent-1", "parent-1")

    def test_absent(self):
        assert sk.extract_agent_ids({}) == (None, None)


def test_first_user_message_skips_assistant():
    data = {"messages": [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "question"},
    ]}
    assert sk.first_user_message_text(data) == "question"
