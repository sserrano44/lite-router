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


class TestSystemText:
    def test_anthropic_top_level_string(self):
        assert sk.system_text({"system": "you are helpful"}) == "you are helpful"

    def test_anthropic_top_level_blocks(self):
        data = {"system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
        assert sk.system_text(data) == "ab"

    def test_openai_system_message_string(self):
        data = {"messages": [
            {"role": "system", "content": "openai sys"},
            {"role": "user", "content": "hi"},
        ]}
        assert sk.system_text(data) == "openai sys"

    def test_openai_system_message_parts(self):
        data = {"messages": [
            {"role": "system", "content": [{"type": "text", "text": "part1 "},
                                            {"type": "text", "text": "part2"}]},
            {"role": "user", "content": "hi"},
        ]}
        assert sk.system_text(data) == "part1 part2"

    def test_top_level_wins_over_messages(self):
        data = {"system": "top", "messages": [{"role": "system", "content": "msg"}]}
        assert sk.system_text(data) == "top"

    def test_first_system_message_only(self):
        data = {"messages": [
            {"role": "system", "content": "first"},
            {"role": "system", "content": "second"},
        ]}
        assert sk.system_text(data) == "first"

    def test_absent_is_empty(self):
        assert sk.system_text({"messages": [{"role": "user", "content": "hi"}]}) == ""
        assert sk.system_text({}) == ""


class TestSessionIdFromHeaders:
    def test_priority_order(self):
        headers = {"x-claude-code-session-id": "cc", "x-session-id": "generic"}
        assert sk.session_id_from_headers(headers) == "cc"

    def test_generic_fallback(self):
        assert sk.session_id_from_headers({"x-session-id": "generic"}) == "generic"

    def test_none_present(self):
        assert sk.session_id_from_headers({"user-agent": "x"}) == ""

    def test_custom_header_list(self):
        headers = {"x-my-session": "abc"}
        assert sk.session_id_from_headers(headers, ("x-my-session",)) == "abc"

    def test_whitespace_stripped_and_skipped(self):
        headers = {"x-claude-code-session-id": "   ", "x-session-id": "real"}
        assert sk.session_id_from_headers(headers) == "real"


def test_derived_key_stable_across_openai_turns():
    """OpenCode-style: no session header, system in messages[0]; the derived
    key must be identical when the same opener is replayed on a later turn."""
    turn1 = {"messages": [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "add a login endpoint"},
    ]}
    turn2 = {"messages": [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "add a login endpoint"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "now add tests"},
    ]}
    k1, s1 = sk.derive_session_key(turn1, {}, "alias")
    k2, s2 = sk.derive_session_key(turn2, {}, "alias")
    assert s1 == s2 == "derived"
    assert k1 == k2
