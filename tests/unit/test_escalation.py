from session_router import escalation
from session_router.escalation import ScanState


def _user(text):
    return {"role": "user", "content": text}


def _assistant(text="ok"):
    return {"role": "assistant", "content": text}


def _tool_result(text, is_error=False):
    block = {"type": "tool_result", "tool_use_id": "t", "content": text}
    if is_error:
        block["is_error"] = True
    return {"role": "user", "content": [block]}


def detect(messages, headers=None, scan=None, policies=None):
    return escalation.detect(messages, headers or {}, scan or ScanState(),
                             policies.escalation)


class TestRetryPatterns:
    def test_still_failing(self, policies):
        signal, _ = detect([_user("build it"), _assistant(), _user("it's still failing")],
                           policies=policies)
        assert signal and signal.reason == "retry_text"

    def test_anchored_no_fires(self, policies):
        signal, _ = detect([_user("No, that's not what I asked")], policies=policies)
        assert signal and signal.reason == "retry_text"

    def test_anchored_no_does_not_match_note(self, policies):
        signal, _ = detect([_user("note the docs say otherwise, please refactor")],
                           policies=policies)
        assert signal is None

    def test_retry_only_on_final_message(self, policies):
        msgs = [_user("still failing"), _assistant(), _user("thanks, looks good")]
        signal, _ = detect(msgs, policies=policies)
        assert signal is None

    def test_case_insensitive(self, policies):
        signal, _ = detect([_user("STILL FAILING after that")], policies=policies)
        assert signal is not None

    def test_final_message_with_tool_results_not_retry(self, policies):
        # tool_result content echoing user-like text must not be read as a retry
        signal, _ = detect([_tool_result("try again later")], policies=policies)
        assert signal is None


class TestToolFailures:
    def test_two_consecutive_failures(self, policies):
        msgs = [_user("run tests"), _assistant(),
                _tool_result("FAILED tests/test_x.py"), _assistant(),
                _tool_result("Traceback (most recent call last):\n  ...")]
        signal, state = detect(msgs, policies=policies)
        assert signal and signal.reason == "tool_failures"
        assert state.consec_failures == 2

    def test_failure_success_failure_no_signal(self, policies):
        msgs = [_tool_result("FAILED x"), _assistant(),
                _tool_result("all good"), _assistant(),
                _tool_result("error TS2345: nope")]
        signal, state = detect(msgs, policies=policies)
        assert signal is None
        assert state.consec_failures == 1

    def test_is_error_flag_counts(self, policies):
        msgs = [_tool_result("something", is_error=True), _tool_result("boom", is_error=True)]
        signal, _ = detect(msgs, policies=policies)
        assert signal and signal.reason == "tool_failures"

    def test_exit_code_marker(self, policies):
        msgs = [_tool_result("ExitCode: 1"), _tool_result("ExitCode: 2")]
        signal, _ = detect(msgs, policies=policies)
        assert signal is not None

    def test_exit_code_zero_clean(self, policies):
        msgs = [_tool_result("ExitCode: 0"), _tool_result("ExitCode: 0")]
        signal, state = detect(msgs, policies=policies)
        assert signal is None
        assert state.consec_failures == 0


class TestHeader:
    def test_escalate_header(self, policies):
        signal, state = detect([_user("hi")], headers={"x-router-escalate": "true"},
                               policies=policies)
        assert signal and signal.reason == "header"
        assert state.escalated_at_msg_count == 1

    def test_header_false_ignored(self, policies):
        signal, _ = detect([_user("hi")], headers={"x-router-escalate": "false"},
                           policies=policies)
        assert signal is None


class TestWatermark:
    def test_no_refire_on_unchanged_history(self, policies):
        msgs = [_tool_result("FAILED a"), _tool_result("FAILED b")]
        signal, state = detect(msgs, policies=policies)
        assert signal is not None
        # Same history replayed next request: watermark blocks a rescan.
        signal2, state2 = detect(msgs, scan=state, policies=policies)
        assert signal2 is None
        assert state2.msg_count == 2

    def test_consume_resets_counters(self, policies):
        msgs = [_tool_result("FAILED a"), _tool_result("FAILED b")]
        signal, state = detect(msgs, policies=policies)
        state = escalation.consume(state, len(msgs))
        assert state.consec_failures == 0
        assert state.escalated_at_msg_count == 2
        # One more failure after the consumed escalation: streak restarts at 1.
        msgs2 = msgs + [_tool_result("FAILED c")]
        signal2, state2 = detect(msgs2, scan=state, policies=policies)
        assert signal2 is None
        assert state2.consec_failures == 1

    def test_history_shrink_resets(self, policies):
        state = ScanState(msg_count=50, consec_failures=1, escalated_at_msg_count=40)
        msgs = [_user("hello")] * 5
        signal, new_state = detect(msgs, scan=state, policies=policies)
        assert signal is None
        assert new_state.msg_count == 5
        assert new_state.consec_failures == 0

    def test_scan_bounded_to_tail(self, policies):
        # Failures buried beyond the 10-message tail are never scanned.
        msgs = [_tool_result("FAILED old"), _tool_result("FAILED old2")]
        msgs += [_assistant()] * 20
        msgs += [_user("continue please")]
        signal, state = detect(msgs, policies=policies)
        assert signal is None
        assert state.msg_count == len(msgs)

    def test_retry_fence_after_escalation(self, policies):
        # Final message predates the escalation fence -> no retry signal.
        msgs = [_user("try again")]
        state = ScanState(msg_count=0, escalated_at_msg_count=5)
        signal, _ = detect(msgs, scan=state, policies=policies)
        assert signal is None


def _oai_tool(text):
    """OpenAI-shape tool result: a role=tool message with string content."""
    return {"role": "tool", "tool_call_id": "t", "content": text}


def _oai_assistant_toolcall():
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": "t", "type": "function",
                            "function": {"name": "bash", "arguments": "{}"}}]}


class TestOpenAIShape:
    def test_two_consecutive_tool_failures(self, policies):
        msgs = [
            {"role": "system", "content": "sys"},
            _user("run tests"),
            _oai_assistant_toolcall(),
            _oai_tool("FAILED tests/test_x.py"),
            _oai_assistant_toolcall(),
            _oai_tool("Traceback (most recent call last):\n  ..."),
        ]
        signal, state = detect(msgs, policies=policies)
        assert signal and signal.reason == "tool_failures"
        assert state.consec_failures == 2

    def test_failure_success_failure_no_signal(self, policies):
        msgs = [
            _oai_tool("FAILED x"), _oai_assistant_toolcall(),
            _oai_tool("all good"), _oai_assistant_toolcall(),
            _oai_tool("error TS2345: nope"),
        ]
        signal, state = detect(msgs, policies=policies)
        assert signal is None
        assert state.consec_failures == 1

    def test_retry_text_on_final_user_message(self, policies):
        msgs = [
            {"role": "system", "content": "sys"},
            _user("build it"),
            _oai_assistant_toolcall(),
            _oai_tool("ok"),
            _user("that's wrong, try again"),
        ]
        signal, _ = detect(msgs, policies=policies)
        assert signal and signal.reason == "retry_text"

    def test_tool_result_not_read_as_retry(self, policies):
        # A final role=tool message echoing retry-like text is not retry text.
        signal, _ = detect([_oai_tool("try again later")], policies=policies)
        assert signal is None

    def test_watermark_no_refire(self, policies):
        msgs = [_oai_tool("FAILED a"), _oai_tool("FAILED b")]
        signal, state = detect(msgs, policies=policies)
        assert signal is not None
        signal2, state2 = detect(msgs, scan=state, policies=policies)
        assert signal2 is None
        assert state2.msg_count == 2


def test_non_list_messages_safe(policies):
    signal, state = detect(None, policies=policies)  # type: ignore[arg-type]
    assert signal is None
    assert state.msg_count == 0
