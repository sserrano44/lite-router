"""LiteAutoRouter — LiteLLM pre-call hook implementing session-pinned routing.

Contract with the proxy (verified against LiteLLM source):
- `async_pre_call_hook` must be defined directly on this class (the proxy
  checks `vars(cls)`), returns a dict to replace `data`, and fires for
  /v1/messages with call_type == "anthropic_messages".
- The hook must NEVER raise: any internal failure degrades to default_model.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

try:  # litellm only exists inside the proxy; tests run without it
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # pragma: no cover

    class CustomLogger:  # type: ignore[no-redef]
        pass


from router_common.events import DecisionEvent, EventType

from session_router import (
    client as client_mod,
    config,
    escalation,
    overrides,
    session_key as sk,
    state_machine as sm,
)
from session_router.classifier_client import ClassifierClient
from session_router.decision_log import DecisionLog
from session_router.session_store import SessionStore

logger = logging.getLogger("lite_router")

ROUTABLE_CALL_TYPES = ("anthropic_messages", "acompletion", "completion")


class LiteAutoRouter(CustomLogger):
    def __init__(
        self,
        store: SessionStore | None = None,
        classifier: ClassifierClient | None = None,
        decision_log: DecisionLog | None = None,
    ):
        self.store = store or SessionStore()
        self.classifier = classifier or ClassifierClient()
        self.decision_log = decision_log or DecisionLog()
        # Dedupe override events per (session, model) so Claude Code's
        # frequent side-channel calls don't flood router_decisions.
        self._override_seen: OrderedDict[tuple[str, str], None] = OrderedDict()

    # ------------------------------------------------------------------ hooks

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            return await self._route(user_api_key_dict, data, call_type)
        except Exception:
            logger.warning("router failed open", exc_info=True)
            try:
                if data.get("model") == config.ROUTER_VIRTUAL_MODEL:
                    data["model"] = config.policies_holder.get().default_model
                    return data
            except Exception:
                pass
            return None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Best-effort enrichment only; decision events are written in the
        # pre-call path. Kept minimal: spend lives in LiteLLM's own logs.
        return None

    # ------------------------------------------------------------------ core

    async def _route(self, user_api_key_dict, data: dict, call_type) -> dict | None:
        if call_type not in ROUTABLE_CALL_TYPES:
            return None
        model = data.get("model")
        headers = sk.extract_headers(data)
        client = client_mod.detect_client(headers, call_type)

        if model != config.ROUTER_VIRTUAL_MODEL:
            self._maybe_log_override(model, headers, client)
            return None
        if not config.ROUTER_ENABLED:
            return None  # model_list alias maps lite-auto -> default deployment

        t0 = time.perf_counter()
        policies = config.policies_holder.get()

        # Client housekeeping calls (title generation, summarization) ride the
        # same session id as the real conversation and often arrive first. Route
        # them to the cheapest tier WITHOUT classifying or pinning, so the user's
        # actual task is what defines the session's tier.
        if policies.is_side_channel(sk.system_text(data)):
            data["model"] = policies.cheapest_tier().model
            logger.debug("side-channel request routed cheap, not pinned")
            return data

        api_key_alias = (
            getattr(user_api_key_dict, "key_alias", None)
            or getattr(user_api_key_dict, "user_id", None)
            or "unknown"
        )
        base_key, key_source = sk.derive_session_key(
            data, headers, api_key_alias, config.ROUTER_SESSION_HEADERS
        )
        agent_id, parent_agent_id = sk.extract_agent_ids(headers)
        is_subagent = parent_agent_id is not None
        subagent_routing = config.SUBAGENT_ROUTING_ENABLED and is_subagent
        key = f"{base_key}:{agent_id}" if (subagent_routing and agent_id) else base_key
        event_agent_id = agent_id if is_subagent else None
        messages = data.get("messages") or []
        msg_count = len(messages) if isinstance(messages, list) else 0

        record = await self.store.get_and_refresh(key)

        if record and record.get("state") == sm.STATE_PINNED:
            decided_model = await self._handle_pinned(
                key, record, messages, headers, policies, event_agent_id,
                api_key_alias, client,
            )
        elif record:  # "classifying" placeholder — another request won the race
            decided_model = policies.default_model
        else:
            decided_model = await self._handle_first_request(
                key, data, headers, policies, api_key_alias, msg_count,
                subagent_routing, base_key, event_agent_id, client,
            )

        stash = {
            "session_key": key,
            "key_source": key_source,
            "model": decided_model,
            "shadow": config.SHADOW_MODE,
        }
        meta = data.get("litellm_metadata")
        if not isinstance(meta, dict):
            meta = data.setdefault("metadata", {}) if not isinstance(
                data.get("metadata"), dict
            ) else data["metadata"]
        meta["lite_router"] = stash

        data["model"] = policies.default_model if config.SHADOW_MODE else decided_model
        if config.ROUTER_TIMING_LOG:
            logger.info(
                "router timing key=%s ms=%.2f model=%s",
                key, (time.perf_counter() - t0) * 1000, decided_model,
            )
        return data

    async def _handle_pinned(
        self, key, record, messages, headers, policies, event_agent_id,
        api_key_alias, client,
    ) -> str:
        scan = escalation.ScanState.from_dict(record.get("scan"))
        signal, new_scan = escalation.detect(messages, headers, scan, policies.escalation)
        updated = None
        if signal is not None:
            escalated = sm.escalate(record, policies)
            if escalated is not None:
                new_scan = escalation.consume(new_scan, len(messages))
                escalated["scan"] = new_scan.to_dict()
                updated = escalated
                self.decision_log.emit(DecisionEvent(
                    session_key=key,
                    event_type=EventType.ESCALATED,
                    model=escalated["model"],
                    policy_name=escalated["policy_name"],
                    api_key_alias=api_key_alias,
                    shadow=config.SHADOW_MODE,
                    agent_id=event_agent_id,
                    client=client,
                    detail={"reason": signal.reason, **signal.detail,
                            "from_model": record["model"],
                            "escalations": escalated["escalations"]},
                ))
        if updated is None and new_scan.to_dict() != record.get("scan"):
            updated = dict(record)
            updated["scan"] = new_scan.to_dict()
        if updated is not None:
            await self.store.update(key, updated)
            record = updated
        return record["model"]

    async def _handle_first_request(
        self, key, data, headers, policies, api_key_alias, msg_count,
        subagent_routing, base_key, event_agent_id, client,
    ) -> str:
        claimed = await self.store.claim_for_classification(
            key, sm.classifying_placeholder(policies)
        )
        if not claimed:
            # Lost the race (or Redis is down): default for this one request.
            return policies.default_model

        first_msg = sk.first_user_message_text(data)
        fmh = sk.first_message_hash(data)
        system = sk.system_text(data)
        shadow = config.SHADOW_MODE

        if subagent_routing:
            # R1b: pin one tier below the parent session's pin (floor: cheapest).
            parent = await self.store.get_and_refresh(base_key)
            if parent and parent.get("state") == sm.STATE_PINNED:
                tier = sm.subagent_tier(policies, parent)
            else:
                tier = policies.default_tier()
            record = sm.build_pin_record(
                policies,
                classify=sm.ClassifyResult(tier.name, tier.model, 1.0),
                path_override=False, first_message_hash=fmh,
                api_key_alias=api_key_alias, msg_count=msg_count,
            )
            record["subagent"] = True
            await self.store.write_pin(key, record)
            self.decision_log.emit(DecisionEvent(
                session_key=key, event_type=EventType.PINNED, model=record["model"],
                policy_name=record["policy_name"], first_message_hash=fmh,
                api_key_alias=api_key_alias, shadow=shadow, agent_id=event_agent_id,
                client=client, detail={"subagent": True, "parent_key": base_key},
            ))
            return record["model"]

        # Repo-hint auto-detection parses Claude Code's system-prompt env block;
        # other clients get the header-only override path (empty hints).
        hints = (
            overrides.extract_repo_hints(system)
            if client == client_mod.CLIENT_CLAUDE_CODE
            else overrides.RepoHints()
        )
        matched = overrides.match_path_override(hints, headers, policies)
        classify = None
        classify_latency = None
        if matched is None:
            classify = await self.classifier.classify(
                first_msg, system[:1000],
                {"cwd": hints.cwd, "claude_md_excerpt": hints.claude_md_excerpt},
            )
            classify_latency = classify.latency_ms if classify else None

        record = sm.build_pin_record(
            policies, classify=classify, path_override=matched is not None,
            first_message_hash=fmh, api_key_alias=api_key_alias, msg_count=msg_count,
        )
        await self.store.write_pin(key, record)

        if matched is None:
            self.decision_log.emit(DecisionEvent(
                session_key=key,
                event_type=EventType.CLASSIFIED if classify else EventType.FALLBACK,
                model=record["model"], policy_name=record["policy_name"],
                confidence=record["confidence"], first_message_hash=fmh,
                api_key_alias=api_key_alias, latency_ms=classify_latency,
                shadow=shadow, agent_id=event_agent_id, client=client,
                detail={} if classify else {"reason": "classifier_unavailable"},
                raw_first_message=(
                    first_msg[:4000] if config.CAPTURE_FIRST_MESSAGES and first_msg else None
                ),
                system_excerpt=system[:1000] if config.CAPTURE_FIRST_MESSAGES else None,
            ))
        self.decision_log.emit(DecisionEvent(
            session_key=key, event_type=EventType.PINNED, model=record["model"],
            policy_name=record["policy_name"], confidence=record["confidence"],
            first_message_hash=fmh, api_key_alias=api_key_alias,
            shadow=shadow, agent_id=event_agent_id, client=client,
            detail={"path_override": matched} if matched else {},
        ))
        return record["model"]

    def _maybe_log_override(self, model, headers: dict[str, str], client: str) -> None:
        """R11: concrete model name bypasses the router; log once per session+model."""
        if not isinstance(model, str) or not model:
            return
        session_id = sk.session_id_from_headers(headers, config.ROUTER_SESSION_HEADERS)
        if not session_id:
            return  # session not resolvable — skip (cheap path for curl etc.)
        dedupe_key = (session_id, model)
        if dedupe_key in self._override_seen:
            return
        self._override_seen[dedupe_key] = None
        if len(self._override_seen) > 4096:
            self._override_seen.popitem(last=False)
        self.decision_log.emit(DecisionEvent(
            session_key=session_id, event_type=EventType.OVERRIDE, model=model,
            shadow=config.SHADOW_MODE, client=client,
            detail={"requested_model": model},
        ))


proxy_handler_instance = LiteAutoRouter()
