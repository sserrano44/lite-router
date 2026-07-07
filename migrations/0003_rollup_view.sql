-- Per-session rollup for Metabase (R19). Spend joins happen in Metabase
-- against LiteLLM's own spend logs on api_key_alias + time window.
CREATE OR REPLACE VIEW router_session_rollup AS
SELECT
    session_key,
    min(ts) AS started_at,
    max(ts) AS last_event_at,
    bool_or(shadow) AS shadow,
    max(api_key_alias) AS api_key_alias,
    max(CASE WHEN event_type = 'classified' THEN policy_name END) AS initial_policy,
    max(CASE WHEN event_type = 'pinned' THEN policy_name END) AS pinned_policy,
    max(CASE WHEN event_type = 'pinned' THEN model END) AS pinned_model,
    count(*) FILTER (WHERE event_type = 'escalated') AS escalations,
    bool_or(event_type = 'fallback') AS had_fallback,
    bool_or(event_type = 'override') AS had_override,
    count(DISTINCT agent_id) FILTER (WHERE agent_id IS NOT NULL) AS subagents,
    count(*) AS events
FROM router_decisions
GROUP BY session_key;
