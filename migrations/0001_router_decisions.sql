CREATE TABLE IF NOT EXISTS router_decisions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_key text NOT NULL,
    ts timestamptz NOT NULL DEFAULT now(),
    event_type text NOT NULL
        CHECK (event_type IN ('classified', 'pinned', 'escalated', 'override', 'fallback')),
    policy_name text,
    model text,
    confidence real,
    first_message_hash text,
    api_key_alias text,
    latency_ms integer,
    shadow boolean NOT NULL DEFAULT false,
    agent_id text,
    detail jsonb NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS router_decisions_session_ts_idx ON router_decisions (session_key, ts);
CREATE INDEX IF NOT EXISTS router_decisions_ts_idx ON router_decisions (ts);
