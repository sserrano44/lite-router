-- Raw first user messages: the future training set (R20).
-- Restricted access; 90-day retention enforced by scripts/retention_purge.py.
CREATE TABLE IF NOT EXISTS router_first_messages (
    first_message_hash text PRIMARY KEY,
    session_key text NOT NULL,
    ts timestamptz NOT NULL DEFAULT now(),
    raw_message text NOT NULL,
    system_excerpt text
);

CREATE INDEX IF NOT EXISTS router_first_messages_ts_idx ON router_first_messages (ts);

REVOKE ALL ON router_first_messages FROM PUBLIC;
