-- Client label (claude-code / opencode / generic) for each decision, so
-- sessions from different agents/CLIs are distinguishable in analytics.
ALTER TABLE router_decisions ADD COLUMN IF NOT EXISTS client text;

CREATE INDEX IF NOT EXISTS router_decisions_client_idx ON router_decisions (client);
