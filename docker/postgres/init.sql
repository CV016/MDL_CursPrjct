-- =============================================================================
-- AI-DOC INTERACT — PostgreSQL initialisation script
-- This script runs automatically on first container boot via the
-- /docker-entrypoint-initdb.d/ mechanism.
-- It is idempotent: all statements use IF NOT EXISTS guards.
-- =============================================================================

-- Enable the pgcrypto extension for gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- users — application accounts managed by the auth service
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username      VARCHAR(64)  NOT NULL UNIQUE,
    email         VARCHAR(254) NOT NULL UNIQUE,
    -- bcrypt hash of the plaintext password (cost factor 12)
    password_hash VARCHAR(72)  NOT NULL,
    is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Partial index: fast lookups for active users only
CREATE INDEX IF NOT EXISTS idx_users_username_active
    ON users (username)
    WHERE is_active = TRUE;

-- ---------------------------------------------------------------------------
-- sessions — issued JWT sessions (for audit / revocation support)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    session_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID        NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    -- Stored JTI (JWT ID) claim — allows individual token revocation
    jti         VARCHAR(64) NOT NULL UNIQUE,
    issued_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ NOT NULL,
    revoked     BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_jti     ON sessions (jti);

-- ---------------------------------------------------------------------------
-- experiment_logs — records every inference request for MLflow correlation
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiment_logs (
    log_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          UUID        NOT NULL,
    user_id         UUID        REFERENCES users (user_id) ON DELETE SET NULL,
    -- A or B variant label from the gateway A/B router
    variant         CHAR(1)     NOT NULL CHECK (variant IN ('A', 'B')),
    service         VARCHAR(32) NOT NULL,  -- 'summarizer' | 'question_gen'
    -- MLflow run ID for this inference call
    mlflow_run_id   VARCHAR(64),
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    latency_ms      NUMERIC(10, 2),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_exp_logs_doc_id  ON experiment_logs (doc_id);
CREATE INDEX IF NOT EXISTS idx_exp_logs_variant ON experiment_logs (variant);

-- ---------------------------------------------------------------------------
-- feedback — thumbs-up / thumbs-down ratings from the frontend
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id      UUID        NOT NULL,
    user_id     UUID        REFERENCES users (user_id) ON DELETE SET NULL,
    variant     CHAR(1)     NOT NULL CHECK (variant IN ('A', 'B')),
    -- 1 = positive, -1 = negative
    score       SMALLINT    NOT NULL CHECK (score IN (-1, 1)),
    comment     TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_doc_id  ON feedback (doc_id);
CREATE INDEX IF NOT EXISTS idx_feedback_variant ON feedback (variant);
CREATE INDEX IF NOT EXISTS idx_feedback_score   ON feedback (score);

-- ---------------------------------------------------------------------------
-- Trigger: keep users.updated_at in sync automatically
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
    RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_users_updated_at ON users;
CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
