CREATE TABLE IF NOT EXISTS patrol_runs (
    id          SERIAL PRIMARY KEY,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at    TIMESTAMPTZ,
    robot       TEXT NOT NULL DEFAULT 'B2',
    source      TEXT NOT NULL,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id              BIGSERIAL PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES patrol_runs(id),
    event_no        INTEGER NOT NULL,
    track_id        TEXT NOT NULL,
    detected_at     TIMESTAMPTZ NOT NULL,
    stored_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    x_m             REAL,
    y_m             REAL,
    distance_m      REAL,
    bearing_deg     REAL,
    position_status TEXT NOT NULL DEFAULT 'ok' CHECK (position_status IN ('ok', 'no_robot_pose', 'missing')),
    n_points        INTEGER,
    robot_x         REAL, robot_y REAL, robot_yaw REAL,
    source          TEXT NOT NULL DEFAULT 'radar',
    status          TEXT NOT NULL DEFAULT 'new' CHECK (status IN ('new', 'ack', 'false_alarm')),
    handled_at      TIMESTAMPTZ,
    UNIQUE (run_id, event_no)
);

CREATE TABLE IF NOT EXISTS operator_actions (
    id        BIGSERIAL PRIMARY KEY,
    alert_id  BIGINT NOT NULL REFERENCES alerts(id),
    action    TEXT NOT NULL CHECK (action IN ('ack', 'false_alarm')),
    acted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    operator  TEXT
);

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = current_schema() AND table_name = 'alerts' AND column_name = 'stored_at') THEN
        ALTER TABLE alerts ADD COLUMN stored_at TIMESTAMPTZ;
        UPDATE alerts SET stored_at = detected_at;
        ALTER TABLE alerts ALTER COLUMN stored_at SET DEFAULT now(), ALTER COLUMN stored_at SET NOT NULL;
    END IF;
END $$;
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS position_status TEXT NOT NULL DEFAULT 'ok';
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'radar';
ALTER TABLE alerts ALTER COLUMN x_m DROP NOT NULL;
ALTER TABLE alerts ALTER COLUMN y_m DROP NOT NULL;
ALTER TABLE alerts ALTER COLUMN distance_m DROP NOT NULL;
ALTER TABLE alerts ALTER COLUMN bearing_deg DROP NOT NULL;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'alerts_position_status_check'
                   AND conrelid = 'alerts'::regclass) THEN
        ALTER TABLE alerts ADD CONSTRAINT alerts_position_status_check
            CHECK (position_status IN ('ok', 'no_robot_pose', 'missing'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS alerts_detected_at ON alerts (detected_at DESC);
CREATE INDEX IF NOT EXISTS alerts_run_status ON alerts (run_id, status);
