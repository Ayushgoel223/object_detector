-- ============================================================
-- BlindAid v2 — MySQL Schema
-- Run once: mysql -u root -p blindaid < schema.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS blindaid
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE blindaid;

-- ── Session Log ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS session_log (
    id            BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_id    VARCHAR(64) NOT NULL UNIQUE,
    start_time    DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    end_time      DATETIME(3),
    total_frames  INT UNSIGNED DEFAULT 0,
    cam_fps_avg   FLOAT DEFAULT 0,
    inf_fps_avg   FLOAT DEFAULT 0,
    INDEX idx_session_start (session_id, start_time)
) ENGINE=InnoDB;

-- ── Detection Events ──────────────────────────────────────
-- Only significant events stored (not every frame)
CREATE TABLE IF NOT EXISTS detection_events (
    id            BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_id    VARCHAR(64) NOT NULL,
    timestamp     DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    frame_id      BIGINT UNSIGNED NOT NULL,
    label         VARCHAR(64) NOT NULL,
    confidence    FLOAT NOT NULL,
    zone          ENUM('left','ahead','right') NOT NULL,
    urgency       ENUM('SAFE','FAR','NEAR','CRITICAL') NOT NULL,
    bbox_json     JSON,                          -- {x1,y1,x2,y2,cx,cy,area_fraction}
    INDEX idx_det_session  (session_id, timestamp),
    INDEX idx_det_label    (label),
    INDEX idx_det_urgency  (urgency),
    FOREIGN KEY (session_id) REFERENCES session_log(session_id)
        ON DELETE CASCADE
) ENGINE=InnoDB;

-- ── OCR Events ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ocr_events (
    id                  BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_id          VARCHAR(64) NOT NULL,
    timestamp           DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    frame_id            BIGINT UNSIGNED NOT NULL,
    raw_text            TEXT NOT NULL,
    cleaned_text        VARCHAR(512),
    confidence          FLOAT NOT NULL,
    zone                ENUM('left','center','right') NOT NULL,
    semantic_tag        VARCHAR(64),             -- 'EXIT','STAIRS','DANGER' etc.
    word_dynamics_json  JSON,                    -- per-char confidence array
    INDEX idx_ocr_session  (session_id, timestamp),
    INDEX idx_ocr_tag      (semantic_tag),
    FOREIGN KEY (session_id) REFERENCES session_log(session_id)
        ON DELETE CASCADE
) ENGINE=InnoDB;

-- ── Training Samples ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS training_samples (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    image_path      VARCHAR(512) NOT NULL,
    label           VARCHAR(64),                 -- NULL for unsupervised
    annotation_json JSON,                        -- YOLO bbox format
    source          VARCHAR(128) NOT NULL,        -- 'youtube','roboflow','openimages','synthetic'
    split           ENUM('train','val','test') DEFAULT 'train',
    phase           ENUM('supervised','unsupervised','rl') DEFAULT 'supervised',
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_ts_source (source),
    INDEX idx_ts_split  (split),
    INDEX idx_ts_phase  (phase)
) ENGINE=InnoDB;

-- ── Training Runs ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS training_runs (
    id          BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    run_id      VARCHAR(64) NOT NULL,
    phase       ENUM('supervised','unsupervised','rl') NOT NULL,
    epoch       INT UNSIGNED NOT NULL,
    map50       FLOAT,
    map50_95    FLOAT,
    precision   FLOAT,
    recall      FLOAT,
    loss        FLOAT,
    notes       TEXT,
    timestamp   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_run_id    (run_id),
    INDEX idx_run_phase (phase, epoch)
) ENGINE=InnoDB;

-- ── RL Episodes ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rl_episodes (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    episode         INT UNSIGNED NOT NULL,
    step            INT UNSIGNED NOT NULL,
    state_json      JSON NOT NULL,               -- 256-dim encoded state
    action          TINYINT UNSIGNED NOT NULL,   -- 0-5 action index
    reward          FLOAT NOT NULL,
    next_state_json JSON NOT NULL,
    done            TINYINT(1) NOT NULL DEFAULT 0,
    q_values_json   JSON,                        -- debug: Q(s,a) for all actions
    timestamp       DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    INDEX idx_rl_episode (episode, step)
) ENGINE=InnoDB;

-- ── Cluster Centroids (Phase 2 Unsupervised) ─────────────
CREATE TABLE IF NOT EXISTS cluster_centroids (
    id              INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    cluster_id      INT UNSIGNED NOT NULL,
    label_hint      VARCHAR(64),                 -- inferred label if known
    centroid_json   JSON NOT NULL,               -- 512-dim float array
    member_count    INT UNSIGNED DEFAULT 0,
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_cc_cluster (cluster_id)
) ENGINE=InnoDB;
