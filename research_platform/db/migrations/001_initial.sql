-- StockRNN Research Platform initial metadata schema.
-- Target: MySQL 8.0+; this migration stores metadata only.
-- Do not run against the frozen strategy files or model artifacts.

CREATE TABLE IF NOT EXISTS platform_schema_version (
    version VARCHAR(32) NOT NULL PRIMARY KEY,
    applied_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS strategies (
    id CHAR(36) NOT NULL PRIMARY KEY,
    strategy_id VARCHAR(128) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    status ENUM('draft','research','training','backtested','candidate','approved','frozen','retired') NOT NULL DEFAULT 'draft',
    parent_strategy_id CHAR(36) NULL,
    description TEXT NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    CONSTRAINT fk_strategies_parent FOREIGN KEY (parent_strategy_id) REFERENCES strategies(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS strategy_versions (
    id CHAR(36) NOT NULL PRIMARY KEY,
    strategy_id CHAR(36) NOT NULL,
    version VARCHAR(64) NOT NULL,
    status ENUM('draft','research','training','backtested','candidate','approved','frozen','retired') NOT NULL DEFAULT 'draft',
    config_json JSON NOT NULL,
    data_version VARCHAR(255) NULL,
    factor_version_ids JSON NULL,
    model_version VARCHAR(255) NULL,
    artifact_manifest JSON NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    UNIQUE KEY uq_strategy_version (strategy_id, version),
    CONSTRAINT fk_strategy_versions_strategy FOREIGN KEY (strategy_id) REFERENCES strategies(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS factors (
    id CHAR(36) NOT NULL PRIMARY KEY,
    factor_id VARCHAR(128) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    status ENUM('draft','research','candidate','approved','retired') NOT NULL DEFAULT 'draft',
    economic_hypothesis TEXT NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS factor_versions (
    id CHAR(36) NOT NULL PRIMARY KEY,
    factor_id CHAR(36) NOT NULL,
    version VARCHAR(64) NOT NULL,
    formula TEXT NOT NULL,
    input_columns JSON NOT NULL,
    parameters_json JSON NULL,
    time_boundary VARCHAR(255) NULL,
    code_artifact_id CHAR(36) NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    UNIQUE KEY uq_factor_version (factor_id, version),
    CONSTRAINT fk_factor_versions_factor FOREIGN KEY (factor_id) REFERENCES factors(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS experiments (
    id CHAR(36) NOT NULL PRIMARY KEY,
    experiment_id VARCHAR(128) NOT NULL UNIQUE,
    strategy_version_id CHAR(36) NULL,
    objective TEXT NOT NULL,
    config_json JSON NOT NULL,
    status ENUM('draft','queued','running','succeeded','failed','cancelled') NOT NULL DEFAULT 'draft',
    created_by VARCHAR(128) NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    CONSTRAINT fk_experiments_strategy_version FOREIGN KEY (strategy_version_id) REFERENCES strategy_versions(id),
    KEY idx_experiments_status_created (status, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS experiment_runs (
    id CHAR(36) NOT NULL PRIMARY KEY,
    experiment_id CHAR(36) NOT NULL,
    run_key VARCHAR(255) NOT NULL UNIQUE,
    status ENUM('queued','running','succeeded','failed','cancelled','retrying') NOT NULL DEFAULT 'queued',
    seed INT NULL,
    window_name VARCHAR(128) NULL,
    started_at TIMESTAMP(6) NULL,
    finished_at TIMESTAMP(6) NULL,
    error_type VARCHAR(255) NULL,
    error_message TEXT NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    CONSTRAINT fk_experiment_runs_experiment FOREIGN KEY (experiment_id) REFERENCES experiments(id),
    KEY idx_experiment_runs_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS training_jobs (
    id CHAR(36) NOT NULL PRIMARY KEY,
    experiment_run_id CHAR(36) NOT NULL,
    task_id VARCHAR(255) NULL UNIQUE,
    status ENUM('queued','running','succeeded','failed','cancelled','retrying') NOT NULL DEFAULT 'queued',
    progress DECIMAL(6,3) NOT NULL DEFAULT 0,
    current_epoch INT NULL,
    total_epochs INT NULL,
    log_artifact_id CHAR(36) NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    CONSTRAINT fk_training_jobs_run FOREIGN KEY (experiment_run_id) REFERENCES experiment_runs(id),
    KEY idx_training_jobs_status_updated (status, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS artifacts (
    id CHAR(36) NOT NULL PRIMARY KEY,
    artifact_type VARCHAR(64) NOT NULL,
    relative_path VARCHAR(1024) NOT NULL,
    sha256 CHAR(64) NULL,
    size_bytes BIGINT NULL,
    experiment_run_id CHAR(36) NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    CONSTRAINT fk_artifacts_run FOREIGN KEY (experiment_run_id) REFERENCES experiment_runs(id),
    KEY idx_artifacts_type_created (artifact_type, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS audit_events (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    actor_type VARCHAR(32) NOT NULL,
    actor_id VARCHAR(128) NULL,
    event_type VARCHAR(128) NOT NULL,
    entity_type VARCHAR(64) NULL,
    entity_id VARCHAR(128) NULL,
    payload_json JSON NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    KEY idx_audit_entity_created (entity_type, entity_id, created_at),
    KEY idx_audit_event_created (event_type, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT IGNORE INTO platform_schema_version(version) VALUES ('001_initial');
