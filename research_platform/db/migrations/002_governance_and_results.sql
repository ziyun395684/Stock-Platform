-- Research platform follow-up metadata tables and indexes.
CREATE TABLE IF NOT EXISTS backtest_runs (
    id CHAR(36) NOT NULL PRIMARY KEY,
    experiment_run_id CHAR(36) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'queued',
    config_json JSON NOT NULL,
    started_at TIMESTAMP(6) NULL,
    finished_at TIMESTAMP(6) NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    KEY idx_backtest_run_status_created (status, created_at),
    CONSTRAINT fk_backtest_runs_experiment_run FOREIGN KEY (experiment_run_id) REFERENCES experiment_runs(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS metrics (
    id CHAR(36) NOT NULL PRIMARY KEY,
    experiment_run_id CHAR(36) NOT NULL,
    metric_name VARCHAR(128) NOT NULL,
    metric_value DECIMAL(20,8) NULL,
    split VARCHAR(32) NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    KEY idx_metrics_run_name (experiment_run_id, metric_name),
    KEY idx_metrics_split_created (split, created_at),
    CONSTRAINT fk_metrics_experiment_run FOREIGN KEY (experiment_run_id) REFERENCES experiment_runs(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE INDEX idx_runs_status_created ON experiment_runs(status, created_at);
CREATE INDEX idx_training_jobs_status_updated ON training_jobs(status, updated_at);
CREATE INDEX idx_artifacts_type_created ON artifacts(artifact_type, created_at);
