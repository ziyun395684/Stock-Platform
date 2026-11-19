-- Strategy lineage and governance follow-up migration.
ALTER TABLE strategy_versions ADD COLUMN parent_version_id CHAR(36) NULL;
ALTER TABLE agent_sessions ADD COLUMN plan_json JSON NULL;
ALTER TABLE approvals ADD COLUMN report_artifact_ids JSON NULL;
ALTER TABLE strategy_versions ADD KEY idx_strategy_versions_parent (parent_version_id);
ALTER TABLE strategy_versions ADD CONSTRAINT fk_strategy_versions_parent FOREIGN KEY (parent_version_id) REFERENCES strategy_versions(id);
INSERT INTO platform_schema_version(version) VALUES ('003_strategy_lineage') ON DUPLICATE KEY UPDATE version=VALUES(version);
