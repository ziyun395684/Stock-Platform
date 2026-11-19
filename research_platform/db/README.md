# Platform database

This directory will contain versioned MySQL migrations and seed data.

Rules:

- Use the existing Docker MySQL service; do not create a second database in the MVP.
- Store only platform metadata here: strategies, factors, experiments, jobs, metrics, artifacts, approvals, and audit events.
- Keep model files, predictions, reports, and checkpoints under `output/platform/` or object storage; store their paths and SHA-256 in MySQL.
- Never commit credentials or production data.
