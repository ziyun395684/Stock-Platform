# StockRNN Research Platform

This package is the platform layer around the existing StockRNN research engine.
It will provide:

- FastAPI APIs;
- MySQL-backed strategy and experiment metadata;
- Redis/Celery asynchronous factor, training, backtest, and report jobs;
- Vue frontend integration;
- research agents that orchestrate deterministic tools.

The existing `data/`, `model_and_train/`, `research/`, `tests/`, and frozen strategy
assets remain the source of truth for quantitative computation. The platform must
not overwrite frozen strategies or connect to brokers in the MVP.

## Phase 0 smoke start

Install platform dependencies, then run from the repository root:

```powershell
python -m pip install -r research_platform/backend/requirements.txt
python -m uvicorn research_platform.backend.app.main:app --reload
```

Health endpoints:

- `GET /health`
- `GET /api/health`
- `GET /api/dependencies`
- `GET/POST /api/strategies`
- `GET /api/strategies/{strategy_id}`
- `POST /api/strategies/{strategy_id}/versions`
- `GET/POST /api/experiments`
- `POST /api/experiments/{experiment_id}/runs`
- `GET /api/experiments/{experiment_id}/runs/{run_id}`
  - supports `engine_probe`, `factor_baseline`, `factor_quality`, `factor_stratified_returns`, `factor_correlation`, `factor_decay`, `factor_regime`, `factor_common_train`, `transformer_train`, and `factor_common_oos`, `factor_sensitivity`, and `strategy_gate` task types;
  - returns MySQL run state and immutable artifact metadata.
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/reports` and `GET /api/reports/{artifact_id}` return read-only report artifact summaries with experiment/run filters;
- `GET /api/factors` returns factor artifact summaries with experiment, artifact type, and limit filters;
- `factor_compute` is the registered Worker adapter for common-factor panel generation and writes a factor-column manifest;
- `database_backup.py` builds password-free MySQL backup/restore commands and SHA-256/size manifests; command execution remains an explicitly approved operations step;
- `python -m research_platform.scripts.database_backup backup <file.sql>` defaults to dry-run; add `--execute` only during an approved operations window, and use `restore <file.sql>` for a checksum-validated restore drill;
- `GET /api/models` returns Transformer checkpoint/model metadata artifact summaries;
- `GET/POST /api/approvals` stores human approval decisions;
- `POST /api/agents/sessions` and `/api/agents/sessions/{session_id}/tasks` provide governed Agent planning metadata; `GET /api/agents/sessions`, `/tasks`, `/summary`, and `/next-steps` expose session task status, persisted plans, result counts, and retry/review suggestions; `PATCH /api/agents/sessions/{id}/status` supports draft/confirmed/running/completed/cancelled; strategy approvals are queryable at `GET /api/strategies/{strategy_id}/approvals`; `approved`/`freeze_requested` decisions gate MySQL strategy transitions, while Agent task status changes are audited through `agent-audit`; `ResearchGoal`/`ResearchPlan` schemas capture objective, baseline, constraints, budget, task list, estimated runs, and approval requirement; tools are restricted to factor/report/model/artifact reads plus experiment/run API names, and arbitrary code or direct DB writes are rejected; it never auto-freezes a strategy. Strategy transitions in MySQL write `audit_events` records; startup creates newly declared metadata tables when absent;
- `POST /api/agents/tasks/{task_id}/submit` maps an approved Agent task through the same task registry used by the run API and records the Celery submission audit event;
- `POST /api/agents/plan/execute` persists a confirmed research plan, creates one experiment/run per task, and queues each run through the governed Agent task bridge;
- `GET /api/agents/sessions/{session_id}/state` returns one polling payload with session status, task progress, failure details, and next-step suggestions;
- `GET /api/agents/sessions/{session_id}/feedback` reads completed run artifacts, aggregates return/drawdown/turnover metrics, and returns governed next-research suggestions;
- `POST /api/agents/sessions/{session_id}/next-plan` maps feedback actions to registered research tasks and returns a user-confirmed next `ResearchPlan`; it never auto-executes or accepts arbitrary code;
- `GET /api/experiments/compare?ids=id1,id2` aggregates completed run metrics from bounded JSON/CSV artifacts and returns return, drawdown, Sharpe, turnover, completion counts, and best-result pointers;
- `GET /api/agents/sessions/{session_id}/report/markdown` exports a Chinese read-only research report containing objective, run status, metrics, failures, suggestions, and the no-broker/no-live-trading boundary;
- The frontend Agent Console polls execution state, shows per-task progress and failures, and can export the current report as JSON;
- The frontend also exposes Phase 10 next-plan generation, Markdown report download, and real experiment comparison metrics;
- Completed Agent sessions show a read-only result-analysis panel; this platform does not place live orders or connect to brokers;
- Agent task submission is guarded by `tool_name=submit_run`, task status, and idempotent reuse of an existing queued Celery task ID;
- HTTP and validation failures use a stable `{error: {code, message, path, details}}` response shape;
  - supports `?status=queued|running|succeeded|failed|cancelled|retrying`;
  - returns progress, task ID, timestamps, and readable errors;
- Vue 任务中心已接入 `/api/jobs`，展示状态、进度、关联实验运行、错误信息和 artifact 数量；任务 API 同时返回 artifact 路径、SHA-256 和大小;
- `GET /api/experiments/{experiment_id}/runs` 返回实验运行列表和 artifacts，Vue 实验中心展示运行/产物数量，并可展开查看配置 JSON、运行状态和 artifact 类型；
- Vue 因子实验室汇总 Factor Quality、Stratified Returns、Correlation、Decay、Regime 任务及产物路径，支持按实验、ETF/symbol 和日期筛选；任务中心支持按状态筛选、运行中任务取消、失败任务重跑并每 15 秒自动刷新，已完成/失败任务会被终态保护；失败重跑创建新 run，不覆盖旧 artifact；任务列表展示 current_epoch/total_epochs（Worker 提供时）；任务报告接口 `/api/jobs/{job_id}/report` 提取 JSON 数值指标并展示 CSV 摘要，任务中心支持打开报告和下载 artifact；因子实验室支持按实验筛选；Vue 实验中心可提交 Factor Quality/Correlation/Decay/Regime/OOS 异步运行，支持复制实验和最多三项实验并排比较；策略中心支持状态筛选和 `POST /api/strategies/{strategy_id}/transition` 合法状态迁移；运行提交支持 `config.idempotency_key`，同一实验重复提交会返回已有 run；失败任务重跑 run 会返回 `retry_of` 来源；Celery 默认软超时 3300 秒、硬超时 3600 秒，并对 `frozen` 策略显示只读标识；点击策略 ID 可展开版本状态、父策略血缘、model/data/factor 摘要和配置 JSON；artifact 可通过 `/api/artifacts/{artifact_id}/download` 安全下载，也可通过 `/api/artifacts/{artifact_id}/preview` 查看最多 50 行的 UTF-8 CSV/JSON 摘要；CSV/JSON 同时返回结构化 columns/rows，前端以表格展示相关性矩阵和指标；预览中的数值列自动绘制 IC/Rank IC/衰减折线，矩阵单元格按正负相关性着色。
- `GET /api/runtime-config`

Documentation index: [`docs/README.md`](../docs/README.md). Architecture baseline: [`docs/adr/0001-platform-scope.md`](../docs/adr/0001-platform-scope.md). `migration_runner.apply_migrations()` applies ordered `db/migrations/*.sql` files once and records versions in `platform_schema_version`. Migration 002 adds `backtest_runs`, `metrics`, and result-query indexes; migration 003 adds strategy version lineage. Frozen strategies can be cloned through `POST /api/strategies/{strategy_id}/clone` into a research child. Confirmed Agent plans can be persisted to a session with `session_id`; the `agent_sessions.plan_json` field stores the plan. Agent plans can be budget-checked via `POST /api/agents/plan/validate`, confirmed via `POST /api/agents/plan/confirm`, or cancelled via `POST /api/agents/plan/cancel`; approvals support `target_type`, `target_id`, and `reviewer` filters; approval payloads can include `report_artifact_ids`, which are checked against the artifact table before persistence; over-budget plans are rejected. API startup registers schema version `003_strategy_lineage`.

The dependencies endpoint checks MySQL and Redis reachability with short
timeouts. The runtime-config endpoint exposes only safe diagnostics; it never
returns credentials.

## Current local status

- MySQL container: `mysql`, host port `3306`, database `stockrnn_platform`;
- migration `001_initial` has been applied and verified;
- local credentials are stored only in ignored `research_platform/.env.local`;
- Redis container: `stockrnn-redis`, host port `6379`, persistent volume
  `stockrnn_redis_data`;
- FastAPI dependency diagnostics report both MySQL and Redis reachable;
- Celery `smoke_task` was submitted through Redis and returned successfully;
- formal API/Worker Docker Compose deployment is running on port `8000`;
- Vue frontend is available on port `5173` when the frontend Compose service is started;
- containerized API health and dependency checks pass;
- containerized Celery `docker-smoke` task succeeded;
- containerized API created and read back a strategy through MySQL;
- `engine_probe` was submitted through the API/Celery/Redis path and completed;
- latest `experiment_runs` and `training_jobs` rows are both `succeeded` with progress `100`;
- a short-window Factor Baseline run produced CSV/JSON artifacts under
  `output/platform/runs/efd841fc-9138-4d6a-a109-28f17e7bf3d6/` with SHA-256 metadata;
- the rebuilt container Worker includes numpy, pandas, PyYAML, and scikit-learn;
- containerized Factor Baseline validation succeeded under
  `output/platform/runs/f112e24a-8d67-4bcb-893a-5556c34e1bed/`;
- containerized Factor Quality Rank IC validation succeeded under
  `output/platform/runs/5d0c19ad-e7ae-4192-83a0-68dde8ead6d0/`;
- containerized factor Spearman-correlation validation succeeded under
  `output/platform/runs/f2449fe6-e68e-4817-8068-784c1150aae7/`;
- containerized multi-horizon factor-decay validation succeeded under
  `output/platform/runs/43efda7f-4cb1-48ee-9a8f-821e4e0da6fc/`;
- containerized market-regime factor validation succeeded under
  `output/platform/runs/5fcd822a-2725-495c-aa2d-562120d5fe98/`;
- containerized Factor-common deterministic training/evaluation succeeded under
  `output/platform/runs/e37db10b-527f-449f-a0b8-222b88ec815f/`.
  The `transformer_train` adapter invokes the existing StockRNN time-split LSTM-Transformer rolling engine, indexes its CSV report, training log, metadata, generated `.keras` checkpoints, predictions, validation predictions, rolling config, and any engine-emitted transaction/NAV files. The metadata JSON also summarizes mean net return, drawdown, Rank IC, Sharpe, turnover, and trade count when present; the report API recursively exposes nested numeric metrics. It validates `training_windows` and `seeds` before enqueueing.

The deterministic Factor-common adapter records `checkpoint: null` rather than fabricating a neural checkpoint, validates `training_windows` (500/750/1000) and `seeds` (7/42/123), and writes separate training-log and model-metadata artifacts; the Vue run form exposes `training_windows` and `seeds` inputs and includes them in submitted config;
- containerized non-overlapping Factor-common OOS validation succeeded under
  `output/platform/runs/3809c67b-46cd-4868-9261-0ea6d7d4a2ae/`;
- containerized Factor-common Top-K/cost sensitivity validation succeeded under
  `output/platform/runs/a22a3664-8004-4f8d-aa3d-b3bf520c0e75/`;
- containerized strategy-gate validation succeeded under
  `output/platform/runs/5558537a-b492-4a66-850f-c91df53a6bfe/`; the frozen baseline
  `long_term_v1_quotes2` remained read-only and no automatic promotion occurred.


Database migration: review and run `db/migrations/001_initial.sql` against the
platform database only. The migration creates metadata tables and never moves
or changes existing StockRNN data, models, or frozen strategies.
