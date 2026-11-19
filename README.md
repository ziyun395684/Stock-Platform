# StockRNN Research Platform

一个面向量化研究的本地平台：通过中文自然语言输入研究目标，由受治理的 Agent 生成研究计划，再通过 FastAPI、Celery、Redis 和 MySQL 管理实验、运行、artifact 与报告。

本项目只做研究编排、回测、模型训练、纸交易和结果分析，**不连接券商，不自动下单，不进行实盘交易**。

## 项目结构

```text
research_platform/      # FastAPI、Celery Worker、Vue 前端、迁移、测试
data/                   # 数据处理源码，不包含行情文件
model_and_train/        # 训练和回测源码，不包含 checkpoint
research/               # 研究任务和报告源码，不包含历史输出
config.yaml             # 引擎配置模板
requirements-engine.txt # StockRNN 引擎依赖
```

平台 Worker 会调用 `data`、`model_and_train` 和 `research` 中的源码。行情数据、模型文件和实验输出不随仓库提交，需要在本地准备或通过挂载目录提供：

```text
quotes2/                         # 本地历史行情目录，可通过任务 config.data_dir 指定
output/platform/                 # 平台生成的 artifact 和运行输出
model_and_train/checkpoints/     # 本地模型文件，可选
```

## 快速启动

推荐使用 conda 的 Python 3.11 环境：

```powershell
conda create -n p311 python=3.11 -y
conda activate p311
pip install -r research_platform/backend/requirements.txt
```

如果需要运行 Transformer 训练或完整的原始 StockRNN 引擎，再安装：

```powershell
pip install -r requirements-engine.txt
```

复制 `.env.example` 为 `.env`，配置 MySQL 和 Redis。然后启动：

```powershell
python -m uvicorn research_platform.backend.app.main:app --reload --port 8000
celery -A research_platform.workers.celery_app:celery_app worker --loglevel=INFO --pool=solo
cd research_platform/frontend
pnpm install
pnpm dev
```

访问：

- 前端：http://localhost:5173
- API：http://localhost:8000
- OpenAPI：http://localhost:8000/docs

## Docker 启动

MySQL 和 Redis 默认使用宿主机服务。先复制 `research_platform/.env.docker.example` 为 `research_platform/.env.docker` 并填写数据库密码，然后从仓库根目录执行：

```powershell
docker compose --env-file research_platform/.env.docker `
  -f research_platform/docker/docker-compose.platform.yml up --build -d
```

## Agent 研究流程

1. 用户输入中文研究目标。
2. 规划器生成白名单任务，例如因子质量、相关性、训练、样本外验证和成本敏感性。
3. 用户校验并确认预算和计划。
4. 平台创建 experiment/run/agent_task。
5. Celery Worker 执行注册过的研究任务。
6. Agent 汇总状态和 artifact 指标，生成反馈、下一轮待确认计划和中文 Markdown 报告。

当前架构是“一个受治理 Agent 编排器 + 多个专业 Worker”，不是多个可自主协商的数字员工系统。Agent 不执行任意 Python，不直接写数据库，也不具有券商或下单工具。

## 主要接口

- `GET /api/dependencies`
- `GET/POST /api/experiments`
- `POST /api/experiments/{experiment_id}/runs`
- `GET /api/jobs`
- `POST /api/agents/plan/draft`
- `POST /api/agents/plan/validate`
- `POST /api/agents/plan/execute`
- `GET /api/agents/sessions/{session_id}/state`
- `GET /api/agents/sessions/{session_id}/feedback`
- `POST /api/agents/sessions/{session_id}/next-plan`
- `GET /api/agents/sessions/{session_id}/report/markdown`
- `GET /api/experiments/compare?ids=id1,id2`

## 测试与构建

```powershell
conda run --no-capture-output -n p311 python -m pytest research_platform/tests -q
cd research_platform/frontend
pnpm build
```

## GitHub 上传前检查

不要提交 `.env`、数据库密码、`quotes2/` 行情数据、`output/` 实验产物、模型文件、`node_modules/` 和 Python 缓存。

```powershell
git status --short
git add .
git diff --cached --stat
git commit -m "chore: publish StockRNN research platform"
git branch -M main
git remote add origin <你的 GitHub 仓库地址>
git push -u origin main
```

当前未附带许可证。公开到 GitHub 前，请根据你的数据、模型和原始 StockRNN 代码授权情况补充合适的 `LICENSE` 文件。

