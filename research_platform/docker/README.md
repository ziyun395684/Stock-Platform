# Platform Docker notes

`docker-compose.platform.yml` is the runnable API/Worker compose file. It does
not create a second MySQL or Redis service; both are existing host services.

1. Copy `research_platform/.env.docker.example` to `research_platform/.env.docker`.
2. Replace the database password locally; keep the file uncommitted.
3. From the repository root run:

```powershell
docker compose --env-file research_platform/.env.docker -f research_platform/docker/docker-compose.platform.yml up --build
```

The containers use `host.docker.internal` to reach the existing MySQL and Redis
ports. The API is exposed at `http://127.0.0.1:8000`; the Worker consumes the
same Redis queue; and the Vue frontend is exposed at `http://127.0.0.1:5173`.
Nginx proxies frontend `/api/*` requests to FastAPI. The compose file
intentionally does not run migrations automatically; review and apply
`db/migrations/001_initial.sql` first.

The frontend MVP includes the research dashboard, strategy center, experiment
center, API health state, and MySQL/Redis dependency state. Quantitative logic
remains in the backend and Worker rather than running in the browser.

`docker-compose.platform.example.yml` remains a reference variant for custom
Docker networks.
