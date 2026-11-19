from pathlib import Path

from sqlalchemy import create_engine

from research_platform.backend.app import database
from research_platform.backend.app.artifact_repository import record_artifact
from research_platform.backend.app.job_repository import get_job, list_jobs
from research_platform.backend.app.models import Base
from research_platform.backend.app.report_repository import list_result_artifacts
from research_platform.backend.app.repositories import create_experiment
from research_platform.backend.app.run_repository import create_run, get_run, update_run


def test_experiment_run_job_and_artifact_index_round_trip(tmp_path: Path) -> None:
    previous = database._engine
    engine = create_engine(f"sqlite:///{tmp_path / 'platform.db'}", future=True)
    database._engine = engine
    try:
        Base.metadata.create_all(engine)
        experiment = create_experiment({
            "experiment_id": "sqlite-integration",
            "objective": "repository round trip",
            "config": {"selection_split": "validation"},
        })
        run = create_run(experiment["id"], "sqlite-integration:run-1", {"seed": 42, "window_name": "500d"})
        artifact_id = record_artifact(run["id"], "report", "output/platform/report.json", "a" * 64, 12)
        update_run(run["id"], "succeeded")

        loaded = get_run(run["id"])
        assert loaded is not None
        assert loaded["status"] == "succeeded"
        assert loaded["seed"] == 42
        assert loaded["artifacts"][0]["id"] == artifact_id

        job = list_jobs()
        assert len(job) == 1
        assert job[0]["run_status"] == "succeeded"
        assert get_job(job[0]["id"])["artifacts"][0]["artifact_type"] == "report"

        reports = list_result_artifacts(experiment["id"])
        assert reports[0]["id"] == artifact_id
        assert reports[0]["run_key"] == "sqlite-integration:run-1"
    finally:
        database._engine = previous
        engine.dispose()
