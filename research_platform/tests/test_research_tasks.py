from __future__ import annotations

from research_platform.workers.celery_app import celery_app
from research_platform.backend.app.task_registry import celery_task_names, registered_task_names, task_for


def test_api_task_registry_matches_supported_celery_tasks() -> None:
    expected = {"engine_probe", "factor_compute", "factor_baseline", "factor_quality", "factor_stratified_returns", "factor_correlation", "factor_decay", "factor_regime", "factor_common_train", "transformer_train", "factor_common_oos", "factor_sensitivity", "strategy_gate", "signal_snapshot", "paper_portfolio"}
    assert registered_task_names() == expected
    assert expected.issubset(celery_task_names())
    assert task_for("factor_common_oos").name == "research_platform.factor_common_oos"


def test_api_task_registry_rejects_unknown_task() -> None:
    import pytest
    with pytest.raises(ValueError, match="unsupported research task type"):
        task_for("unknown")


def test_backtest_engine_contract_accepts_required_finite_output() -> None:
    import pandas as pd
    from research_platform.backend.app.engine_contract import validate_backtest_result

    validate_backtest_result(pd.DataFrame({"total_return_net": [0.1], "max_drawdown": [0.05]}))


def test_backtest_engine_contract_rejects_missing_empty_and_nonfinite_output() -> None:
    import pandas as pd
    import pytest
    from research_platform.backend.app.engine_contract import validate_backtest_result

    with pytest.raises(ValueError, match="缺少必要字段"):
        validate_backtest_result(pd.DataFrame({"total_return_net": [0.1]}))
    with pytest.raises(ValueError, match="为空"):
        validate_backtest_result(pd.DataFrame({"total_return_net": [], "max_drawdown": []}))
    with pytest.raises(ValueError, match="非有限值"):
        validate_backtest_result(pd.DataFrame({"total_return_net": [float("inf")], "max_drawdown": [0.1]}))


def test_engine_probe_is_registered() -> None:
    assert "research_platform.engine_probe" in celery_app.tasks


def test_existing_engine_contract_is_importable() -> None:
    from model_and_train.factor_baseline import FACTOR_MODES
    from model_and_train.portfolio_backtest import PortfolioBacktest
    from research.factor_baseline.run_factor_baselines import run

    assert {"common", "roc", "trend", "ma_energy"}.issubset(FACTOR_MODES)
    assert PortfolioBacktest.__name__ == "PortfolioBacktest"
    assert run.__name__ == "run"
