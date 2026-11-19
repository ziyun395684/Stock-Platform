import pandas as pd

from research_platform.backend.app.regression_contract import compare_backtest_reports, compare_factor_columns


def test_factor_regression_ignores_row_order_and_checks_values() -> None:
    expected = pd.DataFrame({"trade_date": [2, 1], "symbol": ["B", "A"], "roc_5": [0.2, 0.1]})
    actual = pd.DataFrame({"trade_date": [1, 2], "symbol": ["A", "B"], "roc_5": [0.1, 0.2]})
    assert compare_factor_columns(expected, actual, ["roc_5"])["equal"] is True
    actual.loc[1, "roc_5"] = 0.3
    result = compare_factor_columns(expected, actual, ["roc_5"])
    assert result["equal"] is False
    assert result["reason"] == "value_mismatch"


def test_backtest_regression_detects_metric_drift() -> None:
    expected = pd.DataFrame({"factor_mode": ["common"], "top_k": [1], "cost_rate": [0.003], "total_return_net": [0.12], "max_drawdown": [0.05]})
    actual = expected.copy()
    assert compare_backtest_reports(expected, actual)["equal"] is True
    actual.loc[0, "total_return_net"] = 0.13
    assert compare_backtest_reports(expected, actual)["reason"] == "value_mismatch"

