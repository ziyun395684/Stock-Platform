from research_platform.backend.app.candidate_report import summarize_observations


def test_candidate_report_keeps_all_dimensions_and_rejects_best_window_only() -> None:
    observations = [
        {"window": 500, "seed": 7, "cost_rate": 0.003, "regime": "bull", "total_return_net": 0.10},
        {"window": 750, "seed": 42, "cost_rate": 0.003, "regime": "bull", "total_return_net": -0.02},
    ]
    report = summarize_observations(observations, total_runs=2)
    assert report["coverage"]["windows"] == ["500", "750"]
    assert report["coverage"]["seeds"] == ["42", "7"]
    assert report["overall"]["count"] == 2
    assert report["selection_policy"].startswith("aggregate all")
    assert report["recommendation"]["status"] == "insufficient_evidence"


def test_candidate_report_requires_multiple_costs_and_regimes() -> None:
    observations = [
        {"window": 500, "seed": 7, "cost_rate": 0.003, "regime": "bull", "total_return_net": 0.10},
        {"window": 750, "seed": 42, "cost_rate": 0.005, "regime": "bear", "total_return_net": 0.02},
    ]
    report = summarize_observations(observations, total_runs=2)
    assert report["recommendation"]["eligible"] is True
    assert abs(report["overall"]["mean"] - 0.06) < 1e-12
