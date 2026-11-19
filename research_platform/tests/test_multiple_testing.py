from research_platform.backend.app.multiple_testing import benjamini_hochberg, normal_two_sided_p


def test_benjamini_hochberg_preserves_order_and_bounds() -> None:
    values = benjamini_hochberg([0.001, 0.04, 0.8, 0.02])
    assert len(values) == 4
    assert all(0 <= value <= 1 for value in values)
    assert values[0] <= values[2]


def test_normal_rank_ic_p_value_handles_invalid_and_signal_values() -> None:
    assert normal_two_sided_p(0.1, 0.02, 20) < 0.001
    assert normal_two_sided_p(0.1, 0.0, 20) == 1.0
