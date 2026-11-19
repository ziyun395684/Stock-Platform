from research_platform.backend.app.factor_validation import validate_factor_definition
from research_platform.backend.app.schemas import FactorDefinition, FactorVersionDefinition


def test_factor_definition_requires_research_metadata() -> None:
    factor = FactorDefinition(
        factor_id="momentum_20",
        name="20-day momentum",
        economic_hypothesis="Recent strength persists over the short horizon.",
        version=FactorVersionDefinition(
            version="v1",
            formula="close / close.shift(20) - 1",
            input_columns=["close"],
            time_boundary="available through t close",
        ),
    )
    assert validate_factor_definition(factor) == []


def test_factor_definition_rejects_future_data_markers() -> None:
    factor = FactorDefinition(
        factor_id="leaky",
        name="Leaky factor",
        economic_hypothesis="invalid",
        version=FactorVersionDefinition(
            version="v1",
            formula="next_close / close - 1",
            input_columns=["close"],
            time_boundary="t close",
        ),
    )
    assert validate_factor_definition(factor)
