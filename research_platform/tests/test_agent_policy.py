from research_platform.backend.app.agent_repository import is_tool_allowed


def test_agent_allows_only_research_tools() -> None:
    assert is_tool_allowed("list_factors") is True
    assert is_tool_allowed("submit_run") is True
    assert is_tool_allowed("arbitrary_python") is False


def test_agent_rejects_order_and_broker_tools() -> None:
    for tool_name in ("broker.place_order", "order_submit", "trade.execute", "execute_order"):
        assert is_tool_allowed(tool_name) is False
