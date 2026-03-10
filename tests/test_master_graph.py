import pytest
from unittest.mock import AsyncMock, patch

from agent_chatroom.master_graph import (
    node_open_next_subtopic,
    node_plan_generation,
    node_topic_replan_or_close,
    route_after_replan,
    route_after_open_next_subtopic,
)


@pytest.mark.asyncio
async def test_node_plan_generation():
    state = {"topic_id": 1}

    mock_gemini = AsyncMock(
        return_value={
            "action": "create_plan",
            "subtopics": [
                {"summary": "Subtopic 1", "detail": "Detail 1"},
                {"summary": "Subtopic 2", "detail": "Detail 2"},
            ],
        }
    )

    with patch("agent_chatroom.master_graph.ask_gemini_cli", new=mock_gemini):
        with patch("agent_chatroom.master_graph.api.create_plan", return_value=7):
            with patch(
                "agent_chatroom.master_graph.api.get_topic",
                return_value={"id": 1, "summary": "Topic Summary", "detail": "Topic Detail"},
            ):
                new_state = await node_plan_generation(state)

    assert new_state["plan_id"] == 7
    assert new_state["next_action"] == "open_next_subtopic"


@pytest.mark.asyncio
async def test_node_open_next_subtopic():
    state = {"topic_id": 1, "plan_id": 7}
    plan = {
        "id": 7,
        "current_index": 0,
        "content": '[{"summary": "Subtopic 1", "detail": "Detail 1"}]',
    }

    mock_gemini = AsyncMock(
        return_value={
            "action": "post_message",
            "content": "This is the grounding brief.",
        }
    )

    with patch(
        "agent_chatroom.master_graph.api.get_topic",
        return_value={"id": 1, "summary": "Topic Summary", "detail": "Topic Detail"},
    ):
        with patch("agent_chatroom.master_graph.api.get_active_plan", return_value=plan):
            with patch("agent_chatroom.master_graph.ask_gemini_cli", new=mock_gemini):
                with patch("agent_chatroom.master_graph.api.create_subtopic", return_value=10):
                    with patch("agent_chatroom.master_graph.api.persist_message", new=AsyncMock(return_value=99)):
                        with patch("agent_chatroom.master_graph.api.update_subtopic_start_msg"):
                            with patch("agent_chatroom.master_graph.api.advance_plan_cursor"):
                                with patch("agent_chatroom.master_graph.api.set_topic_status"):
                                    new_state = await node_open_next_subtopic(state)

    assert new_state["current_subtopic_id"] == 10
    assert new_state["plan_id"] == 7
    assert new_state["next_action"] == "run_subtopic"


@pytest.mark.asyncio
async def test_node_topic_replan_or_close_defers_on_error():
    state = {"topic_id": 1}

    with patch(
        "agent_chatroom.master_graph.api.get_topic",
        return_value={"id": 1, "summary": "Topic Summary", "detail": "Topic Detail"},
    ):
        with patch(
            "agent_chatroom.master_graph.api.get_current_subtopics",
            return_value=[{"summary": "Subtopic 1", "conclusion": "Conclusion 1"}],
        ):
            with patch(
                "agent_chatroom.master_graph.ask_gemini_cli",
                new=AsyncMock(return_value={"error": "timeout"}),
            ):
                with patch("agent_chatroom.master_graph.api.post_message") as post_message:
                    with patch("agent_chatroom.master_graph.api.set_topic_status") as set_status:
                        result = await node_topic_replan_or_close(state)

    assert result["deferred"] is True
    assert result["topic_complete"] is False
    assert result["next_action"] == "defer_topic"
    assert route_after_replan(result) == "defer_topic"
    post_message.assert_not_called()
    set_status.assert_not_called()


def test_route_after_open_next_subtopic_replans_when_no_subtopic_was_opened():
    assert route_after_open_next_subtopic({"next_action": "replan_or_close"}) == "replan_or_close"
    assert route_after_open_next_subtopic({"current_subtopic_id": 0}) == "replan_or_close"
    assert route_after_open_next_subtopic({"current_subtopic_id": 10, "next_action": "run_subtopic"}) == "run_subtopic"
