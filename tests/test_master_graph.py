import json
import pytest
from unittest.mock import AsyncMock, patch

from grox_chat.master_graph import (
    ask_gemini_cli,
    _collect_votes,
    node_inspect_topic_state,
    node_open_next_subtopic,
    node_plan_generation,
    node_topic_replan_or_close,
    route_after_generate_plan,
    route_after_replan,
    route_after_open_next_subtopic,
)


@pytest.mark.asyncio
async def test_ask_gemini_cli_defaults_to_flash_first():
    mock_result = AsyncMock()
    mock_result.text = '{"ok": true}'
    mock_result.provider_used = "gemini_flash"
    mock_result.fallback_used = False
    mock_result.search_used = False
    with patch("grox_chat.master_graph.llm_call_with_web", new=AsyncMock(return_value=mock_result)) as llm_call_with_web:
        result = await ask_gemini_cli("System", "Context", "skynet")

    assert result == {"ok": True}
    assert llm_call_with_web.await_args.kwargs["provider_profile"] == "gemini_flash"
    assert llm_call_with_web.await_args.kwargs["model"] == "gemini-3.0-flash"
    assert llm_call_with_web.await_args.kwargs["require_json"] is True
    assert llm_call_with_web.await_args.kwargs["search_budget"] == 2


@pytest.mark.asyncio
async def test_ask_gemini_cli_retries_invalid_json_before_succeeding():
    bad_result = AsyncMock()
    bad_result.text = "not json"
    bad_result.provider_used = "minimax"
    bad_result.fallback_used = True
    bad_result.search_used = False

    good_result = AsyncMock()
    good_result.text = '{"ok": true, "step": 2}'
    good_result.provider_used = "minimax"
    good_result.fallback_used = True
    good_result.search_used = False

    with patch("grox_chat.master_graph.llm_call_with_web", new=AsyncMock(side_effect=[bad_result, good_result])) as llm_call_with_web:
        result = await ask_gemini_cli("System", "Context", "skynet")

    assert result == {"ok": True, "step": 2}
    assert llm_call_with_web.await_count == 2


@pytest.mark.asyncio
async def test_node_plan_generation():
    state = {"topic_id": 1}

    with patch(
        "grox_chat.master_graph.api.get_topic",
        return_value={"id": 1, "summary": "Topic Summary", "detail": "Topic Detail"},
    ):
        with patch(
            "grox_chat.master_graph._propose_subtopics",
            new=AsyncMock(
                return_value={
                    "candidates": [
                        {"summary": "Subtopic 1", "detail": "Detail 1"},
                        {"summary": "Subtopic 2", "detail": "Detail 2"},
                    ],
                    "error": None,
                }
            ),
        ):
            with patch(
                "grox_chat.master_graph._collect_votes",
                new=AsyncMock(return_value={"yes_votes": 8, "successful_votes": 10, "failed_votes": 0}),
            ):
                with patch("grox_chat.master_graph.api.create_plan", return_value=7):
                    new_state = await node_plan_generation(state)

    assert new_state["plan_id"] == 7
    assert new_state["next_action"] == "open_next_subtopic"


@pytest.mark.asyncio
async def test_collect_votes_persists_reasoned_vote_records():
    fake_agent = AsyncMock()
    fake_agent.vote_detail = AsyncMock(
        side_effect=[
            {"decision": True, "decision_label": "yes", "reason": "non-redundant axis", "raw_response": '{"vote":"yes","reason":"non-redundant axis"}'},
            {"decision": False, "decision_label": "no", "reason": "too overlapping", "raw_response": '{"vote":"no","reason":"too overlapping"}'},
        ]
    )

    with patch("grox_chat.master_graph.voting_agents", return_value=["dreamer", "critic"]):
        with patch("grox_chat.master_graph.get_agent", return_value=fake_agent):
            with patch("grox_chat.master_graph.api.insert_vote_record") as insert_vote_record:
                tally = await _collect_votes(
                    "Vote prompt",
                    topic_id=11,
                    subtopic_id=None,
                    round_number=None,
                    vote_kind="candidate_admission",
                    subject="Subtopic A",
                )

    assert tally == {"yes_votes": 1, "successful_votes": 2, "failed_votes": 0}
    assert insert_vote_record.call_count == 2
    assert insert_vote_record.call_args_list[0].args[:10] == (
        11,
        None,
        None,
        "candidate_admission",
        "Subtopic A",
        "Vote prompt",
        "dreamer",
        True,
        "yes",
        "non-redundant axis",
    )


@pytest.mark.asyncio
async def test_node_plan_generation_truncates_to_four_subtopics():
    state = {"topic_id": 1}
    model_subtopics = [
        {"summary": f"Subtopic {idx}", "detail": f"Detail {idx}"}
        for idx in range(1, 6)
    ]

    with patch(
        "grox_chat.master_graph.api.get_topic",
        return_value={"id": 1, "summary": "Topic Summary", "detail": "Topic Detail"},
    ):
        with patch(
            "grox_chat.master_graph._propose_subtopics",
            new=AsyncMock(return_value={"candidates": model_subtopics, "error": None}),
        ):
            with patch(
                "grox_chat.master_graph._collect_votes",
                new=AsyncMock(return_value={"yes_votes": 8, "successful_votes": 10, "failed_votes": 0}),
            ):
                with patch("grox_chat.master_graph.api.create_plan", return_value=7) as create_plan:
                    new_state = await node_plan_generation(state)

    stored_subtopics = json.loads(create_plan.call_args.args[1])
    assert len(stored_subtopics) == 4
    assert new_state["plan_id"] == 7


@pytest.mark.asyncio
async def test_node_plan_generation_closes_after_three_empty_cycles():
    state = {"topic_id": 1}

    with patch(
        "grox_chat.master_graph.api.get_topic",
        return_value={"id": 1, "summary": "Topic Summary", "detail": "Topic Detail"},
    ):
        with patch(
            "grox_chat.master_graph._propose_subtopics",
            new=AsyncMock(
                side_effect=[
                    {"candidates": [{"summary": "A", "detail": "A"}], "error": None},
                    {"candidates": [{"summary": "B", "detail": "B"}], "error": None},
                    {"candidates": [{"summary": "C", "detail": "C"}], "error": None},
                ]
            ),
        ):
            with patch(
                "grox_chat.master_graph._collect_votes",
                new=AsyncMock(return_value={"yes_votes": 0, "successful_votes": 10, "failed_votes": 0}),
            ):
                with patch("grox_chat.master_graph.aget_embedding", new=AsyncMock(return_value=None)):
                    with patch("grox_chat.master_graph.api.set_topic_status") as set_status:
                        with patch("grox_chat.master_graph.api.post_message") as post_message:
                            result = await node_plan_generation(state)

    assert result["topic_complete"] is True
    assert result["next_action"] == "close_topic"
    set_status.assert_called_once_with(1, "Closed")
    post_message.assert_called_once()


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
        "grox_chat.master_graph.api.get_topic",
        return_value={"id": 1, "summary": "Topic Summary", "detail": "Topic Detail"},
    ):
        with patch("grox_chat.master_graph.api.get_active_plan", return_value=plan):
            with patch("grox_chat.master_graph.ask_gemini_cli", new=mock_gemini):
                with patch("grox_chat.master_graph.api.create_subtopic", return_value=10):
                    with patch("grox_chat.master_graph.api.persist_message", new=AsyncMock(return_value=99)):
                        with patch("grox_chat.master_graph.api.update_subtopic_start_msg"):
                            with patch("grox_chat.master_graph.api.advance_plan_cursor"):
                                with patch("grox_chat.master_graph.api.set_topic_status"):
                                    new_state = await node_open_next_subtopic(state)

    assert new_state["current_subtopic_id"] == 10
    assert new_state["plan_id"] == 7
    assert new_state["next_action"] == "run_subtopic"


@pytest.mark.asyncio
async def test_node_topic_replan_or_close_closes_when_room_votes_no():
    state = {"topic_id": 1}

    with patch(
        "grox_chat.master_graph.api.get_topic",
        return_value={"id": 1, "summary": "Topic Summary", "detail": "Topic Detail"},
    ):
        with patch(
            "grox_chat.master_graph.api.get_current_subtopics",
            return_value=[{"summary": "Subtopic 1", "conclusion": "Conclusion 1"}],
        ):
            with patch(
                "grox_chat.master_graph._collect_votes",
                new=AsyncMock(return_value={"yes_votes": 0, "successful_votes": 10, "failed_votes": 0}),
            ):
                with patch("grox_chat.master_graph.aget_embedding", new=AsyncMock(return_value=None)):
                    with patch("grox_chat.master_graph.api.post_message") as post_message:
                        with patch("grox_chat.master_graph.api.set_topic_status") as set_status:
                            result = await node_topic_replan_or_close(state)

    assert result["topic_complete"] is True
    assert result["next_action"] == "close_topic"
    assert route_after_replan(result) == "close_topic"
    post_message.assert_called_once()
    set_status.assert_called_once_with(1, "Closed")


@pytest.mark.asyncio
async def test_node_topic_replan_or_close_refills_and_truncates_to_four():
    state = {"topic_id": 1}
    replanned = [
        {"summary": "A", "detail": "Detail A"},
        {"summary": "B", "detail": "Detail B"},
        {"summary": "C", "detail": "Detail C"},
    ]
    proposal_calls = 0

    async def _proposal_side_effect(*_args, **_kwargs):
        nonlocal proposal_calls
        proposal_calls += 1
        return {"candidates": replanned if proposal_calls == 1 else [], "error": None}

    with patch(
        "grox_chat.master_graph.api.get_topic",
        return_value={"id": 1, "summary": "Topic Summary", "detail": "Topic Detail"},
    ):
        with patch(
            "grox_chat.master_graph.api.get_current_subtopics",
            return_value=[{"summary": "Done 1", "conclusion": "Conclusion 1"}],
        ):
            with patch(
                "grox_chat.master_graph._collect_votes",
                new=AsyncMock(
                    side_effect=[
                        {"yes_votes": 8, "successful_votes": 10, "failed_votes": 0},
                        {"yes_votes": 8, "successful_votes": 10, "failed_votes": 0},
                        {"yes_votes": 8, "successful_votes": 10, "failed_votes": 0},
                        {"yes_votes": 8, "successful_votes": 10, "failed_votes": 0},
                    ]
                ),
            ):
                with patch(
                    "grox_chat.master_graph._propose_subtopics",
                    new=AsyncMock(side_effect=_proposal_side_effect),
                ):
                    with patch("grox_chat.master_graph.api.create_plan", return_value=9) as create_plan:
                        result = await node_topic_replan_or_close(state)

    stored_subtopics = json.loads(create_plan.call_args.args[1])
    assert len(stored_subtopics) == 3
    assert result["plan_id"] == 9
    assert result["next_action"] == "open_next_subtopic"


def test_inspect_topic_state_opens_next_subtopic_before_replanning():
    active_plan = {
        "id": 3,
        "current_index": 1,
        "content": json.dumps(
            [
                {"summary": "Done", "detail": "Done detail"},
                {"summary": "Next", "detail": "Next detail"},
            ]
        ),
    }

    with patch("grox_chat.master_graph.api.get_active_plan", return_value=active_plan):
        with patch("grox_chat.master_graph.api.get_open_subtopic", return_value=None):
            with patch(
                "grox_chat.master_graph.api.get_current_subtopics",
                return_value=[{"id": 1, "summary": "Done"}],
            ):
                result = node_inspect_topic_state({"topic_id": 1})

    assert result["next_action"] == "open_next_subtopic"
    assert result["plan_id"] == 3


def test_route_after_open_next_subtopic_replans_when_no_subtopic_was_opened():
    assert route_after_open_next_subtopic({"next_action": "replan_or_close"}) == "replan_or_close"
    assert route_after_open_next_subtopic({"current_subtopic_id": 0}) == "replan_or_close"
    assert route_after_open_next_subtopic({"current_subtopic_id": 10, "next_action": "run_subtopic"}) == "run_subtopic"


def test_route_after_generate_plan_honors_close_and_defer():
    assert route_after_generate_plan({"topic_complete": True, "next_action": "close_topic"}) == "close_topic"
    assert route_after_generate_plan({"deferred": True, "next_action": "defer_topic"}) == "defer_topic"
    assert route_after_generate_plan({"plan_id": 7, "next_action": "open_next_subtopic"}) == "open_next_subtopic"


@pytest.mark.asyncio
async def test_node_plan_generation_defers_on_vote_execution_failure():
    state = {"topic_id": 1}

    with patch(
        "grox_chat.master_graph.api.get_topic",
        return_value={"id": 1, "summary": "Topic Summary", "detail": "Topic Detail"},
    ):
        with patch(
            "grox_chat.master_graph._propose_subtopics",
            new=AsyncMock(return_value={"candidates": [{"summary": "A", "detail": "A"}], "error": None}),
        ):
            with patch(
                "grox_chat.master_graph._collect_votes",
                new=AsyncMock(return_value={"yes_votes": 2, "successful_votes": 8, "failed_votes": 2}),
            ):
                result = await node_plan_generation(state)

    assert result["deferred"] is True
    assert result["next_action"] == "defer_topic"


@pytest.mark.asyncio
async def test_node_plan_generation_defers_on_proposal_error():
    state = {"topic_id": 1}

    with patch(
        "grox_chat.master_graph.api.get_topic",
        return_value={"id": 1, "summary": "Topic Summary", "detail": "Topic Detail"},
    ):
        with patch(
            "grox_chat.master_graph._propose_subtopics",
            new=AsyncMock(return_value={"candidates": [], "error": "provider outage"}),
        ):
            result = await node_plan_generation(state)

    assert result["deferred"] is True
    assert result["next_action"] == "defer_topic"


@pytest.mark.asyncio
async def test_node_topic_replan_or_close_defers_on_replan_vote_failure():
    state = {"topic_id": 1}

    with patch(
        "grox_chat.master_graph.api.get_topic",
        return_value={"id": 1, "summary": "Topic Summary", "detail": "Topic Detail"},
    ):
        with patch(
            "grox_chat.master_graph.api.get_current_subtopics",
            return_value=[{"summary": "Done 1", "conclusion": "Conclusion 1"}],
        ):
            with patch(
                "grox_chat.master_graph._collect_votes",
                new=AsyncMock(return_value={"yes_votes": 0, "successful_votes": 9, "failed_votes": 1}),
            ):
                result = await node_topic_replan_or_close(state)

    assert result["deferred"] is True
    assert result["next_action"] == "defer_topic"


@pytest.mark.asyncio
async def test_node_topic_replan_or_close_passes_completed_subtopics_into_proposals():
    state = {"topic_id": 1}
    current_subtopics = [
        {"summary": "Done 1", "conclusion": "Conclusion 1"},
        {"summary": "Done 2", "conclusion": "Conclusion 2"},
    ]

    with patch(
        "grox_chat.master_graph.api.get_topic",
        return_value={"id": 1, "summary": "Topic Summary", "detail": "Topic Detail"},
    ):
        with patch(
            "grox_chat.master_graph.api.get_current_subtopics",
            return_value=current_subtopics,
        ):
            with patch(
                "grox_chat.master_graph._collect_votes",
                new=AsyncMock(
                    side_effect=[
                        {"yes_votes": 8, "successful_votes": 10, "failed_votes": 0},
                        {"yes_votes": 8, "successful_votes": 10, "failed_votes": 0},
                    ]
                ),
            ):
                with patch(
                    "grox_chat.master_graph._propose_subtopics",
                    new=AsyncMock(return_value={"candidates": [], "error": None}),
                ) as propose:
                    with patch("grox_chat.master_graph.aget_embedding", new=AsyncMock(return_value=None)):
                        with patch("grox_chat.master_graph.api.post_message"):
                            with patch("grox_chat.master_graph.api.set_topic_status"):
                                await node_topic_replan_or_close(state)

    assert propose.await_args.args[3] == ["Done 1", "Done 2"]
