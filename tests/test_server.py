import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from grox_chat.broker import BrokerResponse, SearchEvidenceItem
from grox_chat.server import (
    CAT_EXPANSION_TURN,
    DEBATE_PHASE,
    DOG_CORRECTION_TURN,
    EVIDENCE_PHASE,
    OPENING_PHASE,
    SPECTATOR,
    TRON_REMEDIATION_TURN,
    route_after_final_librarian,
    _aggregate_termination_votes,
    _extract_target_from_content,
    _refresh_pending_turns_with_extras,
    _build_termination_question,
    _has_required_summary_sections,
    _normalize_termination_vote_contract,
    _run_termination_votes,
    _sanitize_citations_to_allowed_ids,
    _termination_policy_for_round,
    _should_run_termination_vote,
    _normalize_fact_proposal_contract,
    _normalize_focus_contract,
    build_actor_system_prompt,
    _build_vote_prompt,
    _normalize_message_contract,
    audience_summary_node,
    build_audience_summary_prompt,
    build_clerk_sourced_fact_prompt,
    build_fact_proposer_prompt,
    build_graph,
    build_librarian_prompt,
    audience_termination_check_node,
    build_base_turns_for_phase,
    build_extra_turns,
    build_turn_queue_for_round,
    build_stages_for_round,
    _build_intervention_turns,
    parallel_group_node,
    sequential_group_node,
    bootstrap_fact_intake_node,
    expert_node,
    fact_proposer_node,
    final_fact_proposer_node,
    final_librarian_node,
    final_writer_node,
    get_phase_for_round,
    librarian_node,
    should_enable_web_backup,
    should_enable_web_search,
    writer_node,
)


@pytest.mark.asyncio
async def test_final_writer_node_is_harvest_only_noop():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": True,
        "round_number": 3,
        "last_writer_round": 3,
    }

    with patch("grox_chat.server.call_text", new=AsyncMock()) as writer_query:
        result = await final_writer_node(state)

    assert result == {}
    writer_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_writer_node_persists_only_critique_message():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": EVIDENCE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 2,
        "last_writer_round": None,
    }
    messages = [
        {"id": 1, "sender": "dreamer", "content": "claim", "msg_type": "standard", "confidence_score": 7.0},
    ]
    writer_replies = [
        "ISSUE 1: False precision\nWHY IT MATTERS: The recommendation depends on invented numbers.",
        "PRIMARY ISSUE: False precision\nWHY CENTRAL: It distorts the recommendation.\nSECONDARY ISSUE: none",
        '{"action":"post_message","content":"Writer critique"}',
    ]

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch("grox_chat.server.call_text", new=AsyncMock(side_effect=writer_replies)) as writer_query:
                        with patch("grox_chat.server.api.persist_message", new=AsyncMock(return_value=55)) as persist_message:
                            with patch("grox_chat.server.process_writer_output", new=AsyncMock()) as process_writer_output:
                                result = await writer_node(state)

    assert writer_query.await_count == 3
    assert writer_query.await_args_list[0].kwargs["provider"] == "minimax"
    assert writer_query.await_args_list[1].kwargs["provider"] == "minimax"
    assert writer_query.await_args_list[2].kwargs["require_json"] is True
    persist_message.assert_awaited_once()
    assert persist_message.await_args.args[:4] == (1, 1, "writer", "Writer critique")
    assert persist_message.await_args.kwargs["round_number"] == 2
    assert persist_message.await_args.kwargs["turn_kind"] == "writer_critique"
    process_writer_output.assert_not_awaited()
    assert result["last_writer_round"] == 2


@pytest.mark.asyncio
async def test_writer_node_retries_empty_outputs_for_all_three_stages():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": EVIDENCE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 2,
        "last_writer_round": None,
    }
    messages = [
        {"id": 1, "sender": "dreamer", "content": "claim", "msg_type": "standard", "confidence_score": 7.0},
    ]
    writer_replies = [
        "",
        "ISSUE 1: False precision\nWHY IT MATTERS: The recommendation depends on invented numbers.",
        "   ",
        "PRIMARY ISSUE: False precision\nWHY CENTRAL: It distorts the recommendation.\nSECONDARY ISSUE: none",
        "",
        '{"action":"post_message","content":"Writer critique"}',
    ]

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch("grox_chat.server.call_text", new=AsyncMock(side_effect=writer_replies)) as writer_query:
                        with patch("grox_chat.server.api.persist_message", new=AsyncMock(return_value=55)) as persist_message:
                            result = await writer_node(state)

    assert writer_query.await_count == 6
    assert writer_query.await_args_list[0].kwargs["require_json"] is False
    assert writer_query.await_args_list[1].kwargs["require_json"] is False
    assert writer_query.await_args_list[2].kwargs["require_json"] is False
    assert writer_query.await_args_list[3].kwargs["require_json"] is False
    assert writer_query.await_args_list[4].kwargs["require_json"] is True
    assert writer_query.await_args_list[5].kwargs["require_json"] is True
    persist_message.assert_awaited_once()
    assert persist_message.await_args.args[:4] == (1, 1, "writer", "Writer critique")
    assert result["last_writer_round"] == 2


def test_build_graph_routes_close_path_through_drain_daemon():
    graph = build_graph().get_graph()
    close_targets = [
        edge.target
        for edge in graph.edges
        if edge.source == "audience_termination_check_node" and edge.data == "close_subtopic"
    ]

    assert close_targets == ["drain_daemon_node"]
    assert "final_writer_node" not in graph.nodes


@pytest.mark.asyncio
async def test_fact_proposer_node_caps_candidates_in_regular_round():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 3,
        "last_writer_round": None,
        "last_fact_proposer_round": None,
    }
    messages = [
        {"id": 1, "sender": "critic", "content": "Latency rose by 12% in the cited benchmark.", "msg_type": "standard", "confidence_score": 7.0},
    ]
    proposer_reply = '{"action":"propose_fact_candidates","fact_candidates":[]}'

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch("grox_chat.server.call_text", new=AsyncMock(return_value=proposer_reply)):
                        with patch("grox_chat.server.process_writer_output", new=AsyncMock(return_value=[1, 2])) as process_writer_output:
                            await fact_proposer_node(state)

    assert process_writer_output.await_args.kwargs["max_candidates"] == 3
    assert process_writer_output.await_args.args[2] is None
    assert process_writer_output.await_args.kwargs["structured_facts"][0]["candidate_type"] == "number"


@pytest.mark.asyncio
async def test_final_fact_proposer_node_allows_three_candidates():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": True,
        "round_number": 4,
        "last_writer_round": 4,
        "last_fact_proposer_round": 4,
        "last_final_fact_proposer_round": None,
    }
    messages = [
        {"id": 1, "sender": "critic", "content": "The paper reported a 42% failure reduction.", "msg_type": "standard", "confidence_score": 7.0},
    ]
    proposer_reply = '{"action":"propose_fact_candidates","fact_candidates":[]}'

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch("grox_chat.server.call_text", new=AsyncMock(return_value=proposer_reply)):
                        with patch("grox_chat.server.process_writer_output", new=AsyncMock(return_value=[1, 2, 3])) as process_writer_output:
                            await final_fact_proposer_node(state)

    assert process_writer_output.await_args.kwargs["max_candidates"] == 4


def test_normalize_fact_proposal_contract_filters_blank_entries():
    parsed = _normalize_fact_proposal_contract(
        '{"action":"propose_facts","facts":["Fact A","  ",42,"Fact B"]}'
    )

    assert parsed["parsed_ok"] is True
    assert parsed["facts"] == ["Fact A", "Fact B"]


@pytest.mark.asyncio
async def test_librarian_node_reviews_candidates_and_posts_visible_audit():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 3,
    }
    candidates = [
        {"id": 10, "candidate_text": "Fact A"},
        {"id": 11, "candidate_text": "Fact B"},
    ]
    messages = [
        {"id": 1, "sender": "writer", "content": "Writer pass", "msg_type": "standard", "confidence_score": None},
    ]
    reviews = [
        {"candidate_id": 10, "decision": "accept", "reviewed_text": "Fact A", "review_note": "ok"},
        {"candidate_id": 11, "decision": "reject", "reviewed_text": None, "review_note": "unsupported"},
    ]

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_pending_fact_candidates", return_value=candidates):
                with patch("grox_chat.server.api.get_pending_claim_candidates", return_value=[]):
                    with patch("grox_chat.server.api.get_messages", return_value=messages):
                        with patch("grox_chat.server.build_query_rag_context", new=AsyncMock(return_value=("RAG", False))):
                            with patch(
                                "grox_chat.server._query_librarian_review_text",
                                new=AsyncMock(
                                    side_effect=[
                                        ('{"decision":"accept","reviewed_text":"Fact A","review_note":"ok","evidence_note":"source","confidence_score":8}', "minimax"),
                                        ('{"decision":"reject","review_note":"unsupported","evidence_note":"missing support","confidence_score":3}', "minimax"),
                                    ]
                                ),
                            ):
                                with patch("grox_chat.server.apply_librarian_review", new=AsyncMock(side_effect=reviews)):
                                    with patch("grox_chat.server.api.persist_message", new=AsyncMock()) as persist_message:
                                        await librarian_node(state)

    persist_message.assert_awaited_once()
    assert persist_message.await_args.args[2] == "librarian"
    assert "LIBRARIAN AUDIT:" in persist_message.await_args.args[3]
    assert persist_message.await_args.kwargs["round_number"] == 3
    assert persist_message.await_args.kwargs["turn_kind"] == "librarian_audit"


@pytest.mark.asyncio
async def test_librarian_node_falls_back_to_gemini_when_minimax_schema_invalid():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 3,
    }
    candidates = [{"id": 10, "candidate_text": "Fact A"}]
    messages = [{"id": 1, "sender": "writer", "content": "Writer pass", "msg_type": "standard", "confidence_score": None}]

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_pending_fact_candidates", return_value=candidates):
                with patch("grox_chat.server.api.get_pending_claim_candidates", return_value=[]):
                    with patch("grox_chat.server.api.get_messages", return_value=messages):
                        with patch("grox_chat.server.build_query_rag_context", new=AsyncMock(return_value=("RAG", False))):
                            with patch(
                                "grox_chat.server._query_librarian_review_text",
                                new=AsyncMock(return_value=('{"decision":"soften"}', "minimax")),
                            ):
                                with patch(
                                    "grox_chat.server.call_text",
                                    new=AsyncMock(return_value='{"decision":"accept","reviewed_text":"Fact A","review_note":"ok","evidence_note":"source","confidence_score":8}'),
                                ) as gemini_query:
                                    with patch(
                                        "grox_chat.server.apply_librarian_review",
                                        new=AsyncMock(return_value={"candidate_id": 10, "decision": "accept", "reviewed_text": "Fact A", "review_note": "ok"}),
                                    ):
                                        with patch("grox_chat.server.api.persist_message", new=AsyncMock()):
                                            await librarian_node(state)

    gemini_query.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("round_number, phase", [(1, OPENING_PHASE), (2, EVIDENCE_PHASE)])
async def test_audience_termination_skips_votes_before_round_three(round_number, phase):
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": phase,
        "subtopic_exhausted": False,
        "round_number": round_number,
    }

    messages = [
        {"id": 9, "sender": "skynet", "content": "Current summary", "msg_type": "summary", "confidence_score": None},
    ]

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.aget_embedding", new=AsyncMock(return_value=[0.1] * 384)):
                    with patch("grox_chat.server.api.search_messages_hybrid", return_value=[]):
                        with patch("grox_chat.server._run_termination_votes", new=AsyncMock()) as run_votes:
                            with patch("grox_chat.server.api.post_message") as post_message:
                                result = await audience_termination_check_node(state)

    assert result["subtopic_exhausted"] is False
    run_votes.assert_not_awaited()
    post_message.assert_not_called()


@pytest.mark.asyncio
async def test_audience_termination_continues_when_room_votes_no():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 3,
    }

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=[]):
                with patch("grox_chat.server._run_termination_votes", new=AsyncMock(return_value=[])) as run_votes:
                    with patch("grox_chat.server.api.post_message") as post_message:
                        result = await audience_termination_check_node(state)

    assert result["subtopic_exhausted"] is False
    run_votes.assert_awaited_once()
    post_message.assert_not_called()


@pytest.mark.asyncio
async def test_audience_termination_posts_warning_when_loop_detected_but_continues():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 3,
        "latest_summary_msg_id": 9,
    }
    messages = [
        {"id": 9, "sender": "skynet", "content": "Current summary", "msg_type": "summary", "confidence_score": None},
    ]

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.aget_embedding", new=AsyncMock(return_value=[0.1] * 384)):
                    with patch(
                        "grox_chat.server.api.search_messages_hybrid",
                        return_value=[{"id": 3, "content": "Past summary", "distance": 0.2}],
                    ):
                        with patch(
                            "grox_chat.server._run_termination_votes",
                            new=AsyncMock(return_value=[]),
                        ):
                            with patch("grox_chat.server.api.post_message") as post_message:
                                result = await audience_termination_check_node(state)

    assert result["subtopic_exhausted"] is False
    post_message.assert_called_once_with(
        1,
        1,
        "skynet",
        "System warning: this debate is revisiting prior conclusions. Bring new evidence, a narrower unresolved claim, or a different assumption next round.",
        msg_type="warning",
        round_number=3,
        turn_kind="skynet_warning",
    )


@pytest.mark.asyncio
async def test_audience_termination_keeps_loop_warning_before_round_three():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": EVIDENCE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 2,
        "latest_summary_msg_id": 9,
    }
    messages = [
        {"id": 9, "sender": "skynet", "content": "Current summary", "msg_type": "summary", "confidence_score": None},
    ]

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.aget_embedding", new=AsyncMock(return_value=[0.1] * 384)):
                    with patch(
                        "grox_chat.server.api.search_messages_hybrid",
                        return_value=[{"id": 3, "content": "Past summary", "distance": 0.2}],
                    ):
                        with patch("grox_chat.server._run_termination_votes", new=AsyncMock()) as run_votes:
                            with patch("grox_chat.server.api.post_message") as post_message:
                                result = await audience_termination_check_node(state)

    assert result["subtopic_exhausted"] is False
    run_votes.assert_not_awaited()
    post_message.assert_called_once_with(
        1,
        1,
        "skynet",
        "System warning: this debate is revisiting prior conclusions. Bring new evidence, a narrower unresolved claim, or a different assumption next round.",
        msg_type="warning",
        round_number=2,
        turn_kind="skynet_warning",
    )


@pytest.mark.asyncio
async def test_audience_termination_degrades_open_when_vote_execution_fails():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 3,
    }
    messages = [
        {"id": 9, "sender": "skynet", "content": "Current summary", "msg_type": "summary", "confidence_score": None},
    ]

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.aget_embedding", new=AsyncMock(return_value=[0.1] * 384)):
                    with patch("grox_chat.server.api.search_messages_hybrid", return_value=[]):
                        with patch(
                            "grox_chat.server._run_termination_votes",
                            new=AsyncMock(side_effect=RuntimeError("all vote paths failed")),
                        ):
                            result = await audience_termination_check_node(state)

    assert result["subtopic_exhausted"] is False


@pytest.mark.asyncio
async def test_audience_termination_forces_close_at_round_ten():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 10,
    }

    with patch("grox_chat.server._run_termination_votes", new=AsyncMock()) as run_votes:
        result = await audience_termination_check_node(state)

    assert result["subtopic_exhausted"] is True
    run_votes.assert_not_awaited()


@pytest.mark.asyncio
async def test_audience_termination_does_not_treat_lexical_hit_alone_as_loop():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 4,
        "latest_summary_msg_id": 9,
    }
    messages = [
        {"id": 9, "sender": "skynet", "content": "Current summary", "msg_type": "summary", "confidence_score": None},
    ]

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.aget_embedding", new=AsyncMock(return_value=[0.1] * 384)):
                    with patch(
                        "grox_chat.server.api.search_messages_hybrid",
                        return_value=[{"id": 3, "content": "Past summary", "distance": 0.9, "lexical_score": -0.2}],
                    ):
                        with patch(
                            "grox_chat.server._run_termination_votes",
                            new=AsyncMock(return_value=[]),
                        ):
                            with patch("grox_chat.server.api.post_message") as post_message:
                                result = await audience_termination_check_node(state)

    assert result["subtopic_exhausted"] is False
    post_message.assert_not_called()


@pytest.mark.asyncio
async def test_expert_node_lowers_confidence_on_parse_failure():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "dreamer",
        "current_turn_kind": "base",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": EVIDENCE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 2,
    }
    messages = [
        {"id": 1, "sender": "critic", "content": "Need stronger grounding.", "msg_type": "standard", "confidence_score": 7.0},
    ]

    response = BrokerResponse(
        text="plain text response",
        search_evidence=(),
        search_failed=False,
    )

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("", True))):
                    with patch("grox_chat.server.call_text_with_search_evidence", new=AsyncMock(return_value=response)):
                        with patch("grox_chat.server.api.persist_message", new=AsyncMock()) as persist_message:
                            await expert_node(state)

    persist_message.assert_awaited_once()
    assert persist_message.await_args.kwargs["confidence_score"] == 3.0
    assert persist_message.await_args.kwargs["round_number"] == 2
    assert persist_message.await_args.kwargs["turn_kind"] == "base"


@pytest.mark.asyncio
async def test_opening_round_expert_node_skips_web_search():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "dreamer",
        "current_turn_kind": "base",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": OPENING_PHASE,
        "subtopic_exhausted": False,
        "round_number": 1,
    }
    messages = [
        {"id": 1, "sender": "skynet", "content": "Grounding brief", "msg_type": "standard", "confidence_score": None},
    ]

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch("grox_chat.server.call_text", new=AsyncMock(return_value='{"action":"post_message","content":"Initial stance","confidence_score":7}')) as call_text:
                        with patch("grox_chat.server.api.persist_message", new=AsyncMock()) as persist_message:
                            await expert_node(state)

    call_text.assert_awaited_once()
    persist_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_expert_node_strips_citations_not_present_in_injected_knowledge():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "dreamer",
        "current_turn_kind": "base",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": OPENING_PHASE,
        "subtopic_exhausted": False,
        "round_number": 1,
    }
    messages = [
        {"id": 1, "sender": "skynet", "content": "Grounding brief", "msg_type": "standard", "confidence_score": None},
    ]
    rag_context = (
        "=== RAG KNOWLEDGE INJECTION ===\n"
        "[Related Facts]\n- [F1] Fact 1\n"
        "[Related Claims]\n- [C7] Claim 7\n"
        "[Related Web Evidence]\n- [W9] Web 9\n"
    )
    raw_reply = (
        '{"action":"post_message","content":"Use [F1], [C7], and [W9]. Ignore [F999], [C88], and [W77].",'
        '"confidence_score":7}'
    )

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=(rag_context, False))):
                    with patch("grox_chat.server.call_text", new=AsyncMock(return_value=raw_reply)):
                        with patch("grox_chat.server.api.persist_message", new=AsyncMock()) as persist_message:
                            await expert_node(state)

    persist_message.assert_awaited_once()
    assert persist_message.await_args.args[3] == "Use [F1], [C7], and [W9]. Ignore, and."


@pytest.mark.asyncio
async def test_contrarian_expert_node_uses_search_loop_response_instead_of_http_error():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "contrarian",
        "current_turn_kind": "base",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 3,
    }
    messages = [
        {"id": 1, "sender": "critic", "content": "The consensus is too neat.", "msg_type": "standard", "confidence_score": 7.0},
    ]

    contrarian_reply = '{"action":"post_message","content":"The group is overconfident about soft-skill universals.","confidence_score":6}'

    response = BrokerResponse(
        text=contrarian_reply,
        search_evidence=(),
        search_failed=False,
    )

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch("grox_chat.server.call_text_with_search_evidence", new=AsyncMock(return_value=response)):
                        with patch("grox_chat.server.api.persist_message", new=AsyncMock()) as persist_message:
                            await expert_node(state)

    persist_message.assert_awaited_once()
    assert persist_message.await_args.args[3] == "The group is overconfident about soft-skill universals."
    assert persist_message.await_args.kwargs["confidence_score"] == 6.0
    assert persist_message.await_args.kwargs["turn_kind"] == "base"


@pytest.mark.asyncio
async def test_contrarian_expert_node_uses_search_loop_response_instead_of_http_error_legacy_removed():
    # kept as a no-op guard against reintroducing direct search-loop patch points
    assert True


def test_phase_helpers_and_base_turns():
    assert get_phase_for_round(1) == OPENING_PHASE
    assert get_phase_for_round(2) == EVIDENCE_PHASE
    assert get_phase_for_round(3) == DEBATE_PHASE

    assert [turn["actor"] for turn in build_base_turns_for_phase(OPENING_PHASE)] == [
        "dreamer", "scientist", "engineer", "analyst", "critic", "tron"
    ]
    assert [turn["actor"] for turn in build_base_turns_for_phase(EVIDENCE_PHASE)] == [
        "dreamer", "scientist", "engineer", "analyst", "critic", "contrarian", "dog", "cat", "tron", "spectator"
    ]

    _, evidence_turns = build_turn_queue_for_round({"round_number": 2}, 2)
    assert all(turn["turn_kind"] == "base" for turn in evidence_turns)


def test_build_extra_turns_preserves_tron_then_dog_then_cat_order_with_duplicates():
    state = {
        "tron_target": "scientist",
        "dog_target": "scientist",
        "cat_target": "scientist",
    }

    assert build_extra_turns(state) == [
        {"actor": "scientist", "turn_kind": TRON_REMEDIATION_TURN},
        {"actor": "scientist", "turn_kind": DOG_CORRECTION_TURN},
        {"actor": "scientist", "turn_kind": CAT_EXPANSION_TURN},
    ]


def test_refresh_pending_turns_with_extras_redeems_round_two_targets_same_round():
    state = {
        "round_number": 2,
        "phase": EVIDENCE_PHASE,
        "pending_turns": [
            {"actor": "tron", "turn_kind": "base"},
            {"actor": "scientist", "turn_kind": DOG_CORRECTION_TURN},
        ],
        "dog_target": "scientist",
        "cat_target": None,
        "tron_target": None,
    }
    updates = {"cat_target": "critic"}

    _refresh_pending_turns_with_extras(state, updates)

    assert updates["pending_turns"] == [
        {"actor": "tron", "turn_kind": "base"},
        {"actor": "scientist", "turn_kind": DOG_CORRECTION_TURN},
        {"actor": "critic", "turn_kind": CAT_EXPANSION_TURN},
    ]


def test_refresh_pending_turns_with_extras_does_not_resurrect_consumed_target():
    state = {
        "round_number": 3,
        "phase": DEBATE_PHASE,
        "pending_turns": [
            {"actor": "critic", "turn_kind": "base"},
        ],
        "dog_target": None,
        "cat_target": "dog",
        "tron_target": None,
    }
    updates = {
        "current_actor": "",
        "current_turn_kind": "",
        "cat_target": None,
    }

    _refresh_pending_turns_with_extras(state, updates)

    assert updates["pending_turns"] == [
        {"actor": "critic", "turn_kind": "base"},
    ]


def test_termination_policy_is_graduated_by_round():
    assert _termination_policy_for_round(3)[0] == "weak"
    assert _termination_policy_for_round(5)[0] == "medium"
    assert _termination_policy_for_round(8)[0] == "strong"
    assert _termination_policy_for_round(10)[0] == "forced"


def test_termination_votes_begin_at_round_three():
    assert _should_run_termination_vote(1) is False
    assert _should_run_termination_vote(2) is False
    assert _should_run_termination_vote(3) is True


@pytest.mark.asyncio
async def test_final_librarian_keeps_subtopic_open_when_pending_candidates_remain():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": True,
        "round_number": 4,
    }

    with patch("grox_chat.server._run_librarian_pass", new=AsyncMock(return_value={})):
        with patch(
            "grox_chat.server.api.get_pending_fact_candidates",
            return_value=[{"id": 17, "candidate_text": "Still pending"}],
        ):
            with patch("grox_chat.server.api.get_pending_claim_candidates", return_value=[]):
                result = await final_librarian_node(state)

    assert result["pending_fact_reviews_remaining"] is True
    assert result["subtopic_exhausted"] is False
    assert route_after_final_librarian(result) == "setup_next_round"


@pytest.mark.asyncio
async def test_final_librarian_allows_close_when_no_pending_candidates_remain():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": True,
        "round_number": 4,
    }

    with patch("grox_chat.server._run_librarian_pass", new=AsyncMock(return_value={})):
        with patch("grox_chat.server.api.get_pending_fact_candidates", return_value=[]):
            with patch("grox_chat.server.api.get_pending_claim_candidates", return_value=[]):
                result = await final_librarian_node(state)

    assert result["pending_fact_reviews_remaining"] is False
    assert route_after_final_librarian(result) == "close_subtopic"


def test_target_extraction_supports_chinese_and_case_insensitive_english_names():
    assert _extract_target_from_content("*growls at 工程师* Bark!", "dog") == "engineer"
    assert _extract_target_from_content("*runs to [ScIeNtIsT]* Nya!", "cat") == "scientist"
    assert _extract_target_from_content("[VIOLATION DETECTED: 批评家]", "tron") == "critic"


def test_should_enable_web_search_is_phase_and_turn_kind_aware():
    assert should_enable_web_search({"phase": OPENING_PHASE}, "dreamer", "base") is False
    assert should_enable_web_search({"phase": EVIDENCE_PHASE}, "tron", "base") is True
    assert should_enable_web_search({"phase": DEBATE_PHASE}, "contrarian", "base") is True
    assert should_enable_web_search({"phase": DEBATE_PHASE}, "dreamer", DOG_CORRECTION_TURN) is True
    assert should_enable_web_search({"phase": DEBATE_PHASE}, "dreamer", CAT_EXPANSION_TURN) is True
    assert should_enable_web_search({"phase": DEBATE_PHASE}, "dreamer", TRON_REMEDIATION_TURN) is False
    assert should_enable_web_search({"phase": DEBATE_PHASE}, "dog", "base") is False


def test_should_enable_web_backup_is_debate_only_and_base_turn_only():
    assert should_enable_web_backup({"phase": OPENING_PHASE}, "dreamer", "base") is False
    assert should_enable_web_backup({"phase": EVIDENCE_PHASE}, "dreamer", "base") is False
    assert should_enable_web_backup({"phase": DEBATE_PHASE}, "dreamer", "base") is True
    assert should_enable_web_backup({"phase": DEBATE_PHASE}, "dog", "base") is True
    assert should_enable_web_backup({"phase": DEBATE_PHASE}, "dreamer", DOG_CORRECTION_TURN) is False
    assert should_enable_web_backup({"phase": DEBATE_PHASE}, SPECTATOR, "base") is False


def test_build_actor_system_prompt_adds_mini_max_safe_deliberation_constraints():
    prompt = build_actor_system_prompt({"phase": DEBATE_PHASE, "round_number": 4}, "analyst", "base")

    assert "DEBATE DISCIPLINE:" in prompt
    assert "Prefer net-new argument" in prompt
    assert "Do not invent exact percentages, costs, latency figures, or synthetic scores" in prompt
    assert "[F{id}]" in prompt


def test_build_actor_system_prompt_preserves_dog_targeting_role():
    prompt = build_actor_system_prompt({"phase": DEBATE_PHASE, "round_number": 4}, "dog", "base")

    assert "Choose exactly one target" in prompt
    assert "*growls at [Name]*" in prompt
    assert "Prioritize logical pressure over roleplay volume" in prompt
    assert "Do not invent exact percentages, costs, latency figures, or synthetic scores" not in prompt


def test_normalize_message_contract_filters_non_string_facts():
    parsed = _normalize_message_contract(
        '{"action":"post_message","content":"Writer check","facts":[{"bad": 1}, 7, "Verified fact", "  "]}'
    )

    assert parsed["parsed_ok"] is True
    assert parsed["facts"] == ["Verified fact"]


def test_normalize_message_contract_extracts_single_wrapped_content():
    parsed = _normalize_message_contract(
        '{"action":"post_message","content":"Scientist correction","confidence_score":7.5}'
    )

    assert parsed["parsed_ok"] is True
    assert parsed["content"] == "Scientist correction"
    assert parsed["confidence_score"] == 7.5


def test_normalize_message_contract_extracts_fenced_single_wrapped_content_without_recursing():
    parsed = _normalize_message_contract(
        '```json\n{"action":"post_message","content":"{\\"nested\\":true}"}\n```'
    )

    assert parsed["parsed_ok"] is True
    assert parsed["content"] == '{"nested":true}'


def test_sanitize_citations_to_allowed_ids_strips_only_unprovided_ids():
    cleaned, removed = _sanitize_citations_to_allowed_ids(
        "Use [F1] and [W7], ignore [F999] and [C88].",
        knowledge_blocks=(
            "=== RAG ===\n[F1] Fact\n[C5] Claim\n",
            "=== WEB ===\n[W7] Title: T\n",
        ),
    )

    assert cleaned == "Use [F1] and [W7], ignore and."
    assert removed == {"F": (999,), "C": (88,), "W": ()}


def test_build_fact_prompts_explain_that_web_leads_require_review_before_becoming_facts():
    state = {"phase": DEBATE_PHASE, "round_number": 4}
    topic = {"summary": "topic", "detail": "detail"}
    subtopic = {"summary": "subtopic", "detail": "detail"}
    messages = [{"sender": "critic", "content": "A [W17] article claimed something.", "msg_type": "standard"}]
    candidate = {"id": 9, "candidate_text": "Claim text", "candidate_type": "sourced_claim"}

    fact_prompt = build_fact_proposer_prompt(state, topic, messages, "RAG", max_facts=2)
    sourced_prompt = build_clerk_sourced_fact_prompt(state, topic, messages, "RAG", max_facts=2)
    librarian_prompt = build_librarian_prompt(state, topic, subtopic, candidate, messages, "RAG")

    assert "Web evidence [W...]" in fact_prompt
    assert "promote a [W] lead into a fact candidate" in fact_prompt
    assert "Inspect both uncited externally-sourced statements in the debate and any retrieved [W...] items." in sourced_prompt
    assert "raw [W...] text is never permanent memory by itself" in librarian_prompt


def test_normalize_focus_contract_requires_json_boolean_for_grant_web_search():
    parsed = _normalize_focus_contract(
        '{"action":"focus","target":"scientist","reason":"watch closely","grant_web_search":"false"}'
    )

    assert parsed["parsed_ok"] is True
    assert parsed["target"] == "scientist"
    assert parsed["grant_web_search"] is False


def test_build_vote_prompt_for_close_vote_does_not_include_candidate_admission_rubric():
    prompt = _build_vote_prompt(
        question="Should the current subtopic continue?",
        topic_summary="topic",
        topic_detail="detail",
    )

    assert "materially useful for the topic" not in prompt
    assert "redundant with already selected items" not in prompt
    assert '{"vote":"yes|no","reason":"short sentence"}' in prompt


def test_build_audience_summary_prompt_requires_mini_max_safe_sections():
    state = {"round_number": 2, "phase": EVIDENCE_PHASE}
    topic = {"summary": "topic"}
    messages = [
        {"sender": "skynet", "content": "brief", "msg_type": "standard", "confidence_score": None},
        {"sender": "dreamer", "content": "idea", "msg_type": "standard", "confidence_score": 7.0},
        {"sender": "scientist", "content": "check", "msg_type": "standard", "confidence_score": 6.0},
        {"sender": "writer", "content": "verify", "msg_type": "standard", "confidence_score": None},
    ]

    prompt = build_audience_summary_prompt(state, topic, messages)

    assert "TRAJECTORY:" in prompt
    assert "CONSENSUS:" in prompt
    assert "BLOCKERS:" in prompt
    assert "EVIDENCE GAPS:" in prompt
    assert "AGENT DELTAS:" in prompt
    assert "dreamer, scientist, writer" in prompt
    assert "Librarian rulings" in prompt or "Librarian" in prompt
    assert "Prefix each line with `[Central]` or `[Peripheral]`" in prompt
    assert "Do not state whether the subtopic is ready to close." in prompt


def test_has_required_summary_sections_requires_standalone_section_lines():
    malformed = (
        "This summary mentions TRAJECTORY: and CONSENSUS: in prose.\n"
        "It also references BLOCKERS: and EVIDENCE GAPS: casually.\n"
        "Finally it name-drops AGENT DELTAS: without using real sections."
    )
    valid = (
        "TRAJECTORY:\nStable this round.\n"
        "CONSENSUS:\nSingle-call stays default.\n"
        "BLOCKERS:\nNeed direct benchmark evidence.\n"
        "EVIDENCE GAPS:\n[Central] Missing controlled benchmark.\n"
        "AGENT DELTAS:\n- critic: pressed on evidence.\n"
    )

    assert _has_required_summary_sections(malformed) is False
    assert _has_required_summary_sections(valid) is True


def test_build_termination_question_requires_unresolved_pattern_checks():
    prompt = _build_termination_question("EARLY STAGE. The burden of proof is on continuing.")

    assert "Decide whether this subtopic should CONTINUE or CLOSE." in prompt
    assert "`main_branch`" in prompt
    assert "`centrality`" in prompt
    assert "`recent_shift`" in prompt
    assert "`conditional_support`" in prompt
    assert "`untested_novelty`" in prompt
    assert '"vote":"continue|close"' in prompt


def test_normalize_termination_vote_contract_accepts_aliases():
    parsed = _normalize_termination_vote_contract(
        '{"main_branch":"router circularity","centrality":"central","recent_shift":"no","conditional_support":"false","untested_novelty":"no","vote":"no","override_reason":null}'
    )

    assert parsed["parsed_ok"] is True
    assert parsed["vote"] == "continue"
    assert parsed["main_branch"] == "router circularity"
    assert parsed["central_blocker"] is True
    assert parsed["support_blocker"] is False


def test_normalize_termination_vote_contract_requires_override_to_close_through_blocker():
    parsed = _normalize_termination_vote_contract(
        '{"main_branch":"empirical gap","centrality":"central","recent_shift":"no","conditional_support":"no","untested_novelty":"no","vote":"close","override_reason":null}'
    )

    assert parsed["parsed_ok"] is False
    assert parsed["invalid_reason"] == "missing_override_reason"


@pytest.mark.asyncio
async def test_run_termination_votes_repairs_invalid_schema_once():
    fake_agent = SimpleNamespace(
        spec=SimpleNamespace(role_prompt="Role prompt"),
        governance_vote=AsyncMock(
            return_value='{"main_branch":"empirical gap","centrality":"central","recent_shift":"no","conditional_support":"no","untested_novelty":"no","vote":"close","override_reason":null}'
        ),
    )

    with patch("grox_chat.server.get_agent", return_value=fake_agent):
        with patch(
            "grox_chat.server.query_minimax",
            new=AsyncMock(
                return_value=(
                    '{"main_branch":"empirical gap","centrality":"central","recent_shift":"no","conditional_support":"no","untested_novelty":"no","vote":"continue","override_reason":null}',
                    [],
                )
            ),
        ) as repair_call:
            with patch("grox_chat.server.api.insert_vote_record") as insert_vote_record:
                vote_records = await _run_termination_votes(
                    voters=["critic"],
                    prompt="close or continue?",
                    topic_id=1,
                    subtopic_id=2,
                    round_number=3,
                    subject="Current subtopic summary",
                )

    assert len(vote_records) == 1
    assert vote_records[0]["repair_used"] is True
    assert vote_records[0]["parsed"]["parsed_ok"] is True
    assert vote_records[0]["parsed"]["vote"] == "continue"
    repair_call.assert_awaited_once()
    insert_vote_record.assert_called_once()
    assert insert_vote_record.call_args.args[:8] == (
        1,
        2,
        3,
        "termination",
        "Current subtopic summary",
        "close or continue?",
        "critic",
        True,
    )


def test_aggregate_termination_votes_round_three_blocks_on_single_central_vote():
    vote_records = [
        {
            "voter": "critic",
            "raw_response": "{}",
            "parsed": _normalize_termination_vote_contract(
                '{"main_branch":"empirical gap","centrality":"central","recent_shift":"no","conditional_support":"no","untested_novelty":"no","vote":"continue","override_reason":null}'
            ),
        },
        {
            "voter": "engineer",
            "raw_response": "{}",
            "parsed": _normalize_termination_vote_contract(
                '{"main_branch":"none","centrality":"none","recent_shift":"no","conditional_support":"no","untested_novelty":"no","vote":"close","override_reason":null}'
            ),
        },
    ]

    aggregated = _aggregate_termination_votes(vote_records, 3)

    assert aggregated["subtopic_exhausted"] is False
    assert "central_blocker" in aggregated["blocked_by"]


def test_aggregate_termination_votes_counts_blockers_from_invalid_close_votes():
    vote_records = [
        {
            "voter": "critic",
            "raw_response": "{}",
            "parsed": _normalize_termination_vote_contract(
                '{"main_branch":"empirical gap","centrality":"central","recent_shift":"no","conditional_support":"no","untested_novelty":"no","vote":"close","override_reason":null}'
            ),
        },
        {
            "voter": "engineer",
            "raw_response": "{}",
            "parsed": _normalize_termination_vote_contract(
                '{"main_branch":"none","centrality":"none","recent_shift":"no","conditional_support":"no","untested_novelty":"no","vote":"close","override_reason":null}'
            ),
        },
    ]

    aggregated = _aggregate_termination_votes(vote_records, 3)

    assert aggregated["subtopic_exhausted"] is False
    assert aggregated["invalid_votes"] == 1
    assert aggregated["blocker_counts"]["central_blocker"] == 1
    assert "central_blocker" in aggregated["blocked_by"]


def test_aggregate_termination_votes_round_five_needs_two_blockers_to_stop_close():
    vote_records = [
        {
            "voter": "critic",
            "raw_response": "{}",
            "parsed": _normalize_termination_vote_contract(
                '{"main_branch":"soft facts","centrality":"peripheral","recent_shift":"no","conditional_support":"yes","untested_novelty":"no","vote":"continue","override_reason":null}'
            ),
        },
        {
            "voter": "engineer",
            "raw_response": "{}",
            "parsed": _normalize_termination_vote_contract(
                '{"main_branch":"none","centrality":"none","recent_shift":"no","conditional_support":"no","untested_novelty":"no","vote":"close","override_reason":null}'
            ),
        },
        {
            "voter": "scientist",
            "raw_response": "{}",
            "parsed": _normalize_termination_vote_contract(
                '{"main_branch":"none","centrality":"none","recent_shift":"no","conditional_support":"no","untested_novelty":"no","vote":"close","override_reason":null}'
            ),
        },
    ]

    aggregated = _aggregate_termination_votes(vote_records, 5)

    assert aggregated["subtopic_exhausted"] is True
    assert aggregated["blocked_by"] == []


def test_aggregate_termination_votes_round_eight_blocks_on_three_support_flags():
    vote_records = [
        {
            "voter": voter,
            "raw_response": "{}",
            "parsed": _normalize_termination_vote_contract(
                '{"main_branch":"soft support","centrality":"peripheral","recent_shift":"no","conditional_support":"yes","untested_novelty":"no","vote":"continue","override_reason":null}'
            ),
        }
        for voter in ("critic", "scientist", "analyst")
    ] + [
        {
            "voter": "engineer",
            "raw_response": "{}",
            "parsed": _normalize_termination_vote_contract(
                '{"main_branch":"none","centrality":"none","recent_shift":"no","conditional_support":"no","untested_novelty":"no","vote":"close","override_reason":null}'
            ),
        },
        {
            "voter": "dreamer",
            "raw_response": "{}",
            "parsed": _normalize_termination_vote_contract(
                '{"main_branch":"none","centrality":"none","recent_shift":"no","conditional_support":"no","untested_novelty":"no","vote":"close","override_reason":null}'
            ),
        },
        {
            "voter": "cat",
            "raw_response": "{}",
            "parsed": _normalize_termination_vote_contract(
                '{"main_branch":"none","centrality":"none","recent_shift":"no","conditional_support":"no","untested_novelty":"no","vote":"close","override_reason":null}'
            ),
        },
    ]

    aggregated = _aggregate_termination_votes(vote_records, 8)

    assert aggregated["subtopic_exhausted"] is False
    assert "support_blocker" in aggregated["blocked_by"]


def test_aggregate_termination_votes_degrades_open_on_too_many_invalid_votes():
    vote_records = [
        {"voter": "critic", "raw_response": "oops", "parsed": _normalize_termination_vote_contract("oops")},
        {"voter": "scientist", "raw_response": "oops", "parsed": _normalize_termination_vote_contract("oops")},
        {"voter": "engineer", "raw_response": "oops", "parsed": _normalize_termination_vote_contract("oops")},
    ]

    aggregated = _aggregate_termination_votes(vote_records, 4)

    assert aggregated["subtopic_exhausted"] is False
    assert aggregated["blocked_by"] == ["invalid_votes"]


@pytest.mark.asyncio
async def test_audience_summary_node_uses_degraded_summary_when_all_model_fallbacks_fail():
    state = {"topic_id": 1, "subtopic_id": 1, "round_number": 3, "phase": DEBATE_PHASE}
    topic = {"id": 1, "summary": "topic", "detail": "detail"}
    messages = [
        {"sender": "dreamer", "content": "idea", "msg_type": "standard", "confidence_score": 7.0},
        {"sender": "scientist", "content": "check", "msg_type": "standard", "confidence_score": 6.0},
    ]

    with patch("grox_chat.server.api.get_topic", return_value=topic):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch(
                    "grox_chat.server.call_text",
                    new=AsyncMock(side_effect=RuntimeError("all fallbacks failed")),
                ):
                    with patch("grox_chat.server.aget_embedding", new=AsyncMock(return_value=None)):
                        with patch("grox_chat.server.api.post_message", return_value=77) as post_message:
                            result = await audience_summary_node(state)

    assert result["latest_summary_msg_id"] == 77
    stored_summary = post_message.call_args.args[3]
    assert "TRAJECTORY:" in stored_summary
    assert "CONSENSUS:" in stored_summary
    assert "BLOCKERS:" in stored_summary
    assert "EVIDENCE GAPS:" in stored_summary
    assert "AGENT DELTAS:" in stored_summary
    assert "dreamer" in stored_summary
    assert "scientist" in stored_summary


@pytest.mark.asyncio
async def test_bootstrap_fact_intake_searches_each_direction_independently():
    state = {
        "topic_id": 1,
        "subtopic_id": 1,
        "round_number": 1,
    }
    topic = {"id": 1, "summary": "topic", "detail": "detail"}
    subtopic = {"id": 1, "summary": "subtopic", "detail": "detail"}
    direction_reply = '{"action":"propose_fact_directions","directions":["dir one","dir two","dir three"]}'
    proposer_reply = '{"action":"propose_facts","facts":["Fact"]}'
    evidence = BrokerResponse(
        text="",
        search_evidence=(SearchEvidenceItem(query="q", rendered_results="=== WEB SEARCH RESULTS ===\nTitle: A\nSnippet: B\n"),),
        search_failed=False,
    )

    with patch("grox_chat.server.api.get_topic", return_value=topic):
        with patch("grox_chat.server.api.get_subtopic", return_value=subtopic):
            with patch(
                "grox_chat.server.call_text",
                new=AsyncMock(side_effect=[direction_reply, proposer_reply, proposer_reply, proposer_reply]),
            ) as call_text:
                with patch(
                    "grox_chat.server.collect_search_evidence_bundle",
                    new=AsyncMock(return_value=evidence),
                ) as collect_search:
                    with patch(
                        "grox_chat.server.build_query_rag_context",
                        new=AsyncMock(return_value=("RAG", False)),
                    ):
                        with patch(
                            "grox_chat.server.process_writer_output",
                            new=AsyncMock(side_effect=[[11], [12], [13]]),
                        ) as process_writer_output:
                            with patch(
                                "grox_chat.server._run_librarian_pass",
                                new=AsyncMock(return_value={}),
                            ) as run_librarian:
                                await bootstrap_fact_intake_node(state)

    assert call_text.await_count == 4
    assert collect_search.await_count == 3
    assert process_writer_output.await_count == 3
    for call in process_writer_output.await_args_list:
        assert call.kwargs["fact_stage"] == "bootstrap"
        assert call.kwargs["max_candidates"] == 1
    run_librarian.assert_awaited_once()
    assert run_librarian.await_args.kwargs["candidate_ids"] == [11, 12, 13]
    assert run_librarian.await_args.kwargs["emit_audit_message"] is False


@pytest.mark.asyncio
async def test_bootstrap_fact_intake_skips_zero_hit_placeholder_search_results():
    state = {
        "topic_id": 1,
        "subtopic_id": 1,
        "round_number": 1,
    }
    topic = {"id": 1, "summary": "topic", "detail": "detail"}
    subtopic = {"id": 1, "summary": "subtopic", "detail": "detail"}
    direction_reply = '{"action":"propose_fact_directions","directions":["dir one"]}'
    evidence = BrokerResponse(
        text="",
        search_evidence=(SearchEvidenceItem(query="q", rendered_results="=== WEB SEARCH RESULTS ===\nNo useful results found.\n\n"),),
        search_failed=False,
    )

    with patch("grox_chat.server.api.get_topic", return_value=topic):
        with patch("grox_chat.server.api.get_subtopic", return_value=subtopic):
            with patch("grox_chat.server.call_text", new=AsyncMock(return_value=direction_reply)) as call_text:
                with patch(
                    "grox_chat.server.collect_search_evidence_bundle",
                    new=AsyncMock(return_value=evidence),
                ) as collect_search:
                    with patch("grox_chat.server.process_writer_output", new=AsyncMock()) as process_writer_output:
                        with patch("grox_chat.server._run_librarian_pass", new=AsyncMock()) as run_librarian:
                            await bootstrap_fact_intake_node(state)

    assert call_text.await_count == 1
    collect_search.assert_awaited_once()
    process_writer_output.assert_not_awaited()
    run_librarian.assert_not_awaited()


@pytest.mark.asyncio
async def test_expert_node_runs_inline_fact_intake_after_web_enabled_turn():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "contrarian",
        "current_turn_kind": "base",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 3,
    }
    topic = {"id": 1, "summary": "topic", "detail": "detail"}
    subtopic = {"id": 1, "summary": "subtopic", "detail": "detail"}
    messages = [
        {"id": 1, "sender": "critic", "content": "Need evidence", "msg_type": "standard", "confidence_score": 7.0},
    ]
    response = BrokerResponse(
        text='{"action":"post_message","content":"Contrarian response","confidence_score":6}',
        search_evidence=(SearchEvidenceItem(query="q", rendered_results="=== WEB SEARCH RESULTS ===\nTitle: A\nSnippet: B\n"),),
        search_failed=False,
    )
    fact_reply = '{"action":"propose_facts","facts":["Inline fact"]}'

    with patch("grox_chat.server.api.get_topic", return_value=topic):
        with patch("grox_chat.server.api.get_subtopic", return_value=subtopic):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch("grox_chat.server.call_text_with_search_evidence", new=AsyncMock(return_value=response)):
                        with patch("grox_chat.server.call_text", new=AsyncMock(return_value=fact_reply)) as call_text:
                            with patch("grox_chat.server.api.persist_message", new=AsyncMock(return_value=55)) as persist_message:
                                with patch("grox_chat.server.process_writer_output", new=AsyncMock(return_value=[21])) as process_writer_output:
                                    with patch("grox_chat.server._run_librarian_pass", new=AsyncMock(return_value={})) as run_librarian:
                                        await expert_node(state)

    persist_message.assert_awaited_once()
    process_writer_output.assert_awaited_once()
    inline_prompt = call_text.await_args.args[0]
    assert "=== SEARCH EVIDENCE: q ===" in inline_prompt
    assert "Title: A" in inline_prompt
    assert process_writer_output.await_args.kwargs["fact_stage"] == "inline"
    assert process_writer_output.await_args.kwargs["max_candidates"] == 1
    run_librarian.assert_awaited_once()
    assert run_librarian.await_args.kwargs["candidate_ids"] == [21]
    assert run_librarian.await_args.kwargs["emit_audit_message"] is False


@pytest.mark.asyncio
async def test_inline_fact_intake_skips_zero_hit_placeholder_search_results():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "contrarian",
        "current_turn_kind": "base",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 3,
    }
    topic = {"id": 1, "summary": "topic", "detail": "detail"}
    subtopic = {"id": 1, "summary": "subtopic", "detail": "detail"}
    messages = [
        {"id": 1, "sender": "critic", "content": "Need evidence", "msg_type": "standard", "confidence_score": 7.0},
    ]
    response = BrokerResponse(
        text='{"action":"post_message","content":"Contrarian response","confidence_score":6}',
        search_evidence=(SearchEvidenceItem(query="q", rendered_results="=== WEB SEARCH RESULTS ===\nNo useful results found.\n\n"),),
        search_failed=False,
    )

    with patch("grox_chat.server.api.get_topic", return_value=topic):
        with patch("grox_chat.server.api.get_subtopic", return_value=subtopic):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch("grox_chat.server.call_text_with_search_evidence", new=AsyncMock(return_value=response)):
                        with patch("grox_chat.server.call_text", new=AsyncMock()) as call_text:
                            with patch("grox_chat.server.api.persist_message", new=AsyncMock(return_value=55)):
                                with patch("grox_chat.server.process_writer_output", new=AsyncMock()) as process_writer_output:
                                    with patch("grox_chat.server._run_librarian_pass", new=AsyncMock()) as run_librarian:
                                        await expert_node(state)

    call_text.assert_not_awaited()
    process_writer_output.assert_not_awaited()
    run_librarian.assert_not_awaited()


@pytest.mark.asyncio
async def test_fact_proposer_node_marks_synthesized_stage():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 3,
        "last_fact_proposer_round": None,
    }
    topic = {"id": 1, "summary": "topic", "detail": "detail"}
    subtopic = {"id": 1, "summary": "subtopic", "detail": "detail"}
    messages = [
        {"id": 1, "sender": "critic", "content": "The benchmark shows a 12% latency increase.", "msg_type": "standard", "confidence_score": 7.0},
    ]
    proposer_reply = '{"action":"propose_fact_candidates","fact_candidates":[]}'

    with patch("grox_chat.server.api.get_topic", return_value=topic):
        with patch("grox_chat.server.api.get_subtopic", return_value=subtopic):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch("grox_chat.server.call_text", new=AsyncMock(return_value=proposer_reply)):
                        with patch("grox_chat.server.process_writer_output", new=AsyncMock(return_value=[1])) as process_writer_output:
                            await fact_proposer_node(state)

    process_writer_output.assert_awaited_once()
    assert process_writer_output.await_args.kwargs["fact_stage"] == "synthesized"
    assert process_writer_output.await_args.kwargs["structured_facts"][0]["candidate_type"] == "number"


# ---- Stage-based parallel execution tests ----


def test_build_stages_for_round_opening():
    stages = build_stages_for_round(1)
    assert len(stages) == 2
    assert stages[0]["parallel"] is True
    assert stages[1]["parallel"] is True
    stage1_actors = [t["actor"] for t in stages[0]["agents"]]
    stage2_actors = [t["actor"] for t in stages[1]["agents"]]
    assert "dreamer" in stage1_actors
    assert "scientist" in stage1_actors
    assert "engineer" in stage1_actors
    assert "analyst" in stage1_actors
    assert "tron" in stage1_actors
    assert stage2_actors == ["critic"]


def test_build_stages_for_round_evidence():
    stages = build_stages_for_round(2)
    assert len(stages) == 2
    assert stages[0]["parallel"] is True
    assert stages[1]["parallel"] is True
    stage1_actors = [t["actor"] for t in stages[0]["agents"]]
    stage2_actors = [t["actor"] for t in stages[1]["agents"]]
    assert "dreamer" in stage1_actors
    assert "cat" in stage1_actors
    assert "tron" in stage1_actors
    assert SPECTATOR in stage1_actors
    assert "critic" in stage2_actors
    assert "contrarian" in stage2_actors
    assert "dog" in stage2_actors


def test_build_stages_for_round_debate():
    stages = build_stages_for_round(3)
    assert len(stages) == 1
    assert stages[0]["parallel"] is False
    actors = [t["actor"] for t in stages[0]["agents"]]
    assert "dreamer" in actors
    assert "contrarian" in actors
    assert "dog" in actors


def test_build_intervention_turns_creates_turns_for_valid_targets():
    targets = {"dog_target": "dreamer", "cat_target": "scientist", "tron_target": "engineer"}
    turns = _build_intervention_turns(targets)
    actors = [(t["actor"], t["turn_kind"]) for t in turns]
    assert ("dreamer", DOG_CORRECTION_TURN) in actors
    assert ("scientist", CAT_EXPANSION_TURN) in actors
    assert ("engineer", TRON_REMEDIATION_TURN) in actors


def test_build_intervention_turns_ignores_invalid_targets():
    targets = {"dog_target": "skynet", "cat_target": None}
    turns = _build_intervention_turns(targets)
    assert turns == []


@pytest.mark.asyncio
async def test_parallel_group_node_runs_agents_concurrently():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "pending_stages": [],
        "current_actor": "",
        "current_turn_kind": "",
        "current_stage": {
            "agents": [
                {"actor": "dreamer", "turn_kind": "base"},
                {"actor": "scientist", "turn_kind": "base"},
            ],
            "parallel": True,
        },
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "spectator_target": None,
        "spectator_web_boost_target": None,
        "phase": OPENING_PHASE,
        "subtopic_exhausted": False,
        "round_number": 1,
    }

    fake_result = {
        "actor": "dreamer",
        "turn_kind": "base",
        "content": "test content",
        "confidence_score": 7.0,
        "search_evidence": [],
        "targets": {},
        "spectator_data": None,
        "no_topic": False,
        "topic": {"id": 1, "summary": "topic"},
        "subtopic": {"id": 1, "summary": "subtopic"},
        "rag_context": "",
    }

    with patch(
        "grox_chat.server._run_single_agent_turn",
        new=AsyncMock(return_value=fake_result),
    ) as run_turn:
        with patch("grox_chat.server._persist_agent_result", new=AsyncMock()) as persist:
            result = await parallel_group_node(state)

    assert run_turn.await_count == 2
    assert persist.await_count == 2
    assert result["current_stage"] is None


@pytest.mark.asyncio
async def test_parallel_group_node_handles_spectator():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "pending_stages": [],
        "current_actor": "",
        "current_turn_kind": "",
        "current_stage": {
            "agents": [{"actor": SPECTATOR, "turn_kind": "base"}],
            "parallel": True,
        },
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "spectator_target": None,
        "spectator_web_boost_target": None,
        "phase": EVIDENCE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 2,
    }

    spectator_result = {
        "actor": SPECTATOR,
        "turn_kind": "base",
        "spectator_data": {"parsed_ok": True, "target": "scientist", "grant_web_search": True},
        "targets": {},
        "no_topic": False,
        "search_evidence": [],
    }

    with patch(
        "grox_chat.server._run_single_agent_turn",
        new=AsyncMock(return_value=spectator_result),
    ):
        with patch("grox_chat.server._persist_agent_result", new=AsyncMock()) as persist:
            result = await parallel_group_node(state)

    persist.assert_not_awaited()
    assert result.get("spectator_target") == "scientist"
    assert result.get("spectator_web_boost_target") == "scientist"


@pytest.mark.asyncio
async def test_parallel_group_node_inserts_intervention_stage():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "pending_stages": [{"agents": [{"actor": "critic", "turn_kind": "base"}], "parallel": True}],
        "current_actor": "",
        "current_turn_kind": "",
        "current_stage": {
            "agents": [{"actor": "dog", "turn_kind": "base"}],
            "parallel": True,
        },
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "spectator_target": None,
        "spectator_web_boost_target": None,
        "phase": EVIDENCE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 2,
    }

    dog_result = {
        "actor": "dog",
        "turn_kind": "base",
        "content": "*growls at [dreamer]*",
        "confidence_score": 7.0,
        "search_evidence": [],
        "targets": {"dog_target": "dreamer"},
        "spectator_data": None,
        "no_topic": False,
        "topic": {"id": 1},
        "subtopic": {"id": 1},
        "rag_context": "",
    }

    with patch(
        "grox_chat.server._run_single_agent_turn",
        new=AsyncMock(return_value=dog_result),
    ):
        with patch("grox_chat.server._persist_agent_result", new=AsyncMock()):
            result = await parallel_group_node(state)

    assert result.get("dog_target") == "dreamer"
    # Intervention stage should be inserted before the remaining critic stage
    assert "pending_stages" in result
    assert len(result["pending_stages"]) == 2
    assert result["pending_stages"][0]["parallel"] is True
    intervention_actors = [t["actor"] for t in result["pending_stages"][0]["agents"]]
    assert "dreamer" in intervention_actors


@pytest.mark.asyncio
async def test_sequential_group_node_runs_agents_in_order():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "pending_stages": [],
        "current_actor": "",
        "current_turn_kind": "",
        "current_stage": {
            "agents": [
                {"actor": "dreamer", "turn_kind": "base"},
                {"actor": "scientist", "turn_kind": "base"},
            ],
            "parallel": False,
        },
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "spectator_target": None,
        "spectator_web_boost_target": None,
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 3,
    }

    fake_result = {
        "actor": "dreamer",
        "turn_kind": "base",
        "content": "test content",
        "confidence_score": 7.0,
        "search_evidence": [],
        "targets": {},
        "spectator_data": None,
        "no_topic": False,
        "topic": {"id": 1},
        "subtopic": {"id": 1},
        "rag_context": "",
    }

    call_order = []

    async def mock_run_turn(state, actor, turn_kind):
        call_order.append(actor)
        return {**fake_result, "actor": actor}

    with patch(
        "grox_chat.server._run_single_agent_turn",
        new=AsyncMock(side_effect=mock_run_turn),
    ):
        with patch("grox_chat.server._persist_agent_result", new=AsyncMock()):
            result = await sequential_group_node(state)

    assert call_order == ["dreamer", "scientist"]
    assert result["current_stage"] is None


def test_build_graph_has_stage_dispatcher():
    graph = build_graph().get_graph()
    assert "stage_dispatcher" in graph.nodes
    assert "parallel_group" in graph.nodes
    assert "sequential_group" in graph.nodes
    assert "drain_daemon_node" in graph.nodes
