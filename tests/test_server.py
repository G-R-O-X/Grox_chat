import pytest
from unittest.mock import AsyncMock, patch

from grox_chat.server import (
    CAT_EXPANSION_TURN,
    DEBATE_PHASE,
    DOG_CORRECTION_TURN,
    EVIDENCE_PHASE,
    OPENING_PHASE,
    TRON_REMEDIATION_TURN,
    route_after_final_librarian,
    _extract_target_from_content,
    _refresh_pending_turns_with_extras,
    _termination_policy_for_round,
    _normalize_fact_proposal_contract,
    _normalize_message_contract,
    audience_summary_node,
    build_audience_summary_prompt,
    audience_termination_check_node,
    build_base_turns_for_phase,
    build_extra_turns,
    build_turn_queue_for_round,
    expert_node,
    fact_proposer_node,
    final_fact_proposer_node,
    final_librarian_node,
    final_writer_node,
    get_phase_for_round,
    librarian_node,
    should_enable_web_search,
    writer_node,
)


@pytest.mark.asyncio
async def test_final_writer_node_skips_same_round_duplicate_pass():
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

    with patch("grox_chat.server.query_gemini_cli", new=AsyncMock()) as writer_query:
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
    writer_reply = '{"action":"post_message","content":"Writer critique"}'

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch("grox_chat.server.query_gemini_cli", new=AsyncMock(return_value=writer_reply)):
                        with patch("grox_chat.server.api.persist_message", new=AsyncMock(return_value=55)) as persist_message:
                            with patch("grox_chat.server.process_writer_output", new=AsyncMock()) as process_writer_output:
                                result = await writer_node(state)

    persist_message.assert_awaited_once()
    assert persist_message.await_args.args[:4] == (1, 1, "writer", "Writer critique")
    assert persist_message.await_args.kwargs["round_number"] == 2
    assert persist_message.await_args.kwargs["turn_kind"] == "writer_critique"
    process_writer_output.assert_not_awaited()
    assert result["last_writer_round"] == 2


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
        {"id": 1, "sender": "critic", "content": "claim", "msg_type": "standard", "confidence_score": 7.0},
    ]
    proposer_reply = '{"action":"propose_facts","facts":["Fact 1","Fact 2","Fact 3"]}'

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch("grox_chat.server.react_search_loop", new=AsyncMock(return_value=(proposer_reply, False))):
                        with patch("grox_chat.server.process_writer_output", new=AsyncMock(return_value=[1, 2])) as process_writer_output:
                            await fact_proposer_node(state)

    assert process_writer_output.await_args.kwargs["max_candidates"] == 2
    assert process_writer_output.await_args.args[2] is None
    assert process_writer_output.await_args.kwargs["structured_facts"] == ["Fact 1", "Fact 2", "Fact 3"]


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
        {"id": 1, "sender": "critic", "content": "claim", "msg_type": "standard", "confidence_score": 7.0},
    ]
    proposer_reply = '{"action":"propose_facts","facts":["Fact 1","Fact 2","Fact 3","Fact 4"]}'

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch("grox_chat.server.react_search_loop", new=AsyncMock(return_value=(proposer_reply, False))):
                        with patch("grox_chat.server.process_writer_output", new=AsyncMock(return_value=[1, 2, 3])) as process_writer_output:
                            await final_fact_proposer_node(state)

    assert process_writer_output.await_args.kwargs["max_candidates"] == 3


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
                with patch("grox_chat.server.api.get_messages", return_value=messages):
                    with patch("grox_chat.server.build_query_rag_context", new=AsyncMock(return_value=("RAG", False))):
                        with patch(
                            "grox_chat.server._query_librarian_review_text",
                            new=AsyncMock(return_value=('{"decision":"soften"}', "minimax")),
                        ):
                            with patch(
                                "grox_chat.server.query_gemini_cli",
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
async def test_audience_termination_is_suppressed_before_round_three():
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
    }

    with patch("grox_chat.server.query_with_fallback", new=AsyncMock()) as query_gemini_cli:
        with patch("grox_chat.server.api.post_message") as post_message:
            result = await audience_termination_check_node(state)

    assert result["subtopic_exhausted"] is False
    query_gemini_cli.assert_not_awaited()
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
        {"id": 9, "sender": "audience", "content": "Current summary", "msg_type": "summary", "confidence_score": None},
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
                            "grox_chat.server.query_with_fallback",
                            new=AsyncMock(return_value='{"is_done": false, "warning": "Move to new evidence."}'),
                        ):
                            with patch("grox_chat.server.api.post_message") as post_message:
                                result = await audience_termination_check_node(state)

    assert result["subtopic_exhausted"] is False
    post_message.assert_called_once_with(
        1,
        1,
        "audience",
        "Move to new evidence.",
        msg_type="warning",
        round_number=3,
        turn_kind="audience_warning",
    )


@pytest.mark.asyncio
async def test_audience_termination_degrades_open_when_all_model_fallbacks_fail():
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
        {"id": 9, "sender": "audience", "content": "Current summary", "msg_type": "summary", "confidence_score": None},
    ]

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.aget_embedding", new=AsyncMock(return_value=[0.1] * 384)):
                    with patch("grox_chat.server.api.search_messages_hybrid", return_value=[]):
                        with patch(
                            "grox_chat.server.query_with_fallback",
                            new=AsyncMock(side_effect=RuntimeError("all fallbacks failed")),
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

    with patch("grox_chat.server.query_with_fallback", new=AsyncMock()) as query_with_fallback:
        result = await audience_termination_check_node(state)

    assert result["subtopic_exhausted"] is True
    query_with_fallback.assert_not_awaited()


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
        {"id": 9, "sender": "audience", "content": "Current summary", "msg_type": "summary", "confidence_score": None},
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
                            "grox_chat.server.query_with_fallback",
                            new=AsyncMock(return_value='{"is_done": false}'),
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

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("", True))):
                    with patch("grox_chat.server.react_search_loop", new=AsyncMock(return_value=("plain text response", False))):
                        with patch("grox_chat.server.api.persist_message", new=AsyncMock()) as persist_message:
                            await expert_node(state)

    persist_message.assert_awaited_once()
    assert persist_message.await_args.kwargs["confidence_score"] == 2.5
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
        {"id": 1, "sender": "audience", "content": "Grounding brief", "msg_type": "standard", "confidence_score": None},
    ]

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch("grox_chat.server.query_minimax", new=AsyncMock(return_value=('{"action":"post_message","content":"Initial stance","confidence_score":7}', []))) as query_minimax:
                        with patch("grox_chat.server.react_search_loop", new=AsyncMock()) as react_search_loop:
                            with patch("grox_chat.server.api.persist_message", new=AsyncMock()) as persist_message:
                                await expert_node(state)

    react_search_loop.assert_not_awaited()
    query_minimax.assert_awaited_once()
    persist_message.assert_awaited_once()


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

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch("grox_chat.server.react_search_loop", new=AsyncMock(return_value=(contrarian_reply, False))):
                        with patch("grox_chat.server.api.persist_message", new=AsyncMock()) as persist_message:
                            await expert_node(state)

    persist_message.assert_awaited_once()
    assert persist_message.await_args.args[3] == "The group is overconfident about soft-skill universals."
    assert persist_message.await_args.kwargs["confidence_score"] == 6.0
    assert persist_message.await_args.kwargs["turn_kind"] == "base"


def test_phase_helpers_and_base_turns():
    assert get_phase_for_round(1) == OPENING_PHASE
    assert get_phase_for_round(2) == EVIDENCE_PHASE
    assert get_phase_for_round(3) == DEBATE_PHASE

    assert [turn["actor"] for turn in build_base_turns_for_phase(OPENING_PHASE)] == [
        "dreamer", "scientist", "engineer", "analyst", "critic", "tron"
    ]
    assert [turn["actor"] for turn in build_base_turns_for_phase(EVIDENCE_PHASE)] == [
        "dreamer", "scientist", "engineer", "analyst", "critic", "contrarian", "dog", "cat", "tron"
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


def test_normalize_message_contract_filters_non_string_facts():
    parsed = _normalize_message_contract(
        '{"action":"post_message","content":"Writer check","facts":[{"bad": 1}, 7, "Verified fact", "  "]}'
    )

    assert parsed["parsed_ok"] is True
    assert parsed["facts"] == ["Verified fact"]


def test_build_audience_summary_prompt_requires_per_agent_positions_first():
    state = {"round_number": 2, "phase": EVIDENCE_PHASE}
    topic = {"summary": "topic"}
    messages = [
        {"sender": "audience", "content": "brief", "msg_type": "standard", "confidence_score": None},
        {"sender": "dreamer", "content": "idea", "msg_type": "standard", "confidence_score": 7.0},
        {"sender": "scientist", "content": "check", "msg_type": "standard", "confidence_score": 6.0},
        {"sender": "writer", "content": "verify", "msg_type": "standard", "confidence_score": None},
    ]

    prompt = build_audience_summary_prompt(state, topic, messages)

    assert "AGENT POSITIONS:" in prompt
    assert "SYNTHESIS:" in prompt
    assert "OPEN QUESTIONS:" in prompt
    assert "dreamer, scientist, writer" in prompt


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
                    "grox_chat.server.query_with_fallback",
                    new=AsyncMock(side_effect=RuntimeError("all fallbacks failed")),
                ):
                    with patch("grox_chat.server.aget_embedding", new=AsyncMock(return_value=None)):
                        with patch("grox_chat.server.api.post_message", return_value=77) as post_message:
                            result = await audience_summary_node(state)

    assert result["latest_summary_msg_id"] == 77
    stored_summary = post_message.call_args.args[3]
    assert "AGENT POSITIONS:" in stored_summary
    assert "dreamer" in stored_summary
    assert "scientist" in stored_summary
