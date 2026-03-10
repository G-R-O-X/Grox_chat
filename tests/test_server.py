import pytest
from unittest.mock import AsyncMock, patch

from agent_chatroom.server import (
    CAT_EXPANSION_TURN,
    DEBATE_PHASE,
    DOG_CORRECTION_TURN,
    EVIDENCE_PHASE,
    OPENING_PHASE,
    TRON_REMEDIATION_TURN,
    _extract_target_from_content,
    _normalize_message_contract,
    audience_summary_node,
    build_audience_summary_prompt,
    audience_termination_check_node,
    build_base_turns_for_phase,
    build_extra_turns,
    expert_node,
    final_writer_node,
    get_phase_for_round,
    should_enable_web_search,
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

    with patch("agent_chatroom.server.query_with_fallback", new=AsyncMock()) as writer_query:
        result = await final_writer_node(state)

    assert result == {}
    writer_query.assert_not_awaited()


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

    with patch("agent_chatroom.server.query_with_fallback", new=AsyncMock()) as query_gemini_cli:
        with patch("agent_chatroom.server.api.post_message") as post_message:
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

    with patch("agent_chatroom.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("agent_chatroom.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("agent_chatroom.server.api.get_messages", return_value=messages):
                with patch("agent_chatroom.server.aget_embedding", new=AsyncMock(return_value=[0.1] * 384)):
                    with patch(
                        "agent_chatroom.server.api.search_messages_hybrid",
                        return_value=[{"id": 3, "content": "Past summary", "distance": 0.2}],
                    ):
                        with patch(
                            "agent_chatroom.server.query_with_fallback",
                            new=AsyncMock(return_value='{"is_done": false, "warning": "Move to new evidence."}'),
                        ):
                            with patch("agent_chatroom.server.api.post_message") as post_message:
                                result = await audience_termination_check_node(state)

    assert result["subtopic_exhausted"] is False
    post_message.assert_called_once_with(1, 1, "audience", "Move to new evidence.", msg_type="warning")


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

    with patch("agent_chatroom.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("agent_chatroom.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("agent_chatroom.server.api.get_messages", return_value=messages):
                with patch("agent_chatroom.server.aget_embedding", new=AsyncMock(return_value=[0.1] * 384)):
                    with patch("agent_chatroom.server.api.search_messages_hybrid", return_value=[]):
                        with patch(
                            "agent_chatroom.server.query_with_fallback",
                            new=AsyncMock(side_effect=RuntimeError("all fallbacks failed")),
                        ):
                            result = await audience_termination_check_node(state)

    assert result["subtopic_exhausted"] is False


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

    with patch("agent_chatroom.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("agent_chatroom.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("agent_chatroom.server.api.get_messages", return_value=messages):
                with patch("agent_chatroom.server.assemble_rag_context", new=AsyncMock(return_value=("", True))):
                    with patch("agent_chatroom.server.react_search_loop", new=AsyncMock(return_value=("plain text response", False))):
                        with patch("agent_chatroom.server.api.persist_message", new=AsyncMock()) as persist_message:
                            await expert_node(state)

    persist_message.assert_awaited_once()
    assert persist_message.await_args.kwargs["confidence_score"] == 2.5


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

    with patch("agent_chatroom.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("agent_chatroom.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("agent_chatroom.server.api.get_messages", return_value=messages):
                with patch("agent_chatroom.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch("agent_chatroom.server.query_minimax", new=AsyncMock(return_value=('{"action":"post_message","content":"Initial stance","confidence_score":7}', []))) as query_minimax:
                        with patch("agent_chatroom.server.react_search_loop", new=AsyncMock()) as react_search_loop:
                            with patch("agent_chatroom.server.api.persist_message", new=AsyncMock()) as persist_message:
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

    with patch("agent_chatroom.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("agent_chatroom.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("agent_chatroom.server.api.get_messages", return_value=messages):
                with patch("agent_chatroom.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch("agent_chatroom.server.react_search_loop", new=AsyncMock(return_value=(contrarian_reply, False))):
                        with patch("agent_chatroom.server.api.persist_message", new=AsyncMock()) as persist_message:
                            await expert_node(state)

    persist_message.assert_awaited_once()
    assert persist_message.await_args.args[3] == "The group is overconfident about soft-skill universals."
    assert persist_message.await_args.kwargs["confidence_score"] == 6.0


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

    with patch("agent_chatroom.server.api.get_topic", return_value=topic):
        with patch("agent_chatroom.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("agent_chatroom.server.api.get_messages", return_value=messages):
                with patch(
                    "agent_chatroom.server.query_with_fallback",
                    new=AsyncMock(side_effect=RuntimeError("all fallbacks failed")),
                ):
                    with patch("agent_chatroom.server.aget_embedding", new=AsyncMock(return_value=None)):
                        with patch("agent_chatroom.server.api.post_message", return_value=77) as post_message:
                            result = await audience_summary_node(state)

    assert result["latest_summary_msg_id"] == 77
    stored_summary = post_message.call_args.args[3]
    assert "AGENT POSITIONS:" in stored_summary
    assert "dreamer" in stored_summary
    assert "scientist" in stored_summary
