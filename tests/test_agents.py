from unittest.mock import AsyncMock, patch

import pytest

from grox_chat.agents import (
    SKYNET,
    SPECTATOR,
    can_special_target,
    get_agent,
    is_deliberator,
    is_npc,
    is_special,
    parse_vote_response,
    voting_agents,
)
from grox_chat.server import BASE_TURN, DEBATE_PHASE, EVIDENCE_PHASE, expert_node, should_enable_web_search


def test_agent_role_matrix_and_voters():
    assert SKYNET in voting_agents()
    assert SPECTATOR in voting_agents()
    assert "writer" not in voting_agents()
    assert "librarian" not in voting_agents()

    assert is_deliberator("dreamer") is True
    assert is_special("dog") is True
    assert is_special(SPECTATOR) is True
    assert is_npc("writer") is True
    assert is_npc("fact_proposer") is True


def test_special_roles_can_target_only_deliberators():
    assert can_special_target("dreamer") is True
    assert can_special_target("scientist") is True
    assert can_special_target("dog") is False
    assert can_special_target("cat") is False
    assert can_special_target("tron") is False
    assert can_special_target(SPECTATOR) is False
    assert can_special_target("writer") is False
    assert can_special_target("librarian") is False
    assert can_special_target(SKYNET) is False


@pytest.mark.asyncio
async def test_spectator_vote_uses_strict_json_contract():
    spectator = get_agent(SPECTATOR)
    with patch("grox_chat.agents.call_text", new=AsyncMock(return_value='{"vote":"yes"}')):
        assert await spectator.vote("Vote on this.", allow_web=True) is True


@pytest.mark.asyncio
async def test_spectator_turn_sets_next_round_focus_and_web_boost():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": SPECTATOR,
        "current_turn_kind": BASE_TURN,
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
    messages = [
        {"id": 1, "sender": SKYNET, "content": "Grounding brief", "msg_type": "standard", "confidence_score": None},
    ]

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch(
                        "grox_chat.server.call_text",
                        new=AsyncMock(return_value='{"action":"focus","target":"Scientist","reason":"best shot","grant_web_search":true}'),
                    ):
                        updates = await expert_node(state)

    assert updates["spectator_target"] == "scientist"
    assert updates["spectator_web_boost_target"] == "scientist"


@pytest.mark.asyncio
async def test_targeted_spectator_turn_gets_web_boost_and_clears_after_use():
    state = {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "scientist",
        "current_turn_kind": BASE_TURN,
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "spectator_target": "scientist",
        "spectator_web_boost_target": "scientist",
        "phase": DEBATE_PHASE,
        "subtopic_exhausted": False,
        "round_number": 3,
    }
    messages = [
        {"id": 1, "sender": SKYNET, "content": "summary", "msg_type": "summary", "confidence_score": None},
    ]

    assert should_enable_web_search(state, "scientist", BASE_TURN) is True

    with patch("grox_chat.server.api.get_topic", return_value={"id": 1, "summary": "topic", "detail": "detail"}):
        with patch("grox_chat.server.api.get_subtopic", return_value={"id": 1, "summary": "subtopic", "detail": "detail"}):
            with patch("grox_chat.server.api.get_messages", return_value=messages):
                with patch("grox_chat.server.assemble_rag_context", new=AsyncMock(return_value=("RAG", False))):
                    with patch(
                        "grox_chat.server.call_text",
                        new=AsyncMock(return_value='{"action":"post_message","content":"Focused answer","confidence_score":8}'),
                    ) as call_text:
                        with patch("grox_chat.server.api.persist_message", new=AsyncMock()):
                            updates = await expert_node(state)

    call_text.assert_awaited_once()
    assert updates["spectator_target"] is None
    assert updates["spectator_web_boost_target"] is None


def test_parse_vote_response_is_tri_state():
    assert parse_vote_response('{"vote":"yes"}') is True
    assert parse_vote_response('{"vote":"no"}') is False
    assert parse_vote_response('{"vote":"maybe"}') is None
    assert parse_vote_response("plain text drift") is None
