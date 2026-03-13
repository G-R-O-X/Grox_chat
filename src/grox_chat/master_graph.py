import json
import logging
from typing import Any, Dict, List, TypedDict as TypingTypedDict

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from . import api
from .agents import SKYNET, get_agent, voting_agents
from .broker import (
    PROFILE_GEMINI_FLASH,
    PROFILE_GEMINI_PRO,
    llm_call_with_web,
)
from .embedding import aget_embedding
from .server import run_subtopic_graph
from .structured_retry import retry_structured_output, usable_text_output

logger = logging.getLogger(__name__)
SUBTOPIC_CANDIDATE_COUNT = 3
SUBTOPIC_VOTE_CYCLE_LIMIT = 3
DECISION_PASS_RATIO = 2 / 3


def _is_usable_json_text(text: str) -> bool:
    if not usable_text_output(text):
        return False
    try:
        return isinstance(_parse_json_object(text), dict)
    except Exception:
        return False


class TopicState(TypedDict, total=False):
    topic_id: int
    plan_id: int
    current_subtopic_id: int
    next_action: str
    topic_complete: bool
    deferred: bool


class VoteTally(TypingTypedDict):
    yes_votes: int
    successful_votes: int
    failed_votes: int


def _parse_json_object(output: str) -> Dict[str, Any]:
    import re

    match = re.search(r"\{.*\}", output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return json.loads(output)


def _parse_plan_content(plan: Dict[str, Any] | None) -> List[Dict[str, str]]:
    if not plan:
        return []
    try:
        content = json.loads(plan["content"])
        if isinstance(content, list):
            return [item for item in content if isinstance(item, dict)]
    except Exception:
        pass
    return []


def _sanitize_subtopics(raw_subtopics: Any, limit: int) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    if not isinstance(raw_subtopics, list):
        return cleaned

    for item in raw_subtopics:
        if not isinstance(item, dict):
            continue
        summary = item.get("summary")
        detail = item.get("detail")
        if not isinstance(summary, str) or not summary.strip():
            continue
        if not isinstance(detail, str) or not detail.strip():
            continue
        cleaned.append({"summary": summary.strip(), "detail": detail.strip()})
        if len(cleaned) >= limit:
            break
    return cleaned


async def ask_gemini_cli(
    system_prompt: str,
    context: str,
    role: str,
    model: str = "gemini-3.0-flash",
    *,
    topic_id: int = 0,
) -> Dict[str, Any]:
    prompt = f"{system_prompt}\n\nHere is the context of the chatroom:\n{context}"
    provider_profile = (
        PROFILE_GEMINI_PRO if "pro" in (model or "").lower() else PROFILE_GEMINI_FLASH
    )
    logger.info(
        "[%s] Starting orchestration call profile=%s model=%s allow_web=%s prompt_chars=%s context_chars=%s",
        role,
        provider_profile,
        model,
        True,
        len(prompt),
        len(context),
    )

    try:
        result = await retry_structured_output(
            stage_name=f"{role} orchestration",
            logger=logger,
            is_usable=lambda item: _is_usable_json_text(item.text),
            invoke=lambda: llm_call_with_web(
                prompt,
                provider_profile=provider_profile,
                role=role,
                require_json=True,
                model=model,
                search_budget=2,
                temperature=0.7,
                system_prompt=system_prompt,
                topic_id=topic_id,
            ),
        )
        if result is None:
            raise RuntimeError("orchestration structured retry exhausted")
        output = result.text
        logger.info(
            "[%s] Orchestration broker call succeeded provider_used=%s fallback_used=%s search_used=%s text_chars=%s; attempting JSON parse.",
            role,
            result.provider_used,
            result.fallback_used,
            result.search_used,
            len(output or ""),
        )
        parsed = _parse_json_object(output)
        logger.info(
            "[%s] Orchestration JSON parse succeeded keys=%s",
            role,
            sorted(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__,
        )
        return parsed
    except Exception as e:
        logger.error("[%s] Orchestration call failed: %s", role, e)
        return {"error": str(e)}


def _decision_passes(yes_votes: int, total_votes: int) -> bool:
    if total_votes <= 0:
        return False
    return (yes_votes / total_votes) > DECISION_PASS_RATIO


def _build_vote_prompt(
    *,
    topic: dict,
    question: str,
    candidate_summary: str = "",
    candidate_detail: str = "",
    selected: list[str] | None = None,
    rejected: list[str] | None = None,
) -> str:
    selected_block = ", ".join(selected or []) or "none"
    rejected_block = ", ".join(rejected or []) or "none"
    lines = [
        f"Topic: {topic['summary']}",
        f"Topic Detail: {topic['detail']}",
    ]
    if candidate_summary:
        lines.append(f"Candidate Subtopic: {candidate_summary}")
    if candidate_detail:
        lines.append(f"Candidate Detail: {candidate_detail}")
    lines.extend(
        [
            f"Already selected candidates: {selected_block}",
            f"Already rejected candidates: {rejected_block}",
            "",
            question,
            'Reply with strict JSON: {"vote":"yes|no","reason":"short sentence"}.',
        ]
    )
    return "\n".join(lines)


async def _collect_votes(
    prompt: str,
    *,
    topic_id: int,
    subtopic_id: int | None,
    round_number: int | None,
    vote_kind: str,
    subject: str,
) -> VoteTally:
    yes_votes = 0
    successful_votes = 0
    failed_votes = 0
    for voter in voting_agents():
        agent = get_agent(voter)
        try:
            payload = await agent.vote_detail(prompt, allow_web=False)
        except Exception as exc:
            logger.warning("[skynet] Vote execution failed for %s: %s", voter, exc)
            api.insert_vote_record(
                topic_id,
                subtopic_id,
                round_number,
                vote_kind,
                subject,
                prompt,
                voter,
                False,
                None,
                None,
                "",
                metadata_json=json.dumps({"invalid_reason": f"exception:{type(exc).__name__}"}),
            )
            failed_votes += 1
            continue
        if payload is None:
            logger.warning("[skynet] Vote from %s was invalid or malformed.", voter)
            api.insert_vote_record(
                topic_id,
                subtopic_id,
                round_number,
                vote_kind,
                subject,
                prompt,
                voter,
                False,
                None,
                None,
                "",
                metadata_json=json.dumps({"invalid_reason": "invalid_or_malformed"}),
            )
            failed_votes += 1
            continue
        successful_votes += 1
        decision = payload["decision"]
        yes_votes += int(decision)
        api.insert_vote_record(
            topic_id,
            subtopic_id,
            round_number,
            vote_kind,
            subject,
            prompt,
            voter,
            True,
            payload["decision_label"],
            payload["reason"],
            payload["raw_response"],
        )
    return {
        "yes_votes": yes_votes,
        "successful_votes": successful_votes,
        "failed_votes": failed_votes,
    }


async def _propose_subtopics(
    topic: dict,
    selected: list[str],
    rejected: list[str],
    unavailable: list[str] | None = None,
) -> dict[str, Any]:
    system_prompt = (
        "You are Skynet. Propose exactly 4 candidate subtopics for this topic. "
        "Avoid duplicating already selected, rejected, or completed subtopics. "
        "All JSON string values must be written in English only. "
        'Output strictly JSON using this schema: {"action":"create_plan","subtopics":[{"summary":"...","detail":"..."}]}.'
    )
    completed = unavailable or []
    context = (
        f"Topic: {topic['summary']}\n"
        f"Topic Detail: {topic['detail']}\n"
        f"Already selected candidates: {', '.join(selected) or 'none'}\n"
        f"Already rejected candidates: {', '.join(rejected) or 'none'}\n"
        f"Already completed subtopics: {', '.join(completed) or 'none'}"
    )
    data = await ask_gemini_cli(system_prompt, context, SKYNET, topic_id=topic["id"])
    if isinstance(data, dict) and data.get("error"):
        return {"candidates": [], "error": str(data["error"])}
    candidates = _sanitize_subtopics(data.get("subtopics", []) if isinstance(data, dict) else [], limit=SUBTOPIC_CANDIDATE_COUNT)
    seen = set(selected) | set(rejected) | set(completed)
    deduped: list[dict[str, str]] = []
    for candidate in candidates:
        summary = candidate["summary"]
        if summary in seen:
            continue
        seen.add(summary)
        deduped.append(candidate)
        if len(deduped) >= SUBTOPIC_CANDIDATE_COUNT:
            break
    return {"candidates": deduped, "error": None}


def node_inspect_topic_state(state: TopicState) -> TopicState:
    topic_id = state["topic_id"]
    active_plan = api.get_active_plan(topic_id)
    open_subtopic = api.get_open_subtopic(topic_id)

    if open_subtopic:
        return {
            "plan_id": active_plan["id"] if active_plan else 0,
            "current_subtopic_id": open_subtopic["id"],
            "next_action": "run_subtopic",
        }

    if active_plan:
        planned_subtopics = _parse_plan_content(active_plan)
        if active_plan.get("current_index", 0) < len(planned_subtopics):
            return {
                "plan_id": active_plan["id"],
                "next_action": "open_next_subtopic",
            }

    if api.get_current_subtopics(topic_id):
        return {"next_action": "replan_or_close"}

    return {"next_action": "generate_plan"}


def route_from_inspect(state: TopicState) -> str:
    return state.get("next_action", "generate_plan")


async def node_plan_generation(state: TopicState) -> TopicState:
    topic = api.get_topic(state["topic_id"])
    if not topic:
        return {"topic_complete": True, "next_action": "close_topic"}
    selected: list[dict[str, str]] = []
    rejected: list[str] = []

    for cycle in range(1, SUBTOPIC_VOTE_CYCLE_LIMIT + 1):
        proposal = await _propose_subtopics(
            topic,
            [item["summary"] for item in selected],
            rejected,
        )
        if proposal.get("error"):
            logger.warning("[skynet] Deferring topic after proposal-generation failure: %s", proposal["error"])
            return {"deferred": True, "topic_complete": False, "next_action": "defer_topic"}
        candidates = proposal["candidates"]
        if not candidates:
            continue
        for candidate in candidates:
            prompt = _build_vote_prompt(
                topic=topic,
                question=(
                    "Should this subtopic be admitted to the discussion plan? "
                    "Vote YES only if it materially helps resolve the topic and is not redundant with already selected subtopics."
                ),
                candidate_summary=candidate["summary"],
                candidate_detail=candidate["detail"],
                selected=[item["summary"] for item in selected],
                rejected=rejected,
            )
            tally = await _collect_votes(
                prompt,
                topic_id=topic["id"],
                subtopic_id=None,
                round_number=None,
                vote_kind="candidate_admission",
                subject=candidate["summary"],
            )
            if tally["failed_votes"] > 0:
                logger.warning("[skynet] Deferring topic after vote execution failures during plan generation.")
                return {"deferred": True, "topic_complete": False, "next_action": "defer_topic"}
            if _decision_passes(tally["yes_votes"], tally["successful_votes"]):
                selected.append(candidate)
                if len(selected) >= SUBTOPIC_CANDIDATE_COUNT:
                    break
            else:
                rejected.append(candidate["summary"])
        if len(selected) >= SUBTOPIC_CANDIDATE_COUNT:
            break

    if not selected:
        final_summary = (
            f"Topic '{topic['summary']}' is closed because the room could not reach basic consensus on any discussable subtopic after "
            f"{SUBTOPIC_VOTE_CYCLE_LIMIT} proposal cycles. Please restate or narrow the topic."
        )
        emb = await aget_embedding(final_summary)
        if emb:
            api.insert_message_with_embedding(
                state["topic_id"],
                None,
                SKYNET,
                final_summary,
                msg_type="summary",
                embedding=emb,
            )
        else:
            api.post_message(state["topic_id"], None, SKYNET, final_summary, msg_type="summary")
        api.set_topic_status(state["topic_id"], "Closed")
        return {"topic_complete": True, "next_action": "close_topic"}

    plan_id = api.create_plan(
        state["topic_id"],
        json.dumps(selected[:SUBTOPIC_CANDIDATE_COUNT]),
        current_index=0,
    )
    return {"plan_id": plan_id, "next_action": "open_next_subtopic"}


async def node_open_next_subtopic(state: TopicState) -> TopicState:
    topic = api.get_topic(state["topic_id"])
    plan = api.get_active_plan(state["topic_id"])
    if not topic or not plan:
        return {"next_action": "replan_or_close"}

    subtopics = _parse_plan_content(plan)
    plan_index = plan.get("current_index", 0)
    if plan_index >= len(subtopics):
        return {"next_action": "replan_or_close"}

    next_subtopic = subtopics[plan_index]
    system_prompt = (
        "You are Skynet. Create a detailed grounding brief for the next subtopic. "
        "All JSON string values must be written in English only. "
        "Output strictly JSON using this schema: "
        "{\"action\":\"post_message\",\"content\":\"grounding brief text\"}"
    )
    context = (
        f"Topic: {topic['summary']}\n"
        f"Topic Detail: {topic['detail']}\n"
        f"Subtopic: {next_subtopic['summary']}\n"
        f"Subtopic Detail: {next_subtopic['detail']}"
    )
    data = await ask_gemini_cli(system_prompt, context, SKYNET, topic_id=state["topic_id"])
    brief_content = data.get("content") if isinstance(data, dict) else None
    if not brief_content:
        brief_content = f"Grounding Brief: {next_subtopic['detail']}"

    subtopic_id = api.create_subtopic(state["topic_id"], next_subtopic["summary"], next_subtopic["detail"])
    start_msg_id = await api.persist_message(state["topic_id"], subtopic_id, SKYNET, brief_content)
    api.update_subtopic_start_msg(subtopic_id, start_msg_id)
    api.advance_plan_cursor(plan["id"])
    api.set_topic_status(state["topic_id"], "Running")

    return {
        "plan_id": plan["id"],
        "current_subtopic_id": subtopic_id,
        "next_action": "run_subtopic",
    }


async def node_run_subtopic(state: TopicState) -> TopicState:
    await run_subtopic_graph(
        state["topic_id"],
        state["current_subtopic_id"],
        plan_id=state.get("plan_id", 0),
    )
    return {}


async def node_conclude_subtopic(state: TopicState) -> TopicState:
    topic = api.get_topic(state["topic_id"])
    subtopic = api.get_subtopic(state["current_subtopic_id"])
    messages = api.get_messages(state["topic_id"], subtopic_id=state["current_subtopic_id"], limit=40)

    if not topic or not subtopic:
        return {}

    ctx = f"Topic: {topic['summary']}\nSubtopic: {subtopic['summary']}\n"
    for message in messages:
        ctx += f"[{message['sender']}]: {message['content'][:300]}\n"

    system_prompt = (
        "You are Skynet. Write the final conclusion for this completed subtopic. "
        "All JSON string values must be written in English only. "
        "Output strictly JSON using this schema: "
        "{\"action\":\"close_subtopic\",\"content\":\"final conclusion\"}"
    )
    data = await ask_gemini_cli(
        system_prompt,
        ctx,
        SKYNET,
        model="gemini-3.0-flash",
        topic_id=state["topic_id"],
    )
    conclusion = data.get("content") if isinstance(data, dict) else None
    if not conclusion:
        conclusion = f"Subtopic '{subtopic['summary']}' exhausted."

    emb = await aget_embedding(conclusion)
    if emb:
        api.insert_message_with_embedding(
            state["topic_id"],
            state["current_subtopic_id"],
            SKYNET,
            conclusion,
            msg_type="summary",
            embedding=emb,
        )
    else:
        api.post_message(state["topic_id"], state["current_subtopic_id"], SKYNET, conclusion, msg_type="summary")

    api.close_subtopic(state["current_subtopic_id"], conclusion)
    return {"current_subtopic_id": 0}


async def node_topic_replan_or_close(state: TopicState) -> TopicState:
    topic = api.get_topic(state["topic_id"])
    subtopics = api.get_current_subtopics(state["topic_id"])
    if not topic:
        return {"topic_complete": True}
    ctx = f"Topic: {topic['summary']}\nDetail: {topic['detail']}\n"
    for subtopic in subtopics:
        conclusion = subtopic.get("conclusion") or "(No conclusion recorded)"
        ctx += f"Subtopic: {subtopic['summary']}\nConclusion: {conclusion[:400]}\n"

    replan_vote_prompt = _build_vote_prompt(
        topic=topic,
        question=(
            "Should the room open additional subtopics for this topic? "
            "Vote YES only if the completed subtopics are still insufficient to support a final answer."
        ),
        candidate_summary="",
        candidate_detail=ctx,
    )
    tally = await _collect_votes(
        replan_vote_prompt,
        topic_id=topic["id"],
        subtopic_id=None,
        round_number=None,
        vote_kind="replan_gate",
        subject="Should the topic replan?",
    )
    if tally["failed_votes"] > 0:
        logger.warning("[skynet] Deferring topic after vote execution failures during replan gate.")
        return {"deferred": True, "topic_complete": False, "next_action": "defer_topic"}
    if not _decision_passes(tally["yes_votes"], tally["successful_votes"]):
        final_summary = f"Topic '{topic['summary']}' is complete."
        emb = await aget_embedding(final_summary)
        if emb:
            api.insert_message_with_embedding(
                state["topic_id"],
                None,
                SKYNET,
                final_summary,
                msg_type="summary",
                embedding=emb,
            )
        else:
            api.post_message(state["topic_id"], None, SKYNET, final_summary, msg_type="summary")
        api.set_topic_status(state["topic_id"], "Closed")
        return {"topic_complete": True, "next_action": "close_topic"}

    selected: list[dict[str, str]] = []
    rejected: list[str] = []
    completed_summaries = [item["summary"] for item in subtopics if isinstance(item.get("summary"), str)]
    for _cycle in range(1, SUBTOPIC_VOTE_CYCLE_LIMIT + 1):
        proposal = await _propose_subtopics(
            topic,
            [item["summary"] for item in selected],
            rejected,
            completed_summaries,
        )
        if proposal.get("error"):
            logger.warning("[skynet] Deferring topic after replanning proposal-generation failure: %s", proposal["error"])
            return {"deferred": True, "topic_complete": False, "next_action": "defer_topic"}
        candidates = proposal["candidates"]
        if not candidates:
            continue
        for candidate in candidates:
            prompt = _build_vote_prompt(
                topic=topic,
                question=(
                    "Should this newly proposed subtopic be admitted during replanning? "
                    "Vote YES only if it adds needed coverage beyond what has already been completed."
                ),
                candidate_summary=candidate["summary"],
                candidate_detail=candidate["detail"],
                selected=[item["summary"] for item in selected],
                rejected=rejected,
            )
            tally = await _collect_votes(
                prompt,
                topic_id=topic["id"],
                subtopic_id=None,
                round_number=None,
                vote_kind="replan_admission",
                subject=candidate["summary"],
            )
            if tally["failed_votes"] > 0:
                logger.warning("[skynet] Deferring topic after vote execution failures during replanning.")
                return {"deferred": True, "topic_complete": False, "next_action": "defer_topic"}
            if _decision_passes(tally["yes_votes"], tally["successful_votes"]):
                selected.append(candidate)
                if len(selected) >= SUBTOPIC_CANDIDATE_COUNT:
                    break
            else:
                rejected.append(candidate["summary"])
        if len(selected) >= SUBTOPIC_CANDIDATE_COUNT:
            break

    if not selected:
        final_summary = f"Topic '{topic['summary']}' is complete."
        emb = await aget_embedding(final_summary)
        if emb:
            api.insert_message_with_embedding(
                state["topic_id"],
                None,
                SKYNET,
                final_summary,
                msg_type="summary",
                embedding=emb,
            )
        else:
            api.post_message(state["topic_id"], None, SKYNET, final_summary, msg_type="summary")
        api.set_topic_status(state["topic_id"], "Closed")
        return {"topic_complete": True, "next_action": "close_topic"}

    plan_id = api.create_plan(
        state["topic_id"],
        json.dumps(selected[:SUBTOPIC_CANDIDATE_COUNT]),
        current_index=0,
    )
    return {"plan_id": plan_id, "topic_complete": False, "next_action": "open_next_subtopic"}


def route_after_replan(state: TopicState) -> str:
    if state.get("deferred"):
        return "defer_topic"
    if state.get("topic_complete"):
        return "close_topic"
    return "open_next_subtopic"


def route_after_generate_plan(state: TopicState) -> str:
    if state.get("deferred"):
        return "defer_topic"
    if state.get("topic_complete"):
        return "close_topic"
    return "open_next_subtopic"


def route_after_open_next_subtopic(state: TopicState) -> str:
    if state.get("next_action") == "replan_or_close" or not state.get("current_subtopic_id"):
        return "replan_or_close"
    return "run_subtopic"


def close_topic_node(state: TopicState) -> TopicState:
    logger.info("[skynet] Topic complete.")
    return {"topic_complete": True}


def defer_topic_node(state: TopicState) -> TopicState:
    logger.info("[skynet] Topic deferred after transient orchestration failure.")
    return {"deferred": True, "topic_complete": False}


def build_master_graph():
    builder = StateGraph(TopicState)

    builder.add_node("inspect_topic_state", node_inspect_topic_state)
    builder.add_node("generate_plan", node_plan_generation)
    builder.add_node("open_next_subtopic", node_open_next_subtopic)
    builder.add_node("run_subtopic", node_run_subtopic)
    builder.add_node("conclude_subtopic", node_conclude_subtopic)
    builder.add_node("replan_or_close", node_topic_replan_or_close)
    builder.add_node("defer_topic", defer_topic_node)
    builder.add_node("close_topic", close_topic_node)

    builder.add_edge(START, "inspect_topic_state")
    builder.add_conditional_edges(
        "inspect_topic_state",
        route_from_inspect,
        {
            "generate_plan": "generate_plan",
            "open_next_subtopic": "open_next_subtopic",
            "run_subtopic": "run_subtopic",
            "replan_or_close": "replan_or_close",
        },
    )
    builder.add_conditional_edges(
        "generate_plan",
        route_after_generate_plan,
        {
            "open_next_subtopic": "open_next_subtopic",
            "defer_topic": "defer_topic",
            "close_topic": "close_topic",
        },
    )
    builder.add_conditional_edges(
        "open_next_subtopic",
        route_after_open_next_subtopic,
        {
            "run_subtopic": "run_subtopic",
            "replan_or_close": "replan_or_close",
        },
    )
    builder.add_edge("run_subtopic", "conclude_subtopic")
    builder.add_edge("conclude_subtopic", "inspect_topic_state")
    builder.add_conditional_edges(
        "replan_or_close",
        route_after_replan,
        {
            "open_next_subtopic": "open_next_subtopic",
            "defer_topic": "defer_topic",
            "close_topic": "close_topic",
        },
    )
    builder.add_edge("defer_topic", END)
    builder.add_edge("close_topic", END)

    return builder.compile()
