import json
import logging
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from . import api
from .embedding import aget_embedding
from .llm_router import query_with_fallback
from .server import run_subtopic_graph

logger = logging.getLogger(__name__)


class TopicState(TypedDict, total=False):
    topic_id: int
    plan_id: int
    current_subtopic_id: int
    next_action: str
    topic_complete: bool
    deferred: bool


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


async def ask_gemini_cli(system_prompt: str, context: str, role: str, model: str = "gemini-3.1-pro-preview") -> Dict[str, Any]:
    prompt = f"{system_prompt}\n\nHere is the context of the chatroom:\n{context}"
    logger.info(f"[{role}] Asking Gemini CLI ({model})...")

    try:
        output = await query_with_fallback(
            prompt=prompt,
            model=model,
            temperature=0.7,
            system_instruction=system_prompt,
            use_google_search=True,
            enable_fallback=True,
            fallback_role=role,
        )
        return _parse_json_object(output)
    except Exception as e:
        logger.error(f"[{role}] Gemini CLI exception: {e}")
        return {"error": str(e)}


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

    system_prompt = (
        "You are the Audience. Break the topic into an ordered list of subtopics. "
        "All JSON string values must be written in English only. "
        "Output strictly JSON using this schema: "
        "{\"action\":\"create_plan\",\"subtopics\":[{\"summary\":\"...\",\"detail\":\"...\"}]}"
    )
    context = f"Topic: {topic['summary']}\nDetail: {topic['detail']}"
    data = await ask_gemini_cli(system_prompt, context, "audience")

    subtopics = data.get("subtopics", []) if isinstance(data, dict) else []
    if not subtopics:
        subtopics = [{"summary": topic["summary"], "detail": topic["detail"]}]

    plan_id = api.create_plan(state["topic_id"], json.dumps(subtopics), current_index=0)
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
        "You are the Audience. Create a detailed grounding brief for the next subtopic. "
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
    data = await ask_gemini_cli(system_prompt, context, "audience")
    brief_content = data.get("content") if isinstance(data, dict) else None
    if not brief_content:
        brief_content = f"Grounding Brief: {next_subtopic['detail']}"

    subtopic_id = api.create_subtopic(state["topic_id"], next_subtopic["summary"], next_subtopic["detail"])
    start_msg_id = await api.persist_message(state["topic_id"], subtopic_id, "audience", brief_content)
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
        "You are the Audience. Write the final conclusion for this completed subtopic. "
        "All JSON string values must be written in English only. "
        "Output strictly JSON using this schema: "
        "{\"action\":\"close_subtopic\",\"content\":\"final conclusion\"}"
    )
    data = await ask_gemini_cli(system_prompt, ctx, "audience", model="gemini-3.0-flash")
    conclusion = data.get("content") if isinstance(data, dict) else None
    if not conclusion:
        conclusion = f"Subtopic '{subtopic['summary']}' exhausted."

    emb = await aget_embedding(conclusion)
    if emb:
        api.insert_message_with_embedding(
            state["topic_id"],
            state["current_subtopic_id"],
            "audience",
            conclusion,
            msg_type="summary",
            embedding=emb,
        )
    else:
        api.post_message(state["topic_id"], state["current_subtopic_id"], "audience", conclusion, msg_type="summary")

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

    system_prompt = (
        "You are the Audience. Decide whether the topic needs more subtopics or should be closed. "
        "All JSON string values must be written in English only. "
        "Output strictly JSON using one of these schemas: "
        "{\"action\":\"create_plan\",\"subtopics\":[{\"summary\":\"...\",\"detail\":\"...\"}]}"
        " or "
        "{\"action\":\"close_topic\",\"content\":\"final topic summary\"}"
    )
    data = await ask_gemini_cli(system_prompt, ctx, "audience")
    action = data.get("action") if isinstance(data, dict) else None

    if isinstance(data, dict) and data.get("error"):
        logger.warning("[audience] Replan failed transiently: %s", data["error"])
        return {"deferred": True, "topic_complete": False, "next_action": "defer_topic"}

    if action == "create_plan" and data.get("subtopics"):
        plan_id = api.create_plan(state["topic_id"], json.dumps(data["subtopics"]), current_index=0)
        return {"plan_id": plan_id, "topic_complete": False, "next_action": "open_next_subtopic"}

    if action != "close_topic":
        logger.warning("[audience] Replan returned invalid output; deferring topic.")
        return {"deferred": True, "topic_complete": False, "next_action": "defer_topic"}

    final_summary = None
    if isinstance(data, dict):
        final_summary = data.get("content")
    if not final_summary:
        final_summary = f"Topic '{topic['summary']}' is complete."

    emb = await aget_embedding(final_summary)
    if emb:
        api.insert_message_with_embedding(
            state["topic_id"],
            None,
            "audience",
            final_summary,
            msg_type="summary",
            embedding=emb,
        )
    else:
        api.post_message(state["topic_id"], None, "audience", final_summary, msg_type="summary")
    api.set_topic_status(state["topic_id"], "Closed")
    return {"topic_complete": True, "next_action": "close_topic"}


def route_after_replan(state: TopicState) -> str:
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
    logger.info("[audience] Topic complete.")
    return {"topic_complete": True}


def defer_topic_node(state: TopicState) -> TopicState:
    logger.info("[audience] Topic deferred after transient orchestration failure.")
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
    builder.add_edge("generate_plan", "open_next_subtopic")
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
