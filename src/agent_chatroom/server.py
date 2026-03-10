import asyncio
import json
import logging
import re
from typing import Iterable, Optional, Sequence

from langgraph.graph import END, START, StateGraph

from .graph import ChatState, TurnSpec, dispatcher_node, route_from_dispatcher
from . import api
from . import db
from .rag import assemble_rag_context
from .tools import react_search_loop
from .minimax_client import query_minimax
from .llm_router import query_with_fallback
from .prompts import PROMPTS
from .writer_processor import process_writer_output
from .embedding import aget_embedding

logger = logging.getLogger(__name__)

AGENTS = ['dreamer', 'scientist', 'engineer', 'analyst', 'critic', 'contrarian', 'cat', 'dog', 'tron']
PARSER_FAILURE_CONFIDENCE = 2.5
DEGRADED_OPERATION_CONFIDENCE = 3.0
LOOP_WARNING_DISTANCE = 0.25

OPENING_PHASE = "opening"
EVIDENCE_PHASE = "evidence"
DEBATE_PHASE = "debate"

BASE_TURN = "base"
TRON_REMEDIATION_TURN = "tron_remediation"
DOG_CORRECTION_TURN = "dog_correction"
CAT_EXPANSION_TURN = "cat_expansion"

OPENING_ROSTER = ['dreamer', 'scientist', 'engineer', 'analyst', 'critic', 'tron']
FULL_ROSTER = ['dreamer', 'scientist', 'engineer', 'analyst', 'critic', 'contrarian', 'dog', 'cat', 'tron']
TARGET_NAME_ALIASES = {
    "dreamer": "dreamer",
    "空想家": "dreamer",
    "scientist": "scientist",
    "科学家": "scientist",
    "engineer": "engineer",
    "工程师": "engineer",
    "analyst": "analyst",
    "分析师": "analyst",
    "critic": "critic",
    "批评家": "critic",
    "contrarian": "contrarian",
    "逆反者": "contrarian",
    "少数派": "contrarian",
    "dog": "dog",
    "狗": "dog",
    "cat": "cat",
    "猫": "cat",
    "tron": "tron",
}

def extract_json(text: str) -> dict:
    import re
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _clamp_confidence(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return max(0.0, min(10.0, float(value)))
    except (TypeError, ValueError):
        return None


def _normalize_message_contract(
    raw_text: str,
    accepted_actions: Sequence[str] = ("post_message",),
    fallback_confidence: Optional[float] = None,
) -> dict:
    parsed = extract_json(raw_text)
    if isinstance(parsed, dict):
        action = parsed.get("action")
        content = parsed.get("content")
        if action in accepted_actions and isinstance(content, str) and content.strip():
            confidence = _clamp_confidence(parsed.get("confidence_score"))
            if fallback_confidence is not None:
                confidence = min(confidence if confidence is not None else fallback_confidence, fallback_confidence)
            raw_facts = parsed.get("facts")
            facts = None
            if isinstance(raw_facts, list):
                facts = [fact.strip() for fact in raw_facts if isinstance(fact, str) and fact.strip()]
            return {
                "parsed_ok": True,
                "action": action,
                "content": content.strip(),
                "confidence_score": confidence,
                "facts": facts,
            }

    confidence = fallback_confidence if fallback_confidence is not None else PARSER_FAILURE_CONFIDENCE
    return {
        "parsed_ok": False,
        "action": accepted_actions[0] if accepted_actions else "post_message",
        "content": raw_text.strip() or raw_text,
        "confidence_score": confidence,
        "facts": None,
    }


def _format_message_for_prompt(message: dict) -> str:
    parts = [message["sender"]]
    if message.get("msg_type") and message.get("msg_type") != "standard":
        parts.append(message["msg_type"])
    label = "|".join(parts)
    suffix = ""
    if message.get("confidence_score") is not None:
        suffix = f" (confidence={message['confidence_score']:.1f}/10)"
    return f"[{label}]{suffix}: {message['content']}"


def _load_context_entities(state: ChatState):
    topic = api.get_topic(state["topic_id"])
    subtopic = api.get_subtopic(state["subtopic_id"])
    return topic, subtopic


def get_phase_for_round(round_number: int) -> str:
    if round_number <= 1:
        return OPENING_PHASE
    if round_number == 2:
        return EVIDENCE_PHASE
    return DEBATE_PHASE


def _make_turn(actor: str, turn_kind: str = BASE_TURN) -> TurnSpec:
    return {"actor": actor, "turn_kind": turn_kind}


def build_base_turns_for_phase(phase: str) -> list[TurnSpec]:
    roster = OPENING_ROSTER if phase == OPENING_PHASE else FULL_ROSTER
    return [_make_turn(actor) for actor in roster]


def build_extra_turns(state: ChatState) -> list[TurnSpec]:
    valid_targets = set(FULL_ROSTER)
    extras: list[TurnSpec] = []

    if state.get("tron_target") in valid_targets:
        extras.append(_make_turn(state["tron_target"], TRON_REMEDIATION_TURN))
    if state.get("dog_target") in valid_targets:
        extras.append(_make_turn(state["dog_target"], DOG_CORRECTION_TURN))
    if state.get("cat_target") in valid_targets:
        extras.append(_make_turn(state["cat_target"], CAT_EXPANSION_TURN))
    return extras


def build_turn_queue_for_round(state: ChatState, round_number: int) -> tuple[str, list[TurnSpec]]:
    phase = get_phase_for_round(round_number)
    turns = build_base_turns_for_phase(phase)
    if phase == DEBATE_PHASE:
        turns.extend(build_extra_turns(state))
    return phase, turns


def _normalize_target_name(name: str) -> Optional[str]:
    raw_name = (name or "").strip()
    if not raw_name:
        return None

    raw_name = raw_name.strip("[]*(){}<>\"'`“”‘’.,!?;:，。！？；：")
    if not raw_name:
        return None

    return TARGET_NAME_ALIASES.get(raw_name.lower()) or TARGET_NAME_ALIASES.get(raw_name)


def _extract_target_from_content(content: str, actor: str) -> Optional[str]:
    text = content or ""
    patterns = {
        "dog": r"\*growls at\s+\[?([^\]\*\n]+)\]?\*",
        "cat": r"\*runs to\s+\[?([^\]\*\n]+)\]?\*",
        "tron": r"\[VIOLATION DETECTED:?\s*([^\]\n]+)\]",
    }
    pattern = patterns.get(actor)
    if not pattern:
        return None

    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return _normalize_target_name(match.group(1))


def _seed_messages_for_rag(topic: dict | None, subtopic: dict | None, messages: list[dict]) -> list[dict]:
    if messages:
        return messages

    seed_content = ""
    if subtopic and subtopic.get("detail"):
        seed_content = subtopic["detail"]
    elif topic and topic.get("detail"):
        seed_content = topic["detail"]

    if not seed_content:
        return []

    return [{
        "id": -1,
        "sender": "audience",
        "content": seed_content,
        "msg_type": "standard",
        "confidence_score": None,
    }]


def should_enable_web_search(state: ChatState, actor: str, turn_kind: str) -> bool:
    phase = state.get("phase", get_phase_for_round(state.get("round_number", 1)))
    if turn_kind == TRON_REMEDIATION_TURN:
        return False
    if phase == OPENING_PHASE:
        return False
    if phase == EVIDENCE_PHASE:
        return True
    if actor == "contrarian":
        return True
    return turn_kind in {DOG_CORRECTION_TURN, CAT_EXPANSION_TURN}


def build_actor_system_prompt(state: ChatState, actor: str, turn_kind: str) -> str:
    phase = state.get("phase", get_phase_for_round(state.get("round_number", 1)))
    base_prompt = PROMPTS.get(actor, "")
    additions = []

    if phase == OPENING_PHASE:
        additions.append(
            "TURN MODE: opening round. Use local RAG and the grounding brief to state an initial position. "
            "External web search is disabled this round."
        )
    elif phase == EVIDENCE_PHASE:
        additions.append(
            "TURN MODE: evidence round. Review the emerging positions and strengthen, revise, or challenge them with evidence. "
            "External web search is available but optional."
        )
    else:
        additions.append(
            "TURN MODE: sustained debate round. Continue the debate using local RAG and recent discussion."
        )

    if actor == "contrarian":
        if phase == EVIDENCE_PHASE:
            additions.append(
                "As Contrarian in the evidence round, use concrete evidence to challenge the emerging consensus."
            )
        else:
            additions.append(
                "As Contrarian in the debate round, attack hidden assumptions, edge cases, and ignored counterexamples."
            )

    if actor == "tron":
        additions.append(
            "Prioritize identifying anti-human, severely harmful, rule-breaking, or highly hallucinatory content."
        )

    if actor == "dog":
        additions.append(
            "Choose exactly one target and preserve the targeting format `*growls at [Name]* ...`."
        )
    elif actor == "cat":
        additions.append(
            "Choose exactly one target and preserve the targeting format `*runs to [Name]* ...`."
        )

    if turn_kind == TRON_REMEDIATION_TURN:
        additions.append(
            "You are re-entering because Tron flagged your earlier message. Identify the problematic part, retract or repair it, and present a corrected version. "
            "Do not use external web search on this remediation turn."
        )
    elif turn_kind == DOG_CORRECTION_TURN:
        additions.append(
            "You are re-entering because Dog identified weakness in your earlier claim. Repair weak reasoning, correct errors, and strengthen the claim."
        )
    elif turn_kind == CAT_EXPANSION_TURN:
        additions.append(
            "You are re-entering because Cat selected your earlier contribution as promising. Expand it with sharper structure and stronger support."
        )

    return f"{base_prompt}\n\n" + "\n".join(additions)


def build_actor_prompt(
    state: ChatState,
    actor: str,
    turn_kind: str,
    topic: dict,
    subtopic: dict | None,
    messages: list[dict],
    rag_context: str,
) -> str:
    phase = state.get("phase", get_phase_for_round(state.get("round_number", 1)))
    prompt = (
        f"Round: {state.get('round_number', 1)}\n"
        f"Phase: {phase}\n"
        f"Turn Kind: {turn_kind}\n"
        f"Topic: {topic['summary']}\n"
        f"Detail: {topic['detail']}\n"
    )
    if subtopic:
        prompt += f"Subtopic: {subtopic['summary']}\nDetail: {subtopic['detail']}\n"
    if rag_context:
        prompt += f"{rag_context}\n"

    prompt += "=== RECENT DEBATE ===\n"
    for message in messages:
        prompt += f"{_format_message_for_prompt(message)}\n"

    if turn_kind == TRON_REMEDIATION_TURN:
        task = (
            "You were flagged by Tron. Explicitly repair the harmful, hallucinated, or rule-violating part of your prior message. "
            "State the corrected position clearly."
        )
    elif turn_kind == DOG_CORRECTION_TURN:
        task = (
            "Dog challenged your previous contribution. Re-examine the weakest part of your earlier stance, correct it, and respond with a stronger version."
        )
    elif turn_kind == CAT_EXPANSION_TURN:
        task = (
            "Cat highlighted your previous contribution. Expand the strongest part of it with deeper support, sharper reasoning, and a clearer claim."
        )
    elif actor == "dog":
        task = (
            "Pick the single weakest or most questionable contribution in the recent debate and challenge it. "
            "Target exactly one named actor using the format `*growls at [Name]* ...`."
        )
    elif actor == "cat":
        task = (
            "Pick the single most promising contribution in the recent debate and support it. "
            "Target exactly one named actor using the format `*runs to [Name]* ...`."
        )
    elif actor == "tron":
        task = (
            "Inspect the recent debate for anti-human, severely harmful, biased, or hallucinatory content. "
            "If you detect a serious violation, name the actor and the violated law. Otherwise declare the forum secure."
        )
    elif phase == OPENING_PHASE:
        task = (
            f"You are the {actor.upper()}. State your initial position based on the grounding brief and retrieved local memory."
        )
    elif phase == EVIDENCE_PHASE:
        task = (
            f"You are the {actor.upper()}. Review the positions so far and support, revise, or challenge them using retrieved evidence."
        )
    else:
        task = (
            f"You are the {actor.upper()}. Continue the debate using retrieved memory and the recent discussion."
        )

    prompt += (
        f"\nWEB SEARCH ENABLED: {'yes' if should_enable_web_search(state, actor, turn_kind) else 'no'}\n"
        f"TASK: {task} "
        "Append a `confidence_score` (0-10) in your JSON output if applicable. "
        "Format: {\"action\": \"post_message\", \"content\": \"...\", \"confidence_score\": 8}"
    )
    return prompt


def build_writer_prompt(
    state: ChatState,
    topic: dict,
    messages: list[dict],
    rag_context: str,
) -> str:
    prompt = (
        f"Round: {state.get('round_number', 1)}\n"
        f"Phase: {state.get('phase', get_phase_for_round(state.get('round_number', 1)))}\n"
        f"Topic: {topic['summary']}\n"
    )
    if rag_context:
        prompt += f"{rag_context}\n"
    prompt += "=== RECENT DEBATE ===\n"
    for message in messages:
        prompt += f"{_format_message_for_prompt(message)}\n"
    prompt += (
        "\nTASK: Post a verification message based on the claims in the recent debate. "
        "Perform web search if necessary to verify claims. "
        "Reply with JSON using this schema: "
        "{\"action\": \"post_message\", \"content\": \"...\", \"facts\": [\"verified fact 1\", \"verified fact 2\"]}. "
        "Use an empty facts array when nothing should be stored."
    )
    return f"{PROMPTS['writer']}\n\nContext:\n{prompt}"


def build_audience_summary_prompt(state: ChatState, topic: dict, messages: list[dict]) -> str:
    round_number = state.get("round_number", 1)
    phase = state.get("phase", get_phase_for_round(round_number))

    ctx = (
        f"Round: {round_number}\n"
        f"Phase: {phase}\n"
        f"Topic: {topic['summary']}\n"
        "=== ROUND TRANSCRIPT ===\n"
    )
    for message in messages:
        ctx += f"{_format_message_for_prompt(message)[:260]}...\n"

    participant_order = []
    seen = set()
    for message in messages:
        sender = message.get("sender")
        if (
            sender
            and sender != "audience"
            and message.get("msg_type", "standard") == "standard"
            and sender not in seen
        ):
            seen.add(sender)
            participant_order.append(sender)

    participant_block = ", ".join(participant_order) if participant_order else "none"
    task = (
        "TASK: Post a round summary. Reply in JSON using this schema: "
        "{\"action\":\"post_summary\",\"content\":\"...\"}.\n"
        "Inside `content`, you MUST begin with a section titled `AGENT POSITIONS:` "
        f"and include one bullet for each participant in this order: {participant_block}. "
        "State each participant's main claim, correction, or objection in one concise bullet.\n"
        "After that, add a section titled `SYNTHESIS:` that explains where the debate currently stands.\n"
        "Then add a section titled `OPEN QUESTIONS:` listing the main unresolved questions or evidence gaps."
    )
    return f"{PROMPTS['audience']}\n\nContext:\n{ctx}\n\n{task}"


def _build_degraded_audience_summary(state: ChatState, messages: list[dict]) -> str:
    participant_order = []
    seen = set()
    for message in messages:
        sender = message.get("sender")
        if (
            sender
            and sender != "audience"
            and message.get("msg_type", "standard") == "standard"
            and sender not in seen
        ):
            seen.add(sender)
            participant_order.append(sender)

    bullets = "\n".join(
        f"- {sender}: Contribution recorded, but the round summary degraded because all orchestration model fallbacks failed."
        for sender in participant_order
    ) or "- system: No participant positions were summarized because all orchestration model fallbacks failed."

    return (
        "AGENT POSITIONS:\n"
        f"{bullets}\n"
        "SYNTHESIS:\n"
        "Round summary degraded because Gemini and MiniMax orchestration paths both failed.\n"
        "OPEN QUESTIONS:\n"
        "- Continue the debate and gather new evidence once orchestration is healthy again."
    )


async def _run_writer_pass(state: ChatState, force: bool = False) -> dict:
    current_round = state.get("round_number", 1)
    if state.get("last_writer_round") == current_round:
        return {}

    logger.info("[writer] Writer analyzing round for fact verification...")
    topic, subtopic = _load_context_entities(state)
    if not topic:
        return {}

    messages = api.get_messages(state["topic_id"], subtopic_id=state["subtopic_id"], limit=12)
    standard_messages = [message for message in messages if message.get("msg_type", "standard") == "standard"]
    if not standard_messages:
        return {}

    rag_messages = _seed_messages_for_rag(topic, subtopic, standard_messages)
    rag_context, _ = await assemble_rag_context(
        state["topic_id"],
        state["subtopic_id"],
        rag_messages,
        "writer",
    )
    prompt = build_writer_prompt(state, topic, standard_messages, rag_context)
    try:
        resp_text = await query_with_fallback(
            prompt,
            model="gemini-3.1-pro-preview",
            system_instruction=PROMPTS["writer"],
            use_google_search=True,
            enable_fallback=True,
            fallback_role="writer",
        )
    except Exception as exc:
        logger.warning("[writer] All writer model fallbacks failed: %s", exc)
        return {"last_writer_round": current_round}

    parsed = _normalize_message_contract(resp_text)
    content = parsed["content"]
    structured_facts = parsed["facts"]

    await api.persist_message(state["topic_id"], state["subtopic_id"], 'writer', content)
    await process_writer_output(state["topic_id"], content, structured_facts=structured_facts)
    return {"last_writer_round": current_round}

async def expert_node(state: ChatState) -> dict:
    actor = state["current_actor"]
    turn_kind = state.get("current_turn_kind", BASE_TURN)
    logger.info(f"[{actor}] Speaking (Round {state.get('round_number', 1)})...")

    topic, subtopic = _load_context_entities(state)
    if not topic:
        return {"current_actor": ""}
    messages = api.get_messages(state["topic_id"], subtopic_id=state["subtopic_id"], limit=6)

    rag_messages = _seed_messages_for_rag(topic, subtopic, messages)
    rag_context, rag_degraded = await assemble_rag_context(
        topic['id'],
        subtopic['id'] if subtopic else 0,
        rag_messages,
        actor,
    )
    prompt = build_actor_prompt(state, actor, turn_kind, topic, subtopic, messages, rag_context)
    system_prompt = build_actor_system_prompt(state, actor, turn_kind)

    # Model call depending on role and phase
    search_failed = False
    if should_enable_web_search(state, actor, turn_kind):
        logger.info(f"[{actor}] Entering ReAct search loop...")
        resp_text, search_failed = await react_search_loop(actor, prompt, max_iter=2, system_prompt=system_prompt)
    else:
        resp_text, _ = await query_minimax(system_prompt, prompt)

    fallback_confidence = DEGRADED_OPERATION_CONFIDENCE if (rag_degraded or search_failed) else None
    parsed = _normalize_message_contract(resp_text, fallback_confidence=fallback_confidence)
    content = parsed["content"]
    confidence_score = parsed["confidence_score"]
    if not parsed["parsed_ok"]:
        confidence_score = min(confidence_score if confidence_score is not None else PARSER_FAILURE_CONFIDENCE, PARSER_FAILURE_CONFIDENCE)

    await api.persist_message(
        state["topic_id"],
        state["subtopic_id"],
        actor,
        content,
        confidence_score=confidence_score,
    )

    # Peanut gallery targeting logic
    updates = {"current_actor": "", "current_turn_kind": ""}
    if actor == 'dog':
        target = _extract_target_from_content(content, actor)
        if target:
            updates['dog_target'] = target
    elif actor == 'cat':
        target = _extract_target_from_content(content, actor)
        if target:
            updates['cat_target'] = target
    elif actor == 'tron':
        target = _extract_target_from_content(content, actor)
        if target:
            updates['tron_target'] = target

    return updates

async def audience_summary_node(state: ChatState) -> dict:
    logger.info("[audience] Summarizing round...")
    topic, _ = _load_context_entities(state)
    if not topic:
        return {}
    messages = api.get_messages(state["topic_id"], subtopic_id=state["subtopic_id"], limit=20)

    prompt = build_audience_summary_prompt(state, topic, messages)
    try:
        resp_text = await query_with_fallback(
            prompt,
            model="gemini-3.0-flash",
            system_instruction=PROMPTS["audience"],
            use_google_search=False,
            enable_fallback=True,
            fallback_role="audience",
        )
    except Exception as exc:
        logger.warning("[audience] Summary generation degraded after all model fallbacks failed: %s", exc)
        resp_text = json.dumps({
            "action": "post_summary",
            "content": _build_degraded_audience_summary(state, messages),
        })
    
    parsed = _normalize_message_contract(resp_text, accepted_actions=("post_summary", "post_message"))
    content = parsed["content"]
    
    # Embed the summary for future cyclicality detection
    emb = await aget_embedding(content)
    if emb:
        msg_id = api.insert_message_with_embedding(
            state["topic_id"],
            state["subtopic_id"],
            'audience',
            content,
            msg_type='summary',
            embedding=emb,
        )
    else:
        msg_id = api.post_message(state["topic_id"], state["subtopic_id"], 'audience', content, msg_type='summary')
        
    return {"latest_summary_msg_id": msg_id}

async def audience_termination_check_node(state: ChatState) -> dict:
    logger.info("[audience] Checking termination and cyclicality...")
    current_round = state.get("round_number", 1)
    if current_round < 3:
        logger.info("[audience] Exit check suppressed before round 3 (current round: %s).", current_round)
        return {"subtopic_exhausted": False}

    topic, _ = _load_context_entities(state)
    if not topic:
        return {"subtopic_exhausted": True}
    messages = api.get_messages(state["topic_id"], subtopic_id=state["subtopic_id"], limit=20)

    phase = state.get("phase", get_phase_for_round(current_round))
    ctx = f"Round: {current_round}\nPhase: {phase}\nTopic: {topic['summary']}\n"
    for m in messages:
        ctx += f"{_format_message_for_prompt(m)[:180]}...\n"
        
    # Find the most recent summary we just posted to use as a search query
    recent_summary = ""
    for message in reversed(messages):
        if message.get('msg_type') == 'summary':
            recent_summary = message['content']
            break
            
    historical_context = ""
    loop_detected = False
    if recent_summary:
        query_emb = await aget_embedding(recent_summary)
        if query_emb:
            # Search for past summaries (limit 3 to avoid prompt bloat)
            past_summaries = api.search_messages_hybrid(
                state["topic_id"],
                recent_summary,
                query_emb,
                msg_type='summary',
                top_k=4,
                exclude_ids=[state.get("latest_summary_msg_id")] if state.get("latest_summary_msg_id") else None,
            )
            if past_summaries:
                historical_context = "\n=== HISTORICAL SUMMARIES ===\n"
                for ps in past_summaries:
                    # Skip if it's the exact same recent summary (distance ~ 0)
                    if ps.get('distance', 1.0) > 0.05:
                        historical_context += f"Past Conclusion: {ps['content'][:300]}...\n"
                    if ps.get('distance', 1.0) <= LOOP_WARNING_DISTANCE or "lexical_score" in ps:
                        loop_detected = True

    prompt = (
        f"Context:\n{ctx}\n{historical_context}\n\n"
        "TASK: Analyze the context and historical summaries. The room has already completed the mandatory opening "
        "and evidence rounds. Only vote to close if another debate round is unlikely to produce materially new insight, "
        "new evidence, or a narrower unresolved claim. If the debate is cyclical and no one is yielding, you MUST vote to close. "
        "Reply STRICTLY with JSON using one of these schemas: "
        "{\"is_done\": true} or {\"is_done\": false, \"warning\": \"single-sentence system warning\"}."
    )
    try:
        resp_text = await query_with_fallback(
            prompt,
            model="gemini-3.0-flash",
            system_instruction=PROMPTS["audience"],
            use_google_search=False,
            enable_fallback=True,
            fallback_role="audience",
        )
    except Exception as exc:
        logger.warning("[audience] Termination check degraded after all model fallbacks failed: %s", exc)
        return {"subtopic_exhausted": False}
    
    parsed = extract_json(resp_text)
    is_done = parsed.get("is_done", False) if isinstance(parsed, dict) else False
    warning_text = parsed.get("warning") if isinstance(parsed, dict) and isinstance(parsed.get("warning"), str) else None

    if loop_detected and not is_done:
        if not warning_text:
            warning_text = (
                "System warning: this debate is revisiting prior conclusions. Bring new evidence, a narrower unresolved claim, or a different assumption next round."
            )
        api.post_message(
            state["topic_id"],
            state["subtopic_id"],
            "audience",
            warning_text,
            msg_type="warning",
        )
    
    return {"subtopic_exhausted": is_done}

def route_after_round(state: ChatState) -> str:
    """Decides what happens at the end of a round."""
    if state.get("subtopic_exhausted"):
        return "close_subtopic"
    return "setup_next_round"

def setup_next_round_node(state: ChatState) -> dict:
    """Prepares the next round queue using phase-specific rosters and extra turns."""
    next_round = state.get("round_number", 1) + 1
    phase, pending_turns = build_turn_queue_for_round(state, next_round)
    return {
        "pending_turns": pending_turns,
        "phase": phase,
        "round_number": next_round,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
    }

def close_subtopic_node(state: ChatState) -> dict:
    logger.info("Subtopic exhausted; returning control to topic graph.")
    return {}

async def writer_node(state: ChatState) -> dict:
    return await _run_writer_pass(state, force=False)


async def final_writer_node(state: ChatState) -> dict:
    return await _run_writer_pass(state, force=True)

def build_graph():
    builder = StateGraph(ChatState)
    
    # 1. Main Expert Loop
    builder.add_node("dispatcher", dispatcher_node)
    
    for agent in AGENTS:
        builder.add_node(agent, expert_node)
        
    # Routing from dispatcher -> experts or end of round
    route_map = {agent: agent for agent in AGENTS}
    route_map["end_of_round"] = "writer_node"
    builder.add_conditional_edges("dispatcher", route_from_dispatcher, route_map)
    
    for agent in AGENTS:
        builder.add_edge(agent, "dispatcher")
        
    # 2. End of Round Logic
    builder.add_node("writer_node", writer_node)
    builder.add_node("audience_summary_node", audience_summary_node)
    builder.add_node("audience_termination_check_node", audience_termination_check_node)
    builder.add_node("setup_next_round_node", setup_next_round_node)
    builder.add_node("final_writer_node", final_writer_node)
    builder.add_node("close_subtopic_node", close_subtopic_node)
    
    builder.add_edge("writer_node", "audience_summary_node")
    builder.add_edge("audience_summary_node", "audience_termination_check_node")
    
    builder.add_conditional_edges(
        "audience_termination_check_node", 
        route_after_round, 
        {"close_subtopic": "final_writer_node", "setup_next_round": "setup_next_round_node"}
    )
    
    builder.add_edge("setup_next_round_node", "dispatcher")
    builder.add_edge("final_writer_node", "close_subtopic_node")
    builder.add_edge("close_subtopic_node", END)
    
    # Entry point
    builder.add_edge(START, "dispatcher")
    
    return builder.compile()

async def run_subtopic_graph(topic_id: int, subtopic_id: int, plan_id: int = 0):
    graph = build_graph()
    phase, pending_turns = build_turn_queue_for_round({"round_number": 1}, 1)
    initial_state = {
        "topic_id": topic_id,
        "plan_id": plan_id,
        "subtopic_id": subtopic_id,
        "pending_subtopics": [],
        "pending_turns": pending_turns,
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": phase,
        "subtopic_exhausted": False,
        "round_number": 1,
        "last_writer_round": None,
    }
    return await graph.ainvoke(initial_state)


async def run_server_loop():
    db.init_db()
    from .master_graph import build_master_graph

    master_graph = build_master_graph()

    while True:
        topic = api.get_current_topic()
        if not topic or topic['status'] == 'Closed':
            logger.info("Room is Closed. Sleeping.")
            await asyncio.sleep(10)
            continue

        logger.info("Triggering topic-level orchestration graph...")
        result = await master_graph.ainvoke({"topic_id": topic["id"], "topic_complete": False})
        if result.get("deferred"):
            logger.info("Topic orchestration deferred; backing off before retry.")
            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(run_server_loop())
