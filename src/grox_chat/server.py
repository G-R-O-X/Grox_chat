import asyncio
import json
import logging
import re
from typing import Iterable, Optional, Sequence

from langgraph.graph import END, START, StateGraph

from .graph import ChatState, TurnSpec, dispatcher_node, route_from_dispatcher
from . import api
from . import db
from .agents import (
    DELIBERATORS,
    SKYNET,
    SPECTATOR,
    VOTING_PARTICIPANTS,
    can_special_target,
    get_agent,
    get_agent_spec,
    ordinary_deliberators,
    voting_agents,
)
from .rag import assemble_rag_context, build_query_rag_context
from .tools import react_search_loop
from .minimax_client import query_minimax
from .llm_router import query_with_fallback
from .external.gemini_cli_client import query_gemini_cli, warmup_gemini_cli
from .prompts import PROMPTS
from .writer_processor import process_writer_output
from .librarian_processor import (
    apply_librarian_review,
    build_librarian_audit_message,
    parse_librarian_review,
)
from .embedding import aget_embedding

logger = logging.getLogger(__name__)

AGENTS = list(DELIBERATORS) + ['cat', 'dog', 'tron', SPECTATOR]
PARSER_FAILURE_CONFIDENCE = 2.5
DEGRADED_OPERATION_CONFIDENCE = 3.0
LOOP_WARNING_DISTANCE = 0.25
WRITER_FACT_LIMIT = 2
FINAL_WRITER_FACT_LIMIT = 3

OPENING_PHASE = "opening"
EVIDENCE_PHASE = "evidence"
DEBATE_PHASE = "debate"

BASE_TURN = "base"
TRON_REMEDIATION_TURN = "tron_remediation"
DOG_CORRECTION_TURN = "dog_correction"
CAT_EXPANSION_TURN = "cat_expansion"
WRITER_CRITIQUE_TURN = "writer_critique"
LIBRARIAN_AUDIT_TURN = "librarian_audit"
AUDIENCE_SUMMARY_TURN = "skynet_summary"
AUDIENCE_WARNING_TURN = "skynet_warning"

OPENING_ROSTER = ['dreamer', 'scientist', 'engineer', 'analyst', 'critic', 'tron']
FULL_ROSTER = ['dreamer', 'scientist', 'engineer', 'analyst', 'critic', 'contrarian', 'dog', 'cat', 'tron', SPECTATOR]
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
}

SUBTOPIC_CANDIDATE_COUNT = 4
SUBTOPIC_VOTE_CYCLE_LIMIT = 3
DECISION_PASS_RATIO = 2 / 3

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


def _parse_single_json_wrapper(text: str) -> Optional[dict]:
    stripped = (text or "").strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


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
    parsed = _parse_single_json_wrapper(raw_text) or extract_json(raw_text)
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


def _normalize_fact_proposal_contract(raw_text: str) -> dict:
    parsed = extract_json(raw_text)
    if isinstance(parsed, dict):
        action = parsed.get("action")
        raw_facts = parsed.get("facts")
        if action == "propose_facts" and isinstance(raw_facts, list):
            facts = [fact.strip() for fact in raw_facts if isinstance(fact, str) and fact.strip()]
            return {
                "parsed_ok": True,
                "facts": facts,
            }

    return {
        "parsed_ok": False,
        "facts": [],
    }


def _normalize_focus_contract(raw_text: str) -> dict:
    parsed = _parse_single_json_wrapper(raw_text) or extract_json(raw_text)
    if isinstance(parsed, dict):
        action = parsed.get("action")
        target = parsed.get("target")
        reason = parsed.get("reason")
        if action == "focus" and isinstance(target, str):
            normalized_target = _normalize_target_name(target)
            if normalized_target and can_special_target(normalized_target):
                return {
                    "parsed_ok": True,
                    "target": normalized_target,
                    "reason": reason.strip() if isinstance(reason, str) else "",
                    "grant_web_search": bool(parsed.get("grant_web_search", False)),
                }
    return {
        "parsed_ok": False,
        "target": None,
        "reason": "",
        "grant_web_search": False,
    }


def _decision_passes(yes_votes: int, total_votes: int) -> bool:
    if total_votes <= 0:
        return False
    return (yes_votes / total_votes) > DECISION_PASS_RATIO


def _build_vote_prompt(
    *,
    question: str,
    topic_summary: str,
    topic_detail: str,
    candidate_summary: Optional[str] = None,
    candidate_detail: Optional[str] = None,
    selected: Optional[Sequence[str]] = None,
    rejected: Optional[Sequence[str]] = None,
) -> str:
    selected_block = ", ".join(selected or []) or "none"
    rejected_block = ", ".join(rejected or []) or "none"
    lines = [
        f"Topic: {topic_summary}",
        f"Topic Detail: {topic_detail}",
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
            "TASK:",
            question,
            "Vote YES only if the candidate is materially useful for the topic and not redundant with already selected items.",
            "Vote NO if it is redundant, low-value, or off-topic.",
            'Reply with strict JSON: {"vote":"yes"} or {"vote":"no"}.',
        ]
    )
    return "\n".join(lines)


async def _run_votes(
    *,
    voters: Sequence[str],
    prompt: str,
    allow_web: bool = False,
) -> tuple[int, int, dict[str, bool]]:
    decisions: dict[str, bool] = {}
    yes_votes = 0
    for voter in voters:
        agent = get_agent(voter)
        try:
            decision = await agent.vote(prompt, allow_web=allow_web)
        except Exception:
            decision = False
        decisions[voter] = decision
        yes_votes += int(decision)
    return yes_votes, len(voters), decisions


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
    valid_targets = set(ordinary_deliberators())
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
    return phase, turns


def _replace_extra_turns(pending_turns: list[TurnSpec], extra_turns: list[TurnSpec]) -> list[TurnSpec]:
    base_turns = [turn for turn in pending_turns if turn.get("turn_kind", BASE_TURN) == BASE_TURN]
    return base_turns + extra_turns


def _refresh_pending_turns_with_extras(state: ChatState, updates: dict) -> None:
    phase = updates.get("phase") or state.get("phase", get_phase_for_round(state.get("round_number", 1)))
    if phase == OPENING_PHASE:
        return

    merged_state = dict(state)
    merged_state.update(updates)
    pending_turns = list(updates.get("pending_turns", state.get("pending_turns", [])))
    updates["pending_turns"] = _replace_extra_turns(pending_turns, build_extra_turns(merged_state))


def _clear_consumed_extra_target(turn_kind: str, updates: dict) -> None:
    if turn_kind == TRON_REMEDIATION_TURN:
        updates["tron_target"] = None
    elif turn_kind == DOG_CORRECTION_TURN:
        updates["dog_target"] = None
    elif turn_kind == CAT_EXPANSION_TURN:
        updates["cat_target"] = None


def _pending_extra_turns(state: ChatState) -> list[TurnSpec]:
    return [
        turn
        for turn in state.get("pending_turns", [])
        if turn.get("turn_kind", BASE_TURN) != BASE_TURN
    ]


def _termination_policy_for_round(round_number: int) -> tuple[str, str]:
    if round_number <= 3:
        return (
            "weak",
            "Use a weak close threshold. Bias toward continuing unless the core subtopic is clearly resolved and no concrete unresolved branch remains.",
        )
    if round_number <= 6:
        return (
            "medium",
            "Use a balanced close threshold. Close if another round is unlikely to add materially new evidence, a sharper claim, or a narrower unresolved branch.",
        )
    if round_number <= 9:
        return (
            "strong",
            "Use a strong close threshold. Bias toward closing unless you can name a specific unresolved branch or evidence gap worth another round.",
        )
    return ("forced", "Round 10 is a forced close.")


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
        "sender": SKYNET,
        "content": seed_content,
        "msg_type": "standard",
        "confidence_score": None,
    }]


def should_enable_web_search(state: ChatState, actor: str, turn_kind: str) -> bool:
    if actor == state.get("spectator_web_boost_target") and turn_kind == BASE_TURN:
        return True
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

    if actor == state.get("spectator_target") and turn_kind == BASE_TURN:
        additions.append(
            "You feel that someone is watching you. You are filled with determination. Focus on the single most decisive unresolved point and make this turn count."
        )

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
    elif actor == SPECTATOR:
        additions.append(
            "Do not debate directly. Select exactly one ordinary deliberator for a next-round focus boost."
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
    elif actor == SPECTATOR:
        task = (
            "Choose the one ordinary deliberator most likely to unlock the next round. "
            'Reply with JSON using this schema: {"action":"focus","target":"scientist","reason":"...","grant_web_search":true}.'
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
        "Format for normal turns: {\"action\": \"post_message\", \"content\": \"...\", \"confidence_score\": 8}"
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
        "\nTASK: Post a critique message based on the claims in the recent debate. "
        "Focus on weak reasoning, hallucination risk, overclaiming, missing evidence, or conceptual drift. "
        "Do not propose facts in this step. "
        "Reply with JSON using this schema: "
        "{\"action\": \"post_message\", \"content\": \"...\"}."
    )
    return f"{PROMPTS['writer']}\n\nContext:\n{prompt}"


def build_fact_proposer_prompt(
    state: ChatState,
    topic: dict,
    messages: list[dict],
    rag_context: str,
    max_facts: int,
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
        "\nTASK: Propose candidate facts for long-term memory using local context plus web research. "
        f"Return at most {max_facts} candidate facts. "
        "Only include specific, reusable, evidence-worthy factual claims. "
        "Do not include opinions, interpretations, or broad summaries. "
        "Reply with JSON using this schema: "
        "{\"action\": \"propose_facts\", \"facts\": [\"candidate fact 1\", \"candidate fact 2\"]}. "
        "Use an empty facts array when nothing should be proposed."
    )
    return f"{PROMPTS['fact_proposer']}\n\nContext:\n{prompt}"


def build_librarian_prompt(
    state: ChatState,
    topic: dict,
    subtopic: dict | None,
    candidate: dict,
    messages: list[dict],
    rag_context: str,
) -> str:
    prompt = (
        f"Round: {state.get('round_number', 1)}\n"
        f"Phase: {state.get('phase', get_phase_for_round(state.get('round_number', 1)))}\n"
        f"Topic: {topic['summary']}\n"
    )
    if subtopic:
        prompt += f"Subtopic: {subtopic['summary']}\n"
    prompt += (
        f"Candidate ID: {candidate['id']}\n"
        f"Candidate Fact: {candidate['candidate_text']}\n"
    )
    if rag_context:
        prompt += f"{rag_context}\n"
    prompt += "=== RECENT TRANSCRIPT ===\n"
    for message in messages:
        prompt += f"{_format_message_for_prompt(message)}\n"
    prompt += (
        "\nTASK: Verify whether this candidate fact should enter permanent memory. "
        "You MUST rely on both the local context above and web-grounded verification. "
        "Decision rules: accept if the claim is specific and supported; soften if the core idea is supportable but the wording is too broad, too absolute, or too strong; reject if unsupported, speculative, or merely interpretive. "
        "Absolute formulations such as `no evidence`, `always`, `never`, `proves`, or `definitively` must be softened or rejected unless the evidence explicitly supports them. "
        "Reply with STRICT JSON using this schema: "
        "{\"action\": \"review_fact\", \"decision\": \"accept|soften|reject\", \"reviewed_text\": \"...\", \"review_note\": \"...\", \"evidence_note\": \"...\", \"confidence_score\": 8}."
    )
    return f"{PROMPTS['librarian']}\n\nContext:\n{prompt}"


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
            and sender != SKYNET
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
    return f"{PROMPTS['skynet']}\n\nContext:\n{ctx}\n\n{task}"


def _build_degraded_audience_summary(state: ChatState, messages: list[dict]) -> str:
    participant_order = []
    seen = set()
    for message in messages:
        sender = message.get("sender")
        if (
            sender
            and sender != SKYNET
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


async def _run_writer_critique_pass(state: ChatState) -> dict:
    current_round = state.get("round_number", 1)
    if state.get("last_writer_round") == current_round:
        return {}

    logger.info("[writer] Writer analyzing round for critique...")
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
        resp_text = await query_gemini_cli(
            prompt,
            model="gemini-3.0-flash",
            temperature=0.7,
            max_tokens=8192,
            system_instruction=PROMPTS["writer"],
            use_google_search=False,
            enable_fallback=True,
        )
    except Exception as exc:
        logger.warning("[writer] All writer critique model fallbacks failed: %s", exc)
        return {"last_writer_round": current_round}

    parsed = _normalize_message_contract(resp_text)
    content = parsed["content"]
    await api.persist_message(
        state["topic_id"],
        state["subtopic_id"],
        'writer',
        content,
        round_number=current_round,
        turn_kind=WRITER_CRITIQUE_TURN,
    )
    return {"last_writer_round": current_round}


async def _run_fact_proposer_pass(state: ChatState, force: bool = False) -> dict:
    current_round = state.get("round_number", 1)
    if force:
        if state.get("last_final_fact_proposer_round") == current_round:
            return {}
    elif state.get("last_fact_proposer_round") == current_round:
        return {}

    logger.info("[fact_proposer] Proposing candidate facts from the round...")
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
    max_facts = FINAL_WRITER_FACT_LIMIT if force else WRITER_FACT_LIMIT
    prompt = build_fact_proposer_prompt(state, topic, standard_messages, rag_context, max_facts=max_facts)
    try:
        resp_text, _ = await react_search_loop(
            "fact_proposer",
            prompt,
            max_iter=2,
            system_prompt=PROMPTS["fact_proposer"],
        )
    except Exception as exc:
        logger.warning("[fact_proposer] MiniMax fact proposal failed: %s", exc)
        marker_key = "last_final_fact_proposer_round" if force else "last_fact_proposer_round"
        return {marker_key: current_round}

    if (resp_text or "").strip().startswith("Error:"):
        logger.warning("[fact_proposer] MiniMax fact proposal returned an error sentinel: %s", resp_text)
        marker_key = "last_final_fact_proposer_round" if force else "last_fact_proposer_round"
        return {marker_key: current_round}

    parsed = _normalize_fact_proposal_contract(resp_text)
    if not parsed["parsed_ok"]:
        logger.warning("[fact_proposer] Invalid fact proposal contract; skipping candidate creation.")
        marker_key = "last_final_fact_proposer_round" if force else "last_fact_proposer_round"
        return {marker_key: current_round}

    await process_writer_output(
        state["topic_id"],
        state["subtopic_id"],
        None,
        "",
        structured_facts=parsed["facts"],
        max_candidates=max_facts,
    )
    marker_key = "last_final_fact_proposer_round" if force else "last_fact_proposer_round"
    return {marker_key: current_round}


async def _query_librarian_review_text(prompt: str) -> tuple[str, str]:
    try:
        resp_text, _ = await react_search_loop(
            "librarian",
            prompt,
            max_iter=2,
            system_prompt=PROMPTS["librarian"],
        )
        if (resp_text or "").strip().startswith("Error:"):
            raise RuntimeError(resp_text.strip())
        return resp_text, "minimax"
    except Exception as exc:
        logger.warning("[librarian] MiniMax review failed, escalating to Gemini Flash + Google: %s", exc)

    resp_text = await query_gemini_cli(
        prompt,
        model="gemini-3.0-flash",
        temperature=0.7,
        max_tokens=8192,
        system_instruction=PROMPTS["librarian"],
        use_google_search=True,
        enable_fallback=True,
    )
    return resp_text, "gemini"


async def _run_librarian_pass(state: ChatState) -> dict:
    logger.info("[librarian] Reviewing pending fact candidates...")
    topic, subtopic = _load_context_entities(state)
    if not topic or not subtopic:
        return {}

    pending_candidates = api.get_pending_fact_candidates(state["topic_id"], state["subtopic_id"])
    if not pending_candidates:
        return {}

    messages = api.get_messages(state["topic_id"], subtopic_id=state["subtopic_id"], limit=12)
    recent_message_ids = [message["id"] for message in messages if "id" in message]
    review_results = []

    for candidate in pending_candidates:
        rag_context, _ = await build_query_rag_context(
            state["topic_id"],
            candidate["candidate_text"],
            exclude_ids=recent_message_ids,
        )
        prompt = build_librarian_prompt(state, topic, subtopic, candidate, messages, rag_context)
        try:
            resp_text, provider = await _query_librarian_review_text(prompt)
            try:
                review = parse_librarian_review(resp_text, candidate["candidate_text"])
            except Exception:
                if provider != "minimax":
                    raise
                logger.warning(
                    "[librarian] MiniMax review for candidate %s was not valid JSON/schema; retrying with Gemini Flash.",
                    candidate["id"],
                )
                resp_text = await query_gemini_cli(
                    prompt,
                    model="gemini-3.0-flash",
                    temperature=0.7,
                    max_tokens=8192,
                    system_instruction=PROMPTS["librarian"],
                    use_google_search=True,
                    enable_fallback=True,
                )
                review = parse_librarian_review(resp_text, candidate["candidate_text"])
            review_results.append(
                await apply_librarian_review(state["topic_id"], candidate, review)
            )
        except Exception as exc:
            logger.warning(
                "[librarian] Failed to review candidate %s; leaving pending: %s",
                candidate["id"],
                exc,
            )

    if not review_results:
        return {}

    audit_message = build_librarian_audit_message(review_results)
    await api.persist_message(
        state["topic_id"],
        state["subtopic_id"],
        "librarian",
        audit_message,
        round_number=state.get("round_number", 1),
        turn_kind=LIBRARIAN_AUDIT_TURN,
    )
    return {}

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

    updates = {"current_actor": "", "current_turn_kind": ""}

    if actor == SPECTATOR:
        parsed_focus = _normalize_focus_contract(resp_text)
        if parsed_focus["parsed_ok"]:
            updates["spectator_target"] = parsed_focus["target"]
            updates["spectator_web_boost_target"] = (
                parsed_focus["target"] if parsed_focus["grant_web_search"] else None
            )
        return updates

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
        round_number=state.get("round_number", 1),
        turn_kind=turn_kind,
    )

    # Peanut gallery targeting logic
    _clear_consumed_extra_target(turn_kind, updates)
    if turn_kind == BASE_TURN and actor == state.get("spectator_target"):
        updates["spectator_target"] = None
        updates["spectator_web_boost_target"] = None

    if turn_kind == BASE_TURN and actor == 'dog':
        target = _extract_target_from_content(content, actor)
        if target:
            updates['dog_target'] = target
    elif turn_kind == BASE_TURN and actor == 'cat':
        target = _extract_target_from_content(content, actor)
        if target:
            updates['cat_target'] = target
    elif turn_kind == BASE_TURN and actor == 'tron':
        target = _extract_target_from_content(content, actor)
        if target:
            updates['tron_target'] = target

    if any(key in updates for key in {"dog_target", "cat_target", "tron_target"}):
        _refresh_pending_turns_with_extras(state, updates)

    return updates

async def audience_summary_node(state: ChatState) -> dict:
    logger.info("[skynet] Summarizing round...")
    topic, _ = _load_context_entities(state)
    if not topic:
        return {}
    current_round = state.get("round_number", 1)
    messages = api.get_messages(state["topic_id"], subtopic_id=state["subtopic_id"], limit=20)

    prompt = build_audience_summary_prompt(state, topic, messages)
    try:
        resp_text = await query_with_fallback(
            prompt,
            model="gemini-3.0-flash",
            system_instruction=PROMPTS["skynet"],
            use_google_search=False,
            enable_fallback=True,
            fallback_role=SKYNET,
        )
    except Exception as exc:
        logger.warning("[skynet] Summary generation degraded after all model fallbacks failed: %s", exc)
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
            SKYNET,
            content,
            msg_type='summary',
            embedding=emb,
            round_number=current_round,
            turn_kind=AUDIENCE_SUMMARY_TURN,
        )
    else:
        msg_id = api.post_message(
            state["topic_id"],
            state["subtopic_id"],
            SKYNET,
            content,
            msg_type='summary',
            round_number=current_round,
            turn_kind=AUDIENCE_SUMMARY_TURN,
        )
        
    return {"latest_summary_msg_id": msg_id}

async def audience_termination_check_node(state: ChatState) -> dict:
    logger.info("[skynet] Checking termination and cyclicality...")
    current_round = state.get("round_number", 1)
    if current_round >= 10:
        logger.info("[skynet] Forcing subtopic close at round %s.", current_round)
        return {"subtopic_exhausted": True}
    if _pending_extra_turns(state):
        logger.info("[skynet] Deferring close because extra turns are still pending.")
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
                    if ps.get('distance', 1.0) <= LOOP_WARNING_DISTANCE:
                        loop_detected = True

    stage, stage_guidance = _termination_policy_for_round(current_round)
    decision_prompt = _build_vote_prompt(
        question=(
            "Should this subtopic be closed now? Vote YES only if the room has extracted enough value and another round is unlikely to add meaningful new evidence or a sharper unresolved branch. "
            f"Current termination guidance: {stage_guidance}"
        ),
        topic_summary=topic["summary"],
        topic_detail=ctx + historical_context,
        selected=None,
        rejected=None,
    )
    try:
        yes_votes, total_votes, _ = await _run_votes(
            voters=voting_agents(),
            prompt=decision_prompt,
            allow_web=False,
        )
        is_done = _decision_passes(yes_votes, total_votes)
    except Exception as exc:
        logger.warning("[skynet] Termination vote degraded open after vote failure: %s", exc)
        yes_votes, total_votes = 0, 0
        is_done = False
    warning_text = None

    if loop_detected and not is_done:
        if not warning_text:
            warning_text = (
                "System warning: this debate is revisiting prior conclusions. Bring new evidence, a narrower unresolved claim, or a different assumption next round."
            )
        api.post_message(
            state["topic_id"],
            state["subtopic_id"],
            SKYNET,
            warning_text,
            msg_type="warning",
            round_number=current_round,
            turn_kind=AUDIENCE_WARNING_TURN,
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
        "spectator_target": state.get("spectator_target"),
        "spectator_web_boost_target": state.get("spectator_web_boost_target"),
        "pending_fact_reviews_remaining": False,
    }

def close_subtopic_node(state: ChatState) -> dict:
    logger.info("Subtopic exhausted; returning control to topic graph.")
    return {}

async def writer_node(state: ChatState) -> dict:
    return await _run_writer_critique_pass(state)


async def final_writer_node(state: ChatState) -> dict:
    return await _run_writer_critique_pass(state)


async def fact_proposer_node(state: ChatState) -> dict:
    return await _run_fact_proposer_pass(state, force=False)


async def final_fact_proposer_node(state: ChatState) -> dict:
    return await _run_fact_proposer_pass(state, force=True)


async def librarian_node(state: ChatState) -> dict:
    return await _run_librarian_pass(state)


async def final_librarian_node(state: ChatState) -> dict:
    await _run_librarian_pass(state)
    pending_candidates = api.get_pending_fact_candidates(state["topic_id"], state["subtopic_id"])
    if pending_candidates:
        logger.warning(
            "[librarian] %s fact candidates remain pending; delaying subtopic close for another round.",
            len(pending_candidates),
        )
        return {"pending_fact_reviews_remaining": True, "subtopic_exhausted": False}
    return {"pending_fact_reviews_remaining": False}


def route_after_final_librarian(state: ChatState) -> str:
    if state.get("pending_fact_reviews_remaining"):
        return "setup_next_round"
    return "close_subtopic"

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
    builder.add_node("fact_proposer_node", fact_proposer_node)
    builder.add_node("librarian_node", librarian_node)
    builder.add_node("audience_summary_node", audience_summary_node)
    builder.add_node("audience_termination_check_node", audience_termination_check_node)
    builder.add_node("setup_next_round_node", setup_next_round_node)
    builder.add_node("final_writer_node", final_writer_node)
    builder.add_node("final_fact_proposer_node", final_fact_proposer_node)
    builder.add_node("final_librarian_node", final_librarian_node)
    builder.add_node("close_subtopic_node", close_subtopic_node)
    
    builder.add_edge("writer_node", "fact_proposer_node")
    builder.add_edge("fact_proposer_node", "librarian_node")
    builder.add_edge("librarian_node", "audience_summary_node")
    builder.add_edge("audience_summary_node", "audience_termination_check_node")
    
    builder.add_conditional_edges(
        "audience_termination_check_node", 
        route_after_round, 
        {"close_subtopic": "final_writer_node", "setup_next_round": "setup_next_round_node"}
    )
    
    builder.add_edge("setup_next_round_node", "dispatcher")
    builder.add_edge("final_writer_node", "final_fact_proposer_node")
    builder.add_edge("final_fact_proposer_node", "final_librarian_node")
    builder.add_conditional_edges(
        "final_librarian_node",
        route_after_final_librarian,
        {
            "setup_next_round": "setup_next_round_node",
            "close_subtopic": "close_subtopic_node",
        },
    )
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
        "spectator_target": None,
        "spectator_web_boost_target": None,
        "phase": phase,
        "subtopic_exhausted": False,
        "round_number": 1,
        "last_writer_round": None,
        "last_fact_proposer_round": None,
        "last_final_fact_proposer_round": None,
        "pending_fact_reviews_remaining": False,
    }
    return await graph.ainvoke(initial_state)


async def run_server_loop():
    db.init_db()
    from .master_graph import build_master_graph

    try:
        await asyncio.wait_for(warmup_gemini_cli(), timeout=45)
        logger.info("[GeminiCLI] Warmup completed.")
    except Exception as exc:
        logger.warning("[GeminiCLI] Warmup failed; continuing with lazy fallback: %s", exc)

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
