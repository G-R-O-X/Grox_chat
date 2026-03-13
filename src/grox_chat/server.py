import asyncio
import json
import logging
import re
from typing import Any, Optional, Sequence

from langgraph.graph import END, START, StateGraph

from .graph import ChatState, TurnSpec, dispatcher_node, route_from_dispatcher
from . import api
from . import db
from .agents import (
    DELIBERATORS,
    SKYNET,
    SPECTATOR,
    can_special_target,
    get_agent,
    get_agent_spec,
    ordinary_deliberators,
    special_agents,
    voting_agents,
)
from .broker import (
    SearchEvidenceItem,
    call_text,
    call_text_with_search_evidence,
    collect_search_evidence_bundle,
    is_gemini_enabled,
    shutdown_broker,
)
from .minimax_client import query_minimax
from .rag import assemble_rag_context, build_query_rag_context
from .external.gemini_cli_client import warmup_gemini_cli
from .logging_utils import configure_logging
from .prompts import PROMPTS
from .writer_processor import process_clerk_claim_output, process_writer_output
from .librarian_processor import (
    apply_claim_review,
    apply_librarian_review,
    build_librarian_audit_message,
    parse_claim_review,
    parse_librarian_review,
)
from .embedding import aget_embedding
from .structured_retry import retry_structured_output, usable_text_output

logger = logging.getLogger(__name__)

AGENTS = list(DELIBERATORS) + ['cat', 'dog', 'tron', SPECTATOR]
PARSER_FAILURE_CONFIDENCE = 2.5
DEGRADED_OPERATION_CONFIDENCE = 3.0
LOOP_WARNING_DISTANCE = 0.25
WRITER_FACT_LIMIT = 2
FINAL_WRITER_FACT_LIMIT = 3
BOOTSTRAP_FACT_DIRECTION_LIMIT = 3
INLINE_FACT_LIMIT = 1

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
TERMINATION_MAX_INVALID_VOTES = 2
ROUND3_CLOSE_RATIO = 0.8
ROUND46_CLOSE_RATIO = 2 / 3
ROUND79_CLOSE_RATIO = 0.6
SUMMARY_SECTION_HEADERS = (
    "TRAJECTORY:",
    "CONSENSUS:",
    "BLOCKERS:",
    "EVIDENCE GAPS:",
    "AGENT DELTAS:",
)
DELIBERATION_DISCIPLINE_LINES = (
    "DEBATE DISCIPLINE:",
    "- Prefer net-new argument, explicit correction, or narrowed disagreement over praise, empty agreement, or broad recap.",
    "- If you challenge a claim, identify the specific sentence, assumption, metric, or causal link you are attacking.",
)
WRITER_ANALYSIS_SYSTEM_PROMPT = """You are the Writer and a meta-Critic observing a multi-agent debate.
Your job in this pass is to analyze the round, not to produce final JSON output.

CRITICAL INSTRUCTION:
Write in English only.
Return concise plain text only.
Do not use markdown fences, thinking tags, or extra commentary outside the requested format.
"""
WRITER_STAGE_MAX_ATTEMPTS = 2
FACT_CITATION_PROTOCOL = (
    "KNOWLEDGE CITATION PROTOCOL:\n"
    "- Cite stored facts as `[F{id}]`.\n"
    "- Cite stored claims as `[C{id}]`.\n"
    "- Cite web evidence as `[W{id}]`, but describe it as unverified web evidence.\n"
    "- `[W...]` items may guide verification, but they are not permanent facts unless later admitted as `[F...]`.\n"
    "- Do not invent IDs.\n"
    "- Summaries and historical messages are context only. Do not cite them as evidence."
)
NUMBER_FACT_LIMIT = 3
FINAL_NUMBER_FACT_LIMIT = 4
SOURCED_FACT_LIMIT = 2
FINAL_SOURCED_FACT_LIMIT = 3
CLAIM_LIMIT = 2
FINAL_CLAIM_LIMIT = 3
NUMERIC_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z])(?:\$)?\d[\d,]*(?:\.\d+)?%?")
CITATION_ID_PATTERN = re.compile(r"\[(F|C|W)(\d+)\]")

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


def _extract_allowed_citation_ids(*knowledge_blocks: str) -> dict[str, set[int]]:
    allowed = {"F": set(), "C": set(), "W": set()}
    for block in knowledge_blocks:
        for prefix, raw_id in CITATION_ID_PATTERN.findall(block or ""):
            allowed[prefix].add(int(raw_id))
    return allowed


def _sanitize_citations_to_allowed_ids(
    content: str,
    *,
    knowledge_blocks: Sequence[str],
) -> tuple[str, dict[str, tuple[int, ...]]]:
    allowed = _extract_allowed_citation_ids(*knowledge_blocks)
    removed: dict[str, list[int]] = {"F": [], "C": [], "W": []}

    def _replace(match: re.Match[str]) -> str:
        prefix, raw_id = match.groups()
        citation_id = int(raw_id)
        if citation_id in allowed[prefix]:
            return match.group(0)
        removed[prefix].append(citation_id)
        return ""

    cleaned = CITATION_ID_PATTERN.sub(_replace, content or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"[ ]+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?])(?:\s*\1)+", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    deduped_removed = {
        prefix: tuple(dict.fromkeys(values))
        for prefix, values in removed.items()
    }
    return cleaned.strip(), deduped_removed


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


def _normalize_fact_direction_contract(raw_text: str) -> dict:
    parsed = _parse_single_json_wrapper(raw_text) or extract_json(raw_text)
    if isinstance(parsed, dict):
        action = parsed.get("action")
        raw_directions = parsed.get("directions")
        if action == "propose_fact_directions" and isinstance(raw_directions, list):
            directions = [direction.strip() for direction in raw_directions if isinstance(direction, str) and direction.strip()]
            return {"parsed_ok": True, "directions": directions}
    return {"parsed_ok": False, "directions": []}


def _normalize_focus_contract(raw_text: str) -> dict:
    parsed = _parse_single_json_wrapper(raw_text) or extract_json(raw_text)
    if isinstance(parsed, dict):
        action = parsed.get("action")
        target = parsed.get("target")
        reason = parsed.get("reason")
        if action == "focus" and isinstance(target, str):
            normalized_target = _normalize_target_name(target)
            if normalized_target and can_special_target(normalized_target):
                raw_grant = parsed.get("grant_web_search", False)
                if isinstance(raw_grant, bool):
                    grant_web_search = raw_grant
                else:
                    grant_web_search = False
                return {
                    "parsed_ok": True,
                    "target": normalized_target,
                    "reason": reason.strip() if isinstance(reason, str) else "",
                    "grant_web_search": grant_web_search,
                }
    return {
        "parsed_ok": False,
        "target": None,
        "reason": "",
        "grant_web_search": False,
    }


def _structured_message_is_usable(text: str, accepted_actions: Sequence[str] = ("post_message",)) -> bool:
    if not usable_text_output(text):
        return False
    parsed = _normalize_message_contract(text, accepted_actions=accepted_actions)
    return parsed.get("parsed_ok", False) and bool(parsed.get("content", "").strip())


def _fact_direction_output_is_usable(text: str) -> bool:
    if not usable_text_output(text):
        return False
    parsed = _normalize_fact_direction_contract(text)
    return parsed.get("parsed_ok", False)


def _fact_list_output_is_usable(text: str) -> bool:
    if not usable_text_output(text):
        return False
    parsed = _normalize_fact_proposal_contract(text)
    return parsed.get("parsed_ok", False)


def _normalize_clerk_fact_candidates_contract(raw_text: str) -> dict:
    parsed = _parse_single_json_wrapper(raw_text) or extract_json(raw_text)
    if isinstance(parsed, dict) and parsed.get("action") == "propose_fact_candidates":
        raw_candidates = parsed.get("fact_candidates")
        if isinstance(raw_candidates, list):
            candidates = []
            for item in raw_candidates:
                if not isinstance(item, dict):
                    continue
                candidate_text = item.get("candidate_text")
                source_excerpt = item.get("source_excerpt")
                source_refs = item.get("source_refs_json") or item.get("source_refs")
                if not isinstance(candidate_text, str) or not candidate_text.strip():
                    continue
                if not isinstance(source_excerpt, str) or not source_excerpt.strip():
                    continue
                if not isinstance(source_refs, list):
                    continue
                normalized_refs = [ref.strip() for ref in source_refs if isinstance(ref, str) and ref.strip()]
                if not normalized_refs:
                    continue
                candidates.append(
                    {
                        "candidate_text": candidate_text.strip(),
                        "candidate_type": "sourced_claim",
                        "source_refs": normalized_refs,
                        "source_excerpt": source_excerpt.strip(),
                    }
                )
            return {"parsed_ok": True, "fact_candidates": candidates}
    return {"parsed_ok": False, "fact_candidates": []}


def _fact_candidates_output_is_usable(text: str) -> bool:
    if not usable_text_output(text):
        return False
    parsed = _normalize_clerk_fact_candidates_contract(text)
    return parsed.get("parsed_ok", False)


def _normalize_clerk_claim_candidates_contract(raw_text: str) -> dict:
    parsed = _parse_single_json_wrapper(raw_text) or extract_json(raw_text)
    if isinstance(parsed, dict) and parsed.get("action") == "propose_claim_candidates":
        raw_candidates = parsed.get("claim_candidates")
        if isinstance(raw_candidates, list):
            candidates = []
            for item in raw_candidates:
                if not isinstance(item, dict):
                    continue
                candidate_text = item.get("candidate_text")
                rationale_short = item.get("rationale_short")
                support_fact_ids = item.get("support_fact_ids_json") or item.get("support_fact_ids")
                if not isinstance(candidate_text, str) or not candidate_text.strip():
                    continue
                if not isinstance(rationale_short, str) or not rationale_short.strip():
                    continue
                if not isinstance(support_fact_ids, list):
                    continue
                normalized_ids: list[int] = []
                for fact_id in support_fact_ids:
                    try:
                        normalized_ids.append(int(fact_id))
                    except (TypeError, ValueError):
                        continue
                if not normalized_ids:
                    continue
                candidates.append(
                    {
                        "candidate_text": candidate_text.strip(),
                        "support_fact_ids": normalized_ids,
                        "rationale_short": rationale_short.strip(),
                    }
                )
            return {"parsed_ok": True, "claim_candidates": candidates}
    return {"parsed_ok": False, "claim_candidates": []}


def _claim_candidates_output_is_usable(text: str) -> bool:
    if not usable_text_output(text):
        return False
    parsed = _normalize_clerk_claim_candidates_contract(text)
    return parsed.get("parsed_ok", False)


async def _call_text_with_structured_retry(
    *,
    stage_name: str,
    invoke,
    validator,
):
    return await retry_structured_output(
        stage_name=stage_name,
        invoke=invoke,
        is_usable=validator,
        logger=logger,
    )


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
        ]
    )
    if candidate_summary:
        lines.extend(
            [
                "Vote YES only if the candidate is materially useful for the topic and not redundant with already selected items.",
                "Vote NO if it is redundant, low-value, or off-topic.",
            ]
        )
    lines.append('Reply with strict JSON: {"vote":"yes|no","reason":"short sentence"}.')
    return "\n".join(lines)


async def _run_votes(
    *,
    voters: Sequence[str],
    prompt: str,
    allow_web: bool = False,
) -> tuple[int, int, int, dict[str, Optional[bool]]]:
    decisions: dict[str, Optional[bool]] = {}
    yes_votes = 0
    successful_votes = 0
    failed_votes = 0
    for voter in voters:
        agent = get_agent(voter)
        try:
            decision = await agent.vote(prompt, allow_web=allow_web)
        except Exception:
            decision = None
        decisions[voter] = decision
        if decision is None:
            failed_votes += 1
            continue
        successful_votes += 1
        yes_votes += int(decision)
    return yes_votes, successful_votes, failed_votes, decisions


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
            "EARLY STAGE. The burden of proof is on continuing. If any central blocker remains, or if the recommendation is still shifting, you should continue.",
        )
    if round_number <= 6:
        return (
            "medium",
            "MID STAGE. Close only when the remaining disagreement is peripheral or repetitive. Continue if the recommendation is still unstable or a central branch remains.",
        )
    if round_number <= 9:
        return (
            "strong",
            "LATE STAGE. The burden of proof is on closing, but you must continue if a severe central blocker still makes the current recommendation unstable or unsafe.",
        )
    return ("forced", "Round 10 is a forced close.")


def _should_run_termination_vote(round_number: int) -> bool:
    return round_number >= 3


def _build_termination_question(stage_guidance: str) -> str:
    return (
        "Decide whether this subtopic should CONTINUE or CLOSE.\n"
        f"Current stage guidance: {stage_guidance}\n"
        "Fill every field before you choose the final vote.\n"
        "Field rules:\n"
        "- `main_branch`: name the main unresolved branch; use `none` only if no meaningful blocker remains.\n"
        "- `centrality`: use `central`, `mixed`, `peripheral`, or `none`.\n"
        "- `recent_shift`: use `yes`, `no`, or `unclear` based on whether the framing, governing metric, or recommendation changed in the last 1-2 rounds.\n"
        "- `conditional_support`: use `yes` if the current recommendation still relies on softened, caveated, or weakly validated facts.\n"
        "- `untested_novelty`: use `yes` if a new framework, router, metric, or failure model affecting the recommendation has not yet been stress-tested.\n"
        "A subtopic can be legitimately CLOSED if it reaches ONE of the following end states:\n"
        "1. [Hard Consensus]: A logically sound conclusion supported by evidence with no remaining central blockers.\n"
        "2. [Constructive Suspension - Empirical Gap]: The team hits the boundary of non-executable constraints (e.g., requires physical testing), BUT has generated a rigorous, falsifiable [EXPERIMENTAL BLUEPRINT] (must include exact variables/metrics) to resolve the unknown.\n"
        "3. [Constructive Suspension - Trade-off]: The team hits an unresolvable value conflict and produces a detailed [DECISION MATRIX] showing under what specific conditions Path A or Path B should be chosen.\n"
        "Default voting policy:\n"
        "- If `centrality` is `central` or `mixed`, default to `continue`, UNLESS a Constructive Suspension state has been reached.\n"
        "- If `recent_shift` is `yes` or `unclear`, default to `continue`.\n"
        "- If `conditional_support` is `yes`, default to `continue`.\n"
        "- If `untested_novelty` is `yes`, default to `continue`.\n"
        "- If you vote `close`, `reason` MUST contain a short explanation of WHICH end state was reached (Hard Consensus, Empirical Gap, or Trade-off) and why.\n"
        "- If you vote `continue`, `reason` MUST explain what specific blocker or untested novelty remains.\n"
        'Reply with strict JSON only: {"main_branch":"...","centrality":"central|mixed|peripheral|none","recent_shift":"yes|no|unclear","conditional_support":"yes|no","untested_novelty":"yes|no","vote":"continue|close","reason":"... (Mandatory explanation for your vote)"}.'
    )


def _build_termination_vote_prompt(*, topic_summary: str, topic_detail: str, stage_guidance: str) -> str:
    return "\n".join(
        [
            f"Topic: {topic_summary}",
            f"Topic Detail: {topic_detail}",
            "",
            "TASK: SUBTOPIC CLOSURE GOVERNANCE",
            _build_termination_question(stage_guidance),
        ]
    )


def _build_termination_vote_repair_prompt(*, original_prompt: str, invalid_text: str, invalid_reason: str) -> str:
    return (
        "Original governance task:\n"
        f"{original_prompt}\n\n"
        "Invalid governance response:\n"
        f"{invalid_text}\n\n"
        f"Validation failure: {invalid_reason}\n\n"
        "Rewrite the response into valid JSON using exactly this schema:\n"
        '{"main_branch":"...","centrality":"central|mixed|peripheral|none","recent_shift":"yes|no|unclear","conditional_support":"yes|no","untested_novelty":"yes|no","vote":"continue|close","reason":"... (Mandatory)"}\n'
        "Preserve the original intent when possible.\n"
        "Output JSON only. Do not add markdown fences, commentary, or extra keys."
    )



async def _repair_summary_by_decomposition(flawed_content: str, provider: str = "minimax") -> str:
    logger.warning("[skynet] Summary missing headers. Initiating decomposition repair...")
    
    async def extract_section(header: str) -> str:
        prompt = (
            f"Here is a raw, unformatted summary of a debate. Your task is to extract ONLY the information "
            f"pertaining to the section '{header}'.\n"
            f"Re-write it so your output starts EXACTLY with '{header}'.\n"
            f"If the information is completely missing from the text, output exactly '{header}\nUnknown.'\n\n"
            f"Raw summary:\n{flawed_content}"
        )
        
        for attempt in range(2):
            try:
                resp = await call_text(
                    prompt,
                    provider=provider,
                    strategy="direct",
                    allow_web=False,
                    system_instruction="You are a strict text formatting assistant. Extract only what is requested. Output plain text, absolutely no markdown fences like ``` or ```json.",
                    model="MiniMax-M2.5" if provider == "minimax" else "gemini-3.0-flash",
                    fallback_role="skynet",
                    require_json=False,
                )
                if not resp:
                    continue
                    
                # Strip any markdown fences just in case
                resp = re.sub(r"^```[a-zA-Z]*\n|```$", "", resp.strip(), flags=re.MULTILINE).strip()
                
                # Check if it properly starts with the required header
                if not resp.startswith(header):
                    # Force the prefix if it generated useful content but forgot the header
                    if len(resp) > 20:
                        return f"{header}\n{resp}"
                    continue # Try again if it's completely malformed
                    
                return resp
            except Exception as e:
                logger.warning(f"[skynet] Attempt {attempt+1} failed to extract {header}: {e}")
                
        # Ultimate fallback if both attempts fail or return garbage
        return f"{header}\nUnknown."

    tasks = [extract_section(header) for header in SUMMARY_SECTION_HEADERS]
    results = await asyncio.gather(*tasks)
    return "\n\n".join(results)


def _has_required_summary_sections(content: str) -> bool:
    line_cursor = -1
    lines = (content or "").splitlines()
    for header in SUMMARY_SECTION_HEADERS:
        position = next(
            (index for index, line in enumerate(lines) if index > line_cursor and line.strip().startswith(header)),
            -1,
        )
        if position < 0:
            return False
        line_cursor = position
    return True


def _normalize_yes_no(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "y", "true", "1"}:
            return "yes"
        if normalized in {"no", "n", "false", "0"}:
            return "no"
    return None


def _normalize_centrality(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    aliases = {
        "central": "central",
        "core": "central",
        "mixed": "mixed",
        "peripheral": "peripheral",
        "secondary": "peripheral",
        "none": "none",
    }
    return aliases.get(normalized)


def _normalize_recent_shift(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        normalized = value.strip().lower()
        aliases = {
            "yes": "yes",
            "y": "yes",
            "true": "yes",
            "changed": "yes",
            "no": "no",
            "n": "no",
            "false": "no",
            "stable": "no",
            "unclear": "unclear",
            "unknown": "unclear",
            "maybe": "unclear",
        }
        return aliases.get(normalized)
    return None


def _normalize_termination_vote_label(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "close" if value else "continue"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"close", "yes", "approve"}:
            return "close"
        if normalized in {"continue", "no", "reject", "keep_open"}:
            return "continue"
    return None


def _empty_termination_vote(reason: str) -> dict[str, Any]:
    return {
        "parsed_ok": False,
        "main_branch": "none",
        "centrality": "none",
        "recent_shift": "unclear",
        "conditional_support": "no",
        "untested_novelty": "no",
        "vote": "continue",
        "reason": None,
        "central_blocker": False,
        "volatility_blocker": True,
        "support_blocker": False,
        "novelty_blocker": False,
        "invalid_reason": reason,
    }


def _normalize_termination_vote_contract(raw_text: str) -> dict[str, Any]:
    parsed = _parse_single_json_wrapper(raw_text) or extract_json(raw_text)
    if not isinstance(parsed, dict):
        return _empty_termination_vote("invalid_json")

    centrality = _normalize_centrality(parsed.get("centrality"))
    recent_shift = _normalize_recent_shift(parsed.get("recent_shift"))
    conditional_support = _normalize_yes_no(parsed.get("conditional_support"))
    untested_novelty = _normalize_yes_no(parsed.get("untested_novelty"))
    vote = _normalize_termination_vote_label(parsed.get("vote"))

    if not all((centrality, recent_shift, conditional_support, untested_novelty, vote)):
        return _empty_termination_vote("invalid_fields")

    raw_branch = parsed.get("main_branch")
    if isinstance(raw_branch, str) and raw_branch.strip():
        main_branch = raw_branch.strip()
    elif centrality == "none":
        main_branch = "none"
    else:
        main_branch = "unspecified"

    raw_override = parsed.get("override_reason")
    override_reason = raw_override.strip() if isinstance(raw_override, str) and raw_override.strip() else None

    central_blocker = centrality in {"central", "mixed"}
    volatility_blocker = recent_shift in {"yes", "unclear"}
    support_blocker = conditional_support == "yes"
    novelty_blocker = untested_novelty == "yes"
    has_blocker = central_blocker or volatility_blocker or support_blocker or novelty_blocker

    if vote == "close" and has_blocker and not override_reason:
        result = _empty_termination_vote("missing_override_reason")
        result.update(
            {
                "main_branch": main_branch,
                "centrality": centrality,
                "recent_shift": recent_shift,
                "conditional_support": conditional_support,
                "untested_novelty": untested_novelty,
                "vote": vote,
                "central_blocker": central_blocker,
                "volatility_blocker": volatility_blocker,
                "support_blocker": support_blocker,
                "novelty_blocker": novelty_blocker,
            }
        )
        return result

    return {
        "parsed_ok": True,
        "main_branch": main_branch,
        "centrality": centrality,
        "recent_shift": recent_shift,
        "conditional_support": conditional_support,
        "untested_novelty": untested_novelty,
        "vote": vote,
        "override_reason": override_reason,
        "central_blocker": central_blocker,
        "volatility_blocker": volatility_blocker,
        "support_blocker": support_blocker,
        "novelty_blocker": novelty_blocker,
        "invalid_reason": None,
    }


async def _run_termination_votes(
    *,
    voters: Sequence[str],
    prompt: str,
    topic_id: int,
    subtopic_id: int,
    round_number: int,
    subject: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for voter in voters:
        agent = get_agent(voter)
        raw_response = ""
        repair_response = ""
        repair_used = False
        try:
            raw_response = await agent.governance_vote(prompt)
            parsed = _normalize_termination_vote_contract(raw_response)
            if not parsed["parsed_ok"] and raw_response.strip():
                repair_used = True
                try:
                    repair_response, _ = await query_minimax(
                        system_prompt=(
                            f"{agent.spec.role_prompt}\n\n"
                            "GOVERNANCE JSON REPAIR MODE:\n"
                            "You are repairing a subtopic termination governance vote that failed validation.\n"
                            "Preserve the original intent when possible.\n"
                            "Output valid JSON only with the exact requested keys.\n"
                            "Do not add markdown fences, commentary, or extra keys."
                        ).strip(),
                        question=_build_termination_vote_repair_prompt(
                            original_prompt=prompt,
                            invalid_text=raw_response,
                            invalid_reason=parsed["invalid_reason"] or "unknown",
                        ),
                        temperature=0.1,
                        max_tokens=1024,
                    )
                    repaired = _normalize_termination_vote_contract(repair_response)
                    if repaired["parsed_ok"]:
                        parsed = repaired
                    else:
                        logger.warning(
                            "[GovVote] agent=%s repair failed invalid_reason=%s repair_response=%s",
                            voter,
                            repaired["invalid_reason"],
                            repair_response,
                        )
                except Exception as exc:
                    logger.warning("[GovVote] agent=%s repair failed with exception: %s", voter, exc)
        except Exception as exc:
            parsed = _empty_termination_vote(f"exception:{type(exc).__name__}")
            logger.warning("[GovVote] agent=%s execution failed: %s", voter, exc)
        record = {
            "voter": voter,
            "raw_response": raw_response,
            "repair_used": repair_used,
            "repair_response": repair_response,
            "parsed": parsed,
        }
        api.insert_vote_record(
            topic_id,
            subtopic_id,
            round_number,
            "termination",
            subject,
            prompt,
            voter,
            bool(parsed["parsed_ok"]),
            parsed["vote"] if parsed["parsed_ok"] else None,
            parsed["override_reason"] if parsed["parsed_ok"] else None,
            raw_response,
            metadata_json=json.dumps(
                {
                    "main_branch": parsed["main_branch"],
                    "centrality": parsed["centrality"],
                    "recent_shift": parsed["recent_shift"],
                    "conditional_support": parsed["conditional_support"],
                    "untested_novelty": parsed["untested_novelty"],
                    "override_reason": parsed["override_reason"],
                    "invalid_reason": parsed["invalid_reason"],
                    "repair_used": repair_used,
                    "repair_response": repair_response,
                },
                ensure_ascii=True,
            ),
        )
        logger.info(
            "[GovVote] agent=%s parsed_ok=%s vote=%s centrality=%s recent_shift=%s conditional_support=%s untested_novelty=%s override_reason=%s invalid_reason=%s repair_used=%s raw_response=%s repair_response=%s",
            voter,
            parsed["parsed_ok"],
            parsed["vote"],
            parsed["centrality"],
            parsed["recent_shift"],
            parsed["conditional_support"],
            parsed["untested_novelty"],
            parsed["override_reason"],
            parsed["invalid_reason"],
            repair_used,
            raw_response,
            repair_response,
        )
        records.append(record)
    return records


def _termination_thresholds_for_round(round_number: int) -> dict[str, float | int]:
    if round_number <= 3:
        return {
            "close_ratio": ROUND3_CLOSE_RATIO,
            "central_blocker": 1,
            "volatility_blocker": 1,
            "support_blocker": 1,
            "novelty_blocker": 1,
        }
    if round_number <= 6:
        return {
            "close_ratio": ROUND46_CLOSE_RATIO,
            "central_blocker": 2,
            "volatility_blocker": 2,
            "support_blocker": 2,
            "novelty_blocker": 2,
        }
    return {
        "close_ratio": ROUND79_CLOSE_RATIO,
        "central_blocker": 2,
        "volatility_blocker": 2,
        "support_blocker": 3,
        "novelty_blocker": 3,
    }


def _aggregate_termination_votes(vote_records: Sequence[dict[str, Any]], round_number: int) -> dict[str, Any]:
    valid_votes = [record["parsed"] for record in vote_records if record.get("parsed", {}).get("parsed_ok")]
    blocker_signal_votes = [
        record["parsed"]
        for record in vote_records
        if record.get("parsed", {}).get("parsed_ok")
        or record.get("parsed", {}).get("invalid_reason") == "missing_override_reason"
    ]
    invalid_votes = len(vote_records) - len(valid_votes)
    close_votes = sum(1 for parsed in valid_votes if parsed["vote"] == "close")
    close_ratio = close_votes / len(valid_votes) if valid_votes else 0.0
    blocker_counts = {
        "central_blocker": sum(1 for parsed in blocker_signal_votes if parsed["central_blocker"]),
        "volatility_blocker": sum(1 for parsed in blocker_signal_votes if parsed["volatility_blocker"]),
        "support_blocker": sum(1 for parsed in blocker_signal_votes if parsed["support_blocker"]),
        "novelty_blocker": sum(1 for parsed in blocker_signal_votes if parsed["novelty_blocker"]),
    }

    if invalid_votes > TERMINATION_MAX_INVALID_VOTES:
        return {
            "subtopic_exhausted": False,
            "valid_votes": len(valid_votes),
            "invalid_votes": invalid_votes,
            "close_votes": close_votes,
            "close_ratio": close_ratio,
            "blocker_counts": blocker_counts,
            "blocked_by": ["invalid_votes"],
        }

    thresholds = _termination_thresholds_for_round(round_number)
    blocked_by = [
        category
        for category, count in blocker_counts.items()
        if count >= thresholds[category]
    ]
    subtopic_exhausted = bool(valid_votes) and not blocked_by and close_ratio >= thresholds["close_ratio"]
    return {
        "subtopic_exhausted": subtopic_exhausted,
        "valid_votes": len(valid_votes),
        "invalid_votes": invalid_votes,
        "close_votes": close_votes,
        "close_ratio": close_ratio,
        "blocker_counts": blocker_counts,
        "blocked_by": blocked_by,
    }


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
    if actor == SPECTATOR:
        return False
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


def should_enable_web_backup(state: ChatState, actor: str, turn_kind: str) -> bool:
    if actor == SPECTATOR:
        return False
    if turn_kind != BASE_TURN:
        return False
    phase = state.get("phase", get_phase_for_round(state.get("round_number", 1)))
    return phase == DEBATE_PHASE


def build_actor_system_prompt(state: ChatState, actor: str, turn_kind: str) -> str:
    phase = state.get("phase", get_phase_for_round(state.get("round_number", 1)))
    base_prompt = PROMPTS.get(actor, "")
    additions = []

    additions.append(FACT_CITATION_PROTOCOL)

    if actor in ordinary_deliberators():
        additions.extend(DELIBERATION_DISCIPLINE_LINES)

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
        additions.append(
            "Do not just oppose the conclusion. Identify the hidden assumption the room is relying on and explain how the debate changes if it is false."
        )
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

    if actor == "analyst":
        additions.append(
            "Do not invent exact percentages, costs, latency figures, or synthetic scores unless they are explicitly grounded in accepted facts or provided evidence. Use variables, inequalities, or relative comparisons when data is missing."
        )
    elif actor == "scientist":
        additions.append(
            "You may identify empirical uncertainty, but you may not stop there. First give the strongest theoretical conclusion justified by current facts and first-principles reasoning, then state what remains empirically unresolved."
        )
    elif actor == "engineer":
        additions.append(
            "Do not introduce hybrid, tiered, or router-heavy architectures unless you first name the concrete failure mode they solve and why a simpler design is insufficient."
        )

    if actor == "dog":
        additions.append(
            "Choose exactly one target and preserve the targeting format `*growls at [Name]* ...`."
        )
        additions.append(
            "Prioritize logical pressure over roleplay volume. Hunt false precision, compromise by evasion, unsupported deferment, and missing causal links."
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
    *,
    latest_summary: str = "",
    include_output_contract: bool = True,
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
    if latest_summary:
        prompt += f"=== LATEST SUMMARY ===\n{latest_summary}\n"

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
        f"TASK: {task}"
    )
    if include_output_contract:
        prompt += (
            " Append a `confidence_score` (0-10) in your JSON output if applicable. "
            "Format for normal turns: {\"action\": \"post_message\", \"content\": \"...\", \"confidence_score\": 8}"
        )
    return prompt


def _build_writer_context_block(
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
    return prompt


def build_writer_diagnosis_prompt(
    state: ChatState,
    topic: dict,
    messages: list[dict],
    rag_context: str,
) -> str:
    prompt = _build_writer_context_block(state, topic, messages, rag_context)
    prompt += f"\n{FACT_CITATION_PROTOCOL}\n"
    prompt += (
        "\nTASK: Diagnose the 2-3 most consequential reasoning failures in the recent debate. "
        "Prefer issues such as false precision, premature compromise, empirical deferral, hidden assumptions, missing causal links, overclaiming, unsupported framing shifts, or conceptual drift. "
        "Output plain text only using this format:\n"
        "ISSUE 1: ...\n"
        "WHY IT MATTERS: ...\n"
        "ISSUE 2: ...\n"
        "WHY IT MATTERS: ...\n"
        "ISSUE 3: ...\n"
        "WHY IT MATTERS: ...\n"
        "If fewer than 3 issues matter, stop early."
    )
    return prompt


def build_writer_selection_prompt(
    state: ChatState,
    topic: dict,
    messages: list[dict],
    rag_context: str,
    diagnosis: str,
) -> str:
    prompt = _build_writer_context_block(state, topic, messages, rag_context)
    prompt += f"\n{FACT_CITATION_PROTOCOL}\n"
    prompt += (
        "\n=== WRITER DIAGNOSIS ===\n"
        f"{diagnosis.strip()}\n"
        "\nTASK: Select the single most central issue for the next critique message. "
        "Choose the issue that most affects the room's current recommendation, evidence quality, or closure readiness. "
        "You may name one secondary issue only if it directly sharpens the main point. "
        "Output plain text only using this format:\n"
        "PRIMARY ISSUE: ...\n"
        "WHY CENTRAL: ...\n"
        "SECONDARY ISSUE: ... or none"
    )
    return prompt


def build_writer_prompt(
    state: ChatState,
    topic: dict,
    messages: list[dict],
    rag_context: str,
    diagnosis: str = "",
    focus: str = "",
) -> str:
    prompt = _build_writer_context_block(state, topic, messages, rag_context)
    prompt += f"\n{FACT_CITATION_PROTOCOL}\n"
    if diagnosis.strip():
        prompt += f"\n=== WRITER DIAGNOSIS ===\n{diagnosis.strip()}\n"
    if focus.strip():
        prompt += f"\n=== SELECTED CENTRAL ISSUE ===\n{focus.strip()}\n"
    prompt += (
        "\nTASK: Post a critique message based on the claims in the recent debate. "
        "Center the critique on the selected primary issue. "
        "You may mention at most one secondary issue only if it directly sharpens the same critique. "
        "Focus on weak reasoning, hallucination risk, overclaiming, missing evidence, or conceptual drift. "
        "Do not propose facts, do not summarize the whole round, and do not judge whether the room should close. "
        "Reply with JSON using this schema: "
        "{\"action\": \"post_message\", \"content\": \"...\"}."
    )
    return prompt


def _writer_text_response_is_usable(text: str) -> bool:
    stripped = (text or "").strip()
    return bool(stripped) and not stripped.startswith("Error:")


def _writer_compose_response_is_usable(text: str) -> bool:
    parsed = _normalize_message_contract(text)
    return bool(parsed.get("parsed_ok")) and bool(parsed.get("content", "").strip())


async def _call_writer_stage_with_retry(
    *,
    stage_name: str,
    prompt: str,
    system_instruction: str,
    temperature: float,
    max_tokens: int,
    require_json: bool = False,
    validator=None,
) -> str:
    validator = validator or _writer_text_response_is_usable
    response = await retry_structured_output(
        stage_name=stage_name,
        logger=logger,
        attempts=WRITER_STAGE_MAX_ATTEMPTS,
        is_usable=validator,
        invoke=lambda: call_text(
            prompt,
            provider="minimax",
            strategy="direct",
            allow_web=False,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            fallback_role="writer",
            require_json=require_json,
        ),
    )
    return response or ""


def build_fact_proposer_prompt(
    state: ChatState,
    topic: dict,
    messages: list[dict],
    rag_context: str,
    max_facts: int,
    fact_stage: str = "synthesized",
    focus_label: str | None = None,
) -> str:
    prompt = (
        f"Round: {state.get('round_number', 1)}\n"
        f"Phase: {state.get('phase', get_phase_for_round(state.get('round_number', 1)))}\n"
        f"Topic: {topic['summary']}\n"
    )
    prompt += f"{FACT_CITATION_PROTOCOL}\n"
    prompt += (
        "Web evidence [W...] may be used as an unverified lead. "
        "Only promote a [W] lead into a fact candidate when you can restate it conservatively with explicit source refs and a short source excerpt.\n"
    )
    if focus_label:
        prompt += f"Fact Stage: {fact_stage}\nFocus: {focus_label}\n"
    if rag_context:
        prompt += f"{rag_context}\n"
    prompt += "=== RECENT DEBATE ===\n"
    for message in messages:
        prompt += f"{_format_message_for_prompt(message)}\n"
    if fact_stage == "bootstrap":
        prompt += (
            "\nTASK: From this single bootstrap fact direction and its web evidence, propose externally verifiable baseline facts. "
            f"Return at most {max_facts} candidate facts. "
            "Only include data-like, factual, reusable claims grounded in the cited search evidence. "
            "Do not include broad conclusions, opinions, or synthesis."
        )
    elif fact_stage == "inline":
        prompt += (
            "\nTASK: From this turn's web evidence, propose at most one immediately reusable hard fact for shared memory. "
            "Prefer a fact that later speakers could reuse instead of repeating the same search. "
            "Do not include opinions, interpretations, or broad summaries."
        )
    else:
        prompt += (
            "\nTASK: Propose candidate synthesized facts for long-term memory using local context plus web research when relevant. "
            f"Return at most {max_facts} candidate facts. "
            "Only include specific, reusable, evidence-worthy claims or cautious working conclusions. "
            "Do not include opinions or broad narrative summaries."
        )
    prompt += (
        " Reply with JSON using this schema: "
        "{\"action\": \"propose_facts\", \"facts\": [\"candidate fact 1\", \"candidate fact 2\"]}. "
        "Use an empty facts array when nothing should be proposed."
    )
    return f"{PROMPTS['fact_proposer']}\n\nContext:\n{prompt}"


def build_clerk_sourced_fact_prompt(
    state: ChatState,
    topic: dict,
    messages: list[dict],
    rag_context: str,
    *,
    max_facts: int,
) -> str:
    prompt = (
        f"Round: {state.get('round_number', 1)}\n"
        f"Phase: {state.get('phase', get_phase_for_round(state.get('round_number', 1)))}\n"
        f"Topic: {topic['summary']}\n"
        f"{FACT_CITATION_PROTOCOL}\n"
    )
    if rag_context:
        prompt += f"{rag_context}\n"
    prompt += "=== RECENT DEBATE ===\n"
    for message in messages:
        prompt += f"{_format_message_for_prompt(message)}\n"
    prompt += (
        "\nTASK: Extract at most "
        f"{max_facts} externally-sourced conclusion candidates that appear in the recent debate without a supporting [F...] citation. "
        "Inspect both uncited externally-sourced statements in the debate and any retrieved [W...] items. "
        "Only include claims that look like paper conclusions, official statistics, reputable web conclusions, or expert-source claims. "
        "[W...] items are leads only: if a [W] item is worth keeping, rewrite it as a conservative fact candidate rather than copying raw web wording into permanent memory. "
        "Each candidate MUST include a short source reference list and a short source excerpt. "
        "Do not include internally-derived conclusions, summaries, or unsupported opinions. "
        'Reply with strict JSON only: {"action":"propose_fact_candidates","fact_candidates":[{"candidate_text":"...","source_refs_json":["..."],"source_excerpt":"..."}]}.'
    )
    return f"{PROMPTS['fact_proposer']}\n\nContext:\n{prompt}"


def build_clerk_claim_prompt(
    state: ChatState,
    topic: dict,
    messages: list[dict],
    rag_context: str,
    *,
    cited_fact_context: str,
    max_claims: int,
) -> str:
    prompt = (
        f"Round: {state.get('round_number', 1)}\n"
        f"Phase: {state.get('phase', get_phase_for_round(state.get('round_number', 1)))}\n"
        f"Topic: {topic['summary']}\n"
        f"{FACT_CITATION_PROTOCOL}\n"
    )
    if rag_context:
        prompt += f"{rag_context}\n"
    prompt += "=== VERIFIED FACTS REFERENCED THIS ROUND ===\n"
    prompt += f"{cited_fact_context}\n"
    prompt += "=== RECENT DEBATE ===\n"
    for message in messages:
        prompt += f"{_format_message_for_prompt(message)}\n"
    prompt += (
        "\nTASK: Extract at most "
        f"{max_claims} derived claim candidates that are explicitly supported by cited facts [F...]. "
        "Only propose a claim if the current messages already contain a visible evidence chain that relies on accepted facts. "
        "Do not propose claims based only on uncited chat text. "
        'Reply with strict JSON only: {"action":"propose_claim_candidates","claim_candidates":[{"candidate_text":"...","support_fact_ids_json":[1,2],"rationale_short":"..."}]}.'
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
    fact_stage = candidate.get("fact_stage", "synthesized")
    prompt = (
        f"Round: {state.get('round_number', 1)}\n"
        f"Phase: {state.get('phase', get_phase_for_round(state.get('round_number', 1)))}\n"
        f"Topic: {topic['summary']}\n"
    )
    prompt += f"{FACT_CITATION_PROTOCOL}\n"
    if subtopic:
        prompt += f"Subtopic: {subtopic['summary']}\n"
    prompt += (
        f"Candidate ID: {candidate['id']}\n"
        f"Candidate Fact: {candidate['candidate_text']}\n"
        f"Fact Stage: {fact_stage}\n"
        f"Candidate Type: {candidate.get('candidate_type', 'sourced_claim')}\n"
    )
    if candidate.get("evidence_note"):
        prompt += f"Evidence Note:\n{candidate['evidence_note']}\n"
    if candidate.get("source_refs_json"):
        prompt += f"Source Refs: {candidate['source_refs_json']}\n"
    if candidate.get("source_excerpt"):
        prompt += f"Source Excerpt: {candidate['source_excerpt']}\n"
    if rag_context:
        prompt += f"{rag_context}\n"
    prompt += "=== RECENT TRANSCRIPT ===\n"
    for message in messages:
        prompt += f"{_format_message_for_prompt(message)}\n"
    if fact_stage in {"bootstrap", "inline"}:
        prompt += (
            "\nTASK: Verify whether this candidate fact should enter permanent memory as an externally checkable factual claim. "
            "For bootstrap and inline facts, be strict: prefer concrete, externally verifiable claims grounded in the current web evidence. "
            "Reject weak, interpretive, or overstated wording aggressively."
        )
    else:
        prompt += (
            "\nTASK: Verify whether this candidate fact should enter permanent memory as a synthesized working conclusion. "
            "For synthesized facts, cautious consolidation is allowed, but the wording must stay conservative and evidence-grounded."
        )
    prompt += (
        " You MUST rely on both the local context above and web-grounded verification. "
        "If the candidate is grounded in [W...] leads, you may promote it into a durable [F...] only after verification and conservative rewriting; raw [W...] text is never permanent memory by itself. "
        "Decision rules: accept if the claim is specific and supported; soften if the core idea is supportable but the wording is too broad, too absolute, or too strong; reject if unsupported, speculative, or merely interpretive. "
        "Use `correct` when the candidate points at a real fact but the value or wording must be repaired before storage. "
        "Absolute formulations such as `no evidence`, `always`, `never`, `proves`, or `definitively` must be softened or rejected unless the evidence explicitly supports them. "
        "Reply with STRICT JSON using this schema: "
        "{\"action\": \"review_fact\", \"decision\": \"accept|correct|soften|reject\", \"verification_status\": \"accepted|corrected|unsupported|refuted\", \"reviewed_text\": \"...\", \"review_note\": \"...\", \"evidence_note\": \"...\", \"source_refs_json\": [\"...\"], \"source_excerpt\": \"...\", \"confidence_score\": 8}."
    )
    return f"{PROMPTS['librarian']}\n\nContext:\n{prompt}"


def build_claim_review_prompt(
    state: ChatState,
    topic: dict,
    subtopic: dict | None,
    candidate: dict,
    messages: list[dict],
    support_facts: Sequence[dict],
    rag_context: str,
) -> str:
    prompt = (
        f"Round: {state.get('round_number', 1)}\n"
        f"Phase: {state.get('phase', get_phase_for_round(state.get('round_number', 1)))}\n"
        f"Topic: {topic['summary']}\n"
        f"{FACT_CITATION_PROTOCOL}\n"
    )
    if subtopic:
        prompt += f"Subtopic: {subtopic['summary']}\n"
    prompt += (
        f"Claim Candidate ID: {candidate['id']}\n"
        f"Claim Candidate: {candidate['candidate_text']}\n"
    )
    if rag_context:
        prompt += f"{rag_context}\n"
    prompt += "=== SUPPORT FACTS ===\n"
    for fact in support_facts:
        prompt += f"[F{fact['id']}] {fact['content']}\n"
    prompt += "=== RECENT TRANSCRIPT ===\n"
    for message in messages:
        prompt += f"{_format_message_for_prompt(message)}\n"
    prompt += (
        "\nTASK: Review whether this derived claim should enter the claim table. "
        "Accept only if the cited facts genuinely support the claim. "
        "Soften if the direction is supportable but the wording is still too strong. "
        "Reject if the reasoning overreaches, skips steps, or is not actually supported by the cited facts. "
        'Reply with STRICT JSON only: {"action":"review_claim","decision":"accept|soften|reject","reviewed_text":"...","review_note":"...","supported_fact_ids":[1,2],"claim_score":7}.'
    )
    return f"{PROMPTS['librarian']}\n\nContext:\n{prompt}"


def build_bootstrap_fact_direction_prompt(topic: dict, subtopic: dict) -> str:
    prompt = (
        f"Topic: {topic['summary']}\n"
        f"Subtopic: {subtopic['summary']}\n"
        f"Subtopic Detail: {subtopic.get('detail', '')}\n"
        f"{FACT_CITATION_PROTOCOL}\n"
        "TASK: Propose up to 3 fact directions that are worth checking before round 1 starts. "
        "These should be baseline external facts or data points that would reduce early hallucination and improve the quality of the first discussion round. "
        "Prefer directions that can be checked on reputable sources and reused later in the subtopic. "
        "Reply with JSON using this schema: "
        "{\"action\":\"propose_fact_directions\",\"directions\":[\"direction 1\",\"direction 2\",\"direction 3\"]}."
    )
    return f"{PROMPTS['skynet']}\n\nContext:\n{prompt}"


def _render_search_evidence_note(search_evidence: Sequence[SearchEvidenceItem]) -> str:
    lines: list[str] = []
    for item in search_evidence:
        status = "error" if item.had_error else "ok"
        snippet = item.rendered_results.strip().replace("\n", " ")[:240]
        lines.append(f"query={item.query} status={status} evidence={snippet}")
    return "\n".join(lines)


def _render_search_evidence_context(search_evidence: Sequence[SearchEvidenceItem]) -> str:
    chunks: list[str] = []
    for item in search_evidence:
        if item.had_error:
            continue
        rendered = (item.rendered_results or "").strip()
        if not rendered:
            continue
        without_header = rendered.replace("=== WEB SEARCH RESULTS ===", "", 1).strip()
        if not without_header or without_header == "No useful results found.":
            continue
        chunks.append(f"=== SEARCH EVIDENCE: {item.query} ===\n{rendered}")
    return "\n\n".join(chunks)


def _has_usable_search_evidence(search_evidence: Sequence[SearchEvidenceItem]) -> bool:
    return bool(_render_search_evidence_context(search_evidence))


def build_audience_summary_prompt(state: ChatState, topic: dict, messages: list[dict]) -> str:
    round_number = state.get("round_number", 1)
    phase = state.get("phase", get_phase_for_round(round_number))

    ctx = (
        f"Round: {round_number}\n"
        f"Phase: {phase}\n"
        f"Topic: {topic['summary']}\n"
        f"{FACT_CITATION_PROTOCOL}\n"
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
        "Inside `content`, you MUST use exactly these section headers in this exact order:\n"
        "TRAJECTORY:\n"
        "CONSENSUS:\n"
        "BLOCKERS:\n"
        "EVIDENCE GAPS:\n"
        "AGENT DELTAS:\n"
        "Section rules:\n"
        "- `TRAJECTORY`: 1-2 short sentences stating whether the room changed framing, governing metric, or recommendation this round.\n"
        "- `CONSENSUS`: state the strongest agreement. If the room is in an empirical impasse, explicitly state that 'Consensus is that empirical data is required' and record any specific experimental blueprints or decision matrices proposed.\n"
        "- `BLOCKERS`: name the main unresolved branch. If the blocker is a pure lack of empirical data that cannot be simulated, state this clearly.\n"
        "- `EVIDENCE GAPS`: list only the gaps that could justify another round. Prefix each line with `[Central]` or `[Peripheral]`.\n"
        f"- `AGENT DELTAS`: include one bullet for each participant in this order: {participant_block}. State only what changed, what was conceded, or what new attack/correction was introduced this round.\n"
        "Do not state whether the subtopic is ready to close."
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
        "TRAJECTORY:\n"
        "Round summary degraded because Gemini and MiniMax orchestration paths both failed.\n"
        "CONSENSUS:\n"
        "No reliable consensus summary is available from this round.\n"
        "BLOCKERS:\n"
        "Unknown due to degraded summary generation.\n"
        "EVIDENCE GAPS:\n"
        "[Central] Continue the debate once orchestration is healthy again.\n"
        "AGENT DELTAS:\n"
        f"{bullets}"
    )


def _current_round_standard_messages(state: ChatState, *, include_npc: bool = False, limit: int = 24) -> list[dict]:
    current_round = state.get("round_number", 1)
    messages = api.get_messages(state["topic_id"], subtopic_id=state["subtopic_id"], limit=limit)
    filtered = [
        message
        for message in messages
        if message.get("msg_type", "standard") == "standard"
        and (message.get("round_number") in {None, current_round})
    ]
    if include_npc:
        return filtered
    return [
        message
        for message in filtered
        if message.get("sender") not in {SKYNET, "writer", "librarian", "fact_proposer"}
    ]


def _extract_fact_ids_from_text(text: str) -> list[int]:
    return [int(match.group(1)) for match in re.finditer(r"\[F(\d+)\]", text or "")]


def _looks_like_metadata_number(token: str, context: str) -> bool:
    normalized_context = context.lower()
    if any(marker in normalized_context for marker in ("round ", "message ", "msg ")):
        return True
    digits = token.replace("$", "").replace("%", "").replace(",", "")
    if digits.isdigit():
        value = int(digits)
        if 1900 <= value <= 2099:
            return True
    return False


def _extract_number_fact_candidates(messages: Sequence[dict]) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()
    for message in messages:
        content = message.get("content", "")
        for line in (content or "").splitlines():
            stripped = line.strip()
            if not stripped or re.search(r"\[F\d+\]", stripped):
                continue
            for match in NUMERIC_TOKEN_PATTERN.finditer(stripped):
                token = match.group(0)
                if _looks_like_metadata_number(token, stripped):
                    continue
                candidate_text = stripped
                normalized = " ".join(candidate_text.split())
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                candidates.append(
                    {
                        "candidate_text": normalized,
                        "candidate_type": "number",
                        "source_refs": [],
                        "source_excerpt": normalized,
                    }
                )
                break
    return candidates


def _render_fact_lookup_context(facts: Sequence[dict]) -> str:
    lines: list[str] = []
    for fact in facts:
        lines.append(f"[F{fact['id']}] {fact['content']}")
    return "\n".join(lines)


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
    diagnosis = await _call_writer_stage_with_retry(
        stage_name="Writer diagnosis",
        prompt=build_writer_diagnosis_prompt(state, topic, standard_messages, rag_context),
        system_instruction=WRITER_ANALYSIS_SYSTEM_PROMPT,
        temperature=0.4,
        max_tokens=1200,
    )

    focus = ""
    if diagnosis.strip():
        focus = await _call_writer_stage_with_retry(
            stage_name="Writer focus selection",
            prompt=build_writer_selection_prompt(state, topic, standard_messages, rag_context, diagnosis),
            system_instruction=WRITER_ANALYSIS_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=700,
        )

    prompt = build_writer_prompt(
        state,
        topic,
        standard_messages,
        rag_context,
        diagnosis=diagnosis,
        focus=focus,
    )
    resp_text = await _call_writer_stage_with_retry(
        stage_name="Writer compose",
        prompt=prompt,
        system_instruction=PROMPTS["writer"],
        temperature=0.5,
        max_tokens=4096,
        require_json=True,
        validator=_writer_compose_response_is_usable,
    )
    if not resp_text.strip():
        logger.warning("[writer] All writer critique model fallbacks failed.")
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

    logger.info("[fact_proposer] Clerk extracting fact and claim candidates from the round...")
    topic, subtopic = _load_context_entities(state)
    if not topic:
        return {}

    standard_messages = _current_round_standard_messages(state, include_npc=False, limit=24)
    if not standard_messages:
        return {}

    rag_messages = _seed_messages_for_rag(topic, subtopic, standard_messages)
    rag_context, _ = await assemble_rag_context(
        state["topic_id"],
        state["subtopic_id"],
        rag_messages,
        "fact_proposer",
    )
    number_limit = FINAL_NUMBER_FACT_LIMIT if force else NUMBER_FACT_LIMIT
    sourced_limit = FINAL_SOURCED_FACT_LIMIT if force else SOURCED_FACT_LIMIT
    claim_limit = FINAL_CLAIM_LIMIT if force else CLAIM_LIMIT

    number_candidates = _extract_number_fact_candidates(standard_messages)[:number_limit]
    if number_candidates:
        await process_writer_output(
            state["topic_id"],
            state["subtopic_id"],
            None,
            "",
            structured_facts=number_candidates,
            fact_stage="synthesized",
            round_number=current_round,
            max_candidates=number_limit,
        )

    sourced_prompt = build_clerk_sourced_fact_prompt(
        state,
        topic,
        standard_messages,
        rag_context,
        max_facts=sourced_limit,
    )
    sourced_text = await _call_text_with_structured_retry(
        stage_name="Clerk sourced fact pass",
        validator=_fact_candidates_output_is_usable,
        invoke=lambda: call_text(
            sourced_prompt,
            provider="minimax",
            strategy="react",
            allow_web=True,
            system_instruction=PROMPTS["fact_proposer"],
            fallback_role="fact_proposer",
            require_json=True,
            topic_id=state["topic_id"],
            subtopic_id=state["subtopic_id"],
        ),
    )
    if sourced_text:
        parsed_sourced = _normalize_clerk_fact_candidates_contract(sourced_text)
        if parsed_sourced["parsed_ok"] and parsed_sourced["fact_candidates"]:
            await process_writer_output(
                state["topic_id"],
                state["subtopic_id"],
                None,
                "",
                structured_facts=parsed_sourced["fact_candidates"],
                fact_stage="synthesized",
                round_number=current_round,
                max_candidates=sourced_limit,
            )

    cited_fact_ids = sorted(
        {
            fact_id
            for message in standard_messages
            for fact_id in _extract_fact_ids_from_text(message.get("content", ""))
        }
    )
    if cited_fact_ids:
        support_facts = api.get_facts_by_ids(state["topic_id"], cited_fact_ids)
        if support_facts:
            claim_prompt = build_clerk_claim_prompt(
                state,
                topic,
                standard_messages,
                rag_context,
                cited_fact_context=_render_fact_lookup_context(support_facts),
                max_claims=claim_limit,
            )
            claim_text = await _call_text_with_structured_retry(
                stage_name="Clerk claim pass",
                validator=_claim_candidates_output_is_usable,
                invoke=lambda: call_text(
                    claim_prompt,
                    provider="minimax",
                    strategy="direct",
                    allow_web=False,
                    system_instruction=PROMPTS["fact_proposer"],
                    fallback_role="fact_proposer",
                    require_json=True,
                ),
            )
            if claim_text:
                parsed_claims = _normalize_clerk_claim_candidates_contract(claim_text)
                if parsed_claims["parsed_ok"] and parsed_claims["claim_candidates"]:
                    await process_clerk_claim_output(
                        state["topic_id"],
                        state["subtopic_id"],
                        None,
                        parsed_claims["claim_candidates"],
                        max_candidates=claim_limit,
                    )

    marker_key = "last_final_fact_proposer_round" if force else "last_fact_proposer_round"
    return {marker_key: current_round}


async def _query_librarian_review_text(prompt: str, *, stage_name: str, validator) -> tuple[str, str]:
    minimax_text = await _call_text_with_structured_retry(
        stage_name=stage_name,
        validator=validator,
        invoke=lambda: call_text(
            prompt,
            provider="minimax",
            strategy="react",
            allow_web=True,
            system_instruction=PROMPTS["librarian"],
            fallback_role="librarian",
            require_json=True,
        ),
    )
    if minimax_text:
        return minimax_text, "minimax"

    logger.warning("[librarian] MiniMax review exhausted retries, escalating to Gemini Flash + Google.")
    resp_text = await call_text(
        prompt,
        provider="gemini",
        strategy="direct",
        allow_web=True,
        system_instruction=PROMPTS["librarian"],
        model="gemini-3.0-flash",
        temperature=0.7,
        max_tokens=8192,
        fallback_role="librarian",
        require_json=True,
    )
    return resp_text, "gemini"


async def _run_librarian_pass(
    state: ChatState,
    *,
    candidate_ids: Optional[Sequence[int]] = None,
    emit_audit_message: bool = True,
) -> dict:
    logger.info("[librarian] Reviewing pending fact and claim candidates...")
    topic, subtopic = _load_context_entities(state)
    if not topic or not subtopic:
        return {}

    pending_candidates = api.get_pending_fact_candidates(state["topic_id"], state["subtopic_id"])
    if candidate_ids is not None:
        allowed_ids = set(candidate_ids)
        pending_candidates = [candidate for candidate in pending_candidates if candidate["id"] in allowed_ids]
    pending_claims = api.get_pending_claim_candidates(state["topic_id"], state["subtopic_id"])

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
            resp_text, provider = await _query_librarian_review_text(
                prompt,
                stage_name=f"Librarian fact review {candidate['id']}",
                validator=lambda text: usable_text_output(text) and bool(extract_json(text)),
            )
            try:
                review = parse_librarian_review(resp_text, candidate["candidate_text"])
            except Exception:
                if provider != "minimax":
                    raise
                logger.warning(
                    "[librarian] MiniMax review for candidate %s was not valid JSON/schema; retrying with Gemini Flash.",
                    candidate["id"],
                )
                resp_text = await call_text(
                    prompt,
                    provider="gemini",
                    strategy="direct",
                    allow_web=True,
                    system_instruction=PROMPTS["librarian"],
                    model="gemini-3.0-flash",
                    temperature=0.7,
                    max_tokens=8192,
                    fallback_role="librarian",
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

    for candidate in pending_claims:
        support_ids = _extract_fact_ids_from_text(candidate.get("support_fact_ids_json", "")) if isinstance(candidate.get("support_fact_ids_json"), str) else []
        try:
            if isinstance(candidate.get("support_fact_ids_json"), str):
                support_ids = [int(item) for item in json.loads(candidate["support_fact_ids_json"] or "[]")]
        except Exception:
            support_ids = []
        support_facts = api.get_facts_by_ids(state["topic_id"], support_ids)
        if not support_facts:
            logger.warning("[librarian] Claim candidate %s has no valid support facts; rejecting in place.", candidate["id"])
            api.update_claim_candidate_review(
                candidate["id"],
                "reject",
                review_note="No valid support facts were available for review.",
            )
            review_results.append(
                {
                    "candidate_id": candidate["id"],
                    "record_kind": "claim",
                    "decision": "reject",
                    "review_note": "No valid support facts were available for review.",
                }
            )
            continue
        rag_context, _ = await build_query_rag_context(
            state["topic_id"],
            candidate["candidate_text"],
            exclude_ids=recent_message_ids,
        )
        prompt = build_claim_review_prompt(state, topic, subtopic, candidate, messages, support_facts, rag_context)
        try:
            resp_text, provider = await _query_librarian_review_text(
                prompt,
                stage_name=f"Librarian claim review {candidate['id']}",
                validator=lambda text: usable_text_output(text) and bool(extract_json(text)),
            )
            try:
                review = parse_claim_review(resp_text, candidate["candidate_text"], support_ids)
            except Exception:
                if provider != "minimax":
                    raise
                resp_text = await call_text(
                    prompt,
                    provider="gemini",
                    strategy="direct",
                    allow_web=True,
                    system_instruction=PROMPTS["librarian"],
                    model="gemini-3.0-flash",
                    temperature=0.7,
                    max_tokens=8192,
                    fallback_role="librarian",
                    require_json=True,
                )
                review = parse_claim_review(resp_text, candidate["candidate_text"], support_ids)
            review_results.append(await apply_claim_review(state["topic_id"], candidate, review))
        except Exception as exc:
            logger.warning(
                "[librarian] Failed to review claim candidate %s; leaving pending: %s",
                candidate["id"],
                exc,
            )

    if not review_results or not emit_audit_message:
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


async def bootstrap_fact_intake_node(state: ChatState) -> dict:
    logger.info("[skynet] Bootstrapping baseline facts for the subtopic...")
    topic, subtopic = _load_context_entities(state)
    if not topic or not subtopic:
        return {}

    direction_prompt = build_bootstrap_fact_direction_prompt(topic, subtopic)
    direction_text = await _call_text_with_structured_retry(
        stage_name="Bootstrap fact direction generation",
        validator=_fact_direction_output_is_usable,
        invoke=lambda: call_text(
            direction_prompt,
            provider="gemini",
            strategy="direct",
            allow_web=False,
            system_instruction=PROMPTS["skynet"],
            model="gemini-3.0-flash",
            fallback_role=SKYNET,
            require_json=True,
        ),
    )
    if not direction_text:
        logger.warning("[skynet] Bootstrap fact direction generation failed.")
        return {}

    parsed_directions = _normalize_fact_direction_contract(direction_text)
    if not parsed_directions["parsed_ok"]:
        logger.warning("[skynet] Bootstrap fact directions were not parseable; skipping bootstrap fact intake.")
        return {}

    created_candidate_ids: list[int] = []
    for direction in parsed_directions["directions"][:BOOTSTRAP_FACT_DIRECTION_LIMIT]:
        try:
            evidence_response = await collect_search_evidence_bundle(
                "fact_proposer",
                f"Gather evidence for this bootstrap fact direction:\n{direction}",
                max_iter=1,
                system_prompt=PROMPTS["fact_proposer"],
                topic_id=state["topic_id"],
                subtopic_id=state["subtopic_id"],
            )
        except Exception as exc:
            logger.warning("[skynet] Bootstrap search failed for direction '%s': %s", direction, exc)
            continue

        if not _has_usable_search_evidence(evidence_response.search_evidence):
            continue

        evidence_note = _render_search_evidence_note(evidence_response.search_evidence)
        synthetic_messages = [
            {
                "sender": SKYNET,
                "content": f"Bootstrap fact direction: {direction}\n\n{evidence_note}",
                "msg_type": "standard",
            }
        ]
        try:
            rag_context, _ = await build_query_rag_context(
                state["topic_id"],
                direction,
            )
        except Exception as exc:
            logger.warning("[skynet] Bootstrap RAG context failed for direction '%s': %s", direction, exc)
            rag_context = ""

        proposer_prompt = build_fact_proposer_prompt(
            state,
            topic,
            synthetic_messages,
            rag_context,
            max_facts=1,
            fact_stage="bootstrap",
            focus_label=direction,
        )
        proposer_text = await _call_text_with_structured_retry(
            stage_name=f"Bootstrap fact proposal {direction[:48]}",
            validator=_fact_list_output_is_usable,
            invoke=lambda: call_text(
                proposer_prompt,
                provider="minimax",
                strategy="direct",
                allow_web=False,
                system_instruction=PROMPTS["fact_proposer"],
                fallback_role="fact_proposer",
                require_json=True,
            ),
        )
        if not proposer_text:
            continue

        parsed = _normalize_fact_proposal_contract(proposer_text)
        if not parsed["parsed_ok"]:
            continue

        candidate_ids = await process_writer_output(
            state["topic_id"],
            state["subtopic_id"],
            None,
            "",
            structured_facts=parsed["facts"],
            fact_stage="bootstrap",
            evidence_note=evidence_note,
            round_number=state.get("round_number"),
            max_candidates=1,
        )
        created_candidate_ids.extend(candidate_ids)

    if created_candidate_ids:
        await _run_librarian_pass(
            state,
            candidate_ids=created_candidate_ids,
            emit_audit_message=False,
        )
    return {}


async def _run_inline_fact_intake(
    state: ChatState,
    *,
    actor: str,
    topic: dict,
    subtopic: dict | None,
    rag_context: str,
    search_evidence: Sequence[SearchEvidenceItem],
) -> None:
    if actor == SPECTATOR:
        return
    if actor not in DELIBERATORS and actor not in special_agents():
        return
    if not _has_usable_search_evidence(search_evidence):
        return

    messages = api.get_messages(state["topic_id"], subtopic_id=state["subtopic_id"], limit=8)
    evidence_context = _render_search_evidence_context(search_evidence)
    if not evidence_context:
        return
    prompt_messages = messages + [
        {
            "sender": "web_search",
            "content": evidence_context,
            "msg_type": "standard",
        }
    ]
    prompt = build_fact_proposer_prompt(
        state,
        topic,
        prompt_messages,
        rag_context,
        max_facts=1,
        fact_stage="inline",
        focus_label=f"Turn actor: {actor}",
    )
    proposer_text = await _call_text_with_structured_retry(
        stage_name=f"Inline fact proposal {actor}",
        validator=_fact_list_output_is_usable,
        invoke=lambda: call_text(
            prompt,
            provider="minimax",
            strategy="direct",
            allow_web=False,
            system_instruction=PROMPTS["fact_proposer"],
            fallback_role="fact_proposer",
            require_json=True,
        ),
    )
    if not proposer_text:
        return

    parsed = _normalize_fact_proposal_contract(proposer_text)
    if not parsed["parsed_ok"]:
        return

    candidate_ids = await process_writer_output(
        state["topic_id"],
        state["subtopic_id"],
        None,
        "",
        structured_facts=parsed["facts"],
        fact_stage="inline",
        evidence_note=_render_search_evidence_note(search_evidence),
        round_number=state.get("round_number"),
        max_candidates=INLINE_FACT_LIMIT,
    )
    if candidate_ids:
        await _run_librarian_pass(
            state,
            candidate_ids=candidate_ids,
            emit_audit_message=False,
        )

async def expert_node(state: ChatState) -> dict:
    actor = state["current_actor"]
    turn_kind = state.get("current_turn_kind", BASE_TURN)
    logger.info(f"[{actor}] Speaking (Round {state.get('round_number', 1)})...")

    topic, subtopic = _load_context_entities(state)
    if not topic:
        return {"current_actor": ""}
    messages = api.get_messages(state["topic_id"], subtopic_id=state["subtopic_id"], limit=6)
    summary_messages = api.get_messages(
        state["topic_id"],
        subtopic_id=state["subtopic_id"],
        limit=1,
        msg_type="summary",
    )
    latest_summary = summary_messages[-1]["content"] if summary_messages else ""

    rag_messages = _seed_messages_for_rag(topic, subtopic, messages)
    system_prompt = build_actor_system_prompt(state, actor, turn_kind)
    planner_prompt = build_actor_prompt(
        state,
        actor,
        turn_kind,
        topic,
        subtopic,
        messages,
        "",
        latest_summary=latest_summary,
        include_output_contract=False,
    )
    rag_context, rag_degraded = await assemble_rag_context(
        topic["id"],
        subtopic["id"] if subtopic else 0,
        rag_messages,
        actor,
        planner_system_prompt=system_prompt,
        planner_context=planner_prompt,
        latest_summary=latest_summary,
        allow_web_backup=should_enable_web_backup(state, actor, turn_kind),
    )
    prompt = build_actor_prompt(
        state,
        actor,
        turn_kind,
        topic,
        subtopic,
        messages,
        rag_context,
        latest_summary=latest_summary,
    )

    # Model call depending on role and phase
    search_failed = False
    search_evidence: Sequence[SearchEvidenceItem] = ()
    if should_enable_web_search(state, actor, turn_kind):
        logger.info(f"[{actor}] Entering ReAct search loop...")
        try:
            response = await _call_text_with_structured_retry(
                stage_name=f"{actor} web turn",
                validator=lambda item: (
                    bool(_normalize_focus_contract(item.text)["parsed_ok"])
                    if actor == SPECTATOR
                    else _structured_message_is_usable(item.text, accepted_actions=("post_message",))
                ),
                invoke=lambda: call_text_with_search_evidence(
                    prompt,
                    provider="minimax",
                    strategy="react",
                    allow_web=True,
                    system_instruction=system_prompt,
                    fallback_role=actor,
                    topic_id=state["topic_id"],
                    subtopic_id=state["subtopic_id"],
                ),
            )
            if response is None:
                raise RuntimeError("structured retry exhausted")
            resp_text = response.text
            search_evidence = response.search_evidence
            search_failed = response.search_failed
        except Exception as exc:
            logger.warning("[%s] Web-enhanced broker call failed: %s", actor, exc)
            resp_text = str(exc)
            search_failed = True
    else:
        try:
            resp_text = await _call_text_with_structured_retry(
                stage_name=f"{actor} direct turn",
                validator=lambda text: (
                    bool(_normalize_focus_contract(text)["parsed_ok"])
                    if actor == SPECTATOR
                    else _structured_message_is_usable(text, accepted_actions=("post_message",))
                ),
                invoke=lambda: call_text(
                    prompt,
                    provider=get_agent_spec(actor).default_provider,
                    strategy="direct",
                    allow_web=False,
                    system_instruction=system_prompt,
                    fallback_role=actor,
                    require_json=True,
                ),
            )
            if resp_text is None:
                raise RuntimeError("structured retry exhausted")
        except Exception as exc:
            logger.warning("[%s] Direct broker call failed: %s", actor, exc)
            resp_text = str(exc)

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
    citation_knowledge_blocks = [rag_context] + [
        item.rendered_results for item in search_evidence if item.rendered_results
    ]
    if latest_summary:
        citation_knowledge_blocks.append(latest_summary)
    for msg in messages:
        if msg.get("content"):
            citation_knowledge_blocks.append(msg["content"])

    content, removed_citations = _sanitize_citations_to_allowed_ids(
        parsed["content"],
        knowledge_blocks=citation_knowledge_blocks,
    )
    if any(removed_citations.values()):
        logger.info(
            "[%s] Stripped citations not present in injected knowledge F=%s C=%s W=%s",
            actor,
            removed_citations["F"],
            removed_citations["C"],
            removed_citations["W"],
        )
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

    if search_evidence:
        await _run_inline_fact_intake(
            state,
            actor=actor,
            topic=topic,
            subtopic=subtopic,
            rag_context=rag_context,
            search_evidence=search_evidence,
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
    resp_text = await _call_text_with_structured_retry(
        stage_name="Round summary generation",
        validator=lambda text: _structured_message_is_usable(text, accepted_actions=("post_summary", "post_message")),
        invoke=lambda: call_text(
            prompt,
            provider="gemini",
            strategy="direct",
            allow_web=False,
            system_instruction=PROMPTS["skynet"],
            model="gemini-3.0-flash",
            fallback_role=SKYNET,
            require_json=True,
        ),
    )
    if not resp_text:
        resp_text = json.dumps({
            "action": "post_summary",
            "content": _build_degraded_audience_summary(state, messages),
        })
    
    parsed = _normalize_message_contract(resp_text, accepted_actions=("post_summary", "post_message"))
    content = parsed["content"]
    if not _has_required_summary_sections(content):
        # Attempt to repair the malformed summary using decomposition
        content = await _repair_summary_by_decomposition(content)
        # Final sanity check: if the repair also fails to produce the headers, then degrade.
        if not _has_required_summary_sections(content):
            logger.warning("[skynet] Decomposition repair failed to produce required sections; using degraded fallback.")
            content = _build_degraded_audience_summary(state, messages)
    
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
    if current_round >= 7:
        logger.info("[skynet] Forcing subtopic close at round %s.", current_round)
        return {"subtopic_exhausted": True}

    topic, subtopic = _load_context_entities(state)
    if not topic or not subtopic:
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
    decision_prompt = _build_termination_vote_prompt(
        topic_summary=topic["summary"],
        topic_detail=ctx + historical_context,
        stage_guidance=stage_guidance,
    )
    if not _should_run_termination_vote(current_round):
        logger.info(
            "[skynet] Skipping termination vote at round %s because early rounds are for stance-taking and evidence gathering.",
            current_round,
        )
        is_done = False
    elif _pending_extra_turns(state):
        logger.info("[skynet] Deferring close because extra turns are still pending.")
        is_done = False
    else:
        try:
            vote_records = await _run_termination_votes(
                voters=voting_agents(),
                prompt=decision_prompt,
                topic_id=state["topic_id"],
                subtopic_id=state["subtopic_id"],
                round_number=current_round,
                subject=subtopic["summary"],
            )
            aggregation = _aggregate_termination_votes(vote_records, current_round)
            logger.info(
                "[skynet] Termination aggregation round=%s valid_votes=%s invalid_votes=%s close_votes=%s close_ratio=%.2f blocker_counts=%s blocked_by=%s subtopic_exhausted=%s",
                current_round,
                aggregation["valid_votes"],
                aggregation["invalid_votes"],
                aggregation["close_votes"],
                aggregation["close_ratio"],
                aggregation["blocker_counts"],
                aggregation["blocked_by"],
                aggregation["subtopic_exhausted"],
            )
            if aggregation["invalid_votes"] > TERMINATION_MAX_INVALID_VOTES:
                logger.warning(
                    "[skynet] Termination vote degraded open because %s governance votes were invalid.",
                    aggregation["invalid_votes"],
                )
            is_done = aggregation["subtopic_exhausted"]
        except Exception as exc:
            logger.warning("[skynet] Termination vote degraded open after vote failure: %s", exc)
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
    logger.info("[writer] Final writer critique skipped because close-path work is harvest-only.")
    return {}


async def fact_proposer_node(state: ChatState) -> dict:
    return await _run_fact_proposer_pass(state, force=False)


async def final_fact_proposer_node(state: ChatState) -> dict:
    return await _run_fact_proposer_pass(state, force=True)


async def librarian_node(state: ChatState) -> dict:
    return await _run_librarian_pass(state)


async def final_librarian_node(state: ChatState) -> dict:
    await _run_librarian_pass(state)
    pending_candidates = api.get_pending_fact_candidates(state["topic_id"], state["subtopic_id"])
    pending_claims = api.get_pending_claim_candidates(state["topic_id"], state["subtopic_id"])
    if pending_candidates or pending_claims:
        logger.warning(
            "[librarian] %s fact candidates and %s claim candidates remain pending; delaying subtopic close for another round.",
            len(pending_candidates),
            len(pending_claims),
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
    builder.add_node("bootstrap_fact_intake_node", bootstrap_fact_intake_node)
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
        {"close_subtopic": "final_fact_proposer_node", "setup_next_round": "setup_next_round_node"}
    )
    
    builder.add_edge("setup_next_round_node", "dispatcher")
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
    builder.add_edge(START, "bootstrap_fact_intake_node")
    builder.add_edge("bootstrap_fact_intake_node", "dispatcher")
    
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

    if is_gemini_enabled():
        try:
            await asyncio.wait_for(warmup_gemini_cli(), timeout=45)
            logger.info("[GeminiCLI] Warmup completed.")
        except Exception as exc:
            logger.warning("[GeminiCLI] Warmup failed; continuing with lazy fallback: %s", exc)
    else:
        logger.info("[GeminiCLI] Warmup skipped because ENABLE_GEMINI is not enabled.")

    master_graph = build_master_graph()
    try:
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
    finally:
        await shutdown_broker()

if __name__ == "__main__":
    configure_logging()
    asyncio.run(run_server_loop())
