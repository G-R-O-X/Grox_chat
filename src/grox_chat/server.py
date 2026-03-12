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
    lines.append('Reply with strict JSON: {"vote":"yes"} or {"vote":"no"}.')
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
        "Default voting policy:\n"
        "- If `centrality` is `central` or `mixed`, default to `continue`.\n"
        "- If `recent_shift` is `yes` or `unclear`, default to `continue`.\n"
        "- If `conditional_support` is `yes`, default to `continue`.\n"
        "- If `untested_novelty` is `yes`, default to `continue`.\n"
        "- Vote `close` only when blockers are gone and another round would mostly repeat the same ground.\n"
        "- If you vote `close` while any blocker is present, `override_reason` must be a short non-empty sentence.\n"
        'Reply with strict JSON only: {"main_branch":"...","centrality":"central|mixed|peripheral|none","recent_shift":"yes|no|unclear","conditional_support":"yes|no","untested_novelty":"yes|no","vote":"continue|close","override_reason":"... or null"}.'
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
        '{"main_branch":"...","centrality":"central|mixed|peripheral|none","recent_shift":"yes|no|unclear","conditional_support":"yes|no","untested_novelty":"yes|no","vote":"continue|close","override_reason":"... or null"}\n'
        "Preserve the original intent when possible.\n"
        "Output JSON only. Do not add markdown fences, commentary, or extra keys."
    )


def _has_required_summary_sections(content: str) -> bool:
    line_cursor = -1
    lines = (content or "").splitlines()
    for header in SUMMARY_SECTION_HEADERS:
        position = next(
            (index for index, line in enumerate(lines) if index > line_cursor and line.strip() == header),
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
        "override_reason": None,
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
    fact_stage: str = "synthesized",
    focus_label: str | None = None,
) -> str:
    prompt = (
        f"Round: {state.get('round_number', 1)}\n"
        f"Phase: {state.get('phase', get_phase_for_round(state.get('round_number', 1)))}\n"
        f"Topic: {topic['summary']}\n"
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
    if subtopic:
        prompt += f"Subtopic: {subtopic['summary']}\n"
    prompt += (
        f"Candidate ID: {candidate['id']}\n"
        f"Candidate Fact: {candidate['candidate_text']}\n"
        f"Fact Stage: {fact_stage}\n"
    )
    if candidate.get("evidence_note"):
        prompt += f"Evidence Note:\n{candidate['evidence_note']}\n"
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
        "Decision rules: accept if the claim is specific and supported; soften if the core idea is supportable but the wording is too broad, too absolute, or too strong; reject if unsupported, speculative, or merely interpretive. "
        "Absolute formulations such as `no evidence`, `always`, `never`, `proves`, or `definitively` must be softened or rejected unless the evidence explicitly supports them. "
        "Reply with STRICT JSON using this schema: "
        "{\"action\": \"review_fact\", \"decision\": \"accept|soften|reject\", \"reviewed_text\": \"...\", \"review_note\": \"...\", \"evidence_note\": \"...\", \"confidence_score\": 8}."
    )
    return f"{PROMPTS['librarian']}\n\nContext:\n{prompt}"


def build_bootstrap_fact_direction_prompt(topic: dict, subtopic: dict) -> str:
    prompt = (
        f"Topic: {topic['summary']}\n"
        f"Subtopic: {subtopic['summary']}\n"
        f"Subtopic Detail: {subtopic.get('detail', '')}\n"
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
        "- `CONSENSUS`: state the strongest current agreement and include the main caveats from recent Librarian rulings or softened facts.\n"
        "- `BLOCKERS`: name the main unresolved branch blocking closure; write `None` only if no meaningful blocker remains.\n"
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
        resp_text = await call_text(
            prompt,
            provider="gemini",
            strategy="direct",
            allow_web=False,
            system_instruction=PROMPTS["writer"],
            model="gemini-3.0-flash",
            temperature=0.7,
            max_tokens=8192,
            fallback_role="writer",
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
    prompt = build_fact_proposer_prompt(
        state,
        topic,
        standard_messages,
        rag_context,
        max_facts=max_facts,
        fact_stage="synthesized",
    )
    try:
        resp_text = await call_text(
            prompt,
            provider="minimax",
            strategy="react",
            allow_web=True,
            system_instruction=PROMPTS["fact_proposer"],
            fallback_role="fact_proposer",
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
        fact_stage="synthesized",
        max_candidates=max_facts,
    )
    marker_key = "last_final_fact_proposer_round" if force else "last_fact_proposer_round"
    return {marker_key: current_round}


async def _query_librarian_review_text(prompt: str) -> tuple[str, str]:
    try:
        resp_text = await call_text(
            prompt,
            provider="minimax",
            strategy="react",
            allow_web=True,
            system_instruction=PROMPTS["librarian"],
            fallback_role="librarian",
        )
        return resp_text, "minimax"
    except Exception as exc:
        logger.warning("[librarian] MiniMax review failed, escalating to Gemini Flash + Google: %s", exc)

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
    return resp_text, "gemini"


async def _run_librarian_pass(
    state: ChatState,
    *,
    candidate_ids: Optional[Sequence[int]] = None,
    emit_audit_message: bool = True,
) -> dict:
    logger.info("[librarian] Reviewing pending fact candidates...")
    topic, subtopic = _load_context_entities(state)
    if not topic or not subtopic:
        return {}

    pending_candidates = api.get_pending_fact_candidates(state["topic_id"], state["subtopic_id"])
    if candidate_ids is not None:
        allowed_ids = set(candidate_ids)
        pending_candidates = [candidate for candidate in pending_candidates if candidate["id"] in allowed_ids]
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
    try:
        direction_text = await call_text(
            direction_prompt,
            provider="gemini",
            strategy="direct",
            allow_web=False,
            system_instruction=PROMPTS["skynet"],
            model="gemini-3.0-flash",
            fallback_role=SKYNET,
        )
    except Exception as exc:
        logger.warning("[skynet] Bootstrap fact direction generation failed: %s", exc)
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
        try:
            proposer_text = await call_text(
                proposer_prompt,
                provider="minimax",
                strategy="direct",
                allow_web=False,
                system_instruction=PROMPTS["fact_proposer"],
                fallback_role="fact_proposer",
            )
        except Exception as exc:
            logger.warning("[fact_proposer] Bootstrap fact proposal failed for '%s': %s", direction, exc)
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
    try:
        proposer_text = await call_text(
            prompt,
            provider="minimax",
            strategy="direct",
            allow_web=False,
            system_instruction=PROMPTS["fact_proposer"],
            fallback_role="fact_proposer",
        )
    except Exception as exc:
        logger.warning("[fact_proposer] Inline fact proposal failed for %s: %s", actor, exc)
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
    search_evidence: Sequence[SearchEvidenceItem] = ()
    if should_enable_web_search(state, actor, turn_kind):
        logger.info(f"[{actor}] Entering ReAct search loop...")
        try:
            response = await call_text_with_search_evidence(
                prompt,
                provider="minimax",
                strategy="react",
                allow_web=True,
                system_instruction=system_prompt,
                fallback_role=actor,
            )
            resp_text = response.text
            search_evidence = response.search_evidence
            search_failed = response.search_failed
        except Exception as exc:
            logger.warning("[%s] Web-enhanced broker call failed: %s", actor, exc)
            resp_text = str(exc)
            search_failed = True
    else:
        try:
            resp_text = await call_text(
                prompt,
                provider=get_agent_spec(actor).default_provider,
                strategy="direct",
                allow_web=False,
                system_instruction=system_prompt,
                fallback_role=actor,
            )
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
    try:
        resp_text = await call_text(
            prompt,
            provider="gemini",
            strategy="direct",
            allow_web=False,
            system_instruction=PROMPTS["skynet"],
            model="gemini-3.0-flash",
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
    if not _has_required_summary_sections(content):
        logger.warning("[skynet] Summary missing required sections; using degraded fallback.")
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
    if current_round >= 10:
        logger.info("[skynet] Forcing subtopic close at round %s.", current_round)
        return {"subtopic_exhausted": True}

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
