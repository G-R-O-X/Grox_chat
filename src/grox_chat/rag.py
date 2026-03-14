import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import api
from .broker import PROFILE_MINIMAX, get_or_collect_search_evidence_item, llm_call
from .embedding import aget_embedding
from .json_utils import extract_json_object as _extract_json_object
from .reranker import arerank
from .structured_retry import retry_structured_output, usable_text_output

logger = logging.getLogger(__name__)

RERANK_THRESHOLD = 0.3
WEB_BACKUP_SEARCH_FAILURE_SENTINEL = "No useful results found."


def _normalize_query_planner_contract(raw_text: str) -> dict:
    if not usable_text_output(raw_text):
        return {"parsed_ok": False, "query": ""}
    parsed = _extract_json_object(raw_text)
    if not isinstance(parsed, dict):
        return {"parsed_ok": False, "query": ""}
    query = parsed.get("query")
    if not isinstance(query, str) or not query.strip():
        return {"parsed_ok": False, "query": ""}
    return {"parsed_ok": True, "query": query.strip()}


def _planner_output_is_usable(text: str) -> bool:
    parsed = _normalize_query_planner_contract(text)
    return parsed["parsed_ok"]


async def _select_relevant_records(
    query: str,
    records: Sequence[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not records:
        return []

    docs = [record["content"] for record in records]
    ranked_indices = await arerank(query, docs, top_k=top_k)

    selected = []
    for idx, score in ranked_indices:
        if score >= RERANK_THRESHOLD:
            selected.append({**records[idx], "score": score})
    return selected


def _render_knowledge_guide(include_web: bool) -> str:
    lines = [
        "GUIDE:",
        "- [F...] are verified or librarian-reviewed facts. Prefer them as evidence.",
        "- [C...] are derived claims supported by facts. They are weaker than [F...].",
        "- Summaries and Messages provide context only. They are not evidence and must not be cited as evidence.",
    ]
    if include_web:
        lines.append(
            "- [W...] are unverified web search results. They may be cited, but must be described as unverified web evidence and only become durable facts after clerk/librarian review promotes them into [F...]."
        )
    return "\n".join(lines)


def _render_section(title: str, records: Iterable[Dict[str, Any]], label: str) -> str:
    rows = list(records)
    if not rows:
        return ""

    section = [title]
    for record in rows:
        if label == "Fact":
            section.append(f"- [F{record['id']}] {record['content']}")
        elif label == "Claim":
            section.append(f"- [C{record['id']}] {record['content']}")
        elif label == "Web":
            section.append(f"- [W{record['id']}] {record['content']}")
        else:
            section.append(f"- [{label}: {record['id']}] {record['content']}")
    return "\n".join(section) + "\n"


def _recent_debate_slice(recent_messages: List[Dict[str, Any]], max_messages: int = 6) -> str:
    lines = []
    for message in recent_messages[-max_messages:]:
        lines.append(f"[{message['sender']}]: {message['content'][:260]}")
    return "\n".join(lines)


def _render_rag_context(
    *,
    facts: Sequence[Dict[str, Any]],
    claims: Sequence[Dict[str, Any]],
    summaries: Sequence[Dict[str, Any]],
    messages: Sequence[Dict[str, Any]],
    web_block: str = "",
) -> str:
    include_web = bool(web_block)
    sections = [
        "=== RAG KNOWLEDGE INJECTION ===",
        _render_knowledge_guide(include_web),
        _render_section("[Related Facts]", facts, "Fact").rstrip(),
        _render_section("[Related Claims]", claims, "Claim").rstrip(),
        _render_section("[Relevant Historical Summaries]", summaries, "Summary").rstrip(),
        _render_section("[Relevant Historical Messages]", messages, "Message").rstrip(),
    ]
    if web_block:
        sections.extend(
            [
                "[Related Web Evidence]",
                "Database had no relevant stored knowledge for this query. The following [W] items come from web search and have not been verified by the Librarian.",
                web_block.rstrip(),
            ]
        )
    return "\n".join(section for section in sections if section) + "\n\n"


def _has_local_knowledge(records: dict[str, Sequence[Dict[str, Any]]]) -> bool:
    return any(records.get(name) for name in ("facts", "claims", "summaries", "messages"))


def _has_usable_web_results(rendered_results: str) -> bool:
    text = (rendered_results or "").strip()
    if not text:
        return False
    without_header = text.replace("=== WEB SEARCH RESULTS ===", "", 1).strip()
    if not without_header:
        return False
    return WEB_BACKUP_SEARCH_FAILURE_SENTINEL not in without_header


async def _generate_query_text(
    *,
    recent_messages: List[Dict[str, Any]],
    current_speaker: str,
    planner_system_prompt: str,
    planner_context: str,
    latest_summary: str,
) -> tuple[str, bool]:
    last_message = recent_messages[-1]["content"]
    degraded = False

    if planner_context:
        system_prompt = (
            f"{planner_system_prompt}\n\n"
            "RETRIEVAL PLANNER MODE:\n"
            "You are planning retrieval for the upcoming turn, not writing the final answer.\n"
            "Read the role instructions, latest summary, task, and recent debate.\n"
            "Return strict JSON only using this schema: {\"query\":\"...\"}.\n"
            "The query must be concise, factual, and optimized for retrieving relevant local knowledge."
        ).strip()
        planner_prompt = planner_context
    else:
        recent_debate = _recent_debate_slice(recent_messages)
        summary_block = f"Latest Summary:\n{latest_summary}\n" if latest_summary else ""
        system_prompt = (
            "You are a retrieval query planner. Analyze the debate context and upcoming speaker.\n"
            "Return strict JSON only using this schema: {\"query\":\"...\"}.\n"
            "The query must be concise, factual, and optimized for retrieving relevant local knowledge."
        )
        planner_prompt = (
            f"Upcoming Speaker: {current_speaker}\n"
            f"{summary_block}"
            f"Recent Debate:\n{recent_debate}"
        )

    logger.info("[RAG] Formulating query for %s...", current_speaker)
    raw_text = await retry_structured_output(
        stage_name=f"RAG query planner {current_speaker}",
        invoke=lambda: llm_call(
            planner_prompt,
            system_prompt=system_prompt,
            provider_profile=PROFILE_MINIMAX,
            role=current_speaker,
            max_tokens=512,
            require_json=True,
        ),
        is_usable=lambda item: _planner_output_is_usable(item.text),
        logger=logger,
    )

    if raw_text is not None:
        parsed = _normalize_query_planner_contract(raw_text.text)
        if parsed["parsed_ok"]:
            return parsed["query"], degraded

    degraded = True
    fallback_query = (latest_summary or last_message or "")[:200].strip()
    logger.warning("[RAG] Query formulation failed for %s, falling back to raw context.", current_speaker)
    return fallback_query, degraded


async def _collect_local_rag_records(
    topic_id: int,
    query_text: str,
    *,
    exclude_ids: Optional[Sequence[int]] = None,
    fact_top_k: int = 12,
    claim_top_k: int = 8,
    summary_top_k: int = 8,
    message_top_k: int = 8,
    selected_fact_top_k: int = 3,
    selected_claim_top_k: int = 2,
    selected_summary_top_k: int = 2,
    selected_message_top_k: int = 2,
) -> tuple[dict[str, Sequence[Dict[str, Any]]], bool]:
    query = (query_text or "").strip()
    empty_records = {"facts": (), "claims": (), "summaries": (), "messages": ()}
    if not query:
        return empty_records, True

    query_emb = await aget_embedding(query)
    if not query_emb:
        logger.warning("[RAG] Embedding failed.")
        return empty_records, True

    try:
        candidate_facts = api.search_facts_hybrid(topic_id, query, query_emb, top_k=fact_top_k)
        candidate_claims = api.search_claims_hybrid(topic_id, query, query_emb, top_k=claim_top_k)
        candidate_summaries = api.search_messages_hybrid(
            topic_id,
            query,
            query_emb,
            msg_type="summary",
            top_k=summary_top_k,
            exclude_ids=exclude_ids,
        )
        candidate_messages = api.search_messages_hybrid(
            topic_id,
            query,
            query_emb,
            msg_type="standard",
            top_k=message_top_k,
            exclude_ids=exclude_ids,
        )
    except Exception as exc:
        logger.warning("[RAG] Retrieval failed: %s", exc)
        return empty_records, True

    if not candidate_facts and not candidate_claims and not candidate_summaries and not candidate_messages:
        return empty_records, False

    try:
        selected_facts = await _select_relevant_records(query, candidate_facts, top_k=selected_fact_top_k)
        selected_claims = await _select_relevant_records(query, candidate_claims, top_k=selected_claim_top_k)
        selected_summaries = await _select_relevant_records(query, candidate_summaries, top_k=selected_summary_top_k)
        selected_messages = await _select_relevant_records(query, candidate_messages, top_k=selected_message_top_k)
    except Exception as exc:
        logger.warning("[RAG] Reranking failed: %s", exc)
        return empty_records, True

    return {
        "facts": tuple(selected_facts),
        "claims": tuple(selected_claims),
        "summaries": tuple(selected_summaries),
        "messages": tuple(selected_messages),
    }, False


async def assemble_rag_context(
    topic_id: int,
    subtopic_id: int,
    recent_messages: List[Dict[str, Any]],
    current_speaker: str,
    *,
    planner_system_prompt: str = "",
    planner_context: str = "",
    latest_summary: str = "",
    allow_web_backup: bool = False,
) -> Tuple[str, bool]:
    """
    Implements the local RAG pipeline.
    1. Generate an actor-shaped retrieval query.
    2. Embed the query.
    3. Retrieve Facts / Claims / Summaries / Messages from local memory.
    4. Rerank the retrieved records.
    5. If local memory is empty and debate-time web backup is allowed, fetch or reuse [W] evidence.
    6. Assemble the final prompt.
    """
    if not recent_messages:
        return "", False

    query_text, planner_degraded = await _generate_query_text(
        recent_messages=recent_messages,
        current_speaker=current_speaker,
        planner_system_prompt=planner_system_prompt,
        planner_context=planner_context,
        latest_summary=latest_summary,
    )
    if not query_text:
        return "", True

    logger.info("[RAG] Generated Query: %s", query_text)

    recent_message_ids = [m["id"] for m in recent_messages if "id" in m]
    records, retrieval_degraded = await _collect_local_rag_records(
        topic_id,
        query_text,
        exclude_ids=recent_message_ids,
    )
    if _has_local_knowledge(records):
        rag_text = _render_rag_context(
            facts=records["facts"],
            claims=records["claims"],
            summaries=records["summaries"],
            messages=records["messages"],
        )
        return rag_text, planner_degraded or retrieval_degraded

    degraded = planner_degraded or retrieval_degraded
    if not allow_web_backup:
        return "", degraded

    evidence_item = await get_or_collect_search_evidence_item(
        query_text,
        topic_id=topic_id,
        subtopic_id=subtopic_id,
        role=current_speaker,
    )
    if evidence_item.had_error:
        degraded = True
    if not _has_usable_web_results(evidence_item.rendered_results):
        return "", degraded

    rag_text = _render_rag_context(
        facts=records["facts"],
        claims=records["claims"],
        summaries=records["summaries"],
        messages=records["messages"],
        web_block=evidence_item.rendered_results,
    )
    return rag_text, degraded


async def build_query_rag_context(
    topic_id: int,
    query_text: str,
    *,
    exclude_ids: Optional[Sequence[int]] = None,
    fact_top_k: int = 12,
    summary_top_k: int = 8,
    message_top_k: int = 8,
    selected_fact_top_k: int = 3,
    selected_summary_top_k: int = 2,
    selected_message_top_k: int = 2,
) -> Tuple[str, bool]:
    records, retrieval_degraded = await _collect_local_rag_records(
        topic_id,
        query_text,
        exclude_ids=exclude_ids,
        fact_top_k=fact_top_k,
        claim_top_k=8,
        summary_top_k=summary_top_k,
        message_top_k=message_top_k,
        selected_fact_top_k=selected_fact_top_k,
        selected_claim_top_k=2,
        selected_summary_top_k=selected_summary_top_k,
        selected_message_top_k=selected_message_top_k,
    )
    if not _has_local_knowledge(records):
        return "", retrieval_degraded

    rag_text = _render_rag_context(
        facts=records["facts"],
        claims=records["claims"],
        summaries=records["summaries"],
        messages=records["messages"],
    )
    return rag_text, retrieval_degraded
