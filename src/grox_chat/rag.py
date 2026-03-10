import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from . import api
from .embedding import aget_embedding
from .reranker import arerank
from .minimax_client import query_minimax

logger = logging.getLogger(__name__)

RERANK_THRESHOLD = 0.3


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


def _render_section(title: str, records: Iterable[Dict[str, Any]], label: str) -> str:
    rows = list(records)
    if not rows:
        return ""

    section = [title]
    for record in rows:
        section.append(f"- [{label}: {record['id']}] {record['content']}")
    return "\n".join(section) + "\n"


def _recent_debate_slice(recent_messages: List[Dict[str, Any]], max_messages: int = 4) -> str:
    lines = []
    for message in recent_messages[-max_messages:]:
        lines.append(f"[{message['sender']}]: {message['content'][:220]}")
    return "\n".join(lines)


async def assemble_rag_context(
    topic_id: int,
    subtopic_id: int,
    recent_messages: List[Dict[str, Any]],
    current_speaker: str,
) -> Tuple[str, bool]:
    """
    Implements the local RAG pipeline.
    1. Distill recent context into an English query.
    2. Embed the query.
    3. Retrieve Top-K Facts via sqlite-vec.
    4. Rerank the facts against the query.
    5. Assemble the final prompt.
    """
    
    if not recent_messages:
        return "", False
        
    # Step 1: Query Formulation
    last_message = recent_messages[-1]['content']
    recent_debate = _recent_debate_slice(recent_messages)
    degraded = False
    
    # We use MiniMax to quickly distill the semantic core of the current debate into a tight query
    distill_prompt = (
        "You are a search query generator. Analyze the following recent debate context and the upcoming speaker. "
        "Output a highly concise, factual search query in English that captures the core technical or factual dispute. "
        "ONLY output the query string, nothing else."
    )
    distill_context = f"Upcoming Speaker: {current_speaker}\nRecent Debate:\n{recent_debate}"
    
    logger.info(f"[RAG] Formulating query for {current_speaker}...")
    query_ch, _ = await query_minimax(
        distill_prompt,
        distill_context,
        max_tokens=8192,
        recover_pseudo_tool_query=True,
    )
    query_ch = query_ch.strip()
    
    if "Error" in query_ch or not query_ch:
        logger.warning("[RAG] Query formulation failed, falling back to raw message.")
        degraded = True
        query_ch = last_message[:200]
        
    logger.info(f"[RAG] Generated Query: {query_ch}")
    
    # Step 2: Embedding
    query_emb = await aget_embedding(query_ch)
    if not query_emb:
        logger.warning("[RAG] Embedding failed.")
        return "", True
        
    # Step 3: Retrieval (KNN)
    try:
        recent_message_ids = [m["id"] for m in recent_messages if "id" in m]
        candidate_facts = api.search_facts_hybrid(topic_id, query_ch, query_emb, top_k=12)
        candidate_summaries = api.search_messages_hybrid(
            topic_id,
            query_ch,
            query_emb,
            msg_type="summary",
            top_k=8,
            exclude_ids=recent_message_ids,
        )
        candidate_messages = api.search_messages_hybrid(
            topic_id,
            query_ch,
            query_emb,
            msg_type="standard",
            top_k=8,
            exclude_ids=recent_message_ids,
        )
    except Exception as exc:
        logger.warning("[RAG] Retrieval failed: %s", exc)
        return "", True

    if not candidate_facts and not candidate_summaries and not candidate_messages:
        return "", degraded

    # Step 4: Reranking
    try:
        selected_facts = await _select_relevant_records(query_ch, candidate_facts, top_k=3)
        selected_summaries = await _select_relevant_records(query_ch, candidate_summaries, top_k=2)
        selected_messages = await _select_relevant_records(query_ch, candidate_messages, top_k=2)
    except Exception as exc:
        logger.warning("[RAG] Reranking failed: %s", exc)
        return "", True

    if not selected_facts and not selected_summaries and not selected_messages:
        return "", degraded

    # Step 5: Assembly
    sections = [
        "=== RAG KNOWLEDGE INJECTION ===",
        _render_section("[Related Verified Facts]", selected_facts, "Fact").rstrip(),
        _render_section("[Relevant Historical Summaries]", selected_summaries, "Summary").rstrip(),
        _render_section("[Relevant Historical Messages]", selected_messages, "Message").rstrip(),
    ]
    rag_text = "\n".join(section for section in sections if section)
    return rag_text + "\n\n", degraded
