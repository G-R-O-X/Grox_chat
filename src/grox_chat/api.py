from .db import (
    supersede_facts as db_supersede_facts,

    get_web_evidence_for_topic as db_get_web_evidence,

    claim_candidate_exists as db_claim_candidate_exists,
    close_subtopic as db_close_subtopic,
    create_claim_candidate as db_create_claim_candidate,
    create_fact_candidate as db_create_fact_candidate,
    fact_exists as db_fact_exists,
    fact_candidate_exists as db_fact_candidate_exists,
    get_db,
    get_claim_candidates as db_get_claim_candidates,
    get_open_subtopic as db_get_open_subtopic,
    get_db_path,
    get_facts_by_ids as db_get_facts_by_ids,
    get_fact_by_content as db_get_fact_by_content,
    get_fact_candidates as db_get_fact_candidates,
    get_vote_records as db_get_vote_records,
    insert_claim as db_insert_claim,
    insert_vote_record as db_insert_vote_record,
    insert_web_evidence as db_insert_web_evidence,
    insert_fact,
    insert_fact_with_embedding,
    insert_message_with_embedding,
    search_claims_lexical,
    search_facts,
    search_facts_lexical,
    search_messages,
    search_messages_lexical,
    search_web_evidence_cross_topic as db_search_web_evidence_cross_topic,
    search_web_evidence_same_topic as db_search_web_evidence_same_topic,
    update_claim_candidate_review as db_update_claim_candidate_review,
    update_fact_candidate_review as db_update_fact_candidate_review,
    update_plan_cursor,
    update_subtopic_start_msg,
)
from .embedding import aget_embedding

# Expose db functions that don't need additional api logic
__all__ = [
    'get_current_topic', 'get_topic', 'create_plan', 'get_plan', 'get_current_subtopics',
    'get_latest_subtopic', 'get_subtopic', 'get_messages', 'create_topic', 'set_topic_status',
    'create_subtopic', 'post_message', 'search_facts', 'insert_fact', 'insert_fact_with_embedding',
    'insert_message_with_embedding', 'search_messages', 'search_facts_lexical',
    'search_messages_lexical', 'search_facts_hybrid', 'search_messages_hybrid', 'get_active_plan',
    'advance_plan_cursor', 'update_subtopic_start_msg', 'close_subtopic',
    'get_open_subtopic', 'get_db_path', 'persist_message', 'fact_exists', 'get_fact_by_content',
    'create_fact_candidate', 'create_fact_candidate_with_stage', 'get_pending_fact_candidates', 'get_fact_candidates', 'get_facts',
    'fact_candidate_exists', 'update_fact_candidate_review', 'create_claim_candidate', 'get_pending_claim_candidates',
    'get_claim_candidates', 'claim_candidate_exists', 'update_claim_candidate_review', 'insert_claim', 'get_facts_by_ids',
    'search_claims_hybrid', 'insert_web_evidence', 'search_web_evidence_same_topic', 'search_web_evidence_cross_topic',
    'insert_vote_record', 'get_vote_records'
]

def _dict(row):
    return dict(row) if row else None


def _merge_ranked_rows(*groups, limit: int):
    ranked_groups = [list(group) for group in groups if group]
    merged = []
    seen = set()
    offsets = [0] * len(ranked_groups)

    while len(merged) < limit:
        progressed = False
        for index, group in enumerate(ranked_groups):
            while offsets[index] < len(group):
                row = group[offsets[index]]
                offsets[index] += 1
                row_id = row["id"]
                if row_id in seen:
                    continue
                seen.add(row_id)
                merged.append(row)
                progressed = True
                break
            if len(merged) >= limit:
                break
        if not progressed:
            break
    return merged

def get_current_topic():
    with get_db() as conn:
        row = conn.execute("SELECT * FROM Topic ORDER BY id DESC LIMIT 1").fetchone()
        return _dict(row)

def get_topic(topic_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM Topic WHERE id = ?", (topic_id,)).fetchone()
        return _dict(row)

def create_plan(topic_id: int, content: str, current_index: int = 0) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO Plan (topic_id, content, current_index) VALUES (?, ?, ?)",
            (topic_id, content, current_index),
        )
        return cursor.lastrowid

def get_plan(topic_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM Plan WHERE topic_id = ? ORDER BY id DESC LIMIT 1", (topic_id,)).fetchone()
        return _dict(row)


def get_active_plan(topic_id: int):
    return get_plan(topic_id)

def get_current_subtopics(topic_id):
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM Subtopic WHERE topic_id = ? ORDER BY id ASC", (topic_id,)).fetchall()
        return [_dict(r) for r in rows]

def get_latest_subtopic(topic_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM Subtopic WHERE topic_id = ? ORDER BY id DESC LIMIT 1", (topic_id,)).fetchone()
        return _dict(row)

def get_subtopic(subtopic_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM Subtopic WHERE id = ?", (subtopic_id,)).fetchone()
        return _dict(row)


def get_messages(topic_id, subtopic_id=None, limit=50, msg_type=None):
    with get_db() as conn:
        clauses = ["topic_id = ?"]
        params = [topic_id]
        if subtopic_id is not None:
            clauses.append("subtopic_id = ?")
            params.append(subtopic_id)
        if msg_type is not None:
            clauses.append("msg_type = ?")
            params.append(msg_type)
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM Message WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()
        # Return in chronological order
        return [_dict(r) for r in reversed(rows)]

def create_topic(summary, detail):
    with get_db() as conn:
        cursor = conn.execute("INSERT INTO Topic (summary, detail, status) VALUES (?, ?, 'Started')", (summary, detail))
        return cursor.lastrowid

def set_topic_status(topic_id, status):
    if status not in ['Closed', 'Started', 'Running']:
        raise ValueError("Invalid status")
    with get_db() as conn:
        conn.execute("UPDATE Topic SET status = ? WHERE id = ?", (status, topic_id))

def create_subtopic(topic_id, summary, detail, start_msg_id=None):
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO Subtopic (topic_id, summary, detail, start_msg_id, status) VALUES (?, ?, ?, ?, 'Open')",
            (topic_id, summary, detail, start_msg_id),
        )
        return cursor.lastrowid

def post_message(
    topic_id,
    subtopic_id,
    sender,
    content,
    msg_type='standard',
    confidence_score=None,
    round_number=None,
    turn_kind=None,
):
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO Message (topic_id, subtopic_id, sender, content, msg_type, confidence_score, round_number, turn_kind)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (topic_id, subtopic_id, sender, content, msg_type, confidence_score, round_number, turn_kind)
        )
        conn.execute(
            "INSERT OR REPLACE INTO messages_fts(rowid, content, topic_id, msg_type, sender) VALUES (?, ?, ?, ?, ?)",
            (cursor.lastrowid, content, str(topic_id), msg_type, sender),
        )
        return cursor.lastrowid


async def persist_message(
    topic_id,
    subtopic_id,
    sender,
    content,
    msg_type='standard',
    confidence_score=None,
    round_number=None,
    turn_kind=None,
):
    if msg_type == 'standard':
        embedding = await aget_embedding(content)
        if embedding:
            return insert_message_with_embedding(
                topic_id,
                subtopic_id,
                sender,
                content,
                msg_type,
                embedding,
                confidence_score,
                round_number,
                turn_kind,
            )
    return post_message(
        topic_id,
        subtopic_id,
        sender,
        content,
        msg_type,
        confidence_score,
        round_number,
        turn_kind,
    )


def advance_plan_cursor(plan_id: int):
    plan = None
    with get_db() as conn:
        plan = conn.execute("SELECT current_index FROM Plan WHERE id = ?", (plan_id,)).fetchone()
    if plan is None:
        raise ValueError(f"Plan {plan_id} not found")
    update_plan_cursor(plan_id, plan["current_index"] + 1)


def close_subtopic(subtopic_id: int, conclusion: str):
    db_close_subtopic(subtopic_id, conclusion)


def get_open_subtopic(topic_id: int):
    return db_get_open_subtopic(topic_id)


def fact_exists(topic_id: int, content: str, source: str | None = None):
    return db_fact_exists(topic_id, content, source)


def get_fact_by_content(topic_id: int, content: str):
    return db_get_fact_by_content(topic_id, content)


def create_fact_candidate(topic_id: int, subtopic_id: int, writer_msg_id: int | None, candidate_text: str, **kwargs) -> int:
    return db_create_fact_candidate(topic_id, subtopic_id, writer_msg_id, candidate_text, **kwargs)


def create_fact_candidate_with_stage(
    topic_id: int,
    subtopic_id: int,
    writer_msg_id: int | None,
    candidate_text: str,
    *,
    fact_stage: str,
    candidate_type: str = "sourced_claim",
    evidence_note: str | None = None,
    source_refs_json: str | None = None,
    source_excerpt: str | None = None,
    verification_status: str | None = None,
    round_number: int | None = None,
) -> int:
    return db_create_fact_candidate(
        topic_id,
        subtopic_id,
        writer_msg_id,
        candidate_text,
        fact_stage=fact_stage,
        candidate_type=candidate_type,
        evidence_note=evidence_note,
        source_refs_json=source_refs_json,
        source_excerpt=source_excerpt,
        verification_status=verification_status,
        round_number=round_number,
    )


def create_claim_candidate(
    topic_id: int,
    subtopic_id: int,
    clerk_msg_id: int | None,
    candidate_text: str,
    *,
    support_fact_ids_json: str | None = None,
    rationale_short: str | None = None,
) -> int:
    return db_create_claim_candidate(
        topic_id,
        subtopic_id,
        clerk_msg_id,
        candidate_text,
        support_fact_ids_json=support_fact_ids_json,
        rationale_short=rationale_short,
    )


def get_pending_fact_candidates(topic_id: int, subtopic_id: int):
    return db_get_fact_candidates(topic_id, subtopic_id=subtopic_id, status="pending")


def get_pending_claim_candidates(topic_id: int, subtopic_id: int):
    return db_get_claim_candidates(topic_id, subtopic_id=subtopic_id, status="pending")


def get_fact_candidates(topic_id: int, subtopic_id: int | None = None, status: str | None = None, limit: int | None = None):
    candidates = db_get_fact_candidates(topic_id, subtopic_id=subtopic_id, status=status)
    if limit is not None:
        return candidates[:limit]
    return candidates


def get_claim_candidates(topic_id: int, subtopic_id: int | None = None, status: str | None = None, limit: int | None = None):
    candidates = db_get_claim_candidates(topic_id, subtopic_id=subtopic_id, status=status)
    if limit is not None:
        return candidates[:limit]
    return candidates


def get_facts(topic_id: int, limit: int | None = None):
    with get_db() as conn:
        params = [topic_id]
        query = "SELECT * FROM Fact WHERE topic_id = ? ORDER BY id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [_dict(row) for row in rows]


def fact_candidate_exists(topic_id: int, candidate_text: str, statuses=None):
    return db_fact_candidate_exists(topic_id, candidate_text, statuses=statuses)


def claim_candidate_exists(topic_id: int, candidate_text: str, statuses=None):
    return db_claim_candidate_exists(topic_id, candidate_text, statuses=statuses)


def update_fact_candidate_review(
    candidate_id: int,
    status: str,
    reviewed_text: str | None = None,
    review_note: str | None = None,
    evidence_note: str | None = None,
    confidence_score: float | None = None,
    reviewer: str | None = None,
    accepted_fact_id: int | None = None,
):
    db_update_fact_candidate_review(
        candidate_id,
        status,
        reviewed_text=reviewed_text,
        review_note=review_note,
        evidence_note=evidence_note,
        confidence_score=confidence_score,
        reviewer=reviewer,
        accepted_fact_id=accepted_fact_id,
    )


def update_claim_candidate_review(
    candidate_id: int,
    status: str,
    reviewed_text: str | None = None,
    review_note: str | None = None,
    claim_score: float | None = None,
    accepted_claim_id: int | None = None,
):
    db_update_claim_candidate_review(
        candidate_id,
        status,
        reviewed_text=reviewed_text,
        review_note=review_note,
        claim_score=claim_score,
        accepted_claim_id=accepted_claim_id,
    )


def insert_claim(
    topic_id: int,
    subtopic_id: int | None,
    content: str,
    *,
    support_fact_ids_json: str | None = None,
    rationale_short: str | None = None,
    claim_score: float | None = None,
    status: str = "active",
    candidate_id: int | None = None,
) -> int:
    return db_insert_claim(
        topic_id,
        subtopic_id,
        content,
        support_fact_ids_json=support_fact_ids_json,
        rationale_short=rationale_short,
        claim_score=claim_score,
        status=status,
        candidate_id=candidate_id,
    )


def insert_web_evidence(
    origin_topic_id: int,
    origin_subtopic_id: int | None,
    query_text: str,
    title: str,
    snippet: str,
    url: str,
    source_domain: str,
    result_rank: int,
    search_provider: str,
    search_role: str,
) -> int:
    return db_insert_web_evidence(
        origin_topic_id,
        origin_subtopic_id,
        query_text,
        title,
        snippet,
        url,
        source_domain,
        result_rank,
        search_provider,
        search_role,
    )


def insert_vote_record(
    topic_id: int,
    subtopic_id: int | None,
    round_number: int | None,
    vote_kind: str,
    subject: str,
    prompt_text: str,
    voter: str,
    parsed_ok: bool,
    decision: str | None,
    reason: str | None,
    raw_response: str,
    metadata_json: str | None = None,
) -> int:
    return db_insert_vote_record(
        topic_id,
        subtopic_id,
        round_number,
        vote_kind,
        subject,
        prompt_text,
        voter,
        parsed_ok,
        decision,
        reason,
        raw_response,
        metadata_json,
    )


def get_vote_records(
    topic_id: int,
    *,
    subtopic_id: int | None = None,
    vote_kind: str | None = None,
    round_number: int | None = None,
    limit: int | None = None,
):
    return db_get_vote_records(
        topic_id,
        subtopic_id=subtopic_id,
        vote_kind=vote_kind,
        round_number=round_number,
        limit=limit,
    )


def get_facts_by_ids(topic_id: int, fact_ids: list[int]):
    return db_get_facts_by_ids(topic_id, fact_ids)


def search_facts_hybrid(topic_id: int, query_text: str, query_embedding, top_k: int = 12):
    dense = search_facts(topic_id, query_embedding, top_k=top_k)
    lexical = search_facts_lexical(topic_id, query_text, top_k=top_k)
    return _merge_ranked_rows(dense, lexical, limit=top_k)


def search_claims_hybrid(topic_id: int, query_text: str, query_embedding=None, top_k: int = 8):
    _ = query_embedding
    return search_claims_lexical(topic_id, query_text, top_k=top_k)


def search_messages_hybrid(topic_id: int, query_text: str, query_embedding, msg_type: str = None, top_k: int = 8, exclude_ids=None):
    dense = search_messages(topic_id, query_embedding, msg_type=msg_type, top_k=top_k, exclude_ids=exclude_ids)
    lexical = search_messages_lexical(topic_id, query_text, msg_type=msg_type, top_k=top_k, exclude_ids=exclude_ids)
    return _merge_ranked_rows(dense, lexical, limit=top_k)


def search_web_evidence_same_topic(topic_id: int, query_text: str, top_k: int = 8, max_age_days: int = 30):
    return db_search_web_evidence_same_topic(topic_id, query_text, top_k=top_k, max_age_days=max_age_days)


def search_web_evidence_cross_topic(topic_id: int, query_text: str, top_k: int = 8, max_age_days: int = 30):
    return db_search_web_evidence_cross_topic(topic_id, query_text, top_k=top_k, max_age_days=max_age_days)

def get_web_evidence_for_topic(topic_id: int) -> list[dict]:
    return db_get_web_evidence(topic_id)

def supersede_facts(fact_ids: list[int]) -> None:
    db_supersede_facts(fact_ids)


def get_claims(topic_id: int, limit: int | None = None):
    with get_db() as conn:
        params = [topic_id]
        query = "SELECT * FROM Claim WHERE topic_id = ? ORDER BY id DESC"
        if limit is not None:
            query += f" LIMIT {limit}"
        rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]
