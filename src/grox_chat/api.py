from .db import (
    close_subtopic as db_close_subtopic,
    fact_exists as db_fact_exists,
    get_db,
    get_open_subtopic as db_get_open_subtopic,
    get_db_path,
    insert_fact_with_embedding,
    insert_message_with_embedding,
    search_facts,
    search_facts_lexical,
    search_messages,
    search_messages_lexical,
    update_plan_cursor,
    update_subtopic_start_msg,
)
from .embedding import aget_embedding

# Expose db functions that don't need additional api logic
__all__ = [
    'get_current_topic', 'get_topic', 'create_plan', 'get_plan', 'get_current_subtopics',
    'get_latest_subtopic', 'get_subtopic', 'get_messages', 'create_topic', 'set_topic_status',
    'create_subtopic', 'post_message', 'search_facts', 'insert_fact_with_embedding',
    'insert_message_with_embedding', 'search_messages', 'search_facts_lexical',
    'search_messages_lexical', 'search_facts_hybrid', 'search_messages_hybrid', 'get_active_plan',
    'advance_plan_cursor', 'update_subtopic_start_msg', 'close_subtopic',
    'get_open_subtopic', 'get_db_path', 'persist_message', 'fact_exists'
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

def post_message(topic_id, subtopic_id, sender, content, msg_type='standard', confidence_score=None):
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO Message (topic_id, subtopic_id, sender, content, msg_type, confidence_score) VALUES (?, ?, ?, ?, ?, ?)",
            (topic_id, subtopic_id, sender, content, msg_type, confidence_score)
        )
        conn.execute(
            "INSERT OR REPLACE INTO messages_fts(rowid, content, topic_id, msg_type, sender) VALUES (?, ?, ?, ?, ?)",
            (cursor.lastrowid, content, str(topic_id), msg_type, sender),
        )
        return cursor.lastrowid


async def persist_message(topic_id, subtopic_id, sender, content, msg_type='standard', confidence_score=None):
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
            )
    return post_message(topic_id, subtopic_id, sender, content, msg_type, confidence_score)


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


def fact_exists(topic_id: int, content: str, source: str = "Writer"):
    return db_fact_exists(topic_id, content, source)


def search_facts_hybrid(topic_id: int, query_text: str, query_embedding, top_k: int = 12):
    dense = search_facts(topic_id, query_embedding, top_k=top_k)
    lexical = search_facts_lexical(topic_id, query_text, top_k=top_k)
    return _merge_ranked_rows(dense, lexical, limit=top_k)


def search_messages_hybrid(topic_id: int, query_text: str, query_embedding, msg_type: str = None, top_k: int = 8, exclude_ids=None):
    dense = search_messages(topic_id, query_embedding, msg_type=msg_type, top_k=top_k, exclude_ids=exclude_ids)
    lexical = search_messages_lexical(topic_id, query_text, msg_type=msg_type, top_k=top_k, exclude_ids=exclude_ids)
    return _merge_ranked_rows(dense, lexical, limit=top_k)
