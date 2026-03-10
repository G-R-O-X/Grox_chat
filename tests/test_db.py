import os
import pytest

from agent_chatroom.db import (
    get_db,
    get_db_path,
    init_db,
    insert_fact_with_embedding,
    insert_message_with_embedding,
    search_facts,
    search_facts_lexical,
    search_messages,
    search_messages_lexical,
)
from agent_chatroom import api

@pytest.fixture(autouse=True)
def setup_teardown():
    # Use a test database
    os.environ["TESTING"] = "1"
    db_path = get_db_path()
    if os.path.exists(db_path):
        os.remove(db_path)
    init_db()
    yield
    if os.path.exists(db_path):
        os.remove(db_path)

def test_db_schema_upgrades():
    with get_db() as conn:
        # Check if Plan table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Plan'")
        assert cursor.fetchone() is not None

        # Check if Fact table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Fact'")
        assert cursor.fetchone() is not None

        # Check Subtopic updates
        cursor = conn.execute("PRAGMA table_info(Subtopic)")
        columns = [row['name'] for row in cursor.fetchall()]
        assert 'start_msg_id' in columns
        assert 'conclusion' in columns
        assert 'status' in columns

        cursor = conn.execute("PRAGMA table_info(Plan)")
        columns = [row['name'] for row in cursor.fetchall()]
        assert 'current_index' in columns

        cursor = conn.execute("PRAGMA table_info(Message)")
        columns = [row['name'] for row in cursor.fetchall()]
        assert 'confidence_score' in columns
        
        # Check vector tables
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_facts'")
        assert cursor.fetchone() is not None
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_messages'")
        assert cursor.fetchone() is not None
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='facts_fts'")
        assert cursor.fetchone() is not None
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'")
        assert cursor.fetchone() is not None

def test_sqlite_vec_extension():
    with get_db() as conn:
        # Check if vec_version function exists (part of sqlite-vec)
        cursor = conn.execute("SELECT vec_version()")
        version = cursor.fetchone()[0]
        assert version is not None
        assert isinstance(version, str)

def test_fact_insertion():
    with get_db() as conn:
        # Insert Topic
        cursor = conn.execute("INSERT INTO Topic (summary, detail) VALUES ('Test Topic', 'Test Detail')")
        topic_id = cursor.lastrowid
        
        # Insert Fact
        cursor = conn.execute(
            "INSERT INTO Fact (topic_id, content, source) VALUES (?, ?, ?)",
            (topic_id, "Test Fact Content", "Writer Verification")
        )
        fact_id = cursor.lastrowid
        assert fact_id is not None
        
        # Verify Fact
        cursor = conn.execute("SELECT * FROM Fact WHERE id = ?", (fact_id,))
        fact = cursor.fetchone()
        assert fact['content'] == "Test Fact Content"
        assert fact['source'] == "Writer Verification"

def test_plan_insertion():
    with get_db() as conn:
        # Insert Topic
        cursor = conn.execute("INSERT INTO Topic (summary, detail) VALUES ('Test Topic', 'Test Detail')")
        topic_id = cursor.lastrowid
        
        # Insert Plan
        cursor = conn.execute(
            "INSERT INTO Plan (topic_id, content) VALUES (?, ?)",
            (topic_id, "Test Plan Content")
        )
        plan_id = cursor.lastrowid
        assert plan_id is not None

def test_vector_search():
    with get_db() as conn:
        # 1. Insert Topic
        cursor = conn.execute("INSERT INTO Topic (summary, detail) VALUES ('Test Topic', 'Test Detail')")
        topic_id = cursor.lastrowid
        cursor = conn.execute("INSERT INTO Topic (summary, detail) VALUES ('Other Topic', 'Other Detail')")
        other_topic_id = cursor.lastrowid
    
    # 2. Insert facts with embeddings
    emb1 = [0.1] * 384
    emb2 = [0.9] * 384
    emb3 = [0.11] * 384
    
    fact1_id = insert_fact_with_embedding(topic_id, "Fact about cats", "Source 1", emb1)
    fact2_id = insert_fact_with_embedding(topic_id, "Fact about dogs", "Source 2", emb2)
    insert_fact_with_embedding(other_topic_id, "Fact from another topic", "Source 3", emb3)
    
    assert fact1_id is not None
    assert fact2_id is not None
    
    # 3. Search for facts
    query_emb = [0.11] * 384 # Close to emb1
    results = search_facts(topic_id, query_emb, top_k=1)
    
    assert len(results) == 1
    assert results[0]['id'] == fact1_id
    assert results[0]['content'] == "Fact about cats"


def test_message_vector_search_is_topic_scoped():
    with get_db() as conn:
        cursor = conn.execute("INSERT INTO Topic (summary, detail) VALUES ('Topic A', 'Detail A')")
        topic_a = cursor.lastrowid
        cursor = conn.execute("INSERT INTO Topic (summary, detail) VALUES ('Topic B', 'Detail B')")
        topic_b = cursor.lastrowid

    emb = [0.2] * 384
    msg_a = insert_message_with_embedding(topic_a, None, "audience", "Summary A", "summary", emb)
    insert_message_with_embedding(topic_b, None, "audience", "Summary B", "summary", emb)

    results = search_messages(topic_a, emb, msg_type="summary", top_k=5, exclude_ids=[msg_a])
    assert results == []


def test_fact_lexical_search():
    with get_db() as conn:
        cursor = conn.execute("INSERT INTO Topic (summary, detail) VALUES ('Topic A', 'Detail A')")
        topic_id = cursor.lastrowid

    insert_fact_with_embedding(topic_id, "Vector memory supports lexical recall", "Writer", [0.3] * 384)
    results = search_facts_lexical(topic_id, "lexical recall", top_k=5)

    assert len(results) == 1
    assert results[0]["content"] == "Vector memory supports lexical recall"


def test_message_lexical_search_indexes_plain_post_message():
    with get_db() as conn:
        cursor = conn.execute("INSERT INTO Topic (summary, detail) VALUES ('Topic A', 'Detail A')")
        topic_id = cursor.lastrowid

    msg_id = api.post_message(topic_id, None, "dreamer", "Lexical fallback message", "standard", confidence_score=2.5)
    results = search_messages_lexical(topic_id, "fallback", msg_type="standard", top_k=5)

    assert len(results) == 1
    assert results[0]["id"] == msg_id
    assert results[0]["confidence_score"] == 2.5
