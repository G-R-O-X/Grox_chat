import os
import pytest

from grox_chat.db import (
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
from grox_chat import api

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
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='FactCandidate'")
        assert cursor.fetchone() is not None
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Claim'")
        assert cursor.fetchone() is not None
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ClaimCandidate'")
        assert cursor.fetchone() is not None
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='WebEvidence'")
        assert cursor.fetchone() is not None
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='VoteRecord'")
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
        assert 'round_number' in columns
        assert 'turn_kind' in columns

        cursor = conn.execute("PRAGMA table_info(Fact)")
        columns = [row['name'] for row in cursor.fetchall()]
        assert 'subtopic_id' in columns
        assert 'fact_stage' in columns
        assert 'fact_type' in columns
        assert 'verification_status' in columns
        assert 'source_kind' in columns
        assert 'source_refs_json' in columns
        assert 'source_excerpt' in columns
        assert 'candidate_id' in columns
        assert 'review_status' in columns
        assert 'evidence_note' in columns
        assert 'confidence_score' in columns

        cursor = conn.execute("PRAGMA table_info(FactCandidate)")
        columns = [row['name'] for row in cursor.fetchall()]
        assert 'fact_stage' in columns
        assert 'candidate_type' in columns
        assert 'source_refs_json' in columns
        assert 'source_excerpt' in columns
        assert 'verification_status' in columns
        assert 'round_number' in columns

        cursor = conn.execute("PRAGMA table_info(Claim)")
        columns = [row['name'] for row in cursor.fetchall()]
        assert 'support_fact_ids_json' in columns
        assert 'rationale_short' in columns
        assert 'claim_score' in columns
        assert 'status' in columns
        assert 'candidate_id' in columns

        cursor = conn.execute("PRAGMA table_info(ClaimCandidate)")
        columns = [row['name'] for row in cursor.fetchall()]
        assert 'support_fact_ids_json' in columns
        assert 'rationale_short' in columns
        assert 'status' in columns
        assert 'review_note' in columns
        assert 'claim_score' in columns
        assert 'accepted_claim_id' in columns
        
        # Check vector tables
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_facts'")
        assert cursor.fetchone() is not None
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_messages'")
        assert cursor.fetchone() is not None
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='facts_fts'")
        assert cursor.fetchone() is not None
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'")
        assert cursor.fetchone() is not None
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='claims_fts'")
        assert cursor.fetchone() is not None
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='web_evidence_fts'")
        assert cursor.fetchone() is not None


def test_vote_record_insert_and_query():
    with get_db() as conn:
        cursor = conn.execute("INSERT INTO Topic (summary, detail) VALUES ('Topic A', 'Detail A')")
        topic_id = cursor.lastrowid
        cursor = conn.execute(
            "INSERT INTO Subtopic (topic_id, summary, detail, status) VALUES (?, ?, ?, 'Open')",
            (topic_id, "Subtopic A", "Subtopic Detail"),
        )
        subtopic_id = cursor.lastrowid

    api.insert_vote_record(
        topic_id,
        subtopic_id,
        3,
        "termination",
        "Current subtopic summary",
        "Prompt snapshot",
        "critic",
        True,
        "continue",
        "central blocker remains",
        '{"vote":"continue"}',
        metadata_json='{"centrality":"central"}',
    )

    rows = api.get_vote_records(topic_id, subtopic_id=subtopic_id, vote_kind="termination", round_number=3)

    assert len(rows) == 1
    assert rows[0]["voter"] == "critic"
    assert rows[0]["decision"] == "continue"
    assert rows[0]["reason"] == "central blocker remains"
    assert rows[0]["raw_response"] == '{"vote":"continue"}'

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
    msg_a = insert_message_with_embedding(topic_a, None, "skynet", "Summary A", "summary", emb)
    insert_message_with_embedding(topic_b, None, "skynet", "Summary B", "summary", emb)

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


def test_message_round_and_turn_metadata_persist():
    with get_db() as conn:
        cursor = conn.execute("INSERT INTO Topic (summary, detail) VALUES ('Topic A', 'Detail A')")
        topic_id = cursor.lastrowid

    msg_id = api.post_message(
        topic_id,
        None,
        "dreamer",
        "Structured turn metadata",
        "standard",
        confidence_score=7.5,
        round_number=3,
        turn_kind="cat_expansion",
    )

    with get_db() as conn:
        row = conn.execute("SELECT * FROM Message WHERE id = ?", (msg_id,)).fetchone()

    assert row["round_number"] == 3
    assert row["turn_kind"] == "cat_expansion"


def test_claim_candidate_and_claim_persist():
    with get_db() as conn:
        cursor = conn.execute("INSERT INTO Topic (summary, detail) VALUES ('Topic A', 'Detail A')")
        topic_id = cursor.lastrowid
        cursor = conn.execute(
            "INSERT INTO ClaimCandidate (topic_id, subtopic_id, candidate_text, support_fact_ids_json, rationale_short) VALUES (?, ?, ?, ?, ?)",
            (topic_id, None, "Supported claim", "[1,2]", "Both facts point in the same direction."),
        )
        claim_candidate_id = cursor.lastrowid
        cursor = conn.execute(
            "INSERT INTO Claim (topic_id, subtopic_id, content, support_fact_ids_json, rationale_short, claim_score, candidate_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (topic_id, None, "Supported claim", "[1,2]", "Both facts point in the same direction.", 7.5, claim_candidate_id),
        )
        claim_id = cursor.lastrowid

    with get_db() as conn:
        claim_candidate = conn.execute("SELECT * FROM ClaimCandidate WHERE id = ?", (claim_candidate_id,)).fetchone()
        claim = conn.execute("SELECT * FROM Claim WHERE id = ?", (claim_id,)).fetchone()

    assert claim_candidate["candidate_text"] == "Supported claim"
    assert claim["content"] == "Supported claim"
    assert claim["candidate_id"] == claim_candidate_id
