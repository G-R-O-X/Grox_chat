import contextlib
import os
import re
import sqlite3
import struct
from typing import Any, Dict, Iterable, List, Optional

def get_db_path() -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    if os.environ.get("TESTING") == "1":
        return os.path.join(base_dir, "test_chatroom.db")
    return os.path.join(base_dir, "chatroom.db")


DB_PATH = get_db_path()

@contextlib.contextmanager
def get_db():
    conn = sqlite3.connect(get_db_path(), timeout=10.0)
    
    # Load sqlite-vec
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as e:
        print(f"Warning: Failed to load sqlite-vec extension: {e}")

    conn.row_factory = sqlite3.Row
    try:
        # Enable Write-Ahead Logging for better concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def serialize_f32(vector: List[float]) -> bytes:
    """Serializes a list of floats into a format sqlite-vec expects (blob of f32s)."""
    return struct.pack(f"{len(vector)}f", *vector)


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_def: str) -> None:
    if column_name not in _table_columns(conn, table_name):
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")


def _insert_fact_fts(conn: sqlite3.Connection, fact_id: int, topic_id: int, content: str, source: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO facts_fts(rowid, content, topic_id, source) VALUES (?, ?, ?, ?)",
        (fact_id, content, str(topic_id), source),
    )


def _insert_message_fts(
    conn: sqlite3.Connection,
    msg_id: int,
    topic_id: int,
    sender: str,
    content: str,
    msg_type: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO messages_fts(rowid, content, topic_id, msg_type, sender) VALUES (?, ?, ?, ?, ?)",
        (msg_id, content, str(topic_id), msg_type, sender),
    )


def _backfill_fts(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO facts_fts(rowid, content, topic_id, source)
        SELECT Fact.id, Fact.content, CAST(Fact.topic_id AS TEXT), Fact.source
        FROM Fact
        """
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO messages_fts(rowid, content, topic_id, msg_type, sender)
        SELECT Message.id, Message.content, CAST(Message.topic_id AS TEXT), Message.msg_type, Message.sender
        FROM Message
        """
    )


def _build_fts_query(query_text: str) -> Optional[str]:
    tokens = re.findall(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]+", query_text or "")
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS Topic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT NOT NULL,
                detail TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Closed', -- Closed, Started, Running
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS Plan (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                current_index INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            );

            CREATE TABLE IF NOT EXISTS Subtopic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                summary TEXT NOT NULL,
                detail TEXT NOT NULL,
                start_msg_id INTEGER,
                conclusion TEXT,
                status TEXT NOT NULL DEFAULT 'Open', -- Open, Closed
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            );

            CREATE TABLE IF NOT EXISTS Message (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                subtopic_id INTEGER,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                msg_type TEXT NOT NULL DEFAULT 'standard', -- standard, summary
                confidence_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(subtopic_id) REFERENCES Subtopic(id)
            );
            
            CREATE TABLE IF NOT EXISTS Fact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            );
            
            -- Virtual table for storing embeddings of Facts using sqlite-vec
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_facts USING vec0(
                fact_id INTEGER PRIMARY KEY,
                embedding float[384]
            );
            
            -- Virtual table for storing embeddings of Summaries using sqlite-vec
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages USING vec0(
                msg_id INTEGER PRIMARY KEY,
                embedding float[384]
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                content,
                topic_id UNINDEXED,
                source UNINDEXED
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content,
                topic_id UNINDEXED,
                msg_type UNINDEXED,
                sender UNINDEXED
            );
        """)
        _ensure_column(conn, "Plan", "current_index", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "Subtopic", "start_msg_id", "INTEGER")
        _ensure_column(conn, "Subtopic", "conclusion", "TEXT")
        _ensure_column(conn, "Subtopic", "status", "TEXT NOT NULL DEFAULT 'Open'")
        _ensure_column(conn, "Message", "confidence_score", "REAL")
        _backfill_fts(conn)

def insert_fact_with_embedding(topic_id: int, content: str, source: str, embedding: List[float]) -> int:
    """Insert a fact and its corresponding embedding into the database."""
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO Fact (topic_id, content, source) VALUES (?, ?, ?)",
            (topic_id, content, source)
        )
        fact_id = cursor.lastrowid
        _insert_fact_fts(conn, fact_id, topic_id, content, source)
        
        conn.execute(
            "INSERT INTO vec_facts(fact_id, embedding) VALUES (?, ?)",
            (fact_id, serialize_f32(embedding))
        )
        return fact_id


def fact_exists(topic_id: int, content: str, source: str) -> bool:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM Fact
            WHERE topic_id = ? AND content = ? AND source = ?
            LIMIT 1
            """,
            (topic_id, content, source),
        ).fetchone()
        return row is not None

def search_facts(topic_id: int, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for the most semantically similar facts using sqlite-vec."""
    with get_db() as conn:
        query = """
            SELECT Fact.*, vec_distance_L2(vec_facts.embedding, ?) as distance
            FROM vec_facts
            JOIN Fact ON Fact.id = vec_facts.fact_id
            WHERE Fact.topic_id = ?
            ORDER BY distance
            LIMIT ?
        """
        rows = conn.execute(query, (serialize_f32(query_embedding), topic_id, top_k)).fetchall()
        return [dict(row) for row in rows]


def search_facts_lexical(topic_id: int, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    match_query = _build_fts_query(query_text)
    if not match_query:
        return []

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT Fact.*, bm25(facts_fts) as lexical_score
            FROM facts_fts
            JOIN Fact ON Fact.id = facts_fts.rowid
            WHERE facts_fts MATCH ? AND Fact.topic_id = ?
            ORDER BY lexical_score
            LIMIT ?
            """,
            (match_query, topic_id, top_k),
        ).fetchall()
        return [dict(row) for row in rows]


def insert_message_with_embedding(
    topic_id: int,
    subtopic_id: int,
    sender: str,
    content: str,
    msg_type: str,
    embedding: List[float] = None,
    confidence_score: Optional[float] = None,
) -> int:
    """Insert a message and its embedding."""
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO Message (topic_id, subtopic_id, sender, content, msg_type, confidence_score) VALUES (?, ?, ?, ?, ?, ?)",
            (topic_id, subtopic_id, sender, content, msg_type, confidence_score)
        )
        msg_id = cursor.lastrowid
        _insert_message_fts(conn, msg_id, topic_id, sender, content, msg_type)
        
        if embedding:
            conn.execute(
                "INSERT INTO vec_messages(msg_id, embedding) VALUES (?, ?)",
                (msg_id, serialize_f32(embedding))
            )
        return msg_id

def _exclude_clause(column_name: str, values: Optional[Iterable[int]]) -> tuple[str, list[int]]:
    items = [int(v) for v in values or []]
    if not items:
        return "", []
    placeholders = ", ".join("?" for _ in items)
    return f" AND {column_name} NOT IN ({placeholders})", items


def search_messages(
    topic_id: int,
    query_embedding: List[float],
    msg_type: str = None,
    top_k: int = 5,
    exclude_ids: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    """Search for semantically similar messages (e.g. summaries)."""
    with get_db() as conn:
        exclude_sql, exclude_params = _exclude_clause("Message.id", exclude_ids)
        if msg_type:
            query = """
                SELECT Message.*, vec_distance_L2(vec_messages.embedding, ?) as distance
                FROM vec_messages
                JOIN Message ON Message.id = vec_messages.msg_id
                WHERE Message.topic_id = ? AND Message.msg_type = ?
            """ + exclude_sql + """
                ORDER BY distance
                LIMIT ?
            """
            params = [serialize_f32(query_embedding), topic_id, msg_type, *exclude_params, top_k]
        else:
            query = """
                SELECT Message.*, vec_distance_L2(vec_messages.embedding, ?) as distance
                FROM vec_messages
                JOIN Message ON Message.id = vec_messages.msg_id
                WHERE Message.topic_id = ?
            """ + exclude_sql + """
                ORDER BY distance
                LIMIT ?
            """
            params = [serialize_f32(query_embedding), topic_id, *exclude_params, top_k]
            
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def search_messages_lexical(
    topic_id: int,
    query_text: str,
    msg_type: str = None,
    top_k: int = 5,
    exclude_ids: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    match_query = _build_fts_query(query_text)
    if not match_query:
        return []

    with get_db() as conn:
        exclude_sql, exclude_params = _exclude_clause("Message.id", exclude_ids)
        if msg_type:
            query = """
                SELECT Message.*, bm25(messages_fts) as lexical_score
                FROM messages_fts
                JOIN Message ON Message.id = messages_fts.rowid
                WHERE messages_fts MATCH ? AND Message.topic_id = ? AND Message.msg_type = ?
            """ + exclude_sql + """
                ORDER BY lexical_score
                LIMIT ?
            """
            params = [match_query, topic_id, msg_type, *exclude_params, top_k]
        else:
            query = """
                SELECT Message.*, bm25(messages_fts) as lexical_score
                FROM messages_fts
                JOIN Message ON Message.id = messages_fts.rowid
                WHERE messages_fts MATCH ? AND Message.topic_id = ?
            """ + exclude_sql + """
                ORDER BY lexical_score
                LIMIT ?
            """
            params = [match_query, topic_id, *exclude_params, top_k]

        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def update_plan_cursor(plan_id: int, current_index: int) -> None:
    with get_db() as conn:
        conn.execute("UPDATE Plan SET current_index = ? WHERE id = ?", (current_index, plan_id))


def update_subtopic_start_msg(subtopic_id: int, start_msg_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE Subtopic SET start_msg_id = ? WHERE id = ?",
            (start_msg_id, subtopic_id),
        )


def close_subtopic(subtopic_id: int, conclusion: str) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE Subtopic SET status = 'Closed', conclusion = ? WHERE id = ?",
            (conclusion, subtopic_id),
        )


def get_open_subtopic(topic_id: int) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM Subtopic
            WHERE topic_id = ? AND status = 'Open'
            ORDER BY id DESC
            LIMIT 1
            """,
            (topic_id,),
        ).fetchone()
        return dict(row) if row else None
