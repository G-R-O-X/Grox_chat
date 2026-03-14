import contextlib
import logging
import os
import re
import sqlite3
import struct
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

MAX_CONTENT_LEN = 8000
MAX_SUMMARY_LEN = 500


def _truncate(text: str | None, max_len: int) -> str | None:
    if text is None:
        return None
    return text[:max_len] if len(text) > max_len else text

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
        logger.warning("Failed to load sqlite-vec extension: %s", e)

    conn.row_factory = sqlite3.Row
    try:
        # Enable Write-Ahead Logging for better concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def serialize_f32(vector: List[float]) -> bytes:
    """Serializes a list of floats into a format sqlite-vec expects (blob of f32s)."""
    return struct.pack(f"{len(vector)}f", *vector)


_VALID_TABLES = frozenset({
    "Topic", "Plan", "Subtopic", "Message", "FactCandidate",
    "Fact", "ClaimCandidate", "Claim", "WebEvidence", "VoteRecord",
})
_VALID_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    if table_name not in _VALID_TABLES:
        raise ValueError(f"Invalid table name: {table_name!r}")
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_def: str) -> None:
    if table_name not in _VALID_TABLES:
        raise ValueError(f"Invalid table name: {table_name!r}")
    if not _VALID_COLUMN_RE.match(column_name):
        raise ValueError(f"Invalid column name: {column_name!r}")
    if column_name in _table_columns(conn, table_name):
        return
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


def _insert_claim_fts(conn: sqlite3.Connection, claim_id: int, topic_id: int, content: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO claims_fts(rowid, content, topic_id) VALUES (?, ?, ?)",
        (claim_id, content, str(topic_id)),
    )


def _insert_web_evidence_fts(
    conn: sqlite3.Connection,
    web_id: int,
    origin_topic_id: int,
    source_domain: str,
    content: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO web_evidence_fts(rowid, content, origin_topic_id, source_domain) VALUES (?, ?, ?, ?)",
        (web_id, content, str(origin_topic_id), source_domain),
    )


def _backfill_fts(conn: sqlite3.Connection) -> None:
    fact_count = conn.execute("SELECT COUNT(*) FROM Fact").fetchone()[0]
    if fact_count:
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO facts_fts(rowid, content, topic_id, source)
                SELECT
                    Fact.id,
                    CASE
                        WHEN Fact.summary IS NOT NULL AND Fact.summary != '' THEN Fact.summary || char(10) || char(10) || Fact.content
                        ELSE Fact.content
                    END,
                    CAST(Fact.topic_id AS TEXT), Fact.source
                FROM Fact
                """
            )
        except sqlite3.OperationalError as exc:
            logger.warning("facts_fts backfill failed; attempting FTS rebuild fallback: %s", exc)
            try:
                conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')")
            except sqlite3.OperationalError as rebuild_exc:
                logger.warning("facts_fts rebuild fallback also failed; continuing without legacy backfill: %s", rebuild_exc)

    message_count = conn.execute("SELECT COUNT(*) FROM Message").fetchone()[0]
    if not message_count:
        return

    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO messages_fts(rowid, content, topic_id, msg_type, sender)
            SELECT 
                Message.id, 
                CASE 
                    WHEN Message.summary IS NOT NULL AND Message.summary != '' THEN Message.summary || char(10) || char(10) || Message.content 
                    ELSE Message.content 
                END, 
                CAST(Message.topic_id AS TEXT), Message.msg_type, Message.sender
            FROM Message
            """
        )
    except sqlite3.OperationalError as exc:
        logger.warning("messages_fts backfill failed; attempting FTS rebuild fallback: %s", exc)
        try:
            conn.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")
        except sqlite3.OperationalError as rebuild_exc:
            logger.warning("messages_fts rebuild fallback also failed; continuing without legacy backfill: %s", rebuild_exc)

    claim_count = conn.execute("SELECT COUNT(*) FROM Claim").fetchone()[0]
    if claim_count:
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO claims_fts(rowid, content, topic_id)
                SELECT Claim.id,
                    CASE
                        WHEN Claim.summary IS NOT NULL AND Claim.summary != '' THEN Claim.summary || char(10) || char(10) || Claim.content
                        ELSE Claim.content
                    END,
                    CAST(Claim.topic_id AS TEXT)
                FROM Claim
                """
            )
        except sqlite3.OperationalError as exc:
            logger.warning("claims_fts backfill failed; attempting FTS rebuild fallback: %s", exc)
            try:
                conn.execute("INSERT INTO claims_fts(claims_fts) VALUES ('rebuild')")
            except sqlite3.OperationalError as rebuild_exc:
                logger.warning("claims_fts rebuild fallback also failed; continuing without legacy backfill: %s", rebuild_exc)

    web_count = conn.execute("SELECT COUNT(*) FROM WebEvidence").fetchone()[0]
    if web_count:
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO web_evidence_fts(rowid, content, origin_topic_id, source_domain)
                SELECT
                    WebEvidence.id,
                    TRIM(
                        COALESCE(WebEvidence.title, '') || char(10) || char(10) ||
                        COALESCE(WebEvidence.summary, '') || char(10) || char(10) ||
                        COALESCE(WebEvidence.snippet, '') || char(10) || char(10) ||
                        COALESCE(WebEvidence.query_text, '') || char(10) || char(10) ||
                        COALESCE(WebEvidence.source_domain, '')
                    ),
                    CAST(WebEvidence.origin_topic_id AS TEXT),
                    COALESCE(WebEvidence.source_domain, '')
                FROM WebEvidence
                """
            )
        except sqlite3.OperationalError as exc:
            logger.warning("web_evidence_fts backfill failed; attempting FTS rebuild fallback: %s", exc)
            try:
                conn.execute("INSERT INTO web_evidence_fts(web_evidence_fts) VALUES ('rebuild')")
            except sqlite3.OperationalError as rebuild_exc:
                logger.warning("web_evidence_fts rebuild fallback also failed; continuing without legacy backfill: %s", rebuild_exc)


_FTS5_RESERVED = frozenset({"AND", "OR", "NOT", "NEAR"})

def _build_fts_query(query_text: str) -> Optional[str]:
    tokens = re.findall(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]+", query_text or "")
    if not tokens:
        return None
    safe_tokens = [
        f'"{token.replace(chr(34), chr(34)+chr(34))}"'
        for token in tokens
        if token.upper() not in _FTS5_RESERVED
    ]
    if not safe_tokens:
        return None
    return " OR ".join(safe_tokens)

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
                round_number INTEGER,
                turn_kind TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(subtopic_id) REFERENCES Subtopic(id)
            );

            CREATE TABLE IF NOT EXISTS FactCandidate (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                subtopic_id INTEGER,
                writer_msg_id INTEGER,
                candidate_text TEXT NOT NULL,
                fact_stage TEXT NOT NULL DEFAULT 'synthesized',
                candidate_type TEXT NOT NULL DEFAULT 'sourced_claim',
                status TEXT NOT NULL DEFAULT 'pending',
                reviewed_text TEXT,
                review_note TEXT,
                evidence_note TEXT,
                source_refs_json TEXT,
                source_excerpt TEXT,
                verification_status TEXT,
                confidence_score REAL,
                round_number INTEGER,
                reviewer TEXT,
                accepted_fact_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                reviewed_at DATETIME,
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(subtopic_id) REFERENCES Subtopic(id),
                FOREIGN KEY(writer_msg_id) REFERENCES Message(id),
                FOREIGN KEY(accepted_fact_id) REFERENCES Fact(id)
            );
            
            CREATE TABLE IF NOT EXISTS Fact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                subtopic_id INTEGER,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                fact_stage TEXT NOT NULL DEFAULT 'synthesized',
                fact_type TEXT NOT NULL DEFAULT 'sourced_claim',
                verification_status TEXT,
                source_kind TEXT,
                source_refs_json TEXT,
                source_excerpt TEXT,
                candidate_id INTEGER,
                review_status TEXT,
                evidence_note TEXT,
                confidence_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(subtopic_id) REFERENCES Subtopic(id),
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(candidate_id) REFERENCES FactCandidate(id)
            );

            CREATE TABLE IF NOT EXISTS ClaimCandidate (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                subtopic_id INTEGER,
                clerk_msg_id INTEGER,
                candidate_text TEXT NOT NULL,
                summary TEXT,
                support_fact_ids_json TEXT,
                rationale_short TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                review_note TEXT,
                reviewed_text TEXT,
                claim_score REAL,
                accepted_claim_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                reviewed_at DATETIME,
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(subtopic_id) REFERENCES Subtopic(id),
                FOREIGN KEY(clerk_msg_id) REFERENCES Message(id)
            );

            CREATE TABLE IF NOT EXISTS Claim (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                subtopic_id INTEGER,
                content TEXT NOT NULL,
                summary TEXT,
                support_fact_ids_json TEXT,
                rationale_short TEXT,
                claim_score REAL,
                status TEXT NOT NULL DEFAULT 'active',
                candidate_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(subtopic_id) REFERENCES Subtopic(id),
                FOREIGN KEY(candidate_id) REFERENCES ClaimCandidate(id)
            );

            CREATE TABLE IF NOT EXISTS WebEvidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                origin_topic_id INTEGER NOT NULL,
                origin_subtopic_id INTEGER,
                query_text TEXT NOT NULL,
                title TEXT,
                snippet TEXT,
                url TEXT,
                source_domain TEXT,
                result_rank INTEGER,
                search_provider TEXT,
                search_role TEXT,
                summary TEXT,
                verified INTEGER NOT NULL DEFAULT 0,
                fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(origin_topic_id) REFERENCES Topic(id),
                FOREIGN KEY(origin_subtopic_id) REFERENCES Subtopic(id)
            );

            CREATE TABLE IF NOT EXISTS VoteRecord (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                subtopic_id INTEGER,
                round_number INTEGER,
                vote_kind TEXT NOT NULL,
                subject TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                voter TEXT NOT NULL,
                parsed_ok INTEGER NOT NULL DEFAULT 0,
                decision TEXT,
                reason TEXT,
                raw_response TEXT NOT NULL,
                metadata_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(subtopic_id) REFERENCES Subtopic(id)
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

            CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
                content,
                topic_id UNINDEXED
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS web_evidence_fts USING fts5(
                content,
                origin_topic_id UNINDEXED,
                source_domain UNINDEXED
            );
        """)
        _ensure_column(conn, "Plan", "current_index", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "Subtopic", "start_msg_id", "INTEGER")
        _ensure_column(conn, "Subtopic", "conclusion", "TEXT")
        _ensure_column(conn, "Subtopic", "status", "TEXT NOT NULL DEFAULT 'Open'")
        _ensure_column(conn, "Message", "confidence_score", "REAL")
        _ensure_column(conn, "Message", "round_number", "INTEGER")
        _ensure_column(conn, "Message", "turn_kind", "TEXT")
        _ensure_column(conn, "Fact", "candidate_id", "INTEGER")
        _ensure_column(conn, "Fact", "subtopic_id", "INTEGER")
        _ensure_column(conn, "Fact", "fact_stage", "TEXT NOT NULL DEFAULT 'synthesized'")
        _ensure_column(conn, "Fact", "fact_type", "TEXT NOT NULL DEFAULT 'sourced_claim'")
        _ensure_column(conn, "Fact", "verification_status", "TEXT")
        _ensure_column(conn, "Fact", "source_kind", "TEXT")
        _ensure_column(conn, "Fact", "source_refs_json", "TEXT")
        _ensure_column(conn, "Fact", "source_excerpt", "TEXT")
        _ensure_column(conn, "Topic", "conclusion", "TEXT")
        _ensure_column(conn, "Fact", "review_status", "TEXT")
        _ensure_column(conn, "Fact", "summary", "TEXT")
        _ensure_column(conn, "Claim", "summary", "TEXT")
        _ensure_column(conn, "Message", "summary", "TEXT")
        _ensure_column(conn, "FactCandidate", "summary", "TEXT")
        _ensure_column(conn, "ClaimCandidate", "summary", "TEXT")
        _ensure_column(conn, "WebEvidence", "summary", "TEXT")
        _ensure_column(conn, "FactCandidate", "evidence_note", "TEXT")
        _ensure_column(conn, "Fact", "confidence_score", "REAL")
        _ensure_column(conn, "FactCandidate", "fact_stage", "TEXT NOT NULL DEFAULT 'synthesized'")
        _ensure_column(conn, "FactCandidate", "candidate_type", "TEXT NOT NULL DEFAULT 'sourced_claim'")
        _ensure_column(conn, "FactCandidate", "source_refs_json", "TEXT")
        _ensure_column(conn, "FactCandidate", "source_excerpt", "TEXT")
        _ensure_column(conn, "FactCandidate", "verification_status", "TEXT")
        _ensure_column(conn, "FactCandidate", "round_number", "INTEGER")
        _backfill_fts(conn)

def _insert_fact_row(
    conn: sqlite3.Connection,
    topic_id: int,
    subtopic_id: Optional[int],
    content: str,
    source: str,
    fact_stage: str = "synthesized",
    fact_type: str = "sourced_claim",
    verification_status: Optional[str] = None,
    source_kind: Optional[str] = None,
    source_refs_json: Optional[str] = None,
    source_excerpt: Optional[str] = None,
    candidate_id: Optional[int] = None,
    review_status: Optional[str] = None,
    evidence_note: Optional[str] = None,
    confidence_score: Optional[float] = None,
    summary: Optional[str] = None,
) -> int:
    content = _truncate(content, MAX_CONTENT_LEN)
    summary = _truncate(summary, MAX_SUMMARY_LEN)
    cursor = conn.execute(
        """
        INSERT INTO Fact (
            topic_id,
            subtopic_id,
            content,
            summary,
            source,
            fact_stage,
            fact_type,
            verification_status,
            source_kind,
            source_refs_json,
            source_excerpt,
            candidate_id,
            review_status,
            evidence_note,
            confidence_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            topic_id,
            subtopic_id,
            content,
            summary,
            source,
            fact_stage,
            fact_type,
            verification_status,
            source_kind,
            source_refs_json,
            source_excerpt,
            candidate_id,
            review_status,
            evidence_note,
            confidence_score,
        ),
    )
    fact_id = cursor.lastrowid
    fts_content = content
    if summary:
        fts_content = f"{summary}\n\n{content}"
    _insert_fact_fts(conn, fact_id, topic_id, fts_content, source)
    return fact_id


def insert_fact(
    topic_id: int,
    content: str,
    source: str,
    subtopic_id: Optional[int] = None,
    fact_stage: str = "synthesized",
    fact_type: str = "sourced_claim",
    verification_status: Optional[str] = None,
    source_kind: Optional[str] = None,
    source_refs_json: Optional[str] = None,
    source_excerpt: Optional[str] = None,
    candidate_id: Optional[int] = None,
    review_status: Optional[str] = None,
    evidence_note: Optional[str] = None,
    confidence_score: Optional[float] = None,
    summary: Optional[str] = None,
) -> int:
    with get_db() as conn:
        return _insert_fact_row(
            conn,
            topic_id,
            subtopic_id,
            content,
            source,
            fact_stage=fact_stage,
            fact_type=fact_type,
            verification_status=verification_status,
            source_kind=source_kind,
            source_refs_json=source_refs_json,
            source_excerpt=source_excerpt,
            candidate_id=candidate_id,
            review_status=review_status,
            evidence_note=evidence_note,
            confidence_score=confidence_score,
            summary=summary,
        )


def insert_fact_with_embedding(
    topic_id: int,
    content: str,
    source: str,
    embedding: List[float],
    subtopic_id: Optional[int] = None,
    fact_stage: str = "synthesized",
    fact_type: str = "sourced_claim",
    verification_status: Optional[str] = None,
    source_kind: Optional[str] = None,
    source_refs_json: Optional[str] = None,
    source_excerpt: Optional[str] = None,
    candidate_id: Optional[int] = None,
    review_status: Optional[str] = None,
    evidence_note: Optional[str] = None,
    confidence_score: Optional[float] = None,
    summary: Optional[str] = None,
) -> int:
    """Insert a fact and its corresponding embedding into the database."""
    with get_db() as conn:
        fact_id = _insert_fact_row(
            conn,
            topic_id,
            subtopic_id,
            content,
            source,
            fact_stage=fact_stage,
            fact_type=fact_type,
            verification_status=verification_status,
            source_kind=source_kind,
            source_refs_json=source_refs_json,
            source_excerpt=source_excerpt,
            candidate_id=candidate_id,
            review_status=review_status,
            evidence_note=evidence_note,
            confidence_score=confidence_score,
            summary=summary,
        )
        conn.execute(
            "INSERT INTO vec_facts(fact_id, embedding) VALUES (?, ?)",
            (fact_id, serialize_f32(embedding))
        )
        return fact_id

def create_fact_candidate(
    topic_id: int,
    subtopic_id: int,
    writer_msg_id: Optional[int],
    candidate_text: str,
    summary: Optional[str] = None,
    fact_stage: str = "synthesized",
    candidate_type: str = "sourced_claim",
    evidence_note: Optional[str] = None,
    source_refs_json: Optional[str] = None,
    source_excerpt: Optional[str] = None,
    verification_status: Optional[str] = None,
    round_number: Optional[int] = None,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO FactCandidate (
                topic_id,
                subtopic_id,
                writer_msg_id,
                candidate_text,
                summary,
                fact_stage,
                candidate_type,
                evidence_note,
                source_refs_json,
                source_excerpt,
                verification_status,
                round_number
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_id,
                subtopic_id,
                writer_msg_id,
                candidate_text,
                summary,
                fact_stage,
                candidate_type,
                evidence_note,
                source_refs_json,
                source_excerpt,
                verification_status,
                round_number,
            ),
        )
        return cursor.lastrowid


def create_claim_candidate(
    topic_id: int,
    subtopic_id: int,
    clerk_msg_id: Optional[int],
    candidate_text: str,
    summary: Optional[str] = None,
    support_fact_ids_json: Optional[str] = None,
    rationale_short: Optional[str] = None,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO ClaimCandidate (
                topic_id,
                subtopic_id,
                clerk_msg_id,
                candidate_text,
                summary,
                support_fact_ids_json,
                rationale_short
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (topic_id, subtopic_id, clerk_msg_id, candidate_text, summary, support_fact_ids_json, rationale_short),
        )
        return cursor.lastrowid


def get_fact_candidates(
    topic_id: int,
    subtopic_id: Optional[int] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    with get_db() as conn:
        clauses = ["topic_id = ?"]
        params: list[Any] = [topic_id]
        if subtopic_id is not None:
            clauses.append("subtopic_id = ?")
            params.append(subtopic_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        rows = conn.execute(
            f"SELECT * FROM FactCandidate WHERE {' AND '.join(clauses)} ORDER BY id ASC",
            params,
        ).fetchall()
        return [dict(row) for row in rows]


def get_claim_candidates(
    topic_id: int,
    subtopic_id: Optional[int] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    with get_db() as conn:
        clauses = ["topic_id = ?"]
        params: list[Any] = [topic_id]
        if subtopic_id is not None:
            clauses.append("subtopic_id = ?")
            params.append(subtopic_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        try:
            rows = conn.execute(
                f"SELECT * FROM ClaimCandidate WHERE {' AND '.join(clauses)} ORDER BY id ASC",
                params,
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                return []
            raise
        return [dict(row) for row in rows]


def fact_candidate_exists(
    topic_id: int,
    candidate_text: str,
    statuses: Optional[Iterable[str]] = None,
) -> bool:
    with get_db() as conn:
        params: list[Any] = [topic_id, candidate_text]
        query = """
            SELECT 1
            FROM FactCandidate
            WHERE topic_id = ? AND candidate_text = ?
        """
        status_list = [status for status in (statuses or []) if status]
        if status_list:
            placeholders = ", ".join("?" for _ in status_list)
            query += f" AND status IN ({placeholders})"
            params.extend(status_list)
        query += " LIMIT 1"
        row = conn.execute(query, params).fetchone()
        return row is not None


def claim_candidate_exists(
    topic_id: int,
    candidate_text: str,
    statuses: Optional[Iterable[str]] = None,
) -> bool:
    with get_db() as conn:
        params: list[Any] = [topic_id, candidate_text]
        query = """
            SELECT 1
            FROM ClaimCandidate
            WHERE topic_id = ? AND candidate_text = ?
        """
        status_list = [status for status in (statuses or []) if status]
        if status_list:
            placeholders = ", ".join("?" for _ in status_list)
            query += f" AND status IN ({placeholders})"
            params.extend(status_list)
        query += " LIMIT 1"
        try:
            row = conn.execute(query, params).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                return False
            raise
        return row is not None


def update_fact_candidate_review(
    candidate_id: int,
    status: str,
    reviewed_text: Optional[str] = None,
    review_note: Optional[str] = None,
    evidence_note: Optional[str] = None,
    confidence_score: Optional[float] = None,
    reviewer: Optional[str] = None,
    accepted_fact_id: Optional[int] = None,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            UPDATE FactCandidate
            SET status = ?,
                reviewed_text = ?,
                review_note = ?,
                evidence_note = ?,
                confidence_score = ?,
                reviewer = ?,
                accepted_fact_id = ?,
                reviewed_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                status,
                reviewed_text,
                review_note,
                evidence_note,
                confidence_score,
                reviewer,
                accepted_fact_id,
                candidate_id,
            ),
        )


def update_claim_candidate_review(
    candidate_id: int,
    status: str,
    reviewed_text: Optional[str] = None,
    review_note: Optional[str] = None,
    claim_score: Optional[float] = None,
    accepted_claim_id: Optional[int] = None,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            UPDATE ClaimCandidate
            SET status = ?,
                reviewed_text = ?,
                review_note = ?,
                claim_score = ?,
                accepted_claim_id = ?,
                reviewed_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                status,
                reviewed_text,
                review_note,
                claim_score,
                accepted_claim_id,
                candidate_id,
            ),
        )


def fact_exists(topic_id: int, content: str, source: Optional[str] = None) -> bool:
    with get_db() as conn:
        params: list[Any] = [topic_id, content]
        query = """
            SELECT 1
            FROM Fact
            WHERE topic_id = ? AND content = ?
        """
        if source is not None:
            query += " AND source = ?"
            params.append(source)
        query += " LIMIT 1"
        row = conn.execute(query, params).fetchone()
        return row is not None


def get_fact_by_content(topic_id: int, content: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM Fact
            WHERE topic_id = ? AND content = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (topic_id, content),
        ).fetchone()
        return dict(row) if row else None


def get_facts_by_ids(topic_id: int, fact_ids: Iterable[int]) -> List[Dict[str, Any]]:
    ids = [int(fact_id) for fact_id in fact_ids]
    if not ids:
        return []
    placeholders = ", ".join("?" for _ in ids)
    with get_db() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM Fact
            WHERE topic_id = ? AND id IN ({placeholders})
            ORDER BY id ASC
            """,
            [topic_id, *ids],
        ).fetchall()
        return [dict(row) for row in rows]


def insert_claim(
    topic_id: int,
    subtopic_id: Optional[int],
    content: str,
    summary: Optional[str] = None,
    support_fact_ids_json: Optional[str] = None,
    rationale_short: Optional[str] = None,
    claim_score: Optional[float] = None,
    status: str = "active",
    candidate_id: Optional[int] = None,
) -> int:
    content = _truncate(content, MAX_CONTENT_LEN)
    summary = _truncate(summary, MAX_SUMMARY_LEN)
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO Claim (
                topic_id,
                subtopic_id,
                content,
                summary,
                support_fact_ids_json,
                rationale_short,
                claim_score,
                status,
                candidate_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_id,
                subtopic_id,
                content,
                summary,
                support_fact_ids_json,
                rationale_short,
                claim_score,
                status,
                candidate_id,
            ),
        )
        claim_id = cursor.lastrowid
        fts_content = content
        if summary:
            fts_content = f"{summary}\n\n{content}"
        _insert_claim_fts(conn, claim_id, topic_id, fts_content)
        return claim_id


def insert_web_evidence(
    origin_topic_id: int,
    origin_subtopic_id: Optional[int],
    query_text: str,
    title: str,
    snippet: str,
    url: str,
    source_domain: str,
    result_rank: int,
    search_provider: str,
    search_role: str,
    summary: Optional[str] = None,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO WebEvidence (
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
                summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
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
                summary,
            ),
        )
        web_id = cursor.lastrowid
        content = "\n\n".join(
            part.strip()
            for part in (title or "", summary or "", snippet or "", query_text or "", source_domain or "")
            if isinstance(part, str) and part.strip()
        )
        _insert_web_evidence_fts(conn, web_id, origin_topic_id, source_domain or "", content)
        return web_id


def insert_vote_record(
    topic_id: int,
    subtopic_id: Optional[int],
    round_number: Optional[int],
    vote_kind: str,
    subject: str,
    prompt_text: str,
    voter: str,
    parsed_ok: bool,
    decision: Optional[str],
    reason: Optional[str],
    raw_response: str,
    metadata_json: Optional[str] = None,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO VoteRecord (
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
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_id,
                subtopic_id,
                round_number,
                vote_kind,
                subject,
                prompt_text,
                voter,
                int(parsed_ok),
                decision,
                reason,
                raw_response,
                metadata_json,
            ),
        )
        return cursor.lastrowid


def get_vote_records(
    topic_id: int,
    *,
    subtopic_id: Optional[int] = None,
    vote_kind: Optional[str] = None,
    round_number: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    clauses = ["topic_id = ?"]
    params: list[Any] = [topic_id]
    if subtopic_id is not None:
        clauses.append("subtopic_id = ?")
        params.append(subtopic_id)
    if vote_kind is not None:
        clauses.append("vote_kind = ?")
        params.append(vote_kind)
    if round_number is not None:
        clauses.append("round_number = ?")
        params.append(round_number)

    query = f"SELECT * FROM VoteRecord WHERE {' AND '.join(clauses)} ORDER BY id ASC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

def search_facts(topic_id: int, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for the most semantically similar facts using sqlite-vec."""
    with get_db() as conn:
        query = """
            SELECT Fact.*, vec_distance_L2(vec_facts.embedding, ?) as distance
            FROM vec_facts
            JOIN Fact ON Fact.id = vec_facts.fact_id
            WHERE Fact.topic_id = ? AND (Fact.review_status IS NULL OR Fact.review_status != 'superseded')
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


def search_claims_lexical(topic_id: int, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    match_query = _build_fts_query(query_text)
    if not match_query:
        return []

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT Claim.*, bm25(claims_fts) as lexical_score
            FROM claims_fts
            JOIN Claim ON Claim.id = claims_fts.rowid
            WHERE claims_fts MATCH ? AND Claim.topic_id = ?
            ORDER BY lexical_score
            LIMIT ?
            """,
            (match_query, topic_id, top_k),
        ).fetchall()
        return [dict(row) for row in rows]


def search_web_evidence_same_topic(
    topic_id: int,
    query_text: str,
    top_k: int = 5,
    max_age_days: int = 30,
) -> List[Dict[str, Any]]:
    match_query = _build_fts_query(query_text)
    if not match_query:
        return []

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                WebEvidence.*,
                TRIM(
                    COALESCE(WebEvidence.title, '') || ' ' ||
                    COALESCE(WebEvidence.snippet, '') || ' ' ||
                    COALESCE(WebEvidence.query_text, '') || ' ' ||
                    COALESCE(WebEvidence.source_domain, '')
                ) AS content,
                bm25(web_evidence_fts) AS lexical_score
            FROM web_evidence_fts
            JOIN WebEvidence ON WebEvidence.id = web_evidence_fts.rowid
            WHERE web_evidence_fts MATCH ?
              AND WebEvidence.origin_topic_id = ?
              AND WebEvidence.fetched_at >= datetime('now', ?)
            ORDER BY lexical_score
            LIMIT ?
            """,
            (match_query, topic_id, f"-{int(max_age_days)} days", top_k),
        ).fetchall()
        return [dict(row) for row in rows]


def search_web_evidence_cross_topic(
    topic_id: int,
    query_text: str,
    top_k: int = 5,
    max_age_days: int = 30,
) -> List[Dict[str, Any]]:
    match_query = _build_fts_query(query_text)
    if not match_query:
        return []

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                WebEvidence.*,
                TRIM(
                    COALESCE(WebEvidence.title, '') || ' ' ||
                    COALESCE(WebEvidence.snippet, '') || ' ' ||
                    COALESCE(WebEvidence.query_text, '') || ' ' ||
                    COALESCE(WebEvidence.source_domain, '')
                ) AS content,
                bm25(web_evidence_fts) AS lexical_score
            FROM web_evidence_fts
            JOIN WebEvidence ON WebEvidence.id = web_evidence_fts.rowid
            WHERE web_evidence_fts MATCH ?
              AND WebEvidence.origin_topic_id != ?
              AND WebEvidence.fetched_at >= datetime('now', ?)
            ORDER BY lexical_score
            LIMIT ?
            """,
            (match_query, topic_id, f"-{int(max_age_days)} days", top_k),
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
    round_number: Optional[int] = None,
    turn_kind: Optional[str] = None,
    summary: Optional[str] = None,
) -> int:
    """Insert a message and its embedding."""
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO Message (topic_id, subtopic_id, sender, content, summary, msg_type, confidence_score, round_number, turn_kind)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (topic_id, subtopic_id, sender, content, summary, msg_type, confidence_score, round_number, turn_kind)
        )
        msg_id = cursor.lastrowid
        # For FTS, we combine content and summary
        fts_content = content
        if summary:
            fts_content = f"{summary}\n\n{content}"
            
        _insert_message_fts(conn, msg_id, topic_id, sender, fts_content, msg_type)
        
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
    conclusion = _truncate(conclusion, MAX_CONTENT_LEN)
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


def get_web_evidence_for_topic(topic_id: int) -> List[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT id, query_text, title, snippet, source_domain, url FROM WebEvidence WHERE origin_topic_id = ? ORDER BY id DESC",
            (topic_id,)
        )
        return [dict(row) for row in cursor.fetchall()]


def supersede_facts(fact_ids: List[int]) -> None:
    # TODO: Need a more stable, graph-based way to handle fact invalidation and cascading claim invalidation
    if not fact_ids:
        return
    with get_db() as conn:
        placeholders = ",".join("?" * len(fact_ids))
        conn.execute(
            f"UPDATE Fact SET review_status = 'superseded' WHERE id IN ({placeholders})",
            fact_ids
        )


def update_topic_conclusion(topic_id: int, conclusion: str) -> None:
    conclusion = _truncate(conclusion, MAX_CONTENT_LEN)
    with get_db() as conn:
        conn.execute("UPDATE Topic SET conclusion = ? WHERE id = ?", (conclusion, topic_id))
