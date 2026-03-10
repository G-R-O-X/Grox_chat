import logging
from typing import Iterable, List, Optional

from . import api

logger = logging.getLogger(__name__)


def _normalize_fact_text(text: str) -> str:
    return " ".join(text.split())


def _extract_fact_lines(writer_text: str) -> List[str]:
    fact_lines = []
    for line in writer_text.split('\n'):
        stripped = line.strip()
        if stripped.startswith("FACT:") or stripped.startswith("VERIFIED:"):
            fact_lines.append(stripped.replace("FACT:", "").replace("VERIFIED:", "").strip())
    return fact_lines


async def _store_fact_candidates(
    topic_id: int,
    subtopic_id: int,
    writer_msg_id: Optional[int],
    facts: Iterable[str],
    max_candidates: int | None = None,
) -> list[int]:
    seen: set[str] = set()
    created_ids: list[int] = []
    for fact_content in facts:
        if not isinstance(fact_content, str):
            continue
        normalized = _normalize_fact_text(fact_content)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if api.fact_exists(topic_id, normalized):
            logger.info("[Writer Processor] Skipping final fact duplicate: %s...", normalized[:50])
            continue
        if api.fact_candidate_exists(topic_id, normalized, statuses=("pending",)):
            logger.info("[Writer Processor] Skipping pending fact candidate duplicate: %s...", normalized[:50])
            continue
        candidate_id = api.create_fact_candidate(topic_id, subtopic_id, writer_msg_id, normalized)
        created_ids.append(candidate_id)
        logger.info("[Writer Processor] Created FactCandidate ID: %s", candidate_id)
        if max_candidates is not None and len(created_ids) >= max_candidates:
            break
    return created_ids


async def process_writer_output(
    topic_id: int,
    subtopic_id: int,
    writer_msg_id: Optional[int],
    writer_text: str,
    structured_facts: List[str] | None = None,
    max_candidates: int | None = None,
) -> list[int]:
    """
    Stores structured candidate facts in FactCandidate after normalization and deduplication.
    Permanent Fact insertion is delegated to the Librarian review stage.
    """
    facts = structured_facts if structured_facts is not None else _extract_fact_lines(writer_text)
    return await _store_fact_candidates(
        topic_id,
        subtopic_id,
        writer_msg_id,
        facts,
        max_candidates=max_candidates,
    )
