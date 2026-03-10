import logging
from typing import Iterable, List

from . import api
from .embedding import aget_embedding

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


async def _store_facts(topic_id: int, facts: Iterable[str]) -> None:
    seen: set[str] = set()
    for fact_content in facts:
        if not isinstance(fact_content, str):
            continue
        normalized = _normalize_fact_text(fact_content)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if api.fact_exists(topic_id, normalized, source="Writer"):
            logger.info(f"[Writer Processor] Skipping duplicate fact: {normalized[:50]}...")
            continue
        logger.info(f"[Writer Processor] Extracting and embedding fact: {normalized[:50]}...")
        emb = await aget_embedding(normalized)
        if emb:
            fact_id = api.insert_fact_with_embedding(topic_id, normalized, source="Writer", embedding=emb)
            logger.info(f"[Writer Processor] Inserted Fact ID: {fact_id}")
        else:
            logger.warning("[Writer Processor] Failed to embed fact.")


async def process_writer_output(topic_id: int, writer_text: str, structured_facts: List[str] | None = None):
    """
    Parses the writer's output to extract specific verified facts.
    If facts are found, embeds them and inserts them into the Fact table.
    We expect the Writer to output facts in a structured way, or we use regex to extract them.
    For simplicity, let's assume the Writer lists verified facts starting with 'FACT:' or 'VERIFIED:'.
    """
    facts = structured_facts if structured_facts is not None else _extract_fact_lines(writer_text)
    await _store_facts(topic_id, facts)
