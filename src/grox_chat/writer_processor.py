import logging
import json
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
    facts: Iterable[str | dict],
    fact_stage: str = "synthesized",
    evidence_note: Optional[str] = None,
    round_number: int | None = None,
    max_candidates: int | None = None,
) -> list[int]:
    seen: set[str] = set()
    created_ids: list[int] = []
    for fact_item in facts:
        if isinstance(fact_item, dict):
            fact_content = fact_item.get("candidate_text") or fact_item.get("text") or ""
            candidate_type = fact_item.get("candidate_type", "sourced_claim")
            source_refs_json = json.dumps(fact_item.get("source_refs", []), ensure_ascii=True)
            source_excerpt = fact_item.get("source_excerpt")
            verification_status = fact_item.get("verification_status")
        else:
            fact_content = fact_item
            candidate_type = "sourced_claim"
            source_refs_json = None
            source_excerpt = None
            verification_status = None

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
        candidate_id = api.create_fact_candidate_with_stage(
            topic_id,
            subtopic_id,
            writer_msg_id,
            normalized,
            fact_stage=fact_stage,
            candidate_type=candidate_type,
            evidence_note=evidence_note,
            source_refs_json=source_refs_json,
            source_excerpt=source_excerpt,
            verification_status=verification_status,
            round_number=round_number,
        )
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
    fact_stage: str = "synthesized",
    evidence_note: Optional[str] = None,
    round_number: int | None = None,
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
        fact_stage=fact_stage,
        evidence_note=evidence_note,
        round_number=round_number,
        max_candidates=max_candidates,
    )


async def process_clerk_claim_output(
    topic_id: int,
    subtopic_id: int,
    clerk_msg_id: Optional[int],
    claim_candidates: Iterable[dict],
    *,
    max_candidates: int | None = None,
) -> list[int]:
    seen: set[str] = set()
    created_ids: list[int] = []
    for claim in claim_candidates:
        if not isinstance(claim, dict):
            continue
        candidate_text = claim.get("candidate_text") or ""
        if not isinstance(candidate_text, str):
            continue
        normalized = _normalize_fact_text(candidate_text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if api.claim_candidate_exists(topic_id, normalized, statuses=("pending", "accept", "soften")):
            logger.info("[Writer Processor] Skipping claim duplicate: %s...", normalized[:50])
            continue
        support_fact_ids = claim.get("support_fact_ids") or []
        if not isinstance(support_fact_ids, list) or not support_fact_ids:
            continue
        rationale_short = claim.get("rationale_short")
        summary = claim.get("summary", "").strip()
        created_id = api.create_claim_candidate(
            topic_id,
            subtopic_id,
            clerk_msg_id,
            normalized,
            summary=summary,
            support_fact_ids_json=json.dumps(support_fact_ids, ensure_ascii=True),
            rationale_short=rationale_short if isinstance(rationale_short, str) else None,
        )
        created_ids.append(created_id)
        logger.info("[Writer Processor] Created ClaimCandidate ID: %s", created_id)
        if max_candidates is not None and len(created_ids) >= max_candidates:
            break
    return created_ids
