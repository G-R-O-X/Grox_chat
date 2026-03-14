import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import api
from .embedding import aget_embedding

logger = logging.getLogger(__name__)

VALID_DECISIONS = {"accept", "soften", "reject"}
VALID_FACT_DECISIONS = {"accept", "correct", "soften", "reject"}
VALID_CLAIM_DECISIONS = {"accept", "soften", "reject"}


def _normalize_fact_text(text: str) -> str:
    return " ".join((text or "").split())


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None


def _clamp_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return max(0.0, min(10.0, float(value)))
    except (TypeError, ValueError):
        return None


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _normalize_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    ids: list[int] = []
    for item in value:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return ids


def parse_librarian_review(raw_text: str, candidate_text: str) -> Dict[str, Any]:
    parsed = _extract_json(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("Librarian review did not return valid JSON.")

    decision = str(parsed.get("decision", "")).strip().lower()
    if decision not in VALID_FACT_DECISIONS:
        raise ValueError(f"Invalid librarian decision: {decision}")

    verification_status = str(parsed.get("verification_status", "")).strip().lower()
    if verification_status not in {"accepted", "corrected", "unsupported", "refuted"}:
        verification_status = {
            "accept": "accepted",
            "correct": "corrected",
            "soften": "accepted",
            "reject": "unsupported",
        }[decision]

    reviewed_text = parsed.get("reviewed_text")
    summary = parsed.get("summary")
    if decision in {"accept", "correct"}:
        if not isinstance(reviewed_text, str) or not reviewed_text.strip():
            reviewed_text = candidate_text
        reviewed_text = _normalize_fact_text(reviewed_text)
        if not reviewed_text:
            raise ValueError("Accepted fact must include reviewed text.")
    elif decision == "soften":
        if not isinstance(reviewed_text, str) or not reviewed_text.strip():
            raise ValueError("Softened fact must include rewritten reviewed text.")
        reviewed_text = _normalize_fact_text(reviewed_text)
        if not reviewed_text:
            raise ValueError("Softened fact must include rewritten reviewed text.")
    else:
        reviewed_text = None

    review_note = parsed.get("review_note")
    if not isinstance(review_note, str):
        review_note = ""

    evidence_note = parsed.get("evidence_note")
    if not isinstance(evidence_note, str):
        evidence_note = ""

    return {
        "decision": decision,
        "verification_status": verification_status,
        "reviewed_text": reviewed_text,
        "summary": summary,
        "review_note": review_note.strip(),
        "evidence_note": evidence_note.strip(),
        "source_refs": _normalize_string_list(parsed.get("source_refs_json") or parsed.get("source_refs")),
        "source_excerpt": parsed.get("source_excerpt", "").strip() if isinstance(parsed.get("source_excerpt"), str) else "",
        "confidence_score": _clamp_confidence(parsed.get("confidence_score")),
    }


async def apply_librarian_review(
    topic_id: int,
    candidate: Dict[str, Any],
    review: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_id = candidate["id"]
    decision = review["decision"]
    verification_status = review.get("verification_status")
    reviewed_text = review.get("reviewed_text")
    summary = review.get("summary")
    review_note = review.get("review_note")
    evidence_note = review.get("evidence_note")
    source_refs = review.get("source_refs") or []
    source_excerpt = review.get("source_excerpt") or ""
    confidence_score = review.get("confidence_score")

    accepted_fact_id = None
    stored_text = None

    if decision in {"accept", "correct", "soften"}:
        stored_text = _normalize_fact_text(reviewed_text or candidate["candidate_text"])
        existing_fact = api.get_fact_by_content(topic_id, stored_text)
        if existing_fact:
            accepted_fact_id = existing_fact["id"]
            logger.info("[Librarian] Reusing existing fact %s for candidate %s", accepted_fact_id, candidate_id)
        else:
            embedding = await aget_embedding(summary or stored_text)
            insert_kwargs = {
                "subtopic_id": candidate.get("subtopic_id"),
                "fact_stage": candidate.get("fact_stage", "synthesized"),
                "fact_type": candidate.get("candidate_type", "sourced_claim"),
                "verification_status": verification_status,
                "source_kind": candidate.get("source_kind") or (
                    "internal_verification" if candidate.get("candidate_type") == "number" else "web"
                ),
                "source_refs_json": json.dumps(source_refs, ensure_ascii=True) if source_refs else candidate.get("source_refs_json"),
                "source_excerpt": source_excerpt or candidate.get("source_excerpt"),
                "candidate_id": candidate_id,
                "review_status": decision,
                "evidence_note": evidence_note or None,
                "confidence_score": confidence_score,
            }
            if embedding:
                accepted_fact_id = api.insert_fact_with_embedding(
                    topic_id,
                    stored_text,
                    source="Librarian",
                    embedding=embedding,
                    summary=summary,
                    **insert_kwargs,
                )
            else:
                accepted_fact_id = api.insert_fact(
                    topic_id,
                    stored_text,
                    source="Librarian",
                    summary=summary,
                    **insert_kwargs,
                )

    api.update_fact_candidate_review(
        candidate_id,
        decision,
        reviewed_text=stored_text,
        review_note=review_note or None,
        evidence_note=evidence_note or None,
        confidence_score=confidence_score,
        reviewer="Librarian",
        accepted_fact_id=accepted_fact_id,
    )

    return {
        "candidate_id": candidate_id,
        "record_kind": "fact",
        "candidate_text": candidate["candidate_text"],
        "decision": decision,
        "verification_status": verification_status,
        "reviewed_text": stored_text,
        "review_note": review_note or "",
        "evidence_note": evidence_note or "",
        "confidence_score": confidence_score,
        "accepted_fact_id": accepted_fact_id,
    }


def parse_claim_review(raw_text: str, candidate_text: str, fallback_support_ids: list[int]) -> Dict[str, Any]:
    parsed = _extract_json(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("Claim review did not return valid JSON.")

    decision = str(parsed.get("decision", "")).strip().lower()
    if decision not in VALID_CLAIM_DECISIONS:
        raise ValueError(f"Invalid claim review decision: {decision}")

    reviewed_text = parsed.get("reviewed_text")
    summary = parsed.get("summary")
    if decision == "accept":
        if not isinstance(reviewed_text, str) or not reviewed_text.strip():
            reviewed_text = candidate_text
        reviewed_text = _normalize_fact_text(reviewed_text)
    elif decision == "soften":
        if not isinstance(reviewed_text, str) or not reviewed_text.strip():
            raise ValueError("Softened claim must include reviewed text.")
        reviewed_text = _normalize_fact_text(reviewed_text)
    else:
        reviewed_text = None

    review_note = parsed.get("review_note")
    if not isinstance(review_note, str):
        review_note = ""

    supported_fact_ids = _normalize_int_list(parsed.get("supported_fact_ids") or parsed.get("support_fact_ids"))
    if not supported_fact_ids:
        supported_fact_ids = list(fallback_support_ids)

    return {
        "decision": decision,
        "reviewed_text": reviewed_text,
        "summary": summary,
        "review_note": review_note.strip(),
        "supported_fact_ids": supported_fact_ids,
        "claim_score": _clamp_confidence(parsed.get("claim_score")),
    }


def _filter_valid_supported_fact_ids(topic_id: int, fact_ids: Sequence[int]) -> list[int]:
    if not fact_ids:
        return []
    facts = api.get_facts_by_ids(topic_id, list(fact_ids))
    valid_ids = {fact["id"] for fact in facts}
    invalid_ids = [fid for fid in fact_ids if fid not in valid_ids]
    if invalid_ids:
        logger.warning(
            "[claim] Dropping unsupported fact_ids=%s for topic=%s",
            invalid_ids,
            topic_id,
        )
    return [fid for fid in fact_ids if fid in valid_ids]


async def apply_claim_review(
    topic_id: int,
    candidate: Dict[str, Any],
    review: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_id = candidate["id"]
    decision = review["decision"]
    reviewed_text = review.get("reviewed_text")
    review_note = review.get("review_note")
    claim_score = review.get("claim_score")
    supported_fact_ids = review.get("supported_fact_ids") or []
    supported_fact_ids = _filter_valid_supported_fact_ids(topic_id, supported_fact_ids)

    accepted_claim_id = None
    stored_text = None
    if decision in {"accept", "soften"} and supported_fact_ids:
        stored_text = _normalize_fact_text(reviewed_text or candidate["candidate_text"])
        accepted_claim_id = api.insert_claim(
            topic_id,
            candidate.get("subtopic_id"),
            stored_text,
            support_fact_ids_json=json.dumps(supported_fact_ids, ensure_ascii=True),
            rationale_short=candidate.get("rationale_short"),
            claim_score=claim_score,
            status="active" if decision == "accept" else "contested",
            candidate_id=candidate_id,
        )

    api.update_claim_candidate_review(
        candidate_id,
        decision,
        reviewed_text=stored_text,
        review_note=review_note or None,
        claim_score=claim_score,
        accepted_claim_id=accepted_claim_id,
    )

    return {
        "candidate_id": candidate_id,
        "record_kind": "claim",
        "candidate_text": candidate["candidate_text"],
        "decision": decision,
        "reviewed_text": stored_text,
        "review_note": review_note or "",
        "claim_score": claim_score,
        "accepted_claim_id": accepted_claim_id,
    }


def build_librarian_audit_message(results: Iterable[Dict[str, Any]]) -> str:
    fact_accepted: List[str] = []
    fact_corrected: List[str] = []
    fact_softened: List[str] = []
    fact_rejected: List[str] = []
    claim_accepted: List[str] = []
    claim_softened: List[str] = []
    claim_rejected: List[str] = []

    for result in results:
        decision = result["decision"]
        record_kind = result.get("record_kind", "fact")
        if record_kind == "claim":
            if decision == "accept":
                claim_accepted.append(f"- Claim Candidate {result['candidate_id']}: {result['reviewed_text']}")
            elif decision == "soften":
                claim_softened.append(
                    f"- Claim Candidate {result['candidate_id']}: softened to `{result['reviewed_text']}`"
                )
            else:
                note = result.get("review_note") or "unsupported derivation"
                claim_rejected.append(f"- Claim Candidate {result['candidate_id']}: {note}")
            continue

        if decision == "accept":
            fact_accepted.append(f"- Candidate {result['candidate_id']}: {result['reviewed_text']}")
        elif decision == "correct":
            fact_corrected.append(f"- Candidate {result['candidate_id']}: corrected to `{result['reviewed_text']}`")
        elif decision == "soften":
            fact_softened.append(
                f"- Candidate {result['candidate_id']}: softened to `{result['reviewed_text']}`"
            )
        else:
            note = result.get("review_note") or "unsupported or too interpretive"
            fact_rejected.append(f"- Candidate {result['candidate_id']}: {note}")

    sections = ["LIBRARIAN AUDIT:"]
    if fact_accepted:
        sections.append("FACT ACCEPTED:")
        sections.extend(fact_accepted)
    if fact_corrected:
        sections.append("FACT CORRECTED:")
        sections.extend(fact_corrected)
    if fact_softened:
        sections.append("FACT SOFTENED:")
        sections.extend(fact_softened)
    if fact_rejected:
        sections.append("FACT REJECTED:")
        sections.extend(fact_rejected)
    if claim_accepted:
        sections.append("CLAIM ACCEPTED:")
        sections.extend(claim_accepted)
    if claim_softened:
        sections.append("CLAIM SOFTENED:")
        sections.extend(claim_softened)
    if claim_rejected:
        sections.append("CLAIM REJECTED:")
        sections.extend(claim_rejected)
    return "\n".join(sections)
