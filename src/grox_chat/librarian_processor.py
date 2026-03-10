import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from . import api
from .embedding import aget_embedding

logger = logging.getLogger(__name__)

VALID_DECISIONS = {"accept", "soften", "reject"}


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


def parse_librarian_review(raw_text: str, candidate_text: str) -> Dict[str, Any]:
    parsed = _extract_json(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("Librarian review did not return valid JSON.")

    decision = str(parsed.get("decision", "")).strip().lower()
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Invalid librarian decision: {decision}")

    reviewed_text = parsed.get("reviewed_text")
    if decision == "accept":
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
        "reviewed_text": reviewed_text,
        "review_note": review_note.strip(),
        "evidence_note": evidence_note.strip(),
        "confidence_score": _clamp_confidence(parsed.get("confidence_score")),
    }


async def apply_librarian_review(
    topic_id: int,
    candidate: Dict[str, Any],
    review: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_id = candidate["id"]
    decision = review["decision"]
    reviewed_text = review.get("reviewed_text")
    review_note = review.get("review_note")
    evidence_note = review.get("evidence_note")
    confidence_score = review.get("confidence_score")

    accepted_fact_id = None
    stored_text = None

    if decision in {"accept", "soften"}:
        stored_text = _normalize_fact_text(reviewed_text or candidate["candidate_text"])
        existing_fact = api.get_fact_by_content(topic_id, stored_text)
        if existing_fact:
            accepted_fact_id = existing_fact["id"]
            logger.info("[Librarian] Reusing existing fact %s for candidate %s", accepted_fact_id, candidate_id)
        else:
            embedding = await aget_embedding(stored_text)
            insert_kwargs = {
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
                    **insert_kwargs,
                )
            else:
                accepted_fact_id = api.insert_fact(
                    topic_id,
                    stored_text,
                    source="Librarian",
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
        "candidate_text": candidate["candidate_text"],
        "decision": decision,
        "reviewed_text": stored_text,
        "review_note": review_note or "",
        "evidence_note": evidence_note or "",
        "confidence_score": confidence_score,
        "accepted_fact_id": accepted_fact_id,
    }


def build_librarian_audit_message(results: Iterable[Dict[str, Any]]) -> str:
    accepted: List[str] = []
    softened: List[str] = []
    rejected: List[str] = []

    for result in results:
        decision = result["decision"]
        if decision == "accept":
            accepted.append(f"- Candidate {result['candidate_id']}: {result['reviewed_text']}")
        elif decision == "soften":
            softened.append(
                f"- Candidate {result['candidate_id']}: softened to `{result['reviewed_text']}`"
            )
        else:
            note = result.get("review_note") or "unsupported or too interpretive"
            rejected.append(f"- Candidate {result['candidate_id']}: {note}")

    sections = ["LIBRARIAN AUDIT:"]
    if accepted:
        sections.append("ACCEPTED:")
        sections.extend(accepted)
    if softened:
        sections.append("SOFTENED:")
        sections.extend(softened)
    if rejected:
        sections.append("REJECTED:")
        sections.extend(rejected)
    return "\n".join(sections)
