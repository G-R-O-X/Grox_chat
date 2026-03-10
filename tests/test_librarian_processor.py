import pytest
from unittest.mock import AsyncMock, patch

from grox_chat.librarian_processor import (
    apply_librarian_review,
    build_librarian_audit_message,
    parse_librarian_review,
)


def test_parse_librarian_review_requires_rewritten_text_for_soften():
    with pytest.raises(ValueError, match="Softened fact must include rewritten reviewed text."):
        parse_librarian_review(
            '{"decision":"soften","review_note":"too absolute","evidence_note":"search found mixed evidence","confidence_score":8}',
            "Original claim",
        )


def test_parse_librarian_review_accept_defaults_reviewed_text():
    review = parse_librarian_review(
        '{"decision":"accept","review_note":"supported","evidence_note":"matched source","confidence_score":8}',
        "Original claim",
    )

    assert review["decision"] == "accept"
    assert review["reviewed_text"] == "Original claim"
    assert review["confidence_score"] == 8.0


@pytest.mark.asyncio
async def test_apply_librarian_review_accepts_and_inserts_fact():
    candidate = {"id": 5, "candidate_text": "Verified fact"}

    with patch("grox_chat.librarian_processor.api.get_fact_by_content", return_value=None):
        with patch("grox_chat.librarian_processor.aget_embedding", new=AsyncMock(return_value=[0.1] * 384)):
            with patch("grox_chat.librarian_processor.api.insert_fact_with_embedding", return_value=12) as insert_fact:
                with patch("grox_chat.librarian_processor.api.update_fact_candidate_review") as update_candidate:
                    result = await apply_librarian_review(
                        1,
                        candidate,
                        {
                            "decision": "accept",
                            "reviewed_text": "Verified fact",
                            "review_note": "Supported by evidence.",
                            "evidence_note": "Matched search result.",
                            "confidence_score": 9.0,
                        },
                    )

    assert result["accepted_fact_id"] == 12
    insert_fact.assert_called_once_with(
        1,
        "Verified fact",
        source="Librarian",
        embedding=[0.1] * 384,
        candidate_id=5,
        review_status="accept",
        evidence_note="Matched search result.",
        confidence_score=9.0,
    )
    update_candidate.assert_called_once()


@pytest.mark.asyncio
async def test_apply_librarian_review_softens_and_inserts_plain_fact_when_embedding_fails():
    candidate = {"id": 6, "candidate_text": "There is no evidence at all"}

    with patch("grox_chat.librarian_processor.api.get_fact_by_content", return_value=None):
        with patch("grox_chat.librarian_processor.aget_embedding", new=AsyncMock(return_value=None)):
            with patch("grox_chat.librarian_processor.api.insert_fact", return_value=14) as insert_fact:
                with patch("grox_chat.librarian_processor.api.update_fact_candidate_review") as update_candidate:
                    result = await apply_librarian_review(
                        1,
                        candidate,
                        {
                            "decision": "soften",
                            "reviewed_text": "No supporting empirical evidence was found in the current retrieval set.",
                            "review_note": "Absolute claim softened.",
                            "evidence_note": "Current retrieval set was limited.",
                            "confidence_score": 6.5,
                        },
                    )

    assert result["accepted_fact_id"] == 14
    insert_fact.assert_called_once_with(
        1,
        "No supporting empirical evidence was found in the current retrieval set.",
        source="Librarian",
        candidate_id=6,
        review_status="soften",
        evidence_note="Current retrieval set was limited.",
        confidence_score=6.5,
    )
    update_candidate.assert_called_once()


@pytest.mark.asyncio
async def test_apply_librarian_review_rejects_without_inserting_fact():
    candidate = {"id": 7, "candidate_text": "Unsupported claim"}

    with patch("grox_chat.librarian_processor.api.insert_fact") as insert_fact:
        with patch("grox_chat.librarian_processor.api.insert_fact_with_embedding") as insert_fact_with_embedding:
            with patch("grox_chat.librarian_processor.api.update_fact_candidate_review") as update_candidate:
                result = await apply_librarian_review(
                    1,
                    candidate,
                    {
                        "decision": "reject",
                        "reviewed_text": None,
                        "review_note": "Unsupported by retrieved evidence.",
                        "evidence_note": "Search results did not confirm the claim.",
                        "confidence_score": 3.0,
                    },
                )

    assert result["accepted_fact_id"] is None
    insert_fact.assert_not_called()
    insert_fact_with_embedding.assert_not_called()
    update_candidate.assert_called_once()


def test_build_librarian_audit_message_groups_decisions():
    audit = build_librarian_audit_message(
        [
            {"candidate_id": 1, "decision": "accept", "reviewed_text": "Accepted fact"},
            {"candidate_id": 2, "decision": "soften", "reviewed_text": "Softened fact"},
            {"candidate_id": 3, "decision": "reject", "review_note": "Too speculative"},
        ]
    )

    assert "ACCEPTED:" in audit
    assert "SOFTENED:" in audit
    assert "REJECTED:" in audit
