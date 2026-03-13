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
    assert review["verification_status"] == "accepted"


@pytest.mark.asyncio
async def test_apply_librarian_review_accepts_and_inserts_fact():
    candidate = {"id": 5, "candidate_text": "Verified fact", "fact_stage": "bootstrap"}

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
    assert insert_fact.call_args.args == (1, "Verified fact")
    assert insert_fact.call_args.kwargs["source"] == "Librarian"
    assert insert_fact.call_args.kwargs["embedding"] == [0.1] * 384
    assert insert_fact.call_args.kwargs["fact_stage"] == "bootstrap"
    assert insert_fact.call_args.kwargs["fact_type"] == "sourced_claim"
    assert insert_fact.call_args.kwargs["candidate_id"] == 5
    assert insert_fact.call_args.kwargs["review_status"] == "accept"
    assert insert_fact.call_args.kwargs["evidence_note"] == "Matched search result."
    assert insert_fact.call_args.kwargs["confidence_score"] == 9.0
    update_candidate.assert_called_once()


@pytest.mark.asyncio
async def test_apply_librarian_review_softens_and_inserts_plain_fact_when_embedding_fails():
    candidate = {"id": 6, "candidate_text": "There is no evidence at all", "fact_stage": "inline"}

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
    assert insert_fact.call_args.args == (1, "No supporting empirical evidence was found in the current retrieval set.")
    assert insert_fact.call_args.kwargs["source"] == "Librarian"
    assert insert_fact.call_args.kwargs["fact_stage"] == "inline"
    assert insert_fact.call_args.kwargs["fact_type"] == "sourced_claim"
    assert insert_fact.call_args.kwargs["candidate_id"] == 6
    assert insert_fact.call_args.kwargs["review_status"] == "soften"
    assert insert_fact.call_args.kwargs["evidence_note"] == "Current retrieval set was limited."
    assert insert_fact.call_args.kwargs["confidence_score"] == 6.5
    update_candidate.assert_called_once()


@pytest.mark.asyncio
async def test_apply_librarian_review_rejects_without_inserting_fact():
    candidate = {"id": 7, "candidate_text": "Unsupported claim", "fact_stage": "synthesized"}

    with patch("grox_chat.librarian_processor.api.get_fact_by_content", return_value=None):
        with patch("grox_chat.librarian_processor.aget_embedding", new=AsyncMock(return_value=None)):
            with patch("grox_chat.librarian_processor.api.insert_fact", return_value=22) as insert_fact:
                with patch("grox_chat.librarian_processor.api.insert_fact_with_embedding") as insert_fact_with_embedding:
                    with patch("grox_chat.librarian_processor.api.update_fact_candidate_review") as update_candidate:
                        result = await apply_librarian_review(
                            1,
                            candidate,
                            {
                                "decision": "reject",
                                "verification_status": "unsupported",
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

    assert "FACT ACCEPTED:" in audit
    assert "FACT SOFTENED:" in audit
    assert "FACT REJECTED:" in audit
