import pytest
from unittest.mock import patch

from grox_chat.writer_processor import process_writer_output


@pytest.mark.asyncio
async def test_process_writer_output_respects_explicit_empty_facts():
    with patch("grox_chat.writer_processor.api.fact_exists", return_value=False):
        with patch("grox_chat.writer_processor.api.fact_candidate_exists", return_value=False):
            with patch("grox_chat.writer_processor.api.create_fact_candidate_with_stage") as create_candidate:
                created = await process_writer_output(
                    1,
                    2,
                    3,
                    "Writer quoted earlier notes:\nFACT: old quoted line",
                    structured_facts=[],
                )

    assert created == []
    create_candidate.assert_not_called()


@pytest.mark.asyncio
async def test_process_writer_output_deduplicates_duplicate_structured_facts():
    with patch("grox_chat.writer_processor.api.fact_exists", return_value=False):
        with patch("grox_chat.writer_processor.api.fact_candidate_exists", return_value=False):
            with patch(
                "grox_chat.writer_processor.api.create_fact_candidate_with_stage",
                side_effect=[11],
            ) as create_candidate:
                created = await process_writer_output(
                    1,
                    2,
                    3,
                    "unused",
                    structured_facts=["A   verified fact", "A verified fact"],
                )

    assert created == [11]
    create_candidate.assert_called_once_with(
        1,
        2,
        3,
        "A verified fact",
        fact_stage="synthesized",
        evidence_note=None,
    )


@pytest.mark.asyncio
async def test_process_writer_output_skips_existing_final_fact():
    with patch("grox_chat.writer_processor.api.fact_exists", return_value=True):
        with patch("grox_chat.writer_processor.api.fact_candidate_exists", return_value=False):
            with patch("grox_chat.writer_processor.api.create_fact_candidate_with_stage") as create_candidate:
                created = await process_writer_output(
                    1,
                    2,
                    3,
                    "unused",
                    structured_facts=["Already stored fact"],
                )

    assert created == []
    create_candidate.assert_not_called()


@pytest.mark.asyncio
async def test_process_writer_output_ignores_non_string_fact_entries():
    with patch("grox_chat.writer_processor.api.fact_exists", return_value=False):
        with patch("grox_chat.writer_processor.api.fact_candidate_exists", return_value=False):
            with patch(
                "grox_chat.writer_processor.api.create_fact_candidate_with_stage",
                side_effect=[17],
            ) as create_candidate:
                created = await process_writer_output(
                    1,
                    2,
                    3,
                    "unused",
                    structured_facts=[{"bad": 1}, 99, "Valid fact"],
                )

    assert created == [17]
    create_candidate.assert_called_once_with(
        1,
        2,
        3,
        "Valid fact",
        fact_stage="synthesized",
        evidence_note=None,
    )


@pytest.mark.asyncio
async def test_process_writer_output_caps_candidates():
    with patch("grox_chat.writer_processor.api.fact_exists", return_value=False):
        with patch("grox_chat.writer_processor.api.fact_candidate_exists", return_value=False):
            with patch(
                "grox_chat.writer_processor.api.create_fact_candidate_with_stage",
                side_effect=[1, 2],
            ) as create_candidate:
                created = await process_writer_output(
                    1,
                    2,
                    3,
                    "unused",
                    structured_facts=["Fact one", "Fact two", "Fact three"],
                    max_candidates=2,
                )

    assert created == [1, 2]
    assert create_candidate.call_count == 2


@pytest.mark.asyncio
async def test_process_writer_output_caps_after_filtering_duplicates():
    with patch("grox_chat.writer_processor.api.fact_exists", return_value=False):
        with patch("grox_chat.writer_processor.api.fact_candidate_exists", return_value=False):
            with patch(
                "grox_chat.writer_processor.api.create_fact_candidate_with_stage",
                side_effect=[1, 2],
            ) as create_candidate:
                created = await process_writer_output(
                    1,
                    2,
                    None,
                    "unused",
                    structured_facts=["Fact one", "Fact one", "Fact two"],
                    max_candidates=2,
                )

    assert created == [1, 2]
    assert create_candidate.call_count == 2


@pytest.mark.asyncio
async def test_process_writer_output_caps_after_filtering_blank_entries():
    with patch("grox_chat.writer_processor.api.fact_exists", return_value=False):
        with patch("grox_chat.writer_processor.api.fact_candidate_exists", return_value=False):
            with patch(
                "grox_chat.writer_processor.api.create_fact_candidate_with_stage",
                side_effect=[5],
            ) as create_candidate:
                created = await process_writer_output(
                    1,
                    2,
                    None,
                    "unused",
                    structured_facts=["  ", "Fact two"],
                    max_candidates=2,
                )

    assert created == [5]
    create_candidate.assert_called_once_with(
        1,
        2,
        None,
        "Fact two",
        fact_stage="synthesized",
        evidence_note=None,
    )
