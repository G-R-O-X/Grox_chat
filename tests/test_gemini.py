import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from grox_chat.external import gemini_cli_client
from grox_chat.external.gemini_cli_client import _extract_text, warmup_gemini_cli, _map_model

def test_extract_text_gemini():
    mock_response = {
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "{\"action\": \"post_message\", \"content\": \"hello\"}"}
                        ]
                    }
                }
            ]
        }
    }
    
    text = _extract_text(mock_response)
    assert '{"action": "post_message"' in text
    assert "hello" in text

def test_extract_text_empty():
    with pytest.raises(RuntimeError, match="Empty response"):
        _extract_text({"response": {}})


def test_map_model_preserves_gemini_3_1_pro_preview():
    assert _map_model("gemini-3.1-pro-preview") == "gemini-3.1-pro-preview"


@pytest.mark.asyncio
async def test_query_gemini_cli_coalesces_duplicate_inflight_requests():
    gemini_cli_client._inflight_requests.clear()
    gemini_cli_client._broker_lock = None
    gemini_cli_client._request_semaphore = None

    async def fake_uncached(**kwargs):
        await asyncio.sleep(0.01)
        return "shared answer"

    with patch.object(
        gemini_cli_client,
        "_query_gemini_cli_uncached",
        new=AsyncMock(side_effect=fake_uncached),
    ) as uncached:
        results = await asyncio.gather(
            gemini_cli_client.query_gemini_cli("prompt", model="gemini-3.0-flash"),
            gemini_cli_client.query_gemini_cli("prompt", model="gemini-3.0-flash"),
        )

    assert results == ["shared answer", "shared answer"]
    uncached.assert_awaited_once()


@pytest.mark.asyncio
async def test_warmup_gemini_cli_primes_headers_and_project():
    with patch.object(
        gemini_cli_client,
        "_build_headers",
        new=AsyncMock(return_value={"Authorization": "Bearer test"}),
    ) as build_headers:
        with patch.object(
            gemini_cli_client,
            "_discover_project_id",
            new=AsyncMock(return_value="project-123"),
        ) as discover_project:
            await warmup_gemini_cli()

    build_headers.assert_awaited_once()
    discover_project.assert_awaited_once()
