import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from grox_chat.external import gemini_cli_client
from grox_chat.external.gemini_cli_client import _extract_text, warmup_gemini_cli, _map_model


class _FakeResponse:
    def __init__(self, status, json_data=None, text_data=""):
        self.status = status
        self._json_data = json_data or {}
        self._text_data = text_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text_data


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)

    def post(self, *args, **kwargs):
        return self._responses.pop(0)

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


@pytest.mark.asyncio
async def test_discover_project_id_waits_for_gemini_slot():
    gemini_cli_client._cached_project_id = None
    gemini_cli_client._project_lock = None

    fake_session = _FakeSession(
        [
            _FakeResponse(
                200,
                json_data={"cloudaicompanionProject": "project-123"},
            )
        ]
    )

    with patch.object(
        gemini_cli_client,
        "_get_aio_session",
        return_value=fake_session,
    ):
        with patch.object(
            gemini_cli_client,
            "_build_headers",
            new=AsyncMock(return_value={"Authorization": "Bearer test"}),
        ):
            with patch.object(
                gemini_cli_client,
                "wait_for_gemini_slot",
                new=AsyncMock(),
            ) as wait_for_gemini_slot:
                project_id = await gemini_cli_client._discover_project_id()

    assert project_id == "project-123"
    wait_for_gemini_slot.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_gemini_rest_waits_for_gemini_slot():
    fake_session = _FakeSession(
        [
            _FakeResponse(
                200,
                json_data={
                    "response": {
                        "candidates": [
                            {"content": {"parts": [{"text": "ok"}]}}
                        ]
                    }
                },
            )
        ]
    )

    with patch.object(
        gemini_cli_client,
        "_discover_project_id",
        new=AsyncMock(return_value="project-123"),
    ):
        with patch.object(
            gemini_cli_client,
            "_build_headers",
            new=AsyncMock(return_value={"Authorization": "Bearer test"}),
        ):
            with patch.object(
                gemini_cli_client,
                "_get_aio_session",
                return_value=fake_session,
            ):
                with patch.object(
                    gemini_cli_client,
                    "wait_for_gemini_slot",
                    new=AsyncMock(),
                ) as wait_for_gemini_slot:
                    with patch.object(
                        gemini_cli_client,
                        "wait_after_gemini_response",
                        new=AsyncMock(),
                    ) as wait_after_gemini_response:
                        result = await gemini_cli_client._call_gemini_rest(
                            prompt="prompt",
                            model="gemini-3-flash-preview",
                            temperature=0.7,
                            max_tokens=32,
                            system_instruction=None,
                            thinking_level="NONE",
                            use_google_search=False,
                        )

    assert result == "ok"
    wait_for_gemini_slot.assert_awaited_once()
    wait_after_gemini_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_gemini_rest_disables_thinking_for_2_5_flash():
    captured = {}

    class CaptureSession:
        def post(self, url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return _FakeResponse(
                200,
                json_data={
                    "response": {
                        "candidates": [{"content": {"parts": [{"text": "ok"}]}}]
                    }
                },
            )

    with patch.object(
        gemini_cli_client,
        "_discover_project_id",
        new=AsyncMock(return_value="project-123"),
    ):
        with patch.object(
            gemini_cli_client,
            "_build_headers",
            new=AsyncMock(return_value={"Authorization": "Bearer test"}),
        ):
            with patch.object(
                gemini_cli_client,
                "_get_aio_session",
                return_value=CaptureSession(),
            ):
                with patch.object(
                    gemini_cli_client,
                    "wait_for_gemini_slot",
                    new=AsyncMock(),
                ):
                    with patch.object(
                        gemini_cli_client,
                        "wait_after_gemini_response",
                        new=AsyncMock(),
                    ):
                        result = await gemini_cli_client._call_gemini_rest(
                            prompt="prompt",
                            model="gemini-2.5-flash",
                            temperature=0.7,
                            max_tokens=32,
                            system_instruction=None,
                            thinking_level="HIGH",
                            use_google_search=False,
                        )

    assert result == "ok"
    generation_config = captured["json"]["request"]["generationConfig"]
    assert "thinkingConfig" not in generation_config
