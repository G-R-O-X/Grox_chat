import os
from unittest.mock import AsyncMock, patch

import pytest

from grox_chat.minimax_client import (
    ENGLISH_ONLY_INSTRUCTION,
    _extract_pseudo_tool_markup,
    _extract_text_and_tools,
    _get_minimax_api_host,
    query_minimax,
    minimax_search,
)

def test_extract_text_and_tools():
    mock_data = {
        "content": [
            {"type": "thinking", "text": "I should search for this."},
            {"type": "text", "text": "Let me look that up."},
            {
                "type": "tool_use",
                "id": "call_123",
                "name": "web_search",
                "input": {"query": "Latest LangGraph features"}
            }
        ]
    }
    
    text, tools = _extract_text_and_tools(mock_data)
    
    assert text == "Let me look that up."
    assert len(tools) == 1
    assert tools[0]["name"] == "web_search"
    assert tools[0]["input"]["query"] == "Latest LangGraph features"

def test_extract_no_tools():
    mock_data = {
        "content": [
            {"type": "text", "text": "Just a normal message."}
        ]
    }
    
    text, tools = _extract_text_and_tools(mock_data)
    
    assert text == "Just a normal message."
    assert len(tools) == 0


def test_extract_pseudo_tool_markup_surfaces_tool_metadata():
    text = """
<minimax:tool_call>
<invoke name="ddg-search_search">
<parameter name="query">Asch 1946 primacy effect impression formation study</parameter>
<parameter name="max_results">3</parameter>
</invoke>
</minimax:tool_call>
"""

    cleaned_text, tools = _extract_pseudo_tool_markup(text)

    assert cleaned_text == ""
    assert len(tools) == 1
    assert tools[0]["name"] == "ddg-search_search"
    assert tools[0]["input"]["query"] == "Asch 1946 primacy effect impression formation study"


def test_extract_text_and_tools_strips_pseudo_tool_markup_but_keeps_plain_text():
    mock_data = {
        "content": [
            {
                "type": "text",
                "text": (
                    "Here is the answer.\n"
                    "<minimax:tool_call>\n"
                    "<invoke name=\"ddg-search_search\">\n"
                    "<parameter name=\"query\">ignored query</parameter>\n"
                    "</invoke>\n"
                    "</minimax:tool_call>"
                ),
            }
        ]
    }

    text, tools = _extract_text_and_tools(mock_data)

    assert text == "Here is the answer."
    assert len(tools) == 1
    assert tools[0]["input"]["query"] == "ignored query"


@pytest.mark.asyncio
async def test_query_minimax_ignores_tools_payload_for_messages_api():
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"content": [{"type": "text", "text": "ok"}]}

    async def fake_post(self, url, headers=None, json=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    fake_client = type("FakeClient", (), {"post": fake_post})()

    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key", "MINIMAX_EN": "1"}, clear=True):
        with patch("grox_chat.minimax_client.wait_for_slot", new=AsyncMock()):
            with patch("grox_chat.minimax_client._get_http_client", return_value=fake_client):
                text, tools = await query_minimax(
                    system_prompt="system",
                    question="question",
                    tools=[{"type": "function", "function": {"name": "web_search"}}],
                )

    assert text == "ok"
    assert tools == []
    assert captured["url"] == "https://api.minimax.io/anthropic/v1/messages"
    assert "tools" not in captured["json"]
    assert captured["json"]["system"].startswith("system")
    assert ENGLISH_ONLY_INSTRUCTION in captured["json"]["system"]
    assert captured["json"]["messages"][0]["content"].startswith(captured["json"]["system"])
    assert captured["json"]["messages"][0]["content"].endswith(captured["json"]["system"])


@pytest.mark.asyncio
async def test_query_minimax_rejects_pseudo_tool_markup_by_default():
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<minimax:tool_call>\n"
                            "<invoke name=\"ddg-search_search\">\n"
                            "<parameter name=\"query\">thin slicing first impression workplace</parameter>\n"
                            "</invoke>\n"
                            "</minimax:tool_call>"
                        ),
                    }
                ]
            }

    async def fake_post(self, url, headers=None, json=None):
        return FakeResponse()

    fake_client = type("FakeClient", (), {"post": fake_post})()

    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key", "MINIMAX_EN": "0"}, clear=True):
        with patch("grox_chat.minimax_client.wait_for_slot", new=AsyncMock()):
            with patch("grox_chat.minimax_client._get_http_client", return_value=fake_client):
                text, tools = await query_minimax(
                    system_prompt="system",
                    question="question",
                )

    assert text == "Error: MiniMax emitted pseudo-tool markup in text-only mode"
    assert len(tools) == 1
    assert tools[0]["name"] == "ddg-search_search"


@pytest.mark.asyncio
async def test_query_minimax_recovers_pseudo_tool_query_when_opted_in():
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<minimax:tool_call>\n"
                            "<invoke name=\"ddg-search_search\">\n"
                            "<parameter name=\"query\">thin slicing first impression workplace</parameter>\n"
                            "</invoke>\n"
                            "</minimax:tool_call>"
                        ),
                    }
                ]
            }

    async def fake_post(self, url, headers=None, json=None):
        return FakeResponse()

    fake_client = type("FakeClient", (), {"post": fake_post})()

    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key", "MINIMAX_EN": "0"}, clear=True):
        with patch("grox_chat.minimax_client.wait_for_slot", new=AsyncMock()):
            with patch("grox_chat.minimax_client._get_http_client", return_value=fake_client):
                text, tools = await query_minimax(
                    system_prompt="system",
                    question="question",
                    recover_pseudo_tool_query=True,
                )

    assert text == "thin slicing first impression workplace"
    assert len(tools) == 1
    assert tools[0]["name"] == "ddg-search_search"


def test_minimax_defaults_to_domestic_host():
    with patch.dict(os.environ, {}, clear=True):
        assert _get_minimax_api_host() == "https://api.minimaxi.com"


def test_minimax_international_host_enabled_with_env_flag():
    with patch.dict(os.environ, {"MINIMAX_EN": "1"}, clear=True):
        assert _get_minimax_api_host() == "https://api.minimax.io"


@pytest.mark.asyncio
async def test_minimax_search_uses_domestic_host_by_default():
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    async def fake_post(self, url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return FakeResponse()

    fake_client = type("FakeClient", (), {"post": fake_post})()

    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key", "MINIMAX_EN": "0"}, clear=True):
        with patch("grox_chat.minimax_client.wait_for_slot", new=AsyncMock()):
            with patch("grox_chat.minimax_client._get_http_client", return_value=fake_client):
                result = await minimax_search("left foot or right foot")

    assert result == {"ok": True}
    assert captured["url"] == "https://api.minimaxi.com/v1/coding_plan/search"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
