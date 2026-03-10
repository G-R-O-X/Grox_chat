import pytest
from agent_chatroom.external.gemini_cli_client import _extract_text

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
