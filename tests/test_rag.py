import pytest
from unittest.mock import patch, AsyncMock
from agent_chatroom.rag import assemble_rag_context

@pytest.mark.asyncio
async def test_assemble_rag_context_empty():
    res, degraded = await assemble_rag_context(1, 1, [], "dreamer")
    assert res == ""
    assert degraded is False

@pytest.mark.asyncio
async def test_assemble_rag_context():
    recent_messages = [{"id": 9, "content": "This is a test message", "sender": "critic"}]
    
    mock_query_minimax = AsyncMock(return_value=("测试查询", []))
    mock_embedding = AsyncMock(return_value=[0.1] * 384)
    
    mock_facts = [
        {"id": 1, "content": "Fact 1", "source": "Writer"},
        {"id": 2, "content": "Fact 2", "source": "Writer"}
    ]
    mock_summaries = [
        {"id": 3, "content": "Summary 1", "source": "Audience"},
        {"id": 4, "content": "Summary 2", "source": "Audience"},
    ]
    mock_messages = [
        {"id": 5, "content": "Historical message 1", "source": "engineer"},
        {"id": 6, "content": "Historical message 2", "source": "critic"},
    ]
    
    # Reranker returns indices and scores
    mock_rerank = AsyncMock(side_effect=[
        [(0, 0.9), (1, 0.1)],
        [(0, 0.8), (1, 0.1)],
        [(0, 0.7), (1, 0.1)],
    ])
    
    with patch("agent_chatroom.rag.query_minimax", new=mock_query_minimax):
        with patch("agent_chatroom.rag.aget_embedding", new=mock_embedding):
            with patch("agent_chatroom.rag.api.search_facts_hybrid", return_value=mock_facts):
                with patch("agent_chatroom.rag.api.search_messages_hybrid", side_effect=[mock_summaries, mock_messages]):
                    with patch("agent_chatroom.rag.arerank", new=mock_rerank):
                        res, degraded = await assemble_rag_context(1, 1, recent_messages, "dreamer")

                        assert "RAG KNOWLEDGE INJECTION" in res
                        assert degraded is False
                        assert "[Fact: 1]" in res
                        assert "Fact 1" in res
                        assert "[Summary: 3]" in res
                        assert "Summary 1" in res
                        assert "[Message: 5]" in res
                        assert "Historical message 1" in res
                        # Fact 2 is dropped because score 0.1 < 0.3
                        assert "[Fact: 2]" not in res
