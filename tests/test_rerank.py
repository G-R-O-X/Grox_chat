from agent_chatroom.reranker import rerank

def test_rerank_basic():
    query = "What is the capital of France?"
    documents = [
        "London is a great city.",
        "Paris is the capital of France.",
        "The weather is nice today."
    ]
    
    results = rerank(query, documents)
    
    # Check return structure
    assert len(results) == 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    
    # Check sorting
    assert results[0][1] >= results[1][1]
    assert results[1][1] >= results[2][1]
    
    # Check correctness: Document 1 (Paris) should be the highest
    assert results[0][0] == 1
    
    # Check scores are normalized between 0 and 1
    assert 0.0 <= results[0][1] <= 1.0

def test_rerank_empty():
    assert rerank("", ["doc1"]) == []
    assert rerank("query", []) == []

def test_rerank_top_k():
    query = "test query"
    documents = ["doc1", "doc2", "doc3"]
    results = rerank(query, documents, top_k=2)
    assert len(results) == 2
