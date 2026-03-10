import logging
import threading
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

_reranker_model = None
_initialization_attempted = False
_reranker_model_lock = threading.Lock()

def get_reranker_model():
    """Get or create the CrossEncoder model handle (thread-safe singleton)."""
    global _reranker_model, _initialization_attempted

    if _initialization_attempted:
        return _reranker_model

    with _reranker_model_lock:
        if _initialization_attempted:
            return _reranker_model

        _initialization_attempted = True

        try:
            from sentence_transformers import CrossEncoder
            import torch

            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            logger.info(f"[Reranker] Loading {DEFAULT_RERANKER_MODEL} on {device}...")
            _reranker_model = CrossEncoder(
                DEFAULT_RERANKER_MODEL,
                max_length=512,
                device=device,
            )
            logger.info(f"[Reranker] Loaded on {device}")

        except ImportError as e:
            logger.warning(f"[Reranker] sentence-transformers not installed: {e}")
        except Exception as e:
            logger.warning(f"[Reranker] Failed to load reranker model: {e}")

    return _reranker_model

def rerank(
    query: str,
    documents: List[str],
    top_k: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """
    Rerank documents by relevance to query using CrossEncoder.
    Returns List of (original_index, score) sorted by score descending.
    """
    model = get_reranker_model()
    if model is None:
        return []

    if not documents or not query or not query.strip():
        return []

    try:
        pairs = [[query, doc] for doc in documents]
        scores = model.predict(pairs, show_progress_bar=False)

        import math
        def sigmoid(x):
            try:
                return 1 / (1 + math.exp(-x))
            except OverflowError:
                return 0.0 if x < 0 else 1.0

        normalized_scores = [sigmoid(float(s)) for s in scores]

        indexed_scores = list(enumerate(normalized_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None and top_k > 0:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores

    except Exception as e:
        logger.error(f"[Reranker] Ranking failed: {e}")
        return []

async def arerank(
    query: str,
    documents: List[str],
    top_k: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """Async wrapper for rerank to prevent event loop blocking."""
    import asyncio
    return await asyncio.to_thread(rerank, query, documents, top_k)
