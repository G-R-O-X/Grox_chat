import logging
import threading
from typing import List, Optional
import asyncio

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_embedding_model = None
_initialization_attempted = False
_embedding_model_lock = threading.Lock()

def get_embedding_model():
    """
    Get or create the SentenceTransformer model handle (thread-safe singleton).
    """
    global _embedding_model, _initialization_attempted

    if _initialization_attempted:
        return _embedding_model

    with _embedding_model_lock:
        if _initialization_attempted:
            return _embedding_model

        _initialization_attempted = True

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # Determine device: MPS > CUDA > CPU
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            logger.info(f"[Embedding] Loading {DEFAULT_EMBEDDING_MODEL} on {device}...")
            _embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL, device=device)
            logger.info(f"[Embedding] Loaded successfully on {device}")
        except ImportError as e:
            logger.warning(f"[Embedding] sentence-transformers not installed: {e}")
        except Exception as e:
            logger.warning(f"[Embedding] Failed to load embedding model: {e}")

    return _embedding_model

def get_embedding(text: str) -> Optional[List[float]]:
    """
    Get embedding for text.
    """
    if not text or not text.strip():
        logger.debug("[Embedding] Empty text provided")
        return None

    model = get_embedding_model()
    if model is None:
        return None

    try:
        embedding = model.encode(text)
        if embedding is not None:
            return embedding.tolist()
        return None
    except Exception as e:
        logger.debug(f"[Embedding] Error: {e}")
        return None

def get_embeddings_batch(texts: List[str]) -> Optional[List[List[float]]]:
    """
    Get embeddings for multiple texts.
    """
    if not texts:
        return []

    model = get_embedding_model()
    if model is None:
        return None

    try:
        embeddings = model.encode(texts)
        if embeddings is not None:
            return embeddings.tolist()
        return None
    except Exception as e:
        logger.debug(f"[Embedding] Batch error: {e}")
        return None

async def aget_embedding(text: str) -> Optional[List[float]]:
    """Async wrapper for get_embedding to prevent event loop blocking."""
    return await asyncio.to_thread(get_embedding, text)

async def aget_embeddings_batch(texts: List[str]) -> Optional[List[List[float]]]:
    """Async wrapper for get_embeddings_batch to prevent event loop blocking."""
    return await asyncio.to_thread(get_embeddings_batch, texts)
