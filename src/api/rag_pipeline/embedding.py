"""
Embedding model functionality for the Ollama RAG system.
"""

import torch
from rag_pipeline.config import EMBEDDING_MODEL, logger
from sentence_transformers import SentenceTransformer


def get_ch_embedding_model():
    """Load and return the embedding model."""
    try:
        model_name = EMBEDDING_MODEL
        model = SentenceTransformer(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Successfully loaded Embedding model: {model_name}")
        return model
    except Exception as e:
        logger.exception(f"Error loading Embedding model: {e}")
        raise


def embed_query(query, model):
    """Generate an embedding for the given query."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Encode query and move to the same device as the model
    embedding = model.encode(query, show_progress_bar=False, device=device)
    return embedding.tolist()
