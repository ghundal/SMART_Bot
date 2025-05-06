"""
Embedding model utilities for the Ollama RAG system.

This module provides functions to load a sentence-transformer embedding model and generate
vector embeddings for user queries. These embeddings are used for semantic search and
retrieval-augmented generation (RAG) workflows.
"""

import torch
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, logger


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
