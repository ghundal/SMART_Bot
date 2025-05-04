"""
Module for direct reranking using transformer-based models without prompting.
"""

import os
import subprocess
import tempfile
from typing import List, Dict, Any
import numpy as np

from .config import logger


def rerank_chunks(
    chunks: List[Dict[str, Any]], query: str, model_name: str
) -> List[Dict[str, Any]]:
    """
    Rerank chunks using a transformer-based reranker model via Ollama.
    This uses direct comparison rather than prompting.

    Args:
        chunks: List of document chunks to rerank
        query: User query to compare against
        model_name: Name of the reranker model in Ollama

    Returns:
        List of reranked chunks with scores
    """
    try:
        # Verify that chunks is not empty
        if not chunks:
            logger.warning("No chunks to rerank")
            return []

        # Prepare the query and documents for the reranker
        # For BGE reranker, we'll use Ollama's embedding API
        query_embedding = get_embedding(query, model_name)

        # Get embeddings for each chunk
        scored_chunks = []
        for chunk in chunks:
            chunk_copy = chunk.copy()

            # Get embedding for this chunk
            chunk_embedding = get_embedding(chunk["chunk_text"], model_name)

            # Calculate similarity
            if query_embedding and chunk_embedding:
                # Use cosine similarity
                similarity = cosine_similarity(query_embedding, chunk_embedding)
                # Scale to 0-10 range for compatibility with existing code
                score = float(similarity * 10)
            else:
                # Fallback if embedding generation fails
                score = 0

            chunk_copy["llm_score"] = score
            scored_chunks.append(chunk_copy)

        # Sort by score in descending order
        reranked_chunks = sorted(scored_chunks, key=lambda x: x["llm_score"], reverse=True)
        logger.info(f"Reranked {len(reranked_chunks)} chunks using transformer model")

        return reranked_chunks

    except Exception as e:
        logger.exception(f"Error in transformer reranking: {str(e)}")
        # Return original chunks without scoring on error
        return [chunk.copy() for chunk in chunks]


def get_embedding(text: str, model_name: str) -> List[float]:
    """
    Get embedding for a text using Ollama's embedding endpoint.

    Args:
        text: Text to embed
        model_name: Ollama model to use

    Returns:
        List of embedding values or None on error
    """
    try:
        # Create a temporary file with the text
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(text)
            text_file = f.name

        # Use Ollama CLI to generate embeddings
        cmd = ["ollama", "embeddings", "-m", model_name, "-f", text_file]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Clean up temporary file
        os.unlink(text_file)

        if result.returncode != 0:
            logger.error(f"Error generating embedding: {result.stderr}")
            return None

        # Parse the embedding output
        # The format depends on Ollama's output, may need adjustment
        embedding_str = result.stdout.strip()

        # Parse the embedding values (format may vary)
        try:
            # Try parsing as JSON first (newer Ollama versions)
            import json

            embedding_data = json.loads(embedding_str)
            if isinstance(embedding_data, dict) and "embedding" in embedding_data:
                return embedding_data["embedding"]
            elif isinstance(embedding_data, list):
                return embedding_data
        except json.JSONDecodeError:
            # Fallback: Try parsing as space-separated values
            embedding_values = [float(x) for x in embedding_str.split()]
            return embedding_values

        logger.warning("Could not parse embedding output format")
        return None

    except Exception as e:
        logger.exception(f"Error getting embedding: {str(e)}")
        return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score (0-1)
    """
    try:
        if not a or not b or len(a) != len(b):
            return 0.0

        a_array = np.array(a)
        b_array = np.array(b)

        # Calculate cosine similarity
        dot_product = np.dot(a_array, b_array)
        norm_a = np.linalg.norm(a_array)
        norm_b = np.linalg.norm(b_array)

        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)

        # Ensure the result is between 0 and 1
        return max(0.0, min(1.0, float(similarity)))

    except Exception as e:
        logger.exception(f"Error calculating cosine similarity: {str(e)}")
        return 0.0
