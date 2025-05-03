"""
Transformer-based reranker using Hugging Face models.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Any, Dict, List
import logging

# Configure logger
logger = logging.getLogger(__name__)


class TransformerReranker:
    """
    A reranker that uses a transformer model to score document relevance.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Initialize the transformer reranker.

        Args:
            model_name: Name of the Hugging Face model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the transformer model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            logger.info(f"Loaded transformer reranker model: {self.model_name}")
        except Exception as e:
            logger.exception(f"Error loading transformer model: {str(e)}")
            raise

    def rerank(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Rerank chunks based on relevance to the query.

        Args:
            chunks: List of document chunks to rerank
            query: The search query

        Returns:
            Reranked list of chunks with scores
        """
        try:
            # Create pairs of [query, chunk_text] for each chunk
            pairs = [[query, chunk["chunk_text"]] for chunk in chunks]

            # Tokenize pairs and get scores
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
                )
                # Get relevance scores
                scores = (
                    self.model(**inputs, return_dict=True)
                    .logits.view(
                        -1,
                    )
                    .float()
                )

            # Add scores to chunks
            reranking_results = []
            for i, chunk in enumerate(chunks):
                chunk_with_score = chunk.copy()
                # Convert score to a 0-10 range (original scores can be unbounded)
                # Apply sigmoid to get a value between 0 and 1, then scale to 0-10
                normalized_score = torch.sigmoid(scores[i]).item() * 10
                chunk_with_score["llm_score"] = normalized_score
                reranking_results.append(chunk_with_score)

            # Sort by score in descending order
            reranked_chunks = sorted(reranking_results, key=lambda x: x["llm_score"], reverse=True)
            logger.info(
                f"Reranked {len(reranked_chunks)} chunks using transformer model {self.model_name}"
            )
            return reranked_chunks

        except Exception as e:
            logger.exception(f"Error in transformer reranking: {str(e)}")
            # Return original chunks if reranking fails
            return chunks


def rerank_chunks(
    chunks: List[Dict[str, Any]], query: str, model_name: str = "BAAI/bge-reranker-base"
) -> List[Dict[str, Any]]:
    """
    Rerank chunks based on relevance to the query using a transformer model.

    Args:
        chunks: List of document chunks to rerank
        query: The search query
        model_name: Name of the transformer model to use

    Returns:
        Reranked list of chunks with scores
    """
    try:
        # Create reranker instance and rerank chunks
        reranker = TransformerReranker(model_name)
        reranked_chunks = reranker.rerank(chunks, query)
        return reranked_chunks

    except Exception as e:
        logger.exception(f"Error in transformer reranking: {str(e)}")
        # Return original chunks if reranking fails
        return chunks
