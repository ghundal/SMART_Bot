"""
Unit tests for the embedding.py module.

Tests the embedding model functionality for the Ollama RAG system including:
- Loading the embedding model
- Generating embeddings for queries
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np


class TestEmbedding(unittest.TestCase):
    def setUp(self):
        """Set up common test environment before each test"""
        self.mock_model = MagicMock()
        self.mock_embedding = np.array([0.1, 0.2, 0.3])

        self.mock_model.to.return_value = self.mock_model
        self.mock_model.encode.return_value = self.mock_embedding

    @patch("api.rag_pipeline.embedding.EMBEDDING_MODEL", "mock-embedding-model")
    @patch("api.rag_pipeline.embedding.logger", MagicMock())
    @patch("api.rag_pipeline.embedding.SentenceTransformer")
    @patch("torch.cuda.is_available", return_value=False)
    def test_get_ch_embedding_model_cpu(
        self, mock_cuda_available, mock_sentence_transformer, *args
    ):
        """Test loading the embedding model on CPU"""
        mock_sentence_transformer.return_value = self.mock_model

        from api.rag_pipeline.embedding import get_ch_embedding_model

        result = get_ch_embedding_model()

        mock_sentence_transformer.assert_called_once_with("mock-embedding-model")
        self.mock_model.to.assert_called_once_with("cpu")
        self.assertEqual(result, self.mock_model)

    @patch("api.rag_pipeline.embedding.EMBEDDING_MODEL", "mock-embedding-model")
    @patch("api.rag_pipeline.embedding.logger", MagicMock())
    @patch("api.rag_pipeline.embedding.SentenceTransformer")
    @patch("torch.cuda.is_available", return_value=True)
    def test_get_ch_embedding_model_gpu(
        self, mock_cuda_available, mock_sentence_transformer, *args
    ):
        """Test loading the embedding model on GPU"""
        mock_sentence_transformer.return_value = self.mock_model

        from api.rag_pipeline.embedding import get_ch_embedding_model

        result = get_ch_embedding_model()

        mock_sentence_transformer.assert_called_once_with("mock-embedding-model")
        self.mock_model.to.assert_called_once_with("cuda")
        self.assertEqual(result, self.mock_model)

    @patch("api.rag_pipeline.embedding.EMBEDDING_MODEL", "mock-embedding-model")
    @patch("api.rag_pipeline.embedding.logger")
    @patch("api.rag_pipeline.embedding.SentenceTransformer")
    def test_get_ch_embedding_model_exception(self, mock_sentence_transformer, mock_logger, *args):
        """Test exception handling when loading the embedding model"""
        mock_sentence_transformer.side_effect = Exception("Model not found")

        from api.rag_pipeline.embedding import get_ch_embedding_model

        with self.assertRaises(Exception):
            get_ch_embedding_model()

        mock_logger.exception.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    def test_embed_query_cpu(self, mock_cuda_available):
        """Test generating an embedding for a query on CPU"""
        from api.rag_pipeline.embedding import embed_query

        result = embed_query("test query", self.mock_model)

        self.mock_model.encode.assert_called_once_with(
            "test query", show_progress_bar=False, device="cpu"
        )
        self.assertEqual(result, self.mock_embedding.tolist())

    @patch("torch.cuda.is_available", return_value=True)
    def test_embed_query_gpu(self, mock_cuda_available):
        """Test generating an embedding for a query on GPU"""
        from api.rag_pipeline.embedding import embed_query

        result = embed_query("test query", self.mock_model)

        self.mock_model.encode.assert_called_once_with(
            "test query", show_progress_bar=False, device="cuda"
        )
        self.assertEqual(result, self.mock_embedding.tolist())


if __name__ == "__main__":
    unittest.main()
