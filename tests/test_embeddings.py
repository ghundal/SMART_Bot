import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add the src directory to the path so we can import our module
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Import the module to test
from src.api.rag_pipeline.embedding import embed_query, get_ch_embedding_model


class TestEmbeddingModel(unittest.TestCase):
    """Test cases for the embedding model functionality."""

    @patch("src.api.rag_pipeline.embedding.SentenceTransformer")
    @patch("src.api.rag_pipeline.embedding.torch.cuda.is_available")
    def test_get_ch_embedding_model_cuda(self, mock_cuda_available, mock_sentence_transformer):
        """Test loading embedding model with CUDA available."""
        # Mock CUDA availability
        mock_cuda_available.return_value = True

        # Setup mock model
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_sentence_transformer.return_value = mock_model

        # Call the function
        model = get_ch_embedding_model()

        # Verify model was loaded with the correct name
        mock_sentence_transformer.assert_called_once_with("all-mpnet-base-v2")

        # Verify model was moved to CUDA
        mock_model.to.assert_called_once_with("cuda")

        # Verify the returned model is the expected one
        self.assertEqual(model, mock_model)

    @patch("src.api.rag_pipeline.embedding.SentenceTransformer")
    @patch("src.api.rag_pipeline.embedding.torch.cuda.is_available")
    def test_get_ch_embedding_model_cpu(self, mock_cuda_available, mock_sentence_transformer):
        """Test loading embedding model with CUDA not available."""
        # Mock CUDA availability
        mock_cuda_available.return_value = False

        # Setup mock model
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_sentence_transformer.return_value = mock_model

        # Call the function
        model = get_ch_embedding_model()

        # Verify model was loaded with the correct name
        mock_sentence_transformer.assert_called_once_with("all-mpnet-base-v2")

        # Verify model was moved to CPU
        mock_model.to.assert_called_once_with("cpu")

        # Verify the returned model is the expected one
        self.assertEqual(model, mock_model)

    @patch("src.api.rag_pipeline.embedding.SentenceTransformer")
    def test_get_ch_embedding_model_exception(self, mock_sentence_transformer):
        """Test error handling when loading model fails."""
        # Setup mock to raise exception
        mock_sentence_transformer.side_effect = Exception("Model loading failed")

        # Verify that the exception is propagated
        with self.assertRaises(Exception):
            get_ch_embedding_model()

    @patch("src.api.rag_pipeline.embedding.torch.cuda.is_available")
    def test_embed_query_cuda(self, mock_cuda_available):
        """Test query embedding with CUDA available."""
        # Mock CUDA availability
        mock_cuda_available.return_value = True

        # Setup mock model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        # Call the function
        result = embed_query("test query", mock_model)

        # Verify encode was called with the right parameters
        mock_model.encode.assert_called_once_with(
            "test query", show_progress_bar=False, device="cuda"
        )

        # Verify the result is the expected list
        self.assertEqual(result, [0.1, 0.2, 0.3])

    @patch("src.api.rag_pipeline.embedding.torch.cuda.is_available")
    def test_embed_query_cpu(self, mock_cuda_available):
        """Test query embedding with CUDA not available."""
        # Mock CUDA availability
        mock_cuda_available.return_value = False

        # Setup mock model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        # Call the function
        result = embed_query("test query", mock_model)

        # Verify encode was called with the right parameters
        mock_model.encode.assert_called_once_with(
            "test query", show_progress_bar=False, device="cpu"
        )

        # Verify the result is the expected list
        self.assertEqual(result, [0.1, 0.2, 0.3])

    def test_embed_query_exception(self):
        """Test error handling when embedding fails."""
        # Setup mock model that raises an exception
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Embedding failed")

        # Verify that the exception is propagated
        with self.assertRaises(Exception):
            embed_query("test query", mock_model)


if __name__ == "__main__":
    unittest.main()
