"""
Unit tests for the src/api/rag_pipeline/transformer_reranker.py module.

Tests the transformer-based reranking functionality including:
- Chunk reranking
- Embedding generation
- Cosine similarity calculation
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json
import numpy as np

# Add src directory to path to find the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src")))

# Import the module under test
from api.rag_pipeline.transformer_reranker import rerank_chunks, get_embedding, cosine_similarity


class TestTransformerReranker(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        # Create sample chunks for testing
        self.test_chunks = [
            {"document_id": "doc1", "chunk_text": "Text about machine learning", "page_number": 1},
            {"document_id": "doc2", "chunk_text": "Text about neural networks", "page_number": 2},
            {"document_id": "doc3", "chunk_text": "Text about data science", "page_number": 3},
        ]

        # Test query
        self.test_query = "machine learning applications"

        # Test model name
        self.test_model = "bge-reranker-large"

        # Sample embeddings
        self.query_embedding = [0.1, 0.2, 0.3, 0.4]
        self.chunk_embeddings = [
            [0.1, 0.2, 0.3, 0.4],  # Very similar to query (1.0)
            [0.05, 0.1, 0.15, 0.2],  # Moderately similar to query (~0.5)
            [-0.1, -0.2, -0.3, -0.4],  # Opposite of query (-1.0, will be clamped to 0)
        ]

        # Expected similarity scores (0-1 range)
        self.expected_similarities = [1.0, 0.5, 0.0]

        # Patch logger to avoid actual logging during tests
        self.logger_patch = patch("api.rag_pipeline.transformer_reranker.logger")
        self.mock_logger = self.logger_patch.start()

    def tearDown(self):
        """Clean up after each test"""
        self.logger_patch.stop()

    @patch("api.rag_pipeline.transformer_reranker.get_embedding")
    @patch("api.rag_pipeline.transformer_reranker.cosine_similarity")
    def test_rerank_chunks_success(self, mock_cosine_sim, mock_get_embedding):
        """Test successful reranking of chunks"""
        # Set up mocks
        mock_get_embedding.side_effect = [
            self.query_embedding,  # First call for query
            self.chunk_embeddings[0],  # Second call for chunk 1
            self.chunk_embeddings[1],  # Third call for chunk 2
            self.chunk_embeddings[2],  # Fourth call for chunk 3
        ]

        # Set up cosine similarity mock
        mock_cosine_sim.side_effect = self.expected_similarities

        # Call the function
        result = rerank_chunks(self.test_chunks, self.test_query, self.test_model)

        # Verify the calls
        self.assertEqual(mock_get_embedding.call_count, 4)
        self.assertEqual(mock_cosine_sim.call_count, 3)

        # Verify the results
        self.assertEqual(len(result), 3)

        # Chunks should be sorted by score in descending order
        self.assertEqual(result[0]["document_id"], "doc1")  # Score 10.0
        self.assertEqual(result[1]["document_id"], "doc2")  # Score 5.0
        self.assertEqual(result[2]["document_id"], "doc3")  # Score 0.0

        # Verify the scores (scaled to 0-10 range)
        self.assertEqual(result[0]["llm_score"], 10.0)
        self.assertEqual(result[1]["llm_score"], 5.0)
        self.assertEqual(result[2]["llm_score"], 0.0)

        # Verify logging
        self.mock_logger.info.assert_called_once_with("Reranked 3 chunks using transformer model")

    def test_rerank_chunks_empty(self):
        """Test reranking with empty chunk list"""
        # Call with empty chunks
        result = rerank_chunks([], self.test_query, self.test_model)

        # Verify result is empty
        self.assertEqual(result, [])

        # Verify warning was logged
        self.mock_logger.warning.assert_called_once_with("No chunks to rerank")

    @patch("api.rag_pipeline.transformer_reranker.get_embedding")
    def test_rerank_chunks_embedding_failure(self, mock_get_embedding):
        """Test reranking when embedding generation fails"""
        # Set up mocks
        mock_get_embedding.return_value = None  # Embedding generation fails

        # Call the function
        result = rerank_chunks(self.test_chunks, self.test_query, self.test_model)

        # Verify the calls - FIXED: First call should be for the query
        self.assertEqual(mock_get_embedding.call_count, 4)  # Query + 3 chunks
        mock_get_embedding.assert_any_call(self.test_query, self.test_model)

        # Verify the results (original chunks, with score 0)
        self.assertEqual(len(result), 3)
        for chunk in result:
            self.assertEqual(chunk["llm_score"], 0)

    @patch("api.rag_pipeline.transformer_reranker.get_embedding")
    def test_rerank_chunks_exception(self, mock_get_embedding):
        """Test reranking with an exception"""
        # Set up mock to raise exception
        mock_get_embedding.side_effect = Exception("Test exception")

        # Call the function
        result = rerank_chunks(self.test_chunks, self.test_query, self.test_model)

        # Verify the results (original chunks returned without modification)
        self.assertEqual(len(result), 3)
        for i, chunk in enumerate(result):
            self.assertEqual(chunk["document_id"], self.test_chunks[i]["document_id"])
            self.assertNotIn("llm_score", chunk)  # No score added

        # Verify exception was logged
        self.mock_logger.exception.assert_called_once()

    @patch("tempfile.NamedTemporaryFile")
    @patch("subprocess.run")
    @patch("os.unlink")
    def test_get_embedding_success_json(self, mock_unlink, mock_subprocess_run, mock_temp_file):
        """Test successful embedding generation with JSON output"""
        # Set up mocks
        mock_file = MagicMock()
        mock_file.name = "temp_text_file"
        mock_temp_file.return_value.__enter__.return_value = mock_file

        # Mock successful subprocess result with JSON output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"embedding": self.query_embedding})
        mock_subprocess_run.return_value = mock_result

        # Call the function
        result = get_embedding("Test text", self.test_model)

        # Verify the calls
        mock_temp_file.assert_called_once()
        mock_subprocess_run.assert_called_once_with(
            ["ollama", "embeddings", "-m", self.test_model, "-f", "temp_text_file"],
            capture_output=True,
            text=True,
        )
        mock_unlink.assert_called_once_with("temp_text_file")

        # Verify the result
        self.assertEqual(result, self.query_embedding)

    @patch("tempfile.NamedTemporaryFile")
    @patch("subprocess.run")
    @patch("os.unlink")
    def test_get_embedding_success_list(self, mock_unlink, mock_subprocess_run, mock_temp_file):
        """Test successful embedding generation with list output"""
        # Set up mocks
        mock_file = MagicMock()
        mock_file.name = "temp_text_file"
        mock_temp_file.return_value.__enter__.return_value = mock_file

        # Mock successful subprocess result with JSON array output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(self.query_embedding)
        mock_subprocess_run.return_value = mock_result

        # Call the function
        result = get_embedding("Test text", self.test_model)

        # Verify the result
        self.assertEqual(result, self.query_embedding)

    @patch("tempfile.NamedTemporaryFile")
    @patch("subprocess.run")
    @patch("os.unlink")
    def test_get_embedding_success_plain_text(
        self, mock_unlink, mock_subprocess_run, mock_temp_file
    ):
        """Test successful embedding generation with plain text output"""
        # Set up mocks
        mock_file = MagicMock()
        mock_file.name = "temp_text_file"
        mock_temp_file.return_value.__enter__.return_value = mock_file

        # Mock successful subprocess result with space-separated values
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0.1 0.2 0.3 0.4"
        mock_subprocess_run.return_value = mock_result

        # Call the function
        result = get_embedding("Test text", self.test_model)

        # Verify the result
        self.assertEqual(result, self.query_embedding)

    @patch("tempfile.NamedTemporaryFile")
    @patch("subprocess.run")
    @patch("os.unlink")
    def test_get_embedding_subprocess_error(self, mock_unlink, mock_subprocess_run, mock_temp_file):
        """Test embedding generation with subprocess error"""
        # Set up mocks
        mock_file = MagicMock()
        mock_file.name = "temp_text_file"
        mock_temp_file.return_value.__enter__.return_value = mock_file

        # Mock failed subprocess result
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error generating embedding"
        mock_subprocess_run.return_value = mock_result

        # Call the function
        result = get_embedding("Test text", self.test_model)

        # Verify the calls
        mock_temp_file.assert_called_once()
        mock_subprocess_run.assert_called_once()
        mock_unlink.assert_called_once_with("temp_text_file")

        # Verify the result
        self.assertIsNone(result)

        # Verify error was logged
        self.mock_logger.error.assert_called_once_with(
            "Error generating embedding: Error generating embedding"
        )

    @patch("tempfile.NamedTemporaryFile")
    @patch("subprocess.run")
    @patch("os.unlink")
    def test_get_embedding_invalid_output(self, mock_unlink, mock_subprocess_run, mock_temp_file):
        """Test embedding generation with invalid output format"""
        self.mock_logger.reset_mock()

        # Setup mock temp file
        mock_file = MagicMock()
        mock_file.name = "temp_text_file"
        mock_temp_file.return_value.__enter__.return_value = mock_file

        # Return invalid, non-parsable stdout
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "non-json, non-list, non-numeric"
        mock_subprocess_run.return_value = mock_result

        # Run the function
        result = get_embedding("Test text", self.test_model)

        # Validate output
        self.assertIsNone(result)

    @patch("tempfile.NamedTemporaryFile")
    def test_get_embedding_exception(self, mock_temp_file):
        """Test embedding generation with exception"""
        # Set up mock to raise exception
        mock_temp_file.side_effect = Exception("Test exception")

        # Call the function
        result = get_embedding("Test text", self.test_model)

        # Verify the result
        self.assertIsNone(result)

        # Verify exception was logged
        self.mock_logger.exception.assert_called_once_with(
            "Error getting embedding: Test exception"
        )

    def test_cosine_similarity_normal(self):
        """Test cosine similarity calculation with normal vectors"""
        # Define test vectors
        a = [1, 2, 3]
        b = [2, 3, 4]

        # Calculate expected result
        a_array = np.array(a)
        b_array = np.array(b)
        dot_product = np.dot(a_array, b_array)
        norm_a = np.linalg.norm(a_array)
        norm_b = np.linalg.norm(b_array)
        expected = dot_product / (norm_a * norm_b)

        # Call the function
        result = cosine_similarity(a, b)

        # Verify the result
        self.assertAlmostEqual(result, expected)

    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors (should be 1.0)"""
        # Define test vector
        a = [1, 2, 3]

        # Call the function
        result = cosine_similarity(a, a)

        # Verify the result
        self.assertEqual(result, 1.0)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors (should be 0.0)"""
        # Define orthogonal vectors
        a = [1, 0, 0]
        b = [0, 1, 0]

        # Call the function
        result = cosine_similarity(a, b)

        # Verify the result
        self.assertEqual(result, 0.0)

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity with opposite vectors (should be 0.0 after clamping)"""
        # Define opposite vectors
        a = [1, 2, 3]
        b = [-1, -2, -3]

        # Call the function
        result = cosine_similarity(a, b)

        # Verify the result (would be -1.0, but clamped to 0.0)
        self.assertEqual(result, 0.0)

    def test_cosine_similarity_empty(self):
        """Test cosine similarity with empty vectors"""
        # Call the function with empty vectors
        result = cosine_similarity([], [])

        # Verify the result
        self.assertEqual(result, 0.0)

    def test_cosine_similarity_different_lengths(self):
        """Test cosine similarity with vectors of different lengths"""
        # Define vectors of different lengths
        a = [1, 2, 3]
        b = [1, 2]

        # Call the function
        result = cosine_similarity(a, b)

        # Verify the result
        self.assertEqual(result, 0.0)

    def test_cosine_similarity_zero_norm(self):
        """Test cosine similarity with zero norm vector"""
        # Define a zero vector
        a = [0, 0, 0]
        b = [1, 2, 3]

        # Call the function
        result = cosine_similarity(a, b)

        # Verify the result
        self.assertEqual(result, 0.0)

    def test_cosine_similarity_exception(self):
        """Test cosine similarity with exception"""
        # Mock np.dot to raise exception
        with patch("numpy.dot", side_effect=Exception("Test exception")):
            # Call the function
            result = cosine_similarity([1, 2, 3], [4, 5, 6])

            # Verify the result
            self.assertEqual(result, 0.0)

            # Verify exception was logged
            self.mock_logger.exception.assert_called_once_with(
                "Error calculating cosine similarity: Test exception"
            )


if __name__ == "__main__":
    unittest.main()
