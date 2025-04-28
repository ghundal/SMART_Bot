import os
import sys
import unittest
from unittest.mock import MagicMock, patch


# Add the src directory to the path so we can import our module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the module to test
from src.api.rag_pipeline.ollama import query_ollama_with_hybrid_search_multilingual


class TestOllamaRAG(unittest.TestCase):
    """Test cases for the main Ollama RAG functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock session
        self.mock_session = MagicMock()

        # Create mock embedding model
        self.mock_embedding_model = MagicMock()

        # Default test parameters
        self.test_question = "What is deep learning?"
        self.test_email = "user@example.com"
        self.test_model = "gemma3:12b"

        # Sample embedding
        self.sample_embedding = [0.1] * 384

        # Sample context chunks
        self.sample_chunks = [
            {
                "chunk_id": 1,
                "chunk_text": "Deep learning is a subset of machine learning.",
                "document_id": "doc1",
                "page_number": 1,
            },
            {
                "chunk_id": 2,
                "chunk_text": "Neural networks are the foundation of deep learning.",
                "document_id": "doc2",
                "page_number": 3,
            },
            {
                "chunk_id": 3,
                "chunk_text": "Transformers are a type of deep learning model.",
                "document_id": "doc3",
                "page_number": 5,
            },
        ]

        # Sample sorted results
        self.sample_sorted_results = [
            {"chunk": chunk, "score": 0.9 - i * 0.1} for i, chunk in enumerate(self.sample_chunks)
        ]

        # Sample document metadata
        self.sample_metadata = {
            "doc1": {
                "class_name": "Introduction to AI",
                "authors": "John Doe",
                "term": "Spring 2025",
            },
            "doc2": {"class_name": "Neural Networks", "authors": "Jane Smith", "term": "Fall 2024"},
            "doc3": {"class_name": "Advanced ML", "authors": "Bob Johnson", "term": "Winter 2024"},
        }

    @patch("src.api.rag_pipeline.ollama.check_query_safety_with_llama_guard")
    @patch("src.api.rag_pipeline.ollama.detect_language")
    @patch("src.api.rag_pipeline.ollama.embed_query")
    @patch("src.api.rag_pipeline.ollama.hybrid_search")
    @patch("src.api.rag_pipeline.ollama.rerank_with_llm")
    @patch("src.api.rag_pipeline.ollama.retrieve_document_metadata")
    @patch("src.api.rag_pipeline.ollama.query_llm")
    @patch("src.api.rag_pipeline.ollama.log_audit")
    def test_query_ollama_english(
        self,
        mock_log_audit,
        mock_query_llm,
        mock_retrieve_metadata,
        mock_rerank,
        mock_hybrid_search,
        mock_embed_query,
        mock_detect_language,
        mock_safety_check,
    ):
        """Test querying Ollama with an English question."""
        # Configure mocks
        mock_safety_check.return_value = (True, "")
        mock_detect_language.return_value = "en"
        mock_embed_query.return_value = self.sample_embedding
        mock_hybrid_search.return_value = (self.sample_chunks, self.sample_sorted_results)
        mock_rerank.return_value = self.sample_chunks
        mock_retrieve_metadata.return_value = self.sample_metadata
        mock_query_llm.return_value = (
            "Deep learning is a subset of machine learning that uses neural networks."
        )

        # Call function
        result = query_ollama_with_hybrid_search_multilingual(
            self.mock_session,
            self.test_question,
            self.mock_embedding_model,
            self.test_email,
            self.test_model,
        )

        # Verify the correct functions were called
        mock_safety_check.assert_called_once_with(self.test_question)
        mock_detect_language.assert_called_once_with(self.test_question)
        mock_embed_query.assert_called_once_with(self.test_question, self.mock_embedding_model)
        mock_hybrid_search.assert_called_once()
        mock_rerank.assert_called_once()
        mock_retrieve_metadata.assert_called_once()
        mock_query_llm.assert_called_once()
        mock_log_audit.assert_called_once()

        # Verify the result structure
        self.assertEqual(result["original_question"], self.test_question)
        self.assertEqual(result["detected_language"], "en")
        self.assertIsNone(result["english_question"])
        self.assertEqual(result["context_count"], 3)
        self.assertIn("Deep learning", result["response"])
        self.assertEqual(len(result["top_documents"]), 3)

    @patch("src.api.rag_pipeline.ollama.check_query_safety_with_llama_guard")
    @patch("src.api.rag_pipeline.ollama.detect_language")
    @patch("src.api.rag_pipeline.ollama.translate_text")
    @patch("src.api.rag_pipeline.ollama.embed_query")
    @patch("src.api.rag_pipeline.ollama.hybrid_search")
    @patch("src.api.rag_pipeline.ollama.rerank_with_llm")
    @patch("src.api.rag_pipeline.ollama.retrieve_document_metadata")
    @patch("src.api.rag_pipeline.ollama.query_llm")
    @patch("src.api.rag_pipeline.ollama.log_audit")
    def test_query_ollama_non_english(
        self,
        mock_log_audit,
        mock_query_llm,
        mock_retrieve_metadata,
        mock_rerank,
        mock_hybrid_search,
        mock_embed_query,
        mock_translate,
        mock_detect_language,
        mock_safety_check,
    ):
        """Test querying Ollama with a non-English question."""
        # Configure mocks
        mock_safety_check.return_value = (True, "")
        mock_detect_language.return_value = "es"
        mock_translate.side_effect = [
            "What is deep learning?",  # Spanish to English
            "El aprendizaje profundo es un subconjunto del aprendizaje automático que utiliza redes neuronales.",  # English to Spanish
        ]
        mock_embed_query.return_value = self.sample_embedding
        mock_hybrid_search.return_value = (self.sample_chunks, self.sample_sorted_results)
        mock_rerank.return_value = self.sample_chunks
        mock_retrieve_metadata.return_value = self.sample_metadata
        mock_query_llm.return_value = (
            "Deep learning is a subset of machine learning that uses neural networks."
        )

        # Call function with Spanish question
        result = query_ollama_with_hybrid_search_multilingual(
            self.mock_session,
            "¿Qué es el aprendizaje profundo?",
            self.mock_embedding_model,
            self.test_email,
            self.test_model,
        )

        # Verify translation was called twice (question and response)
        self.assertEqual(mock_translate.call_count, 2)

        # Verify the result structure
        self.assertEqual(result["original_question"], "¿Qué es el aprendizaje profundo?")
        self.assertEqual(result["detected_language"], "es")
        self.assertEqual(result["english_question"], "What is deep learning?")
        self.assertEqual(result["context_count"], 3)
        # Response should be in Spanish
        self.assertIn("aprendizaje profundo", result["response"])
        self.assertEqual(len(result["top_documents"]), 3)

    @patch("src.api.rag_pipeline.ollama.check_query_safety_with_llama_guard")
    def test_query_ollama_unsafe_original(self, mock_safety_check):
        """Test handling of unsafe original query."""
        # Configure mock to indicate safety issue
        mock_safety_check.return_value = (False, "Contains inappropriate content")

        # Call function
        result = query_ollama_with_hybrid_search_multilingual(
            self.mock_session,
            "How to hack a website?",
            self.mock_embedding_model,
            self.test_email,
            self.test_model,
        )

        # Verify safety check was called
        mock_safety_check.assert_called_once()

        # Verify the result indicates safety issue
        self.assertTrue(result["safety_issue"])
        self.assertIn("cannot process this request", result["response"])

    @patch("src.api.rag_pipeline.ollama.check_query_safety_with_llama_guard")
    @patch("src.api.rag_pipeline.ollama.detect_language")
    @patch("src.api.rag_pipeline.ollama.translate_text")
    def test_query_ollama_unsafe_translated(
        self, mock_translate, mock_detect_language, mock_safety_check
    ):
        """Test handling of unsafe translated query."""
        # Configure mocks
        mock_detect_language.return_value = "fr"

        # First safety check passes, second fails
        mock_safety_check.side_effect = [
            (True, ""),  # Original query is safe
            (False, "Contains inappropriate content"),  # Translated query is unsafe
        ]

        # Mock translations
        mock_translate.side_effect = [
            "How to hack a website?",  # French to English
            "Je ne peux pas traiter cette demande: Contains inappropriate content",  # English to French
        ]

        # Call function with French question
        result = query_ollama_with_hybrid_search_multilingual(
            self.mock_session,
            "Comment pirater un site web?",
            self.mock_embedding_model,
            self.test_email,
            self.test_model,
        )

        # Verify safety checks were called twice
        self.assertEqual(mock_safety_check.call_count, 2)

        # Verify translation was called twice
        self.assertEqual(mock_translate.call_count, 2)

        # Verify the result indicates safety issue
        self.assertTrue(result["safety_issue"])
        self.assertIn("ne peux pas traiter cette demande", result["response"])

    @patch("src.api.rag_pipeline.ollama.check_query_safety_with_llama_guard")
    @patch("src.api.rag_pipeline.ollama.detect_language")
    @patch("src.api.rag_pipeline.ollama.embed_query")
    def test_query_ollama_exception(
        self, mock_embed_query, mock_detect_language, mock_safety_check
    ):
        """Test handling of exceptions."""
        # Configure mocks
        mock_safety_check.return_value = (True, "")
        mock_detect_language.return_value = "en"
        mock_embed_query.side_effect = Exception("Embedding error")

        # Call function
        result = query_ollama_with_hybrid_search_multilingual(
            self.mock_session,
            self.test_question,
            self.mock_embedding_model,
            self.test_email,
            self.test_model,
        )

        # Verify error info is in result
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Embedding error")
        self.assertIn("Sorry", result["response"])

    @patch("src.api.rag_pipeline.ollama.check_query_safety_with_llama_guard")
    @patch("src.api.rag_pipeline.ollama.detect_language")
    @patch("src.api.rag_pipeline.ollama.embed_query")
    @patch("src.api.rag_pipeline.ollama.hybrid_search")
    @patch("src.api.rag_pipeline.ollama.rerank_with_llm")
    @patch("src.api.rag_pipeline.ollama.retrieve_document_metadata")
    @patch("src.api.rag_pipeline.ollama.query_llm")
    @patch("src.api.rag_pipeline.ollama.log_audit")
    def test_query_ollama_with_chat_history(
        self,
        mock_log_audit,
        mock_query_llm,
        mock_retrieve_metadata,
        mock_rerank,
        mock_hybrid_search,
        mock_embed_query,
        mock_detect_language,
        mock_safety_check,
    ):
        """Test querying Ollama with chat history."""
        # Configure mocks
        mock_safety_check.return_value = (True, "")
        mock_detect_language.return_value = "en"
        mock_embed_query.return_value = self.sample_embedding
        mock_hybrid_search.return_value = (self.sample_chunks, self.sample_sorted_results)
        mock_rerank.return_value = self.sample_chunks
        mock_retrieve_metadata.return_value = self.sample_metadata
        mock_query_llm.return_value = (
            "Deep learning is a subset of machine learning that uses neural networks."
        )

        # Sample chat history
        chat_history = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a field of AI..."},
            {"role": "user", "content": "Tell me more about neural networks"},
            {"role": "assistant", "content": "Neural networks are computing systems..."},
        ]

        # Call function with chat history
        result = query_ollama_with_hybrid_search_multilingual(
            self.mock_session,
            "How does it relate to deep learning?",
            self.mock_embedding_model,
            self.test_email,
            self.test_model,
            chat_history=chat_history,
        )

        # Verify query_llm was called with conversation context
        args, kwargs = mock_query_llm.call_args
        prompt = args[0]
        self.assertIn("PREVIOUS CONVERSATION:", prompt)
        self.assertIn("User: What is machine learning?", prompt)
        self.assertIn("User: Tell me more about neural networks", prompt)

        # Verify the result structure
        self.assertEqual(result["original_question"], "How does it relate to deep learning?")
        self.assertEqual(result["context_count"], 3)
        self.assertIn("Deep learning", result["response"])


if __name__ == "__main__":
    unittest.main()
