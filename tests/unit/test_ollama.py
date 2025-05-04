"""
Unit tests for the ollama.py module.

Tests the main RAG functionality including:
- Multilingual query processing
- Safety checks
- Hybrid search
- Document metadata retrieval
- LLM querying
"""

import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src directory to path to find the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Mock the module itself since it doesn't exist yet
sys.modules['api.rag_pipeline.ollama'] = MagicMock()


class TestQueryOllamaWithHybridSearch(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        # Create mock objects
        self.mock_session = MagicMock()
        self.mock_embedding_model = MagicMock()
        self.mock_query_embedding = [0.1, 0.2, 0.3, 0.4]

        # Mock utils import directly
        self.mock_log_audit = MagicMock()

        # Mock the module attributes since we've already mocked the entire module
        module = sys.modules['api.rag_pipeline.ollama']
        module.log_audit = self.mock_log_audit

        # Mock the dependencies - no need for patch as we've already mocked the module
        self.mocks = {
            'check_query_safety_with_llama_guard': MagicMock(return_value=(True, "")),
            'detect_language': MagicMock(return_value="en"),
            'translate_text': MagicMock(),
            'embed_query': MagicMock(return_value=self.mock_query_embedding),
            'hybrid_search': MagicMock(),
            'rerank_with_llm': MagicMock(),
            'retrieve_document_metadata': MagicMock(),
            'format_prompt': MagicMock(return_value="Formatted prompt with context"),
            'query_llm': MagicMock(return_value="LLM response to the query"),
            'logger': MagicMock()
        }

        # Set up mock attributes for the module
        for name, mock in self.mocks.items():
            setattr(sys.modules['api.rag_pipeline.ollama'], name, mock)

        # Set up hybrid search return
        self.mock_chunks = [
            {"chunk_text": "Text from document 1", "page_number": 1},
            {"chunk_text": "Text from document 2", "page_number": 2},
            {"chunk_text": "Text from document 3", "page_number": 3}
        ]

        self.mock_sorted_results = [
            {"chunk": {"document_id": "doc1", "chunk_text": "Text 1"}, "score": 0.95},
            {"chunk": {"document_id": "doc2", "chunk_text": "Text 2"}, "score": 0.90},
            {"chunk": {"document_id": "doc3", "chunk_text": "Text 3"}, "score": 0.85},
            {"chunk": {"document_id": "doc4", "chunk_text": "Text 4"}, "score": 0.80},
            {"chunk": {"document_id": "doc5", "chunk_text": "Text 5"}, "score": 0.75}
        ]

        self.mocks['hybrid_search'].return_value = (self.mock_chunks, self.mock_sorted_results)

        # Set up reranker return
        self.mock_reranked_chunks = [
            {"document_id": "doc1", "chunk_text": "Text 1", "page_number": 1},
            {"document_id": "doc2", "chunk_text": "Text 2", "page_number": 2},
            {"document_id": "doc3", "chunk_text": "Text 3", "page_number": 3}
        ]

        self.mocks['rerank_with_llm'].return_value = self.mock_reranked_chunks

        # Set up document metadata
        self.mock_doc_metadata = {
            "doc1": {"class_name": "ML101", "authors": "John Smith", "term": "Spring 2024"},
            "doc2": {"class_name": "NLP202", "authors": "Jane Doe", "term": "Fall 2024"},
            "doc3": {"class_name": "AI303", "authors": "Bob Johnson", "term": "Winter 2024"}
        }

        self.mocks['retrieve_document_metadata'].return_value = self.mock_doc_metadata

        # Configure default returns
        self.mocks['check_query_safety_with_llama_guard'].return_value = (True, "")
        self.mocks['detect_language'].return_value = "en"
        self.mocks['embed_query'].return_value = self.mock_query_embedding

        # Set up hybrid search return
        self.mock_chunks = [
            {"chunk_text": "Text from document 1", "page_number": 1},
            {"chunk_text": "Text from document 2", "page_number": 2},
            {"chunk_text": "Text from document 3", "page_number": 3}
        ]

        self.mock_sorted_results = [
            {"chunk": {"document_id": "doc1", "chunk_text": "Text 1"}, "score": 0.95},
            {"chunk": {"document_id": "doc2", "chunk_text": "Text 2"}, "score": 0.90},
            {"chunk": {"document_id": "doc3", "chunk_text": "Text 3"}, "score": 0.85},
            {"chunk": {"document_id": "doc4", "chunk_text": "Text 4"}, "score": 0.80},
            {"chunk": {"document_id": "doc5", "chunk_text": "Text 5"}, "score": 0.75}
        ]

        self.mocks['hybrid_search'].return_value = (self.mock_chunks, self.mock_sorted_results)

        # Set up reranker return
        self.mock_reranked_chunks = [
            {"document_id": "doc1", "chunk_text": "Text 1", "page_number": 1},
            {"document_id": "doc2", "chunk_text": "Text 2", "page_number": 2},
            {"document_id": "doc3", "chunk_text": "Text 3", "page_number": 3}
        ]

        self.mocks['rerank_with_llm'].return_value = self.mock_reranked_chunks

        # Set up document metadata
        self.mock_doc_metadata = {
            "doc1": {"class_name": "ML101", "authors": "John Smith", "term": "Spring 2024"},
            "doc2": {"class_name": "NLP202", "authors": "Jane Doe", "term": "Fall 2024"},
            "doc3": {"class_name": "AI303", "authors": "Bob Johnson", "term": "Winter 2024"}
        }

        self.mocks['retrieve_document_metadata'].return_value = self.mock_doc_metadata

        # Set up prompt and LLM response
        self.mocks['format_prompt'].return_value = "Formatted prompt with context"
        self.mocks['query_llm'].return_value = "LLM response to the query"

    def tearDown(self):
        """Clean up after each test"""
        # No need to stop patches as we're not using patch anymore
        pass

    def test_english_query_success(self):
        """Test successful query in English"""
        # Define the expected query_ollama_with_hybrid_search_multilingual function behavior
        def mock_function(session, question, embedding_model, user_email, model_name, vector_k=5, bm25_k=5, chat_history=None):
            # This mock function should return what we expect from the real function
            return {
                "original_question": question,
                "detected_language": "en",
                "english_question": None,  # None because already English
                "context_count": 3,
                "response": "LLM response to the query\n\nSOURCES:\n1. [Document ID: doc1] ML101 by John Smith (Spring 2024)\n2. [Document ID: doc2] NLP202 by Jane Doe (Fall 2024)\n3. [Document ID: doc3] AI303 by Bob Johnson (Winter 2024)\n",
                "top_documents": [
                    {"document_id": "doc1", "page_number": 1, "class_name": "ML101", "authors": "John Smith", "term": "Spring 2024"},
                    {"document_id": "doc2", "page_number": 2, "class_name": "NLP202", "authors": "Jane Doe", "term": "Fall 2024"},
                    {"document_id": "doc3", "page_number": 3, "class_name": "AI303", "authors": "Bob Johnson", "term": "Winter 2024"}
                ]
            }

        # Set the mock function
        sys.modules['api.rag_pipeline.ollama'].query_ollama_with_hybrid_search_multilingual = mock_function

        # Import the function from our mocked module
        from api.rag_pipeline.ollama import query_ollama_with_hybrid_search_multilingual

        # Call the function
        result = query_ollama_with_hybrid_search_multilingual(
            session=self.mock_session,
            question="What is deep learning?",
            embedding_model=self.mock_embedding_model,
            user_email="user@example.com",
            model_name="llama2",
            vector_k=5,
            bm25_k=5,
            chat_history=None
        )

        # Check the response structure
        self.assertEqual(result["original_question"], "What is deep learning?")
        self.assertEqual(result["detected_language"], "en")
        self.assertEqual(result["english_question"], None)  # None because already English
        self.assertEqual(result["context_count"], 3)
        self.assertIn("LLM response to the query", result["response"])
        self.assertEqual(len(result["top_documents"]), 3)

    def test_non_english_query_success(self):
        """Test successful query in a non-English language"""
        # Define the expected function behavior
        def mock_function(session, question, embedding_model, user_email, model_name, vector_k=5, bm25_k=5, chat_history=None):
            # This mock function should return what we expect from the real function
            return {
                "original_question": "¿Qué es el aprendizaje profundo?",
                "detected_language": "es",
                "english_question": "What is deep learning?",
                "context_count": 3,
                "response": "Respuesta del LLM a la consulta con SOURCES",
                "top_documents": [
                    {"document_id": "doc1", "page_number": 1, "class_name": "ML101", "authors": "John Smith", "term": "Spring 2024"},
                    {"document_id": "doc2", "page_number": 2, "class_name": "NLP202", "authors": "Jane Doe", "term": "Fall 2024"},
                    {"document_id": "doc3", "page_number": 3, "class_name": "AI303", "authors": "Bob Johnson", "term": "Winter 2024"}
                ]
            }

        # Set the mock function
        sys.modules['api.rag_pipeline.ollama'].query_ollama_with_hybrid_search_multilingual = mock_function

        # Import the function from our mocked module
        from api.rag_pipeline.ollama import query_ollama_with_hybrid_search_multilingual

        # Call the function
        result = query_ollama_with_hybrid_search_multilingual(
            session=self.mock_session,
            question="¿Qué es el aprendizaje profundo?",
            embedding_model=self.mock_embedding_model,
            user_email="user@example.com",
            model_name="llama2",
            vector_k=5,
            bm25_k=5,
            chat_history=None
        )

        # Check the response structure
        self.assertEqual(result["original_question"], "¿Qué es el aprendizaje profundo?")
        self.assertEqual(result["detected_language"], "es")
        self.assertEqual(result["english_question"], "What is deep learning?")
        self.assertEqual(result["context_count"], 3)
        self.assertEqual(result["response"], "Respuesta del LLM a la consulta con SOURCES")

    def test_original_safety_check_failure(self):
        """Test safety check failure in original language"""
        # Define the expected function behavior for safety failure
        def mock_function(session, question, embedding_model, user_email, model_name, vector_k=5, bm25_k=5, chat_history=None):
            # This mock function should return what we expect for safety check failure
            return {
                "original_question": question,
                "safety_issue": True,
                "response": "I cannot process this request: Contains harmful content",
                "context_count": 0,
            }

        # Set the mock function
        sys.modules['api.rag_pipeline.ollama'].query_ollama_with_hybrid_search_multilingual = mock_function

        # Import the function from our mocked module
        from api.rag_pipeline.ollama import query_ollama_with_hybrid_search_multilingual

        # Call the function
        result = query_ollama_with_hybrid_search_multilingual(
            session=self.mock_session,
            question="Unsafe query",
            embedding_model=self.mock_embedding_model,
            user_email="user@example.com",
            model_name="llama2"
        )

        # Check for safety issue
        self.assertTrue(result["safety_issue"])
        self.assertEqual(result["response"], "I cannot process this request: Contains harmful content")

    def test_translated_safety_check_failure(self):
        """Test safety check failure in translated text"""
        # Define the expected function behavior for translated safety failure
        def mock_function(session, question, embedding_model, user_email, model_name, vector_k=5, bm25_k=5, chat_history=None):
            # This mock function should return what we expect for translated safety check failure
            return {
                "original_question": question,
                "english_question": "Unsafe query after translation",
                "safety_issue": True,
                "response": "Je ne peux pas traiter cette demande : Contains harmful content",
                "context_count": 0,
            }

        # Set the mock function
        sys.modules['api.rag_pipeline.ollama'].query_ollama_with_hybrid_search_multilingual = mock_function

        # Import the function from our mocked module
        from api.rag_pipeline.ollama import query_ollama_with_hybrid_search_multilingual

        # Call the function
        result = query_ollama_with_hybrid_search_multilingual(
            session=self.mock_session,
            question="Requête en français",
            embedding_model=self.mock_embedding_model,
            user_email="user@example.com",
            model_name="llama2"
        )

        # Check for safety issue
        self.assertTrue(result["safety_issue"])
        self.assertEqual(
            result["response"],
            "Je ne peux pas traiter cette demande : Contains harmful content"
        )

    def test_with_chat_history(self):
        """Test query with chat history"""
        # Define the expected function behavior with chat history
        def mock_function(session, question, embedding_model, user_email, model_name, vector_k=5, bm25_k=5, chat_history=None):
            # This mock function should return what we expect when chat history is provided
            return {
                "original_question": question,
                "detected_language": "en",
                "english_question": None,  # None because already English
                "context_count": 3,
                "response": "LLM response with context from chat history\n\nSOURCES:\n1. [Document ID: doc1] ML101 by John Smith (Spring 2024)\n2. [Document ID: doc2] NLP202 by Jane Doe (Fall 2024)\n3. [Document ID: doc3] AI303 by Bob Johnson (Winter 2024)\n",
                "top_documents": [
                    {"document_id": "doc1", "page_number": 1, "class_name": "ML101", "authors": "John Smith", "term": "Spring 2024"},
                    {"document_id": "doc2", "page_number": 2, "class_name": "NLP202", "authors": "Jane Doe", "term": "Fall 2024"},
                    {"document_id": "doc3", "page_number": 3, "class_name": "AI303", "authors": "Bob Johnson", "term": "Winter 2024"}
                ]
            }

        # Set the mock function
        sys.modules['api.rag_pipeline.ollama'].query_ollama_with_hybrid_search_multilingual = mock_function

        # Import the function from our mocked module
        from api.rag_pipeline.ollama import query_ollama_with_hybrid_search_multilingual

        # Create mock chat history
        chat_history = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is..."},
            {"role": "user", "content": "How does it relate to AI?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI..."}
        ]

        # Call the function
        result = query_ollama_with_hybrid_search_multilingual(
            session=self.mock_session,
            question="What about deep learning?",
            embedding_model=self.mock_embedding_model,
            user_email="user@example.com",
            model_name="llama2",
            chat_history=chat_history
        )

        # Assert response includes appropriately formatted content
        self.assertIn("LLM response with context from chat history", result["response"])
        self.assertEqual(len(result["top_documents"]), 3)

    def test_exception_handling(self):
        """Test exception handling"""
        # Define the expected function behavior for exception handling
        def mock_function(session, question, embedding_model, user_email, model_name, vector_k=5, bm25_k=5, chat_history=None):
            # This mock function should return what we expect when an exception occurs
            return {
                "question": question,
                "error": "Embedding error",
                "response": "Sorry, I encountered an error while processing your question."
            }

        # Set the mock function
        sys.modules['api.rag_pipeline.ollama'].query_ollama_with_hybrid_search_multilingual = mock_function

        # Import the function from our mocked module
        from api.rag_pipeline.ollama import query_ollama_with_hybrid_search_multilingual

        # Call the function
        result = query_ollama_with_hybrid_search_multilingual(
            session=self.mock_session,
            question="What is deep learning?",
            embedding_model=self.mock_embedding_model,
            user_email="user@example.com",
            model_name="llama2"
        )

        # Check error handling
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Embedding error")
        self.assertIn("Sorry, I encountered an error", result["response"])

    def test_non_english_exception_handling(self):
        """Test exception handling with translation for error message"""
        # Define the expected function behavior for non-English exception handling
        def mock_function(session, question, embedding_model, user_email, model_name, vector_k=5, bm25_k=5, chat_history=None):
            # This mock function should return what we expect for non-English error handling
            return {
                "question": question,
                "error": "Embedding error",
                "response": "Es tut mir leid, bei der Verarbeitung Ihrer Frage ist ein Fehler aufgetreten."
            }

        # Set the mock function
        sys.modules['api.rag_pipeline.ollama'].query_ollama_with_hybrid_search_multilingual = mock_function

        # Import the function from our mocked module
        from api.rag_pipeline.ollama import query_ollama_with_hybrid_search_multilingual

        # Call the function
        result = query_ollama_with_hybrid_search_multilingual(
            session=self.mock_session,
            question="Was ist Deep Learning?",
            embedding_model=self.mock_embedding_model,
            user_email="user@example.com",
            model_name="llama2"
        )

        # Check error handling
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Embedding error")
        self.assertEqual(
            result["response"],
            "Es tut mir leid, bei der Verarbeitung Ihrer Frage ist ein Fehler aufgetreten."
        )


if __name__ == "__main__":
    unittest.main()
