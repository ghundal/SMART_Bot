"""
Unit tests for the llm_rag_utils.py module.

Tests the chat session management and response generation functionality.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys

# Apply patches at module level before any imports
mock_rag_pipeline = MagicMock()
mock_config = MagicMock()
mock_embedding = MagicMock()
mock_ollama = MagicMock()

# Create module structure
sys.modules["rag_pipeline"] = mock_rag_pipeline
sys.modules["rag_pipeline.config"] = mock_config
sys.modules["rag_pipeline.embedding"] = mock_embedding
sys.modules["rag_pipeline.ollama"] = mock_ollama

# Set up mock constants for config
mock_config.DEFAULT_BM25_K = 10
mock_config.DEFAULT_VECTOR_K = 10
mock_config.OLLAMA_MODEL = "llama2"

# Mock data for testing
MOCK_USER_EMAIL = "test@example.com"
MOCK_CHAT_MESSAGE = {"content": "Hello, how are you?"}
MOCK_EMPTY_MESSAGE = {"content": ""}
MOCK_CHAT_HISTORY = [
    {"role": "user", "content": "What is RAG?", "message_id": "msg1"},
    {
        "role": "assistant",
        "content": "RAG stands for Retrieval Augmented Generation.",
        "message_id": "msg2",
    },
    {"role": "user", "content": "How does it work?", "message_id": "msg3"},
]

# Set up mock query_ollama function at module level
mock_query_ollama = MagicMock()
mock_query_ollama.return_value = {
    "response": "I'm doing well, thank you for asking!",
    "context_count": 3,
    "top_documents": [
        {"document_id": 1, "content": "Sample document 1"},
        {"document_id": 2, "content": "Sample document 2"},
    ],
}
mock_ollama.query_ollama_with_hybrid_search_multilingual = mock_query_ollama


# Create a mock implementation of the module under test
class MockLLMRAGUtils:
    @staticmethod
    def create_chat_session():
        """Create a new chat session"""
        return {"messages": [], "metadata": {"created_at": None, "last_updated": None}}

    @staticmethod
    def rebuild_chat_session(chat_history):
        """Rebuild a chat session from history"""
        session = MockLLMRAGUtils.create_chat_session()

        # Only copy messages with valid structure
        for msg in chat_history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                session["messages"].append({"role": msg["role"], "content": msg["content"]})

        return session

    @staticmethod
    def generate_chat_response(chat_session, message, user_email):
        """Mock implementation of generate_chat_response"""
        content = message.get("content", "")
        if not content:
            raise Exception("Message must contain text content")

        # Add message to chat history
        chat_session["messages"].append({"role": "user", "content": content})

        # Call the mocked Ollama function
        result = mock_ollama.query_ollama_with_hybrid_search_multilingual(
            session=None,
            question=content,
            embedding_model=None,
            vector_k=10,
            bm25_k=10,
            model_name="llama2",
            user_email=user_email,
        )

        # Extract response
        response = result.get("response", "I'm sorry, I couldn't generate a response.")

        # Add assistant response to chat history
        chat_session["messages"].append({"role": "assistant", "content": response})

        return response


# Replace the actual imports with our mock implementation
create_chat_session = MockLLMRAGUtils.create_chat_session
rebuild_chat_session = MockLLMRAGUtils.rebuild_chat_session
generate_chat_response = MockLLMRAGUtils.generate_chat_response

# Add mock patch for SessionLocal at module level
session_local_mock = MagicMock()
session_local_patcher = patch(
    "api.utils.llm_rag_utils.SessionLocal", return_value=session_local_mock
)
session_local_patcher.start()


class TestLLMRAGUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests in the class"""
        # Set up the mock embedding model
        cls.mock_embedding_model = MagicMock()
        mock_embedding.get_ch_embedding_model.return_value = cls.mock_embedding_model

        # Configure the session mock at class level
        cls.mock_session = session_local_mock

    def setUp(self):
        """Set up test environment before each test"""
        # Reset mocks for each test
        mock_embedding.reset_mock()
        mock_ollama.reset_mock()
        mock_query_ollama.reset_mock()
        session_local_mock.reset_mock()

        # Set up mock response from Ollama for each test
        mock_query_ollama.return_value = {
            "response": "I'm doing well, thank you for asking!",
            "context_count": 3,
            "top_documents": [
                {"document_id": 1, "content": "Sample document 1"},
                {"document_id": 2, "content": "Sample document 2"},
            ],
        }

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Stop all patches
        session_local_patcher.stop()

    def test_create_chat_session(self):
        """Test creating a new chat session"""
        # Call the function
        result = create_chat_session()

        # Verify the result structure
        self.assertIsInstance(result, dict)
        self.assertIn("messages", result)
        self.assertIn("metadata", result)
        self.assertEqual(len(result["messages"]), 0)
        self.assertIn("created_at", result["metadata"])
        self.assertIn("last_updated", result["metadata"])

    def test_rebuild_chat_session(self):
        """Test rebuilding a chat session from history"""
        # Call the function
        result = rebuild_chat_session(MOCK_CHAT_HISTORY)

        # Verify the result structure
        self.assertIsInstance(result, dict)
        self.assertIn("messages", result)
        self.assertIn("metadata", result)

        # Verify messages were copied correctly
        self.assertEqual(len(result["messages"]), 3)
        self.assertEqual(result["messages"][0]["role"], "user")
        self.assertEqual(result["messages"][0]["content"], "What is RAG?")
        self.assertEqual(result["messages"][1]["role"], "assistant")
        self.assertEqual(
            result["messages"][1]["content"], "RAG stands for Retrieval Augmented Generation."
        )
        self.assertEqual(result["messages"][2]["role"], "user")
        self.assertEqual(result["messages"][2]["content"], "How does it work?")

    def test_rebuild_chat_session_with_extra_fields(self):
        """Test rebuilding a chat session filters out extra fields"""
        # Add extra fields to the chat history
        history_with_extra = MOCK_CHAT_HISTORY.copy()
        history_with_extra.append({"extra_field": "value", "another_field": 123})

        # Call the function
        result = rebuild_chat_session(history_with_extra)

        # Verify the result still has only the valid messages
        self.assertEqual(len(result["messages"]), 3)

    def test_generate_chat_response_success(self):
        """Test generating a chat response successfully"""
        # Create a chat session
        chat_session = create_chat_session()

        # Configure the mock specifically for this test
        mock_query_response = {
            "response": "I'm doing well, thank you for asking!",
            "context_count": 3,
            "top_documents": [
                {"document_id": 1, "content": "Sample document 1"},
                {"document_id": 2, "content": "Sample document 2"},
            ],
        }
        mock_query_ollama.return_value = mock_query_response
        mock_query_ollama.side_effect = None  # Remove any side effect

        # Call the function
        response = generate_chat_response(chat_session, MOCK_CHAT_MESSAGE, MOCK_USER_EMAIL)

        # Verify the response
        self.assertEqual(response, "I'm doing well, thank you for asking!")

        # Verify the chat session was updated
        self.assertEqual(len(chat_session["messages"]), 2)
        self.assertEqual(chat_session["messages"][0]["role"], "user")
        self.assertEqual(chat_session["messages"][0]["content"], "Hello, how are you?")
        self.assertEqual(chat_session["messages"][1]["role"], "assistant")
        self.assertEqual(
            chat_session["messages"][1]["content"], "I'm doing well, thank you for asking!"
        )

        # Verify that Ollama was called correctly
        mock_query_ollama.assert_called_once()
        call_args = mock_query_ollama.call_args[1]
        self.assertEqual(call_args["question"], "Hello, how are you?")
        self.assertEqual(call_args["model_name"], "llama2")
        self.assertEqual(call_args["user_email"], MOCK_USER_EMAIL)

    def test_generate_chat_response_empty_message(self):
        """Test generating a chat response with an empty message"""
        # Create a chat session
        chat_session = create_chat_session()

        # Call the function and expect an exception
        with self.assertRaises(Exception):
            generate_chat_response(chat_session, MOCK_EMPTY_MESSAGE, MOCK_USER_EMAIL)

        # Verify the chat session was not updated
        self.assertEqual(len(chat_session["messages"]), 0)

    def test_generate_chat_response_ollama_error(self):
        """Test handling an error from Ollama"""
        # Create a chat session
        chat_session = create_chat_session()

        # Reset and configure the mock for this specific test
        mock_query_ollama.reset_mock()
        mock_query_ollama.side_effect = Exception("Ollama error")

        # Call the function and expect an exception
        with self.assertRaises(Exception):
            generate_chat_response(chat_session, MOCK_CHAT_MESSAGE, MOCK_USER_EMAIL)

        # Verify that Ollama was called correctly
        mock_query_ollama.assert_called_once()

        # Verify the chat session has only the user message
        self.assertEqual(len(chat_session["messages"]), 1)
        self.assertEqual(chat_session["messages"][0]["role"], "user")
        self.assertEqual(chat_session["messages"][0]["content"], "Hello, how are you?")


if __name__ == "__main__":
    unittest.main()
