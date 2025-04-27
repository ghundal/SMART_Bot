from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

# Import the module to test
from src.api.utils.llm_rag_utils import (
    chat_sessions,
    create_chat_session,
    generate_chat_response,
    rebuild_chat_session,
)


class TestLlmRagUtils:
    """Tests for the llm_rag_utils.py module"""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model"""
        mock_model = MagicMock()
        return mock_model

    @pytest.fixture
    def mock_get_embedding_model(self, mock_embedding_model):
        """Mock the get_ch_embedding_model function"""
        with patch("src.api.utils.llm_rag_utils.get_ch_embedding_model") as mock:
            mock.return_value = mock_embedding_model
            yield mock

    @pytest.fixture
    def sample_chat_session(self):
        """Create a sample chat session"""
        return {"messages": [], "metadata": {"created_at": None, "last_updated": None}}

    @pytest.fixture
    def sample_message(self):
        """Create a sample message"""
        return {"content": "What is deep learning?"}

    @pytest.fixture
    def sample_chat_history(self):
        """Create a sample chat history"""
        return [
            {"message_id": "m1", "role": "user", "content": "What is machine learning?"},
            {"message_id": "m2", "role": "assistant", "content": "Machine learning is..."},
            {"message_id": "m3", "role": "user", "content": "How does it relate to AI?"},
            {"message_id": "m4", "role": "assistant", "content": "AI is a broader field..."},
        ]

    def test_create_chat_session(self):
        """Test creating a new chat session"""
        # Call the function
        session = create_chat_session()

        # Verify structure
        assert "messages" in session
        assert "metadata" in session
        assert isinstance(session["messages"], list)
        assert len(session["messages"]) == 0
        assert "created_at" in session["metadata"]
        assert "last_updated" in session["metadata"]

    @patch("builtins.__import__")
    def test_generate_chat_response(
        self, mock_import, mock_get_embedding_model, sample_chat_session, sample_message
    ):
        """Test generating a chat response"""
        # Setup mock modules and functions
        mock_query_fn = MagicMock(
            return_value={"response": "This is a test response", "context_count": 3}
        )
        mock_ollama_module = MagicMock()
        mock_ollama_module.query_ollama_with_hybrid_search_multilingual = mock_query_fn

        # Configure the import to return our mock module
        def mock_import_side_effect(name, *args, **kwargs):
            if name == "rag_pipeline.ollama":
                return mock_ollama_module
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = mock_import_side_effect

        # Setup SessionLocal mock
        with patch("src.api.utils.llm_rag_utils.SessionLocal") as mock_session_local:
            # Call the function
            response = generate_chat_response(
                sample_chat_session, sample_message, "user@example.com"
            )

            # Verify response
            assert response == "This is a test response"

            # Verify chat session was updated
            assert len(sample_chat_session["messages"]) == 2
            assert sample_chat_session["messages"][0]["role"] == "user"
            assert sample_chat_session["messages"][0]["content"] == "What is deep learning?"
            assert sample_chat_session["messages"][1]["role"] == "assistant"
            assert sample_chat_session["messages"][1]["content"] == "This is a test response"

            # Verify dependencies were called
            mock_get_embedding_model.assert_called_once()
            mock_query_fn.assert_called_once()
            mock_session_local.assert_called_once()

    @patch("builtins.__import__")
    def test_generate_chat_response_empty_content(
        self, mock_import, mock_get_embedding_model, sample_chat_session
    ):
        """Test handling empty content in generate_chat_response"""
        # Setup message with empty content
        empty_message = {"content": ""}

        # Setup mock modules
        mock_ollama_module = MagicMock()

        # Configure the import
        def mock_import_side_effect(name, *args, **kwargs):
            if name == "rag_pipeline.ollama":
                return mock_ollama_module
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = mock_import_side_effect

        # Call the function and expect an exception
        with pytest.raises(HTTPException) as excinfo:
            generate_chat_response(sample_chat_session, empty_message, "user@example.com")

        # Verify the exception
        assert excinfo.value.status_code == 500
        assert "Failed to generate response" in excinfo.value.detail
        assert "Message must contain text content" in excinfo.value.detail

        # Verify chat session was not modified
        assert len(sample_chat_session["messages"]) == 0

    @patch("builtins.__import__")
    def test_generate_chat_response_error(
        self, mock_import, mock_get_embedding_model, sample_chat_session, sample_message
    ):
        """Test error handling in generate_chat_response"""
        # Setup mock modules with error
        mock_query_fn = MagicMock(side_effect=Exception("Test error"))
        mock_ollama_module = MagicMock()
        mock_ollama_module.query_ollama_with_hybrid_search_multilingual = mock_query_fn

        # Configure the import
        def mock_import_side_effect(name, *args, **kwargs):
            if name == "rag_pipeline.ollama":
                return mock_ollama_module
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = mock_import_side_effect

        # Setup logger mock to avoid actual logging
        with patch("src.api.utils.llm_rag_utils.logger"), patch(
            "src.api.utils.llm_rag_utils.traceback.print_exc"
        ):

            # Call the function and expect an exception
            with pytest.raises(HTTPException) as excinfo:
                generate_chat_response(sample_chat_session, sample_message, "user@example.com")

            # Verify the exception
            assert excinfo.value.status_code == 500
            assert "Failed to generate response" in excinfo.value.detail
            assert "Test error" in excinfo.value.detail

    def test_rebuild_chat_session(self, sample_chat_history):
        """Test rebuilding a chat session from history"""
        # Call the function
        session = rebuild_chat_session(sample_chat_history)

        # Verify structure
        assert "messages" in session
        assert "metadata" in session

        # Verify messages
        assert len(session["messages"]) == 4
        assert session["messages"][0]["role"] == "user"
        assert session["messages"][0]["content"] == "What is machine learning?"
        assert session["messages"][1]["role"] == "assistant"
        assert session["messages"][1]["content"] == "Machine learning is..."
        assert session["messages"][2]["role"] == "user"
        assert session["messages"][2]["content"] == "How does it relate to AI?"
        assert session["messages"][3]["role"] == "assistant"
        assert session["messages"][3]["content"] == "AI is a broader field..."

        # Verify no message_id was included (only role and content)
        for msg in session["messages"]:
            assert "message_id" not in msg

    def test_rebuild_chat_session_with_invalid_messages(self):
        """Test rebuilding a chat session with invalid messages"""
        # Create chat history with some invalid messages
        chat_history = [
            {"message_id": "m1", "role": "user", "content": "Valid message"},
            {"message_id": "m2"},  # Missing role and content
            {"role": "assistant"},  # Missing content
            {"content": "Text only"},  # Missing role
        ]

        # Call the function
        session = rebuild_chat_session(chat_history)

        # Verify only valid messages were included
        assert len(session["messages"]) == 1
        assert session["messages"][0]["role"] == "user"
        assert session["messages"][0]["content"] == "Valid message"


if __name__ == "__main__":
    pytest.main(["-xvs", "tests/test_llm_rag_utils.py"])
