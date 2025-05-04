"""
Unit tests for the chat_api.py module.

Tests the RAG chatbot API endpoints including:
- Creating new chats
- Continuing existing chats
- Retrieving chat history
- Deleting chats
- Processing queries
"""

import json
import sys
import os
import time
import unittest
import uuid
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

# Mock data for testing
MOCK_USER_EMAIL = "test@example.com"
MOCK_SESSION_ID = "test-session-123"
MOCK_CHAT_ID = "12345678-1234-5678-1234-567812345678"
MOCK_MESSAGE_ID = "87654321-8765-4321-8765-432187654321"
MOCK_QUESTION = "What is RAG?"
MOCK_RESPONSE = "RAG stands for Retrieval Augmented Generation."
MOCK_CHAT = {
    "chat_id": MOCK_CHAT_ID,
    "title": MOCK_QUESTION,
    "dts": int(time.time()),
    "messages": [
        {"message_id": MOCK_MESSAGE_ID, "role": "user", "content": MOCK_QUESTION},
        {"message_id": str(uuid.uuid4()), "role": "assistant", "content": MOCK_RESPONSE},
    ],
}
MOCK_CHATS = [MOCK_CHAT]
MOCK_OLLAMA_RESULT = {
    "response": MOCK_RESPONSE,
    "top_documents": [
        {"id": 1, "content": "Document about RAG", "metadata": {"source": "article1.pdf"}},
        {"id": 2, "content": "More about retrieval", "metadata": {"source": "article2.pdf"}},
    ]
}


class TestChatAPI(unittest.TestCase):
    def setUp(self):
        # Create mock objects
        self.mock_db = MagicMock(spec=Session)
        self.mock_session_local = MagicMock(return_value=self.mock_db)

        # Mock the chat manager
        self.mock_chat_manager = MagicMock()
        self.mock_chat_manager.get_recent_chats.return_value = MOCK_CHATS
        self.mock_chat_manager.get_chat.return_value = MOCK_CHAT
        self.mock_chat_manager.save_chat.return_value = True
        self.mock_chat_manager.delete_chat.return_value = True

        # Mock the embedding model
        self.mock_embedding_model = MagicMock()

        # Mock the query function
        self.mock_query_ollama = MagicMock(return_value=MOCK_OLLAMA_RESULT)

        # Mock auth_middleware verify_token
        self.mock_verify_token = AsyncMock(return_value=MOCK_USER_EMAIL)

        # Mock uuid
        self.mock_uuid = MagicMock()
        self.mock_uuid.uuid4.return_value = MOCK_CHAT_ID

        # Set up module patching
        self.setup_module_patching()

    def setup_module_patching(self):
        """Set up all the module patches"""
        # Create mock modules
        self.mock_modules = {
            'utils': MagicMock(),
            'utils.chat_history': MagicMock(),
            'utils.database': MagicMock(),
            'utils.llm_rag_utils': MagicMock(),
            'rag_pipeline': MagicMock(),
            'rag_pipeline.config': MagicMock(),
            'rag_pipeline.embedding': MagicMock(),
            'rag_pipeline.ollama': MagicMock(),
        }

        # Configure mock modules
        self.mock_modules['utils.chat_history'].ChatHistoryManager = MagicMock(return_value=self.mock_chat_manager)
        self.mock_modules['utils.database'].SessionLocal = self.mock_session_local
        self.mock_modules['utils.llm_rag_utils'].chat_sessions = {}
        self.mock_modules['utils.llm_rag_utils'].create_chat_session = MagicMock(return_value={})
        self.mock_modules['utils.llm_rag_utils'].rebuild_chat_session = MagicMock(return_value={})
        self.mock_modules['rag_pipeline.config'].DEFAULT_VECTOR_K = 10
        self.mock_modules['rag_pipeline.config'].DEFAULT_BM25_K = 10
        self.mock_modules['rag_pipeline.config'].OLLAMA_MODEL = "llama2"
        self.mock_modules['rag_pipeline.embedding'].get_ch_embedding_model = MagicMock(return_value=self.mock_embedding_model)
        self.mock_modules['rag_pipeline.ollama'].query_ollama_with_hybrid_search_multilingual = self.mock_query_ollama

        # Add mock modules to sys.modules
        for name, module in self.mock_modules.items():
            sys.modules[name] = module

        # Patch uuid
        self.uuid_patcher = patch('uuid.uuid4', return_value=MOCK_CHAT_ID)
        self.uuid_patcher.start()

        # Patch time
        self.time_patcher = patch('time.time', return_value=12345)
        self.time_patcher.start()

    def tearDown(self):
        # Stop all patches
        self.uuid_patcher.stop()
        self.time_patcher.stop()

        # Remove mock modules
        for name in self.mock_modules:
            if name in sys.modules:
                del sys.modules[name]

    def import_chat_api(self):
        """Import the chat_api module directly from file"""
        # Add src to the Python path
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

        # Import the module directly using its file path
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "chat_api",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/api/routers/chat_api.py"))
        )
        chat_api = importlib.util.module_from_spec(spec)

        # Execute the module
        spec.loader.exec_module(chat_api)

        # Replace dependencies with mocks
        chat_api.verify_token = self.mock_verify_token
        chat_api.chat_manager = self.mock_chat_manager
        chat_api.embedding_model = self.mock_embedding_model
        chat_api.query_ollama_with_hybrid_search_multilingual = self.mock_query_ollama

        return chat_api

    async def test_get_chats(self):
        """Test the get_chats endpoint"""
        # Import the module
        chat_api = self.import_chat_api()

        # Call the get_chats function
        result = await chat_api.get_chats(x_session_id=MOCK_SESSION_ID, limit=10, user_email=MOCK_USER_EMAIL)

        # Verify the chat manager was called correctly
        self.mock_chat_manager.get_recent_chats.assert_called_once_with(MOCK_USER_EMAIL, 10)

        # Verify the result is what we expect
        assert result == MOCK_CHATS

    async def test_get_chat(self):
        """Test the get_chat endpoint"""
        # Import the module
        chat_api = self.import_chat_api()

        # Call the get_chat function
        result = await chat_api.get_chat(
            chat_id=MOCK_CHAT_ID,
            x_session_id=MOCK_SESSION_ID,
            user_email=MOCK_USER_EMAIL
        )

        # Verify the chat manager was called correctly
        self.mock_chat_manager.get_chat.assert_called_once_with(MOCK_CHAT_ID)

        # Verify the result is what we expect
        assert result == MOCK_CHAT

    async def test_get_chat_not_found(self):
        """Test the get_chat endpoint with a chat that doesn't exist"""
        # Import the module
        chat_api = self.import_chat_api()

        # Set up the chat manager to return None
        self.mock_chat_manager.get_chat.return_value = None

        # Call the get_chat function and expect an error
        with pytest.raises(HTTPException) as exc_info:
            await chat_api.get_chat(
                chat_id=MOCK_CHAT_ID,
                x_session_id=MOCK_SESSION_ID,
                user_email=MOCK_USER_EMAIL
            )

        # Verify the correct error was raised
        assert exc_info.value.status_code == 404
        assert "Chat not found" in exc_info.value.detail

    async def test_start_chat_with_llm(self):
        """Test the start_chat_with_llm endpoint"""
        # Import the module
        chat_api = self.import_chat_api()

        # Create a message for the request
        message = {"content": MOCK_QUESTION, "model": "llama2"}

        # Call the start_chat_with_llm function
        result = await chat_api.start_chat_with_llm(
            message=message,
            x_session_id=MOCK_SESSION_ID,
            user_email=MOCK_USER_EMAIL
        )

        # Verify the query function was called correctly
        self.mock_query_ollama.assert_called_once()

        # Verify the chat was saved
        self.mock_chat_manager.save_chat.assert_called_once()

        # Verify the result has the expected structure
        assert "chat_id" in result
        assert "title" in result
        assert "dts" in result
        assert "messages" in result
        assert "top_documents" in result
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    async def test_start_chat_missing_content(self):
        """Test the start_chat_with_llm endpoint with missing content"""
        # Import the module
        chat_api = self.import_chat_api()

        # Create a message for the request with no content
        message = {"model": "llama2"}

        # Call the start_chat_with_llm function and expect an error
        with pytest.raises(HTTPException) as exc_info:
            await chat_api.start_chat_with_llm(
                message=message,
                x_session_id=MOCK_SESSION_ID,
                user_email=MOCK_USER_EMAIL
            )

        # Verify the correct error was raised
        assert exc_info.value.status_code == 400
        assert "Message content is required" in exc_info.value.detail

    async def test_continue_chat_with_llm(self):
        """Test the continue_chat_with_llm endpoint"""
        # Import the module
        chat_api = self.import_chat_api()

        # Create a message for the request
        message = {"content": "Follow-up question", "model": "llama2"}

        # Call the continue_chat_with_llm function
        result = await chat_api.continue_chat_with_llm(
            chat_id=MOCK_CHAT_ID,
            message=message,
            x_session_id=MOCK_SESSION_ID,
            user_email=MOCK_USER_EMAIL
        )

        # Verify the chat manager was called correctly
        self.mock_chat_manager.get_chat.assert_called_once_with(MOCK_CHAT_ID)

        # Verify the query function was called correctly
        self.mock_query_ollama.assert_called_once()

        # Verify the chat was saved
        self.mock_chat_manager.save_chat.assert_called_once()

        # Verify the result has the expected structure
        assert "chat_id" in result
        assert "messages" in result
        assert len(result["messages"]) > 2  # Original messages plus the new ones

    async def test_continue_chat_not_found(self):
        """Test the continue_chat_with_llm endpoint with a chat that doesn't exist"""
        # Import the module
        chat_api = self.import_chat_api()

        # Set up the chat manager to return None
        self.mock_chat_manager.get_chat.return_value = None

        # Create a message for the request
        message = {"content": "Follow-up question", "model": "llama2"}

        # Call the continue_chat_with_llm function and expect an error
        with pytest.raises(HTTPException) as exc_info:
            await chat_api.continue_chat_with_llm(
                chat_id=MOCK_CHAT_ID,
                message=message,
                x_session_id=MOCK_SESSION_ID,
                user_email=MOCK_USER_EMAIL
            )

        # Verify the correct error was raised
        assert exc_info.value.status_code == 404
        assert "Chat not found" in exc_info.value.detail

    async def test_delete_chat(self):
        """Test the delete_chat endpoint"""
        # Import the module
        chat_api = self.import_chat_api()

        # Set up chat_sessions with the mock chat
        self.mock_modules['utils.llm_rag_utils'].chat_sessions = {MOCK_CHAT_ID: {}}

        # Call the delete_chat function
        result = await chat_api.delete_chat(
            chat_id=MOCK_CHAT_ID,
            x_session_id=MOCK_SESSION_ID,
            user_email=MOCK_USER_EMAIL
        )

        # Verify the chat manager was called correctly
        self.mock_chat_manager.delete_chat.assert_called_once_with(MOCK_CHAT_ID)

        # Verify the result has the expected structure
        assert result["status"] == "success"
        assert f"Chat {MOCK_CHAT_ID} deleted successfully" in result["message"]

        # Verify the chat was removed from chat_sessions
        assert MOCK_CHAT_ID not in self.mock_modules['utils.llm_rag_utils'].chat_sessions

    async def test_delete_chat_not_found(self):
        """Test the delete_chat endpoint with a chat that doesn't exist"""
        # Import the module
        chat_api = self.import_chat_api()

        # Set up the chat manager to return False for delete
        self.mock_chat_manager.delete_chat.return_value = False

        # Call the delete_chat function and expect an error
        with pytest.raises(HTTPException) as exc_info:
            await chat_api.delete_chat(
                chat_id=MOCK_CHAT_ID,
                x_session_id=MOCK_SESSION_ID,
                user_email=MOCK_USER_EMAIL
            )

        # Verify the correct error was raised
        assert exc_info.value.status_code == 404
        assert "Chat not found or could not be deleted" in exc_info.value.detail

    async def test_process_query_new_chat(self):
        """Test the process_query endpoint creating a new chat"""
        # Import the module
        chat_api = self.import_chat_api()

        # Create a request object for the query
        class MockRequest:
            def __init__(self):
                self.chat_id = ""  # Empty for new chat
                self.question = MOCK_QUESTION
                self.model_name = "llama2"
                self.session_id = MOCK_SESSION_ID

        request = MockRequest()

        # Call the process_query function
        result = await chat_api.process_query(request, user_email=MOCK_USER_EMAIL)

        # Verify the query function was called correctly
        self.mock_query_ollama.assert_called_once()

        # Verify the chat was saved
        self.mock_chat_manager.save_chat.assert_called_once()

        # Verify the result has the expected structure
        assert "chat_id" in result
        assert "title" in result
        assert "dts" in result
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    async def test_process_query_existing_chat(self):
        """Test the process_query endpoint continuing an existing chat"""
        # Import the module
        chat_api = self.import_chat_api()

        # Create a request object for the query
        class MockRequest:
            def __init__(self):
                self.chat_id = MOCK_CHAT_ID  # Existing chat
                self.question = "Follow-up question"
                self.model_name = "llama2"
                self.session_id = MOCK_SESSION_ID

        request = MockRequest()

        # Call the process_query function
        result = await chat_api.process_query(request, user_email=MOCK_USER_EMAIL)

        # Verify the chat manager was called correctly
        self.mock_chat_manager.get_chat.assert_called_once_with(MOCK_CHAT_ID)

        # Verify the query function was called correctly
        self.mock_query_ollama.assert_called_once()

        # Verify the chat was saved
        self.mock_chat_manager.save_chat.assert_called_once()

        # Verify the result has the expected structure
        assert "chat_id" in result
        assert "messages" in result
        assert len(result["messages"]) > 2  # Original messages plus the new ones

    async def test_process_query_missing_question(self):
        """Test the process_query endpoint with missing question"""
        # Import the module
        chat_api = self.import_chat_api()

        # Create a request object for the query with no question
        class MockRequest:
            def __init__(self):
                self.chat_id = MOCK_CHAT_ID
                self.question = ""
                self.model_name = "llama2"
                self.session_id = MOCK_SESSION_ID

        request = MockRequest()

        # Call the process_query function and expect an error
        with pytest.raises(HTTPException) as exc_info:
            await chat_api.process_query(request, user_email=MOCK_USER_EMAIL)

        # Verify the correct error was raised
        assert exc_info.value.status_code == 400
        assert "Question is required" in exc_info.value.detail

    async def test_process_query_missing_session_id(self):
        """Test the process_query endpoint with missing session ID"""
        # Import the module
        chat_api = self.import_chat_api()

        # Create a request object for the query with no session ID
        class MockRequest:
            def __init__(self):
                self.chat_id = MOCK_CHAT_ID
                self.question = MOCK_QUESTION
                self.model_name = "llama2"
                self.session_id = ""

        request = MockRequest()

        # Call the process_query function and expect an error
        with pytest.raises(HTTPException) as exc_info:
            await chat_api.process_query(request, user_email=MOCK_USER_EMAIL)

        # Verify the correct error was raised
        assert exc_info.value.status_code == 400
        assert "Session ID is required" in exc_info.value.detail


if __name__ == '__main__':
    unittest.main()
