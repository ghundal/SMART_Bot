import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

# Import the router and related components
from src.api.routers.chat_api import (
    QueryRequest,
    continue_chat_with_llm,
    delete_chat,
    get_chat,
    get_chats,
    process_query,
    router,
    start_chat_with_llm,
)


class TestChatAPI:
    """Tests for the chat_api.py module"""

    @pytest.fixture
    def mock_chat_manager(self):
        """Create a mock chat manager"""
        mock_manager = MagicMock()
        mock_manager.get_recent_chats = MagicMock()
        mock_manager.get_chat = MagicMock()
        mock_manager.save_chat = MagicMock()
        mock_manager.delete_chat = MagicMock()
        return mock_manager

    @pytest.fixture
    def mock_chat_data(self):
        """Sample chat data for testing"""
        chat_id = str(uuid.uuid4())
        return {
            "chat_id": chat_id,
            "title": "Test chat",
            "dts": int(time.time()),
            "messages": [
                {"message_id": str(uuid.uuid4()), "role": "user", "content": "Hello"},
                {"message_id": str(uuid.uuid4()), "role": "assistant", "content": "Hi there!"},
            ],
            "top_documents": [
                {
                    "document_id": "doc1",
                    "page_number": 1,
                    "class_name": "Introduction to AI",
                    "authors": "John Doe",
                    "term": "Spring 2025",
                }
            ],
        }

    @pytest.fixture
    def mock_query_result(self):
        """Sample query result for testing"""
        return {
            "response": "This is a test response",
            "top_documents": [
                {
                    "document_id": "doc1",
                    "page_number": 1,
                    "class_name": "Introduction to AI",
                    "authors": "John Doe",
                    "term": "Spring 2025",
                }
            ],
        }

    @pytest.mark.asyncio
    async def test_get_chats(self, mock_chat_manager):
        """Test getting all chats"""
        # Mock data
        mock_chats = [{"chat_id": "1"}, {"chat_id": "2"}]
        mock_chat_manager.get_recent_chats.return_value = mock_chats

        # Test with mocks
        with patch("src.api.routers.chat_api.chat_manager", mock_chat_manager):
            result = await get_chats(
                x_session_id="test-session", limit=10, user_email="test@example.com"
            )

        # Verify calls and results
        mock_chat_manager.get_recent_chats.assert_called_once_with("test-session", 10)
        assert result == mock_chats

    @pytest.mark.asyncio
    async def test_get_chat_success(self, mock_chat_manager, mock_chat_data):
        """Test getting a specific chat successfully"""
        # Mock data
        mock_chat_manager.get_chat.return_value = mock_chat_data

        # Test with mocks
        with patch("src.api.routers.chat_api.chat_manager", mock_chat_manager):
            result = await get_chat(
                chat_id=mock_chat_data["chat_id"],
                x_session_id="test-session",
                user_email="test@example.com",
            )

        # Verify calls and results
        mock_chat_manager.get_chat.assert_called_once_with(
            mock_chat_data["chat_id"], "test-session"
        )
        assert result == mock_chat_data

    @pytest.mark.asyncio
    async def test_get_chat_not_found(self, mock_chat_manager):
        """Test getting a non-existent chat"""
        # Mock data - chat not found
        mock_chat_manager.get_chat.return_value = None

        # Test with mocks
        with patch("src.api.routers.chat_api.chat_manager", mock_chat_manager):
            with pytest.raises(HTTPException) as excinfo:
                await get_chat(
                    chat_id="non-existent-id",
                    x_session_id="test-session",
                    user_email="test@example.com",
                )

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Chat not found" in excinfo.value.detail

    @pytest.mark.asyncio
    async def test_start_chat_with_llm(self, mock_chat_manager, mock_query_result):
        """Test starting a new chat"""
        # Mock the query function
        mock_query = MagicMock(return_value=mock_query_result)

        # Test with mocks
        with patch(
            "src.api.routers.chat_api.query_ollama_with_hybrid_search_multilingual", mock_query
        ), patch("src.api.routers.chat_api.chat_manager", mock_chat_manager), patch(
            "src.api.routers.chat_api.uuid.uuid4", MagicMock(return_value="test-uuid")
        ), patch(
            "src.api.routers.chat_api.time.time", MagicMock(return_value=12345)
        ), patch(
            "src.api.routers.chat_api.SessionLocal", MagicMock()
        ):

            result = await start_chat_with_llm(
                message={"content": "Test question", "model": "test-model"},
                x_session_id="test-session",
                user_email="test@example.com",
            )

        # Verify function calls
        mock_query.assert_called_once()
        mock_chat_manager.save_chat.assert_called_once()

        # Verify result structure
        assert result["chat_id"] == "test-uuid"
        assert result["title"] == "Test question"
        assert result["dts"] == 12345
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Test question"
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][1]["content"] == "This is a test response"
        assert "top_documents" in result

    @pytest.mark.asyncio
    async def test_start_chat_with_empty_content(self, mock_chat_manager):
        """Test starting a chat with empty content"""
        # Test with empty content
        with pytest.raises(HTTPException) as excinfo:
            await start_chat_with_llm(
                message={"content": ""}, x_session_id="test-session", user_email="test@example.com"
            )

        # Verify exception
        assert excinfo.value.status_code == 400
        assert "Message content is required" in excinfo.value.detail

    @pytest.mark.asyncio
    async def test_continue_chat_with_llm(
        self, mock_chat_manager, mock_chat_data, mock_query_result
    ):
        """Test continuing an existing chat"""
        # Mock data
        mock_chat_manager.get_chat.return_value = mock_chat_data

        # Mock the query function
        mock_query = MagicMock(return_value=mock_query_result)

        # Test with mocks
        with patch(
            "src.api.routers.chat_api.query_ollama_with_hybrid_search_multilingual", mock_query
        ), patch("src.api.routers.chat_api.chat_manager", mock_chat_manager), patch(
            "src.api.routers.chat_api.chat_sessions", {mock_chat_data["chat_id"]: {"messages": []}}
        ), patch(
            "src.api.routers.chat_api.uuid.uuid4", MagicMock(return_value="new-uuid")
        ), patch(
            "src.api.routers.chat_api.time.time", MagicMock(return_value=12345)
        ), patch(
            "src.api.routers.chat_api.SessionLocal", MagicMock()
        ):

            result = await continue_chat_with_llm(
                chat_id=mock_chat_data["chat_id"],
                message={"content": "Follow-up question", "model": "test-model"},
                x_session_id="test-session",
                user_email="test@example.com",
            )

        # Verify function calls
        mock_chat_manager.get_chat.assert_called_once_with(
            mock_chat_data["chat_id"], "test-session"
        )
        mock_query.assert_called_once()
        mock_chat_manager.save_chat.assert_called_once()

        # Verify result
        assert result["chat_id"] == mock_chat_data["chat_id"]
        assert result["dts"] == 12345
        assert len(result["messages"]) == 4  # Original 2 + new user + new assistant
        assert result["messages"][2]["role"] == "user"
        assert result["messages"][2]["content"] == "Follow-up question"
        assert result["messages"][3]["role"] == "assistant"
        assert result["messages"][3]["content"] == "This is a test response"

    @pytest.mark.asyncio
    async def test_continue_chat_not_found(self, mock_chat_manager):
        """Test continuing a non-existent chat"""
        # Mock data - chat not found
        mock_chat_manager.get_chat.return_value = None

        # Test with mocks
        with patch("src.api.routers.chat_api.chat_manager", mock_chat_manager):
            with pytest.raises(HTTPException) as excinfo:
                await continue_chat_with_llm(
                    chat_id="non-existent-id",
                    message={"content": "Test question"},
                    x_session_id="test-session",
                    user_email="test@example.com",
                )

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Chat not found" in excinfo.value.detail

    @pytest.mark.asyncio
    async def test_continue_chat_rebuild_session(
        self, mock_chat_manager, mock_chat_data, mock_query_result
    ):
        """Test continuing a chat when the session needs to be rebuilt"""
        # Mock data
        mock_chat_manager.get_chat.return_value = mock_chat_data

        # Mock the query function
        mock_query = MagicMock(return_value=mock_query_result)

        # Mock rebuild function
        mock_rebuild = MagicMock(return_value={"messages": []})

        # Test with mocks
        with patch(
            "src.api.routers.chat_api.query_ollama_with_hybrid_search_multilingual", mock_query
        ), patch("src.api.routers.chat_api.chat_manager", mock_chat_manager), patch(
            "src.api.routers.chat_api.chat_sessions", {}
        ), patch(
            "src.api.routers.chat_api.rebuild_chat_session", mock_rebuild
        ), patch(
            "src.api.routers.chat_api.uuid.uuid4", MagicMock(return_value="new-uuid")
        ), patch(
            "src.api.routers.chat_api.time.time", MagicMock(return_value=12345)
        ), patch(
            "src.api.routers.chat_api.SessionLocal", MagicMock()
        ):

            await continue_chat_with_llm(
                chat_id=mock_chat_data["chat_id"],
                message={"content": "Follow-up question", "model": "test-model"},
                x_session_id="test-session",
                user_email="test@example.com",
            )

        # Verify rebuild function was called
        mock_rebuild.assert_called_once_with(mock_chat_data["messages"])

    @pytest.mark.asyncio
    async def test_delete_chat_success(self, mock_chat_manager):
        """Test deleting a chat successfully"""
        # Mock data
        chat_id = "test-chat-id"
        mock_chat_manager.delete_chat.return_value = True

        # Test with mocks
        with patch("src.api.routers.chat_api.chat_manager", mock_chat_manager), patch(
            "src.api.routers.chat_api.chat_sessions", {chat_id: {"messages": []}}
        ):

            result = await delete_chat(
                chat_id=chat_id, x_session_id="test-session", user_email="test@example.com"
            )

        # Verify function calls
        mock_chat_manager.delete_chat.assert_called_once_with(chat_id, "test-session")

        # Verify result
        assert result["status"] == "success"
        assert "deleted successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_delete_chat_not_found(self, mock_chat_manager):
        """Test deleting a non-existent chat"""
        # Mock data - deletion fails
        mock_chat_manager.delete_chat.return_value = False

        # Test with mocks
        with patch("src.api.routers.chat_api.chat_manager", mock_chat_manager):
            with pytest.raises(HTTPException) as excinfo:
                await delete_chat(
                    chat_id="non-existent-id",
                    x_session_id="test-session",
                    user_email="test@example.com",
                )

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Chat not found or could not be deleted" in excinfo.value.detail

    @pytest.mark.asyncio
    async def test_process_query_new_chat(self, mock_chat_manager, mock_query_result):
        """Test processing a query for a new chat"""
        # Mock the query function
        mock_query = MagicMock(return_value=mock_query_result)

        # Create request
        request = QueryRequest(
            chat_id="", question="Test question", model_name="test-model", session_id="test-session"
        )

        # Test with mocks
        with patch(
            "src.api.routers.chat_api.query_ollama_with_hybrid_search_multilingual", mock_query
        ), patch("src.api.routers.chat_api.chat_manager", mock_chat_manager), patch(
            "src.api.routers.chat_api.uuid.uuid4", MagicMock(return_value="test-uuid")
        ), patch(
            "src.api.routers.chat_api.time.time", MagicMock(return_value=12345)
        ), patch(
            "src.api.routers.chat_api.SessionLocal", MagicMock()
        ), patch(
            "src.api.routers.chat_api.create_chat_session", MagicMock(return_value={"messages": []})
        ):

            result = await process_query(request=request, user_email="test@example.com")

        # Verify result is a new chat
        assert result["chat_id"] == "test-uuid"
        assert "title" in result
        assert "messages" in result
        assert len(result["messages"]) == 2

    @pytest.mark.asyncio
    async def test_process_query_existing_chat(
        self, mock_chat_manager, mock_chat_data, mock_query_result
    ):
        """Test processing a query for an existing chat"""
        # Mock data
        mock_chat_manager.get_chat.return_value = mock_chat_data

        # Mock the query function
        mock_query = MagicMock(return_value=mock_query_result)

        # Create request
        request = QueryRequest(
            chat_id=mock_chat_data["chat_id"],
            question="Follow-up question",
            model_name="test-model",
            session_id="test-session",
        )

        # Test with mocks
        with patch(
            "src.api.routers.chat_api.query_ollama_with_hybrid_search_multilingual", mock_query
        ), patch("src.api.routers.chat_api.chat_manager", mock_chat_manager), patch(
            "src.api.routers.chat_api.chat_sessions", {}
        ), patch(
            "src.api.routers.chat_api.create_chat_session", MagicMock(return_value={"messages": []})
        ), patch(
            "src.api.routers.chat_api.uuid.uuid4", MagicMock(return_value="new-uuid")
        ), patch(
            "src.api.routers.chat_api.time.time", MagicMock(return_value=12345)
        ), patch(
            "src.api.routers.chat_api.SessionLocal", MagicMock()
        ):

            result = await process_query(request=request, user_email="test@example.com")

        # Verify result is the continued chat
        assert result["chat_id"] == mock_chat_data["chat_id"]
        assert "messages" in result
        assert len(result["messages"]) == 4  # Original 2 + new user + new assistant

    @pytest.mark.asyncio
    async def test_process_query_missing_question(self):
        """Test processing a query with missing question"""
        # Create request with empty question
        request = QueryRequest(
            chat_id="", question="", model_name="test-model", session_id="test-session"
        )

        # Test with empty question
        with pytest.raises(HTTPException) as excinfo:
            await process_query(request=request, user_email="test@example.com")

        # Verify exception
        assert excinfo.value.status_code == 400
        assert "Question is required" in excinfo.value.detail

    @pytest.mark.asyncio
    async def test_process_query_missing_session_id(self):
        """Test processing a query with missing session ID"""
        # Create request with empty session ID
        request = QueryRequest(
            chat_id="", question="Test question", model_name="test-model", session_id=""
        )

        # Test with empty session ID
        with pytest.raises(HTTPException) as excinfo:
            await process_query(request=request, user_email="test@example.com")

        # Verify exception
        assert excinfo.value.status_code == 400
        assert "Session ID is required" in excinfo.value.detail


if __name__ == "__main__":
    pytest.main(["-xvs", "tests/test_chat_api.py"])
