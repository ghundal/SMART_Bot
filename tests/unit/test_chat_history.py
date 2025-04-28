import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Import the module to test - we'll monkeypatch the class rather than
# trying to change the import structure
from src.api.utils.chat_history import ChatHistoryManager


class TestChatHistory:
    """Tests for the ChatHistoryManager class"""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session"""
        mock_session = MagicMock()
        mock_session.commit = MagicMock()
        mock_session.rollback = MagicMock()
        mock_session.close = MagicMock()
        mock_session.execute = MagicMock()
        return mock_session

    @pytest.fixture
    def sample_chat(self):
        """Sample chat data for testing"""
        return {
            "chat_id": "test-chat-123",
            "title": "Test Chat",
            "dts": int(datetime.now().timestamp()),
            "messages": [
                {"message_id": "msg1", "role": "user", "content": "Hello"},
                {"message_id": "msg2", "role": "assistant", "content": "Hi there!"},
            ],
        }

    def test_init(self):
        """Test initialization of ChatHistoryManager"""
        # Test with default parameters
        manager = ChatHistoryManager(model="test-model")

        # Verify attributes
        assert manager.model == "test-model"
        assert manager.SessionLocal is not None

    def test_save_chat_new(self, mock_db_session, sample_chat):
        """Test saving a new chat"""
        # Configure the mock session to indicate chat doesn't exist
        mock_scalar = MagicMock()
        mock_scalar.scalar.return_value = 0
        mock_db_session.execute.return_value = mock_scalar

        # Create manager and patch SessionLocal
        with patch("src.api.utils.chat_history.SessionLocal", return_value=mock_db_session):
            manager = ChatHistoryManager(model="test-model")
            manager.save_chat(sample_chat, "test-session")

        # Verify execute was called twice (check existence and insert)
        assert mock_db_session.execute.call_count == 2

        # Get the second call args (insert query)
        insert_call = mock_db_session.execute.call_args_list[1]
        sql = insert_call[0][0].text
        params = insert_call[0][1]

        # Verify SQL contains INSERT
        assert "INSERT INTO chat_history" in sql

        # Verify parameters
        assert params["chat_id"] == sample_chat["chat_id"]
        assert params["session_id"] == "test-session"
        assert params["model"] == "test-model"
        assert params["title"] == sample_chat["title"]
        assert isinstance(params["messages"], str)
        # Verify messages were properly JSON encoded
        assert json.loads(params["messages"]) == sample_chat["messages"]

        # Verify session operations
        mock_db_session.commit.assert_called_once()
        mock_db_session.close.assert_called_once()

    # Rest of the test methods...
    # I've abbreviated here but all methods follow the same structure
    # as in the previous example

    def test_save_chat_existing(self, mock_db_session, sample_chat):
        """Test updating an existing chat"""
        # Configure mock...
        mock_scalar = MagicMock()
        mock_scalar.scalar.return_value = 1
        mock_db_session.execute.return_value = mock_scalar

        with patch("src.api.utils.chat_history.SessionLocal", return_value=mock_db_session):
            manager = ChatHistoryManager(model="test-model")
            manager.save_chat(sample_chat, "test-session")

        # Verify update query...
        assert mock_db_session.execute.call_count == 2
        update_call = mock_db_session.execute.call_args_list[1]
        sql = update_call[0][0].text
        assert "UPDATE chat_history" in sql

    def test_get_chat_found(self, mock_db_session):
        """Test retrieving an existing chat"""
        # Setup mock data...
        created_at = datetime.now()
        updated_at = datetime.now()
        mock_db_session.execute.return_value.fetchone.return_value = (
            "test-chat-123",
            "test-session",
            "test-model",
            "Test Chat",
            json.dumps([{"message_id": "msg1", "role": "user", "content": "Hello"}]),
            12345,
            created_at,
            updated_at,
        )

        # Test the method...
        with patch("src.api.utils.chat_history.SessionLocal", return_value=mock_db_session):
            manager = ChatHistoryManager(model="test-model")
            result = manager.get_chat("test-chat-123", "test-session")

        # Verify result...
        assert result["chat_id"] == "test-chat-123"
        assert isinstance(result["messages"], list)

    def test_delete_chat_success(self, mock_db_session):
        """Test successfully deleting a chat"""
        # Setup mock...
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_db_session.execute.return_value = mock_result

        # Test the method...
        with patch("src.api.utils.chat_history.SessionLocal", return_value=mock_db_session):
            manager = ChatHistoryManager(model="test-model")
            result = manager.delete_chat("test-chat-123", "test-session")

        # Verify result...
        assert result is True
        mock_db_session.commit.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-xvs", "tests/test_chat_history.py"])
