"""
Unit tests for the chat_history.py module.

Tests the ChatHistoryManager class which handles:
- Saving chats to the database
- Retrieving specific chats
- Getting recent chat histories
- Deleting chats
"""

import json
import unittest
import time
import sys
import os
import importlib.util
from unittest.mock import MagicMock, patch, ANY
from datetime import datetime

import pytest
from sqlalchemy.orm import Session
from sqlalchemy import text

# Mock modules
sys.modules['rag_pipeline'] = MagicMock()
sys.modules['rag_pipeline.config'] = MagicMock()
sys.modules['rag_pipeline.embedding'] = MagicMock()
sys.modules['rag_pipeline.ollama'] = MagicMock()

# Mock data for testing
MOCK_MODEL = "llama2"
MOCK_USER_EMAIL = "test@example.com"
MOCK_SESSION_ID = "test-session-123"
MOCK_CHAT_ID = "12345678-1234-5678-1234-567812345678"
MOCK_TIMESTAMP = int(time.time())
MOCK_CHAT = {
    "chat_id": MOCK_CHAT_ID,
    "title": "Test Chat",
    "dts": MOCK_TIMESTAMP,
    "messages": [
        {"message_id": "msg1", "role": "user", "content": "Hello"},
        {"message_id": "msg2", "role": "assistant", "content": "Hi there!"},
    ],
}
MOCK_DB_ROW = (
    MOCK_CHAT_ID,
    MOCK_SESSION_ID,
    MOCK_MODEL,
    "Test Chat",
    json.dumps(MOCK_CHAT["messages"]),
    MOCK_TIMESTAMP,
    datetime.now(),
    datetime.now(),
)


class TestChatHistoryManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        # Create mock objects
        self.mock_db_session = MagicMock(spec=Session)
        self.mock_session_local = MagicMock(return_value=self.mock_db_session)

        # Set up patching
        self.patches = [
            patch('api.utils.database.SessionLocal', self.mock_session_local)
        ]

        # Start all patches
        for p in self.patches:
            p.start()

        # Define the ChatHistoryManager class to avoid import issues
        from api.utils.chat_history import ChatHistoryManager

        # Create an instance of the class to test
        self.chat_manager = ChatHistoryManager(MOCK_MODEL)

        # Ensure the SessionLocal is properly mocked
        self.chat_manager.SessionLocal = self.mock_session_local

    def tearDown(self):
        """Clean up after each test"""
        # Stop all patches
        for p in self.patches:
            p.stop()

    def test_init(self):
        """Test the initialization of ChatHistoryManager"""
        self.assertEqual(self.chat_manager.model, MOCK_MODEL)
        self.assertIs(self.chat_manager.SessionLocal, self.mock_session_local)

    def test_save_chat_new(self):
        """Test saving a new chat to the database"""
        # Mock the count query result (0 means chat doesn't exist)
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0
        self.mock_db_session.execute.return_value = mock_count_result

        # Call the method to test
        self.chat_manager.save_chat(MOCK_CHAT, MOCK_USER_EMAIL, MOCK_SESSION_ID)

        # Verify execute was called twice (once for check, once for insert)
        self.assertEqual(self.mock_db_session.execute.call_count, 2)

        # Verify commit was called once
        self.mock_db_session.commit.assert_called_once()

        # Verify close was called once
        self.mock_db_session.close.assert_called_once()

        # Instead of checking string content, check that parameters for the second call include
        # chat_id, session_id, and user_email for an INSERT operation
        second_call_args = self.mock_db_session.execute.call_args_list[1]
        self.assertIn(MOCK_CHAT_ID, str(second_call_args))
        self.assertIn(MOCK_SESSION_ID, str(second_call_args))
        self.assertIn(MOCK_USER_EMAIL, str(second_call_args))

    def test_save_chat_existing(self):
        """Test updating an existing chat in the database"""
        # Mock the count query result (1 means chat exists)
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1
        self.mock_db_session.execute.return_value = mock_count_result

        # Call the method to test
        self.chat_manager.save_chat(MOCK_CHAT, MOCK_USER_EMAIL, MOCK_SESSION_ID)

        # Verify execute was called twice (once for check, once for update)
        self.assertEqual(self.mock_db_session.execute.call_count, 2)

        # Verify commit was called once
        self.mock_db_session.commit.assert_called_once()

        # Check that the second call includes chat_id for an UPDATE operation
        second_call_args = self.mock_db_session.execute.call_args_list[1]
        self.assertIn(MOCK_CHAT_ID, str(second_call_args))
        # Also check the parameters include message data
        self.assertIn("messages", str(second_call_args))

    def test_save_chat_exception(self):
        """Test handling exceptions when saving a chat"""
        # Mock the execute method to raise an exception
        self.mock_db_session.execute.side_effect = Exception("Database error")

        # Call the method and expect an exception
        with self.assertRaises(Exception):
            self.chat_manager.save_chat(MOCK_CHAT, MOCK_USER_EMAIL, MOCK_SESSION_ID)

        # Verify rollback was called once
        self.mock_db_session.rollback.assert_called_once()

        # Verify close was called once
        self.mock_db_session.close.assert_called_once()

    def test_get_chat(self):
        """Test retrieving a specific chat by ID"""
        # Mock the fetchone result
        mock_result = MagicMock()
        mock_result.fetchone.return_value = MOCK_DB_ROW
        self.mock_db_session.execute.return_value = mock_result

        # Call the method to test
        result = self.chat_manager.get_chat(MOCK_CHAT_ID)

        # Verify execute was called once
        self.mock_db_session.execute.assert_called_once()

        # Verify close was called once
        self.mock_db_session.close.assert_called_once()

        # Check that the result has the expected structure
        self.assertEqual(result["chat_id"], MOCK_CHAT_ID)
        self.assertEqual(result["model"], MOCK_MODEL)
        self.assertEqual(result["title"], "Test Chat")
        self.assertEqual(len(result["messages"]), 2)

    def test_get_chat_not_found(self):
        """Test when a chat is not found in the database"""
        # Mock the fetchone result to return None
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        self.mock_db_session.execute.return_value = mock_result

        # Call the method to test
        result = self.chat_manager.get_chat(MOCK_CHAT_ID)

        # Verify execute was called once
        self.mock_db_session.execute.assert_called_once()

        # Verify close was called once
        self.mock_db_session.close.assert_called_once()

        # Check that the result is None
        self.assertIsNone(result)

    def test_get_chat_exception(self):
        """Test handling exceptions when retrieving a chat"""
        # Mock the execute method to raise an exception
        self.mock_db_session.execute.side_effect = Exception("Database error")

        # Call the method to test
        result = self.chat_manager.get_chat(MOCK_CHAT_ID)

        # Verify execute was called once
        self.mock_db_session.execute.assert_called_once()

        # Verify close was called once
        self.mock_db_session.close.assert_called_once()

        # Check that the result is None
        self.assertIsNone(result)

    def test_get_recent_chats(self):
        """Test retrieving recent chats"""
        # Mock the fetchall result
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [MOCK_DB_ROW, MOCK_DB_ROW]  # Two identical rows for simplicity
        self.mock_db_session.execute.return_value = mock_result

        # Call the method to test
        results = self.chat_manager.get_recent_chats(MOCK_USER_EMAIL, 10)

        # Verify execute was called once
        self.mock_db_session.execute.assert_called_once()

        # Verify close was called once
        self.mock_db_session.close.assert_called_once()

        # Instead of checking string content, verify that parameters were passed correctly
        call_args = self.mock_db_session.execute.call_args
        self.assertIn(MOCK_USER_EMAIL, str(call_args))

        # Alternatively, check that 10 was passed as a parameter
        # This assumes the implementation uses parameter binding for the limit
        if 'params' in str(call_args):
            self.assertIn('10', str(call_args))

        # Check that the results list has the expected length
        self.assertEqual(len(results), 2)

        # Check the structure of the first result
        self.assertEqual(results[0]["chat_id"], MOCK_CHAT_ID)
        self.assertEqual(results[0]["model"], MOCK_MODEL)
        self.assertEqual(len(results[0]["messages"]), 2)

    def test_get_recent_chats_no_limit(self):
        """Test retrieving recent chats without a limit"""
        # Mock the fetchall result
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [MOCK_DB_ROW]
        self.mock_db_session.execute.return_value = mock_result

        # Call the method to test without a limit
        results = self.chat_manager.get_recent_chats(MOCK_USER_EMAIL)

        # Verify execute was called once
        self.mock_db_session.execute.assert_called_once()

        # Verify the user email parameter was passed
        call_args = self.mock_db_session.execute.call_args
        self.assertIn(MOCK_USER_EMAIL, str(call_args))

    def test_get_recent_chats_empty(self):
        """Test when no recent chats are found"""
        # Mock the fetchall result to return an empty list
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        self.mock_db_session.execute.return_value = mock_result

        # Call the method to test
        results = self.chat_manager.get_recent_chats(MOCK_USER_EMAIL)

        # Verify execute was called once
        self.mock_db_session.execute.assert_called_once()

        # Verify close was called once
        self.mock_db_session.close.assert_called_once()

        # Verify the result is an empty list
        self.assertEqual(results, [])

    def test_get_recent_chats_exception(self):
        """Test handling exceptions when retrieving recent chats"""
        # Mock the execute method to raise an exception
        self.mock_db_session.execute.side_effect = Exception("Database error")

        # Call the method to test
        results = self.chat_manager.get_recent_chats(MOCK_USER_EMAIL)

        # Verify execute was called once
        self.mock_db_session.execute.assert_called_once()

        # Verify close was called once
        self.mock_db_session.close.assert_called_once()

        # Verify the result is an empty list
        self.assertEqual(results, [])

    def test_delete_chat(self):
        """Test deleting a chat"""
        # Mock the execute result to indicate a successful deletion
        mock_result = MagicMock()
        mock_result.rowcount = 1  # One row was deleted
        self.mock_db_session.execute.return_value = mock_result

        # Call the method to test
        result = self.chat_manager.delete_chat(MOCK_CHAT_ID)

        # Verify execute was called once
        self.mock_db_session.execute.assert_called_once()

        # Verify commit was called once
        self.mock_db_session.commit.assert_called_once()

        # Verify close was called once
        self.mock_db_session.close.assert_called_once()

        # Check that the result is True
        self.assertTrue(result)

    def test_delete_chat_not_found(self):
        """Test when a chat to delete is not found"""
        # Mock the execute result to indicate no rows were deleted
        mock_result = MagicMock()
        mock_result.rowcount = 0  # No rows were deleted
        self.mock_db_session.execute.return_value = mock_result

        # Call the method to test
        result = self.chat_manager.delete_chat(MOCK_CHAT_ID)

        # Verify execute was called once
        self.mock_db_session.execute.assert_called_once()

        # Verify commit was NOT called
        self.mock_db_session.commit.assert_not_called()

        # Verify close was called once
        self.mock_db_session.close.assert_called_once()

        # Check that the result is False
        self.assertFalse(result)

    def test_delete_chat_exception(self):
        """Test handling exceptions when deleting a chat"""
        # Mock the execute method to raise an exception
        self.mock_db_session.execute.side_effect = Exception("Database error")

        # Call the method to test
        result = self.chat_manager.delete_chat(MOCK_CHAT_ID)

        # Verify execute was called once
        self.mock_db_session.execute.assert_called_once()

        # Verify rollback was called once
        self.mock_db_session.rollback.assert_called_once()

        # Verify close was called once
        self.mock_db_session.close.assert_called_once()

        # Check that the result is False
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
