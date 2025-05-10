"""
Unit tests for the llm_rag_utils.py module.

Tests the chat session management functionality.
"""

import unittest

# Import the actual module functions to test
from api.utils.llm_rag_utils import create_chat_session, rebuild_chat_session, chat_sessions

# Mock data for testing
MOCK_CHAT_HISTORY = [
    {"role": "user", "content": "What is RAG?", "message_id": "msg1"},
    {
        "role": "assistant",
        "content": "RAG stands for Retrieval Augmented Generation.",
        "message_id": "msg2",
    },
    {"role": "user", "content": "How does it work?", "message_id": "msg3"},
]


class TestLLMRAGUtils(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        # Clear the global chat sessions dictionary before each test
        chat_sessions.clear()

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

        # Check that created_at and last_updated are None as specified in the implementation
        self.assertIsNone(result["metadata"]["created_at"])
        self.assertIsNone(result["metadata"]["last_updated"])

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

    def test_rebuild_chat_session_with_missing_fields(self):
        """Test rebuilding a chat session handles missing required fields"""
        # Create history with missing fields
        history_with_missing = [
            {"role": "user", "content": "Valid message"},  # Valid
            {"role": "assistant"},  # Missing content
            {"content": "Missing role"},  # Missing role
            {},  # Empty dict
        ]

        # Call the function
        result = rebuild_chat_session(history_with_missing)

        # Only one valid message should be copied
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0]["role"], "user")
        self.assertEqual(result["messages"][0]["content"], "Valid message")

    def test_global_chat_sessions_dictionary(self):
        """Test that the global chat_sessions dictionary works properly"""
        # Check that it starts empty
        self.assertEqual(len(chat_sessions), 0)

        # Add a session
        chat_id = "test_session_1"
        chat_sessions[chat_id] = create_chat_session()

        # Verify it was added
        self.assertEqual(len(chat_sessions), 1)
        self.assertIn(chat_id, chat_sessions)
        self.assertEqual(len(chat_sessions[chat_id]["messages"]), 0)


if __name__ == "__main__":
    unittest.main()
