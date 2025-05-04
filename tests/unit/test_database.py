"""
Unit tests for the database.py module.

Tests the PostgreSQL connection and audit logging functionality.
"""

import unittest
import os
from unittest.mock import MagicMock, patch, ANY
import numpy as np
import sys

# Create mock modules
sys.modules['sqlalchemy'] = MagicMock()
sys.modules['sqlalchemy.orm'] = MagicMock()
sys.modules['rag_pipeline'] = MagicMock()
sys.modules['rag_pipeline.config'] = MagicMock()

# Mock the logger
mock_logger = MagicMock()
sys.modules['rag_pipeline.config'].logger = mock_logger

# Import after mocking
from api.utils.database import connect_to_postgres, log_audit, SessionLocal


class TestDatabase(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        # Create mock objects
        self.mock_session = MagicMock()
        self.mock_engine = MagicMock()
        self.mock_text = MagicMock()

        # Set up environment variables for testing
        self.env_vars = {
            "DB_HOST": "test-host",
            "DB_PORT": "5432",
            "DB_NAME": "test-db",
            "DB_USER": "test-user",
            "DB_PASSWORD": "test-password"
        }

        # Save original environment variables
        self.original_env = {}
        for key in self.env_vars:
            self.original_env[key] = os.environ.get(key)
            os.environ[key] = self.env_vars[key]

    def tearDown(self):
        """Clean up after each test"""
        # Restore original environment variables
        for key in self.env_vars:
            if self.original_env[key] is None:
                del os.environ[key]
            else:
                os.environ[key] = self.original_env[key]

    @patch('api.utils.database.create_engine')
    def test_connect_to_postgres(self, mock_create_engine):
        """Test the database connection function with environment variables"""
        # Set up mock
        mock_create_engine.return_value = self.mock_engine

        # Call the function
        result = connect_to_postgres()

        # Check that create_engine was called with the correct connection string
        expected_conn_string = (
            f"postgresql://{self.env_vars['DB_USER']}:{self.env_vars['DB_PASSWORD']}"
            f"@{self.env_vars['DB_HOST']}:{self.env_vars['DB_PORT']}/{self.env_vars['DB_NAME']}"
        )
        mock_create_engine.assert_called_once_with(expected_conn_string)

        # Check the result
        self.assertEqual(result, self.mock_engine)

    @patch('api.utils.database.create_engine')
    def test_connect_to_postgres_default_values(self, mock_create_engine):
        """Test the database connection function with default values"""
        # Save current environment variables
        temp_env = {}
        for key in self.env_vars:
            temp_env[key] = os.environ.get(key)
            if key in os.environ:
                del os.environ[key]

        try:
            # Set up mock
            mock_create_engine.return_value = self.mock_engine

            # Call the function
            result = connect_to_postgres()

            # Check that create_engine was called with the default connection string
            expected_conn_string = "postgresql://postgres:postgres@localhost:5432/smart"
            mock_create_engine.assert_called_once_with(expected_conn_string)

            # Check the result
            self.assertEqual(result, self.mock_engine)
        finally:
            # Restore environment variables
            for key, value in temp_env.items():
                if value is not None:
                    os.environ[key] = value

    @patch('api.utils.database.text')
    def test_log_audit(self, mock_text):
        """Test the audit logging function"""
        # Set up mocks
        mock_text.return_value = self.mock_text

        # Mock the execute method to properly handle the params
        def side_effect(text_obj, params=None):
            self.execute_params = params
            return MagicMock()

        self.mock_session.execute.side_effect = side_effect

        # Sample test data
        user_email = "test@example.com"
        query = "test query"
        query_embedding = np.array([0.1, 0.2, 0.3])
        chunks = [
            {"document_id": 1, "chunk_text": "chunk 1"},
            {"document_id": 2, "chunk_text": "chunk 2"}
        ]
        response = "test response"
        detected_language = "fr"

        # Call the function
        log_audit(
            self.mock_session,
            user_email,
            query,
            query_embedding,
            chunks,
            response,
            detected_language
        )

        # Verify the SQL contains expected elements
        mock_text.assert_called_once()
        sql_arg = mock_text.call_args[0][0]
        self.assertIn("INSERT INTO audit", sql_arg)
        self.assertIn("language_code", sql_arg)

        # Verify execute was called
        self.mock_session.execute.assert_called_once()

        # Check the parameters dictionary
        self.assertEqual(self.execute_params.get("user_email"), user_email)
        self.assertEqual(self.execute_params.get("query"), query)
        self.assertEqual(self.execute_params.get("document_ids"), [1, 2])
        self.assertEqual(self.execute_params.get("chunk_texts"), ["chunk 1", "chunk 2"])
        self.assertEqual(self.execute_params.get("response"), response)
        self.assertEqual(self.execute_params.get("language_code"), detected_language)

        # Verify commit was called
        self.mock_session.commit.assert_called_once()

        # Verify logger.info was called
        mock_logger.info.assert_called_once()

    @patch('api.utils.database.text')
    def test_log_audit_with_default_language(self, mock_text):
        """Test the audit logging function with default language"""
        # Set up mocks
        mock_text.return_value = self.mock_text

        # Mock the execute method to properly handle the params
        def side_effect(text_obj, params=None):
            self.execute_params = params
            return MagicMock()

        self.mock_session.execute.side_effect = side_effect

        # Sample test data with no language specified
        user_email = "test@example.com"
        query = "test query"
        query_embedding = np.array([0.1, 0.2, 0.3])
        chunks = [{"document_id": 1, "chunk_text": "chunk 1"}]
        response = "test response"

        # Call the function without language
        log_audit(
            self.mock_session,
            user_email,
            query,
            query_embedding,
            chunks,
            response
        )

        # Check the language_code is set to default "en"
        self.assertEqual(self.execute_params.get("language_code"), "en")

    @patch('api.utils.database.text')
    def test_log_audit_with_empty_chunks(self, mock_text):
        """Test the audit logging function with empty chunks"""
        # Set up mocks
        mock_text.return_value = self.mock_text

        # Mock the execute method to properly handle the params
        def side_effect(text_obj, params=None):
            self.execute_params = params
            return MagicMock()

        self.mock_session.execute.side_effect = side_effect

        # Sample test data with empty chunks
        user_email = "test@example.com"
        query = "test query"
        query_embedding = np.array([0.1, 0.2, 0.3])
        chunks = []
        response = "test response"

        # Call the function
        log_audit(
            self.mock_session,
            user_email,
            query,
            query_embedding,
            chunks,
            response
        )

        # Check the document_ids and chunk_texts are empty lists
        self.assertEqual(self.execute_params.get("document_ids"), [])
        self.assertEqual(self.execute_params.get("chunk_texts"), [])

    @patch('api.utils.database.text')
    def test_log_audit_with_none_chunks(self, mock_text):
        """Test the audit logging function with None chunks"""
        # Set up mocks
        mock_text.return_value = self.mock_text

        # Mock the execute method to properly handle the params
        def side_effect(text_obj, params=None):
            self.execute_params = params
            return MagicMock()

        self.mock_session.execute.side_effect = side_effect

        # Sample test data with None chunks
        user_email = "test@example.com"
        query = "test query"
        query_embedding = np.array([0.1, 0.2, 0.3])
        chunks = None
        response = "test response"

        # Call the function
        log_audit(
            self.mock_session,
            user_email,
            query,
            query_embedding,
            chunks,
            response
        )

        # Check the document_ids and chunk_texts are empty lists
        self.assertEqual(self.execute_params.get("document_ids"), [])
        self.assertEqual(self.execute_params.get("chunk_texts"), [])

    @patch('api.utils.database.text')
    def test_log_audit_with_list_embedding(self, mock_text):
        """Test the audit logging function with a list embedding instead of numpy array"""
        # Set up mocks
        mock_text.return_value = self.mock_text

        # Mock the execute method to properly handle the params
        def side_effect(text_obj, params=None):
            self.execute_params = params
            return MagicMock()

        self.mock_session.execute.side_effect = side_effect

        # Sample test data with list embedding
        user_email = "test@example.com"
        query = "test query"
        query_embedding = [0.1, 0.2, 0.3]  # List instead of numpy array
        chunks = [{"document_id": 1, "chunk_text": "chunk 1"}]
        response = "test response"

        # Call the function
        log_audit(
            self.mock_session,
            user_email,
            query,
            query_embedding,
            chunks,
            response
        )

        # Verify SQL contains the embedding
        mock_text.assert_called_once()
        sql_arg = mock_text.call_args[0][0]
        self.assertIn(str(query_embedding), sql_arg)

    @patch('api.utils.database.text')
    def test_log_audit_exception(self, mock_text):
        """Test exception handling in the audit logging function"""
        # Set up mocks
        mock_text.return_value = self.mock_text
        self.mock_session.execute.side_effect = Exception("Database error")

        # Sample test data
        user_email = "test@example.com"
        query = "test query"
        query_embedding = np.array([0.1, 0.2, 0.3])
        chunks = [{"document_id": 1, "chunk_text": "chunk 1"}]
        response = "test response"

        # Call the function (should not raise exception)
        log_audit(
            self.mock_session,
            user_email,
            query,
            query_embedding,
            chunks,
            response
        )

        # Verify logger.exception was called
        mock_logger.exception.assert_called_once()

        # Verify commit was not called after exception
        self.mock_session.commit.assert_not_called()

    def test_session_local(self):
        """Test that SessionLocal is properly created"""
        # We can't test much about the actual SessionLocal without setting up
        # a real database, but we can verify it exists and has the correct type
        self.assertTrue(callable(SessionLocal))


if __name__ == "__main__":
    unittest.main()
