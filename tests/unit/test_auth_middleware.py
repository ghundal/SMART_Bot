"""
Unit tests for the auth_middleware.py module.

Tests the authentication middleware functions including:
- Database session dependency
- Token extraction from cookies and headers
- Token verification logic
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi import Cookie, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from sqlalchemy.sql import text


class TestAuthMiddleware(unittest.TestCase):
    def setUp(self):
        # Create mock objects
        self.mock_db = MagicMock(spec=Session)
        self.mock_session_local = MagicMock(return_value=self.mock_db)

        # Mock the database execute chain
        self.mock_execute = MagicMock()
        self.mock_db.execute.return_value = self.mock_execute
        self.mock_execute.fetchone = MagicMock()

        # Create mock request
        self.mock_request = MagicMock(spec=Request)

        # Mock utils.database module
        self.mock_utils_module = MagicMock()
        self.mock_utils_module.database = MagicMock()
        self.mock_utils_module.database.SessionLocal = self.mock_session_local

        # Add mock modules to sys.modules
        sys.modules['utils'] = self.mock_utils_module
        sys.modules['utils.database'] = self.mock_utils_module.database

        # Mock sqlalchemy text function
        self.mock_text = MagicMock()
        self.mock_text.side_effect = lambda query: query  # Just return the query string
        self.mock_text_patcher = patch('sqlalchemy.sql.text', self.mock_text)
        self.mock_text_patcher.start()

        # Import auth_middleware
        self.auth_middleware = self.import_auth_middleware()

    def tearDown(self):
        # Stop all patches
        self.mock_text_patcher.stop()

        # Remove mock modules
        if 'utils' in sys.modules:
            del sys.modules['utils']
        if 'utils.database' in sys.modules:
            del sys.modules['utils.database']

    def import_auth_middleware(self):
        """Import the auth_middleware module directly from file"""
        # Add src to the Python path
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

        # Import the module directly using its file path
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "auth_middleware",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/api/routers/auth_middleware.py"))
        )
        auth_middleware = importlib.util.module_from_spec(spec)

        # Execute the module
        spec.loader.exec_module(auth_middleware)

        # Replace dependencies with mocks
        auth_middleware.SessionLocal = self.mock_session_local

        return auth_middleware

    def test_get_db(self):
        """Test the get_db dependency function"""
        # Create a generator from the get_db function
        db_generator = self.auth_middleware.get_db()

        # Get the database session from the generator
        db = next(db_generator)

        # Verify the correct db session was returned
        assert db == self.mock_db

        # Verify close is called when generator is exhausted
        try:
            next(db_generator)  # This should raise StopIteration
        except StopIteration:
            pass

        self.mock_db.close.assert_called_once()

    async def test_get_token_from_request_cookie(self):
        """Test extracting token from cookie"""
        # Set up the test with a cookie
        token = "cookie_token_value"

        # Call the function with a cookie
        result = await self.auth_middleware.get_token_from_request(
            self.mock_request,
            access_token=token,
            token_header=None
        )

        # Verify the token from cookie is returned
        assert result == token

    async def test_get_token_from_request_header(self):
        """Test extracting token from header"""
        # Set up the test with a header but no cookie
        token = "header_token_value"

        # Call the function with a header
        result = await self.auth_middleware.get_token_from_request(
            self.mock_request,
            access_token=None,
            token_header=token
        )

        # Verify the token from header is returned
        assert result == token

    async def test_get_token_from_request_cookie_priority(self):
        """Test that cookie takes priority over header"""
        # Set up the test with both cookie and header
        cookie_token = "cookie_token_value"
        header_token = "header_token_value"

        # Call the function with both
        result = await self.auth_middleware.get_token_from_request(
            self.mock_request,
            access_token=cookie_token,
            token_header=header_token
        )

        # Verify the token from cookie is returned (priority)
        assert result == cookie_token

    async def test_get_token_from_request_none(self):
        """Test behavior when no token is provided"""
        # Call the function with no tokens
        result = await self.auth_middleware.get_token_from_request(
            self.mock_request,
            access_token=None,
            token_header=None
        )

        # Verify that None is returned
        assert result is None

    async def test_verify_token_valid(self):
        """Test token verification with a valid token"""
        # Set up the test with a valid token
        token = "valid_token"
        expected_email = "user@example.com"

        # Configure mock to return a valid result
        self.mock_execute.fetchone.return_value = [expected_email]

        # Call the verify_token function
        result = await self.auth_middleware.verify_token(token, self.mock_db)

        # Verify the database was queried correctly
        self.mock_db.execute.assert_called_once()

        # Verify the correct email was returned
        assert result == expected_email

    async def test_verify_token_missing(self):
        """Test token verification with a missing token"""
        # Call the verify_token function with no token
        with pytest.raises(HTTPException) as exc_info:
            await self.auth_middleware.verify_token(None, self.mock_db)

        # Verify the correct error was raised
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Not authenticated" in exc_info.value.detail

    async def test_verify_token_invalid(self):
        """Test token verification with an invalid token"""
        # Set up the test with an invalid token
        token = "invalid_token"

        # Configure mock to return no result (token not found)
        self.mock_execute.fetchone.return_value = None

        # Call the verify_token function
        with pytest.raises(HTTPException) as exc_info:
            await self.auth_middleware.verify_token(token, self.mock_db)

        # Verify the correct error was raised
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid authentication credentials" in exc_info.value.detail

    async def test_verify_token_db_error(self):
        """Test token verification with a database error"""
        # Set up the test with a token
        token = "error_token"

        # Configure mock to raise an exception
        self.mock_db.execute.side_effect = Exception("Database error")

        # Call the verify_token function
        with pytest.raises(HTTPException) as exc_info:
            await self.auth_middleware.verify_token(token, self.mock_db)

        # Verify the correct error was raised
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Authentication error" in exc_info.value.detail
        assert "Database error" in exc_info.value.detail


if __name__ == '__main__':
    unittest.main()
