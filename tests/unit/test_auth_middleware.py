from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

# Import the functions to test
from src.api.routers.auth_middleware import get_db, get_token_from_request, verify_token


class TestAuthMiddleware:
    """Tests for the auth_middleware.py module"""

    @pytest.fixture
    def mock_db(self):
        """Fixture for mock database session"""
        mock_db = MagicMock(spec=Session)
        mock_db.close = MagicMock()
        return mock_db

    @pytest.fixture
    def mock_request(self):
        """Fixture for mock request object"""
        mock_request = MagicMock(spec=Request)
        return mock_request

    def test_get_db(self, mock_db):
        """Test the get_db function yields a session and closes it afterward"""
        # Patch the SessionLocal to return our mock db
        with patch("src.api.routers.auth_middleware.SessionLocal", return_value=mock_db):
            # Get the generator
            db_gen = get_db()

            # Get the yielded session
            db = next(db_gen)

            # Verify it's our mock
            assert db == mock_db

            # Verify that close is not called yet
            mock_db.close.assert_not_called()

            # Trigger the finally block
            try:
                next(db_gen)
            except StopIteration:
                pass

            # Verify close was called
            mock_db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_token_from_cookie(self, mock_request):
        """Test extracting token from cookies"""
        # Test with cookie present
        token = await get_token_from_request(
            request=mock_request, access_token="cookie-token", token_header=None
        )

        assert token == "cookie-token"

    @pytest.mark.asyncio
    async def test_get_token_from_header(self, mock_request):
        """Test extracting token from Authorization header"""
        # Test with only header token present
        token = await get_token_from_request(
            request=mock_request, access_token=None, token_header="header-token"
        )

        assert token == "header-token"

    @pytest.mark.asyncio
    async def test_get_token_cookie_priority(self, mock_request):
        """Test that cookie token takes priority over header token"""
        # Test with both cookie and header tokens
        token = await get_token_from_request(
            request=mock_request, access_token="cookie-token", token_header="header-token"
        )

        # Cookie should be prioritized
        assert token == "cookie-token"

    @pytest.mark.asyncio
    async def test_get_token_none(self, mock_request):
        """Test behavior when no token is present"""
        # Test with no token
        token = await get_token_from_request(
            request=mock_request, access_token=None, token_header=None
        )

        assert token is None

    @pytest.mark.asyncio
    async def test_verify_token_success(self, mock_db):
        """Test successful token verification"""
        # Mock database query result
        mock_result = MagicMock()
        mock_result.__getitem__.return_value = "user@example.com"
        mock_db.execute.return_value.fetchone.return_value = mock_result

        # Verify token
        result = await verify_token(token="valid-token", db=mock_db)

        # Check that DB query was executed with correct parameters
        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args[0]
        assert "SELECT user_email FROM user_tokens WHERE token = :token" in str(call_args[0])
        assert call_args[1]["token"] == "valid-token"

        # Check result
        assert result == "user@example.com"

    @pytest.mark.asyncio
    async def test_verify_token_no_token(self, mock_db):
        """Test token verification with no token provided"""
        # Try to verify with no token
        with pytest.raises(HTTPException) as excinfo:
            await verify_token(token=None, db=mock_db)

        # Check exception details
        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert excinfo.value.detail == "Not authenticated"
        assert excinfo.value.headers["WWW-Authenticate"] == "Bearer"

        # Verify no database query was executed
        mock_db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_token_invalid(self, mock_db):
        """Test token verification with invalid token"""
        # Mock database query with no result (invalid token)
        mock_db.execute.return_value.fetchone.return_value = None

        # Try to verify with invalid token
        with pytest.raises(HTTPException) as excinfo:
            await verify_token(token="invalid-token", db=mock_db)

        # Check exception details
        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
        # Updated assertion to match the actual format of the error message
        assert "Invalid authentication credentials" in excinfo.value.detail
        assert excinfo.value.headers["WWW-Authenticate"] == "Bearer"

        # Verify database query was executed
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_token_db_error(self, mock_db):
        """Test token verification with database error"""
        # Mock database query raising exception
        mock_db.execute.side_effect = Exception("Database connection error")

        # Try to verify with token
        with pytest.raises(HTTPException) as excinfo:
            await verify_token(token="some-token", db=mock_db)

        # Check exception details
        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Authentication error: Database connection error" in excinfo.value.detail
        assert excinfo.value.headers["WWW-Authenticate"] == "Bearer"

        # Verify database query attempt was made
        mock_db.execute.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-xvs", "tests/test_auth_middleware.py"])
