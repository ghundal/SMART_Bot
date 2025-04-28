import json
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from fastapi import HTTPException, Request
from starlette.responses import RedirectResponse

# Create mock data
MOCK_GOOGLE_SECRETS = {
    "client_id": "test-client-id",
    "client_secret": "test-client-secret",
}

MOCK_TOKEN = {"access_token": "test-access-token", "expires_in": 3600, "expires_at": 1619011200}

MOCK_USER_INFO = {"email": "test@example.com", "name": "Test User"}

# Mock modules before importing the router
with patch("builtins.open", mock_open(read_data=json.dumps(MOCK_GOOGLE_SECRETS))):
    with patch("src.api.routers.auth_google.OAuth") as MockOAuth:
        # Setup the mock OAuth instance
        mock_oauth = MagicMock()
        mock_google = MagicMock()
        mock_oauth.register.return_value = mock_google
        mock_oauth.google = mock_google
        MockOAuth.return_value = mock_oauth

        # Now import the module
        from src.api.routers.auth_google import oauth, router


class TestAuthGoogle:
    """Tests for the Google OAuth router"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        # Create mock request
        self.mock_request = MagicMock(spec=Request)
        self.mock_request.url_for.return_value = "http://testserver/auth"

        # Mock OAuth methods
        oauth.google.authorize_redirect = AsyncMock(return_value="http://redirect.url")
        oauth.google.authorize_access_token = AsyncMock(return_value=MOCK_TOKEN)

        # Setup mock response for get() method
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_USER_INFO
        oauth.google.get = AsyncMock(return_value=mock_response)

        # Mock database session
        self.mock_db = MagicMock()
        self.mock_execute = MagicMock()
        self.mock_db.execute = self.mock_execute
        self.mock_db.commit = MagicMock()
        self.mock_db.close = MagicMock()

    def test_oauth_registration(self):
        """Test that OAuth is properly registered"""
        assert oauth is not None
        assert hasattr(oauth, "google")

    @pytest.mark.asyncio
    async def test_login_endpoint(self):
        """Test the login endpoint directly"""
        # Get the login function from the router
        login_route = next(route for route in router.routes if route.path == "/login")
        login_func = login_route.endpoint

        # Call the function directly with our mock request
        with patch("src.api.routers.auth_google.oauth", oauth):
            response = await login_func(self.mock_request)

        # Verify that oauth.google.authorize_redirect was called
        oauth.google.authorize_redirect.assert_called_once_with(
            self.mock_request, "http://testserver/auth"
        )

        # Verify the response is the redirect URL
        assert response == "http://redirect.url"

    @pytest.mark.asyncio
    async def test_auth_callback_success(self):
        """Test successful authentication callback"""
        # Setup database mock for success case
        self.mock_execute.return_value.fetchone.return_value = [1]  # User exists

        # Get the auth callback function from the router
        auth_route = next(route for route in router.routes if route.path == "/auth")
        auth_func = auth_route.endpoint

        # Mock RedirectResponse
        mock_redirect = MagicMock(spec=RedirectResponse)

        # Call the function with mocks
        with patch("src.api.routers.auth_google.oauth", oauth), patch(
            "src.api.routers.auth_google.SessionLocal", return_value=self.mock_db
        ), patch("src.api.routers.auth_google.RedirectResponse", return_value=mock_redirect), patch(
            "src.api.routers.auth_google.os.getenv", return_value="http://frontend.url"
        ):
            response = await auth_func(self.mock_request)

        # Verify that the necessary methods were called
        oauth.google.authorize_access_token.assert_called_once_with(self.mock_request)
        oauth.google.get.assert_called_once()

        # Verify database operations
        assert self.mock_execute.call_count >= 2  # At least check and insert
        assert self.mock_db.commit.called
        assert self.mock_db.close.called

        # Verify cookies were set
        assert mock_redirect.set_cookie.call_count >= 2

        # Verify the response is the redirect response
        assert response == mock_redirect

    @pytest.mark.asyncio
    async def test_auth_callback_unauthorized_user(self):
        """Test authentication callback with unauthorized user"""
        # Setup database mock for unauthorized case
        self.mock_execute.return_value.fetchone.return_value = None  # User doesn't exist

        # Get the auth callback function
        auth_route = next(route for route in router.routes if route.path == "/auth")
        auth_func = auth_route.endpoint

        # Call the function and expect HTTPException
        with patch("src.api.routers.auth_google.oauth", oauth), patch(
            "src.api.routers.auth_google.SessionLocal", return_value=self.mock_db
        ), pytest.raises(HTTPException) as excinfo:
            await auth_func(self.mock_request)

        # Verify the 403 error was raised
        assert excinfo.value.status_code == 403
        assert "Access denied" in excinfo.value.detail

        # Verify database was checked but not modified
        self.mock_execute.assert_called_once()
        assert not self.mock_db.commit.called
        assert self.mock_db.close.called

    @pytest.mark.asyncio
    async def test_auth_callback_oauth_error(self):
        """Test authentication callback with OAuth error"""
        # Setup OAuth error
        oauth.google.authorize_access_token = AsyncMock(side_effect=Exception("OAuth Error"))

        # Get the auth callback function
        auth_route = next(route for route in router.routes if route.path == "/auth")
        auth_func = auth_route.endpoint

        # Call the function and expect an exception
        with patch("src.api.routers.auth_google.oauth", oauth), patch(
            "src.api.routers.auth_google.SessionLocal", return_value=self.mock_db
        ), pytest.raises(Exception) as excinfo:
            await auth_func(self.mock_request)

        # Verify the error was propagated
        assert "OAuth Error" in str(excinfo.value)

        # Verify no database operations were performed
        assert not self.mock_execute.called
        assert not self.mock_db.commit.called


if __name__ == "__main__":
    pytest.main(["-xvs", "tests/test_auth_google.py"])
