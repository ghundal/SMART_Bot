"""
Authentication Dependencies for SMART System

This module defines FastAPI dependencies to support secure user authentication
via access tokens stored in cookies or passed via HTTP Authorization headers.
It is used to protect API endpoints and verify user identity.

Key Functions:
1. Extracts tokens from either secure cookies or `Authorization: Bearer` headers.
2. Verifies token validity by checking the `user_tokens` table in PostgreSQL.
3. Returns the authenticated user's email if the token is valid.

Requirements:
- PostgreSQL database with a `user_tokens` table containing valid tokens.
- OAuth-based token issuance during login.

Functions:
- `get_db`: Provides a database session.
- `get_token_from_request`: Retrieves a token from cookies or headers.
- `verify_token`: Confirms token validity and retrieves associated user email.
"""

from typing import Optional

from fastapi import Cookie, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
from utils.database import SessionLocal

# Optional OAuth2 scheme for token extraction from Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


def get_db():
    """Dependency for database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_token_from_request(
    request: Request,
    access_token: Optional[str] = Cookie(None),
    token_header: Optional[str] = Depends(oauth2_scheme),
):
    """
    Extract token from cookies or Authorization header
    Prioritizes cookie-based authentication but falls back to header if needed
    """
    # First try to get token from cookie
    if access_token:
        return access_token

    # Fall back to Authorization header
    if token_header:
        return token_header

    # No token found
    return None


async def verify_token(
    token: Optional[str] = Depends(get_token_from_request),
    db: Session = Depends(get_db),
):
    """
    Verify the access token and return the user email if valid
    Works with both cookie-based and header-based authentication
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Check if the token exists in our database
        result = db.execute(
            text("SELECT user_email FROM user_tokens WHERE token = :token"),
            {"token": token},
        ).fetchone()

        if not result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return result[0]  # Return the user email

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication error: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
