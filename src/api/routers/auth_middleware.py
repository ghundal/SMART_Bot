'''
Implements FastAPI dependency functions to verify authentication tokens
from either cookies or headers(docs) by checking them against a database.
'''

from fastapi import Depends, HTTPException, status, Request, Cookie
from fastapi.security import OAuth2PasswordBearer
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
from api.utils.database import SessionLocal

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
    token_header: Optional[str] = Depends(oauth2_scheme)
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
    db: Session = Depends(get_db)
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
            {"token": token}
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