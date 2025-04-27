"""
Initializes the FastAPI application, sets up CORS and session middleware
for authentication, and includes both authentication and chat API routers
under their respective URL prefixes.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.auth_google import router as google_router
from routers.chat_api import router as query_router
from routers.reports import router as reports_router
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI()

# Required middleware for OAuth to use session storage
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "super-secret-dev-key"),
)

# Add CORS middleware to allow frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_URL", "http://localhost:9000"),
        os.getenv("FRONTEND_URL", "http://localhost:3000"),
    ],
    allow_credentials=True,  # Important for cookies to work across domains
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the Google OAuth router under `/auth`
app.include_router(google_router, prefix="/auth")

# Include the chat/query router under `/api`
app.include_router(query_router, prefix="/api")

# Add the reports router under `/api`
app.include_router(reports_router, prefix="/api", tags=["reports"])
