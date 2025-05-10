"""
Main Application Entry Point for SMART RAG System (FastAPI)

This module initializes the FastAPI server for the SMART system, setting up:
- CORS middleware for cross-origin frontend/backend interaction.
- Session middleware required by Google OAuth.
- Mounted route handlers for authentication, chat, reporting, and health checks.

Features:
- Enables secure Google OAuth 2.0 login via `/auth`
- Provides chat-based RAG interface via `/api`
- Offers usage analytics and query reports under `/api/reports`
- Supports cookie-based session persistence and CORS headers
- Includes a lightweight health check endpoint at `/health`

Environment Variables:
- `SESSION_SECRET_KEY`: Key for encrypting session cookies
- `FRONTEND_URL`: Allowed origin(s) for CORS requests

Routers:
- `/auth`: Handles Google OAuth login and callbacks
- `/api`: Manages chat sessions and RAG queries
- `/api/reports`: Exposes reporting and analytics data
- `/health`: Health check for deployment monitoring
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.auth_google import router as google_router
from routers.chat_api import router as query_router
from routers.reports import router as reports_router
from routers.health import router as health_router
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
app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(health_router, tags=["health"])
