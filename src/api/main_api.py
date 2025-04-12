from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from auth_google import router as google_router
import os

app = FastAPI()

# Required middleware for OAuth to use session storage
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "super-secret-dev-key")
)

# Add CORS middleware to allow frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:9000")],
    allow_credentials=True,  # Important for cookies to work across domains
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include the Google OAuth router under `/auth`
app.include_router(google_router, prefix="/auth")
