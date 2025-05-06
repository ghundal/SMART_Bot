"""
Health Check Endpoint for SMART System

This FastAPI router defines a simple health check endpoint to verify that the backend service is running.

Routes:
- `GET /`: Returns HTTP 200 OK if the service is alive.
"""

from fastapi import APIRouter, Request, Response
from starlette.status import HTTP_200_OK

router = APIRouter()


@router.get("/")
async def health(_: Request):
    return Response(status_code=HTTP_200_OK)


@router.get("/eat-mem")
async def health(_: Request):
    if not router.garbage:
        router.garbage = []
    else:
        router.garbage.append([b"0" * 1024 * 1024 * 10])
    return Response(status_code=HTTP_200_OK)
