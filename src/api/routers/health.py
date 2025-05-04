from fastapi import APIRouter, Request, Response
from starlette.status import HTTP_200_OK

router = APIRouter()


@router.get("/")
async def health(_: Request):
    return Response(status_code=HTTP_200_OK)
