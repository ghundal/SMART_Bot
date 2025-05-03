from fastapi import APIRouter, HTTP_200_OK, Request, Response

router = APIRouter()

@router.get("/")
async def health(_: Request):
    return Response(status_code=HTTP_200_OK)
