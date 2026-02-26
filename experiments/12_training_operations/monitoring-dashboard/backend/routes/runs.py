from cache import cache
from fastapi import APIRouter

router = APIRouter()


@router.get("/runs")
async def list_runs():
    return {"runs": cache.all_runs()}
