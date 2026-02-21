from fastapi import APIRouter

from cache import cache

router = APIRouter()


@router.get("/runs")
async def list_runs():
    return {"runs": cache.all_runs()}
