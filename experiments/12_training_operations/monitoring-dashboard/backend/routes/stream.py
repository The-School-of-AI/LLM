import asyncio
import json

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request

from cache import cache
from sse import broadcaster

router = APIRouter()


async def _event_generator(request: Request):
    # Send full snapshot on connect
    snapshot = cache.snapshot()
    yield {"event": "snapshot", "data": json.dumps(snapshot)}

    # Subscribe for deltas
    queue = broadcaster.subscribe()
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                event_type, data = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield {"event": event_type, "data": data}
            except asyncio.TimeoutError:
                # Send keepalive comment
                yield {"comment": "keepalive"}
    finally:
        broadcaster.unsubscribe(queue)


@router.get("/stream")
async def stream(request: Request):
    return EventSourceResponse(_event_generator(request))
