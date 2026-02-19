import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db import close_client
from poller import poll_loop, runs_meta_loop
from routes import checkpoints, metrics, runs, stream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    poll_task = asyncio.create_task(poll_loop())
    meta_task = asyncio.create_task(runs_meta_loop())
    yield
    poll_task.cancel()
    meta_task.cancel()
    for t in (poll_task, meta_task):
        try:
            await t
        except asyncio.CancelledError:
            pass
    close_client()


app = FastAPI(title="Training Metrics Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stream.router)
app.include_router(runs.router)
app.include_router(metrics.router)
app.include_router(checkpoints.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
