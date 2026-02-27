from __future__ import annotations

import asyncio
import json
import logging

logger = logging.getLogger(__name__)

# Each queued item is (event_type, json_payload)
_QueueItem = tuple[str, str]


class Broadcaster:
    def __init__(self) -> None:
        self._clients: set[asyncio.Queue[_QueueItem]] = set()

    def subscribe(self) -> asyncio.Queue[_QueueItem]:
        queue: asyncio.Queue[_QueueItem] = asyncio.Queue(maxsize=64)
        self._clients.add(queue)
        logger.info("SSE client connected (total: %d)", len(self._clients))
        return queue

    def unsubscribe(self, queue: asyncio.Queue[_QueueItem]) -> None:
        self._clients.discard(queue)
        logger.info("SSE client disconnected (total: %d)", len(self._clients))

    async def broadcast(self, data: dict) -> None:
        """Broadcast a delta event (default metric update)."""
        await self.broadcast_event("delta", data)

    async def broadcast_event(self, event_type: str, data: dict) -> None:
        """Broadcast a typed SSE event to all clients."""
        payload = json.dumps(data)
        dead: list[asyncio.Queue[_QueueItem]] = []
        for q in self._clients:
            try:
                q.put_nowait((event_type, payload))
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._clients.discard(q)
            logger.warning("Dropped slow SSE client (total: %d)", len(self._clients))

    @property
    def client_count(self) -> int:
        return len(self._clients)


broadcaster = Broadcaster()
