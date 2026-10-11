"""Standalone BEHAVIOR websocket policy server for policy-only environments."""

import asyncio
from copy import deepcopy
import functools
import http
import json
import logging
import time
import traceback
from typing import Any

import msgpack
import numpy as np
import torch
import websockets
import websockets.asyncio.server as websocket_server

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        logger.info("Starting websocket server on %s:%s", self._host, self._port)
        async with websocket_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=functools.partial(_health_check, metadata=self._metadata),
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))
        previous_total = None
        while True:
            try:
                started = time.monotonic()
                request = unpackb(await websocket.recv(), strict_map_key=False)
                if "reset" in request:
                    self._policy.reset()
                    continue
                infer_started = time.monotonic()
                action = self._policy.act(deepcopy(request))
                infer_ms = (time.monotonic() - infer_started) * 1000
                response = {"action": action.detach().cpu().numpy(), "server_timing": {"infer_ms": infer_ms}}
                if previous_total is not None:
                    response["server_timing"]["prev_total_ms"] = previous_total * 1000
                await websocket.send(packer.pack(response))
                previous_total = time.monotonic() - started
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                logger.error("Policy request failed:\n%s", traceback.format_exc())
                await websocket.close(code=1011, reason="Policy inference failed")
                raise


def _health_check(connection, request, *, metadata=None):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    if request.path == "/metadata":
        return connection.respond(http.HTTPStatus.OK, json.dumps(metadata or {}))
    return None


def _pack_data(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {value.dtype}")
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    return value


def _unpack_data(value):
    if b"__ndarray__" in value:
        return np.ndarray(buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]), shape=value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


Packer = functools.partial(msgpack.Packer, default=_pack_data)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_data)
