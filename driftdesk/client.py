"""
DriftDesk environment client.

Uses the OpenEnv WebSocket API (/ws) for stateful multi-step episodes.
HTTP is only used for the health check endpoint.

The OpenEnv HTTP /reset and /step endpoints are stateless (each call creates
a new env instance), so multi-step episodes MUST go through WebSocket.

WebSocket protocol:
  - Send:    {"type": "reset", "data": {"seed": 42, ...}}
  - Receive: {"type": "observation", "data": {"observation": {...}, "reward": null, "done": false}}
  - Send:    {"type": "step", "data": {"module": "...", "payload": {...}}}
  - Receive: {"type": "observation", "data": {"observation": {...}, "reward": 0.7, "done": true}}
  - Send:    {"type": "close"}

Usage (sync, e.g. training loop):
    client = DriftDeskClient("http://localhost:8000")
    with client.session() as sess:
        obs_dict = sess.reset(seed=42, curriculum_stage=1)
        while not obs_dict["done"]:
            obs_dict = sess.step("airline_rebook", {...})

Usage (direct, single session):
    client = DriftDeskClient("http://localhost:8000")
    client.connect()
    obs = client.reset(seed=42)
    obs = client.step("airline_rebook", {...})
    client.close()

For local training without a server, import DriftDeskEnvironment directly.
"""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import requests
import websocket  # websocket-client (sync WebSocket)


class DriftDeskSession:
    """A single persistent WebSocket session to the DriftDesk server.

    Each session maps to one env instance on the server. Supports one
    concurrent multi-step episode per session.
    """

    def __init__(self, ws_url: str, timeout: float = 30.0) -> None:
        self._url = ws_url
        self._timeout = timeout
        self._ws: Optional[websocket.WebSocket] = None

    def connect(self) -> None:
        self._ws = websocket.create_connection(self._url, timeout=self._timeout)

    def close(self) -> None:
        if self._ws:
            try:
                self._ws.send(json.dumps({"type": "close"}))
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def __enter__(self) -> "DriftDeskSession":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _send_recv(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        assert self._ws is not None, "Not connected. Call connect() first."
        self._ws.send(json.dumps(msg))
        raw = self._ws.recv()
        return json.loads(raw)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        curriculum_stage: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reset the environment. Returns the full {observation, reward, done} dict."""
        data: Dict[str, Any] = {}
        if seed is not None:
            data["seed"] = seed
        if episode_id is not None:
            data["episode_id"] = episode_id
        if curriculum_stage is not None:
            data["curriculum_stage"] = curriculum_stage

        resp = self._send_recv({"type": "reset", "data": data})
        return resp.get("data", resp)  # {"observation": {...}, "reward": null, "done": false}

    def step(self, module: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one step. Returns {observation, reward, done}."""
        resp = self._send_recv(
            {"type": "step", "data": {"module": module, "payload": payload}}
        )
        return resp.get("data", resp)

    def state(self) -> Dict[str, Any]:
        """Get current environment state."""
        resp = self._send_recv({"type": "state"})
        return resp.get("data", resp)


class DriftDeskClient:
    """Client for the DriftDesk environment server.

    For each training episode, use session() context manager to get a
    stateful WebSocket session. The health check uses plain HTTP.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._ws_url = (
            self._base
            .replace("https://", "wss://")
            .replace("http://", "ws://")
        ) + "/ws"
        self._timeout = timeout

    @contextmanager
    def session(self) -> Generator[DriftDeskSession, None, None]:
        """Context manager that opens one WebSocket session for a full episode."""
        sess = DriftDeskSession(self._ws_url, self._timeout)
        sess.connect()
        try:
            yield sess
        finally:
            sess.close()

    def healthz(self) -> bool:
        try:
            resp = requests.get(f"{self._base}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
