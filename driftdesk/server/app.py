"""
FastAPI application for the DriftDesk Environment.

Endpoints (OpenEnv standard):
  POST /reset  — start a new episode
  POST /step   — execute an action
  GET  /state  — inspect current environment state
  GET  /schema — OpenAPI schema for action/observation
  WS   /ws     — WebSocket for persistent sessions
  GET  /healthz — health check
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core.env_server.http_server import create_app
from models import DriftDeskAction, DriftDeskObservation
from server.driftdesk_environment import DriftDeskEnvironment

app = create_app(
    DriftDeskEnvironment,
    DriftDeskAction,
    DriftDeskObservation,
    env_name="driftdesk",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)  # calls main()
