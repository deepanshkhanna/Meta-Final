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
import os

from openenv.core.env_server.http_server import create_app
from driftdesk.models import DriftDeskAction, DriftDeskObservation
from driftdesk.server.driftdesk_environment import DriftDeskEnvironment

app = create_app(
    DriftDeskEnvironment,
    DriftDeskAction,
    DriftDeskObservation,
    env_name="driftdesk",
    max_concurrent_envs=4,
)

from fastapi import HTTPException
from fastapi.responses import PlainTextResponse

# Optional bearer-token auth for the training log endpoint.
# Set TRAINING_LOG_TOKEN in the environment to require ?token=<value>.
_TRAINING_LOG_TOKEN: str | None = os.environ.get("TRAINING_LOG_TOKEN")


@app.get("/training-log")
def training_log(tail: int = 200, token: str | None = None):
    """
    Return the last `tail` lines of training.log.

    If the env var TRAINING_LOG_TOKEN is set, the caller must supply the
    matching value as ?token=<value>. Returns HTTP 403 otherwise.
    Note: tail is capped at 5000 to prevent excessive response sizes.
    """
    if _TRAINING_LOG_TOKEN and token != _TRAINING_LOG_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    tail = min(tail, 5000)

    candidates = ["/app/training.log", "/home/user/app/training.log"]
    for log_path in candidates:
        if os.path.exists(log_path):
            with open(log_path, "r", errors="replace") as f:
                lines = f.readlines()
            return PlainTextResponse(
                f"# log_path={log_path} bytes={os.path.getsize(log_path)}\n"
                + "".join(lines[-tail:])
            )

    return PlainTextResponse(
        "training.log not found. Candidates checked: " + ", ".join(candidates),
        status_code=404,
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
