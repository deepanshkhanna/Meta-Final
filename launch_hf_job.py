#!/usr/bin/env python3
"""
launch_hf_job.py — Launch DriftDesk GRPO training as an HF Job on A100.

Usage:
    python launch_hf_job.py              # launch with default A100 settings
    python launch_hf_job.py --steps 200  # shorter run
    python launch_hf_job.py --hardware a10g-large  # cheaper GPU
    python launch_hf_job.py --status     # check running jobs
    python launch_hf_job.py --stop <job_id>  # cancel a job
    python launch_hf_job.py --logs <job_id>  # stream logs

A100 cost: $0.042/min. 500 steps @ ~15s/step = ~125min = ~$5.25
"""
import argparse, os, sys, textwrap
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
SPACE_REPO = "HelloOjasMutreja/driftdesk-training"
ADAPTER_REPO = "HelloOjasMutreja/driftdesk-grpo-adapter"

# ── Job entrypoint script (runs inside the container) ─────────────────────────
JOB_SCRIPT = textwrap.dedent("""
#!/bin/bash
set -e
echo "=== DriftDesk GRPO Training Job ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"

export HOME=/tmp
export HF_HOME=/tmp/hf
export TRANSFORMERS_CACHE=/tmp/hf
export HF_HUB_CACHE=/tmp/hf
export XDG_CACHE_HOME=/tmp/cache
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p /tmp/hf /tmp/cache /workspace

echo "--- Installing Python deps ---"
pip install -q --no-cache-dir \\
    fastapi==0.115.0 "uvicorn[standard]==0.30.6" pydantic==2.9.2 websocket-client==1.8.0 \\
    websockets==15.0.1 requests==2.32.3 python-dateutil==2.9.0.post0

pip install -q --no-cache-dir --no-deps \\
    transformers==4.46.3 accelerate==1.0.1 \
    bitsandbytes==0.44.1 datasets==3.0.2 "huggingface_hub==0.26.5" \\
    safetensors==0.4.5 sentencepiece==0.2.0 tokenizers==0.20.3 \\
    regex==2024.11.6 tqdm==4.67.0 pyarrow==17.0.0 pandas==2.2.3 \\
    fsspec==2024.6.1 filelock==3.16.1 multiprocess==0.70.16 dill==0.3.8 \\
    xxhash==3.5.0 aiohttp==3.10.10 psutil==6.0.0 packaging==24.1 \\
    pyyaml==6.0.2 rich==13.9.4 \\
    multidict==6.1.0 yarl==1.15.4 frozenlist==1.5.0 aiosignal==1.3.1 \\
    attrs==24.2.0 async-timeout==4.0.3 aiohappyeyeballs==2.4.3 propcache==0.2.0

# Install openenv-core and trl WITH deps so server + GRPOTrainer import correctly
pip install -q --no-cache-dir openenv-core==0.2.2
pip install -q --no-cache-dir --upgrade --no-deps "trl>=0.13,<0.16" peft==0.13.2
pip install -q --no-cache-dir starlette==0.38.6 anyio==4.6.2.post1 sniffio==1.3.1 \
    h11==0.14.0 httpcore==1.0.6 httpx==0.27.2
# openenv-core may have bumped huggingface_hub past transformers 4.46.3's <1.0 requirement.
# Re-pin it (with --no-deps so we don't undo the openenv install).
pip install -q --no-cache-dir --no-deps --force-reinstall "huggingface_hub==0.26.5"
echo "--- Verifying critical imports ---"
python3 -c "from openenv.core.env_server.http_server import create_app; print('openenv OK')" || exit 1
python3 -c "from trl import GRPOConfig, GRPOTrainer; print('trl GRPO OK')" || exit 1

echo "--- Cloning Space repo to /workspace ---"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '{space_repo}',
    repo_type='space',
    local_dir='/workspace',
    ignore_patterns=['*.ipynb', 'driftdesk_grpo_output*', 'driftdesk_sft_warmup*', '__pycache__*']
)
print('Repo cloned.')
"

cd /workspace
export DATA_DIR=/workspace

echo "--- Starting DriftDesk env server on :8000 ---"
nohup uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1 > /tmp/env_server.log 2>&1 &
ENV_PID=$!

# Wait for env server (try /healthz then /health then /docs)
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/healthz > /dev/null 2>&1 \
       || curl -sf http://localhost:8000/health > /dev/null 2>&1 \
       || curl -sf http://localhost:8000/docs > /dev/null 2>&1; then
        echo "Env server ready after ${{i}}s (PID=$ENV_PID)"
        break
    fi
    sleep 1
done
# Probe once more, dump server log if still not up
curl -sf http://localhost:8000/docs > /dev/null 2>&1 || {{
    echo "!!! env server not responding — log tail:"
    tail -50 /tmp/env_server.log || true
}}

echo "--- Launching GRPO training ---"
export DRIFTDESK_ENV_URL=http://localhost:8000
export USE_QUANTIZATION=0
export GRPO_BATCH_SIZE={batch_size}
export GRPO_NUM_GENERATIONS={num_generations}
export GRPO_STEPS={steps}
export GRPO_TEMPERATURE={temperature}
export GRPO_TOP_P=1.0
export CURRICULUM_STAGE={curriculum_stage}
export SAVE_STEPS=10
export MAX_COMPLETION_LEN=128
export LR=5e-6
export LORA_R=32
export GRAD_ACCUM=1

python3 train.py 2>&1 | tee /tmp/training.log
EXIT_CODE=$?
echo "Training finished with exit code $EXIT_CODE"
exit $EXIT_CODE
""")


def launch(args):
    api = HfApi(token=HF_TOKEN)
    script = JOB_SCRIPT.format(
        space_repo=SPACE_REPO,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        steps=args.steps,
        temperature=args.temperature,
        curriculum_stage=args.curriculum_stage,
    )

    # Estimate cost
    steps_per_min = 60 / 15  # ~15s/step on A100 (env calls dominate)
    est_minutes = args.steps / steps_per_min
    cost_per_min = {"a100-large": 0.041667, "a10g-large": 0.025, "t4-medium": 0.01}
    est_cost = est_minutes * cost_per_min.get(args.hardware, 0.042)
    print(f"Launching HF Job:")
    print(f"  Hardware : {args.hardware}")
    print(f"  Steps    : {args.steps} @ ~15s/step = ~{est_minutes:.0f} min")
    print(f"  Est. cost: ${est_cost:.2f}")
    print(f"  Rollouts : {args.batch_size * args.num_generations}/step")
    print(f"  Timeout  : {args.timeout}")
    print()

    job = api.run_job(
        image="pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime",
        command=["bash", "-c", script],
        flavor=args.hardware,
        secrets={"HF_TOKEN": HF_TOKEN},
        env={
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        timeout=args.timeout,
        labels={"project": "driftdesk", "type": "grpo-training"},
    )
    print(f"Job launched! ID: {job.id}")
    print(f"Monitor: https://huggingface.co/jobs/{job.id}")
    print(f"Logs:    python launch_hf_job.py --logs {job.id}")
    return job


def show_status(args):
    import time
    api = HfApi(token=HF_TOKEN)
    for attempt in range(5):
        try:
            jobs = list(api.list_jobs())
            break
        except Exception as e:
            if attempt < 4:
                print(f"Rate limited, retrying in 30s... ({e})")
                time.sleep(30)
            else:
                raise
    if not jobs:
        print("No jobs found.")
        return
    print(f"{'ID':<28} {'STATUS':<20} {'HARDWARE':<15} {'CREATED'}")
    print("-" * 80)
    for j in jobs:
        status = getattr(j.status, 'stage', str(j.status))
        created = getattr(j, 'created_at', '?')
        if created != '?':
            created = created.strftime('%Y-%m-%d %H:%M UTC')
        print(f"{j.id:<28} {status:<20} {getattr(j,'flavor','?'):<15} {created}")


def show_logs(args):
    api = HfApi(token=HF_TOKEN)
    print(f"Streaming logs for job {args.job_id}...")
    for line in api.fetch_job_logs(job_id=args.job_id, follow=True):
        print(line, end="", flush=True)


def stop_job(args):
    api = HfApi(token=HF_TOKEN)
    api.cancel_job(job_id=args.job_id)
    print(f"Job {args.job_id} cancelled.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DriftDesk HF Job launcher")
    subparsers = parser.add_subparsers(dest="cmd")

    # launch (default)
    launch_p = subparsers.add_parser("launch", help="Launch training job")
    launch_p.add_argument("--hardware", default="a100-large",
                          choices=["a100-large", "a10g-large", "t4-medium", "l4x1"])
    launch_p.add_argument("--steps", type=int, default=500)
    launch_p.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    launch_p.add_argument("--num-generations", type=int, default=8, dest="num_generations")
    launch_p.add_argument("--temperature", type=float, default=1.2)
    launch_p.add_argument("--curriculum-stage", type=int, default=1, dest="curriculum_stage")
    launch_p.add_argument("--timeout", default="6h")

    # status
    subparsers.add_parser("status", help="List all jobs")

    # logs
    logs_p = subparsers.add_parser("logs", help="Stream job logs")
    logs_p.add_argument("job_id")

    # stop
    stop_p = subparsers.add_parser("stop", help="Cancel a job")
    stop_p.add_argument("job_id")

    args = parser.parse_args()

    if args.cmd == "status":
        show_status(args)
    elif args.cmd == "logs":
        show_logs(args)
    elif args.cmd == "stop":
        stop_job(args)
    else:
        # default: launch
        if args.cmd != "launch":
            # called with no subcommand — use defaults
            class DefaultArgs:
                hardware = "a100-large"
                steps = 500
                batch_size = 8
                num_generations = 8
                temperature = 1.2
                curriculum_stage = 1
                timeout = "6h"
            args = DefaultArgs()
        launch(args)
