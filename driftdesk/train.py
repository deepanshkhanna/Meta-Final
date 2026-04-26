#!/usr/bin/env python3
"""
train.py — DriftDesk GRPO training orchestrator.

This script is the entry point for headless training (HF Space, HF Jobs,
local runs). It delegates all logic to the driftdesk.training subpackage:

  training/config.py  — all hyperparameters and env vars
  training/model.py   — tokenizer + model (QLoRA / full bf16)
  training/rollout.py — WebSocket session, prompt, reward function
  training/sft.py     — SFT warm-up phase
  training/grpo.py    — GRPO dataset, callbacks, trainer

Run:
  python train.py
  CURRICULUM_STAGE=1 GRPO_STEPS=500 python train.py
  SKIP_SFT=1 python train.py
"""
import os
import subprocess
import sys
import time
import urllib.request

import torch
from huggingface_hub import login, whoami, create_repo

# ── HF login (with retry) ─────────────────────────────────────────────────────
from driftdesk.training.config import HF_TOKEN, ENV_URL, DATA_DIR, BASELINE_CSV

if HF_TOKEN:
    for _attempt in range(5):
        try:
            login(token=HF_TOKEN, add_to_git_credential=False)
            _me = whoami()
            print(f"Logged in as: {_me['name']}")
            break
        except Exception as _e:
            if _attempt < 4:
                print(f"HF login attempt {_attempt+1} failed ({_e}), retrying in 30s...")
                time.sleep(30)
            else:
                print(f"WARNING: HF login failed after 5 attempts: {_e}")
else:
    print("WARNING: HF_TOKEN not set — checkpoints will NOT be pushed to Hub.")

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Auto-start env server (HF Job mode) ───────────────────────────────────────
from driftdesk.training.config import AUTOSTART_ENV_SERVER

if AUTOSTART_ENV_SERVER:
    print("[env] Auto-starting DriftDesk env server on :8000 ...")
    _env_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "driftdesk.server.app:app",
         "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    for _i in range(30):
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=2)
            print(f"[env] Server ready after {_i+1}s")
            break
        except Exception:
            time.sleep(1)
    else:
        print("[env] WARNING: server health check timed out")
    os.environ["DRIFTDESK_ENV_URL"] = "http://localhost:8000"

import transformers, trl, peft, accelerate, bitsandbytes
print(f"transformers {transformers.__version__} | trl {trl.__version__} | peft {peft.__version__}")

# ── HF Hub repo setup ─────────────────────────────────────────────────────────
HF_REPO_ID = None
if HF_TOKEN:
    try:
        _username = whoami()["name"]
        HF_REPO_ID = f"{_username}/driftdesk-grpo-adapter"
        create_repo(HF_REPO_ID, repo_type="model", exist_ok=True, private=False)
        print(f"Hub repo ready: https://huggingface.co/{HF_REPO_ID}")
    except Exception as _e:
        print(f"Hub repo setup failed: {_e}")

from driftdesk.training.config import CURRICULUM_STAGE, GRPO_NUM_GENERATIONS, GRPO_BATCH_SIZE
print(f"ENV_URL  : {ENV_URL}")
print(f"Rollouts : {GRPO_NUM_GENERATIONS * GRPO_BATCH_SIZE} per step")
print(f"Hub repo : {HF_REPO_ID or 'disabled'}")

# ── Load model + tokenizer ────────────────────────────────────────────────────
from driftdesk.training.model import load_model, load_tokenizer

tokenizer = load_tokenizer()
model     = load_model()

# ── Oracle baseline (run once, skip if CSV exists) ────────────────────────────
if not os.path.exists(BASELINE_CSV):
    print("Running oracle baseline eval...")
    _result = subprocess.run(
        [sys.executable, "-m", "driftdesk.eval_harness",
         "--env-url", ENV_URL,
         "--agent", "rule_based",
         "--out-csv", BASELINE_CSV,
         "--curriculum-stage", str(CURRICULUM_STAGE)],
        capture_output=True, text=True,
    )
    print(_result.stdout[-2000:])
    if _result.returncode != 0:
        print("STDERR:", _result.stderr[-800:])
else:
    print(f"Baseline CSV already exists: {BASELINE_CSV}")

# ── SFT warm-up ───────────────────────────────────────────────────────────────
from driftdesk.training.sft import run_sft_warmup

run_sft_warmup(model, tokenizer)

# ── GRPO reward function ───────────────────────────────────────────────────────
from driftdesk.training.rollout import make_grpo_reward_fn

grpo_reward_fn = make_grpo_reward_fn(model, tokenizer)
print("GRPO reward function ready.")

# ── GRPO training ─────────────────────────────────────────────────────────────
from driftdesk.training.grpo import run_grpo_training

run_grpo_training(model, tokenizer, grpo_reward_fn, hf_repo_id=HF_REPO_ID)

print("Done.")

