"""
training/config.py — All hyperparameters and environment variables for the
DriftDesk GRPO + SFT training pipeline.

Every tunable value lives here. Import from this module; do not read
os.environ outside this file.
"""
from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# HuggingFace / identity
# ---------------------------------------------------------------------------
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _pick_data_dir() -> str:
    """Resolve the working directory (writable container layout or CWD)."""
    override = os.environ.get("DATA_DIR")
    if override:
        return override
    for cand in ("/app", "/home/user/app"):
        if os.path.isdir(cand) and os.access(cand, os.W_OK):
            return cand
    return os.getcwd()


DATA_DIR: str        = _pick_data_dir()
ADAPTER_SAVE_PATH    = f"{DATA_DIR}/driftdesk_adapter"
RESULTS_CSV          = f"{DATA_DIR}/grpo_training_results.csv"
BASELINE_CSV         = f"{DATA_DIR}/baseline_eval_results.csv"
SFT_SENTINEL         = f"{DATA_DIR}/driftdesk_sft_warmup/.done"
OUTPUT_DIR           = f"{DATA_DIR}/driftdesk_grpo_output"

# ---------------------------------------------------------------------------
# Environment server
# ---------------------------------------------------------------------------
ENV_URL: str = os.environ.get(
    "DRIFTDESK_ENV_URL",
    "https://lokiontheloose-driftdesk.hf.space",
)
WS_URL: str = ENV_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws"

AUTOSTART_ENV_SERVER: bool = os.environ.get("AUTOSTART_ENV_SERVER", "0") == "1"

MAX_EPISODE_STEPS: int = int(os.environ.get("MAX_EPISODE_STEPS", "10"))

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME    = "Qwen/Qwen2.5-3B-Instruct"
LORA_R        = int(os.environ.get("LORA_R", "32"))
LORA_ALPHA    = int(os.environ.get("LORA_ALPHA", str(LORA_R * 2)))
MAX_SEQ_LEN   = 2048
USE_QUANTIZATION: bool = os.environ.get("USE_QUANTIZATION", "1") == "1"

# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------
GRPO_NUM_GENERATIONS = int(os.environ.get("GRPO_NUM_GENERATIONS", "8"))
GRPO_LEARNING_RATE   = float(os.environ.get("LR", "5e-6"))
GRPO_TEMPERATURE     = float(os.environ.get("GRPO_TEMPERATURE", "1.2"))
GRPO_TOP_P           = float(os.environ.get("GRPO_TOP_P", "1.0"))
GRPO_STEPS           = int(os.environ.get("GRPO_STEPS", "500"))
GRPO_BATCH_SIZE      = int(os.environ.get("GRPO_BATCH_SIZE", "4"))
CURRICULUM_STAGE     = int(os.environ.get("CURRICULUM_STAGE", "1"))  # 0=no-drift 1=cued 2=silent
GRAD_ACCUM           = int(os.environ.get("GRAD_ACCUM", "2"))
MAX_COMPLETION_LEN   = int(os.environ.get("MAX_COMPLETION_LEN", "128"))
SAVE_STEPS           = int(os.environ.get("SAVE_STEPS", "5"))
TC_ZERO_ABORT_AFTER  = int(os.environ.get("TC_ZERO_ABORT_AFTER", "150"))
HUB_PUSH_EVERY       = 5   # push adapter to Hub every N checkpoints

# ---------------------------------------------------------------------------
# Shaped training reward weights
#
# The env reward function (RewardEngine.compute_episode_reward) uses fixed
# weights that balance all 7 components for *evaluation*. During GRPO
# training we use a *shaped* reward that amplifies components with high
# gradient variance (tc, dr) and mutes constants (policy_grounding=1.0,
# loop_penalty is always negative). This is explicitly separate from the
# reported eval reward so results are comparable to the rule-based baseline.
#
# Shaped reward (used in grpo_reward_fn when episode is done):
#   shaped = W_TC * tc + W_DR * dr + W_EFF * eff + W_PS * ps
#
# To change these weights you MUST update the comment in rollout.py that
# cross-references them here. The env reward formula is NOT changed.
# ---------------------------------------------------------------------------
SHAPED_W_TC  = 1.5   # task_completion (primary objective)
SHAPED_W_DR  = 1.2   # drift_recovery (secondary objective)
SHAPED_W_EFF = 0.3   # efficiency
SHAPED_W_PS  = 0.3   # priority_score
# Components intentionally excluded from shaped reward:
#   policy_grounding — always 1.0 (zero gradient)
#   loop_penalty     — always negative, suppresses variance
FORMAT_BONUS = 0.05  # step-level partial reward for valid JSON
MODULE_BONUS_MATCH  = 0.04   # calling a module that is in the task list
MODULE_BONUS_AVAIL  = 0.01   # calling a valid-but-not-assigned module

# ---------------------------------------------------------------------------
# SFT warmup
# ---------------------------------------------------------------------------
SFT_SEED_START = 3000
SFT_SEED_END   = 3200
SFT_EPOCHS     = 3
SFT_BATCH      = 2
MAX_SFT_LEN    = 512
SKIP_SFT: bool = os.environ.get("SKIP_SFT", "0") == "1"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_EVERY     = 50
EVAL_N         = 5
EVAL_SEED_START = 1000
MAX_WS_RETRIES  = 2
