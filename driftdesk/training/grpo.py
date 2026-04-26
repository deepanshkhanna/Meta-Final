"""
training/grpo.py — Dataset construction, training callbacks, and GRPO trainer.

Provides:
  build_grpo_dataset()     — generate prompt dataset from env rollouts
  DeterministicEvalCallback — periodic held-out eval during training
  HubPushCallback           — push adapter to HF Hub every N checkpoints
  EarlyAbortCallback        — stop training if tc=0 for too long
  run_grpo_training()       — assemble and run the GRPOTrainer
"""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path
from typing import List

from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from driftdesk.training.config import (
    DATA_DIR, OUTPUT_DIR, RESULTS_CSV, BASELINE_CSV,
    GRPO_STEPS, GRPO_BATCH_SIZE, GRPO_NUM_GENERATIONS,
    GRPO_LEARNING_RATE, GRPO_TEMPERATURE, GRPO_TOP_P,
    CURRICULUM_STAGE, MAX_COMPLETION_LEN, USE_QUANTIZATION,
    GRAD_ACCUM, SAVE_STEPS, HUB_PUSH_EVERY,
    TC_ZERO_ABORT_AFTER, EVAL_EVERY, EVAL_N, EVAL_SEED_START,
    MAX_EPISODE_STEPS, ENV_URL,
)
from driftdesk.training.rollout import (
    DriftDeskSession, obs_to_prompt, generate_action, parse_action,
    training_log,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_grpo_dataset(n_prompts: int = 200, seed_offset: int = 2000, tokenizer=None) -> Dataset:
    """Generate a dataset of rollout prompts for GRPO training."""
    records = []
    for i in range(n_prompts):
        try:
            sess = DriftDeskSession()
            result = sess.reset(seed=seed_offset + i, curriculum_stage=CURRICULUM_STAGE)
            prompt = obs_to_prompt(result, tokenizer)
            records.append({"prompt": prompt, "seed": seed_offset + i})
            sess.close()
        except Exception as e:
            print(f"  [dataset] seed {seed_offset + i} failed: {e}")
    print(f"Built dataset: {len(records)} prompts")
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class DeterministicEvalCallback(TrainerCallback):
    """Periodic deterministic evaluation on held-out seeds."""

    def __init__(self, model, tokenizer,
                 csv_path: str = f"{DATA_DIR}/grpo_eval_during_training.csv",
                 n_eval: int = EVAL_N, seed_start: int = EVAL_SEED_START) -> None:
        self._model     = model
        self._tokenizer = tokenizer
        self.csv_path   = csv_path
        self.n_eval     = n_eval
        self.seed_start = seed_start
        self._written   = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % EVAL_EVERY != 0:
            return
        rewards, drs, tcs = [], [], []
        for s in range(self.seed_start, self.seed_start + self.n_eval):
            try:
                sess = DriftDeskSession()
                result = sess.reset(seed=s, curriculum_stage=CURRICULUM_STAGE)
                for _ in range(MAX_EPISODE_STEPS):
                    if result.get("done"):
                        break
                    prompt = obs_to_prompt(result, self._tokenizer)
                    text   = generate_action(self._model, self._tokenizer, prompt)
                    action, _ = parse_action(text)
                    if not (isinstance(action, dict) and "module" in action and "payload" in action):
                        break
                    result = sess.step(action["module"], action.get("payload", {}))
                ep_r  = float(result.get("reward") or 0.0)
                obs   = result.get("observation", result)
                comps = (obs.get("reward_components") or {})
                rewards.append(ep_r)
                drs.append(comps.get("drift_recovery", 0.0) or 0.0)
                tcs.append(comps.get("task_completion", 0.0) or 0.0)
                sess.close()
            except Exception as e:
                print(f"  [eval cb] seed {s}: {e}")
        if not rewards:
            return
        row = {
            "step": state.global_step,
            "eval_reward_mean": sum(rewards) / len(rewards),
            "eval_drift_recovery_mean": sum(drs) / len(drs) if drs else 0.0,
            "eval_task_completion_mean": sum(tcs) / len(tcs) if tcs else 0.0,
            "n": len(rewards),
        }
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if not self._written:
                w.writeheader()
                self._written = True
            w.writerow(row)
        print(f"  [eval cb step {state.global_step}] reward={row['eval_reward_mean']:.3f}")


class HubPushCallback(TrainerCallback):
    """Push the adapter to HuggingFace Hub after each checkpoint save."""

    def __init__(self, model, tokenizer, repo_id: str, output_dir: str,
                 push_every: int = HUB_PUSH_EVERY) -> None:
        self._model     = model
        self._tokenizer = tokenizer
        self.repo_id    = repo_id
        self.output_dir = output_dir
        self.push_every = push_every

    def on_save(self, args, state, control, **kwargs):
        if not self.repo_id or state.global_step % self.push_every != 0:
            return
        from huggingface_hub import upload_folder
        tmp = self.output_dir + "/hub_push_tmp"
        try:
            self._model.save_pretrained(tmp)
            self._tokenizer.save_pretrained(tmp)
            upload_folder(
                repo_id=self.repo_id,
                folder_path=tmp,
                repo_type="model",
                commit_message=f"checkpoint step {state.global_step}",
                ignore_patterns=["optimizer.pt", "rng_state*", "scheduler.pt"],
            )
            print(f"  [hub push step {state.global_step}] -> {self.repo_id}")
        except Exception as e:
            print(f"  [hub push] WARN: {e}")


class EarlyAbortCallback(TrainerCallback):
    """Abort training if task_completion stays at 0 for TC_ZERO_ABORT_AFTER steps."""

    def __init__(self) -> None:
        self._tc_zero_streak = 0

    def on_step_end(self, args, state, control, **kwargs):
        if not training_log:
            return
        window = [
            t for t in training_log[-GRPO_BATCH_SIZE * GRPO_NUM_GENERATIONS:]
            if isinstance(t, dict)
        ]
        if not window:
            return
        tc_mean = sum(t.get("tc", 0.0) for t in window) / len(window)
        if tc_mean == 0.0:
            self._tc_zero_streak += 1
            print(
                f"  [watchdog] tc=0 streak={self._tc_zero_streak}/{TC_ZERO_ABORT_AFTER}",
                flush=True,
            )
            if self._tc_zero_streak >= TC_ZERO_ABORT_AFTER:
                print(f"\n{'='*60}", flush=True)
                print(f"FATAL: tc=0 for {TC_ZERO_ABORT_AFTER} consecutive steps.", flush=True)
                print("Model is not completing tasks — stopping to avoid wasted compute.", flush=True)
                print(f"{'='*60}\n", flush=True)
                control.should_training_stop = True
        else:
            self._tc_zero_streak = 0
            print(f"  [watchdog] tc={tc_mean:.3f} — learning signal detected ✓", flush=True)


# ---------------------------------------------------------------------------
# GRPO training entry point
# ---------------------------------------------------------------------------

def run_grpo_training(model, tokenizer, reward_fn, hf_repo_id: str | None = None) -> None:
    """Set up and run GRPOTrainer. Saves the final adapter to ADAPTER_SAVE_PATH."""
    from huggingface_hub import upload_folder

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Resume from latest checkpoint if available
    checkpoints = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    resume_checkpoint = checkpoints[-1] if checkpoints else None
    if resume_checkpoint:
        resume_step = int(resume_checkpoint.rsplit("-", 1)[-1])
        print(f"Resuming from checkpoint: {resume_checkpoint} (step {resume_step})")
        if resume_step >= GRPO_STEPS:
            raise RuntimeError(
                f"GRPO_STEPS={GRPO_STEPS} <= resume_step={resume_step}. "
                f"Set GRPO_STEPS > {resume_step}."
            )
    else:
        print(f"No checkpoint found — starting from scratch. Target: {GRPO_STEPS} steps.")

    train_dataset = build_grpo_dataset(n_prompts=200, seed_offset=2000, tokenizer=tokenizer)

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        max_steps=GRPO_STEPS,
        per_device_train_batch_size=GRPO_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=not USE_QUANTIZATION,
        bf16=not USE_QUANTIZATION,
        learning_rate=GRPO_LEARNING_RATE,
        num_generations=GRPO_NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LEN,
        temperature=GRPO_TEMPERATURE,
        top_p=GRPO_TOP_P,
        beta=0.04,
        logging_steps=1,
        save_steps=SAVE_STEPS,
        save_total_limit=None,
        seed=42,
        report_to="none",
        dataloader_num_workers=0,
    )

    callbacks = [
        DeterministicEvalCallback(model, tokenizer),
        EarlyAbortCallback(),
    ]
    if hf_repo_id:
        callbacks.append(HubPushCallback(model, tokenizer, hf_repo_id, OUTPUT_DIR))

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Patch trainer.log to also write to CSV
    from driftdesk.training import _patch_trainer_log
    _patch_trainer_log(trainer, RESULTS_CSV)

    print(f"Starting GRPO training -> {GRPO_STEPS} steps")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    print(f"Training complete at step {trainer.state.global_step}")

    # Save final adapter
    from driftdesk.training.config import ADAPTER_SAVE_PATH
    Path(ADAPTER_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_SAVE_PATH)
    tokenizer.save_pretrained(ADAPTER_SAVE_PATH)
    print(f"Adapter saved to {ADAPTER_SAVE_PATH}")

    if hf_repo_id:
        upload_folder(
            repo_id=hf_repo_id,
            folder_path=ADAPTER_SAVE_PATH,
            repo_type="model",
            commit_message="Final adapter after training",
            ignore_patterns=["optimizer.pt", "rng_state*", "scheduler.pt"],
        )
        print(f"Final adapter pushed to {hf_repo_id}")
