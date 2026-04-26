"""
training/sft.py — SFT warm-up phase.

Runs 200 oracle (rule-based) episodes and fine-tunes the model to imitate
the oracle's actions. This cold-start fix ensures the model produces valid
JSON before GRPO begins.

The SFT warm-up is gated by a sentinel file (SFT_SENTINEL). Once complete,
subsequent runs skip it. Set SKIP_SFT=1 to bypass unconditionally.
"""
from __future__ import annotations

import gc
import json
import os
import random

import torch

from driftdesk.eval_harness import RuleBasedAgent
from driftdesk.training.config import (
    SFT_SENTINEL, SFT_SEED_START, SFT_SEED_END,
    SFT_EPOCHS, SFT_BATCH, MAX_SFT_LEN, SKIP_SFT,
    CURRICULUM_STAGE, DATA_DIR,
)
from driftdesk.training.rollout import DriftDeskSession, obs_to_prompt


def run_sft_warmup(model, tokenizer) -> None:
    """Run SFT warm-up if not already done; no-op if sentinel exists."""
    if SKIP_SFT and not os.path.exists(SFT_SENTINEL):
        os.makedirs(f"{DATA_DIR}/driftdesk_sft_warmup", exist_ok=True)
        open(SFT_SENTINEL, "w").close()
        print("SKIP_SFT=1 — skipping SFT warmup.")

    if os.path.exists(SFT_SENTINEL):
        print("SFT warm-up already done (sentinel found).")
        return

    print("Running SFT warm-up (cold-start fix)...")
    records = _collect_sft_records(tokenizer)
    _run_sft_loop(model, tokenizer, records)

    os.makedirs(f"{DATA_DIR}/driftdesk_sft_warmup", exist_ok=True)
    open(SFT_SENTINEL, "w").close()
    print("SFT warm-up complete.")


def _collect_sft_records(tokenizer) -> list:
    records = []
    for s in range(SFT_SEED_START, SFT_SEED_END):
        try:
            sess = DriftDeskSession()
            result = sess.reset(seed=s, curriculum_stage=CURRICULUM_STAGE)
            agent = RuleBasedAgent()
            agent.reset(result.get("observation", result).get("tasks", []))
            for _ in range(10):
                action = agent.act(result)
                if action is None:
                    break
                prompt_text = obs_to_prompt(result, tokenizer)
                target = json.dumps(action) + "<|im_end|>"
                records.append(prompt_text + target)
                result = sess.step(action["module"], action.get("payload", {}))
                if result.get("done"):
                    break
            sess.close()
        except Exception as e:
            print(f"  [sft] seed {s}: {e}")
    print(f"Collected {len(records)} SFT (prompt, action) pairs")
    return records


def _run_sft_loop(model, tokenizer, records: list) -> None:
    from torch.optim import AdamW

    gc.collect()
    torch.cuda.empty_cache()

    model.train()
    sft_optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )
    rng = random.Random(42)
    total_loss = 0.0
    n_steps = 0

    for epoch in range(SFT_EPOCHS):
        rng.shuffle(records)
        for b_start in range(0, len(records), SFT_BATCH):
            batch_texts = records[b_start : b_start + SFT_BATCH]
            enc = tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_SFT_LEN,
            ).to(model.device)
            labels = enc["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            if (labels != -100).sum() == 0:
                continue
            try:
                out = model(**enc, labels=labels)
                loss = out.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                sft_optimizer.step()
                sft_optimizer.zero_grad()
                total_loss += loss.item()
                n_steps += 1
                if n_steps % 50 == 0:
                    print(
                        f"  [sft] epoch={epoch+1} step={n_steps} "
                        f"loss={total_loss/n_steps:.4f}",
                        flush=True,
                    )
            except RuntimeError as e:
                print(f"  [sft] skipped batch: {e}")
                sft_optimizer.zero_grad()
                torch.cuda.empty_cache()

    del sft_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"SFT loop done. steps={n_steps} avg_loss={total_loss/max(n_steps, 1):.4f}")
