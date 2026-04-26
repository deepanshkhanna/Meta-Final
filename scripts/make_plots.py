"""Generate training plots for the DriftDesk submission.

Inputs (read from repo root):
- driftdesk/grpo_training_results.csv          (loss, grad_norm, learning_rate)
- driftdesk/local_training.log                  (per-batch [reward_fn] lines)
- driftdesk/step20_eval_results.csv             (50-episode GRPO eval)
- driftdesk/baseline_eval_results.csv           (50-episode oracle baseline)

Outputs (assets/):
- loss_curve.png
- reward_curve.png
- baseline_comparison.png
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# ---------- Loss + grad-norm ----------
df = pd.read_csv(ROOT / "driftdesk" / "grpo_training_results.csv",
                 names=["loss", "grad_norm", "learning_rate"], header=0,
                 on_bad_lines="skip", engine="python")
df = df.dropna(subset=["loss"]).reset_index(drop=True)
df["step"] = df.index + 1

fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.plot(df["step"], df["loss"], color="#1f77b4", lw=1.6, label="loss")
ax1.set_xlabel("training step")
ax1.set_ylabel("loss", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(df["step"], df["grad_norm"], color="#d62728", lw=1.2, alpha=0.7, label="grad_norm")
ax2.set_ylabel("grad_norm", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")

plt.title("DriftDesk GRPO — Loss & Gradient Norm (Batch 1, 55 steps)")
fig.tight_layout()
fig.savefig(ASSETS / "loss_curve.png", dpi=140)
plt.close(fig)
print("wrote assets/loss_curve.png")

# ---------- Per-batch reward ----------
log_path = ROOT / "driftdesk" / "local_training.log"
pat = re.compile(r"\[reward_fn\] batch n=\d+ mean=([\-0-9.]+) min=([\-0-9.]+) max=([\-0-9.]+) scored=(\d+)/(\d+) tc=([\-0-9.]+) dr=([\-0-9.]+)")
rows = []
with log_path.open() as fh:
    for line in fh:
        m = pat.search(line)
        if m:
            rows.append([float(m.group(1)), float(m.group(2)), float(m.group(3)),
                         int(m.group(4)), int(m.group(5)), float(m.group(6)), float(m.group(7))])
rdf = pd.DataFrame(rows, columns=["mean", "min", "max", "scored", "n", "tc", "dr"])
rdf["batch"] = rdf.index + 1
rdf["scored_frac"] = rdf["scored"] / rdf["n"]
rdf["reward_ema"] = rdf["mean"].ewm(span=8).mean()

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.fill_between(rdf["batch"], rdf["min"], rdf["max"], color="#1f77b4", alpha=0.15, label="min/max")
ax.plot(rdf["batch"], rdf["mean"], color="#1f77b4", lw=1.0, alpha=0.6, label="batch mean reward")
ax.plot(rdf["batch"], rdf["reward_ema"], color="#1f77b4", lw=2.2, label="EMA(8)")
ax.plot(rdf["batch"], rdf["scored_frac"] * rdf["mean"].max(), color="#2ca02c", lw=1.4, ls="--",
        alpha=0.7, label="scored fraction (rescaled)")
ax.set_xlabel("training batch")
ax.set_ylabel("reward")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
plt.title("DriftDesk GRPO — Per-Batch Reward Trajectory (55 steps)")
fig.tight_layout()
fig.savefig(ASSETS / "reward_curve.png", dpi=140)
plt.close(fig)
print(f"wrote assets/reward_curve.png  ({len(rdf)} batches)")

# ---------- Baseline vs step-20 GRPO comparison ----------
base = pd.read_csv(ROOT / "driftdesk" / "baseline_eval_results.csv")
grpo = pd.read_csv(ROOT / "driftdesk" / "step20_eval_results.csv")

metrics = ["reward", "task_completion", "drift_recovery", "policy_grounding", "no_spurious_rewrite"]
labels  = ["Mean\nreward", "Task\ncompletion", "Drift\nrecovery", "Policy\ngrounding", "No spurious\nrewrite"]
base_v = [base[m].mean() for m in metrics]
grpo_v = [grpo[m].mean() for m in metrics]

import numpy as np
x = np.arange(len(metrics))
w = 0.38
fig, ax = plt.subplots(figsize=(9, 4.8))
b1 = ax.bar(x - w/2, base_v, w, label=f"Rule baseline (n={len(base)})", color="#7f7f7f")
b2 = ax.bar(x + w/2, grpo_v, w, label=f"GRPO step-20 (n={len(grpo)})", color="#1f77b4")
for bars in (b1, b2):
    for rect in bars:
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.015,
                f"{rect.get_height():.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("score (0–1)")
ax.set_ylim(0, 1.15)
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper right")
plt.title("DriftDesk — Oracle Baseline vs GRPO step-20 (50 deterministic episodes)")
fig.tight_layout()
fig.savefig(ASSETS / "baseline_comparison.png", dpi=140)
plt.close(fig)
print("wrote assets/baseline_comparison.png")

print("done.")
