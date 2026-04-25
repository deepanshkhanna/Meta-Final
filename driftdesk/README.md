---
title: DriftDesk
emoji: 🛫
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# DriftDesk — OpenEnv Hackathon Submission

**Theme 3.2 — Personalized Tasks / World Modeling**

DriftDesk is an interactive RL environment that teaches AI agents to handle **API schema drift** in a realistic customer-service setting. The agent acts as an executive assistant completing service tasks (airline rebooking, bank dispute filing, insurance claims) while the underlying APIs silently change their schemas mid-episode.

---

## Novelty Claim

> *DriftDesk is, to our knowledge, the first OpenEnv-compliant environment in which the primary training signal rewards an agent for **detecting and adapting to mid-episode schema and policy mutations**, with a decomposed `drift_recovery` reward and held-out drift patterns for generalisation evaluation.*

Nearest neighbours and how DriftDesk differs:

| System | What it does | DriftDesk difference |
|--------|--------------|----------------------|
| **τ-bench** (Sierra/Anthropic) | Multi-turn tool-use benchmark with static policy doc | Benchmark only, policies never change mid-episode |
| **AppWorld** (ICLR '24) | 9 apps, 457 static API endpoints, RL-trainable | APIs are fixed per episode |
| **ToolSandbox** (Apple) | Stateful multi-turn tool use | Tool *definitions* are fixed |
| **API-Bank / ToolBench / NESTFUL** | Tool-use benchmarks | Static schema benchmarks, not RL training envs |
| **SWE-Gym** | Real GitHub-issue fixing | Drift is incidental, not the training signal |

---

## Problem Motivation

**Real incidents where schema drift broke AI agents:**
- **LangChain v0.2 migration (June 2024)**: Breaking changes to `LLMChain`, `ConversationalRetrievalChain`, and tool-calling interfaces silently caused agents built against v0.1 to produce malformed tool calls. [Release notes](https://python.langchain.com/docs/versions/v0_2/)
- **Plaid API v2 → v3 migration (2021–2023)**: The `/accounts/get` response schema was restructured; fields moved and renamed. Apps calling the old schema received 400 errors and had no automated recovery path. [Plaid migration guide](https://plaid.com/docs/api/versioning/)

Production AI agents that call APIs will inevitably encounter schema drift — fields that get renamed, added, removed, or restructured. A robust agent must:
1. Detect drift from 422 error responses
2. Distinguish drift errors from transient failures (HTTP 500)
3. Adapt its payload precisely — only changing drifted fields, not spuriously rewriting everything

DriftDesk provides a controlled, decomposed RL signal to learn exactly this skill.

---

## Environment Design

### Modules (3 API domains × 3 schema versions)

| Module | v1 Fields | v2 Change | v99 (held-out) |
|---|---|---|---|
| `airline_rebook` | `flight_id, passenger_name, new_date` | + `reason_code` (FIELD_ADD) | FIELD_RENAME + ENDPOINT_MOVE |
| `bank_dispute` | `account_id, amount, merchant, description` | + `dispute_type` (FIELD_ADD) | FIELD_REMOVE + ENDPOINT_MOVE |
| `insurance_claim` | `claimant_id, incident_date, amount, description` | Field restructure (FIELD_RENAME + FIELD_ADD + FIELD_REMOVE) | Pre-auth token required (FLOW_RESTRUCTURE) |

### Drift Tracks

- **none**: Clean episode — no schema changes. Tests base task completion.
- **cued**: Agent receives a policy document warning about possible changes.
- **silent**: Drift fires mid-episode with no advance notice. Agent must infer from 422 errors.

### Reward Function (7 components)

**Drift episodes** (40% of training):
```
R = 0.25·task_completion + 0.45·drift_recovery + 0.10·policy_grounding
  + 0.10·priority + 0.10·efficiency + annealed·format - loop_penalty
```

**Clean episodes**:
```
R = 0.50·task_completion + 0.10·policy_grounding + 0.20·priority
  + 0.20·efficiency + annealed·format - loop_penalty
```

**Anti-hacking measures**:
- `drift_recovery = avg(error_grounded_edit, first_retry_success, no_spurious_rewrite)` — rewards precise adaptation, not template switching
- `loop_penalty = min(0.30, 0.05 × repeated_failed_calls)` — penalizes hammering the same wrong payload
- Transient errors (fake HTTP 500s, 8% per step) distinguish "retry same payload" from "adapt on drift"
- `format_valid` weight anneals to 0 by step 50 — only rewards format early in training

### Episode Structure

- **10 steps max** per episode
- 3 concurrent tasks with random priorities (0–2)
- Drift fires between steps 2–8 (configurable schedule)
- Agent receives:
  - Initial policy document (v1 schema descriptions, error handling protocol)
  - Current task list with priorities and completion status
  - Last API result (success confirmation or structured error body)

---

## Stack

| Component | Choice |
|---|---|
| Environment framework | openenv-core 0.2.3 (FastAPI + WebSocket) |
| Training algorithm | GRPO (Group Relative Policy Optimization) |
| Model | Qwen2.5-3B-Instruct |
| LoRA | QLoRA 4-bit, r=16 (plain PEFT — Unsloth not used; RTX 5070 Blackwell SM100 requires PyTorch nightly) |
| Training library | HuggingFace TRL `GRPOTrainer` |
| Deployment | HuggingFace Spaces (`https://lokiontheloose-driftdesk.hf.space`) |

---

## Project Structure

```
driftdesk/
├── openenv.yaml                    # OpenEnv manifest
├── requirements.txt
├── Dockerfile
├── schemas.py                      # Schema DSL + registry (3 modules × 3 versions)
├── models.py                       # OpenEnv Action / Observation / State types
├── client.py                       # WebSocket client (stateful sessions)
├── dummy_env.py                    # Stub env for parallel dev (no server needed)
├── eval_harness.py                 # Deterministic 50-episode evaluation
├── driftdesk_grpo_training.ipynb  # Full GRPO training notebook (Colab-ready)
└── server/
    ├── app.py                      # FastAPI app entrypoint
    ├── driftdesk_environment.py    # Main OpenEnv Environment subclass
    ├── drift_controller.py         # Drift schedule + transient error injection
    ├── policy_injector.py          # Episode-start policy document generation
    ├── reward_engine.py            # 7-component decomposed reward
    └── task_modules/
        ├── base.py
        ├── airline.py
        ├── bank.py
        └── insurance.py
```

---

## Running Locally

```bash
pip install -r requirements.txt

# Start the environment server
python3 -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health

# Run oracle agent baseline (50 episodes)
python3 eval_harness.py --env-url http://localhost:8000 --agent rule_based
```

### WebSocket API (multi-step episodes)

```python
import websocket, json

ws = websocket.create_connection("ws://localhost:8000/ws")

# Reset episode
ws.send(json.dumps({"type": "reset", "data": {"seed": 42, "curriculum_stage": 1}}))
obs = json.loads(ws.recv())["data"]  # {"observation": {...}, "reward": null, "done": false}

# Step
ws.send(json.dumps({"type": "step", "data": {"module": "airline_rebook", "payload": {
    "flight_id": "AI-202", "passenger_name": "Jordan Lee", "new_date": "2026-05-10"
}}}))
result = json.loads(ws.recv())["data"]  # {"observation": {...}, "reward": 0.7, "done": true}

ws.send(json.dumps({"type": "close"}))
ws.close()
```

---

## Training (Colab)

Open `driftdesk_grpo_training.ipynb` in Google Colab (A100 recommended):

1. Set `DRIFTDESK_ENV_URL` to your HF Space URL
2. Run all cells — installs deps, loads Qwen2.5-3B-Instruct with QLoRA, trains for 150 GRPO steps
3. Adapter saved to `./driftdesk_adapter`

**Key hyperparameters:**
- Model: `Qwen/Qwen2.5-3B-Instruct`
- LoRA rank: 16, alpha: 32
- GRPO K: 4 samples per prompt
- LR: 5e-6, batch: 4, steps: 150
- Curriculum: start with cued drift (stage 1), advance to silent (stage 2)

---

## Pre-registered Evaluation Metrics

The following 4 slices are pre-registered **before training**. Results will be reported regardless of direction.

Eval set: 50 deterministic episodes, seeds 1000–1049, fixed drift schedule, `curriculum_stage=1` (cued).

| Slice | Description | Oracle baseline | GRPO-trained |
|-------|-------------|:---:|:---:|
| 1. Clean task completion | Episodes with zero drift events: task completion rate | **1.000** | — |
| 2. Drift task completion | Episodes with ≥1 drift event: overall task completion | **1.000** | — |
| 3. Drift-recovery rate | `avg(error_grounded_edit, first_retry_success, no_spurious_rewrite)` on drift eps | **0.431** | — |
| 4. No spurious rewrite | Fraction of transient-error steps where agent did NOT mutate payload | **1.000** | — |

*Table updated after training run.*

### Oracle Agent Baseline (rule-based, **frozen**)

Eval set: 50 episodes, seeds 1000–1049. 38 clean episodes, 12 drift episodes.

| Metric | Score |
|---|---|
| Mean reward | **0.756** |
| Task completion | **1.000** (all eps) |
| Drift-recovery rate | **0.431** (drift eps only) |
| No spurious rewrite | **1.000** (never mutated on transient error) |
| Steps per episode | 3.0 |

The trained GRPO model is expected to improve slice 3 (drift-recovery) and slice 4 (anti-hack) relative to the oracle, with slice 1 as a sanity baseline.

---

## Links

- **HuggingFace Space**: https://lokiontheloose-driftdesk.hf.space — validated `6/6` via `openenv validate`
- **MCP endpoint**: https://lokiontheloose-driftdesk.hf.space/mcp
- **API schema**: https://lokiontheloose-driftdesk.hf.space/schema
- **Training Notebook**: `driftdesk_grpo_training.ipynb`
- **Eval Harness**: `eval_harness.py`

---

## Team

OpenEnv Hackathon India 2026 submission — Theme 3.2 (Personalized Tasks / World Modeling)
