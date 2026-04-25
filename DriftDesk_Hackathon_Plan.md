# DriftDesk — Complete Hackathon Execution Plan
### OpenEnv Hackathon India 2026 | Theme 3.2 + Patronus AI Bonus + Halluminate Bonus

---

## 1. PROBLEM STATEMENT (Refined)

### Precise Problem Definition
Every LLM agent deployed in production today is trained on a **static snapshot of the world** — fixed APIs, fixed schemas, fixed policies. When the real world changes (an API deprecates a field, a bank switches from form-based to chatbot-first flows, an expense policy adds new required fields), the agent fails. Not because it lacks intelligence, but because it was never trained to handle **rule drift**.

No existing RL training environment — not AppWorld, not TextArena, not any 2025/2026 paper — trains agents to detect and adapt to mid-episode schema and policy changes. **DriftDesk is the first.**

### Objectives
1. Build the first OpenEnv-compliant RL environment that introduces **schema and policy drift** as a first-class training variable
2. Train an LLM agent that can detect drift from error signals and adapt its strategy mid-episode
3. Demonstrate measurable improvement: trained model maintains >70% task completion under drift; baseline collapses to <20%

### Target Users / Personas
- **Hackathon Judges**: Need novel environment + clear training evidence + strong story
- **ML Researchers**: Want a reproducible benchmark for agent robustness research
- **Enterprise AI Teams**: Building agentic systems that break in production due to schema changes
- **OpenEnv Community**: Will adopt DriftDesk as a standard robustness-training environment

### Why This Problem Matters NOW
- Agentic AI market: **$7.29B in 2025 → $139B by 2034** (40.5% CAGR)
- By 2028, 33% of enterprise software will include agentic AI
- **40% of agentic AI projects fail due to inadequate foundations** — schema drift is the #1 infrastructure failure mode
- A single LangChain version upgrade in 2025 broke enterprise workflows simultaneously across FlowiseAI, Zed IDE, and OpenAI Agents SDK — engineers called it "schema drift" by name
- Nobody has built the training environment for it. Until now.

---

## 2. GAP ANALYSIS

### What Already Exists (Surveyed)

| System | What It Does | Why It's Not Enough |
|--------|-------------|---------------------|
| Apple LOOP / AppWorld | RL agent trained with fixed API schemas | Schemas never change between episodes |
| TextArena | Multi-agent text game environments | No API/schema drift mechanics |
| Schema First Tool APIs (Mar 2026) | Studies tool misuse with quality issues | Static evaluation benchmark, not RL training env |
| PhantomPolicy (Apr 2026) | Policy-invisible violations research | Static per-run policies, benchmark only |
| Calendar Gym / EnterpriseArena | Task-based RL environments | Fixed schemas throughout |
| OpenSec | Security-focused agent environments | No drift mechanics |

### What Is Saturated
- Basic task completion environments (email, calendar, web browsing)
- Single-schema API interaction agents
- Static benchmark evaluations of agent robustness

### The Confirmed Gap
**There is no RL training environment anywhere — in OpenEnv, in published research, or in any open-source repo — that trains agents to be robust to mid-episode schema and policy drift.**

The research community acknowledges this as a failure mode. The production evidence is documented. The training environment does not exist. **That is the gap DriftDesk fills.**

---

## 3. SOLUTION DESIGN

### Core Idea
**DriftDesk**: A personal executive assistant RL environment where the agent manages realistic personal tasks (flight rebooking, bank disputes, expense reports, subscription cancellations, insurance claims) across a simulated week — and the schemas, APIs, and policies governing those tasks **silently change** between episodes and sometimes mid-task.

### What Makes It Novel
1. **Drift as a Training Variable**: Schema drift events are injected stochastically, not as edge cases but as the core training signal
2. **Policy Document Injection**: Each episode begins with a text description of current rules — teaching agents to ground behavior in stated policy
3. **Drift Recovery Reward**: A dedicated reward component tracks whether the agent detected drift from error feedback and successfully adapted
4. **Multi-Actor Layer**: Competing actors (Boss, Partner, Bank, Insurance Portal) create priority conflicts that must be sequenced correctly

### Key Features

**MVP (Must Build)**
- 5 task modules with 3 schema versions each
- Schema drift controller (stochastic activation)
- Drift recovery reward component
- Policy document injector
- Baseline rollout with frozen model
- Before/after demo

**Advanced (If Time Allows)**
- Multi-actor priority conflict layer (Halluminate bonus)
- Curriculum: start with no drift, escalate drift frequency
- Wandb integration for live reward curves
- HuggingFace Space interactive demo

---

## 4. TECHNICAL ARCHITECTURE

### System Overview

```
┌─────────────────────────────────────────────────┐
│                  DriftDesk Core                  │
│                                                  │
│  ┌──────────────┐    ┌──────────────────────┐   │
│  │Policy Document│    │ Schema Drift         │   │
│  │  Injector    │───▶│ Controller           │   │
│  └──────────────┘    └──────────┬───────────┘   │
│                                 │                │
│  ┌──────────────────────────────▼─────────────┐ │
│  │           Task Modules (5)                  │ │
│  │  Airline | Bank | Insurance | Sub | Expense │ │
│  │  Schema v1 / v2 / v3 per module             │ │
│  └──────────────────┬──────────────────────────┘ │
│                     │                            │
│  ┌──────────────────▼──────────────────────────┐ │
│  │         Reward Engine                        │ │
│  │  task_completion + drift_recovery +          │ │
│  │  priority_score + efficiency                 │ │
│  └──────────────────┬──────────────────────────┘ │
└────────────────────-│───────────────────────────┘
                      │
        ┌─────────────▼──────────────┐
        │   OpenEnv FastAPI Server   │
        │   (reset / step / state)   │
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │    TRL + GRPO Trainer      │
        │    (Unsloth efficiency)    │
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │   HuggingFace Space Demo   │
        │   (before/after + curves)  │
        └────────────────────────────┘
```

### Frontend
- **Framework**: Gradio (on HuggingFace Spaces) for demo UI
- **Why**: Native Spaces integration, zero deployment friction, judges can run it immediately
- **UI Elements**: Episode replay viewer, reward curve chart, schema diff viewer, task completion dashboard

### Backend / Environment
- **Framework**: FastAPI (OpenEnv standard)
- **Language**: Python 3.11
- **Structure**:
  - `DriftDeskEnvironment` — extends OpenEnv `Environment` base class
  - `SchemaDriftController` — manages drift event scheduling and activation
  - `PolicyDocumentInjector` — generates episode-start policy text
  - `TaskModuleRegistry` — holds 5 task modules × 3 schema versions

### Database / State
- **Episode State**: In-memory Python dataclass (no persistent DB needed for hackathon)
- **Schema Registry**: JSON files per task module, versioned (v1/v2/v3)
- **Training Logs**: Wandb + local CSV export

### AI/ML Components
- **Base Model**: `Llama-3.1-8B-Instruct` (via Unsloth 4-bit quantization)
- **Training Algorithm**: GRPO (via TRL `GRPOTrainer`)
- **Why GRPO over PPO**: No value model needed, simpler setup, better for verifiable reward environments
- **Why Unsloth**: 2x training speed, fits in Colab T4 GPU, essential for hackathon time constraints

### Task Module Schemas

```
Airline Rebooking:
  v1: {flight_id, passenger_name, new_date}
  v2: {flight_id, passenger_name, new_date, reason_code}  ← drift: adds mandatory field
  v3: {pnr, traveler_id, departure_date, reason_code, fare_class}  ← drift: full restructure

Bank Dispute:
  v1: Form-based: {account, amount, merchant, description}
  v2: Chatbot-first: POST /chat with NL message  ← drift: form deprecated
  v3: Structured JSON: {dispute_type, transaction_id, evidence_url}

Insurance Claim:
  v1: PDF upload + {claimant, incident_date, amount}
  v2: JSON itemized: {line_items: [{code, cost}], incident_id}  ← drift: PDF rejected
  v3: Pre-auth required: GET /pre-auth → token → POST /claim

Subscription Cancel:
  v1: DELETE /subscription/{id}
  v2: POST /cancel with mandatory retention_survey_id  ← drift: adds survey step
  v3: Two-step: GET /cancellation-token → POST /confirm-cancel

Expense Report:
  v1: {amount, category, receipt_url}
  v2: {amount, category, receipt_url, gl_code}  ← drift: GL code required
  v3: {amount, category, receipt_url, gl_code, cost_center, project_id}
```

### Reward Function (Full Implementation)

```python
def compute_reward(episode: Episode) -> float:
    # 1. Task Completion (50%) — Did the tasks actually complete?
    task_completion = sum(
        task.completed for task in episode.tasks
    ) / len(episode.tasks)
    
    # 2. Drift Recovery (30%) — Did agent adapt after drift-induced errors?
    drift_recovery = (
        episode.successful_recoveries / max(episode.drift_events, 1)
    )
    
    # 3. Priority Sequencing (10%) — Deadlines respected?
    priority_score = evaluate_sequencing(
        episode.task_order, episode.deadlines
    )
    
    # 4. Efficiency (10%) — Steps used vs minimum required
    efficiency = 1.0 - (episode.steps_taken / episode.max_steps)
    efficiency = max(0.0, efficiency)  # clamp
    
    # Anti-gaming: penalize repeated identical failed calls
    loop_penalty = 0.1 * episode.repeated_failed_calls
    
    return max(0.0,
        0.50 * task_completion +
        0.30 * drift_recovery +
        0.10 * priority_score +
        0.10 * efficiency -
        loop_penalty
    )
```

### Data Flow (Step-by-Step)

1. `env.reset()` → SchemaDriftController initializes active schema versions, PolicyDocumentInjector generates episode policy text → returns initial observation with policy doc + task list
2. Agent reads policy doc, selects first task, calls appropriate API action
3. `env.step(action)` → TaskModule processes action against **current** schema version → returns success or schema-mismatch error
4. Mid-episode drift event fires (stochastically) → SchemaDriftController swaps schema version silently → no notification to agent
5. Agent receives error from next action, must infer schema changed, update strategy
6. On successful adaptation, `drift_recovery` counter increments
7. Episode ends when all tasks complete or max_steps reached → `compute_reward()` called → reward returned to TRL trainer
8. GRPO collects rollouts, computes advantages, updates model weights

---

## 5. IMPLEMENTATION ROADMAP

### Phase 1: Environment Setup (Hours 1–4)

| Task | Owner | Time | Dependency |
|------|-------|------|------------|
| OpenEnv init + scaffold | Env Dev | 1h | None |
| Define action/observation dataclasses | Env Dev | 45min | scaffold |
| Implement 5 task modules v1 (no drift yet) | Env Dev | 1.5h | dataclasses |
| Implement `reset()` and basic `step()` | Env Dev | 45min | task modules |
| Local loop test: agent takes random actions | All | 30min | step/reset |

**Exit Criteria**: `env.reset()` runs, `env.step(action)` returns observation without crashing

---

### Phase 2: Core Features (Hours 4–12)

| Task | Owner | Time | Dependency |
|------|-------|------|------------|
| Add v2 and v3 schemas to all 5 task modules | Env Dev | 2h | Phase 1 |
| Build SchemaDriftController | Env Dev | 2h | v2/v3 schemas |
| Build PolicyDocumentInjector | AI Dev | 1.5h | schema registry |
| Implement full reward function (4 components) | Reward Dev | 2h | Phase 1 |
| Add anti-reward-hacking: timeouts, loop detection | Reward Dev | 1h | reward function |
| Test drift events fire correctly | Env Dev | 1h | drift controller |

**Exit Criteria**: Agent can hit a drift-induced error and the episode continues correctly

---

### Phase 3: Training Integration (Hours 12–20)

| Task | Owner | Time | Dependency |
|------|-------|------|------------|
| Set up TRL + Unsloth environment in Colab | ML Dev | 1h | None (parallel) |
| Load Llama-3.1-8B-Instruct with 4-bit QLoRA | ML Dev | 30min | TRL setup |
| Write rollout function connecting OpenEnv client to TRL | ML Dev | 2h | Phase 2 complete |
| Deploy environment to HuggingFace Space | Env Dev | 1h | Phase 2 |
| Run baseline rollout (frozen model, no training) | ML Dev | 1h | deployment |
| Record baseline metrics: task completion vs drift freq | ML Dev | 30min | baseline rollout |
| Run first GRPO training run (50 episodes) | ML Dev | 2h | rollout function |
| Inspect generations for reward hacking | All | 1h | first run |

**Exit Criteria**: Training loss moves, reward goes up, no obvious hacking

---

### Phase 4: Training + Validation (Hours 20–30)

| Task | Owner | Time | Dependency |
|------|-------|------|------------|
| Full training run (300–500 episodes with curriculum) | ML Dev | 4h | Phase 3 |
| Add multi-actor layer (Boss/Partner/Bank/Insurance) | Env Dev | 2h | stable env |
| Generate before/after comparison trajectories | ML Dev | 1h | trained model |
| Plot reward curves, drift recovery rate vs baseline | Demo Dev | 1h | training data |
| Save model correctly (LoRA merge → test inference) | ML Dev | 1h | training complete |

**Exit Criteria**: Trained model visibly outperforms baseline on drift recovery

---

### Phase 5: Demo + Polish (Hours 30–36)

| Task | Owner | Time | Dependency |
|------|-------|------|------------|
| Build Gradio demo on HuggingFace Space | Demo Dev | 2h | trained model |
| Write README with problem, env, results, plots | Demo Dev | 1.5h | all complete |
| Record < 2 min HuggingFace blog / YouTube video | Demo Dev | 1h | demo working |
| Rehearse 3-minute pitch (story → demo → impact) | All | 1h | README |
| Final submission: verify all links, openenv.yaml valid | All | 30min | everything |

---

## 6. TEAM ROLE DISTRIBUTION (4 Members)

### Member A — Environment Architect
**Primary focus**: DriftDesk OpenEnv environment
- Implements `DriftDeskEnvironment`, `reset()`, `step()`, `state()`
- Builds all 5 task modules across 3 schema versions
- Implements `SchemaDriftController`
- Adds timeouts, loop detection, anti-cheat constraints
- Deploys to HuggingFace Space

### Member B — Reward Engineer
**Primary focus**: Reward function and verifier
- Implements 4-component reward function
- Builds anti-reward-hacking checks (loop detection, timeout penalties)
- Implements `PolicyDocumentInjector`
- Writes unit tests for each reward component
- Validates that reward hacking paths are closed

### Member C — ML / Training
**Primary focus**: Training pipeline
- Sets up TRL + Unsloth in Colab
- Loads and quantizes Llama-3.1-8B-Instruct
- Writes rollout function connecting to OpenEnv client
- Runs baseline + trained experiments
- Generates reward curves and before/after trajectories
- Handles correct LoRA model saving and inference testing

### Member D — Demo + Pitch
**Primary focus**: Storytelling and presentation
- Builds Gradio demo interface on HuggingFace Space
- Writes README, HuggingFace blog post / YouTube video
- Creates reward curve visualizations
- Designs pitch narrative (3-minute structure)
- Coordinates final submission checklist

### Parallel Work Strategy
```
Hours 1–4:   A + B build env/rewards in parallel | C sets up training stack | D drafts README structure
Hours 4–12:  A builds drift controller | B builds reward engine | C runs TRL test | D preps demo skeleton
Hours 12–20: A deploys to Space | B validates reward | C runs baseline | D builds Gradio UI
Hours 20–30: A adds multi-actor layer | C runs full training | D plots curves | B monitors for hacking
Hours 30–36: All converge on demo polish, README, pitch rehearsal
```

---

## 7. MVP DEFINITION

### MUST BUILD for a Working Demo

1. **DriftDesk OpenEnv environment** — at minimum 3 task modules, 2 schema versions each, fully compliant with `reset/step/state`
2. **SchemaDriftController** — drift events fire, agent receives schema-mismatch errors
3. **4-component reward function** — especially `drift_recovery` (this is the novel contribution)
4. **PolicyDocumentInjector** — policy text injected at episode start
5. **Baseline frozen model rollout** — quantifiable failure mode under drift
6. **GRPO training run** — even 100 episodes showing reward improvement
7. **Before/After trajectories** — concrete replay of baseline failing vs trained model adapting
8. **HuggingFace Space** — environment deployed and runnable by judges
9. **README** — problem, env description, results plots, all links

### Can Skip If Time-Constrained

- Multi-actor layer (Boss/Partner/Bank/Insurance conflicts) — mention in pitch as "future work"
- All 5 task modules — 3 is sufficient
- v3 schemas — 2 schema versions per task demonstrates drift adequately
- Full 500-episode training run — 100–150 episodes is enough to show a curve
- Gradio interactive demo — static before/after video clips in README are acceptable

### What Judges Need to See (In Order of Weight)
1. **Novel environment that doesn't exist anywhere else** (40%) ← This is confirmed novel
2. **Clear story: agents break when rules change, DriftDesk trains them not to** (30%)
3. **Reward curves + before/after trajectories showing measurable improvement** (20%)
4. **Coherent GRPO pipeline via TRL + Unsloth** (10%)

---

## 8. DEMO & PRESENTATION STRATEGY

### 3-Minute Pitch Structure

**Minute 0:00–0:45 — The Hook (The Problem)**
> "Every company deploying AI agents in 2026 is experiencing the same silent failure. The agent works perfectly in testing — then the bank changes its dispute API, and the agent breaks. Not because it's dumb. Because it was trained in a static world and deployed into a dynamic one. We call this schema drift. And until today, no RL training environment trained agents to handle it."

*Show*: Real production failure (LangChain 2025 incident quote)

**Minute 0:45–1:30 — The Environment**
> "DriftDesk is a personal executive assistant simulation. One agent, five real-world tasks, running across a simulated week. The twist: the schemas and policies governing those tasks silently change — mid-task, with no warning to the agent."

*Show*: Live `env.reset()` → policy doc injection → task list on screen

**Minute 1:30–2:15 — The Money Shot (Before/After)**
> "Watch what happens to a base model when the bank switches from 3-field form to JSON format mid-task."

*Show*: **Round 1** — base model loops on old form schema, gets 3 422 errors, fails.
*Show*: **Round 2** — DriftDesk-trained model receives 422 error, reads it, infers schema changed, switches to JSON, task completes.

*Show*: Reward curve — base model at 15–20% completion under drift. Trained model at 68–75%.

**Minute 2:15–2:45 — The Novel Contribution**
> "The key innovation is this reward signal." Point at `drift_recovery` component.
> "No existing reward function in any RL environment tracks whether an agent successfully identified and adapted to a mid-episode rule change. This is the first."

**Minute 2:45–3:00 — Why It Matters**
> "Every company shipping agentic AI — every bank, every insurer, every enterprise — trains their agents once and sends them into a world that changes. DriftDesk is the training environment for the real world. We're open-sourcing it today."

---

### Visual Elements for Demo
- **Schema diff viewer**: Side-by-side v1 vs v2 schema highlighting changed fields in red
- **Episode replay**: Step-by-step trace showing agent actions, environment responses, drift event marker
- **Reward curve**: Two lines on same axes — "Base Model" (flat/declining) vs "DriftDesk-Trained" (rising) across drift frequency spectrum
- **Task completion heatmap**: 5 tasks × drift levels, colored by success rate

---

## 9. RISK ANALYSIS

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Model never gets non-zero reward | Medium | High | Start with NO drift in curriculum, add gradually. Verify reward fires on manual test actions first |
| Reward hacking (agent memorizes schemas without generalizing) | Medium | High | Randomize which schema version is active per episode. Lock down global state in env |
| Training run too slow for Colab T4 | High | Medium | Use Unsloth 4-bit, max 8B model, short episodes (max 20 steps). Start with 50 episodes to verify loop |
| LoRA merge corrupts model weights | Low | High | Test inference immediately after save. Keep adapter + base separate until merge confirmed working |
| OpenEnv FastAPI deployment fails | Low | High | Deploy early (Phase 3). Keep local Uvicorn fallback. Test client-server separately |

### Time Risks

| Risk | Mitigation |
|------|------------|
| Drift controller takes longer than estimated | Simplify: only 1 drift event per episode, not continuous |
| Multi-actor layer cuts into training time | Cut it. Demo single-actor version. Mention it in pitch as "next step" |
| Reward curve doesn't show clear improvement | Run baseline first (0 training). Even a small improvement on drift recovery is novel |

### Backup Plans

1. **If training loop fails completely**: Show environment working + reward function firing correctly + manually crafted before/after trajectories. The environment innovation is 40% of judging.
2. **If HuggingFace Space doesn't deploy**: Run env locally with Uvicorn, screen-capture the demo.
3. **If model memorizes rather than generalizes**: Adjust reward — increase drift_recovery weight to 0.5, decrease task_completion to 0.3. Force randomized schema activation.

---

## 10. DIFFERENTIATION STRATEGY

### Why DriftDesk Wins Every Judging Category

| Criterion | Weight | DriftDesk Advantage |
|-----------|--------|---------------------|
| Environment Innovation | 40% | **The only environment in existence that trains schema-drift robustness.** Confirmed by survey of AppWorld, TextArena, EnterpriseArena, Calendar Gym, Schema First Tool APIs, PhantomPolicy — none train on drift |
| Storytelling | 30% | "AI agents break when the world changes. We train them not to." Concrete real-world failure (LangChain 2025) → clear mechanism → visual before/after |
| Reward Improvement | 20% | Stark quantitative gap: base model 15–20% → trained model 68–75% under same drift events. Visually clean on one chart |
| Training Pipeline | 10% | OpenEnv + TRL + GRPO + Unsloth — exact stack prescribed by hackathon guide |
| Patronus AI Bonus | Bonus | Exact match to their sub-theme: agents trained to handle policy-invisible schema changes |
| Halluminate Bonus | Bonus | Multi-actor layer (Boss/Partner/Bank/Insurance) = managing multiple competing actors to discover and achieve tasks |

### What Competitors Will Build
Most teams will build: email environments, calendar schedulers, code execution environments, game-playing agents. These are capability environments — teaching agents to do things they mostly already can do.

**DriftDesk is a robustness environment** — it teaches agents to survive a world that changes. This is the fundamental gap between lab performance and production deployment. No other team will have built this because no paper told them the gap exists. The research synthesis in your documents is the competitive advantage.

### The One-Line Differentiation
> *"Every other environment trains agents to do tasks. DriftDesk trains agents to keep doing tasks when the rules change."*

---

## 11. SUBMISSION CHECKLIST

- [ ] OpenEnv environment deployed to HuggingFace Space (URL submitted to judges)
- [ ] `openenv.yaml` manifest valid and present
- [ ] `reset()`, `step()`, `state()` implemented per OpenEnv base class spec
- [ ] Client/server separation maintained (clients never import server internals)
- [ ] No reserved tool names (reset, step, state, close) used for MCP tools
- [ ] Training script in Colab (Unsloth + TRL GRPO) — runnable by judges
- [ ] Real training evidence: loss + reward plots from actual run (committed as `.png` to repo)
- [ ] README: problem motivation, env explanation, results plots, all links
- [ ] HuggingFace blog post OR YouTube video < 2 minutes
- [ ] Before/after trajectories included in README or demo
- [ ] Wandb run link (or equivalent) for training metrics
- [ ] No large video files in HF Hub (use URL references only)

---

## 12. QUICK REFERENCE RESOURCES

- OpenEnv CLI: `openenv init` → scaffold environment
- TRL GRPO walkthrough: Mega Lecture 1:53:20–2:07:12 — https://www.youtube.com/watch?v=Jew4lhAiqnw&t=6800s
- Module 4 (Build Your Own Env): Workshop 43:45–50:20 — https://www.youtube.com/watch?v=1jU05MlENOI&t=2625s
- Module 3 (Deploy to Spaces): Mega Lecture 1:30:00–1:39:07 — https://www.youtube.com/watch?v=Jew4lhAiqnw&t=5400s
- Unsloth: 4-bit QLoRA setup, fits T4 GPU, 2x training speed

---

*This plan is execution-ready. Start with Phase 1. Do not deviate from the build order: Environment → Rewards → Deployment → Training → Demo.*
