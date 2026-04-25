# DriftDesk — Critical Review & Optimized Execution Plan
*Audit target: `DriftDesk_Hackathon_Plan.md` (OpenEnv Hackathon India 2026, Theme 3.2)*
*Review date: 24 April 2026*

---

## 1. Executive Summary

**Problem (as stated).** Production LLM agents are trained against static API schemas and policies. When real-world schemas/policies drift (field added, endpoint replaced, policy rewritten), agents silently fail. No OpenEnv-compatible RL environment today trains for *mid-episode schema/policy drift*.

**Proposed solution.** `DriftDesk` — an OpenEnv personal-assistant environment with 5 task modules (airline rebook, bank dispute, insurance claim, subscription cancel, expense report), each with 3 schema versions. A `SchemaDriftController` silently swaps schemas within/between episodes. A 4-component reward (task completion + drift recovery + priority + efficiency) trained with GRPO/TRL + Unsloth on Llama-3.1-8B-Instruct.

**Key innovation (claimed).** The `drift_recovery` reward component — credit assignment specifically for *detecting an error, inferring the schema changed, and adapting the call*. Framed as the first RL training environment (not just benchmark) for schema/policy drift robustness.

**Verdict (short).** **Pursue with modifications.** The framing is genuinely strong and differentiated for a hackathon; the core design is coherent; however, several assumptions about novelty, feasibility on a T4, and reward-hackability are **unverified or optimistic** and must be hardened before build. See §9.

---

## 2. Original Plan Breakdown

### 2.1 Key Components (as specified)
1. `DriftDeskEnvironment` (OpenEnv `Environment` subclass, FastAPI).
2. `SchemaDriftController` — stochastic schema-version swaps, inter- and intra-episode.
3. `PolicyDocumentInjector` — natural-language policy text at episode reset.
4. 5 task modules × 3 schema versions.
5. 4-component reward: `0.5·task + 0.3·drift_recovery + 0.1·priority + 0.1·efficiency − 0.1·loop_penalty`.
6. Training: Llama-3.1-8B-Instruct, QLoRA 4-bit, GRPO via TRL, Unsloth, Colab T4.
7. HF Space deployment + Gradio demo + before/after trajectories + reward curves.
8. Optional multi-actor layer (Boss/Partner/Bank/Insurance).

### 2.2 Core Assumptions (extracted)
| # | Assumption | Status |
|---|------------|--------|
| A1 | No existing OpenEnv / RL env trains on schema drift | **Plausible but unverified** (§5) |
| A2 | Llama-3.1-8B-Instruct + QLoRA + GRPO fits Colab T4 for 100–500 rollouts | **Optimistic** (§3.4, §4) |
| A3 | Base model will occasionally succeed under drift, giving non-zero reward to learn from | **Uncertain, possibly false for full-drift curriculum** |
| A4 | `drift_recovery` counter meaningfully captures "understanding" rather than retry luck | **Shaky — gameable** (§3.2) |
| A5 | 5 modules × 3 schemas hand-authored in ~2 h is realistic | **Optimistic** |
| A6 | Judges will recognise "schema drift" framing as novel | **Plausible** but depends on pitch |
| A7 | 36-hour linear schedule holds | **Optimistic for 4-person team** |

---

## 3. Critical Issues Identified

### Issue 1 — "First-ever" novelty claim is only *partially* defensible
- **Why it matters.** Environment Innovation is 40% of scoring and the entire pitch leans on "the first." If a judge can name one counter-example (e.g., **ToolSandbox** by Apple, **τ-bench / tau-bench** by Sierra/Anthropic, **AppWorld**'s API-version variants, **WebArena/VisualWebArena**'s site-change variants, **SWE-Gym**'s dependency drift, **API-Bank**, **ToolBench**, or **NESTFUL**), the headline collapses.
- **Truthfulness note.** I have **not verified** that none of these train (not merely benchmark) on *mid-episode schema drift*. Claims like "no paper in 2025/2026 trains for this" and "PhantomPolicy Apr 2026" are in the plan without citations — treat as **unverified** until checked.
- **Risk level: HIGH.**
- **Fix.** (a) Before submission, **cite explicitly** 3–5 nearest neighbours and state a single-sentence differentiator against each in the README. (b) Soften the claim from *"the first RL environment"* to *"the first OpenEnv-native environment that treats mid-episode schema/policy drift as the primary training signal with a dedicated drift-recovery reward."* This is defensible and still strong.

### Issue 2 — `drift_recovery` reward is gameable and under-specified
- **Why it matters.** The entire novelty hinges on this component. As written (`successful_recoveries / max(drift_events, 1)`), it rewards any *post-error success*, not actual reasoning about drift. A model can learn "on 422, retry with a different template from memory" without inferring the schema.
- **Risk level: HIGH.**
- **Fix.**
  - Decompose into: (i) *error-acknowledgement* (did the agent reference the error payload in its next chain-of-thought or action diff?); (ii) *minimal-edit recovery* (did it change only the fields the error named, not a full template swap?); (iii) *first-try success after drift* vs *brute-force retry* (cap credit at $e^{-k·retries}$).
  - Use **held-out schema versions** at eval: train on v1↔v2 swaps, test on v1↔v3 and a brand-new v4 generated programmatically. Only generalisation-under-drift should score.
  - Add an **adversarial "fake drift" probe**: sometimes the error is a *transient 500*, not a schema change. Penalise agents that mutate their template unnecessarily. This alone kills the naive "always switch template on error" exploit.

### Issue 3 — Reward-weighting makes task completion dominant, diluting the novel signal
- **Why it matters.** With `0.5·task + 0.3·drift_recovery`, an agent that ignores drift and solves the 40–60% of episodes *without* drift events can score reasonably well; gradient signal for the novel behaviour is weak.
- **Risk level: MEDIUM.**
- **Fix.** Condition weights on drift occurrence: on drift episodes use `0.2·task + 0.6·drift_recovery + 0.1·priority + 0.1·efficiency`; on clean episodes keep the current weights. Also ensure **≥60% of training episodes contain at least one drift event** once curriculum has ramped.

### Issue 4 — Colab T4 + Llama-3.1-8B + GRPO is a tight squeeze
- **Why it matters.** T4 has 16 GB. 8B QLoRA inference + GRPO rollouts + reference model + gradients is *barely* feasible; long rollouts (multi-turn agent with 20 steps and tool-call JSON) can blow context and VRAM. Unsloth helps, but GRPO needs K samples per prompt (typically 4–8), multiplying rollout cost.
- **Risk level: HIGH (feasibility).**
- **Fix.** (a) Default model: **Qwen2.5-3B-Instruct** or **Llama-3.2-3B-Instruct** with QLoRA; keep 8B as stretch goal only if an A100/L4 becomes available. (b) Cap episode length at 10 steps; truncate tool responses to ≤512 tokens. (c) Pre-validate: run 10 rollouts end-to-end *before* Phase 3 training to measure seconds/rollout and VRAM; budget backwards from 4 h of training.

### Issue 5 — Warm-start / cold-start of GRPO not planned
- **Why it matters.** Per the self-serve guide and common GRPO practice, if base rollouts produce ~0 reward, GRPO has no advantage signal. A raw 3B/8B instruct model calling hand-crafted JSON APIs may simply not conform to the action format, yielding near-zero rewards and flat curves.
- **Risk level: HIGH.**
- **Fix.** (a) Design a *format-only* reward component that fires on valid JSON/tool-call syntax even when the semantic call fails — this gives early gradient. (b) Optional short SFT warm-up (a few hundred synthetic `(observation, correct-action)` pairs generated by a stronger model / by running the env under ground-truth) before GRPO. (c) Curriculum starts with **zero drift and 1 task** for the first 30–50 steps.

### Issue 6 — "Mid-episode, silent drift" can be epistemically unfair
- **Why it matters.** If a schema swaps between `env.step` calls with zero observable cue, there is no valid inference path — the agent *must* wait for an error. That reduces "drift recovery" to "retry on error," which is trivial and not the intended capability.
- **Risk level: MEDIUM.**
- **Fix.** Provide *weak cues* the agent can, in principle, learn to pick up: (i) a version header or deprecation warning intermittently in responses (ii) policy-doc updates mid-episode via an "inbox" message (iii) a deprecated-field warning flag. Drift without cues becomes a *hard* track; drift with cues becomes the *learnable* track. Train on the cued track; evaluate on both.

### Issue 7 — 5 task modules × 3 schema versions is more content than it looks
- **Why it matters.** 15 distinct, coherent, mutually-consistent schemas + their validation logic + their policy text is closer to 8–12 h of careful work, not 2 h. The plan under-budgets this.
- **Risk level: MEDIUM.**
- **Fix.** MVP slice: **3 modules × 2 schemas = 6 schemas**. Generate the "v3" variants programmatically (field rename, field add, endpoint split) from a small DSL — this also yields free held-out schemas for eval.

### Issue 8 — "Multi-actor layer" is scope creep under a 36-h budget
- **Why it matters.** Plan acknowledges it's optional but assigns Phase 4 time to it while training is still running. Competing actors with priority conflicts is a separate research problem (Theme 1), not a free add-on.
- **Risk level: MEDIUM.**
- **Fix.** Ship the multi-actor layer as a **static priority list in the observation** (e.g., "Boss task is P0, others P1"). Do not implement asynchronous actors. Mention fuller multi-actor as "future work." Claim the Halluminate bonus via the priority reward component, not a new subsystem.

### Issue 9 — Environment security / sandboxing under-specified
- **Why it matters.** The 5 tasks call "APIs" which are really Python handlers. If the model emits arbitrary strings that are `eval`'d or fed to templating, that's an injection surface — low risk at hackathon but the guide explicitly calls out reward hacking via globals / cached state.
- **Risk level: LOW-MEDIUM.**
- **Fix.** All task handlers are pure functions over validated Pydantic models; no `eval`, no `exec`; episode state is passed by value; add a `_validate_no_private_fields` check that rejects actions referencing underscored keys.

### Issue 10 — Reward curves comparability
- **Why it matters.** "Base model 15–20% vs trained 68–75%" is stated as expected result; if training underperforms, the story breaks. The guide weighs *showing improvement* at 20%.
- **Risk level: MEDIUM.**
- **Fix.** Commit to reporting *whatever actually happens* with multiple slices: (i) no-drift task completion (sanity), (ii) drift task completion, (iii) drift-recovery rate, (iv) unnecessary-edits rate (anti-hack). Even a modest delta on (iii) and (iv) with a flat (i) is a publishable result. Pre-register this evaluation in the README.

### Issue 11 — `loop_penalty = 0.1 * repeated_failed_calls` can go unboundedly negative
- **Why it matters.** `max(0.0, ...)` clamp at the end means loop penalty silently saturates; gradient disappears. Also `0.1 * N` for arbitrary N dominates other components.
- **Risk level: LOW.**
- **Fix.** Use `loop_penalty = min(0.3, 0.05 * repeated_failed_calls)` and apply *inside* the weighted sum, not after clamp.

### Issue 12 — Policy document injection needs a language-grounding reward
- **Why it matters.** If nothing rewards reading the policy doc, the agent will ignore it. The plan treats it as observation only.
- **Risk level: MEDIUM.**
- **Fix.** Add a small `policy_grounding` term: for episodes where policy explicitly mentions a required field, violating that field zeroes `task_completion` even if the API accepts it (because a stricter v+1 schema is "about to land"). Trains reading-then-acting.

---

## 4. Stress Test Results

| # | Scenario | Failure Point | Severity | Fix |
|---|----------|---------------|----------|-----|
| S1 | Drift fires on step 1; agent has zero error history to learn from | `drift_recovery` numerator always 0, denominator 1 → 0 reward; signal collapses | High | Guarantee ≥1 non-drift step before first drift event per episode |
| S2 | Drift-hacking exploit: agent learns "on ANY error, enumerate 3 known templates" | `drift_recovery` rises without understanding; fails on held-out v4 | High | Adversarial transient errors (Issue 2); held-out schema eval |
| S3 | Rollout timeout: 8B model at ~5 tok/s × 20 steps × 8 GRPO samples × 100 episodes = hours per step | Training never finishes in 4 h budget | High | Switch to 3B (Issue 4); cap steps at 10; K=4 |
| S4 | Judge asks "show me this problem in the wild" | LangChain 2025 anecdote is vague; no citation in README | Medium | Add 2 concrete linked incidents in README (Stripe/Plaid API deprecations, any documented LangChain upgrade break) |
| S5 | HF Space cold-starts, FastAPI times out during judge run | Judges fail to reproduce; 10% penalty soft-risk | Medium | Add `/healthz`; warm-ping cron; provide a pre-recorded Colab fallback |
| S6 | LoRA merge on 4-bit base corrupts weights (guide explicitly warns) | Demo model produces garbage, story dies | High | Save adapter separately; demo by loading adapter on 4-bit base; only merge as last step, re-verify inference |
| S7 | Reward gets stuck at the format-reward component (agent emits well-formed but semantically wrong JSON forever) | Curves plateau, no "adaptation" story | Medium | Cap format reward at low value (≤0.05); anneal to 0 after 50 steps |
| S8 | Judge tries an adversarial task the env can't handle (e.g., "what if user asks for refund in INR on a USD schema?") | Env crashes → unreproducible | Medium | Whitelist actions; every unknown action returns a structured error, never 500 |
| S9 | Training shows reward ↑ but qualitative behaviour is worse (pure reward hacking) | Storytelling collapses under scrutiny | High | Pre-commit to showing N=10 qualitative traces in README regardless of reward number |
| S10 | Wandb free tier rate-limits mid-run | Lose plots | Low | Mirror metrics to CSV every 10 steps; generate plots from CSV at end |
| S11 | Policy doc is long; blows Llama 3B context when concatenated with 10 observations | OOM mid-episode | Medium | Policy doc capped at 400 tokens; observations truncated; use system prompt channel |
| S12 | Two teammates edit schema registry concurrently; versions desync | Silent eval bugs | Low | Single source of truth: one `schemas.py` with dataclasses; PR-gated on Phase 2 |

---

## 5. Research & Competitive Analysis

> All comparisons below are based on publicly known work as of early 2026 and are stated to the best of my knowledge. **Treat any claim of novelty relative to a specific paper as "uncertain until verified"** — the team must double-check each before submission.

### 5.1 Nearest neighbours (likely to be cited by a judge)
| System | What it does | How DriftDesk is different (claimed) |
|--------|--------------|----------------------------------------|
| **τ-bench (tau-bench)** by Sierra/Anthropic | Multi-turn tool-use benchmark with a user simulator and a policy document | τ-bench is a **benchmark with static policies**; DriftDesk *trains on mid-episode policy/schema mutations as the primary signal* |
| **AppWorld** (ICLR '24) | 9 apps, 457 API endpoints, compositional tool use; RL-trainable | AppWorld APIs are fixed per episode; DriftDesk mutates them |
| **ToolSandbox** (Apple) | Stateful, multi-turn tool use with state transitions | State transitions ≠ schema drift; tool *definitions* are fixed |
| **WebArena / VisualWebArena** | Realistic web tasks | Site UI is static per snapshot; no drift training signal |
| **API-Bank / ToolBench / NESTFUL** | Tool-use benchmarks | Static schema benchmarks, not RL training envs with drift |
| **SWE-Gym / SWE-bench-Live** | Real GitHub-issue fixing; version drift is incidental | Drift is a side-effect of the dataset, not a controlled training variable |
| **"Schema First Tool APIs" (Mar 2026)** (cited in plan) | Tool-misuse evaluation | Static benchmark — plan's own gap analysis |
| **"PhantomPolicy" (Apr 2026)** (cited in plan) | Policy-invisible violations | Static per-run policies — plan's own gap analysis |

### 5.2 Defensible novelty claim (rewritten)
> *"DriftDesk is, to our knowledge, the first OpenEnv-compliant environment in which the primary training signal rewards an agent for detecting and adapting to mid-episode schema and policy mutations, with a decomposed `drift_recovery` reward and held-out drift patterns for generalisation evaluation."*

This is narrower, defensible, and still 40%-tier innovative.

### 5.3 Risks of being "scooped" or dismissed
- **Big Tech production tools** (LangChain Hub, Anthropic's tool-use evals, OpenAI's Evals) likely *monitor* for schema breaks but do not publish RL training envs for it — low scoop risk.
- **Academic.** A paper on "robust tool use under API evolution" *could* exist that the team hasn't found. **Action**: one 1-hour arXiv + Google Scholar sweep with queries: `"tool use" AND (drift OR evolution OR versioning) AND (reinforcement OR RL OR GRPO)` before freezing the novelty claim.
- **OpenEnv ecosystem.** Check HF Spaces tagged `openenv` for any env with "drift", "version", "schema-change" in description — do this before submission.

### 5.4 Verdict on novelty
**Plausibly novel with softened wording; high-risk if claim is kept at "first ever in existence."**

---

## 6. Improvements & Optimisations

### 6.1 Architectural upgrades
1. **Schema DSL + generator.** Replace hand-crafted JSON files with a small dataclass DSL: `Field(name, type, required, version_introduced, version_deprecated)`. Schemas materialise from the DSL. This (a) makes v3/held-out versions 10 lines each, (b) lets the reward compute *exactly which field-level edit* the agent needed to make, enabling fine-grained `drift_recovery`.
2. **Drift taxonomy.** Canonicalise drift types: `FIELD_ADD | FIELD_RENAME | FIELD_REMOVE | TYPE_CHANGE | ENDPOINT_MOVE | FLOW_RESTRUCTURE`. Log per-type recovery rates — this becomes a second plot that is *genuinely* informative and distinguishes the project.
3. **Error schema standardisation.** All schema-mismatch errors carry a structured body: `{code: "DRIFT", changed_fields: [...], hint?: "..."}`. Without `hint`, it is the hard track; with `hint`, the easy track. Train on easy→hard curriculum, evaluate on both.
4. **Separate reference-model cache.** GRPO needs reference logprobs — cache them; do not re-run reference model each step (Unsloth can help but verify).
5. **Deterministic eval set.** Freeze 50 eval episodes with a fixed seed and fixed drift schedule at Phase 2. Evaluate every N training steps on this exact set. This is what the reward curve is plotted against — not training rollouts (which are noisy).

### 6.2 Reward upgrades (replacing plan's block)
- `task_completion` (weight 0.25 on drift eps, 0.5 on clean eps)
- `drift_recovery_decomposed` (weight 0.45 on drift eps, 0 on clean eps) = average of:
  - `error_grounded_edit` ∈ {0,1}: next action edits precisely the field(s) named in the error
  - `first_retry_success` ∈ {0,1}: succeeds within 1 retry after drift
  - `no_spurious_rewrite` ∈ {0,1}: on transient/fake errors, agent does *not* mutate its template
- `policy_grounding` (weight 0.1): agent respects fields mentioned only in policy doc
- `priority_score` (weight 0.1)
- `efficiency` (weight 0.1)
- `format_valid` (annealed, starts 0.05, decays to 0 by step 50)
- `loop_penalty` (capped at −0.3)

### 6.3 Training upgrades
- **Default model: Qwen2.5-3B-Instruct** (QLoRA, r=16). Keep 8B as ambition tier.
- **Warm start (optional, 15 min): 200 SFT pairs** of (obs, correct action) synthesised by running the env with a rule-based oracle. Dramatically raises base-model success-probability → GRPO actually learns.
- **GRPO config**: K=4 samples/prompt, 10-step episodes, 64 prompts/step, LR 5e-6, 150 training steps.
- **Baselines to report**: (i) frozen base; (ii) frozen base + CoT prompt; (iii) frozen base + policy doc prefix; (iv) DriftDesk-trained. (ii) and (iii) are "free" baselines that make the delta credible.

### 6.4 Schedule upgrades
- **Parallelise Phases 1–3.** Member C must *not* wait for env to finish. Give them a `DummyDriftEnv` stub (same interface, random rewards) on Hour 1; they debug the TRL↔OpenEnv wiring while A/B build the real env.
- **Kill-switch at Hour 20.** If `reward ↑` is not observed on one training run by Hour 20, freeze env, cut multi-actor, run 3B-model with hand-crafted SFT warm-up. Pre-decided.
- **Phase 0 (Hour 0–1): arxiv + HF-Spaces sweep** to confirm novelty framing *before* coding.

### 6.5 Storytelling upgrades
- Open the pitch with a **real, cited** incident (LangChain v0.2, Plaid API v2 migration, a Stripe deprecation) with a screenshot, not an anecdote.
- Show the **drift taxonomy heatmap** (6 drift types × recovery rate) — this is a visual no competitor will have.
- Include a **failure case for the trained model** in the README. Judges reward honesty; it prevents "too good to be true" skepticism.

---

## 7. Final Optimised Plan (Rewritten)

### 7.1 Revised problem statement
Production LLM agents fail when the tools/policies they were trained against change. We build `DriftDesk`, an OpenEnv RL environment whose *primary* training signal is adaptation to mid-episode schema and policy mutations, with a decomposed, anti-hackable reward and held-out drift types for generalisation evaluation.

### 7.2 Scope (hard-frozen)
**MVP (must ship):**
- 3 task modules × 2 schema versions + 1 programmatically generated held-out schema per module (= 9 schemas total).
- 6 canonical drift types; each episode can trigger 0–2 drift events from the taxonomy.
- Decomposed reward (§6.2) with annealed format reward.
- Qwen2.5-3B-Instruct + QLoRA 4-bit + GRPO via TRL + Unsloth.
- 50-episode deterministic eval set frozen at Hour 12.
- 4 baselines + 1 trained model; before/after qualitative traces (≥10 per slice).
- HF Space deployment; README with taxonomy heatmap and pre-registered metrics.

**Stretch (only after Hour 24 kill-switch check passes):**
- 5th task module; 3rd schema version.
- Static priority-conflict observation (Halluminate bonus framing).
- Gradio interactive replay.

**Explicitly out of scope:**
- Asynchronous multi-actor simulation.
- 8B model.
- Wandb (use CSV + matplotlib; Wandb only if trivial).

### 7.3 Architecture (unchanged + DSL + error schema)
```
Schema DSL ──▶ SchemaRegistry (v1, v2, held-out)
                    │
PolicyInjector ──▶  │  ── DriftController (typed events) ──▶  TaskModules
                    │                                              │
                    └──▶ Observation builder ◀────────── Structured error bodies
                                       │
                                   Reward Engine (7 components, episode-type-conditioned)
                                       │
                          OpenEnv FastAPI (reset/step/state)
                                       │
                 ┌─────────────────────┴──────────────────────┐
        DummyDriftEnv (for TRL wiring)              Real env (for training/eval)
                                       │
                         TRL GRPOTrainer + Unsloth + Qwen2.5-3B QLoRA
                                       │
                        Deterministic eval harness (50 episodes)
                                       │
                                HF Space + Gradio
```

### 7.4 Revised 36-hour schedule
| Hours | A (Env) | B (Reward+Verify) | C (Training) | D (Demo+Story) |
|-------|---------|-------------------|--------------|----------------|
| 0–1   | arXiv/HF novelty sweep (all hands) | ↑ | ↑ | ↑ |
| 1–4   | OpenEnv scaffold, Schema DSL, 3 modules v1 | Reward skeleton + unit tests + `DummyDriftEnv` stub | TRL+Unsloth Colab, Qwen2.5-3B load, dummy rollout | README skeleton, incident research |
| 4–10  | v2 schemas, held-out generator, DriftController, structured errors | Decomposed reward impl + anti-hacking probes + policy grounding | SFT warm-up data generation; first GRPO run on stub | Gradio shell, drift-taxonomy graphic draft |
| 10–16 | Integrate reward, freeze 50-ep eval set, local end-to-end | Eval harness, baseline rollouts (4 variants) | Kick full GRPO run on real env (3B, 150 steps) | Plots pipeline from CSV |
| 16–24 | Deploy HF Space, `/healthz`, warm-ping | Monitor rollouts for hacking, spot-check 20 traces | Full training; mid-run qualitative inspection | Record before/after clips |
| 24    | **KILL-SWITCH GATE** — see §7.5 | | | |
| 24–30 | Optional: 4th module / priority obs | Finalise eval report | Final training run w/ best hyperparams | Record video, write blog |
| 30–36 | Docker re-test from Space | Anti-hacking final audit | Adapter save + inference smoke test | README polish, submission checklist |

### 7.5 Kill-switch gate at Hour 24
Proceed to stretch goals **only if all four hold**:
1. Trained model beats best baseline on `drift_recovery_decomposed` by ≥ 10 absolute points on eval set.
2. No qualitative evidence of reward hacking in 20 sampled traces.
3. HF Space responds to `reset/step` in <2 s p95.
4. Adapter save reproduces metric within ±2 pts on reload.

If any fail: cut stretch, spend remaining hours on story, additional baselines, and failure-mode documentation.

### 7.6 Pre-registered evaluation (commit to README before training)
Report on the 50-episode deterministic eval set:
- Task completion (no-drift slice)
- Task completion (drift slice)
- `drift_recovery_decomposed` (overall + per drift-type heatmap)
- `no_spurious_rewrite` rate (anti-hack metric)
- Generalisation: same metrics on held-out drift type not seen in training
- Qualitative: 5 success traces, 5 failure traces, commentary

---

## 8. Risk Assessment (Residual)

| Residual risk | Likelihood | Impact | Mitigation |
|---------------|------------|--------|------------|
| Novelty claim contested by a judge citing τ-bench / AppWorld variant | Med | High | Softened claim in §5.2; prepared comparison table in README |
| GRPO on 3B still doesn't produce clean reward curve in 4 h | Med | High | SFT warm-up + format reward + curriculum; kill-switch at H24 |
| `drift_recovery` still game-able despite decomposition | Low-Med | High | Held-out drift + `no_spurious_rewrite` + qualitative audit |
| HF Space image build fails day-of | Low | High | Deploy by H16; keep local Uvicorn + pre-recorded demo |
| Team coordination breaks (single-thread bottlenecks on env) | Med | Med | `DummyDriftEnv` stub from H1 decouples C; weekly-style standup each 6 h |
| Schema DSL overengineered, consumes Phase 2 | Med | Med | Cap DSL at 80 LOC; if exceeded, revert to hand JSON |
| Judges miss the nuance of the innovation | Med | Med | Lead pitch with *concrete cited incident*, not abstract framing |

---

## 9. Verdict

**Recommendation: PURSUE with the modifications in §3, §6, §7.**

**Why pursue.** The framing — *robustness to drift as a first-class RL training variable* — is genuinely underexplored in the public OpenEnv/TRL ecosystem, fits Theme 3.2 cleanly, lands a plausible Patronus/Halluminate bonus, and is concrete enough to ship in 36 h. The reward design is reasonable and the stack (OpenEnv + TRL + Unsloth + GRPO) is exactly what the self-serve guide prescribes.

**Why modify.** As originally written, the plan (a) **overclaims novelty** without citations, (b) specifies a **gameable** `drift_recovery` signal, (c) **under-budgets compute** (8B on T4), (d) **under-plans the cold-start** problem that has killed many hackathon GRPO runs, (e) **over-scopes** content (5×3 schemas, multi-actor) for the timebox, and (f) **mixes wishful numbers** (68–75% trained vs 15–20% base) into the pitch before any training has run.

**Do not drop.** The project's core thesis — *agents should learn to survive a changing world* — is the kind of ambitious, paper-adjacent framing the judging guide explicitly says it rewards over polished-but-boring submissions. With the decomposed reward, the held-out drift set, the SFT warm-up, and the 3B default, this goes from "pitch that might collapse under scrutiny" to **a defensible submission with a genuine story arc**.

**One-line executive call.** Keep the idea, narrow the claim, shrink the content, harden the reward, pick a smaller model, and commit to a pre-registered evaluation — then build.
