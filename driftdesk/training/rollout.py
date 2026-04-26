"""
training/rollout.py — WebSocket session, prompt construction, action parsing,
and the GRPO reward function.

REWARD NOTE — shaped vs. reported reward
-----------------------------------------
The env server (RewardEngine.compute_episode_reward) computes a canonical
7-component reward used for eval reporting. The GRPO reward function here
uses a *shaped* reward (see SHAPED_W_* in config.py) that amplifies components
with high gradient variance (task_completion, drift_recovery) and omits
constants (policy_grounding is always 1.0; loop_penalty is always negative).
This shaping is intentional and documented. To compare training vs. eval
rewards, use the reward_components from the env observation — those are the
un-shaped env values.
"""
from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import websocket

from driftdesk.training.config import (
    WS_URL, CURRICULUM_STAGE, MAX_EPISODE_STEPS, MAX_WS_RETRIES,
    SHAPED_W_TC, SHAPED_W_DR, SHAPED_W_EFF, SHAPED_W_PS,
    FORMAT_BONUS, MODULE_BONUS_MATCH, MODULE_BONUS_AVAIL,
    MAX_SEQ_LEN,
)

# V1 schema hints — injected into the prompt so the model knows required fields
# without having to discover them via error responses (diagnosed fix for tc=0).
SCHEMA_HINTS: Dict[str, str] = {
    "airline_rebook": (
        '{"module": "airline_rebook", "payload": '
        '{"flight_id": "...", "passenger_name": "...", "new_date": "YYYY-MM-DD"}}'
    ),
    "bank_dispute": (
        '{"module": "bank_dispute", "payload": '
        '{"account_id": "...", "amount": 0.0, "merchant": "...", "description": "..."}}'
    ),
    "insurance_claim": (
        '{"module": "insurance_claim", "payload": '
        '{"claimant_id": "...", "incident_date": "YYYY-MM-DD", "amount": 0.0, "description": "..."}}'
    ),
}


# ---------------------------------------------------------------------------
# WebSocket session
# ---------------------------------------------------------------------------

class DriftDeskSession:
    """Thin WebSocket wrapper for one episode against the DriftDesk env server."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._ws = websocket.create_connection(WS_URL, timeout=timeout)

    def _rr(self, msg: dict) -> dict:
        self._ws.send(json.dumps(msg))
        raw = self._ws.recv()
        if not raw:
            raise ConnectionError("Empty WebSocket response from environment server")
        return json.loads(raw).get("data", {})

    def reset(self, seed: int | None = None, curriculum_stage: int | None = None) -> dict:
        data: dict = {}
        if seed is not None:
            data["seed"] = seed
        if curriculum_stage is not None:
            data["curriculum_stage"] = curriculum_stage
        return self._rr({"type": "reset", "data": data})

    def step(self, module: str, payload: dict) -> dict:
        return self._rr({"type": "step", "data": {"module": module, "payload": payload}})

    def close(self) -> None:
        try:
            self._ws.send(json.dumps({"type": "close"}))
            self._ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def obs_to_messages(result: dict) -> list:
    """Convert an env observation to a chat-template message list."""
    obs = result.get("observation", result)
    policy   = obs.get("policy_doc", "")
    tasks    = obs.get("tasks", [])
    last_res = obs.get("last_result", {})
    step     = obs.get("step_count", 0)

    pending = [t for t in tasks if not t.get("completed")]
    pending_str = "\n".join(
        f"  [{t['priority']}] {t['module']}: {t['description']}" for t in pending
    )
    pending_modules = [t["module"] for t in pending if t.get("module")]
    hints = [SCHEMA_HINTS[m] for m in pending_modules if m in SCHEMA_HINTS]
    schema_section = (
        "\nPayload schemas for pending tasks:\n" + "\n".join(f"  {h}" for h in hints) + "\n"
        if hints else ""
    )

    system = (
        "You are DriftDesk, an executive-assistant agent. You complete tasks by "
        "calling APIs. Reply with EXACTLY ONE JSON object on a single line, "
        "no prose, no markdown, no commentary. Schema:\n"
        '{"module": "<module_name>", "payload": {<fields>}}\n'
        "On a DRIFT error, copy missing_fields from the error into payload and retry. "
        "On a TRANSIENT_ERROR (http_status 500), retry with the SAME payload — do not change fields."
    )
    user = (
        f"Step {step}. Active policy:\n{policy[:600]}\n\n"
        f"Pending tasks (lower priority number = more urgent):\n{pending_str}\n"
        f"{schema_section}\n"
        f"Last result:\n{json.dumps(last_res)[:400]}\n\n"
        "Extract field values from the task descriptions and respond with ONLY the JSON action."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def obs_to_prompt(result: dict, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        obs_to_messages(result),
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Action parsing + generation
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Tuple[Optional[dict], bool]:
    """Extract a JSON action from model output. Returns (action, is_valid_json)."""
    text = text.strip().split("<|im_end|>")[0].strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end == 0:
        return None, False
    try:
        obj = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None, False
    if not isinstance(obj, dict) or "module" not in obj or "payload" not in obj:
        return obj, True
    return obj, True


def generate_action(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    eos_ids = [
        i for i in [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
        ]
        if i is not None and i >= 0
    ]
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=MAX_SEQ_LEN - max_new_tokens,
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# GRPO reward function
#
# Shaped reward formula (documented in config.py SHAPED_W_* constants):
#   shaped = W_TC * tc + W_DR * dr + W_EFF * eff + W_PS * ps
#
# This diverges from the env eval formula intentionally — see module docstring.
# ---------------------------------------------------------------------------

# Shared training log (appended by grpo_reward_fn; read by EarlyAbortCallback).
training_log: List[dict] = []


def make_grpo_reward_fn(model, tokenizer):
    """Return a GRPO reward function closed over the live model and tokenizer."""

    def grpo_reward_fn(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        rewards = []
        seed_list = kwargs.get("seed", [None] * len(completions))
        if not isinstance(seed_list, (list, tuple)):
            seed_list = [seed_list] * len(completions)

        for i, (completion, prompt) in enumerate(zip(completions, prompts)):
            reward = 0.0
            for attempt in range(MAX_WS_RETRIES + 1):
                sess = None
                try:
                    sess = DriftDeskSession(timeout=45.0)
                    rollout_seed = random.randint(0, 2**31 - 1)
                    result = sess.reset(seed=rollout_seed, curriculum_stage=CURRICULUM_STAGE)

                    action, is_valid_json = parse_action(completion)
                    shape_ok = (
                        isinstance(action, dict)
                        and "module" in action
                        and "payload" in action
                    )
                    partial = (FORMAT_BONUS * (0.5 + 0.5 * float(shape_ok))) if is_valid_json else 0.0

                    if not shape_ok:
                        reward = partial
                        break

                    result = sess.step(action["module"], action.get("payload", {}))

                    for _ in range(MAX_EPISODE_STEPS - 1):
                        if result.get("done"):
                            break
                        next_prompt = obs_to_prompt(result, tokenizer)
                        gen = generate_action(model, tokenizer, next_prompt)
                        next_action, _ = parse_action(gen)
                        if not (
                            isinstance(next_action, dict)
                            and "module" in next_action
                            and "payload" in next_action
                        ):
                            break
                        result = sess.step(next_action["module"], next_action.get("payload", {}))

                    episode_reward = float(result.get("reward") or 0.0)

                    # Shaped reward (amplifies differentiable components).
                    obs   = result.get("observation", result) if isinstance(result, dict) else {}
                    comps = (obs.get("reward_components") or {}) if isinstance(obs, dict) else {}
                    tc  = float(comps.get("task_completion", 0.0) or 0.0)
                    dr  = float(comps.get("drift_recovery", 0.0) or 0.0)
                    eff = float(comps.get("efficiency", 0.0) or 0.0)
                    ps  = float(comps.get("priority_score", 0.0) or 0.0)
                    shaped = SHAPED_W_TC * tc + SHAPED_W_DR * dr + SHAPED_W_EFF * eff + SHAPED_W_PS * ps
                    use_shaped = bool(result.get("done")) and (tc + dr + eff + ps) > 0.0
                    eff_episode_reward = shaped if use_shaped else episode_reward

                    # Module-match bonus: gradient signal before first tc=1 episode.
                    tasks_list   = (obs.get("tasks", []) or []) if isinstance(obs, dict) else []
                    task_modules = {t.get("module", "") for t in tasks_list if t.get("module")}
                    avail_mods   = set((obs.get("available_modules", []) or []) if isinstance(obs, dict) else [])
                    action_module = action.get("module", "") if isinstance(action, dict) else ""
                    module_bonus = (
                        MODULE_BONUS_MATCH if (action_module and action_module in task_modules)
                        else MODULE_BONUS_AVAIL if (action_module and action_module in avail_mods)
                        else 0.0
                    )

                    reward = eff_episode_reward + partial + module_bonus
                    training_log.append({
                        "reward": reward, "episode_reward": episode_reward,
                        "shaped": shaped if use_shaped else None,
                        "tc": tc, "dr": dr, "eff": eff, "ps": ps,
                        "partial": partial, "module_bonus": module_bonus,
                        "done": result.get("done"),
                    })
                    break

                except Exception as e:
                    if attempt < MAX_WS_RETRIES:
                        print(f"  [reward_fn] Retry {attempt + 1} for completion {i}: {e}")
                        time.sleep(2.0 * (attempt + 1))
                    else:
                        print(f"  [reward_fn] Error for completion {i}: {e}")
                finally:
                    if sess:
                        sess.close()

            rewards.append(reward)

        # Batch diagnostics
        nz = sum(1 for r in rewards if r > FORMAT_BONUS + 1e-4)
        recent = [t for t in training_log[-len(rewards):] if isinstance(t, dict)]
        tc_mean = sum(t.get("tc", 0.0) for t in recent) / max(len(recent), 1)
        dr_mean = sum(t.get("dr", 0.0) for t in recent) / max(len(recent), 1)
        mb_mean = sum(t.get("module_bonus", 0.0) for t in recent) / max(len(recent), 1)
        print(
            f"  [reward_fn] batch n={len(rewards)} "
            f"mean={sum(rewards)/max(len(rewards),1):.4f} "
            f"min={min(rewards):.3f} max={max(rewards):.3f} "
            f"scored={nz}/{len(rewards)} "
            f"tc={tc_mean:.3f} dr={dr_mean:.3f} mb={mb_mean:.3f}",
            flush=True,
        )
        return rewards

    return grpo_reward_fn
