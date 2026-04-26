"""
Deterministic evaluation harness for DriftDesk.

Generates N=50 fixed eval episodes (seeded) and evaluates a model or
rule-based agent against them. Outputs per-component reward breakdown and
per-drift-type recovery rates.

Usage:
    python eval_harness.py --env-url http://localhost:8000 --agent rule_based
    python eval_harness.py --env-url http://localhost:8000 --agent <model>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional

from driftdesk.client import DriftDeskClient

EVAL_SEEDS = list(range(1000, 1050))   # 50 deterministic seeds
N_EVAL = len(EVAL_SEEDS)


# ---------------------------------------------------------------------------
# Rule-based oracle agent (for baseline and SFT warm-up data generation)
# ---------------------------------------------------------------------------

class RuleBasedAgent:
    """Simple agent: tries v1 schema first; on DRIFT error adapts to error body."""

    # v1 payloads
    V1_PAYLOADS = {
        "airline_rebook": {
            "flight_id": "AI-202",
            "passenger_name": "Jordan Lee",
            "new_date": "2026-05-10",
        },
        "bank_dispute": {
            "account_id": "ACC-9921",
            "amount": 450.0,
            "merchant": "FastCharge Inc",
            "description": "Unauthorized charge",
        },
        "insurance_claim": {
            "claimant_id": "CLM-5501",
            "incident_date": "2026-04-15",
            "amount": 1200.0,
            "description": "Laptop damaged in flood",
        },
    }

    # Canonical field fill-ins for v2+ fields
    V2_EXTRA = {
        "airline_rebook": {"reason_code": "WORK"},
        "bank_dispute": {"dispute_type": "FRAUD"},
        "insurance_claim": {
            "incident_id": "INC-9001",
            "line_items": [{"code": "LAPTOP", "cost": 1200.0}],
        },
    }

    def __init__(self) -> None:
        self._payloads: Dict[str, Dict[str, Any]] = {}
        self._pre_authed: Dict[str, bool] = {}

    def reset(self, tasks: List[Dict[str, Any]]) -> None:
        self._payloads = {mod: dict(self.V1_PAYLOADS[mod]) for mod in self.V1_PAYLOADS}
        self._pre_authed = {}

    def act(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        tasks = obs.get("observation", obs).get("tasks", obs.get("tasks", []))
        last_result = obs.get("observation", obs).get("last_result", obs.get("last_result", {}))
        last_module = last_result.get("module")

        # Adapt payload on DRIFT error
        if last_result.get("code") == "DRIFT" and last_module:
            missing = last_result.get("missing_fields", [])
            extra = self.V2_EXTRA.get(last_module, {})
            for field in missing:
                if field in extra:
                    self._payloads[last_module][field] = extra[field]
                elif field == "pre_auth_token" and self._pre_authed.get(last_module):
                    pass  # token already obtained, will be added below
            # If pre_auth needed, handle it
            if "pre_auth_token" in missing and not self._pre_authed.get(last_module):
                return {"module": f"{last_module}:pre_auth", "payload": {}}

        # Remove fields that cause unexpected_fields errors
        if last_result.get("code") == "DRIFT" and last_module:
            unexpected = last_result.get("unexpected_fields", [])
            for f in unexpected:
                self._payloads[last_module].pop(f, None)

        # Pick next pending task
        for task in sorted(tasks, key=lambda t: t.get("priority", 99)):
            if not task.get("completed"):
                mod = task["module"]
                payload = dict(self._payloads.get(mod, {}))
                return {"module": mod, "payload": payload}
        return None  # all tasks done

    def record_pre_auth(self, module: str, token: str) -> None:
        self._pre_authed[module] = True
        if module in self._payloads:
            self._payloads[module]["pre_auth_token"] = token


# ---------------------------------------------------------------------------
# Frozen-LLM agent (real before/after baseline; lazily-loaded so the rule_based
# path keeps working without torch/transformers installed)
# ---------------------------------------------------------------------------

class FrozenLLMAgent:
    """Wraps a HuggingFace causal LM behind the same act() interface.

    Uses Qwen native chat template + JSON-only system prompt + EOS stops.
    Loads the base model in 4-bit; if `adapter_path` is given, attaches a PEFT
    adapter on top so we can reuse the same harness for both baselines and the
    trained model.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
    ) -> None:
        import torch  # local import
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        if adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(base, adapter_path)
        else:
            self._model = base
        self._model.eval()
        self._max_new = max_new_tokens
        self._temp = temperature
        im_end = self._tok.convert_tokens_to_ids("<|im_end|>")
        self._eos = [i for i in [self._tok.eos_token_id, im_end] if i is not None and i >= 0]

    def reset(self, tasks: List[Dict[str, Any]]) -> None:
        return None

    def _messages(self, obs: Dict[str, Any]) -> List[Dict[str, str]]:
        policy = obs.get("policy_doc", "")
        tasks = obs.get("tasks", [])
        last_result = obs.get("last_result", {})
        step = obs.get("step_count", 0)
        pending = [t for t in tasks if not t.get("completed")]
        pending_str = "\n".join(
            f"  [{t['priority']}] {t['module']}: {t['description']}" for t in pending
        )
        system = (
            "You are DriftDesk. Reply with EXACTLY ONE JSON object on a single line.\n"
            '{"module": "<module_name>", "payload": {<fields>}}\n'
            "On DRIFT errors, copy missing_fields into payload. "
            "On TRANSIENT_ERROR, retry with the same payload."
        )
        user = (
            f"Step {step}. Policy:\n{policy[:600]}\n\n"
            f"Pending tasks (lower priority = more urgent):\n{pending_str}\n\n"
            f"Last result:\n{json.dumps(last_result)[:400]}\n\n"
            "Respond with ONLY the JSON action."
        )
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    def act(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        import torch
        outer = obs.get("observation", obs)
        prompt = self._tok.apply_chat_template(
            self._messages(outer), tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tok(prompt, return_tensors="pt", truncation=True,
                           max_length=2048 - self._max_new).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self._max_new,
                do_sample=self._temp > 0,
                temperature=max(self._temp, 1e-5),
                top_p=0.95,
                pad_token_id=self._tok.eos_token_id,
                eos_token_id=self._eos,
            )
        text = self._tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=False)
        text = text.split("<|im_end|>")[0].strip()
        s, e = text.find("{"), text.rfind("}") + 1
        if s == -1 or e == 0:
            return None
        try:
            obj = json.loads(text[s:e])
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict) or "module" not in obj or "payload" not in obj:
            return None
        return obj

    def record_pre_auth(self, module: str, token: str) -> None:
        return None  # the model itself decides the next call from observation


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_eval(
    env_url: str,
    agent_type: str = "rule_based",
    out_csv: str = "eval_results.csv",
    curriculum_stage: int = 1,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    client = DriftDeskClient(env_url)
    if agent_type == "rule_based":
        agent: Any = RuleBasedAgent()
    elif agent_type in ("frozen_llm", "trained_llm"):
        agent = FrozenLLMAgent(
            model_name=model_name,
            adapter_path=adapter_path if agent_type == "trained_llm" else None,
        )
    else:
        raise ValueError(f"Unknown agent_type: {agent_type!r}")

    rows: List[Dict[str, Any]] = []

    for i, seed in enumerate(EVAL_SEEDS):
        with client.session() as sess:
            obs_data = sess.reset(seed=seed, curriculum_stage=curriculum_stage)
            obs = obs_data.get("observation", obs_data)
            agent.reset(obs.get("tasks", []))

            episode_reward = 0.0
            steps = 0

            while True:
                action = agent.act(obs_data)
                if action is None:
                    break

                result = sess.step(action["module"], action["payload"])
                obs = result.get("observation", result)

                # Handle pre-auth token
                if action["module"].endswith(":pre_auth"):
                    mod = action["module"].replace(":pre_auth", "")
                    token = obs.get("last_result", {}).get("pre_auth_token", "")
                    if token:
                        agent.record_pre_auth(mod, token)

                steps += 1
                if result.get("done"):
                    episode_reward = result.get("reward") or 0.0
                    break
                if steps >= 15:
                    break

                obs_data = result

            state = sess.state()
        components = state.get("reward_components", {})
        total_drift = state.get("total_drift_events", 0)
        recoveries = state.get("successful_recoveries", 0)

        row = {
            "seed": seed,
            "episode": i,
            "reward": episode_reward,
            "steps": steps,
            "task_completion": components.get("task_completion", 0.0),
            "drift_recovery": components.get("drift_recovery", 0.0),
            "no_spurious_rewrite": components.get("no_spurious_rewrite", 0.0),
            "policy_grounding": components.get("policy_grounding", 0.0),
            "priority_score": components.get("priority_score", 0.0),
            "efficiency": components.get("efficiency", 0.0),
            "loop_penalty": components.get("loop_penalty", 0.0),
            "total_drift_events": total_drift,
            "successful_recoveries": recoveries,
            "agent": agent_type,
            "curriculum_stage": curriculum_stage,
        }
        rows.append(row)
        print(f"  Ep {i+1:02d}/50 seed={seed} reward={episode_reward:.3f} "
              f"tc={row['task_completion']:.2f} dr={row['drift_recovery']:.2f}")

    # Write CSV
    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {out_csv}")

    # Aggregate
    agg = {
        "n_episodes": len(rows),
        "mean_reward": sum(r["reward"] for r in rows) / len(rows),
        "mean_task_completion": sum(r["task_completion"] for r in rows) / len(rows),
        "mean_drift_recovery": sum(r["drift_recovery"] for r in rows) / len(rows),
        "mean_no_spurious_rewrite": sum(r["no_spurious_rewrite"] for r in rows) / len(rows),
        "agent": agent_type,
    }
    print(json.dumps(agg, indent=2))
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--agent", default="rule_based",
                        choices=["rule_based", "frozen_llm", "trained_llm"])
    parser.add_argument("--out-csv", default="eval_results.csv")
    parser.add_argument("--curriculum-stage", type=int, default=1)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter-path", default=None,
                        help="Path to PEFT adapter (used with --agent trained_llm)")
    args = parser.parse_args()
    run_eval(
        env_url=args.env_url,
        agent_type=args.agent,
        out_csv=args.out_csv,
        curriculum_stage=args.curriculum_stage,
        model_name=args.model_name,
        adapter_path=args.adapter_path,
    )
