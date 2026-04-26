"""
training/model.py — Tokenizer and model loading for DriftDesk GRPO training.

Supports:
  - 4-bit QLoRA (consumer GPUs, USE_QUANTIZATION=1)
  - Full bfloat16 (A100/H100, USE_QUANTIZATION=0)
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

from driftdesk.training.config import (
    MODEL_NAME, LORA_R, LORA_ALPHA, USE_QUANTIZATION,
)


def load_tokenizer() -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model() -> AutoModelForCausalLM:
    if USE_QUANTIZATION:
        print("Loading model with 4-bit QLoRA (consumer GPU mode)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        print("Loading model in full bf16 (A100/large GPU mode)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model
