"""
Kaggle/Colab-ready TRL training script for College Admission OpenEnv.

What this script does:
1) Connects to the College Admission environment over HTTP (/reset, /step, /health).
2) Builds an expert demonstration dataset from scripted trajectories.
3) Fine-tunes Qwen2.5 (1.5B class) with Unsloth + TRL (SFT + QLoRA, T4-friendly).
4) Compares:
   - Random policy baseline
   - Untrained model baseline
   - Trained model
5) Saves quantitative outputs and labeled plots:
   - Training loss vs training step
   - Episode return vs episode index
   - Mean task score comparison
6) Optionally pushes artifacts/model to HF Hub and uploads a simple dashboard HF Space.
7) Supports Weights & Biases experiment tracking.

Recommended install cell (Kaggle/Colab):
pip install -q "unsloth>=2025.1.0" "transformers>=4.46.0" "trl>=0.11.0" \
               "peft>=0.13.0" "accelerate>=0.34.0" "bitsandbytes>=0.43.0" \
               "datasets>=2.20.0" "huggingface_hub>=0.25.0" "pandas>=2.1.0" \
               "matplotlib>=3.8.0" "requests>=2.31.0" "wandb>=0.17.0"
"""

from __future__ import annotations

import json
import locale
import os
import random
import re
import shutil
import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from datasets import Dataset
from huggingface_hub import HfApi, create_repo, login
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

if TYPE_CHECKING:
    from trl import SFTTrainer

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except Exception:  # noqa: BLE001
    FastLanguageModel = None  # type: ignore[assignment]
    UNSLOTH_AVAILABLE = False

    def is_bfloat16_supported() -> bool:
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


@dataclass
class Config:
    # Environment/API
    base_url: str = os.getenv(
        "COLLEGE_ENV_BASE_URL",
        "https://Knight09-college-env.hf.space",
    ).rstrip("/")
    request_timeout_s: int = int(os.getenv("REQUEST_TIMEOUT_S", "45"))
    request_retries: int = int(os.getenv("REQUEST_RETRIES", "3"))

    # Model/Training
    # Closest small Qwen2.5 instruction model supported well by Unsloth.
    base_model_name: str = os.getenv("BASE_MODEL_NAME", "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    output_dir: str = os.getenv("OUTPUT_DIR", "./trl_runs/college_env_qwen25_15b_unsloth")
    max_seq_length: int = int(os.getenv("MAX_SEQ_LENGTH", "512"))
    train_episodes_per_template: int = int(os.getenv("TRAIN_EPISODES_PER_TEMPLATE", "110"))
    max_train_steps: int = int(os.getenv("MAX_TRAIN_STEPS", "600"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-4"))
    train_batch_size: int = int(os.getenv("TRAIN_BATCH_SIZE", "2"))
    grad_accum_steps: int = int(os.getenv("GRAD_ACCUM_STEPS", "8"))
    eval_episodes_per_task: int = int(os.getenv("EVAL_EPISODES_PER_TASK", "12"))
    seed: int = int(os.getenv("SEED", "42"))
    use_wandb: bool = os.getenv("USE_WANDB", "1") == "1"
    wandb_project: str = os.getenv("WANDB_PROJECT", "college-admission-openenv")
    wandb_entity: str = os.getenv("WANDB_ENTITY", "")
    wandb_run_name: str = os.getenv("WANDB_RUN_NAME", "qwen25_15b_unsloth_trl")
    wandb_api_key: str = os.getenv("WANDB_API_KEY", "")

    # HF Hub / Space
    push_to_hub: bool = os.getenv("PUSH_TO_HUB", "0") == "1"
    hf_token: str = os.getenv("HF_TOKEN", "")
    hub_model_repo: str = os.getenv("HUB_MODEL_REPO", "")

    # Optional Space dashboard upload
    push_to_space_dashboard: bool = os.getenv("PUSH_TO_SPACE_DASHBOARD", "0") == "1"
    hf_space_repo: str = os.getenv("HF_SPACE_REPO", "")


CFG = Config()


def load_sft_trainer_class() -> type:
    """Load TRL SFTTrainer with a Windows-safe UTF-8 fallback."""
    # TRL reads some template files using locale default encoding; force UTF-8 on
    # Windows to avoid cp1252 decode failures in current TRL releases.
    if os.name == "nt":
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        os.environ.setdefault("PYTHONUTF8", "1")
        original_getpreferredencoding = locale.getpreferredencoding
        original_read_text = Path.read_text

        def _read_text_utf8(self: Path, encoding: Optional[str] = None, errors: Optional[str] = None) -> str:
            return original_read_text(self, encoding=encoding or "utf-8", errors=errors)

        locale.getpreferredencoding = lambda _do_setlocale=True: "UTF-8"  # type: ignore[assignment]
        Path.read_text = _read_text_utf8  # type: ignore[assignment]
        try:
            from trl import SFTTrainer  # type: ignore[import-not-found]
            return SFTTrainer
        finally:
            locale.getpreferredencoding = original_getpreferredencoding  # type: ignore[assignment]
            Path.read_text = original_read_text  # type: ignore[assignment]
    from trl import SFTTrainer  # type: ignore[import-not-found]
    return SFTTrainer


ACTIONS: List[str] = [
    "check_cutoffs",
    "check_status",
    "fill_choices",
    "lock_choices",
    "accept_allotment",
    "upgrade_request",
    "pay_seat_fee",
    "report_to_college",
    "withdraw",
]

COLLEGE_NAMES: List[str] = [
    "IIT Bombay CS",
    "IIT Delhi CS",
    "IIT Madras CS",
    "IIT Kharagpur CS",
    "IIT Roorkee CS",
    "NIT Trichy CS",
    "NIT Warangal CS",
    "NIT Surathkal CS",
    "NIT Calicut CS",
    "VIT Vellore CS",
    "SRM Chennai CS",
]

TASK_TARGET_COLLEGE: Dict[int, str] = {
    1: "NIT Warangal CS",
    2: "IIT Madras CS",
    3: "IIT Bombay CS",
}

SYSTEM_PROMPT = (
    "You are an expert college admission counselling agent for the Indian JEE/CUET process. "
    "Given environment state and short action history, return exactly one next action in strict JSON "
    "with keys: action, target_college, round_number. "
    "Prefer high-reward safe actions; avoid catastrophic withdraw unless explicitly optimal."
)


EXPERT_TEMPLATES: Dict[int, List[List[Dict[str, Any]]]] = {
    1: [
        [
            {"action": "check_status"},
            {"action": "accept_allotment", "target_college": "<ALLOTTED>"},
            {"action": "pay_seat_fee"},
            {"action": "report_to_college"},
        ],
        [
            {"action": "check_cutoffs"},
            {"action": "check_status"},
            {"action": "accept_allotment", "target_college": "<ALLOTTED>"},
            {"action": "pay_seat_fee"},
            {"action": "report_to_college"},
        ],
        [
            {"action": "check_status"},
            {"action": "accept_allotment", "target_college": "<ALLOTTED>"},
            {"action": "check_status"},
            {"action": "pay_seat_fee"},
            {"action": "report_to_college"},
        ],
    ],
    2: [
        [
            {"action": "check_cutoffs"},
            {"action": "fill_choices", "target_college": "<TASK_TARGET>"},
            {"action": "upgrade_request"},
            {"action": "accept_allotment", "target_college": "<TASK_TARGET>"},
            {"action": "pay_seat_fee"},
            {"action": "report_to_college"},
        ],
        [
            {"action": "check_status"},
            {"action": "check_cutoffs"},
            {"action": "fill_choices", "target_college": "<TASK_TARGET>"},
            {"action": "lock_choices"},
            {"action": "upgrade_request"},
            {"action": "accept_allotment", "target_college": "<TASK_TARGET>"},
            {"action": "pay_seat_fee"},
            {"action": "report_to_college"},
        ],
        [
            {"action": "check_cutoffs"},
            {"action": "fill_choices", "target_college": "<TASK_TARGET>"},
            {"action": "upgrade_request"},
            {"action": "check_status"},
            {"action": "accept_allotment", "target_college": "<TASK_TARGET>"},
            {"action": "pay_seat_fee"},
            {"action": "report_to_college"},
        ],
    ],
    3: [
        [
            {"action": "check_status"},
            {"action": "accept_allotment", "target_college": "<ALLOTTED>"},
            {"action": "check_cutoffs"},
            {"action": "fill_choices", "target_college": "<TASK_TARGET>"},
            {"action": "upgrade_request"},
            {"action": "accept_allotment", "target_college": "<TASK_TARGET>"},
            {"action": "pay_seat_fee"},
            {"action": "report_to_college"},
        ],
        [
            {"action": "check_cutoffs"},
            {"action": "check_status"},
            {"action": "accept_allotment", "target_college": "<ALLOTTED>"},
            {"action": "fill_choices", "target_college": "<TASK_TARGET>"},
            {"action": "upgrade_request"},
            {"action": "accept_allotment", "target_college": "<TASK_TARGET>"},
            {"action": "pay_seat_fee"},
            {"action": "report_to_college"},
        ],
        [
            {"action": "check_status"},
            {"action": "accept_allotment", "target_college": "<ALLOTTED>"},
            {"action": "check_cutoffs"},
            {"action": "fill_choices", "target_college": "<TASK_TARGET>"},
            {"action": "lock_choices"},
            {"action": "upgrade_request"},
            {"action": "accept_allotment", "target_college": "<TASK_TARGET>"},
            {"action": "pay_seat_fee"},
            {"action": "report_to_college"},
        ],
    ],
}


def to_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def compact_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "task_id",
        "student_rank",
        "student_category",
        "current_round",
        "allotted_college",
        "choices_filled",
        "seat_fee_paid",
        "deadline_days_left",
        "available_upgrades",
        "steps_taken",
        "reward",
        "done",
        "task_score",
        "message",
    ]
    return {k: obs.get(k) for k in keys}


def build_user_prompt(obs: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    truncated_history = history[-6:]
    history_json = json.dumps(truncated_history, ensure_ascii=False)
    obs_json = json.dumps(compact_observation(obs), ensure_ascii=False)
    return (
        "Environment state (JSON):\n"
        f"{obs_json}\n\n"
        "Recent actions (oldest -> newest, JSON array):\n"
        f"{history_json}\n\n"
        "Return the best next action as strict JSON only."
    )


class CollegeHTTPEnv:
    def __init__(self, base_url: str, timeout_s: int = 45, retries: int = 3) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.retries = retries
        self.session = requests.Session()

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                if method.upper() == "GET":
                    response = self.session.get(url, timeout=self.timeout_s)
                else:
                    response = self.session.post(url, json=payload or {}, timeout=self.timeout_s)
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(1.25 * attempt)
                else:
                    raise RuntimeError(f"Request failed after {self.retries} attempts: {method} {url}") from last_exc
        raise RuntimeError(f"Unexpected request failure for {method} {url}")

    def healthcheck(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def reset(self, task_id: int = 1) -> Dict[str, Any]:
        data = self._request("POST", "/reset", {})
        if "observation" not in data:
            raise ValueError(f"/reset missing 'observation': {data}")
        obs = data["observation"]
        if task_id in (2, 3):
            marker = 22 if task_id == 2 else 33
            data = self._request(
                "POST",
                "/step",
                {
                    "action": {
                        "action": "check_status",
                        "target_college": None,
                        "round_number": marker,
                    }
                },
            )
            if "observation" not in data:
                raise ValueError(f"/step task switch missing 'observation': {data}")
            obs = data["observation"]
        return obs

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool]:
        action_payload = {
            "action": action["action"],
            "target_college": action.get("target_college"),
            "round_number": action.get("round_number", 1),
        }
        wrapped_payload = {"action": action_payload}
        try:
            data = self._request("POST", "/step", wrapped_payload)
        except Exception:
            # Backward compatibility: some custom servers expect a flat payload.
            data = self._request("POST", "/step", action_payload)
        if "observation" not in data:
            raise ValueError(f"/step missing 'observation': {data}")
        obs = data["observation"]
        reward = float(data.get("reward", obs.get("reward", 0.0)))
        done = bool(data.get("done", obs.get("done", False)))
        return obs, reward, done


class LocalCollegeEnv:
    """In-process fallback env when HTTP API is unavailable."""

    def __init__(self) -> None:
        from models import CollegeAction
        from server.college_env_environment import CollegeEnvironment

        self._CollegeAction = CollegeAction
        self._env = CollegeEnvironment()

    @staticmethod
    def _obs_to_dict(obs_obj: Any) -> Dict[str, Any]:
        if hasattr(obs_obj, "model_dump"):
            return obs_obj.model_dump()
        if hasattr(obs_obj, "dict"):
            return obs_obj.dict()
        return dict(obs_obj)

    def healthcheck(self) -> Dict[str, Any]:
        return {"status": "ok", "mode": "local_inprocess"}

    def reset(self, task_id: int = 1) -> Dict[str, Any]:
        obs = self._env._reset_for_task(task_id)
        return self._obs_to_dict(obs)

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool]:
        act = self._CollegeAction(
            action=action["action"],
            target_college=action.get("target_college"),
            round_number=action.get("round_number", 1),
        )
        obs_obj = self._env.step(act)
        obs = self._obs_to_dict(obs_obj)
        reward = float(getattr(obs_obj, "reward", obs.get("reward", 0.0)))
        done = bool(getattr(obs_obj, "done", obs.get("done", False)))
        return obs, reward, done


def resolve_template_action(template_action: Dict[str, Any], obs: Dict[str, Any], task_id: int) -> Dict[str, Any]:
    action = dict(template_action)
    target = action.get("target_college")
    if target == "<ALLOTTED>":
        target = obs.get("allotted_college")
    elif target == "<TASK_TARGET>":
        target = TASK_TARGET_COLLEGE.get(task_id)
    action["target_college"] = target
    action["round_number"] = to_int(obs.get("current_round", 1), 1)
    return action


def collect_demonstrations(env: CollegeHTTPEnv, episodes_per_template: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for task_id, template_list in EXPERT_TEMPLATES.items():
        for template_idx, template in enumerate(template_list):
            for ep in range(episodes_per_template):
                obs = env.reset(task_id=task_id)
                history: List[Dict[str, Any]] = []

                for template_action in template:
                    action = resolve_template_action(template_action, obs, task_id)
                    rows.append(
                        {
                            "task_id": task_id,
                            "template_idx": template_idx,
                            "episode_idx": ep,
                            "prompt": build_user_prompt(obs, history),
                            "response": json.dumps(action, ensure_ascii=False),
                        }
                    )
                    obs, _, done = env.step(action)
                    history.append(action)
                    if done:
                        break
    return pd.DataFrame(rows)


def as_chat_text(tokenizer: AutoTokenizer, prompt: str, response: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            # Some tokenizers expose apply_chat_template but have no chat_template set.
            pass
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{prompt}\n"
        f"<|assistant|>\n{response}"
    )


def build_sft_datasets(tokenizer: AutoTokenizer, df: pd.DataFrame, seed: int) -> Tuple[Dataset, Dataset]:
    records = df.to_dict(orient="records")
    random.Random(seed).shuffle(records)
    split_idx = int(0.95 * len(records))
    train_records = records[:split_idx]
    eval_records = records[split_idx:]

    train_text = [{"text": as_chat_text(tokenizer, r["prompt"], r["response"])} for r in train_records]
    eval_text = [{"text": as_chat_text(tokenizer, r["prompt"], r["response"])} for r in eval_records]

    return Dataset.from_list(train_text), Dataset.from_list(eval_text)


def infer_lora_targets(model: AutoModelForCausalLM) -> List[str]:
    preferred = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found: set[str] = set()
    all_leaf_names: set[str] = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        all_leaf_names.add(leaf)
        if leaf in preferred:
            found.add(leaf)
    if found:
        return sorted(found)

    # Fallbacks for GPT-style naming conventions.
    fallback_order = ["c_attn", "c_proj", "c_fc", "query_key_value", "Wqkv"]
    fallback_found = [name for name in fallback_order if name in all_leaf_names]
    if fallback_found:
        return fallback_found

    # Last-resort conservative fallback for modern decoder-only models.
    return ["q_proj", "v_proj"]


def setup_wandb(cfg: Config) -> List[str]:
    if not cfg.use_wandb:
        return []
    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    if cfg.wandb_entity:
        os.environ["WANDB_ENTITY"] = cfg.wandb_entity
    if cfg.wandb_run_name:
        os.environ["WANDB_NAME"] = cfg.wandb_run_name
    try:
        import wandb
        if cfg.wandb_api_key:
            wandb.login(key=cfg.wandb_api_key)
        print(
            f"[W&B] Enabled project={os.environ.get('WANDB_PROJECT', '')} "
            f"run_name={os.environ.get('WANDB_NAME', '')}"
        )
        return ["wandb"]
    except Exception as exc:  # noqa: BLE001
        print(f"[W&B] Could not initialize wandb ({exc}). Continuing without W&B logging.")
        return []


def load_model_and_tokenizer(model_name: str, max_seq_length: int) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    effective_model_name = model_name
    if (not torch.cuda.is_available() or not UNSLOTH_AVAILABLE) and model_name.startswith("unsloth/"):
        effective_model_name = model_name.replace("unsloth/", "").replace("-bnb-4bit", "")
        print(
            f"[INFO] Using fallback model '{effective_model_name}' "
            f"because Unsloth 4-bit runtime is unavailable."
        )
    if UNSLOTH_AVAILABLE and torch.cuda.is_available():
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.use_cache = False
        return model, tokenizer, "unsloth_4bit"

    tokenizer = AutoTokenizer.from_pretrained(effective_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    backend = "transformers_cpu"
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            backend = "transformers_4bit"
        except Exception as exc:  # noqa: BLE001
            backend = "transformers_fp16"
            print(f"[WARN] BitsAndBytes 4-bit unavailable ({exc}); continuing with fp16 weights.")
    else:
        model_kwargs["torch_dtype"] = torch.float32
        model_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(effective_model_name, **model_kwargs)
    model.config.use_cache = False
    return model, tokenizer, backend


def prepare_model_for_training(
    model: AutoModelForCausalLM,
    lora_targets: List[str],
    seed: int,
    backend: str,
) -> Tuple[AutoModelForCausalLM, Optional[LoraConfig]]:
    if backend == "unsloth_4bit" and UNSLOTH_AVAILABLE:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=lora_targets,
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )
        return model, None

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    return model, peft_config


def build_training_arguments(cfg: Config, out_dir: Path, report_to: List[str]) -> TrainingArguments:
    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(use_cuda and is_bfloat16_supported())
    train_kwargs: Dict[str, Any] = {
        "output_dir": str(out_dir),
        "max_steps": cfg.max_train_steps,
        "per_device_train_batch_size": cfg.train_batch_size,
        "per_device_eval_batch_size": cfg.train_batch_size,
        "gradient_accumulation_steps": cfg.grad_accum_steps,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "fp16": bool(use_cuda and not use_bf16),
        "bf16": bool(use_cuda and use_bf16),
        "logging_steps": 10,
        "eval_steps": 100,
        "save_steps": 100,
        "save_total_limit": 2,
        "gradient_checkpointing": True,
        "optim": "adamw_8bit" if use_cuda else "adamw_torch",
        "report_to": report_to,
        "run_name": cfg.wandb_run_name if report_to else None,
        "seed": cfg.seed,
    }
    for eval_arg_name in ("evaluation_strategy", "eval_strategy"):
        try:
            return TrainingArguments(**train_kwargs, **{eval_arg_name: "steps"})
        except TypeError:
            continue
    return TrainingArguments(**train_kwargs)


def parse_action_text(raw_text: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    default = {
        "action": "check_status",
        "target_college": None,
        "round_number": to_int(obs.get("current_round", 1), 1),
    }
    text = (raw_text or "").strip()
    if not text:
        return default

    parsed: Dict[str, Any] = {}
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    json_blob = json_match.group(0) if json_match else text
    try:
        candidate = json.loads(json_blob)
        if isinstance(candidate, dict):
            parsed = candidate
    except Exception:
        parsed = {}

    action = str(parsed.get("action", "")).strip()
    if action not in ACTIONS:
        lowered = text.lower()
        for a in ACTIONS:
            if a in lowered:
                action = a
                break
    if action not in ACTIONS:
        return default

    target = parsed.get("target_college")
    if action in {"accept_allotment", "fill_choices"} and (target is None or str(target).strip() == ""):
        if action == "accept_allotment":
            target = obs.get("allotted_college")
        else:
            target = TASK_TARGET_COLLEGE.get(to_int(obs.get("task_id", 1), 1))
    elif action not in {"accept_allotment", "fill_choices"}:
        target = None

    round_number = to_int(parsed.get("round_number", obs.get("current_round", 1)), to_int(obs.get("current_round", 1), 1))
    return {"action": action, "target_college": target, "round_number": round_number}


def build_model_policy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    do_sample: bool = False,
    temperature: float = 0.2,
) -> Callable[[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]]:
    if UNSLOTH_AVAILABLE:
        try:
            FastLanguageModel.for_inference(model)
        except Exception:  # noqa: BLE001
            pass
    def _policy(obs: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(obs, history)},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{build_user_prompt(obs, history)}\n<|assistant|>\n"
        else:
            prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{build_user_prompt(obs, history)}\n<|assistant|>\n"

        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": 96,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 0.9

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **generation_kwargs,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return parse_action_text(raw_text, obs)

    return _policy


def random_policy(obs: Dict[str, Any], _history: List[Dict[str, Any]]) -> Dict[str, Any]:
    action = random.choice(ACTIONS)
    target: Optional[str] = None
    if action == "accept_allotment":
        target = obs.get("allotted_college") if random.random() < 0.6 else random.choice(COLLEGE_NAMES)
    elif action == "fill_choices":
        target = random.choice(COLLEGE_NAMES)
    return {
        "action": action,
        "target_college": target,
        "round_number": to_int(obs.get("current_round", 1), 1),
    }


def run_single_episode(
    env: CollegeHTTPEnv,
    task_id: int,
    policy: Callable[[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]],
    max_steps: int = 15,
    capture_trace: bool = False,
) -> Dict[str, Any]:
    obs = env.reset(task_id=task_id)
    history: List[Dict[str, Any]] = []
    total_reward = 0.0
    trace: List[str] = []

    for step in range(1, max_steps + 1):
        action = policy(obs, history)
        obs, reward, done = env.step(action)
        total_reward += reward
        history.append(action)

        if capture_trace:
            trace.append(
                f"step={step:02d} action={action['action']:<18} "
                f"target={action.get('target_college')} reward={reward:+.2f} "
                f"score={float(obs.get('task_score', 0.0)):.3f}"
            )
        if done:
            break

    result = {
        "task_id": task_id,
        "episode_return": total_reward,
        "task_score": float(obs.get("task_score", 0.0)),
        "steps": to_int(obs.get("steps_taken", len(history)), len(history)),
        "done": bool(obs.get("done", False)),
    }
    if capture_trace:
        result["trace"] = trace
    return result


def evaluate_policy(
    env: CollegeHTTPEnv,
    policy_name: str,
    policy: Callable[[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]],
    episodes_per_task: int,
) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
    rows: List[Dict[str, Any]] = []
    traces: Dict[int, List[str]] = {}
    episode_number = 0

    for task_id in (1, 2, 3):
        for ep in range(episodes_per_task):
            episode_number += 1
            capture_trace = ep == 0
            result = run_single_episode(
                env=env,
                task_id=task_id,
                policy=policy,
                max_steps=15,
                capture_trace=capture_trace,
            )
            result["policy"] = policy_name
            result["episode_index"] = episode_number
            rows.append(result)
            if capture_trace:
                traces[task_id] = result.get("trace", [])
    return pd.DataFrame(rows), traces


def plot_training_loss(
    log_history: List[Dict[str, Any]],
    out_path: Path,
    fallback_train_loss: Optional[float] = None,
    fallback_steps: int = 1,
) -> Path:
    train_logs = [x for x in log_history if "loss" in x and "step" in x]
    if train_logs:
        x = [int(item["step"]) for item in train_logs]
        y = [float(item["loss"]) for item in train_logs]
    elif fallback_train_loss is not None:
        # For very short runs (e.g., max_steps=1), trainer may only report aggregate train_loss.
        x = [0, max(1, int(fallback_steps))]
        y = [float(fallback_train_loss), float(fallback_train_loss)]
    else:
        raise ValueError("No training loss logs found in trainer.state.log_history.")

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, linewidth=1.75, label="Train loss")
    plt.xlabel("Training step (optimizer updates)")
    plt.ylabel("Cross-entropy loss (unitless)")
    plt.title("TRL SFT Training Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()
    return out_path


def plot_eval_curves(eval_df: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Episode return curve with EMA smoothing
    plt.figure(figsize=(10, 5))
    for policy_name in eval_df["policy"].unique():
        sub = eval_df[eval_df["policy"] == policy_name].sort_values("episode_index").copy()
        sub["ema_return"] = sub["episode_return"].ewm(span=8, adjust=False).mean()
        plt.plot(sub["episode_index"], sub["ema_return"], linewidth=2, label=policy_name)

    plt.xlabel("Episode index (#)")
    plt.ylabel("Episode return (reward points)")
    plt.title("Policy Performance Across Episodes (EMA)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    reward_curve_path = out_dir / "episode_return_curve.png"
    plt.savefig(reward_curve_path, dpi=170)
    plt.close()

    # 2) Mean final task score bar chart
    summary = (
        eval_df.groupby("policy", as_index=False)["task_score"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_task_score", "std": "std_task_score"})
    )

    plt.figure(figsize=(8.5, 5))
    x_pos = np.arange(len(summary))
    plt.bar(
        x_pos,
        summary["mean_task_score"],
        yerr=summary["std_task_score"].fillna(0.0),
        capsize=6,
    )
    plt.xticks(x_pos, summary["policy"])
    plt.xlabel("Policy")
    plt.ylabel("Final task score (0-1 score units)")
    plt.title("Mean Task Score by Policy")
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    score_bar_path = out_dir / "task_score_bar.png"
    plt.savefig(score_bar_path, dpi=170)
    plt.close()

    return reward_curve_path, score_bar_path


def ensure_hub_inputs(cfg: Config) -> None:
    if not cfg.hf_token:
        raise ValueError("HF_TOKEN is required when push_to_hub or push_to_space_dashboard is enabled.")


def push_to_hub(
    cfg: Config,
    trainer: Any,
    tokenizer: AutoTokenizer,
    summary_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    artifacts: List[Path],
) -> None:
    if not cfg.push_to_hub:
        return
    ensure_hub_inputs(cfg)
    if not cfg.hub_model_repo:
        raise ValueError("Set HUB_MODEL_REPO=your-username/your-model-repo when PUSH_TO_HUB=1.")

    print(f"[HF] Logging in and pushing model/artifacts to {cfg.hub_model_repo} ...")
    login(token=cfg.hf_token)
    create_repo(cfg.hub_model_repo, repo_type="model", private=False, exist_ok=True, token=cfg.hf_token)

    # Push PEFT adapter + tokenizer
    trainer.model.push_to_hub(cfg.hub_model_repo, token=cfg.hf_token)
    tokenizer.push_to_hub(cfg.hub_model_repo, token=cfg.hf_token)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "evaluation_summary.csv"
    eval_csv = output_dir / "evaluation_episodes.csv"
    summary_df.to_csv(summary_csv, index=False)
    eval_df.to_csv(eval_csv, index=False)

    model_card_path = output_dir / "README.md"
    model_card_path.write_text(
        (
            f"# College Env TRL Adapter\n\n"
            f"- Base model: `{cfg.base_model_name}`\n"
            f"- Environment URL: `{cfg.base_url}`\n"
            f"- Train steps: `{cfg.max_train_steps}`\n"
            f"- Eval episodes/task: `{cfg.eval_episodes_per_task}`\n\n"
            f"Artifacts are in the `artifacts/` folder in this repo.\n"
        ),
        encoding="utf-8",
    )

    api = HfApi(token=cfg.hf_token)
    upload_files = artifacts + [summary_csv, eval_csv, model_card_path]
    for file_path in upload_files:
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=f"artifacts/{file_path.name}",
            repo_id=cfg.hub_model_repo,
            repo_type="model",
            token=cfg.hf_token,
        )

    print("[HF] Model + artifacts pushed successfully.")


def push_space_dashboard(cfg: Config, summary_df: pd.DataFrame, artifacts: List[Path]) -> None:
    if not cfg.push_to_space_dashboard:
        return
    ensure_hub_inputs(cfg)
    if not cfg.hf_space_repo:
        raise ValueError("Set HF_SPACE_REPO=your-username/your-space when PUSH_TO_SPACE_DASHBOARD=1.")

    print(f"[HF Space] Creating/updating dashboard space: {cfg.hf_space_repo}")
    login(token=cfg.hf_token)
    create_repo(
        cfg.hf_space_repo,
        repo_type="space",
        private=False,
        space_sdk="gradio",
        exist_ok=True,
        token=cfg.hf_token,
    )

    out_dir = Path(cfg.output_dir)
    space_dir = out_dir / "space_dashboard"
    assets_dir = space_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = assets_dir / "evaluation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    for artifact in artifacts:
        shutil.copy2(artifact, assets_dir / artifact.name)

    readme_text = (
        "---\n"
        "title: College Env Training Dashboard\n"
        "emoji: 📈\n"
        "colorFrom: indigo\n"
        "colorTo: cyan\n"
        "sdk: gradio\n"
        "app_file: app.py\n"
        "pinned: false\n"
        "---\n\n"
        "# College Env TRL Dashboard\n"
        "This Space shows training and evaluation artifacts from `train_trl_kaggle.py`.\n"
    )
    (space_dir / "README.md").write_text(readme_text, encoding="utf-8")

    app_py = (
        "import pandas as pd\n"
        "import gradio as gr\n"
        "\n"
        "summary = pd.read_csv('assets/evaluation_summary.csv')\n"
        "with gr.Blocks(title='College Env TRL Dashboard') as demo:\n"
        "    gr.Markdown('# College Env TRL Dashboard')\n"
        "    gr.Markdown('## Evaluation summary')\n"
        "    gr.Dataframe(value=summary, interactive=False)\n"
        "    gr.Markdown('## Training loss curve')\n"
        "    gr.Image(value='assets/training_loss_curve.png', interactive=False)\n"
        "    gr.Markdown('## Episode return curve')\n"
        "    gr.Image(value='assets/episode_return_curve.png', interactive=False)\n"
        "    gr.Markdown('## Mean task score comparison')\n"
        "    gr.Image(value='assets/task_score_bar.png', interactive=False)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    demo.launch()\n"
    )
    (space_dir / "app.py").write_text(app_py, encoding="utf-8")
    (space_dir / "requirements.txt").write_text("gradio>=5.0.0\npandas>=2.1.0\n", encoding="utf-8")

    api = HfApi(token=cfg.hf_token)
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=cfg.hf_space_repo,
        repo_type="space",
        token=cfg.hf_token,
    )
    print("[HF Space] Dashboard pushed successfully.")


def main() -> None:
    set_seed(CFG.seed)
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)

    out_dir = Path(CFG.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = out_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("College Env TRL Training (Kaggle/Colab-ready)")
    print(f"Environment URL: {CFG.base_url}")
    print(f"Base model:      {CFG.base_model_name}")
    print(f"Output dir:      {out_dir.resolve()}")
    print("=" * 80)

    http_env = CollegeHTTPEnv(
        base_url=CFG.base_url,
        timeout_s=CFG.request_timeout_s,
        retries=CFG.request_retries,
    )
    try:
        health = http_env.healthcheck()
        env: Any = http_env
        print(f"Healthcheck: {health}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] HTTP environment unavailable ({exc}). Falling back to in-process local environment.")
        env = LocalCollegeEnv()
        health = env.healthcheck()
        print(f"Healthcheck: {health}")

    # 1) Build demonstrations
    print("\n[1/7] Collecting expert demonstrations ...")
    demo_df = collect_demonstrations(env, episodes_per_template=CFG.train_episodes_per_template)
    print(f"Collected {len(demo_df)} demonstration pairs.")
    demo_csv = artifact_dir / "demonstrations.csv"
    demo_df.to_csv(demo_csv, index=False)

    # 2) Load base model/tokenizer
    print("\n[2/7] Loading base model + tokenizer ...")
    model, tokenizer, model_backend = load_model_and_tokenizer(CFG.base_model_name, CFG.max_seq_length)
    print(f"Model backend: {model_backend}")
    lora_targets = infer_lora_targets(model)
    print(f"LoRA target modules: {lora_targets}")
    report_to = setup_wandb(CFG)

    # 3) Create SFT datasets
    print("\n[3/7] Building TRL datasets ...")
    train_ds, eval_ds = build_sft_datasets(tokenizer, demo_df, seed=CFG.seed)
    print(f"Train samples: {len(train_ds)} | Eval samples: {len(eval_ds)}")

    # 4) Evaluate random + untrained baselines
    print("\n[4/7] Evaluating baselines (random + untrained) ...")
    random_df, random_traces = evaluate_policy(
        env=env,
        policy_name="random_baseline",
        policy=random_policy,
        episodes_per_task=CFG.eval_episodes_per_task,
    )
    untrained_policy = build_model_policy(model, tokenizer, do_sample=False)
    untrained_df, untrained_traces = evaluate_policy(
        env=env,
        policy_name="untrained_model",
        policy=untrained_policy,
        episodes_per_task=CFG.eval_episodes_per_task,
    )

    # 5) Train with TRL SFT + QLoRA
    print("\n[5/7] Training TRL SFT model ...")
    SFTTrainer = load_sft_trainer_class()
    model_for_training, peft_config = prepare_model_for_training(
        model=model,
        lora_targets=lora_targets,
        seed=CFG.seed,
        backend=model_backend,
    )
    training_args = build_training_arguments(CFG, out_dir, report_to)
    trainer_kwargs: Dict[str, Any] = {
        "model": model_for_training,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "args": training_args,
    }
    trainer_init_params = inspect.signature(SFTTrainer.__init__).parameters
    if "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    if "dataset_text_field" in trainer_init_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_init_params:
        trainer_kwargs["max_seq_length"] = CFG.max_seq_length
    if "packing" in trainer_init_params:
        trainer_kwargs["packing"] = True
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    trainer = SFTTrainer(**trainer_kwargs)
    train_result = trainer.train()
    trainer.save_model(str(out_dir / "final_adapter"))
    tokenizer.save_pretrained(str(out_dir / "final_adapter"))
    print(f"Train runtime (s): {train_result.metrics.get('train_runtime')}")

    # 6) Evaluate trained model
    print("\n[6/7] Evaluating trained model ...")
    trained_policy = build_model_policy(trainer.model, tokenizer, do_sample=False)
    trained_df, trained_traces = evaluate_policy(
        env=env,
        policy_name="trained_model",
        policy=trained_policy,
        episodes_per_task=CFG.eval_episodes_per_task,
    )

    # Merge + save
    eval_df = pd.concat([random_df, untrained_df, trained_df], ignore_index=True)
    eval_csv = artifact_dir / "evaluation_episodes.csv"
    eval_df.to_csv(eval_csv, index=False)

    summary_df = (
        eval_df.groupby(["policy", "task_id"], as_index=False)
        .agg(
            mean_episode_return=("episode_return", "mean"),
            std_episode_return=("episode_return", "std"),
            mean_task_score=("task_score", "mean"),
            std_task_score=("task_score", "std"),
            mean_steps=("steps", "mean"),
        )
        .sort_values(["task_id", "policy"])
    )
    summary_csv = artifact_dir / "evaluation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # 7) Plot curves + push
    print("\n[7/7] Plotting curves and exporting artifacts ...")
    fallback_loss = train_result.metrics.get("train_loss")
    loss_curve = plot_training_loss(
        trainer.state.log_history,
        artifact_dir / "training_loss_curve.png",
        fallback_train_loss=float(fallback_loss) if fallback_loss is not None else None,
        fallback_steps=CFG.max_train_steps,
    )
    reward_curve, score_bar = plot_eval_curves(eval_df, artifact_dir)

    qualitative_path = artifact_dir / "qualitative_traces.txt"
    with qualitative_path.open("w", encoding="utf-8") as f:
        f.write("=== Random baseline (task 3, first episode) ===\n")
        f.write("\n".join(random_traces.get(3, [])))
        f.write("\n\n=== Untrained model (task 3, first episode) ===\n")
        f.write("\n".join(untrained_traces.get(3, [])))
        f.write("\n\n=== Trained model (task 3, first episode) ===\n")
        f.write("\n".join(trained_traces.get(3, [])))
        f.write("\n")

    artifacts = [demo_csv, eval_csv, summary_csv, qualitative_path, loss_curve, reward_curve, score_bar]

    print("\nAggregate comparison (higher is better):")
    overall_summary = (
        eval_df.groupby("policy", as_index=False)
        .agg(
            mean_episode_return=("episode_return", "mean"),
            mean_task_score=("task_score", "mean"),
            mean_steps=("steps", "mean"),
        )
        .sort_values("mean_task_score", ascending=False)
    )
    print(overall_summary.to_string(index=False))

    push_to_hub(
        cfg=CFG,
        trainer=trainer,
        tokenizer=tokenizer,
        summary_df=summary_df,
        eval_df=eval_df,
        artifacts=artifacts,
    )
    push_space_dashboard(cfg=CFG, summary_df=summary_df, artifacts=[loss_curve, reward_curve, score_bar])

    print("\nDone.")
    print(f"Artifacts saved in: {artifact_dir.resolve()}")
    print("Key files:")
    for file_path in artifacts:
        print(f" - {file_path}")


if __name__ == "__main__":
    main()
