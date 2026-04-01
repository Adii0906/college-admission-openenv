"""
Baseline Inference Script — College Admission Counselling Environment
=====================================================================
Uses Meta's Llama 3 via Groq API (FREE and fast!).
Same OpenAI-compatible format — swap to OpenAI anytime.

HOW TO RUN:
    1. Get FREE Groq key at: console.groq.com (no credit card needed)
    2. Create .env file:  copy .env.example to .env and fill GROQ_API_KEY
    3. Install deps:      pip install groq python-dotenv
    4. Run:               python baseline.py

SWITCHING TO OPENAI:
    Uncomment the OpenAI section below and comment out Groq section.
    Set OPENAI_API_KEY in your .env file.
"""

import os
import json
import time
import sys
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env.example shows the vars needed

# ── API Client Setup ──────────────────────────────────────────────────────────

# ── GROQ (currently active — FREE Llama 3) ───────────────────────────────────
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("ERROR: Set GROQ_API_KEY in your .env file")
    print("Get free key at: console.groq.com")
    sys.exit(1)

client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.1-8b-instant"   # Free Llama 3.1 via Groq

# ── OPENAI (uncomment to switch — remember to set OPENAI_API_KEY) ─────────────
# from openai import OpenAI
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     print("ERROR: Set OPENAI_API_KEY in your .env file")
#     sys.exit(1)
# client = OpenAI(api_key=OPENAI_API_KEY)
# MODEL = "gpt-4o-mini"

# ── Import environment ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
try:
    from server.college_env_environment import CollegeEnvironment
    from models import CollegeAction
    from tasks import TASKS, GRADERS
except ImportError as e:
    print(f"ERROR: {e}")
    print("Run this script from the college_env folder")
    sys.exit(1)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Indian college admission counsellor helping students
navigate the JEE/CUET counselling process (JOSAA/CSAB).

You will receive the current state of a student's counselling journey.
Choose ONE action to help the student get the BEST possible college.

CRITICAL RULES:
1. NEVER choose "withdraw" — it is catastrophic and irreversible
2. Start by checking status or cutoffs to understand the situation
3. If an upgrade to a better college is available, always take it
4. Always pay seat fee immediately after accepting allotment
5. "report_to_college" is the final step — only do it after fee is paid

Available actions:
  check_cutoffs, check_status, fill_choices, lock_choices,
  accept_allotment, upgrade_request, pay_seat_fee, report_to_college, withdraw

Respond ONLY with valid JSON — no explanation, no markdown:
{"action": "check_status", "target_college": null, "round_number": 1}"""


def get_llm_action(obs_message: str, step: int) -> dict:
    """Ask the LLM what action to take given the current observation."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Step {step}. Current situation:\n{obs_message}\n\n"
                    "What is your next action? JSON only."
                )}
            ],
            max_tokens=100,
            temperature=0.1,  # Low temp = reproducible
        )

        raw = response.choices[0].message.content.strip()

        # Clean up markdown if present
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError:
        # Fallback: extract action from text
        raw_lower = raw.lower()
        for act in ["check_cutoffs", "check_status", "fill_choices",
                    "accept_allotment", "upgrade_request", "pay_seat_fee",
                    "report_to_college", "lock_choices"]:
            if act in raw_lower:
                return {"action": act, "target_college": None, "round_number": 1}
        return {"action": "check_status", "target_college": None, "round_number": 1}

    except Exception as e:
        print(f"    API error: {e}")
        return {"action": "check_status", "target_college": None, "round_number": 1}


def run_episode(task_id: int, run_num: int) -> float:
    """Run one episode on a task. Returns final score 0.0-1.0."""
    task = TASKS[task_id]
    env = CollegeEnvironment()
    obs = env._reset_for_task(task_id)

    print(f"\n  Run {run_num} | Task {task_id} ({task.difficulty}): {task.name}")
    print(f"  Rank: {task.student_rank} ({task.student_category}) | Goal: {task.target_outcome}")
    print(f"  {'─' * 55}")

    for step in range(task.max_steps):
        action_data = get_llm_action(obs.message, step + 1)
        act_str = action_data.get("action", "check_status")
        college = action_data.get("target_college")
        print(f"  Step {step+1:2}: {act_str:<20} college={college or 'N/A'}")

        try:
            action = CollegeAction(
                action=act_str,
                target_college=college,
                round_number=action_data.get("round_number", 1),
            )
            obs = env.step(action)
            print(f"          reward={obs.reward:+.1f} | score={obs.task_score:.3f} | {obs.message[:70]}")
        except Exception as e:
            print(f"          Error: {e}")
            break

        if obs.done:
            break

        time.sleep(0.3)  # Be nice to the API

    print(f"\n  Final score: {obs.task_score:.3f} / 1.000")
    return obs.task_score


def main():
    print("=" * 60)
    print("College Admission Counselling — Baseline Evaluation")
    print(f"Model: {MODEL} (via {'Groq' if 'groq' in MODEL.lower() or 'llama' in MODEL.lower() else 'OpenAI'})")
    print("=" * 60)

    RUNS = 3  # 3 runs per task for reproducibility
    all_results = {}

    for task_id in [1, 2, 3]:
        task = TASKS[task_id]
        print(f"\n{'═' * 60}")
        print(f"TASK {task_id} ({task.difficulty}): {task.name}")
        print(f"{'═' * 60}")

        scores = []
        for run in range(1, RUNS + 1):
            s = run_episode(task_id, run)
            scores.append(s)

        avg = sum(scores) / len(scores)
        all_results[task_id] = {
            "name": task.name,
            "difficulty": task.difficulty,
            "scores": scores,
            "average": round(avg, 3),
        }
        print(f"\n  Task {task_id} average: {avg:.3f} (individual: {scores})")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("BASELINE RESULTS — copy into README.md")
    print(f"{'=' * 60}")
    print(f"Model: {MODEL} | Runs per task: {RUNS}\n")

    overall = []
    for tid, data in all_results.items():
        print(f"Task {tid} ({data['difficulty']:6}): {data['average']:.3f} | {data['name']}")
        overall.append(data['average'])

    total = sum(overall) / len(overall)
    print(f"\nOverall: {total:.3f} / 1.000")
    print("=" * 60)

    # Save JSON results
    out = {
        "model": MODEL,
        "runs_per_task": RUNS,
        "results": all_results,
        "overall": round(total, 3),
    }
    with open("baseline_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to baseline_results.json")


if __name__ == "__main__":
    main()
