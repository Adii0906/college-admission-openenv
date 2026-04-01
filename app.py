"""
app.py — College Admission Counselling Environment
===================================================
HF Space entry point. Runs:
  - FastAPI/OpenEnv server on port 8000 (background thread)
  - Gradio UI on port 7860 (main process)

The Gradio UI calls the FastAPI server via HTTP requests.
This is the correct architecture — clean separation of concerns.
"""

import sys
import os
import json
import time
import threading
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from tasks import TASKS

FASTAPI_URL = "http://localhost:8000"


# ── Start FastAPI/OpenEnv server in background thread ───────────────────────────
def _start_fastapi():
    """Launch FastAPI on port 8000. Retry on failure."""
    import uvicorn
    from models import CollegeAction, CollegeObservation
    from server.college_env_environment import CollegeEnvironment
    from openenv.core.env_server.http_server import create_app

    fastapi_app = create_app(
        CollegeEnvironment,
        CollegeAction,
        CollegeObservation,
        env_name="college_env",
        max_concurrent_envs=10,
    )
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="error")


threading.Thread(target=_start_fastapi, daemon=True).start()


def _wait_for_server(timeout: int = 30) -> bool:
    """Wait until FastAPI is ready to accept requests."""
    for _ in range(timeout):
        try:
            r = requests.get(f"{FASTAPI_URL}/health", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# Wait for server to start
_server_ready = _wait_for_server(30)


# ── HTTP helpers — Gradio calls FastAPI ─────────────────────────────────────────
def api_reset() -> dict:
    """POST /reset → returns observation dict."""
    try:
        r = requests.post(f"{FASTAPI_URL}/reset", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "observation": {}, "reward": 0, "done": False}


def api_step(action: str, target_college: str | None = None, round_number: int = 1) -> dict:
    """POST /step → returns observation dict."""
    payload = {
        "action": {
            "action": action,
            "target_college": target_college,
            "round_number": round_number,
        }
    }
    try:
        r = requests.post(f"{FASTAPI_URL}/step", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "observation": {}, "reward": 0, "done": False}


def api_state() -> dict:
    """GET /state → returns state dict."""
    try:
        r = requests.get(f"{FASTAPI_URL}/state", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Shared UI state ─────────────────────────────────────────────────────────────
_log = []
_last_obs = {}
_current_task = [1]


def _get_obs(data: dict) -> dict:
    return data.get("observation", {})


def _status_html(obs: dict) -> str:
    if not obs:
        return "<p style='text-align:center;color:#888;padding:20px;font-family:system-ui'>Click Reset to begin</p>"

    task_id = obs.get("task_id", 1)
    task = TASKS.get(task_id, TASKS[1])
    score = obs.get("task_score", 0.0)
    done = obs.get("done", False)
    sp = int(score * 100)
    sc = "#1D9E75" if score >= 0.7 else "#EF9F27" if score >= 0.3 else "#E24B4A"
    dl = obs.get("deadline_days_left", 0)
    dc = "#E24B4A" if dl <= 2 else "#EF9F27" if dl <= 4 else "#1D9E75"
    upg = ", ".join(obs.get("available_upgrades", [])) or "None"
    allot = obs.get("allotted_college") or "Not allotted yet"
    msg = obs.get("message", "")[:350]

    done_banner = ""
    if done:
        if score >= 0.75:
            done_banner = f"<div style='margin:10px 0;padding:12px;background:#E1F5EE;border-radius:8px;text-align:center;font-weight:700;color:#085041;font-size:15px'>✅ SUCCESS! Score: {score:.3f} / 1.000</div>"
        else:
            done_banner = f"<div style='margin:10px 0;padding:12px;background:#FCEBEB;border-radius:8px;text-align:center;font-weight:700;color:#791F1F;font-size:15px'>❌ Episode Ended — Score: {score:.3f} / 1.000. Click Reset.</div>"

    return f"""<div style='font-family:system-ui;font-size:13px;padding:2px'>
  <div style='background:#F4F0FF;border-radius:8px;padding:10px 14px;margin-bottom:10px;border-left:4px solid #7C3AED'>
    <b style='color:#6D28D9'>Task {task_id} — {task.difficulty}:</b> {task.name}
  </div>
  <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px'>
    <div style='background:#F8F8F8;border-radius:8px;padding:8px;text-align:center'>
      <div style='font-size:10px;color:#888'>Rank</div>
      <div style='font-size:17px;font-weight:700;color:#7C3AED'>{obs.get("student_rank",0)}</div>
      <div style='font-size:10px;color:#888'>{obs.get("student_category","")}</div>
    </div>
    <div style='background:#F8F8F8;border-radius:8px;padding:8px;text-align:center'>
      <div style='font-size:10px;color:#888'>Round</div>
      <div style='font-size:17px;font-weight:700;color:#7C3AED'>{obs.get("current_round",1)}/3</div>
    </div>
    <div style='background:#F8F8F8;border-radius:8px;padding:8px;text-align:center'>
      <div style='font-size:10px;color:#888'>Steps left</div>
      <div style='font-size:17px;font-weight:700;color:{dc}'>{dl}</div>
    </div>
    <div style='background:#F8F8F8;border-radius:8px;padding:8px;text-align:center'>
      <div style='font-size:10px;color:#888'>Taken</div>
      <div style='font-size:17px;font-weight:700;color:#333'>{obs.get("steps_taken",0)}</div>
    </div>
  </div>
  <div style='background:#F8F8F8;border-radius:8px;padding:10px;margin-bottom:8px'>
    <b>College:</b> {allot}<br>
    <span style='font-size:11px;color:#555'>
      Branch: {obs.get("allotted_branch") or "N/A"} &nbsp;|&nbsp;
      Choices: {"✓" if obs.get("choices_filled") else "No"} &nbsp;|&nbsp;
      Fee paid: {"✓" if obs.get("seat_fee_paid") else "No"} &nbsp;|&nbsp;
      Upgrades: {upg}
    </span>
  </div>
  <div style='margin-bottom:8px'>
    <div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px'>
      <span style='color:#888'>Task Score</span>
      <b style='color:{sc}'>{score:.3f} / 1.000</b>
    </div>
    <div style='background:#E0E0E0;border-radius:4px;height:10px;overflow:hidden'>
      <div style='height:100%;width:{sp}%;background:{sc};border-radius:4px;transition:width 0.4s'></div>
    </div>
  </div>
  <div style='background:#FFFDF0;border:1px solid #E8D000;border-radius:8px;padding:10px;font-size:12px;color:#333;line-height:1.6'>{msg}</div>
  {done_banner}
</div>"""


def _log_html(log: list) -> str:
    if not log:
        return "<p style='color:#888;font-size:12px;padding:8px;font-family:system-ui'>No actions yet.</p>"
    rows = ""
    for e in log:
        r = e.get("reward", 0)
        c = "#1D9E75" if r > 0 else "#E24B4A" if r < 0 else "#888"
        bg = "#E8F8F0" if r > 0 else "#FEE8E8" if r < 0 else "#F5F5F5"
        rows += f"""<div style='display:flex;gap:8px;padding:7px 0;border-bottom:1px solid #F0F0F0;font-family:system-ui'>
  <span style='background:{bg};color:{c};padding:2px 7px;border-radius:10px;font-weight:700;font-size:10px;white-space:nowrap;flex-shrink:0'>{e.get("action","?").upper()}</span>
  <span style='flex:1;font-size:12px;color:#444;line-height:1.4'>{e.get("msg","")[:90]}</span>
  <span style='font-weight:700;color:{c};font-size:12px;flex-shrink:0'>{r:+.1f}</span>
</div>"""
    return f"<div style='max-height:260px;overflow-y:auto'>{rows}</div>"


def _btns(enabled: bool):
    return [gr.update(interactive=enabled)] * 9


# ── Gradio event handlers ────────────────────────────────────────────────────────
def do_reset(task_str: str):
    global _log
    tid = int(str(task_str).split(":")[0].strip())
    _current_task[0] = tid

    # Reset via FastAPI
    data = api_reset()
    obs = _get_obs(data)

    # Switch tasks using special round_number encoding
    if tid != 1:
        # Task 2 → round_number=22, Task 3 → round_number=33
        task_codes = {1: 11, 2: 22, 3: 33}
        data = api_step("check_status", None, task_codes[tid])
        # That triggers internal task switch, then do a fresh reset
        data = api_reset()
        # Re-switch (reset goes back to task 1)
        data = api_step("check_status", None, task_codes[tid])
        obs = _get_obs(data)

    _last_obs.clear()
    _last_obs.update(obs)
    _log = []

    hint = f"💡 Hint: {TASKS[tid].hints[0]}"
    return [_status_html(obs), _log_html(_log), hint] + _btns(True)


def do_action(act_name: str, college_text: str):
    global _log
    if not _last_obs:
        return ["<p style='color:red;padding:12px'>Please click Reset first!</p>",
                _log_html(_log), ""] + _btns(True)

    college = college_text.strip() if college_text and college_text.strip() else None
    rnd = _last_obs.get("current_round", 1)

    data = api_step(act_name, college, rnd)

    if "error" in data and not data.get("observation"):
        return [f"<p style='color:red;padding:12px'>Server error: {data['error']}</p>",
                _log_html(_log), ""] + _btns(True)

    obs = _get_obs(data)
    reward = data.get("reward", 0.0) or 0.0
    done = data.get("done", False)

    _last_obs.clear()
    _last_obs.update(obs)
    _log.append({"action": act_name, "msg": obs.get("message", "")[:90], "reward": reward})

    task = TASKS[_current_task[0]]
    idx = min(obs.get("steps_taken", 0), len(task.hints) - 1)
    if done:
        hint = "✅ Done! Great job!" if obs.get("task_score", 0) >= 0.75 else "🔄 Try again — click Reset"
    else:
        hint = f"💡 Hint {obs.get('steps_taken', 0)}: {task.hints[idx]}" if task.hints else ""

    return [_status_html(obs), _log_html(_log), hint] + _btns(not done)


# ── Build Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(title="College Admission Counselling") as demo:

    gr.Markdown("""
# 🎓 College Admission Counselling Environment
**OpenEnv RL Environment — Meta AI Hackathon 2025**

Simulate India's JEE/CUET counselling. Help the student secure the best college seat!

> **API (port 8000):** `POST /reset` · `POST /step` · `GET /state` · `GET /health`
""")

    with gr.Row():
        # Left column — controls
        with gr.Column(scale=1, min_width=270):
            gr.Markdown("### 1️⃣ Select Task")
            task_dd = gr.Dropdown(
                choices=[
                    "1: Simple Seat Acceptance (Easy)",
                    "2: Strategic Upgrade Decision (Medium)",
                    "3: Multi-Round Counselling (Hard)",
                ],
                value="1: Simple Seat Acceptance (Easy)",
                label="Task",
            )
            reset_btn = gr.Button("🔄  Reset / New Episode", variant="secondary", size="lg")

            gr.Markdown("### 2️⃣ Take Actions (in order!)")

            college_input = gr.Textbox(
                label="College name (for Accept/Fill actions)",
                placeholder="e.g. NIT Warangal CS",
                max_lines=1,
            )

            btn_cs  = gr.Button("📋  Check Status",         variant="primary")
            btn_cc  = gr.Button("📊  Check Cutoffs",        variant="primary")
            btn_fc  = gr.Button("📝  Fill Choices",         variant="primary")
            btn_acc = gr.Button("✅  Accept Allotment",     variant="primary")
            btn_up  = gr.Button("⬆️  Request Upgrade",      variant="primary")
            btn_pay = gr.Button("💳  Pay Seat Fee",         variant="primary")
            btn_rep = gr.Button("🏫  Report to College",    variant="primary")
            btn_wd  = gr.Button("❌  Withdraw  (−10 pts!)")

            gr.Markdown("""
**Reward guide:**
`check_status` +0.3 · `check_cutoffs` +0.5
`fill_choices` +1.0 · `accept` +2.0
`upgrade` +3.0 · `pay_fee` +2.0
`report` +3.0 · `withdraw` **−10** ☠️
""")

        # Right column — display
        with gr.Column(scale=2):
            gr.Markdown("### Environment State")
            status_out = gr.HTML(value="<p style='text-align:center;color:#888;padding:20px'>Click Reset to begin</p>")
            hint_md    = gr.Markdown("")
            gr.Markdown("### Action Log")
            log_out    = gr.HTML(value="<p style='color:#888;font-size:12px;padding:8px'>No actions yet.</p>")

    gr.Markdown("""---
**Space:** [Knight09/college_admission_env](https://huggingface.co/spaces/Knight09/college_admission_env) ·
**Built with:** [OpenEnv](https://github.com/meta-pytorch/OpenEnv) · India JEE/CUET counselling domain
""")

    outs = [status_out, log_out, hint_md,
            btn_cs, btn_cc, btn_fc, btn_acc, btn_up, btn_pay, btn_rep, btn_wd, reset_btn]

    reset_btn.click(fn=do_reset, inputs=[task_dd], outputs=outs)
    btn_cs .click(fn=lambda c: do_action("check_status",     c), inputs=[college_input], outputs=outs)
    btn_cc .click(fn=lambda c: do_action("check_cutoffs",    c), inputs=[college_input], outputs=outs)
    btn_fc .click(fn=lambda c: do_action("fill_choices",     c), inputs=[college_input], outputs=outs)
    btn_acc.click(fn=lambda c: do_action("accept_allotment", c), inputs=[college_input], outputs=outs)
    btn_up .click(fn=lambda c: do_action("upgrade_request",  c), inputs=[college_input], outputs=outs)
    btn_pay.click(fn=lambda c: do_action("pay_seat_fee",     c), inputs=[college_input], outputs=outs)
    btn_rep.click(fn=lambda c: do_action("report_to_college",c), inputs=[college_input], outputs=outs)
    btn_wd .click(fn=lambda c: do_action("withdraw",         c), inputs=[college_input], outputs=outs)


if __name__ == "__main__":
    print("Waiting for FastAPI server on port 8000...")
    if _server_ready:
        print("FastAPI ready! Starting Gradio on port 7860...")
    else:
        print("Warning: FastAPI may not be ready yet, starting Gradio anyway...")
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
