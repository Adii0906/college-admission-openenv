"""
app.py — College Admission Counselling Environment
===================================================
HF Space entry point (sdk: gradio).

Key architecture:
  - Gradio Blocks serves the UI at /  on port 7860
  - OpenEnv API routes (/reset /step /state /health) 
    are added directly to Gradio's internal FastAPI app
  - HF checker pings /reset on port 7860 → gets 200 OK ✓
  - No Docker, no background threads, no port conflicts
"""

import sys
import os
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from fastapi import Request
from fastapi.responses import JSONResponse

from tasks import TASKS
from models import CollegeAction, CollegeObservation
from server.college_env_environment import CollegeEnvironment

# ── One environment instance (shared between API + UI) ─────────────────────────
_env = CollegeEnvironment()
_ui_log = []
_ui_obs = {}
_ui_task = [1]


# ── Helper: convert observation object to dict ──────────────────────────────────
def _obs_dict(obs) -> dict:
    if isinstance(obs, dict):
        return obs
    return {
        "student_rank":      obs.student_rank,
        "student_category":  obs.student_category,
        "task_id":           obs.task_id,
        "current_round":     obs.current_round,
        "allotted_college":  obs.allotted_college,
        "allotted_branch":   obs.allotted_branch,
        "choices_filled":    obs.choices_filled,
        "seat_fee_paid":     obs.seat_fee_paid,
        "deadline_days_left":obs.deadline_days_left,
        "available_upgrades":obs.available_upgrades,
        "steps_taken":       obs.steps_taken,
        "reward":            float(obs.reward),
        "done":              bool(obs.done),
        "task_score":        float(obs.task_score),
        "message":           str(obs.message),
        "metadata":          {},
    }


# ── Build Gradio UI first (we need demo.app to add routes) ─────────────────────
def _status_html(obs: dict) -> str:
    if not obs:
        return "<p style='text-align:center;color:#888;padding:20px;font-family:system-ui'>Click Reset to begin</p>"

    task_id = int(obs.get("task_id", 1))
    task    = TASKS.get(task_id, TASKS[1])
    score   = float(obs.get("task_score", 0.0))
    done    = bool(obs.get("done", False))
    sp      = int(score * 100)
    sc      = "#1D9E75" if score >= 0.7 else "#EF9F27" if score >= 0.3 else "#E24B4A"
    dl      = int(obs.get("deadline_days_left", 0))
    dc      = "#E24B4A" if dl <= 2 else "#EF9F27" if dl <= 4 else "#1D9E75"
    upg     = ", ".join(obs.get("available_upgrades") or []) or "None"
    allot   = obs.get("allotted_college") or "Not allotted yet"
    msg     = str(obs.get("message", ""))[:350]

    done_html = ""
    if done:
        if score >= 0.75:
            done_html = "<div style='margin:10px 0;padding:12px;background:#E1F5EE;border-radius:8px;text-align:center;font-weight:700;color:#085041;font-size:15px'>✅ SUCCESS! Click Reset for new episode.</div>"
        else:
            done_html = f"<div style='margin:10px 0;padding:12px;background:#FCEBEB;border-radius:8px;text-align:center;font-weight:700;color:#791F1F;font-size:14px'>Episode ended. Score: {score:.3f}. Click Reset.</div>"

    return f"""<div style='font-family:system-ui;font-size:13px;padding:2px'>
  <div style='background:#F4F0FF;border-radius:8px;padding:10px 14px;margin-bottom:10px;border-left:4px solid #7C3AED'>
    <b style='color:#6D28D9'>Task {task_id} — {task.difficulty}:</b> {task.name}
  </div>
  <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px'>
    <div style='background:#F8F8F8;border-radius:8px;padding:8px;text-align:center'>
      <div style='font-size:10px;color:#888'>Rank</div>
      <div style='font-size:16px;font-weight:700;color:#7C3AED'>{obs.get("student_rank",0)}</div>
      <div style='font-size:10px;color:#888'>{obs.get("student_category","")}</div>
    </div>
    <div style='background:#F8F8F8;border-radius:8px;padding:8px;text-align:center'>
      <div style='font-size:10px;color:#888'>Round</div>
      <div style='font-size:16px;font-weight:700;color:#7C3AED'>{obs.get("current_round",1)}/3</div>
    </div>
    <div style='background:#F8F8F8;border-radius:8px;padding:8px;text-align:center'>
      <div style='font-size:10px;color:#888'>Steps left</div>
      <div style='font-size:16px;font-weight:700;color:{dc}'>{dl}</div>
    </div>
    <div style='background:#F8F8F8;border-radius:8px;padding:8px;text-align:center'>
      <div style='font-size:10px;color:#888'>Taken</div>
      <div style='font-size:16px;font-weight:700;color:#333'>{obs.get("steps_taken",0)}</div>
    </div>
  </div>
  <div style='background:#F8F8F8;border-radius:8px;padding:10px;margin-bottom:8px'>
    <b>College:</b> {allot} · Branch: {obs.get("allotted_branch") or "N/A"}<br>
    <span style='font-size:11px;color:#555'>
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
  {done_html}
</div>"""


def _log_html(log: list) -> str:
    if not log:
        return "<p style='color:#888;font-size:12px;padding:8px;font-family:system-ui'>No actions yet.</p>"
    rows = ""
    for e in reversed(log[-10:]):
        r = float(e.get("reward", 0))
        c  = "#1D9E75" if r > 0 else "#E24B4A" if r < 0 else "#888"
        bg = "#E8F8F0" if r > 0 else "#FEE8E8" if r < 0 else "#F5F5F5"
        rows += f"""<div style='display:flex;gap:8px;padding:6px 0;border-bottom:1px solid #F0F0F0;font-family:system-ui'>
  <span style='background:{bg};color:{c};padding:2px 7px;border-radius:10px;font-weight:700;font-size:10px;white-space:nowrap;flex-shrink:0'>{str(e.get("action","?")).upper()}</span>
  <span style='flex:1;font-size:12px;color:#444;line-height:1.4'>{str(e.get("msg",""))[:90]}</span>
  <span style='font-weight:700;color:{c};font-size:12px;flex-shrink:0'>{r:+.1f}</span>
</div>"""
    return f"<div style='max-height:240px;overflow-y:auto'>{rows}</div>"


def _btns(enabled: bool):
    return [gr.update(interactive=enabled)] * 9


def ui_reset(task_str: str):
    global _ui_log
    tid = int(str(task_str).split(":")[0].strip())
    _ui_task[0] = tid
    obs_obj = _env._reset_for_task(tid)
    obs = _obs_dict(obs_obj)
    _ui_obs.clear()
    _ui_obs.update(obs)
    _ui_log = []
    hint = f"💡 Hint: {TASKS[tid].hints[0]}"
    return [_status_html(obs), _log_html(_ui_log), hint] + _btns(True)


def ui_action(act_name: str, college_text: str):
    global _ui_log
    if not _ui_obs:
        return ["<p style='color:red;padding:12px'>Please click Reset first!</p>",
                _log_html(_ui_log), ""] + _btns(True)

    college = college_text.strip() if college_text and college_text.strip() else None
    rnd = int(_ui_obs.get("current_round", 1))

    try:
        action  = CollegeAction(action=act_name, target_college=college, round_number=rnd)
        obs_obj = _env.step(action)
        obs     = _obs_dict(obs_obj)
        reward  = float(obs.get("reward", 0.0))
        done    = bool(obs.get("done", False))

        _ui_obs.clear()
        _ui_obs.update(obs)
        _ui_log.append({"action": act_name, "msg": obs.get("message", "")[:90], "reward": reward})

        task  = TASKS[_ui_task[0]]
        idx   = min(int(obs.get("steps_taken", 0)), len(task.hints) - 1)
        hint  = ("✅ Done! Click Reset for a new episode." if done and obs.get("task_score", 0) >= 0.75
                 else "🔄 Try again — click Reset" if done
                 else f"💡 Hint {obs.get('steps_taken',0)}: {task.hints[idx]}" if task.hints else "")

        return [_status_html(obs), _log_html(_ui_log), hint] + _btns(not done)

    except Exception as e:
        return [f"<p style='color:red;padding:12px'>Error: {e}</p>",
                _log_html(_ui_log), ""] + _btns(True)


with gr.Blocks(title="College Admission Counselling — OpenEnv") as demo:

    gr.Markdown("""
# 🎓 College Admission Counselling Environment
**OpenEnv RL Environment — Meta AI Hackathon 2025**

Simulate India's JEE/CUET counselling. Help the student secure the best college!

> **API endpoints (same port):** `POST /reset` · `POST /step` · `GET /state` · `GET /health` · [`/docs`](/docs)
""")

    with gr.Row():
        with gr.Column(scale=1, min_width=270):
            gr.Markdown("### 1️⃣ Select Task")
            task_dd = gr.Dropdown(
                choices=[
                    "1: Simple Seat Acceptance (Easy)",
                    "2: Strategic Upgrade Decision (Medium)",
                    "3: Multi-Round Counselling (Hard)",
                ],
                value="1: Simple Seat Acceptance (Easy)",
                label="Task Difficulty",
            )
            reset_btn = gr.Button("🔄  Reset / Start Episode", variant="secondary", size="lg")

            gr.Markdown("### 2️⃣ Take Actions")
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
**Rewards:**
`check_status` +0.3 · `check_cutoffs` +0.5
`fill_choices` +1.0 · `accept` +2.0
`upgrade` +3.0 · `pay_fee` +2.0
`report` +3.0 · `withdraw` **−10** ☠️

**Task 1 correct order:**
Status → Accept → Pay Fee → Report
""")

        with gr.Column(scale=2):
            gr.Markdown("### Environment State")
            status_out = gr.HTML(value="<p style='text-align:center;color:#888;padding:20px'>Click Reset to begin</p>")
            hint_md    = gr.Markdown("")
            gr.Markdown("### Action Log")
            log_out    = gr.HTML(value="<p style='color:#888;font-size:12px;padding:8px'>No actions yet.</p>")

    gr.Markdown("---\n**Space:** [Knight09/college_admission_env](https://huggingface.co/spaces/Knight09/college_admission_env) · OpenEnv by Meta + HF · India JEE/CUET domain")

    outs = [status_out, log_out, hint_md,
            btn_cs, btn_cc, btn_fc, btn_acc, btn_up, btn_pay, btn_rep, btn_wd, reset_btn]

    reset_btn.click(fn=ui_reset, inputs=[task_dd], outputs=outs)
    btn_cs .click(fn=lambda c: ui_action("check_status",     c), inputs=[college_input], outputs=outs)
    btn_cc .click(fn=lambda c: ui_action("check_cutoffs",    c), inputs=[college_input], outputs=outs)
    btn_fc .click(fn=lambda c: ui_action("fill_choices",     c), inputs=[college_input], outputs=outs)
    btn_acc.click(fn=lambda c: ui_action("accept_allotment", c), inputs=[college_input], outputs=outs)
    btn_up .click(fn=lambda c: ui_action("upgrade_request",  c), inputs=[college_input], outputs=outs)
    btn_pay.click(fn=lambda c: ui_action("pay_seat_fee",     c), inputs=[college_input], outputs=outs)
    btn_rep.click(fn=lambda c: ui_action("report_to_college",c), inputs=[college_input], outputs=outs)
    btn_wd .click(fn=lambda c: ui_action("withdraw",         c), inputs=[college_input], outputs=outs)


# ── Add OpenEnv API routes to Gradio's internal FastAPI app ────────────────────
# This is the KEY: /reset /step /state /health all work on port 7860
# HF checker pings /reset → Gradio's app handles it → 200 OK ✓

_api_env = CollegeEnvironment()   # separate env instance for API calls


@demo.app.post("/reset")
async def api_reset():
    """OpenEnv reset — starts a new episode. Returns initial observation."""
    obs = _api_env._reset_for_task(1)
    return JSONResponse({
        "observation": _obs_dict(obs),
        "reward": 0.0,
        "done": False,
    })


@demo.app.post("/step")
async def api_step(request: Request):
    """OpenEnv step — takes one action. Returns observation + reward + done."""
    body = await request.json()
    # Support both: {"action": "check_status"} and {"action": {"action": "check_status"}}
    action_data = body.get("action", body)
    if isinstance(action_data, str):
        action_data = {"action": action_data}
    action = CollegeAction(**action_data)
    obs = _api_env.step(action)
    return JSONResponse({
        "observation": _obs_dict(obs),
        "reward": float(obs.reward),
        "done": bool(obs.done),
    })


@demo.app.get("/state")
async def api_state():
    """OpenEnv state — returns current episode state."""
    s = _api_env.state
    return JSONResponse({
        "episode_id": str(s.episode_id),
        "step_count": int(s.step_count),
    })


@demo.app.get("/health")
async def api_health():
    """Health check — returns 200 OK."""
    return JSONResponse({"status": "ok", "environment": "college_admission_env"})


@demo.app.get("/schema")
async def api_schema():
    """Returns action and observation schemas."""
    return JSONResponse({
        "action": CollegeAction.model_json_schema(),
        "observation": CollegeObservation.model_json_schema(),
    })


if __name__ == "__main__":
    print("Starting College Admission Counselling Environment...")
    print("  Gradio UI  → http://localhost:7860/")
    print("  POST /reset → http://localhost:7860/reset")
    print("  POST /step  → http://localhost:7860/step")
    print("  GET  /state → http://localhost:7860/state")
    print("  GET  /docs  → http://localhost:7860/docs")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
