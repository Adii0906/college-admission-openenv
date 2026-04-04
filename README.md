---
title: College Admission Counselling Env
emoji: 🎓
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.29.1"
app_file: app.py
pinned: true
python_version: "3.11"
tags:
  - openenv
  - reinforcement-learning
  - india
  - education
  - real-world
---

# 🎓 College Admission Counselling Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-purple)](https://github.com/meta-pytorch/OpenEnv)

Simulates India's **JEE/CUET college admission counselling** (JOSAA/CSAB style).
1.5 million+ students face this every year. Wrong decisions cost students their dream college.

## API Endpoints (port 7860 — same as Gradio UI)

```
POST /reset     Start new episode → returns observation
POST /step      Take action → returns observation + reward + done
GET  /state     Current episode state
GET  /health    Health check → {"status": "ok"}
GET  /schema    Action + Observation schemas
GET  /docs      Swagger API docs
```

## Step payload format

```json
POST /step
{"action": {"action": "check_status", "target_college": null, "round_number": 1}}

or flat format:
{"action": "check_status"}
```

## Quick test

```bash
# Reset
curl -X POST https://Knight09-college-admission-env.hf.space/reset

# Step
curl -X POST https://Knight09-college-admission-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": "check_status"}'
```

## Action Space

| Action | Reward | Description |
|--------|--------|-------------|
| `check_status` | +0.3 | View allotment and steps left |
| `check_cutoffs` | +0.5 | Check eligible colleges by rank |
| `fill_choices` | +1.0 | Submit college preference list |
| `accept_allotment` | +2.0 | Accept the allotted college |
| `upgrade_request` | +3.0 | Request upgrade to better college |
| `pay_seat_fee` | +2.0 | Pay fee to secure seat |
| `report_to_college` | +3.0 | Final step — done! |
| `withdraw` | **−10.0** | IRREVERSIBLE — avoid! |

## 3 Tasks

| # | Difficulty | Description | Max Steps |
|---|-----------|-------------|-----------|
| 1 | Easy | Accept NIT Warangal CS before deadline | 8 |
| 2 | Medium | Upgrade IIT Kharagpur → IIT Madras CS | 10 |
| 3 | Hard | Multi-round: get IIT Bombay CS | 15 |

## Baseline Scores (Llama 3.1-8B via Groq)

| Task | Difficulty | Score |
|------|-----------|-------|
| 1 | Easy | TBD |
| 2 | Medium | TBD |
| 3 | Hard | TBD |

Run `python inference.py` to reproduce.

## Setup

```bash
pip install "gradio>=5.0.0" openenv-core openai groq requests python-dotenv

# HF Space secrets Currently uing GROQ free api key you can replace with openai key just go to inference.py and uncomment the openai client part
API_BASE_URL = https://api.groq.com/openai/v1
MODEL_NAME   = llama-3.1-8b-instant
HF_TOKEN     = your_groq_key

python app.py        # Starts on localhost:7860
python inference.py  # Baseline evaluation
```

*Built for India's first OpenEnv Hackathon — Meta + Hugging Face.*
