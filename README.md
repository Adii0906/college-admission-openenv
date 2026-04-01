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

## Architecture

```
HF Space
├── Gradio UI     → port 7860  (main, judges interact here)
└── FastAPI/OpenEnv → port 8000  (background, handles /reset /step /state)
```

Gradio calls FastAPI via HTTP. Both run in the same process.

## API Endpoints (port 8000)

```
POST /reset        Start new episode
POST /step         Take action
GET  /state        Current state
GET  /health       Health check → 200
GET  /docs         Interactive API docs
GET  /schema       Action/Observation schemas
```

## Step payload format

```json
POST /step
{"action": {"action": "check_status", "target_college": null, "round_number": 1}}
```

## Action Space

| Action | Reward | Description |
|--------|--------|-------------|
| `check_status` | +0.3 | View allotment and deadline |
| `check_cutoffs` | +0.5 | See eligible colleges by rank |
| `fill_choices` | +1.0 | Submit preference list |
| `accept_allotment` | +2.0 | Accept the college seat |
| `upgrade_request` | +3.0 | Request better college |
| `pay_seat_fee` | +2.0 | Pay fee to secure seat |
| `report_to_college` | +3.0 | Final step — done! |
| `withdraw` | **−10.0** | IRREVERSIBLE — avoid! |

## 3 Tasks

| Task | Difficulty | Max Steps | Perfect Score |
|------|-----------|-----------|---------------|
| 1 — Simple Seat Acceptance | Easy | 8 | 1.000 |
| 2 — Strategic Upgrade | Medium | 10 | 1.000 |
| 3 — Multi-Round Counselling | Hard | 15 | 0.900+ |

## Setup

```bash
# Set secrets in HF Space Settings
API_BASE_URL = https://api.groq.com/openai/v1
MODEL_NAME   = llama-3.1-8b-instant
HF_TOKEN     = your_groq_api_key_here

# Local run
pip install "gradio>=5.0.0" openenv-core openai groq requests python-dotenv
python app.py

# Run inference baseline
python inference.py
```

## Use via Python client

```python
import requests

# Reset
obs = requests.post("https://Knight09-college-admission-env.hf.space/reset").json()

# Step
result = requests.post(
    "https://Knight09-college-admission-env.hf.space/step",
    json={"action": {"action": "check_status", "target_college": None, "round_number": 1}}
).json()
print(result["reward"], result["done"])
```

*Built for India's first OpenEnv Hackathon — Meta + Hugging Face.*
