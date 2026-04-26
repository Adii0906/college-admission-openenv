# College Admission OpenEnv: How it Works and Why It Matters

## What is `college_env`?
`college_env` is a simulated version of India’s college counselling process, designed to behave like a real JEE/CUET-style system such as JOSAA or CSAB.

It lets an agent practice the same choices students face during counselling: checking status, understanding cutoffs, filling preference lists, accepting allotments, asking for upgrades, paying fees, and reporting to college.

## How it works

### 1. A simple HTTP interface
The environment exposes a small OpenEnv HTTP API, so you can interact with it from code or a model.

- `POST /reset` — start a new episode
- `POST /step` — take one action and receive the next observation
- `GET /state` — see the current state
- `GET /schema` — view action and observation definitions
- `GET /docs` — open Swagger documentation

That makes it easy to plug this environment into reinforcement learning agents, baseline scripts, or evaluation code.

### 2. Three counselling tasks
There are three tasks of increasing difficulty:
- Easy: settle for a safe college within a short deadline
- Medium: decide whether to chase an upgrade over multiple rounds
- Hard: manage a longer counselling episode with tighter deadlines and more uncertainty

Each episode starts with a student profile and an initial allotment. The agent has a limited number of moves to reach the best possible outcome.

### 3. Real counselling actions
The action space includes steps that resemble real counselling decisions:
- `check_status`
- `check_cutoffs`
- `fill_choices`
- `lock_choices`
- `accept_allotment`
- `upgrade_request`
- `pay_seat_fee`
- `report_to_college`
- `withdraw`

These are the kinds of choices students and counsellors really make during the admission process.

### 4. Rewards and evaluation
The environment gives rewards based on the actions taken:
- small reward for staying informed (`check_status`, `check_cutoffs`)
- moderate reward for moving forward (`fill_choices`, `accept_allotment`)
- larger reward for securing a seat (`pay_seat_fee`, `report_to_college`)
- heavy penalty for withdrawing early

It also produces a task score and an episode summary so different agents can be compared.

## Why this matters

### Real student impact
Every year, millions of students go through competitive counselling. A wrong decision at the wrong time can mean losing a seat, or even a year of delay.

This environment tries to capture the real pressure of that process:
- limited rounds and strict deadlines
- rank-based eligibility and cutoff awareness
- choosing between a safe admission and a risky upgrade
- procedural steps like fee payment and reporting

### Why AI could help
Training agents on `college_env` helps us understand how models handle a multi-step decision process instead of just answering a prompt.

That could lead to tools for:
- helping students make better counselling choices
- supporting admission advisors with guidance
- testing whether models can follow step-by-step workflows
- teaching AI to respect deadlines and procedures

## Why OpenEnv and RL?

OpenEnv gives us a reusable interface that works with different agents and keeps evaluation consistent.

`college_env` is designed to be a compact benchmark with a strong educational story: real choices, real deadlines, and real consequences.

## In short
`college_env` is a practical environment for studying how AI handles the college counselling process.

It is not just about generating text — it is about planning, prioritizing, and following real-world steps in a high-stakes setting.