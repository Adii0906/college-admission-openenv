# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tasks and Graders for College Admission Counselling Environment.

3 Tasks with increasing difficulty:
    Task 1 (Easy)   - Simple seat acceptance, clear path, 6 steps max
    Task 2 (Medium) - Strategic upgrade across 2 rounds, 10 steps max
    Task 3 (Hard)   - Multi-round counselling with tight deadlines, 15 steps max

Graders return float 0.0 -> 1.0:
    0.0  = completely failed (withdrew or missed deadline with nothing done)
    0.5  = partial success (got a seat but not the optimal one)
    1.0  = perfect (best possible college in correct sequence)

All graders are DETERMINISTIC — same actions always produce same score.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ─────────────────────────────────────────────────────────────
# College data — realistic Indian colleges with cutoff ranks
# ─────────────────────────────────────────────────────────────

COLLEGES: Dict[str, Dict] = {
    "IIT Bombay CS":    {"rank_cutoff": 500,   "score": 100, "fee": 100000},
    "IIT Delhi CS":     {"rank_cutoff": 600,   "score": 98,  "fee": 100000},
    "IIT Madras CS":    {"rank_cutoff": 800,   "score": 96,  "fee": 100000},
    "IIT Kharagpur CS": {"rank_cutoff": 1200,  "score": 90,  "fee": 100000},
    "IIT Roorkee CS":   {"rank_cutoff": 1500,  "score": 87,  "fee": 100000},
    "NIT Trichy CS":    {"rank_cutoff": 3000,  "score": 78,  "fee": 70000},
    "NIT Warangal CS":  {"rank_cutoff": 4000,  "score": 75,  "fee": 70000},
    "NIT Surathkal CS": {"rank_cutoff": 5000,  "score": 72,  "fee": 70000},
    "NIT Calicut CS":   {"rank_cutoff": 7000,  "score": 65,  "fee": 70000},
    "VIT Vellore CS":   {"rank_cutoff": 15000, "score": 50,  "fee": 180000},
    "SRM Chennai CS":   {"rank_cutoff": 25000, "score": 40,  "fee": 150000},
}

CATEGORY_MULTIPLIERS = {
    "GENERAL": 1.0,
    "OBC":     1.3,
    "SC":      2.0,
    "ST":      2.5,
    "EWS":     1.2,
}


def get_eligible_colleges(rank: int, category: str = "GENERAL") -> List[str]:
    """Return colleges the student is eligible for based on rank and category."""
    factor = CATEGORY_MULTIPLIERS.get(category, 1.0)
    return [
        name for name, data in COLLEGES.items()
        if rank <= data["rank_cutoff"] * factor
    ]


def get_best_college(rank: int, category: str = "GENERAL") -> Optional[str]:
    """Return the best (lowest cutoff) college a student qualifies for."""
    eligible = get_eligible_colleges(rank, category)
    if not eligible:
        return None
    return min(eligible, key=lambda c: COLLEGES[c]["rank_cutoff"])


# ─────────────────────────────────────────────────────────────
# Task definitions
# ─────────────────────────────────────────────────────────────

@dataclass
class Task:
    task_id: int
    name: str
    description: str
    difficulty: str
    student_rank: int
    student_category: str
    max_steps: int
    initial_allotment: Optional[str]
    target_outcome: str
    optimal_actions: List[str]
    hints: List[str] = field(default_factory=list)


TASKS: Dict[int, Task] = {
    1: Task(
        task_id=1,
        name="Simple Seat Acceptance",
        difficulty="EASY",
        description=(
            "Rahul has JEE rank 4500 (OBC category). He has been allotted "
            "NIT Warangal CS in Round 1. He must accept the seat, pay the fee, "
            "and report to college before the 3-day deadline. No upgrade is available."
        ),
        student_rank=4500,
        student_category="OBC",
        max_steps=6,
        initial_allotment="NIT Warangal CS",
        target_outcome="accept_seat_and_report",
        optimal_actions=["check_status", "accept_allotment", "pay_seat_fee", "report_to_college"],
        hints=[
            "First check your allotment status",
            "Accept the allotment",
            "Pay the seat acceptance fee",
            "Report to the college to complete admission",
        ]
    ),

    2: Task(
        task_id=2,
        name="Strategic Upgrade Decision",
        difficulty="MEDIUM",
        description=(
            "Priya has JEE rank 1300 (GENERAL). Round 1 allotted IIT Kharagpur CS. "
            "Round 2 shows IIT Madras CS is available as an upgrade. "
            "She must check cutoffs, fill new choices, request upgrade, "
            "accept IIT Madras, and pay the fee. Staying at IIT Kharagpur is a mistake."
        ),
        student_rank=1300,
        student_category="GENERAL",
        max_steps=10,
        initial_allotment="IIT Kharagpur CS",
        target_outcome="upgrade_to_iit_madras",
        optimal_actions=["check_cutoffs", "fill_choices", "upgrade_request", "accept_allotment", "pay_seat_fee"],
        hints=[
            "Check cutoffs to confirm IIT Madras is within reach",
            "Fill new choices with IIT Madras CS at top",
            "Request upgrade in round 2",
            "Accept new allotment (IIT Madras CS)",
            "Pay seat fee immediately",
        ]
    ),

    3: Task(
        task_id=3,
        name="Multi-Round Complex Counselling",
        difficulty="HARD",
        description=(
            "Arjun has JEE rank 550 (GENERAL). He wants IIT Bombay CS (cutoff 500). "
            "Round 1: Gets IIT Delhi CS (his rank just misses Bombay). "
            "Round 2: IIT Bombay CS cutoff relaxes — upgrade opportunity. "
            "Round 3: Final decision with only 1 day left. "
            "Must accept in Round 1 to hold a seat (never leave unallotted), "
            "then upgrade strategically in Round 2. "
            "This genuinely challenges frontier models."
        ),
        student_rank=550,
        student_category="GENERAL",
        max_steps=15,
        initial_allotment="IIT Delhi CS",
        target_outcome="secure_iit_bombay",
        optimal_actions=[
            "check_status", "accept_allotment",   # Round 1: hold seat
            "check_cutoffs", "fill_choices", "upgrade_request",  # Round 2: upgrade
            "accept_allotment", "pay_seat_fee", "report_to_college",  # Round 2/3: complete
        ],
        hints=[
            "CRITICAL: Accept IIT Delhi in Round 1 — never leave yourself unallotted",
            "In Round 2: check if IIT Bombay cutoff has relaxed",
            "Fill new choices with IIT Bombay CS at the top",
            "Request upgrade to IIT Bombay",
            "If allotted IIT Bombay, accept immediately and pay fee",
            "NEVER withdraw — you will lose everything",
        ]
    ),
}


# ─────────────────────────────────────────────────────────────
# Grader functions — deterministic, return float 0.0 to 1.0
# ─────────────────────────────────────────────────────────────

def grade_task_1(episode_log: List[Dict]) -> float:
    """
    Task 1 Grader — Simple seat acceptance.

    Partial scoring:
        +0.25  checked status or cutoffs (smart start)
        +0.25  accepted allotment correctly
        +0.25  paid seat fee
        +0.25  reported to college (full completion)
        -0.50  withdrew (catastrophic)
    """
    score = 0.0
    actions = [e["action"] for e in episode_log]

    if "check_status" in actions or "check_cutoffs" in actions:
        score += 0.25
    if "accept_allotment" in actions:
        score += 0.25
    if "pay_seat_fee" in actions:
        score += 0.25
    if "report_to_college" in actions:
        score += 0.25
    if "withdraw" in actions:
        score -= 0.50

    return round(max(0.0, min(1.0, score)), 3)


def grade_task_2(episode_log: List[Dict]) -> float:
    """
    Task 2 Grader — Strategic upgrade decision.

    Partial scoring:
        +0.10  checked cutoffs (research)
        +0.15  filled choices
        +0.20  requested upgrade
        +0.25  accepted IIT Madras (correct upgrade!)
        +0.20  paid seat fee
        +0.10  completed in <= 8 steps (efficiency bonus)
        -0.40  withdrew
        -0.30  accepted IIT Kharagpur instead of upgrading (missed opportunity)
    """
    score = 0.0
    actions = [e["action"] for e in episode_log]
    colleges = [e.get("target_college") or "" for e in episode_log]

    if "check_cutoffs" in actions:
        score += 0.10
    if "fill_choices" in actions:
        score += 0.15
    if "upgrade_request" in actions:
        score += 0.20

    if "accept_allotment" in actions:
        if "IIT Madras CS" in colleges:
            score += 0.25  # Correct upgrade!
        elif "IIT Kharagpur CS" in colleges:
            score -= 0.30  # Stayed when should have upgraded

    if "pay_seat_fee" in actions:
        score += 0.20
    if len(actions) <= 8:
        score += 0.10  # Efficiency bonus
    if "withdraw" in actions:
        score -= 0.40

    return round(max(0.0, min(1.0, score)), 3)


def grade_task_3(episode_log: List[Dict]) -> float:
    """
    Task 3 Grader — Multi-round complex counselling.

    Partial scoring:
        +0.10  accepted IIT Delhi in round 1 (holding strategy)
        +0.10  checked cutoffs in round 2
        +0.10  filled choices in round 2
        +0.15  requested upgrade in round 2
        +0.25  secured IIT Bombay CS
        +0.15  paid seat fee
        +0.10  reported to college
        +0.05  completed in <= 12 steps
        -0.50  withdrew (catastrophic)
    """
    score = 0.0
    actions = [e["action"] for e in episode_log]
    colleges = [e.get("target_college") or "" for e in episode_log]

    # Round 1: accepted IIT Delhi (holding strategy)
    r1 = [e for e in episode_log if e.get("round_number", 1) == 1]
    if any(e["action"] == "accept_allotment" for e in r1):
        score += 0.10

    if "check_cutoffs" in actions:
        score += 0.10

    # Round 2: filled choices
    r2_fills = [e for e in episode_log
                if e["action"] == "fill_choices" and e.get("round_number", 1) >= 2]
    if r2_fills:
        score += 0.10

    if "upgrade_request" in actions:
        score += 0.15

    # Got IIT Bombay
    if "accept_allotment" in actions and "IIT Bombay CS" in colleges:
        score += 0.25

    if "pay_seat_fee" in actions:
        score += 0.15
    if "report_to_college" in actions:
        score += 0.10
    if len(actions) <= 12:
        score += 0.05

    if "withdraw" in actions:
        score -= 0.50

    return round(max(0.0, min(1.0, score)), 3)


GRADERS = {
    1: grade_task_1,
    2: grade_task_2,
    3: grade_task_3,
}
