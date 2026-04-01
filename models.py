# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the College Admission Counselling Environment.

Simulates India's JEE/CUET college admission counselling process (JOSAA/CSAB).
The AI agent must help a student navigate seat allotment, upgrades, and deadlines.

Real-world context:
    - 1.5 million+ Indian students go through this every year
    - Wrong decisions (not upgrading, missing deadlines) cost students their dream college
    - This environment trains agents to make optimal counselling decisions
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Literal, List, Optional


class CollegeAction(Action):
    """
    What the AI agent can DO during counselling.

    These are the exact actions a student/counsellor takes
    in the real JOSAA/CSAB/state counselling process.
    """

    action: Literal[
        "check_cutoffs",      # Check opening/closing ranks for colleges
        "check_status",       # Check current allotment status
        "fill_choices",       # Fill college preference list
        "lock_choices",       # Lock the preference list (deadline action)
        "accept_allotment",   # Accept the allotted seat
        "upgrade_request",    # Request upgrade to better college in next round
        "pay_seat_fee",       # Pay the seat acceptance fee
        "report_to_college",  # Physically report to allotted college (final step)
        "withdraw",           # Withdraw from process (IRREVERSIBLE!)
    ] = Field(
        ...,
        description=(
            "Action to take in the counselling process. "
            "Correct order matters — wrong sequence loses the seat. "
            "'withdraw' is irreversible and ends the episode with heavy penalty."
        )
    )

    target_college: Optional[str] = Field(
        default=None,
        description=(
            "College name for fill_choices or accept_allotment actions. "
            "e.g. 'IIT Bombay CS' or 'NIT Warangal CS'"
        )
    )

    round_number: Optional[int] = Field(
        default=1,
        description="Counselling round number (1, 2, or 3). Matters for upgrade decisions."
    )


class CollegeObservation(Observation):
    """
    What the AI agent can SEE about the counselling state.

    Mirrors real information a student sees on the JOSAA/CSAB counselling portal.
    """

    # Student profile
    student_rank: int = Field(
        default=0,
        description="Student's JEE/CUET rank. Lower = better. e.g. rank 500 qualifies for IIT Bombay CS."
    )

    student_category: str = Field(
        default="GENERAL",
        description="Admission category: GENERAL / OBC / SC / ST / EWS. Affects cutoff ranks."
    )

    task_id: int = Field(
        default=1,
        description="Current task: 1=Easy (simple acceptance), 2=Medium (upgrade), 3=Hard (multi-round)"
    )

    # Counselling state
    current_round: int = Field(
        default=1,
        description="Current counselling round (1=first allotment, 2=upgrade round, 3=final round)"
    )

    allotted_college: Optional[str] = Field(
        default=None,
        description="College currently allotted to the student. None = no allotment yet."
    )

    allotted_branch: Optional[str] = Field(
        default=None,
        description="Branch/course allotted. e.g. 'Computer Science', 'Mechanical Engineering'"
    )

    choices_filled: bool = Field(
        default=False,
        description="True if student has filled and locked their college preference list."
    )

    seat_fee_paid: bool = Field(
        default=False,
        description="True if seat acceptance fee has been paid. Required to hold the seat."
    )

    deadline_days_left: int = Field(
        default=3,
        description="Days left to complete current required action. 0 = deadline missed!"
    )

    available_upgrades: List[str] = Field(
        default_factory=list,
        description="List of better colleges the student can upgrade to in this round."
    )

    # Episode tracking
    steps_taken: int = Field(
        default=0,
        description="Total actions taken this episode."
    )

    reward: float = Field(
        default=0.0,
        description="Reward for the last action. Positive = good move, Negative = bad move."
    )

    done: bool = Field(
        default=False,
        description="True when episode ends (seat secured, deadline missed, or withdrew)."
    )

    message: str = Field(
        default="",
        description="Human-readable explanation of what just happened and what to do next."
    )

    task_score: float = Field(
        default=0.0,
        description="Current task completion score between 0.0 (failed) and 1.0 (perfect)."
    )
