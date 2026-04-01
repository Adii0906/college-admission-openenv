# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
College Admission Counselling Environment Implementation.

Simulates India's JEE/CUET college admission counselling (JOSAA/CSAB).
The AI agent helps students navigate seat allotment, upgrades,
fee payment, and deadline management.

Real-world basis: 1.5 million+ Indian students go through this every year.
"""

from uuid import uuid4
from typing import Optional, List, Dict

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CollegeAction, CollegeObservation
    from ..tasks import TASKS, GRADERS, get_eligible_colleges, get_best_college, COLLEGES
except ImportError:
    from models import CollegeAction, CollegeObservation
    from tasks import TASKS, GRADERS, get_eligible_colleges, get_best_college, COLLEGES


class CollegeEnvironment(Environment):
    """
    College Admission Counselling Environment.

    3 tasks of increasing difficulty:
      Task 1 (EASY)   — Simple seat acceptance, max 8 steps
      Task 2 (MEDIUM) — Strategic upgrade decision, max 10 steps
      Task 3 (HARD)   — Multi-round counselling, max 15 steps

    Example:
        >>> env = CollegeEnvironment()
        >>> obs = env.reset()
        >>> obs = env.step(CollegeAction(action="check_status"))
        >>> obs = env.step(CollegeAction(action="accept_allotment",
        ...                               target_college="NIT Warangal CS"))
        >>> obs = env.step(CollegeAction(action="pay_seat_fee"))
        >>> obs = env.step(CollegeAction(action="report_to_college"))
        >>> print(obs.task_score)  # 1.0
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id: int = 1
        self._student_rank: int = 0
        self._student_category: str = "GENERAL"
        self._current_round: int = 1
        self._allotted_college: Optional[str] = None
        self._allotted_branch: Optional[str] = None
        self._choices_filled: bool = False
        self._seat_fee_paid: bool = False
        self._deadline_days: int = 8   # Fixed: enough steps to complete
        self._episode_log: List[Dict] = []
        self._done: bool = False
        self._total_reward: float = 0.0

    def reset(self) -> CollegeObservation:
        """Reset to Task 1 (Easy). Returns initial observation."""
        return self._reset_for_task(1)

    def _reset_for_task(self, task_id: int = 1) -> CollegeObservation:
        """Internal: reset environment for a specific task."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = max(1, min(3, task_id))
        task = TASKS[self._task_id]

        self._student_rank = task.student_rank
        self._student_category = task.student_category
        self._current_round = 1
        self._allotted_college = task.initial_allotment
        self._allotted_branch = "Computer Science"
        self._choices_filled = False
        self._seat_fee_paid = False
        self._deadline_days = task.max_steps  # deadline = max steps
        self._episode_log = []
        self._done = False
        self._total_reward = 0.0

        eligible = get_eligible_colleges(self._student_rank, self._student_category)
        upgrades = self._get_upgrades(eligible)

        return CollegeObservation(
            student_rank=self._student_rank,
            student_category=self._student_category,
            task_id=self._task_id,
            current_round=self._current_round,
            allotted_college=self._allotted_college,
            allotted_branch=self._allotted_branch,
            choices_filled=self._choices_filled,
            seat_fee_paid=self._seat_fee_paid,
            deadline_days_left=self._deadline_days,
            available_upgrades=upgrades,
            steps_taken=0,
            reward=0.0,
            done=False,
            task_score=0.0,
            message=(
                f"[Task {self._task_id} — {task.difficulty}] {task.description} "
                f"| Rank: {self._student_rank} ({self._student_category}) "
                f"| Allotment: {self._allotted_college or 'None'} "
                f"| Steps allowed: {task.max_steps}. "
                f"Hint: {task.hints[0]}"
            )
        )

    def step(self, action: CollegeAction) -> CollegeObservation:  # type: ignore[override]
        """
        Process one counselling action.

        Args:
            action: CollegeAction with action name and optional target_college

        Returns:
            CollegeObservation with updated state, reward, and task_score
        """
        # Allow task switching via special round_number: 11=task1, 22=task2, 33=task3
        if action.round_number in [11, 22, 33]:
            task_map = {11: 1, 22: 2, 33: 3}
            return self._reset_for_task(task_map[action.round_number])

        if self._done:
            return self._make_obs(0.0, "Episode done. Call reset() to start again.", force_done=True)

        self._state.step_count += 1
        self._deadline_days -= 1

        reward = 0.0
        message = ""

        # Log for grader
        self._episode_log.append({
            "action": action.action,
            "target_college": action.target_college,
            "round_number": action.round_number or self._current_round,
            "step": self._state.step_count,
        })

        # ── check_status ───────────────────────────────────────
        if action.action == "check_status":
            reward = 0.3
            message = (
                f"Round {self._current_round} | "
                f"Allotted: {self._allotted_college or 'None'} | "
                f"Fee paid: {'Yes' if self._seat_fee_paid else 'No'} | "
                f"Steps left: {self._deadline_days}."
            )

        # ── check_cutoffs ──────────────────────────────────────
        elif action.action == "check_cutoffs":
            eligible = get_eligible_colleges(self._student_rank, self._student_category)
            best = get_best_college(self._student_rank, self._student_category)
            reward = 0.5
            message = (
                f"You qualify for {len(eligible)} colleges. "
                f"Best: {best}. Your rank {self._student_rank} ({self._student_category})."
            )

        # ── fill_choices ───────────────────────────────────────
        elif action.action == "fill_choices":
            if not self._choices_filled:
                self._choices_filled = True
                reward = 1.0
                message = (
                    f"Choices submitted for Round {self._current_round}. "
                    f"Target: {action.target_college or 'not specified'}."
                )
            else:
                reward = -0.5
                message = "Choices already locked this round."

        # ── lock_choices ───────────────────────────────────────
        elif action.action == "lock_choices":
            if self._choices_filled:
                reward = 0.3
                message = "Choices locked. Waiting for allotment result."
            else:
                reward = -1.0
                message = "ERROR: Fill choices before locking!"

        # ── accept_allotment ───────────────────────────────────
        elif action.action == "accept_allotment":
            if self._allotted_college:
                target = action.target_college or self._allotted_college
                if target == self._allotted_college:
                    reward = 2.0
                    message = (
                        f"Allotment accepted: {self._allotted_college}. "
                        "Next: pay the seat fee!"
                    )
                else:
                    reward = -1.0
                    message = (
                        f"ERROR: {target} is not your allotment. "
                        f"Your allotment is {self._allotted_college}."
                    )
            else:
                reward = -1.5
                message = "ERROR: No allotment yet. Check status first."

        # ── upgrade_request ────────────────────────────────────
        elif action.action == "upgrade_request":
            if self._current_round >= 3:
                reward = -1.0
                message = "Round 3 is final. No more upgrades."
            else:
                self._current_round = min(3, self._current_round + 1)
                # Task-specific upgrade outcomes
                if self._task_id == 2 and self._current_round == 2:
                    self._allotted_college = "IIT Madras CS"
                    self._choices_filled = False  # reset for new round
                    reward = 3.0
                    message = "UPGRADE! New allotment: IIT Madras CS. Accept and pay fee."
                elif self._task_id == 3 and self._current_round == 2:
                    self._allotted_college = "IIT Bombay CS"
                    self._choices_filled = False
                    reward = 3.0
                    message = "UPGRADE! New allotment: IIT Bombay CS — dream college! Accept now!"
                else:
                    reward = 0.5
                    message = f"Upgrade processed. Round {self._current_round}. Allotment: {self._allotted_college}."

        # ── pay_seat_fee ───────────────────────────────────────
        elif action.action == "pay_seat_fee":
            if not self._seat_fee_paid:
                self._seat_fee_paid = True
                fee = COLLEGES.get(self._allotted_college or "", {}).get("fee", 70000)
                reward = 2.0
                message = (
                    f"Fee Rs.{fee:,} paid for {self._allotted_college}. "
                    "Seat SECURED! Report to college to complete."
                )
            else:
                reward = 0.1
                message = "Fee already paid. Report to college."

        # ── report_to_college ──────────────────────────────────
        elif action.action == "report_to_college":
            if self._seat_fee_paid:
                self._done = True
                reward = 3.0
                message = (
                    f"SUCCESS! Reported to {self._allotted_college}. "
                    "Admission complete! Congratulations!"
                )
            else:
                reward = -1.5
                message = "ERROR: Pay seat fee before reporting to college!"

        # ── withdraw ───────────────────────────────────────────
        elif action.action == "withdraw":
            self._done = True
            reward = -10.0
            message = (
                "CATASTROPHE! Withdrew from counselling — IRREVERSIBLE. "
                "All allotments lost!"
            )

        # ── deadline and max steps ─────────────────────────────
        task = TASKS[self._task_id]
        if self._state.step_count >= task.max_steps and not self._done:
            self._done = True
            message += f" | Max steps ({task.max_steps}) reached."

        self._total_reward += reward
        return self._make_obs(reward, message)

    def _get_upgrades(self, eligible: List[str]) -> List[str]:
        return [
            c for c in eligible
            if c != self._allotted_college
            and COLLEGES.get(c, {}).get("rank_cutoff", 9999)
            < COLLEGES.get(self._allotted_college or "", {}).get("rank_cutoff", 0)
        ][:3]

    def _make_obs(self, reward: float, message: str, force_done: bool = False) -> CollegeObservation:
        """Build a CollegeObservation from current state."""
        task_score = GRADERS[self._task_id](self._episode_log)
        eligible = get_eligible_colleges(self._student_rank, self._student_category)
        upgrades = self._get_upgrades(eligible)
        return CollegeObservation(
            student_rank=self._student_rank,
            student_category=self._student_category,
            task_id=self._task_id,
            current_round=self._current_round,
            allotted_college=self._allotted_college,
            allotted_branch=self._allotted_branch,
            choices_filled=self._choices_filled,
            seat_fee_paid=self._seat_fee_paid,
            deadline_days_left=max(0, self._deadline_days),
            available_upgrades=upgrades,
            steps_taken=self._state.step_count,
            reward=round(reward, 3),
            done=self._done or force_done,
            task_score=task_score,
            message=message,
        )

    @property
    def state(self) -> State:
        """Return current episode state (episode_id and step_count)."""
        return self._state
