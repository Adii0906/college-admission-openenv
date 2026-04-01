# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""College Admission Counselling Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CollegeAction, CollegeObservation


class CollegeEnv(
    EnvClient[CollegeAction, CollegeObservation, State]
):
    """
    Client for the College Admission Counselling Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example:
        >>> with CollegeEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.message)
        ...
        ...     # Perfect Task 1 run
        ...     result = client.step(CollegeAction(action="check_status"))
        ...     result = client.step(CollegeAction(
        ...         action="accept_allotment",
        ...         target_college="NIT Warangal CS"
        ...     ))
        ...     result = client.step(CollegeAction(action="pay_seat_fee"))
        ...     result = client.step(CollegeAction(action="report_to_college"))
        ...     print(result.observation.task_score)  # 1.0

    Example with Docker:
        >>> client = CollegeEnv.from_docker_image("college_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CollegeAction(action="check_cutoffs"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CollegeAction) -> Dict:
        """Convert CollegeAction to JSON payload for WebSocket step message."""
        return {
            "action": action.action,
            "target_college": action.target_college,
            "round_number": action.round_number,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CollegeObservation]:
        """Parse server response into StepResult[CollegeObservation]."""
        obs_data = payload.get("observation", {})
        observation = CollegeObservation(
            student_rank=obs_data.get("student_rank", 0),
            student_category=obs_data.get("student_category", "GENERAL"),
            task_id=obs_data.get("task_id", 1),
            current_round=obs_data.get("current_round", 1),
            allotted_college=obs_data.get("allotted_college"),
            allotted_branch=obs_data.get("allotted_branch"),
            choices_filled=obs_data.get("choices_filled", False),
            seat_fee_paid=obs_data.get("seat_fee_paid", False),
            deadline_days_left=obs_data.get("deadline_days_left", 3),
            available_upgrades=obs_data.get("available_upgrades", []),
            steps_taken=obs_data.get("steps_taken", 0),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            task_score=obs_data.get("task_score", 0.0),
            message=obs_data.get("message", ""),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
