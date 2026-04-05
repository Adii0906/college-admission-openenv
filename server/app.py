# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the College Env Environment.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import CollegeAction, CollegeObservation
    from .college_env_environment import CollegeEnvironment
except (ModuleNotFoundError, ImportError):
    from models import CollegeAction, CollegeObservation
    from server.college_env_environment import CollegeEnvironment


# Create the FastAPI app
app = create_app(
    CollegeEnvironment,
    CollegeAction,
    CollegeObservation,
    env_name="college_env",
    max_concurrent_envs=1,
)


def main():
    """
    Entry point for running the server directly.

    Usage:
        python -m server.app
        python server/app.py
        python -m server.app --port 8001
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()