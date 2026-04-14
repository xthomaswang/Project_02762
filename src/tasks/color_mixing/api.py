"""Public API for the color mixing task.

All entry points are importable from this module::

    from src.tasks.color_mixing.api import (
        build_iteration_steps,
        analyze_capture,
        fit_observation,
        build_robot_calibration_profile,
    )
"""

from src.tasks.color_mixing.steps import (  # noqa: F401
    Step,
    build_iteration_steps,
    build_control_steps,
    build_tip_check_steps,
    steps_to_runner_dicts,
    build_runner_steps,
    build_calibration_runner_steps,
    build_tip_check_runner_steps,
    execute_steps,
)
from src.tasks.color_mixing.observation import (  # noqa: F401
    CaptureResult,
    analyze_capture,
    fit_observation,
)
from src.tasks.color_mixing.deck import (  # noqa: F401
    DeckProfile,
    build_robot_calibration_profile,
)
