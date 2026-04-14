"""Color mixing assay — active learning optimization of dye volumes.

Public API (re-exported from :mod:`api`)::

    from src.tasks.color_mixing import (
        build_iteration_steps,
        analyze_capture,
        fit_observation,
        build_robot_calibration_profile,
        load_task_config,
        ColorMixingConfig,
        ColorMixingPlugin,
    )
"""

from src.tasks.color_mixing.api import (  # noqa: F401
    build_iteration_steps,
    analyze_capture,
    fit_observation,
    build_robot_calibration_profile,
)
from src.tasks.color_mixing.config import (  # noqa: F401
    ColorMixingConfig,
    load_task_config,
)
from src.tasks.color_mixing.plugin import ColorMixingPlugin  # noqa: F401
