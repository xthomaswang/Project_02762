"""Role-based deck layout resolution and robot setup.

Translates role names (reagent, tiprack, plate, cleaning) to physical
labware IDs on the robot.  No slot numbers are hard-coded here — they
come exclusively from :class:`ColorMixingConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from src.tasks.color_mixing.config import ColorMixingConfig


@dataclass
class DeckProfile:
    """Resolved deck layout: maps roles to loaded labware IDs.

    This is the output of :func:`build_robot_calibration_profile` and
    the input consumed by step execution.
    """
    pipette_ids: Dict[str, str] = field(default_factory=dict)
    reagent_ids: Dict[str, str] = field(default_factory=dict)
    tiprack_ids: Dict[str, str] = field(default_factory=dict)
    plate_id: str = ""
    cleaning_id: str = ""

    # Original config for reference
    config: ColorMixingConfig = field(default=None, repr=False)

    def reagent_labware(self, role: str) -> str:
        """Get labware ID for a reagent role (e.g. 'red')."""
        return self.reagent_ids[role]

    def tiprack_for_mount(self, mount: str) -> str:
        """Get tiprack labware ID for a pipette mount."""
        return self.tiprack_ids[mount]


def build_robot_calibration_profile(
    cfg: ColorMixingConfig,
    robot: Any,
) -> DeckProfile:
    """Load all labware on the robot and return a :class:`DeckProfile`.

    This is the single place where role names are resolved to physical
    labware IDs via the robot API.  After this call, all subsequent
    operations use the profile — never raw slot numbers.

    Args:
        cfg: Loaded color mixing config.
        robot: An ``OT2Client`` instance.

    Returns:
        A :class:`DeckProfile` mapping every role to its labware ID.
    """
    profile = DeckProfile(config=cfg)

    # --- Pipettes ---
    for mount, pcfg in cfg.pipettes.items():
        profile.pipette_ids[mount] = robot.load_pipette(pcfg.name, mount)

    # --- Tipracks ---
    for mount, tcfg in cfg.tipracks.items():
        profile.tiprack_ids[mount] = robot.load_labware(tcfg.labware, tcfg.slot)

    # --- Reagent reservoirs ---
    for role, rcfg in cfg.reagents.items():
        profile.reagent_ids[role] = robot.load_labware(rcfg.labware, rcfg.slot)

    # --- Cleaning reservoir ---
    profile.cleaning_id = robot.load_labware(cfg.cleaning.labware, cfg.cleaning.slot)

    # --- Plate ---
    profile.plate_id = robot.load_labware(cfg.plate.labware, cfg.plate.slot)

    return profile
