"""Robot parking helper for imaging.

Moves the pipette to a safe non-imaging position so the camera has an
unobstructed view of the plate.  The helper uses the existing
``OT2Client.move_to_well`` method -- it does NOT home the robot.

Typical usage (future integration)::

    pos = resolve_park_position(config, robot)
    park_for_imaging(robot, pos)
    # ... capture image ...

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Default park target: slot 11 tiprack, well A1, with a large positive
# Z offset so the pipette hovers well above the labware and stays clear
# of the camera field of view over the plate (slot 1).
DEFAULT_SLOT = "11"
DEFAULT_WELL = "A1"
DEFAULT_OFFSET: Tuple[float, float, float] = (0.0, 0.0, 30.0)


@dataclass(frozen=True)
class ParkPosition:
    """Describes where the pipette should park during imaging.

    Attributes:
        slot: Deck slot containing the labware to park above.
        well: Well name within the labware.
        offset: ``(x, y, z)`` offset from the well top.  A positive
            *z* value keeps the pipette safely above the labware.
    """

    slot: str = DEFAULT_SLOT
    well: str = DEFAULT_WELL
    offset: Tuple[float, float, float] = field(default=DEFAULT_OFFSET)


def resolve_park_position(
    config: Dict[str, Any],
    robot: Any,
) -> ParkPosition:
    """Determine the park position from *config*, falling back to defaults.

    If ``config["imaging"]["park"]`` exists it should contain optional
    keys ``slot``, ``well``, and ``offset`` (a 3-element list).
    Missing keys fall back to the module defaults.

    The resolved slot must be present in ``robot.labware_by_slot``;
    otherwise a ``ValueError`` is raised so the caller gets a clear
    message rather than a cryptic HTTP 404 from the robot.

    Parameters:
        config: The full experiment configuration dictionary (e.g.
            loaded from ``experiment.yaml``).
        robot: An ``OT2Client`` instance (or any object exposing a
            ``labware_by_slot`` mapping).

    Returns:
        A ``ParkPosition`` ready to pass to :func:`park_for_imaging`.
    """

    park_cfg: Dict[str, Any] = (
        config.get("imaging", {}).get("park", {}) or {}
    )

    slot = str(park_cfg.get("slot", DEFAULT_SLOT))
    well = str(park_cfg.get("well", DEFAULT_WELL))
    offset_raw = park_cfg.get("offset", None)

    if offset_raw is not None:
        offset = tuple(float(v) for v in offset_raw)
        if len(offset) != 3:
            raise ValueError(
                f"imaging.park.offset must have exactly 3 elements, "
                f"got {len(offset)}: {offset_raw}"
            )
        offset = (offset[0], offset[1], offset[2])  # narrow tuple type
    else:
        offset = DEFAULT_OFFSET

    # Validate that the target slot is actually loaded on the robot.
    loaded = robot.labware_by_slot
    if slot not in loaded:
        raise ValueError(
            f"Park slot '{slot}' is not loaded on the robot. "
            f"Loaded slots: {sorted(loaded.keys())}. "
            f"Check imaging.park.slot in the config or ensure the "
            f"labware is loaded before calling resolve_park_position()."
        )

    pos = ParkPosition(slot=slot, well=well, offset=offset)
    logger.info("Resolved park position: %s", pos)
    return pos


def park_for_imaging(
    robot: Any,
    park_position: ParkPosition,
) -> None:
    """Move the pipette to *park_position* using ``robot.move_to_well``.

    This is a thin wrapper that resolves the labware ID from the slot
    and delegates to the existing OT-2 HTTP ``moveToWell`` command.
    It does **not** home the robot.

    Parameters:
        robot: An ``OT2Client`` instance.
        park_position: Target position (typically from
            :func:`resolve_park_position`).
    """

    labware_id = robot.get_labware_id(park_position.slot)

    logger.info(
        "Parking pipette: slot=%s labware=%s well=%s offset=%s",
        park_position.slot,
        labware_id,
        park_position.well,
        park_position.offset,
    )

    robot.move_to_well(
        labware_id=labware_id,
        well=park_position.well,
        offset=park_position.offset,
    )
