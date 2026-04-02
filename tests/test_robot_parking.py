"""Tests for src.robot_parking — the imaging park helper."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.robot import (
    DEFAULT_OFFSET,
    DEFAULT_SLOT,
    DEFAULT_WELL,
    ParkPosition,
    park_for_imaging,
    resolve_park_position,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_robot(loaded_slots=None):
    """Return a mock OT2Client with configurable labware_by_slot."""
    robot = MagicMock()
    loaded = loaded_slots or {DEFAULT_SLOT: "labware-id-11"}
    robot.labware_by_slot = loaded
    robot.get_labware_id = MagicMock(side_effect=lambda s: loaded[s])
    return robot


# ------------------------------------------------------------------
# ParkPosition dataclass
# ------------------------------------------------------------------

class TestParkPosition:
    def test_defaults(self):
        pos = ParkPosition()
        assert pos.slot == DEFAULT_SLOT
        assert pos.well == DEFAULT_WELL
        assert pos.offset == DEFAULT_OFFSET

    def test_custom_values(self):
        pos = ParkPosition(slot="5", well="B2", offset=(1.0, 2.0, 50.0))
        assert pos.slot == "5"
        assert pos.well == "B2"
        assert pos.offset == (1.0, 2.0, 50.0)

    def test_frozen(self):
        pos = ParkPosition()
        with pytest.raises(AttributeError):
            pos.slot = "3"  # type: ignore[misc]


# ------------------------------------------------------------------
# resolve_park_position
# ------------------------------------------------------------------

class TestResolveParkPosition:
    def test_default_fallback(self):
        """No imaging.park in config -> defaults to slot 11 / A1 / +30 Z."""
        robot = _make_robot()
        pos = resolve_park_position({}, robot)
        assert pos.slot == "11"
        assert pos.well == "A1"
        assert pos.offset == (0.0, 0.0, 30.0)
        assert pos.offset[2] > 0, "Z offset must be positive for safety"

    def test_explicit_config_override(self):
        """Config imaging.park section overrides defaults."""
        config = {
            "imaging": {
                "park": {
                    "slot": "10",
                    "well": "H12",
                    "offset": [1.0, -2.0, 50.0],
                }
            }
        }
        robot = _make_robot({"10": "labware-id-10"})
        pos = resolve_park_position(config, robot)
        assert pos.slot == "10"
        assert pos.well == "H12"
        assert pos.offset == (1.0, -2.0, 50.0)

    def test_partial_config_override(self):
        """Only some keys in imaging.park -> others fall back to defaults."""
        config = {"imaging": {"park": {"slot": "11"}}}
        robot = _make_robot()
        pos = resolve_park_position(config, robot)
        assert pos.slot == "11"
        assert pos.well == DEFAULT_WELL
        assert pos.offset == DEFAULT_OFFSET

    def test_slot_not_loaded_raises(self):
        """If the requested slot is not on the robot, raise ValueError."""
        robot = _make_robot({"7": "labware-id-7"})
        with pytest.raises(ValueError, match="Park slot '11' is not loaded"):
            resolve_park_position({}, robot)

    def test_slot_not_loaded_error_lists_loaded_slots(self):
        """The error message should list the loaded slots for debugging."""
        robot = _make_robot({"1": "lw-1", "7": "lw-7"})
        with pytest.raises(ValueError, match="Loaded slots"):
            resolve_park_position({}, robot)

    def test_offset_must_have_three_elements(self):
        config = {"imaging": {"park": {"offset": [0.0, 0.0]}}}
        robot = _make_robot()
        with pytest.raises(ValueError, match="exactly 3 elements"):
            resolve_park_position(config, robot)

    def test_empty_park_section_uses_defaults(self):
        """imaging.park: {} -> all defaults."""
        config = {"imaging": {"park": {}}}
        robot = _make_robot()
        pos = resolve_park_position(config, robot)
        assert pos == ParkPosition()

    def test_none_park_section_uses_defaults(self):
        """imaging.park: null -> all defaults."""
        config = {"imaging": {"park": None}}
        robot = _make_robot()
        pos = resolve_park_position(config, robot)
        assert pos == ParkPosition()


# ------------------------------------------------------------------
# park_for_imaging
# ------------------------------------------------------------------

class TestParkForImaging:
    def test_calls_move_to_well_with_correct_args(self):
        """park_for_imaging must delegate to robot.move_to_well."""
        robot = _make_robot()
        pos = ParkPosition(slot="11", well="A1", offset=(0.0, 0.0, 30.0))

        park_for_imaging(robot, pos)

        robot.get_labware_id.assert_called_once_with("11")
        robot.move_to_well.assert_called_once_with(
            labware_id="labware-id-11",
            well="A1",
            offset=(0.0, 0.0, 30.0),
        )

    def test_custom_position_forwarded(self):
        """Custom park position values are forwarded faithfully."""
        loaded = {"5": "labware-id-5"}
        robot = _make_robot(loaded)
        pos = ParkPosition(slot="5", well="B3", offset=(1.0, -1.0, 45.0))

        park_for_imaging(robot, pos)

        robot.get_labware_id.assert_called_once_with("5")
        robot.move_to_well.assert_called_once_with(
            labware_id="labware-id-5",
            well="B3",
            offset=(1.0, -1.0, 45.0),
        )

    def test_does_not_call_home(self):
        """Parking must NOT home the robot."""
        robot = _make_robot()
        pos = ParkPosition()
        park_for_imaging(robot, pos)
        robot.home.assert_not_called()


# ------------------------------------------------------------------
# Integration-style test (resolve + park in sequence)
# ------------------------------------------------------------------

class TestResolveAndPark:
    def test_end_to_end(self):
        """resolve -> park works as a two-step flow."""
        robot = _make_robot({"11": "lw-11"})
        pos = resolve_park_position({}, robot)
        park_for_imaging(robot, pos)

        robot.move_to_well.assert_called_once_with(
            labware_id="lw-11",
            well="A1",
            offset=(0.0, 0.0, 30.0),
        )
