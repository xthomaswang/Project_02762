"""Generate per-column protocol steps from role-driven config.

Single source of truth
~~~~~~~~~~~~~~~~~~~~~~
:func:`build_iteration_steps` produces :class:`Step` dataclass objects —
the **canonical** representation of liquid-handling semantics.

:func:`steps_to_runner_dicts` converts Step objects into runner-ready
dicts (``RunStep``-compatible) by resolving roles to config slot values.
All other ``build_*`` functions are derived from these two primitives so
that protocol semantics are defined in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.tasks.color_mixing.config import ColorMixingConfig


# ======================================================================
# Canonical step model
# ======================================================================

@dataclass
class Step:
    """One liquid-handling action in the per-column protocol."""

    phase: str           # "control" or "experiment"
    action: str          # "transfer", "mix", "use_pipette", "pick_up_tip", "drop_tip", "home"
    reagent: str         # role name: "red", "green", "blue", "water", "mix", ""
    pipette_mount: str   # "left" or "right"
    tiprack_mount: str   # which tiprack to use (by mount key)
    tip_well: str        # well on the tiprack to pick up
    source_role: str     # reagent role for the source labware
    source_well: str     # well on the source labware
    dest_well: str       # well on the plate
    volume: float        # µL to transfer (0 means skip)
    rinse_well: str      # well on the cleaning reservoir

    # Mix-specific (only used when action == "mix")
    mix_cycles: int = 0
    mix_volume: float = 0.0


# ======================================================================
# Canonical builder: build_iteration_steps
# ======================================================================

def build_iteration_steps(
    cfg: ColorMixingConfig,
    volumes: np.ndarray,
    col_idx: int,
    *,
    skip_controls: bool = False,
    mix_cycles_override: Optional[int] = None,
) -> List[Step]:
    """Build the complete step list for one column iteration.

    This is the **single source of truth** for liquid-handling semantics.
    All other step builders derive from or mirror this function.

    Args:
        cfg: Color mixing config.
        volumes: (3,) array [V_red, V_green, V_blue].
        col_idx: 0-based column index on the plate.
        skip_controls: If True, omit Phase 1 (control dispensing).
        mix_cycles_override: Override ``cfg.experiment.mix_cycles``.

    Returns:
        List of :class:`Step` objects in execution order.
    """
    well_col = col_idx + 1
    Vr, Vg, Vb = float(volumes[0]), float(volumes[1]), float(volumes[2])
    mix_cycles = mix_cycles_override or cfg.experiment.mix_cycles
    mix_volume = cfg.experiment.mix_volume_ul
    control_volume = cfg.well_roles.control_volume_ul

    dye_volumes = {"red": Vr, "green": Vg, "blue": Vb}
    steps: List[Step] = []

    # Phase 1: Controls (left pipette)
    if not skip_controls:
        for row, role in cfg.well_roles.controls.items():
            steps.append(Step(
                phase="control",
                action="transfer",
                reagent=role,
                pipette_mount="left",
                tiprack_mount="left",
                tip_well=cfg.tip_wells.get(role, "A1"),
                source_role=role,
                source_well=cfg.reagents[role].well,
                dest_well=f"{row}{well_col}",
                volume=control_volume,
                rinse_well=cfg.cleaning.rinse_wells.get("control", "A6"),
            ))

    # Phase 2: Experiments (right pipette)
    for dye_name in ("red", "green", "blue"):
        vol = dye_volumes[dye_name]
        tip_col = cfg.tip_columns.get(dye_name, 1)
        steps.append(Step(
            phase="experiment",
            action="transfer",
            reagent=dye_name,
            pipette_mount="right",
            tiprack_mount="right",
            tip_well=f"A{tip_col}",
            source_role=dye_name,
            source_well=cfg.reagents[dye_name].well,
            dest_well=f"A{well_col}",
            volume=vol,
            rinse_well=cfg.cleaning.rinse_wells.get(dye_name, "A1"),
        ))

    # Mix
    mix_tip_col = cfg.tip_columns.get("mix", 4)
    steps.append(Step(
        phase="experiment",
        action="mix",
        reagent="mix",
        pipette_mount="right",
        tiprack_mount="right",
        tip_well=f"A{mix_tip_col}",
        source_role="mix",
        source_well="",
        dest_well=f"A{well_col}",
        volume=0.0,
        rinse_well=cfg.cleaning.rinse_wells.get("mix", "A4"),
        mix_cycles=mix_cycles,
        mix_volume=mix_volume,
    ))

    return steps


def build_control_steps(cfg: ColorMixingConfig) -> List[Step]:
    """Build control-only steps for column 1 (calibration).

    Derived from the same logic as the control portion of
    :func:`build_iteration_steps`.
    """
    dummy_volumes = np.array([0.0, 0.0, 0.0])
    all_steps = build_iteration_steps(cfg, dummy_volumes, col_idx=0,
                                      skip_controls=False)
    return [s for s in all_steps if s.phase == "control"]


def build_tip_check_steps(cfg: ColorMixingConfig) -> List[Step]:
    """Build tip pick-up/drop steps for a tip-presence check.

    Returns Step objects with action ``pick_up_tip``, ``drop_tip``,
    and ``home``.
    """
    steps: List[Step] = []
    _blank = Step(
        phase="check", action="", reagent="", pipette_mount="",
        tiprack_mount="", tip_well="", source_role="", source_well="",
        dest_well="", volume=0.0, rinse_well="",
    )

    # Left pipette tips
    for well in (list(cfg.tip_wells.values()) or ["A1"]):
        steps.append(Step(
            **{**_blank.__dict__,
               "action": "pick_up_tip", "pipette_mount": "left",
               "tiprack_mount": "left", "tip_well": well}))
        steps.append(Step(
            **{**_blank.__dict__,
               "action": "drop_tip", "pipette_mount": "left",
               "tiprack_mount": "left"}))

    # Right pipette tips
    for col in (list(cfg.tip_columns.values()) or [1]):
        steps.append(Step(
            **{**_blank.__dict__,
               "action": "pick_up_tip", "pipette_mount": "right",
               "tiprack_mount": "right", "tip_well": f"A{col}"}))
        steps.append(Step(
            **{**_blank.__dict__,
               "action": "drop_tip", "pipette_mount": "right",
               "tiprack_mount": "right"}))

    steps.append(Step(
        **{**_blank.__dict__, "action": "home"}))
    return steps


# ======================================================================
# Adapter: Step -> runner dict
# ======================================================================

def _step_to_runner_dict(
    step: Step,
    cfg: ColorMixingConfig,
    *,
    rinse_cycles: Optional[int] = None,
    rinse_volume: Optional[float] = None,
) -> dict:
    """Convert a single :class:`Step` to a runner-ready dict.

    This is the **only** place where Step objects are mapped to the dict
    format consumed by ``RunStep(**d)``.
    """
    _rinse_kw: dict = {}
    if rinse_cycles is not None:
        _rinse_kw["rinse_cycles"] = rinse_cycles
    if rinse_volume is not None:
        _rinse_kw["rinse_volume"] = rinse_volume

    if step.action == "transfer":
        source_slot = cfg.reagents[step.source_role].slot
        return {
            "name": f"{'Control' if step.phase == 'control' else 'Exp'}: "
                    f"{step.reagent.title()} "
                    f"{'-> ' + step.dest_well if step.phase == 'control' else str(int(step.volume)) + 'uL -> ' + step.dest_well}",
            "kind": "transfer",
            "params": {
                "tiprack_slot": cfg.tipracks[step.tiprack_mount].slot,
                "source_slot": source_slot,
                "dest_slot": cfg.plate.slot,
                "tip_well": step.tip_well,
                "source_well": step.source_well,
                "dest_well": step.dest_well,
                "volume": step.volume,
                "cleaning_slot": cfg.cleaning.slot,
                "rinse_well": step.rinse_well,
                **_rinse_kw,
            },
        }
    elif step.action == "mix":
        return {
            "name": f"Mix {step.dest_well}",
            "kind": "mix",
            "params": {
                "tiprack_slot": cfg.tipracks[step.tiprack_mount].slot,
                "plate_slot": cfg.plate.slot,
                "tip_well": step.tip_well,
                "mix_well": step.dest_well,
                "cycles": step.mix_cycles,
                "volume": step.mix_volume,
                "cleaning_slot": cfg.cleaning.slot,
                "rinse_well": step.rinse_well,
                **_rinse_kw,
            },
        }
    elif step.action == "pick_up_tip":
        return {
            "name": f"Pick up tip {step.tip_well} ({step.pipette_mount})",
            "kind": "pick_up_tip",
            "params": {
                "slot": cfg.tipracks[step.tiprack_mount].slot,
                "well": step.tip_well,
            },
        }
    elif step.action == "drop_tip":
        return {
            "name": f"Drop tip ({step.pipette_mount})",
            "kind": "drop_tip",
            "params": {},
        }
    elif step.action == "home":
        return {
            "name": "Home",
            "kind": "home",
            "params": {},
        }
    else:
        raise ValueError(f"Unknown step action: {step.action!r}")


def steps_to_runner_dicts(
    steps: List[Step],
    cfg: ColorMixingConfig,
    *,
    rinse_cycles: Optional[int] = None,
    rinse_volume: Optional[float] = None,
) -> List[dict]:
    """Convert a list of :class:`Step` objects to runner-ready dicts.

    Inserts ``use_pipette`` steps where the mount changes.
    """
    result: List[dict] = []
    current_mount = None

    for step in steps:
        if step.action in ("pick_up_tip", "drop_tip", "home"):
            # For check steps, insert mount switches explicitly
            if step.pipette_mount and step.pipette_mount != current_mount:
                result.append({
                    "name": f"Select {step.pipette_mount} pipette",
                    "kind": "use_pipette",
                    "params": {"mount": step.pipette_mount},
                })
                current_mount = step.pipette_mount
            result.append(_step_to_runner_dict(step, cfg,
                                               rinse_cycles=rinse_cycles,
                                               rinse_volume=rinse_volume))
            continue

        if step.pipette_mount != current_mount:
            result.append({
                "name": f"Select {step.pipette_mount} pipette",
                "kind": "use_pipette",
                "params": {"mount": step.pipette_mount},
            })
            current_mount = step.pipette_mount

        if step.action == "transfer" and step.volume < 1.0:
            continue

        result.append(_step_to_runner_dict(step, cfg,
                                           rinse_cycles=rinse_cycles,
                                           rinse_volume=rinse_volume))

    return result


# ======================================================================
# Public derived builders (all delegate to canonical + adapter)
# ======================================================================

def build_runner_steps(
    cfg: ColorMixingConfig,
    volumes: np.ndarray,
    col_idx: int,
    *,
    skip_controls: bool = False,
    mix_cycles_override: Optional[int] = None,
    rinse_cycles: Optional[int] = None,
    rinse_volume: Optional[float] = None,
) -> List[dict]:
    """Build runner-ready step dicts for one column's liquid-handling.

    **Derived from** :func:`build_iteration_steps` — no independent
    protocol logic.  Orchestration steps (wait, capture, extract_rgb,
    fit_gp) are NOT included.
    """
    canonical = build_iteration_steps(
        cfg, volumes, col_idx,
        skip_controls=skip_controls,
        mix_cycles_override=mix_cycles_override,
    )
    return steps_to_runner_dicts(canonical, cfg,
                                rinse_cycles=rinse_cycles,
                                rinse_volume=rinse_volume)


def build_calibration_runner_steps(
    cfg: ColorMixingConfig,
    *,
    rinse_cycles: Optional[int] = None,
    rinse_volume: Optional[float] = None,
) -> List[dict]:
    """Build runner-ready step dicts for a calibration column (controls only).

    **Derived from** :func:`build_control_steps`.
    """
    canonical = build_control_steps(cfg)
    return steps_to_runner_dicts(canonical, cfg,
                                rinse_cycles=rinse_cycles,
                                rinse_volume=rinse_volume)


def build_tip_check_runner_steps(cfg: ColorMixingConfig) -> List[dict]:
    """Build runner-ready step dicts for a tip-presence check.

    **Derived from** :func:`build_tip_check_steps`.
    """
    canonical = build_tip_check_steps(cfg)
    return steps_to_runner_dicts(canonical, cfg)


# ======================================================================
# Direct execution (CLI path)
# ======================================================================

def execute_steps(steps: List[Step], profile, ops) -> None:
    """Execute a list of Step objects using an OT2Operations instance.

    Args:
        steps: Steps from :func:`build_iteration_steps`.
        profile: :class:`~.deck.DeckProfile` with resolved labware IDs.
        ops: ``OT2Operations`` instance.
    """
    robot = ops.client
    current_mount = None

    for step in steps:
        if step.pipette_mount != current_mount:
            robot.use_pipette(step.pipette_mount)
            current_mount = step.pipette_mount

        tiprack_id = profile.tiprack_for_mount(step.tiprack_mount)

        if step.action == "transfer":
            if step.volume < 1.0:
                continue
            source_id = profile.reagent_ids[step.source_role]
            ops.transfer(
                tiprack_id=tiprack_id,
                source_id=source_id,
                dest_id=profile.plate_id,
                tip_well=step.tip_well,
                source_well=step.source_well,
                dest_well=step.dest_well,
                volume=step.volume,
                cleaning_id=profile.cleaning_id,
                rinse_col=step.rinse_well,
            )
        elif step.action == "mix":
            ops.mix(
                tiprack_id=tiprack_id,
                labware_id=profile.plate_id,
                tip_well=step.tip_well,
                mix_well=step.dest_well,
                cycles=step.mix_cycles,
                volume=step.mix_volume,
                cleaning_id=profile.cleaning_id,
                rinse_col=step.rinse_well,
            )
