"""Canonical configuration for the color mixing task.

All deck references are role-based (reagent / tiprack / plate / cleaning).
Slot numbers live exclusively in the config YAML — no code path hard-codes them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ------------------------------------------------------------------
# Sub-config dataclasses
# ------------------------------------------------------------------

@dataclass
class PipetteConfig:
    name: str
    mount: str  # "left" or "right"


@dataclass
class ReagentConfig:
    """One dye or liquid source on the deck."""
    labware: str
    slot: str
    well: str = "A1"


@dataclass
class TiprackConfig:
    """A tiprack with role-based tip assignments."""
    labware: str
    slot: str
    mount: str  # which pipette uses this rack


@dataclass
class CleaningConfig:
    """Cleaning reservoir with per-role rinse wells."""
    labware: str
    slot: str
    rinse_wells: Dict[str, str]  # role -> well (e.g. "red" -> "A1")


@dataclass
class PlateConfig:
    """Destination plate."""
    labware: str
    slot: str


@dataclass
class WellRoles:
    """Role-driven well layout on the plate."""
    controls: Dict[str, str]          # row -> role (e.g. "A" -> "red")
    experiment_rows: List[str]        # e.g. ["E", "F", "G", "H"]
    control_volume_ul: float = 200.0


@dataclass
class ExperimentConfig:
    n_initial: int = 5
    n_optimization: int = 7
    max_per_plate: int = 12
    settle_time_seconds: float = 10.0
    mix_cycles: int = 3
    mix_volume_ul: float = 150.0
    batch_mode: bool = False


@dataclass
class MLConfig:
    model: str = "correlated_gp"
    acquisition: str = "EI"
    distance_metric: str = "delta_e_lab"


@dataclass
class CameraConfig:
    device_id: int = 0
    width: int = 1920
    height: int = 1080
    warmup_frames: int = 10


@dataclass
class CleaningProtocol:
    enabled: bool = True
    rinse_cycles: int = 3
    rinse_volume_ul: float = 200.0


@dataclass
class CalibrationConfig:
    grid_path: str = "calibrations/cv_calibration/grid.json"
    roi_scale: float = 0.35


@dataclass
class ParkConfig:
    slot: str = "11"
    well: str = "A1"
    offset: Tuple[float, float, float] = (0.0, 0.0, 30.0)


# ------------------------------------------------------------------
# Top-level config
# ------------------------------------------------------------------

@dataclass
class ColorMixingConfig:
    """Complete, self-contained config for a color mixing experiment.

    Every deck reference is role-based.  The config YAML is the single
    source of truth for slot assignments.
    """

    # Target & search space
    target_color: List[float] = field(default_factory=lambda: [128, 0, 255])
    total_volume_ul: float = 200.0
    volume_bounds_min: List[float] = field(default_factory=lambda: [0, 0, 0])
    volume_bounds_max: List[float] = field(default_factory=lambda: [200, 200, 200])
    convergence_threshold: float = 50.0
    seed: int = 42

    # Robot
    robot_ip: str = "169.254.8.56"
    robot_port: int = 31950

    # Pipettes
    pipettes: Dict[str, PipetteConfig] = field(default_factory=dict)

    # Deck layout (role-driven)
    plate: PlateConfig = field(default_factory=lambda: PlateConfig(
        labware="corning_96_wellplate_360ul_flat", slot="1",
    ))
    reagents: Dict[str, ReagentConfig] = field(default_factory=dict)
    cleaning: CleaningConfig = field(default_factory=lambda: CleaningConfig(
        labware="nest_12_reservoir_15ml", slot="4",
        rinse_wells={"red": "A1", "green": "A2", "blue": "A3",
                     "mix": "A4", "control": "A6"},
    ))
    tipracks: Dict[str, TiprackConfig] = field(default_factory=dict)
    tip_columns: Dict[str, int] = field(default_factory=dict)
    tip_wells: Dict[str, str] = field(default_factory=dict)

    # Well roles
    well_roles: WellRoles = field(default_factory=lambda: WellRoles(
        controls={"A": "red", "B": "green", "C": "blue", "D": "water"},
        experiment_rows=["E", "F", "G", "H"],
    ))

    # Sub-configs
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    cleaning_protocol: CleaningProtocol = field(default_factory=CleaningProtocol)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    park: ParkConfig = field(default_factory=ParkConfig)

    # Output
    output_base_dir: str = "data"

    # Internal: directory of the config file (for resolving relative paths)
    _config_dir: str = field(default=".", repr=False)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def control_rows(self) -> Dict[str, str]:
        return self.well_roles.controls

    @property
    def experiment_rows(self) -> List[str]:
        return self.well_roles.experiment_rows

    def resolve_path(self, rel_path: str) -> str:
        """Resolve *rel_path* relative to the config file directory."""
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.join(self._config_dir, rel_path)

    def reagent_roles(self) -> List[str]:
        """Return sorted list of reagent role names."""
        return sorted(self.reagents.keys())

    def role_for_row(self, row: str) -> Optional[str]:
        """Return the reagent role assigned to a control row, or None."""
        return self.well_roles.controls.get(row)

    def rows_for_role(self, role: str) -> List[str]:
        """Return row letters assigned to a given role."""
        return [r for r, rl in self.well_roles.controls.items() if rl == role]

    def build_deck_dict(self) -> Dict[str, Any]:
        """Return ``{pipettes: {mount: name}, labware: {slot: name}}`` for OpenOT2 DeckConfig."""
        pipettes = {mount: pcfg.name for mount, pcfg in self.pipettes.items()}
        labware: Dict[str, str] = {}
        # Tipracks
        for mount, tcfg in self.tipracks.items():
            labware[tcfg.slot] = tcfg.labware
        # Reagent reservoirs
        for role, rcfg in self.reagents.items():
            labware[rcfg.slot] = rcfg.labware
        # Cleaning reservoir
        labware[self.cleaning.slot] = self.cleaning.labware
        # Plate
        labware[self.plate.slot] = self.plate.labware
        return {"pipettes": pipettes, "labware": labware}

    def build_calibration_targets(self) -> List[Dict[str, Any]]:
        """Build robot calibration targets for the web calibration UI.

        Returns a list of target dicts with name, pipette_mount,
        labware_slot, well, action, and optional volume.
        """
        targets: List[Dict[str, Any]] = []
        total_volume = self.total_volume_ul
        mix_volume = self.experiment.mix_volume_ul

        for mount, pcfg in self.pipettes.items():
            # Tiprack
            tcfg = self.tipracks.get(mount)
            if tcfg:
                if mount == "left":
                    first_well = next(iter(self.tip_wells.values()), "A1")
                else:
                    first_col = next(iter(self.tip_columns.values()), 1)
                    first_well = f"A{first_col}"
                targets.append({
                    "name": f"{mount.title()} tiprack (slot {tcfg.slot})",
                    "pipette_mount": mount,
                    "labware_slot": tcfg.slot,
                    "well": first_well,
                    "action": "pick_up_tip",
                })

            # Reagent reservoirs
            source_volume = total_volume if mount == "left" else mix_volume
            for role in ("red", "green", "blue"):
                rcfg = self.reagents.get(role)
                if not rcfg:
                    continue
                targets.append({
                    "name": f"{role.title()} reservoir ({mount})",
                    "pipette_mount": mount,
                    "labware_slot": rcfg.slot,
                    "well": rcfg.well,
                    "action": "aspirate",
                    "volume": float(source_volume),
                })

            # Water (left only)
            if mount == "left":
                water_rcfg = self.reagents.get("water")
                if water_rcfg:
                    targets.append({
                        "name": "Water reservoir (left)",
                        "pipette_mount": "left",
                        "labware_slot": water_rcfg.slot,
                        "well": water_rcfg.well,
                        "action": "aspirate",
                        "volume": float(total_volume),
                    })

            # Plate
            targets.append({
                "name": f"Plate dispense ({mount})",
                "pipette_mount": mount,
                "labware_slot": self.plate.slot,
                "well": "A1",
                "action": "dispense",
                "volume": float(total_volume if mount == "left" else mix_volume),
            })

            # Cleaning
            if mount == "left":
                rinse_well = self.cleaning.rinse_wells.get("control", "A6")
            else:
                rinse_well = self.cleaning.rinse_wells.get("mix", "A4")
            targets.append({
                "name": f"Cleaning reservoir ({mount})",
                "pipette_mount": mount,
                "labware_slot": self.cleaning.slot,
                "well": rinse_well,
                "action": "aspirate",
                "volume": float(source_volume),
            })

        return targets


# ------------------------------------------------------------------
# YAML loader
# ------------------------------------------------------------------

def load_task_config(config_path: str) -> ColorMixingConfig:
    """Load a color mixing config YAML and return a typed dataclass."""
    config_path = os.path.abspath(config_path)
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    config_dir = os.path.dirname(config_path)

    # --- Pipettes ---
    pipettes = {}
    for mount, pcfg in raw.get("pipettes", {}).items():
        pipettes[mount] = PipetteConfig(name=pcfg["name"], mount=mount)

    # --- Deck ---
    deck = raw.get("deck", {})

    plate_raw = deck.get("plate", {})
    plate = PlateConfig(
        labware=plate_raw.get("labware", "corning_96_wellplate_360ul_flat"),
        slot=str(plate_raw.get("slot", "1")),
    )

    reagents = {}
    for role, rcfg in deck.get("reagents", {}).items():
        reagents[role] = ReagentConfig(
            labware=rcfg.get("labware", "nest_12_reservoir_15ml"),
            slot=str(rcfg["slot"]),
            well=rcfg.get("well", "A1"),
        )

    clean_raw = deck.get("cleaning", {})
    cleaning = CleaningConfig(
        labware=clean_raw.get("labware", "nest_12_reservoir_15ml"),
        slot=str(clean_raw.get("slot", "4")),
        rinse_wells=clean_raw.get("rinse_wells", {}),
    )

    tipracks = {}
    tip_columns: Dict[str, int] = {}
    tip_wells: Dict[str, str] = {}
    for mount, tcfg in deck.get("tipracks", {}).items():
        tipracks[mount] = TiprackConfig(
            labware=tcfg.get("labware", "opentrons_96_filtertiprack_200ul"),
            slot=str(tcfg["slot"]),
            mount=mount,
        )
        for role, col in tcfg.get("tip_columns", {}).items():
            tip_columns[role] = int(col)
        for role, well in tcfg.get("tip_wells", {}).items():
            tip_wells[role] = well

    # --- Well roles ---
    wr_raw = raw.get("well_roles", {})
    controls_raw = wr_raw.get("controls", {"A": "red", "B": "green",
                                            "C": "blue", "D": "water"})
    well_roles = WellRoles(
        controls={str(k): str(v) for k, v in controls_raw.items()},
        experiment_rows=[str(r) for r in wr_raw.get("experiment_rows",
                                                     ["E", "F", "G", "H"])],
        control_volume_ul=float(wr_raw.get("control_volume_ul", 200)),
    )

    # --- Experiment ---
    exp_raw = raw.get("experiment", {})
    experiment = ExperimentConfig(
        n_initial=int(exp_raw.get("n_initial", 5)),
        n_optimization=int(exp_raw.get("n_optimization", 7)),
        max_per_plate=int(exp_raw.get("max_per_plate", 12)),
        settle_time_seconds=float(exp_raw.get("settle_time_seconds", 10)),
        mix_cycles=int(exp_raw.get("mix_cycles", 3)),
        mix_volume_ul=float(exp_raw.get("mix_volume_ul", 150)),
        batch_mode=bool(exp_raw.get("batch_mode", False)),
    )

    # --- ML ---
    ml_raw = raw.get("ml", {})
    ml = MLConfig(
        model=ml_raw.get("model", "correlated_gp"),
        acquisition=ml_raw.get("acquisition", "EI"),
        distance_metric=ml_raw.get("distance_metric", "delta_e_lab"),
    )

    # --- Camera ---
    cam_raw = raw.get("camera", {})
    camera = CameraConfig(
        device_id=int(cam_raw.get("device_id", 0)),
        width=int(cam_raw.get("width", 1920)),
        height=int(cam_raw.get("height", 1080)),
        warmup_frames=int(cam_raw.get("warmup_frames", 10)),
    )

    # --- Cleaning protocol ---
    cl_raw = raw.get("cleaning", {})
    cleaning_protocol = CleaningProtocol(
        enabled=bool(cl_raw.get("enabled", True)),
        rinse_cycles=int(cl_raw.get("rinse_cycles", 3)),
        rinse_volume_ul=float(cl_raw.get("rinse_volume_ul", 200)),
    )

    # --- Calibration ---
    cal_raw = raw.get("calibration", {})
    calibration = CalibrationConfig(
        grid_path=cal_raw.get("grid_path", "calibrations/cv_calibration/grid.json"),
        roi_scale=float(cal_raw.get("roi_scale", 0.35)),
    )

    # --- Imaging park ---
    park_raw = raw.get("imaging", {}).get("park", {})
    offset_raw = park_raw.get("offset", [0.0, 0.0, 30.0])
    park = ParkConfig(
        slot=str(park_raw.get("slot", "11")),
        well=park_raw.get("well", "A1"),
        offset=(float(offset_raw[0]), float(offset_raw[1]), float(offset_raw[2])),
    )

    # --- Volume bounds ---
    vb = raw.get("volume_bounds", {})

    return ColorMixingConfig(
        target_color=[float(v) for v in raw.get("target_color", [128, 0, 255])],
        total_volume_ul=float(raw.get("total_volume_ul", 200)),
        volume_bounds_min=[float(v) for v in vb.get("min", [0, 0, 0])],
        volume_bounds_max=[float(v) for v in vb.get("max", [200, 200, 200])],
        convergence_threshold=float(raw.get("convergence_threshold", 50)),
        seed=int(raw.get("seed", 42)),
        robot_ip=raw.get("robot", {}).get("ip", "169.254.8.56"),
        robot_port=int(raw.get("robot", {}).get("port", 31950)),
        pipettes=pipettes,
        plate=plate,
        reagents=reagents,
        cleaning=cleaning,
        tipracks=tipracks,
        tip_columns=tip_columns,
        tip_wells=tip_wells,
        well_roles=well_roles,
        experiment=experiment,
        ml=ml,
        camera=camera,
        cleaning_protocol=cleaning_protocol,
        calibration=calibration,
        park=park,
        output_base_dir=raw.get("output", {}).get("base_dir", "data"),
        _config_dir=config_dir,
    )
