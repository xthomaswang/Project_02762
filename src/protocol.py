"""
Task-based protocol execution with vision verification.

Merged from ptc.py + ptc_utils.py. Takes robot/vision as explicit parameters
instead of module-level imports.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

from src.robot import OT2Robot
from src import vision
from src import recovery


# ======================================================================
# Labware config loading (from ptc_utils.py)
# ======================================================================

def load_labware_from_config(robot: OT2Robot,
                             config: Dict[str, Any]) -> Dict[str, Any]:
    """Load pipette, tiprack, source, imaging, and dispense labware from config."""
    result: Dict[str, Any] = {}

    pipette_conf = config.get("pipette")
    if pipette_conf is not None:
        pid = robot.load_equipment(0, pipette_conf["name"])
        print(f"[PTC] Pipette loaded: {pid}")
        result["pipette_id"] = pid

    tiprack_conf = config.get("tiprack")
    if tiprack_conf is not None:
        tid = robot.load_equipment(1, tiprack_conf["name"], slot_name=tiprack_conf["slot"])
        print(f"[PTC] Tiprack loaded: {tid} (slot {tiprack_conf['slot']})")
        result["tiprack_id"] = tid

    sources_conf = config.get("sources")
    if sources_conf is not None:
        sources: Dict[str, str] = {}
        for slot in sources_conf["slots"]:
            lid = robot.load_equipment(1, sources_conf["name"], slot_name=slot)
            print(f"[PTC] Source labware loaded: {lid} (slot {slot})")
            sources[slot] = lid
        result["sources"] = sources

    imaging_conf = config.get("imaging")
    if imaging_conf is not None:
        iid = robot.load_equipment(1, imaging_conf["name"], slot_name=imaging_conf["slot"])
        print(f"[PTC] Imaging labware loaded: {iid} (slot {imaging_conf['slot']})")
        result["imaging_labware_id"] = iid

    dispense_conf = config.get("dispense")
    if dispense_conf is not None:
        dname = dispense_conf["name"]
        if "slots" in dispense_conf:
            dslots = dispense_conf["slots"]
        elif "slot" in dispense_conf:
            dslots = [dispense_conf["slot"]]
        else:
            raise ValueError("dispense config must have 'slot' or 'slots'.")
        dispenses: Dict[str, str] = {}
        for slot in dslots:
            did = robot.load_equipment(1, dname, slot_name=slot)
            print(f"[PTC] Dispense labware loaded: {did} (slot {slot})")
            dispenses[slot] = did
        result["dispenses"] = dispenses
        if len(dslots) == 1:
            result["dispense_labware_id"] = next(iter(dispenses.values()))

    return result


# ======================================================================
# Well mapping helpers (from ptc_utils.py)
# ======================================================================

def build_source_wells_by_slot(
    source_slots: List[str],
    source_wells: List[List[str]],
) -> Dict[str, List[str]]:
    """Build mapping: source slot -> list of source wells."""
    if len(source_slots) != len(source_wells):
        raise ValueError("source_slots and source_wells must have the same length.")
    return {slot: [str(w) for w in wells]
            for slot, wells in zip(source_slots, source_wells)}


def build_dispense_wells_by_slot(
    dispense_slots: List[str],
    dispense_wells: Union[str, List[Union[str, List[str]]]],
) -> Dict[str, List[str]]:
    """Normalize dispense definition into slot -> wells mapping."""
    if isinstance(dispense_wells, str):
        return {slot: [dispense_wells] for slot in dispense_slots}
    if not isinstance(dispense_wells, list) or len(dispense_wells) == 0:
        raise ValueError("dispense_wells must be a non-empty str or list.")

    first = dispense_wells[0]
    if isinstance(first, str):
        if len(dispense_slots) == 1:
            return {dispense_slots[0]: [str(w) for w in dispense_wells]}
        raise ValueError("Flat list dispense_wells only allowed with one dispense plate.")

    if isinstance(first, (list, tuple)):
        if len(dispense_wells) != len(dispense_slots):
            raise ValueError("List-of-list length must match dispense_slots length.")
        return {slot: [str(w) for w in wells]
                for slot, wells in zip(dispense_slots, dispense_wells)}

    raise ValueError("Unsupported dispense_wells format.")


# ======================================================================
# Default recovery plans
# ======================================================================

def _default_liquid_recovery() -> List[Dict]:
    return [
        {"type": "dispense", "labware_id": "ctx.source_labware_id",
         "well": "ctx.source_well", "volume": "ctx.volume"},
        {"type": "blow_out", "labware_id": "ctx.source_labware_id",
         "well": "ctx.source_well"},
        {"type": "drop", "labware_id": "ctx.tiprack_id", "well": "ctx.pick_well"},
        {"type": "home"},
    ]


def _default_pickup_recovery() -> List[Dict]:
    return [
        {"type": "drop", "labware_id": "ctx.tiprack_id", "well": "ctx.well_name"},
        {"type": "home"},
    ]


# ======================================================================
# Protocol executor
# ======================================================================

def execute_protocol(config: Dict[str, Any], model: Any,
                     robot: Optional[OT2Robot] = None):
    """
    Main executor for task-based protocol configuration.

    Args:
        config: Dict with 'labware', 'settings', and 'tasks'.
        model: Loaded YOLO model.
        robot: OT2Robot instance (created with default IP if None).
    """
    if robot is None:
        robot = OT2Robot()

    print("\n" + "=" * 60)
    print("  STARTING TASK-BASED PROTOCOL")
    print("=" * 60 + "\n")

    run_id, _ = robot.create_run()
    print(f"[EXECUTOR] Run ID: {run_id}")

    labware_map = load_labware_from_config(robot, config["labware"])

    tiprack_id = labware_map["tiprack_id"]
    img_labware_id = labware_map["imaging_labware_id"]
    img_well = config["settings"]["imaging_well"]
    img_offset = config["settings"].get("imaging_offset", (0, 0, 50))
    base_dir = config["settings"].get("base_dir", "data")

    current_tip_well: Optional[str] = None

    for i, task in enumerate(config.get("tasks", [])):
        task_type = task.get("type")
        print(f"\n[EXECUTOR] Processing Task {i + 1}: {task_type.upper()}")

        if task_type == "pickup":
            success = _execute_pickup(
                robot, task, run_id, model, tiprack_id,
                img_labware_id, img_well, img_offset, base_dir,
            )
            if success:
                current_tip_well = task["well"]
            else:
                print("[EXECUTOR] Protocol stopping due to Pickup Failure.")
                break

        elif task_type == "transfer":
            src_id = labware_map["sources"][task["source_slot"]]
            if "dispenses" in labware_map:
                dest_id = labware_map["dispenses"][task["dest_slot"]]
            else:
                dest_id = labware_map["dispense_labware_id"]
            success = _execute_transfer(
                robot, task, run_id, model, src_id, dest_id,
                img_labware_id, img_well, img_offset, base_dir,
                tiprack_id=tiprack_id,
                current_tip_well=current_tip_well,
            )
            if not success:
                print("[EXECUTOR] Protocol stopping due to Transfer Failure.")
                break

        elif task_type == "drop":
            robot.drop_tips(tiprack_id=tiprack_id, wellname=task["well"])
            current_tip_well = None
            print("[EXECUTOR] Tips dropped.")

    print("\n[EXECUTOR] Protocol Execution Finished.")
    robot.home()


# ======================================================================
# Internal task helpers
# ======================================================================

def _execute_pickup(robot: OT2Robot, task: dict, run_id: str, model: Any,
                    tiprack_id: str, img_id: str, img_well: str,
                    img_offset: tuple, base_dir: str) -> bool:
    well = task["well"]
    robot.pick_up(tiprack_id=tiprack_id, wellname=well)

    result = vision.Predict(
        robot=robot, model=model, run_id=run_id,
        check_type="pickup",
        imaging_labware_id=img_id, imaging_well=img_well,
        imaging_offset=img_offset, base_dir=base_dir,
        step_name=f"pickup_{well}",
        conf=task.get("conf", 0.6),
        expected_tips=task.get("expected_tips", 8),
    )

    if result["passed"]:
        print("[EXECUTOR] Tip Pickup Verified.")
        return True

    print(f"[RECOVERY REASON] Missing: {result.get('missing_positions')}, "
          f"Presence: {result.get('tip_presence')}")
    ctx = {
        "run_id": run_id,
        "pipette_id": robot.pipette_id,
        "tiprack_id": tiprack_id,
        "well_name": well,
        "vision_result": result,
    }
    plan = task.get("on_fail", _default_pickup_recovery())
    recovery.execute_recovery_plan(robot, plan, ctx)
    return False


def _execute_transfer(robot: OT2Robot, task: dict, run_id: str, model: Any,
                      src_id: str, dest_id: str, img_id: str, img_well: str,
                      img_offset: tuple, base_dir: str,
                      tiprack_id: str = "",
                      current_tip_well: Optional[str] = None) -> bool:
    vol = task["volume"]
    src_well = task["source_well"]
    dest_well = task["dest_well"]

    robot.aspirate(
        volume=vol, labware_id=src_id, wellname=src_well,
        origin=task.get("origin", "top"),
        offset=task.get("offset", (0, 0, -35)),
    )

    robot.move(img_id, wellname=img_well, offset=img_offset)

    result = vision.Predict(
        robot=robot, model=model, run_id=run_id,
        check_type="transfer",
        imaging_labware_id=img_id, imaging_well=img_well,
        imaging_offset=img_offset, base_dir=base_dir,
        step_name=f"asp_{src_well}",
        conf=task.get("conf", 0.6),
        volume=vol,
        expected_tips=task.get("expected_tips", 8),
    )
    print(f"[EXECUTOR] Liquid Level Check Result: {result['passed']}")

    if result["passed"]:
        print("[EXECUTOR] Liquid Level Verified. Dispensing.")
        robot.dispense(volume=vol, labware_id=dest_id, wellname=dest_well, origin="bottom")
        robot.blow_out(labware_id=dest_id, wellname=dest_well)
        return True

    print(f"[RECOVERY REASON] {result['detected_levels']}, {result['expected_height_percent']}")
    ctx = {
        "run_id": run_id,
        "pipette_id": robot.pipette_id,
        "source_labware_id": src_id,
        "source_well": src_well,
        "dest_labware_id": dest_id,
        "dest_well": dest_well,
        "tiprack_id": tiprack_id,
        "pick_well": current_tip_well,
        "volume": vol,
        "vision_result": result,
    }
    plan = task.get("on_fail", _default_liquid_recovery())
    recovery.execute_recovery_plan(robot, plan, ctx)
    return False


# ======================================================================
# Convenience helpers (from ptc_utils.py)
# ======================================================================

def auto_recover_tips(robot: OT2Robot, tiprack_id: str, pick_well: str,
                      move_height: float = 20.0,
                      drop_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> bool:
    """Return tips to original well and home."""
    try:
        robot.move(tiprack_id, wellname=pick_well, offset=(0.0, 0.0, move_height))
        robot.drop_tips(tiprack_id=tiprack_id, wellname=pick_well, offset=drop_offset)
        robot.home()
        return True
    except Exception as exc:
        print(f"[PTC WARNING] Tip auto-recovery failed: {exc}")
        return False


def auto_recover_liquid_and_tips(
    robot: OT2Robot,
    source_labware_id: str, source_well: str,
    tiprack_id: str, pick_well: str,
    aspirate_vol: float,
    source_dispense_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    blow_out_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    pick_move_height: float = 20.0,
) -> bool:
    """Recover from failed liquid check: return liquid then return tips."""
    try:
        robot.dispense(volume=aspirate_vol, labware_id=source_labware_id,
                       wellname=source_well, offset=source_dispense_offset)
        robot.blow_out(labware_id=source_labware_id, wellname=source_well,
                       offset=blow_out_offset)
        return auto_recover_tips(robot, tiprack_id, pick_well,
                                 move_height=pick_move_height)
    except Exception as exc:
        print(f"[PTC WARNING] Liquid recovery failed: {exc}")
        return False
