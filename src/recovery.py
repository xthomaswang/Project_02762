"""
Config-driven error recovery for protocol execution.

Ported from ptc_error_recovery.py — takes explicit robot param instead of module ref.
"""

from typing import Any, Dict, List

from src.robot import OT2Robot


def execute_recovery_plan(robot: OT2Robot,
                          recovery_tasks: List[Dict],
                          context: Dict[str, Any]) -> bool:
    """
    Execute a list of recovery tasks based on runtime context.

    Each task dict has a 'type' key and additional params. String values
    starting with 'ctx.' are resolved from *context*.

    Returns True if all steps succeeded, False otherwise.
    """
    print(f"\n[RECOVERY] Starting Task-Based Recovery ({len(recovery_tasks)} steps)...")

    def resolve(key):
        if isinstance(key, str) and key.startswith("ctx."):
            ctx_key = key[4:]
            val = context.get(ctx_key)
            if val is None:
                raise ValueError(f"Context key '{ctx_key}' missing in recovery context.")
            return val
        return key

    try:
        for i, task in enumerate(recovery_tasks):
            task_type = task.get("type")
            print(f"[RECOVERY] Step {i + 1}: {task_type}")

            if task_type == "dispense":
                robot.dispense(
                    volume=resolve(task.get("volume")),
                    labware_id=resolve(task.get("labware_id")),
                    wellname=resolve(task.get("well")),
                    origin="bottom",
                )
            elif task_type == "blow_out":
                robot.blow_out(
                    labware_id=resolve(task.get("labware_id")),
                    wellname=resolve(task.get("well")),
                )
            elif task_type == "drop":
                robot.drop_tips(
                    tiprack_id=resolve(task.get("labware_id")),
                    wellname=resolve(task.get("well")),
                )
            elif task_type == "home":
                robot.home()
            elif task_type == "pause":
                robot.pause_run(message=task.get("message", "Paused by Recovery Protocol"))
            else:
                print(f"[RECOVERY][WARN] Unknown recovery task type: {task_type}")

        print("[RECOVERY] Plan completed successfully.\n")
        return True

    except Exception as e:
        print(f"[RECOVERY][CRITICAL] Recovery plan failed: {e}")
        try:
            robot.home()
        except Exception:
            pass
        return False
