"""Active learning loop manager for the color mixing protocol.

State machine
-------------
idle -> running -> pausing -> paused -> running (resume)
                -> stopping -> stopped
                -> converged
                -> completed
                -> failed
idle -> calibrating -> idle  (calibrate)

The loop runs in a background thread.  ``pause()``, ``resume()``, and
``stop()`` are called from the API thread; synchronisation is handled
via :class:`threading.Event` and atomic flag writes.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from openot2.control.models import (
    RunSequence,
    RunStep,
    RunStatus,
    StepStatus,
    TaskRun,
)
from src.ml import sample_simplex
from src.color.metrics import color_distance
from src.web.handlers import (
    handle_suggest_volumes,
    handle_extract_rgb,
    handle_fit_gp,
    _serialize_reference,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Plan helpers
# ======================================================================

def build_plan(
    config: dict,
    mode: str,
    seed: int = 42,
    start_column: int = 0,
    pre_calibrated: bool = False,
) -> dict:
    """Build a human-readable protocol plan from *config* + *mode*.

    Returns a dict with a top-level summary and per-iteration detail.
    """
    exp = config.get("experiment", {})
    ml = config.get("ml", {})
    n_initial_raw = exp.get("n_initial", 5)
    n_initial = max(n_initial_raw - (1 if pre_calibrated else 0), 0)
    n_opt = exp.get("n_optimization", 7)
    total_volume = float(config.get("total_volume_ul", 200))

    if mode == "quick":
        mix_cycles = 2
        skip_controls_after_first = True
    else:
        mix_cycles = exp.get("mix_cycles", 3)
        skip_controls_after_first = False

    total = n_initial + n_opt
    target_rgb = config.get("target_color", [0, 0, 0])
    threshold = float(
        config.get("convergence_threshold")
        or config.get("convergence", {}).get("rgb_distance_threshold", 50)
    )

    # Pre-sample random volumes for display
    rng = np.random.default_rng(seed=seed)
    random_vols = sample_simplex(n_initial, total_volume, d=3, rng=rng)

    iterations: list[dict] = []
    for i in range(total):
        phase = "random" if i < n_initial else "bo"
        skip = skip_controls_after_first and (pre_calibrated or i > 0)
        entry: dict[str, Any] = {
            "iteration": i + 1,
            "phase": phase,
            "column": ((i + start_column) % 12) + 1,
            "skip_controls": skip,
            "status": "planned",
        }
        if phase == "random":
            v = random_vols[i]
            entry["volumes"] = [
                round(float(v[0]), 1),
                round(float(v[1]), 1),
                round(float(v[2]), 1),
            ]
        else:
            entry["volumes"] = None  # determined by GP at runtime
        iterations.append(entry)

    return {
        "mode": mode,
        "target_rgb": list(target_rgb),
        "convergence_threshold": threshold,
        "total_iterations": total,
        "n_initial": n_initial,
        "n_initial_configured": n_initial_raw,
        "n_optimization": n_opt,
        "mix_cycles": mix_cycles,
        "skip_controls_after_first": skip_controls_after_first,
        "pre_calibrated": pre_calibrated,
        "start_column": start_column,
        "distance_metric": ml.get("distance_metric", "rgb_euclidean"),
        "model_type": ml.get("model", "correlated_gp"),
        "acquisition": ml.get("acquisition", "EI"),
        "iterations": iterations,
    }


# ======================================================================
# ActiveLearningLoop
# ======================================================================

class ActiveLearningLoop:
    """Manages the multi-column active learning loop.

    Each iteration (column) becomes one OpenOT2 run within a sequence.
    The loop dynamically generates steps and wires data between them
    (capture -> extract_rgb -> fit_gp) after each run completes.
    """

    # Valid state transitions ------------------------------------------
    # idle        -> running   (start)
    # running     -> pausing   (pause)
    # running     -> stopping  (stop)
    # running     -> converged (convergence check)
    # running     -> completed (all iterations done)
    # running     -> failed    (fatal error)
    # pausing     -> paused    (runner returned with paused status)
    # paused      -> running   (resume)
    # paused      -> stopped   (stop while paused)
    # stopping    -> stopped   (runner finishes current step)
    # ----------------------------------------------------------------

    def __init__(self, runner, store, config_path: str):
        self.runner = runner
        self.store = store
        self.config = self._load_config(config_path)
        self.config_path = config_path
        self._reset_state()
        self._load_persisted_color_calibration()

    # ------------------------------------------------------------------
    # Internal state management
    # ------------------------------------------------------------------

    def _reset_state(self):
        """Reset ALL mutable state for a fresh run."""
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._sequence_id: Optional[str] = None
        self._state: str = "idle"
        self._mode: str = "quick"
        self._error: Optional[str] = None
        self._converged = False
        self._current_iteration = 0
        self._active_run_id: Optional[str] = None
        self._live_progress: Optional[dict] = None

        # Threading primitives for pause/resume
        self._resume_event = threading.Event()

        # Observation history
        self._all_X: List[List[float]] = []
        self._all_Y: List[List[float]] = []
        self._all_dist: List[float] = []
        self._history: List[dict] = []
        self._cached_reference: Optional[dict] = None

        # Plan (populated at start time)
        self._plan: Optional[dict] = None

        # Calibration state
        self._calibration_done = False
        self._pure_rgbs: Optional[dict] = None  # {"red": [...], "green": [...], "blue": [...]}
        self._water_rgb: Optional[list] = None
        self._gamut: Optional[dict] = None
        self._custom_target: Optional[list] = None  # user-selected target from gamut

    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def _resolve_config_path(self, path: str) -> str:
        """Resolve config-relative paths against the YAML location."""
        if not path:
            return ""
        if os.path.isabs(path):
            return path
        config_dir = os.path.dirname(os.path.abspath(self.config_path))
        return os.path.join(config_dir, path)

    def _resolved_grid_path(self) -> str:
        """Return the absolute grid-calibration path for the current config."""
        raw_path = self.config.get("calibration", {}).get("grid_path", "")
        return self._resolve_config_path(raw_path)

    def _color_calibration_dir(self) -> Path:
        """Return the directory used for persisted color-calibration state."""
        return Path(self.config_path).resolve().parent / "calibrations" / "color_calibration"

    def _color_calibration_path(self) -> Path:
        """Return the canonical saved color-calibration file path."""
        return self._color_calibration_dir() / "color_profile.json"

    def _save_color_calibration(self) -> Path:
        """Persist the current color-calibration state to disk."""
        if not self._pure_rgbs or self._water_rgb is None or self._cached_reference is None:
            raise ValueError("Color calibration state is incomplete; nothing to save")

        path = self._color_calibration_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "pure_rgbs": self._pure_rgbs,
            "water_rgb": self._water_rgb,
            "gamut": self._gamut,
            "cached_reference": self._cached_reference,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _load_persisted_color_calibration(self) -> bool:
        """Load saved color-calibration state if present."""
        path = self._color_calibration_path()
        if not path.exists():
            return False

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pure_rgbs = payload.get("pure_rgbs")
            water_rgb = payload.get("water_rgb")
            cached_reference = payload.get("cached_reference")
            gamut = payload.get("gamut")
            if pure_rgbs is None or water_rgb is None or cached_reference is None:
                logger.warning(
                    "Ignoring incomplete color calibration at %s", path
                )
                return False
            if gamut is None:
                from src.color.metrics import compute_reachable_gamut

                gamut = compute_reachable_gamut(pure_rgbs, np.array(water_rgb))

            self._pure_rgbs = pure_rgbs
            self._water_rgb = water_rgb
            self._cached_reference = cached_reference
            self._gamut = gamut
            self._calibration_done = True
            logger.info("Loaded color calibration from %s", path)
            return True
        except Exception as exc:
            logger.warning("Could not load color calibration from %s: %s", path, exc)
            return False

    def _snapshot_calibration_state(self) -> dict:
        """Preserve calibration data across run restarts."""
        return {
            "calibration_done": self._calibration_done,
            "pure_rgbs": self._pure_rgbs,
            "water_rgb": self._water_rgb,
            "gamut": self._gamut,
            "custom_target": self._custom_target,
            "cached_reference": self._cached_reference,
        }

    def _restore_calibration_state(self, snapshot: dict) -> None:
        """Restore previously captured calibration state."""
        self._calibration_done = snapshot.get("calibration_done", False)
        self._pure_rgbs = snapshot.get("pure_rgbs")
        self._water_rgb = snapshot.get("water_rgb")
        self._gamut = snapshot.get("gamut")
        self._custom_target = snapshot.get("custom_target")
        self._cached_reference = snapshot.get("cached_reference")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, mode: str = "quick") -> str:
        """Start the active learning loop in a background thread.

        Returns the sequence ID for the new run.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Loop is already running")

        # Reset run-time state but keep any completed calibration.
        calibration_state = self._snapshot_calibration_state()
        self._reset_state()
        self._restore_calibration_state(calibration_state)
        self._mode = mode
        self._state = "running"

        # Build plan
        seed = self.config.get("seed", 42)
        start_col = 1 if self._calibration_done else 0
        self._plan = build_plan(
            self.config,
            mode,
            seed=seed,
            start_column=start_col,
            pre_calibrated=self._calibration_done,
        )

        # Create persistent sequence
        seq = RunSequence(
            name=f"color_mixing_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            status="running",
        )
        seq = self.store.create_sequence(seq)
        self._sequence_id = seq.id

        logger.info(
            "=== Active Learning Start ===\n"
            "  Mode: %s\n"
            "  Target RGB: %s\n"
            "  Threshold: %.1f\n"
            "  Iterations: %d (%d random + %d BO)\n"
            "  Distance metric: %s",
            mode,
            self._plan["target_rgb"],
            self._plan["convergence_threshold"],
            self._plan["total_iterations"],
            self._plan["n_initial"],
            self._plan["n_optimization"],
            self._plan["distance_metric"],
        )

        self._thread = threading.Thread(
            target=self._loop_wrapper,
            args=(mode,),
            daemon=True,
        )
        self._thread.start()
        return self._sequence_id

    def pause(self):
        """Request a pause at the next safe-point.

        If the loop is currently running, sets state to ``pausing`` and
        signals the runner to pause at the next checkpoint.
        """
        if self._state == "running":
            self._state = "pausing"
            if self._active_run_id:
                try:
                    self.runner.request_pause(self._active_run_id)
                except Exception as exc:
                    logger.warning("request_pause failed: %s", exc)
            logger.info("Pause requested")

    def resume(self):
        """Resume from a paused state.

        Sets state back to ``running`` and wakes the loop thread so it
        can call ``runner.resume()`` on the active run.
        """
        if self._state == "paused":
            self._state = "running"
            self._resume_event.set()
            logger.info("Resumed")

    def stop(self):
        """Emergency stop — works in ANY active state.

        Signals the runner to pause at the next safe-point.  Works during
        calibration, running, pausing, or paused states.
        """
        self._stop_requested = True
        prev = self._state

        if self._state == "paused":
            self._resume_event.set()
        elif self._state in ("running", "calibrating", "pausing"):
            self._state = "stopping"
            if self._active_run_id:
                try:
                    self.runner.request_pause(self._active_run_id)
                except Exception as exc:
                    logger.warning("request_pause (stop) failed: %s", exc)

        logger.info("E-STOP requested (was %s)", prev)

    def home(self) -> dict:
        """Home the robot.  Only safe when not actively running."""
        if self._state in ("running", "pausing", "paused", "stopping"):
            return {
                "ok": False,
                "error": "Cannot home while protocol is active -- stop it first",
            }
        if self._active_run_id is not None:
            return {
                "ok": False,
                "error": "Cannot home while an active run is still attached",
            }
        try:
            home_step = RunStep(name="Home", kind="home", params={})
            run = TaskRun(name="manual_home", steps=[home_step])
            run = self.store.create_run(run)
            self.runner.run_until_pause_or_done(run.id)
            run = self.store.load_run(run.id)
            if run.status != RunStatus.completed:
                failed_step = next(
                    (step for step in run.steps if step.status == StepStatus.failed),
                    None,
                )
                return {
                    "ok": False,
                    "error": failed_step.error if failed_step and failed_step.error else (
                        f"Home run ended with status {run.status.value}"
                    ),
                    "run_id": run.id,
                }
            return {"ok": True, "run_id": run.id}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def calibrate(self) -> dict:
        """Run a calibration column (controls only) to measure pure dye RGBs.

        Dispenses controls into column 1, captures image, extracts RGB for
        each control well (A=Red, B=Green, C=Blue, D=Water), then computes
        the reachable color gamut.

        Returns dict with ok, pure_rgbs, water_rgb, gamut.
        """
        if self._state not in ("idle", "stopped", "completed", "converged", "failed"):
            return {"ok": False, "error": f"Cannot calibrate in state '{self._state}'"}

        self._state = "calibrating"

        try:
            return self._run_calibration()
        except Exception as exc:
            logger.error("Calibration failed: %s", exc, exc_info=True)
            self._state = "idle"
            return {"ok": False, "error": str(exc)}

    def _run_calibration(self) -> dict:
        """Execute calibration synchronously (called from calibrate())."""
        cfg = self.config
        grid_path = self._resolved_grid_path()
        control_volume = float(cfg.get("total_volume_ul", 200))
        settle_time = cfg.get("experiment", {}).get("settle_time_seconds", 10)
        clean_cfg = cfg.get("cleaning", {})
        rinse_cycles = clean_cfg.get("rinse_cycles", 3)
        rinse_volume = float(clean_cfg.get("rinse_volume_ul", 200))

        # Build calibration steps: controls in column 1 + capture
        steps = []

        # Use left pipette for controls
        steps.append({"name": "Select left pipette", "kind": "use_pipette", "params": {"mount": "left"}})

        controls = [
            ("Red → A1", "7", "A1", "A1", "A6"),
            ("Green → B1", "8", "B1", "B1", "A6"),
            ("Blue → C1", "9", "C1", "C1", "A6"),
            ("Water → D1", "5", "D1", "D1", "A5"),
        ]

        _rinse_kw = {}
        if rinse_cycles is not None:
            _rinse_kw["rinse_cycles"] = rinse_cycles
        if rinse_volume is not None:
            _rinse_kw["rinse_volume"] = rinse_volume

        for label, src_slot, tip, dest, rinse in controls:
            steps.append({
                "name": f"Cal: {label}",
                "kind": "transfer",
                "params": {
                    "tiprack_slot": "10", "source_slot": src_slot,
                    "dest_slot": "1", "tip_well": tip,
                    "source_well": "A1", "dest_well": dest,
                    "volume": control_volume, "cleaning_slot": "4",
                    "rinse_well": rinse, **_rinse_kw,
                },
            })

        # Home + settle + capture
        steps.append({"name": "Home", "kind": "home", "params": {}})
        steps.append({"name": f"Settle {settle_time}s", "kind": "wait", "params": {"seconds": settle_time}})
        steps.append({"name": "Capture calibration", "kind": "capture", "params": {"label": "calibration_col1"}})

        # Create and execute run
        run_steps = [RunStep(**s) for s in steps]
        run = TaskRun(name="Calibration (column 1)", steps=run_steps, metadata={"type": "calibration"})
        run = self.store.create_run(run)

        self._active_run_id = run.id
        self.runner.run_until_pause_or_done(run.id)
        self._active_run_id = None

        # Check if E-Stop was pressed during calibration
        if self._stop_requested:
            self._state = "stopped"
            return {"ok": False, "error": "Calibration stopped by user"}

        # Reload run to get outputs
        run = self.store.load_run(run.id)

        # Check run succeeded
        if run.status != RunStatus.completed:
            self._state = "idle"
            failed_step = next((s for s in run.steps if s.error), None)
            return {"ok": False, "error": f"Calibration run failed: {failed_step.error if failed_step else 'unknown'}"}

        # Get capture output for image path
        capture_output = None
        for s in run.steps:
            if s.kind == "capture" and s.output:
                capture_output = s.output
                break

        if not capture_output:
            self._state = "idle"
            return {"ok": False, "error": "Capture step produced no output"}

        image_path = capture_output.get("image_path") or capture_output.get("path")
        if not image_path:
            self._state = "idle"
            return {"ok": False, "error": "No image path in capture output"}

        # Extract RGB from control wells
        class _P:
            def __init__(self, params): self.params = params

        rgb_result = handle_extract_rgb(_P({
            "image_path": image_path,
            "col": 0,  # column 1 = index 0
            "grid_path": grid_path,
            "cached_reference": None,
            "skip_controls": False,
        }))

        if rgb_result.get("error"):
            self._state = "idle"
            return {"ok": False, "error": f"RGB extraction failed: {rgb_result['error']}"}

        # We need individual control well RGBs, not experiment mean.
        # Re-extract from image directly.
        try:
            import cv2
            from src.vision.extraction import extract_well_rgb
            from src.vision.geometry import load_grid_calibration

            img = cv2.imread(image_path)
            grid = None
            if grid_path:
                try:
                    grid = load_grid_calibration(grid_path)
                except Exception:
                    pass

            # A=0=Red, B=1=Green, C=2=Blue, D=3=Water (column 0)
            red_rgb = extract_well_rgb(img, 0, 0, grid).tolist()
            green_rgb = extract_well_rgb(img, 1, 0, grid).tolist()
            blue_rgb = extract_well_rgb(img, 2, 0, grid).tolist()
            water_rgb = extract_well_rgb(img, 3, 0, grid).tolist()
        except Exception as exc:
            self._state = "idle"
            return {"ok": False, "error": f"Control well extraction failed: {exc}"}

        self._pure_rgbs = {"red": red_rgb, "green": green_rgb, "blue": blue_rgb}
        self._water_rgb = water_rgb

        # Cache the runtime reference for subsequent columns
        if rgb_result.get("runtime_reference"):
            self._cached_reference = rgb_result["runtime_reference"]

        # Compute gamut
        from src.color.metrics import compute_reachable_gamut
        self._gamut = compute_reachable_gamut(self._pure_rgbs, np.array(self._water_rgb))

        try:
            save_path = self._save_color_calibration()
            logger.info("Saved color calibration to %s", save_path)
        except Exception as exc:
            logger.warning("Could not save color calibration: %s", exc)

        self._calibration_done = True
        self._state = "idle"

        logger.info(
            "Calibration complete:\n  Red:   %s\n  Green: %s\n  Blue:  %s\n  Water: %s",
            red_rgb, green_rgb, blue_rgb, water_rgb,
        )

        return {
            "ok": True,
            "pure_rgbs": self._pure_rgbs,
            "water_rgb": self._water_rgb,
            "gamut": self._gamut,
        }

    def tip_check(self) -> dict:
        """Run a tip pick-up / drop check for both pipettes.

        Sequentially picks up and drops tips from the left (slot 10) and
        right (slot 11) tip-racks so the operator can visually verify
        that tips are seated and released correctly.

        Only callable when the loop is idle or in a terminal state.
        Returns ``{"ok": True}`` on success or ``{"ok": False, "error": ...}``.
        """
        if self._state not in ("idle", "stopped", "completed", "converged", "failed"):
            return {"ok": False, "error": f"Cannot run tip check in state '{self._state}'"}

        prev_state = self._state
        self._state = "calibrating"

        try:
            steps: list[dict] = []

            # -- Left pipette: tips from slot 10, wells A1-D1 --
            steps.append({"name": "Select left pipette", "kind": "use_pipette", "params": {"mount": "left"}})
            for well in ("A1", "B1", "C1", "D1"):
                steps.append({
                    "name": f"Pick up tip {well} (left)",
                    "kind": "pick_up_tip",
                    "params": {"slot": "10", "well": well},
                })
                steps.append({
                    "name": f"Drop tip {well} (left)",
                    "kind": "drop_tip",
                    "params": {},
                })

            # -- Right pipette: tips from slot 11, wells A1-A4 --
            steps.append({"name": "Select right pipette", "kind": "use_pipette", "params": {"mount": "right"}})
            for well in ("A1", "A2", "A3", "A4"):
                steps.append({
                    "name": f"Pick up tip {well} (right)",
                    "kind": "pick_up_tip",
                    "params": {"slot": "11", "well": well},
                })
                steps.append({
                    "name": f"Drop tip {well} (right)",
                    "kind": "drop_tip",
                    "params": {},
                })

            # -- Home at the end --
            steps.append({"name": "Home", "kind": "home", "params": {}})

            # Create and execute the run synchronously
            run_steps = [RunStep(**s) for s in steps]
            run = TaskRun(name="Tip Check", steps=run_steps, metadata={"type": "tip_check"})
            run = self.store.create_run(run)

            self._active_run_id = run.id
            self.runner.run_until_pause_or_done(run.id)
            self._active_run_id = None

            # Check result
            run = self.store.load_run(run.id)
            if run.status != RunStatus.completed:
                failed_step = next(
                    (s for s in run.steps if s.status == StepStatus.failed), None
                )
                self._state = prev_state
                return {
                    "ok": False,
                    "error": (
                        failed_step.error
                        if failed_step and failed_step.error
                        else f"Tip check run ended with status {run.status.value}"
                    ),
                    "run_id": run.id,
                }

            self._state = prev_state
            logger.info("Tip check completed successfully")
            return {"ok": True, "run_id": run.id}

        except Exception as exc:
            logger.error("Tip check failed: %s", exc, exc_info=True)
            self._state = prev_state
            return {"ok": False, "error": str(exc)}

    def set_target(self, rgb: list) -> dict:
        """Set a custom target RGB (typically chosen from the gamut)."""
        if len(rgb) != 3:
            return {"ok": False, "error": "Target must be [R, G, B]"}
        self._custom_target = [float(rgb[0]), float(rgb[1]), float(rgb[2])]
        logger.info("Custom target set: %s", self._custom_target)
        return {"ok": True, "target_rgb": self._custom_target}

    def reset(self) -> dict:
        """Reset the loop to idle state.  Only allowed when not running."""
        if self._state in ("running", "pausing", "paused", "stopping"):
            return {"ok": False, "error": "Cannot reset while protocol is active"}
        if self._active_run_id is not None:
            return {"ok": False, "error": "Cannot reset while an active run is still attached"}
        calibration_state = self._snapshot_calibration_state()
        self._reset_state()
        self._restore_calibration_state(calibration_state)
        return {"ok": True}

    def status(self) -> dict:
        """Return a comprehensive snapshot of loop state."""
        start_col = 1 if self._calibration_done else 0
        plan = self._plan or build_plan(
            self.config,
            self._mode,
            self.config.get("seed", 42),
            start_column=start_col,
            pre_calibrated=self._calibration_done,
        )
        best_idx = int(np.argmin(self._all_dist)) if self._all_dist else None
        active_run = self._active_run_snapshot()
        return {
            "state": self._state,
            "sequence_id": self._sequence_id,
            "iteration": self._current_iteration,
            "max_iterations": plan["total_iterations"],
            "n_initial": plan["n_initial"],
            "n_optimization": plan["n_optimization"],
            "convergence_threshold": plan["convergence_threshold"],
            "target_rgb": plan["target_rgb"],
            "distance_metric": plan["distance_metric"],
            "mode": self._mode,
            "best_distance": (
                self._all_dist[best_idx] if best_idx is not None else None
            ),
            "best_iteration": best_idx + 1 if best_idx is not None else None,
            "best_rgb": (
                self._all_Y[best_idx] if best_idx is not None else None
            ),
            "converged": self._converged,
            "error": self._error,
            "history": list(self._history),
            "calibration_done": self._calibration_done,
            "pure_rgbs": self._pure_rgbs,
            "water_rgb": self._water_rgb,
            "gamut_suggested_targets": self._gamut.get("suggested_targets") if self._gamut else None,
            "custom_target": self._custom_target,
            "plan": plan,
            "active_run": active_run,
        }

    def get_plan(self, mode: str = "quick") -> dict:
        """Build and return a plan without starting the loop."""
        seed = self.config.get("seed", 42)
        start_col = 1 if self._calibration_done else 0
        return build_plan(
            self.config,
            mode,
            seed=seed,
            start_column=start_col,
            pre_calibrated=self._calibration_done,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop_wrapper(self, mode: str):
        """Top-level wrapper that catches unhandled exceptions."""
        try:
            self._run_loop(mode)
        except Exception as exc:
            logger.error("Active learning loop crashed: %s", exc, exc_info=True)
            self._error = str(exc)
            self._state = "failed"
        finally:
            self._active_run_id = None
            self._live_progress = None
            # Always write analysis output at termination
            self._save_analysis()
            self._finalize_sequence()

    def _run_loop(self, mode: str):
        cfg = self.config
        plan = self._plan
        target_rgb = np.array(plan["target_rgb"], dtype=float)
        # Override target if user selected from gamut
        if self._custom_target:
            target_rgb = np.array(self._custom_target, dtype=float)
            logger.info("Using custom target from calibration: %s", target_rgb.tolist())
        total_volume = float(cfg.get("total_volume_ul", 200))
        threshold = plan["convergence_threshold"]
        seed = cfg.get("seed", 42)
        n_initial = plan["n_initial"]
        n_opt = plan["n_optimization"]
        total = plan["total_iterations"]
        start_column = plan.get("start_column", 0)
        pre_calibrated = plan.get("pre_calibrated", False)

        bounds_cfg = cfg["volume_bounds"]
        ml_cfg = cfg.get("ml", {})
        model_type = ml_cfg.get("model", "correlated_gp")
        acq_kind = ml_cfg.get("acquisition", "EI")
        distance_metric = plan["distance_metric"]

        settle_time = cfg.get("experiment", {}).get("settle_time_seconds", 10)
        mix_cycles = plan["mix_cycles"]
        skip_after_first = plan["skip_controls_after_first"]
        grid_path = self._resolved_grid_path()

        # Rinse config — quick mode uses 2 cycles instead of config default
        clean_cfg = cfg.get("cleaning", {})
        rinse_volume = float(clean_cfg.get("rinse_volume_ul", 200))
        rinse_cycles = clean_cfg.get("rinse_cycles", 3)
        if mode == "quick":
            rinse_cycles = min(rinse_cycles, 2)

        # Pre-generate random volumes (same seed as plan for consistency)
        rng = np.random.default_rng(seed=seed)
        random_volumes = sample_simplex(n_initial, total_volume, d=3, rng=rng)

        for iteration in range(total):
            # ---- Check stop before starting iteration ----
            if self._stop_requested:
                self._mark_remaining_skipped(iteration)
                self._state = "stopped"
                logger.info("Stopped before iteration %d", iteration + 1)
                return

            col_idx = (iteration + start_column) % 12
            phase = "random" if iteration < n_initial else "bo"
            skip_controls = skip_after_first and (pre_calibrated or iteration > 0)

            # ---- Determine volumes ----
            if phase == "random":
                volumes = random_volumes[iteration].tolist()
            else:
                # Ask GP for suggestion
                class _Params:
                    def __init__(self, params):
                        self.params = params

                suggest_result = handle_suggest_volumes(
                    _Params(
                        {
                            "target_rgb": target_rgb.tolist(),
                            "total_volume": total_volume,
                            "all_X": list(self._all_X),
                            "all_Y": list(self._all_Y),
                            "distance_metric": distance_metric,
                            "acquisition": acq_kind,
                            "model_type": model_type,
                            "bounds_min": bounds_cfg["min"],
                            "bounds_max": bounds_cfg["max"],
                            "phase": "bo",
                            "seed": seed,
                            "iteration": iteration,
                        }
                    )
                )
                volumes = suggest_result["volumes"]
                if suggest_result.get("error"):
                    logger.warning("GP error: %s", suggest_result["error"])

            Vr, Vg, Vb = volumes
            logger.info(
                "Iteration %d/%d [%s] col=%d volumes=(%.1f, %.1f, %.1f)",
                iteration + 1,
                total,
                phase,
                col_idx + 1,
                Vr,
                Vg,
                Vb,
            )

            # Update plan status
            if iteration < len(plan["iterations"]):
                plan["iterations"][iteration]["status"] = "running"
                if plan["iterations"][iteration]["volumes"] is None:
                    plan["iterations"][iteration]["volumes"] = [
                        round(Vr, 1),
                        round(Vg, 1),
                        round(Vb, 1),
                    ]

            # ---- Build steps for this column ----
            steps = self._build_column_steps(
                iteration=iteration,
                col_idx=col_idx,
                volumes=volumes,
                skip_controls=skip_controls,
                settle_time=settle_time,
                mix_cycles=mix_cycles,
                rinse_cycles=rinse_cycles,
                rinse_volume=rinse_volume,
                grid_path=grid_path,
                target_rgb=target_rgb.tolist(),
                distance_metric=distance_metric,
                convergence_threshold=threshold,
            )

            # ---- Create run via store ----
            run_steps = [RunStep(**s) for s in steps]
            run = TaskRun(
                name=f"Iteration {iteration + 1} [{phase}] col {col_idx + 1}",
                steps=run_steps,
                sequence_id=self._sequence_id,
                metadata={
                    "iteration": iteration,
                    "phase": phase,
                    "col_idx": col_idx,
                    "volumes": volumes,
                },
            )

            try:
                run = self.store.create_run(run)
            except Exception as exc:
                msg = f"Failed to create run for iteration {iteration + 1}: {exc}"
                logger.error(msg)
                self._error = msg
                self._mark_plan_item(iteration, "failed")
                self._state = "failed"
                return

            run_id = run.id
            self._active_run_id = run_id
            self._live_progress = None
            run_context = {"run_id": run_id}

            # Add run to sequence
            try:
                seq = self.store.load_sequence(self._sequence_id)
                seq.run_ids.append(run_id)
                self.store.save_sequence(seq)
            except Exception as exc:
                logger.warning("Could not update sequence: %s", exc)

            # ---- Execute the run ----
            try:
                run = self.runner.run_until_pause_or_done(run_id, context=run_context)
            except Exception as exc:
                msg = f"Run execution failed at iteration {iteration + 1}: {exc}"
                logger.error(msg, exc_info=True)
                self._record_history(
                    iteration, phase, col_idx, volumes,
                    None, None, run_id, "failed", str(exc),
                )
                self._error = msg
                self._mark_plan_item(iteration, "failed")
                self._state = "failed"
                return

            # ---- Handle pause/resume cycle ----
            run = self.store.load_run(run_id)

            while run.status == RunStatus.paused:
                self._state = "paused"
                logger.info(
                    "Run paused at iteration %d -- waiting for resume or stop",
                    iteration + 1,
                )

                if self._stop_requested:
                    self._active_run_id = None
                    self._live_progress = None
                    self._mark_remaining_skipped(iteration)
                    self._state = "stopped"
                    logger.info("Stopped while paused at iteration %d", iteration + 1)
                    return

                # Block until resume() or stop() signals us
                self._resume_event.clear()
                self._resume_event.wait()

                if self._stop_requested:
                    self._active_run_id = None
                    self._live_progress = None
                    self._mark_remaining_skipped(iteration)
                    self._state = "stopped"
                    logger.info("Stopped while paused at iteration %d", iteration + 1)
                    return

                # resume() set _state = "running" -- now actually resume the run
                try:
                    run = self.runner.resume(run_id, context=run_context)
                except Exception as exc:
                    msg = f"Resume failed at iteration {iteration + 1}: {exc}"
                    logger.error(msg, exc_info=True)
                    self._error = msg
                    self._mark_plan_item(iteration, "failed")
                    self._active_run_id = None
                    self._live_progress = None
                    self._state = "failed"
                    return

                # Reload to check if it paused again or completed
                run = self.store.load_run(run_id)

            # ---- After run_until_pause_or_done returned with stop requested ----
            if self._stop_requested:
                self._active_run_id = None
                self._live_progress = None
                self._mark_remaining_skipped(iteration)
                self._state = "stopped"
                logger.info("Stopped after iteration %d run completed", iteration + 1)
                return

            # If the run failed inside the runner, propagate
            if run.status == RunStatus.failed:
                failed_step = None
                for s in run.steps:
                    if s.status == StepStatus.failed:
                        failed_step = s
                        break
                err_detail = (
                    failed_step.error if failed_step else "unknown step failure"
                )
                msg = f"Run failed at iteration {iteration + 1}: {err_detail}"
                logger.error(msg)
                self._record_history(
                    iteration, phase, col_idx, volumes,
                    None, None, run_id, "failed", err_detail,
                )
                self._error = msg
                self._mark_plan_item(iteration, "failed")
                self._active_run_id = None
                self._live_progress = None
                self._state = "failed"
                return

            self._state = "running"

            # ---- Data wiring: capture -> extract_rgb -> fit_gp ----
            capture_output = self._find_step_output(run, "capture")
            image_path = None
            if capture_output:
                image_path = (
                    capture_output.get("image_path")
                    or capture_output.get("path")
                )

            if not image_path:
                msg = f"No image captured at iteration {iteration + 1}"
                logger.error(msg)
                self._record_history(
                    iteration, phase, col_idx, volumes,
                    None, None, run_id, "failed", msg,
                )
                self._error = msg
                self._mark_plan_item(iteration, "failed")
                self._active_run_id = None
                self._live_progress = None
                self._state = "failed"
                return

            # Call handle_extract_rgb with the real image path
            class _Params:
                def __init__(self, params):
                    self.params = params

            rgb_result = handle_extract_rgb(
                _Params(
                    {
                        "image_path": image_path,
                        "col": col_idx,
                        "grid_path": grid_path,
                        "cached_reference": self._cached_reference,
                        "skip_controls": skip_controls,
                    }
                )
            )

            if rgb_result.get("error"):
                msg = f"extract_rgb failed: {rgb_result['error']}"
                logger.error(msg)
                self._record_history(
                    iteration, phase, col_idx, volumes,
                    None, None, run_id, "failed", rgb_result["error"],
                )
                self._error = msg
                self._mark_plan_item(iteration, "failed")
                self._active_run_id = None
                self._live_progress = None
                self._state = "failed"
                return

            observed_rgb = rgb_result["mean_rgb"]
            used_calibration = rgb_result.get("used_calibration", False)

            # Cache reference from first column
            if self._cached_reference is None and rgb_result.get(
                "runtime_reference"
            ):
                self._cached_reference = rgb_result["runtime_reference"]

            # Compute distance
            try:
                dist = float(
                    color_distance(
                        np.array(observed_rgb), target_rgb, distance_metric
                    )
                )
            except Exception:
                dist = float("inf")

            # Accumulate observations
            self._all_X.append(list(volumes))
            self._all_Y.append(list(observed_rgb))
            self._all_dist.append(dist)

            # ---- Write results back to run steps ----
            self._write_step_outputs(
                run,
                rgb_result=rgb_result,
                dist=dist,
                volumes=volumes,
                observed_rgb=observed_rgb,
                target_rgb=target_rgb.tolist(),
                distance_metric=distance_metric,
                convergence_threshold=threshold,
                iteration=iteration,
            )

            self._current_iteration = iteration + 1  # 1-based completed count

            self._record_history(
                iteration,
                phase,
                col_idx,
                volumes,
                observed_rgb,
                dist,
                run_id,
                "completed",
                used_calibration=used_calibration,
            )

            # Update plan
            if iteration < len(plan["iterations"]):
                plan["iterations"][iteration]["status"] = "completed"

            logger.info(
                "  Observed: (%.0f, %.0f, %.0f)  Distance: %.1f  Cal: %s",
                observed_rgb[0],
                observed_rgb[1],
                observed_rgb[2],
                dist,
                used_calibration,
            )

            # Clear active run id since this iteration is done
            self._active_run_id = None
            self._live_progress = None

            # ---- Check pause between iterations ----
            # If pause was requested but the run finished before the runner
            # could act on it, honour the pause here between iterations.
            if self._state in ("pausing", "paused"):
                self._state = "paused"
                logger.info(
                    "Paused between iterations (after iteration %d)",
                    iteration + 1,
                )
                self._resume_event.clear()
                self._resume_event.wait()

                if self._stop_requested:
                    self._mark_remaining_skipped(iteration + 1)
                    self._state = "stopped"
                    return

                self._state = "running"

            # ---- Check convergence ----
            if dist < threshold:
                self._converged = True
                self._state = "converged"
                self._mark_remaining_skipped(iteration + 1)
                logger.info(
                    "*** CONVERGED at iteration %d, distance %.1f ***",
                    iteration + 1,
                    dist,
                )
                return

        # Finished all iterations without convergence
        self._state = "completed"
        best_idx = (
            int(np.argmin(self._all_dist)) if self._all_dist else 0
        )
        logger.info(
            "=== Loop Complete ===  %d iterations, best dist=%.1f at iter %d",
            len(self._all_dist),
            self._all_dist[best_idx] if self._all_dist else float("inf"),
            best_idx + 1,
        )

    # ------------------------------------------------------------------
    # Analysis output
    # ------------------------------------------------------------------

    def _save_analysis(self):
        """Save analysis artifacts to data/analysis/{sequence_id}/."""
        if not self._sequence_id:
            return
        if self._state not in (
            "completed",
            "converged",
            "failed",
            "stopped",
        ):
            return

        base = Path(self.store.base_dir) if hasattr(self.store, 'base_dir') else Path("data")
        out_dir = base / "analysis" / self._sequence_id
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create analysis dir %s: %s", out_dir, exc)
            return

        plan = self._plan or {}
        best_idx = int(np.argmin(self._all_dist)) if self._all_dist else None

        # summary.json
        summary = {
            "sequence_id": self._sequence_id,
            "state": self._state,
            "target_rgb": plan.get("target_rgb"),
            "best_distance": (
                self._all_dist[best_idx] if best_idx is not None else None
            ),
            "best_iteration": best_idx + 1 if best_idx is not None else None,
            "best_rgb": (
                self._all_Y[best_idx] if best_idx is not None else None
            ),
            "converged": self._converged,
            "total_iterations_run": self._current_iteration,
            "distances": list(self._all_dist),
            "error": self._error,
        }
        try:
            with open(out_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Failed to write summary.json: %s", exc)

        # history.csv
        if self._history:
            try:
                fieldnames = list(self._history[0].keys())
                with open(out_dir / "history.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self._history)
            except Exception as exc:
                logger.warning("Failed to write history.csv: %s", exc)

        # plan.json
        try:
            with open(out_dir / "plan.json", "w") as f:
                json.dump(plan, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Failed to write plan.json: %s", exc)

        logger.info("Analysis saved to %s", out_dir)

    def _finalize_sequence(self):
        """Update the RunSequence status to match the terminal state."""
        if not self._sequence_id:
            return
        status_map = {
            "completed": "completed",
            "converged": "completed",
            "failed": "failed",
            "stopped": "stopped",
        }
        seq_status = status_map.get(self._state)
        if not seq_status:
            return
        try:
            seq = self.store.load_sequence(self._sequence_id)
            seq.status = seq_status
            self.store.save_sequence(seq)
        except Exception as exc:
            logger.warning("Could not finalize sequence status: %s", exc)

    def _active_run_snapshot(self) -> Optional[dict]:
        """Return the live progress of the currently attached run."""
        if not self._active_run_id:
            return None

        try:
            run = self.store.load_run(self._active_run_id)
        except Exception as exc:
            logger.warning(
                "Failed to load active run %s: %s",
                self._active_run_id,
                exc,
            )
            return {
                "id": self._active_run_id,
                "status": "unavailable",
                "iteration": None,
                "phase": None,
                "column": None,
                "volumes": None,
                "current_step_index": None,
                "current_step_number": None,
                "total_steps": None,
                "current_step": None,
                "steps": [],
                "live_progress": None,
            }

        metadata = run.metadata or {}
        steps = [
            {
                "index": idx,
                "number": idx + 1,
                "id": step.id,
                "name": step.name,
                "kind": step.kind,
                "status": step.status.value,
                "detail": self._format_step_detail(step),
            }
            for idx, step in enumerate(run.steps)
        ]
        current_step = None
        if 0 <= run.current_step_index < len(run.steps):
            step = run.steps[run.current_step_index]
            current_step = {
                "id": step.id,
                "name": step.name,
                "kind": step.kind,
                "status": step.status.value,
                "detail": self._format_step_detail(step),
            }

        return {
            "id": run.id,
            "status": run.status.value,
            "iteration": (
                int(metadata["iteration"]) + 1
                if metadata.get("iteration") is not None
                else None
            ),
            "phase": metadata.get("phase"),
            "column": (
                int(metadata["col_idx"]) + 1
                if metadata.get("col_idx") is not None
                else None
            ),
            "volumes": metadata.get("volumes"),
            "current_step_index": run.current_step_index,
            "current_step_number": (
                min(run.current_step_index + 1, len(run.steps))
                if run.steps
                else 0
            ),
            "total_steps": len(run.steps),
            "current_step": current_step,
            "steps": steps,
            "live_progress": (
                dict(self._live_progress)
                if self._live_progress
                and self._live_progress.get("run_id") == run.id
                else None
            ),
        }

    def _format_step_detail(self, step: RunStep) -> str:
        """Build a short label for the currently running step."""
        params = step.params or {}
        if step.kind == "transfer":
            volume = params.get("volume")
            volume_str = (
                f"{float(volume):.0f}uL"
                if isinstance(volume, (int, float))
                else "--"
            )
            return (
                f"{params.get('source_well', '?')} -> "
                f"{params.get('dest_well', '?')} ({volume_str})"
            )
        if step.kind == "mix":
            return (
                f"{params.get('mix_well', '?')} x{params.get('cycles', '?')} "
                f"({params.get('volume', '--')}uL)"
            )
        if step.kind == "wait":
            return f"{params.get('seconds', '--')}s settle"
        if step.kind == "capture":
            return params.get("label", "capture")
        if step.kind == "extract_rgb":
            return "post-hoc image analysis"
        if step.kind == "fit_gp":
            return "post-hoc GP update"
        if step.kind == "use_pipette":
            return f"mount={params.get('mount', '?')}"
        return step.name

    def update_live_progress(self, payload: dict) -> None:
        """Receive fine-grained robot progress updates from OT2 handlers."""
        run_id = payload.get("run_id")
        if self._active_run_id and run_id and run_id != self._active_run_id:
            return
        self._live_progress = {
            "run_id": run_id or self._active_run_id,
            "step_id": payload.get("step_id"),
            "step_name": payload.get("step_name"),
            "step_kind": payload.get("step_kind"),
            "action": payload.get("action"),
            "detail": payload.get("detail"),
            "cycle": payload.get("cycle"),
            "total_cycles": payload.get("total_cycles"),
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    # Plan helpers
    # ------------------------------------------------------------------

    def _mark_remaining_skipped(self, from_iteration: int):
        """Mark plan items from *from_iteration* onward as skipped."""
        if not self._plan:
            return
        for j in range(from_iteration, len(self._plan["iterations"])):
            self._plan["iterations"][j]["status"] = "skipped"

    # ------------------------------------------------------------------
    # Data wiring: write results back to run steps
    # ------------------------------------------------------------------

    def _write_step_outputs(
        self,
        run: TaskRun,
        *,
        rgb_result: dict,
        dist: float,
        volumes: list,
        observed_rgb: list,
        target_rgb: list,
        distance_metric: str,
        convergence_threshold: float,
        iteration: int,
    ):
        """Write post-hoc data back into the extract_rgb and fit_gp steps."""
        for step in run.steps:
            if step.kind == "extract_rgb":
                step.output = rgb_result
                step.status = StepStatus.succeeded
            elif step.kind == "fit_gp":
                best_idx = int(np.argmin(self._all_dist))
                step.output = {
                    "distance": dist,
                    "converged": dist < convergence_threshold,
                    "best_distance": self._all_dist[best_idx],
                    "best_iteration": best_idx + 1,
                    "all_X": list(self._all_X),
                    "all_Y": list(self._all_Y),
                    "all_dist": list(self._all_dist),
                }
                step.status = StepStatus.succeeded
        try:
            self.store.save_run(run)
        except Exception as exc:
            logger.warning("Could not save run with updated outputs: %s", exc)

    # ------------------------------------------------------------------
    # History recording
    # ------------------------------------------------------------------

    def _mark_plan_item(self, iteration: int, status: str):
        """Update the plan item status for a given iteration."""
        if self._plan and iteration < len(self._plan.get("iterations", [])):
            self._plan["iterations"][iteration]["status"] = status

    def _record_history(
        self,
        iteration,
        phase,
        col_idx,
        volumes,
        observed_rgb,
        distance,
        run_id,
        status,
        error=None,
        used_calibration=False,
    ):
        entry: dict[str, Any] = {
            "iteration": iteration + 1,
            "phase": phase,
            "column": col_idx + 1,
            "volumes": (
                [round(v, 1) for v in volumes] if volumes else None
            ),
            "observed_rgb": (
                [round(float(c), 0) for c in observed_rgb]
                if observed_rgb is not None
                else None
            ),
            "distance": round(distance, 1) if distance is not None else None,
            "run_id": run_id,
            "status": status,
            "used_calibration": used_calibration,
        }
        if error:
            entry["error"] = str(error)
        self._history.append(entry)

    # ------------------------------------------------------------------
    # Step output extraction
    # ------------------------------------------------------------------

    def _find_step_output(
        self, run: TaskRun, kind: str
    ) -> Optional[dict]:
        """Find the output of the first step matching *kind*."""
        for s in run.steps:
            if getattr(s, "kind", None) == kind and getattr(s, "output", None):
                return s.output
        return None

    # ------------------------------------------------------------------
    # Column step builder
    # ------------------------------------------------------------------

    def _build_column_steps(
        self,
        iteration,
        col_idx,
        volumes,
        skip_controls,
        settle_time=10,
        mix_cycles=3,
        rinse_cycles=None,
        rinse_volume=None,
        grid_path="",
        target_rgb=None,
        distance_metric="rgb_euclidean",
        convergence_threshold=50.0,
    ) -> List[Dict[str, Any]]:
        """Generate step dicts for one column.

        Parameter names match the OT-2 handlers exactly.
        """
        well_col = col_idx + 1
        control_volume = float(self.config.get("total_volume_ul", 200))
        mix_volume = self.config.get("experiment", {}).get(
            "mix_volume_ul", 150
        )
        Vr, Vg, Vb = float(volumes[0]), float(volumes[1]), float(volumes[2])

        # Optional per-step rinse overrides (e.g. quick mode uses fewer cycles)
        _rinse_kw: Dict[str, Any] = {}
        if rinse_cycles is not None:
            _rinse_kw["rinse_cycles"] = rinse_cycles
        if rinse_volume is not None:
            _rinse_kw["rinse_volume"] = rinse_volume

        steps: List[Dict[str, Any]] = []

        # ---- Controls (left pipette) ----
        if not skip_controls:
            steps.append(
                {
                    "name": "Select left pipette",
                    "kind": "use_pipette",
                    "params": {"mount": "left"},
                }
            )

            controls = [
                ("Red -> A", "7", "A1", f"A{well_col}", "A6"),
                ("Green -> B", "8", "B1", f"B{well_col}", "A6"),
                ("Blue -> C", "9", "C1", f"C{well_col}", "A6"),
                ("Water -> D", "5", "D1", f"D{well_col}", "A5"),
            ]
            for label, src_slot, tip, dest, rinse in controls:
                steps.append(
                    {
                        "name": f"Control: {label}{well_col}",
                        "kind": "transfer",
                        "params": {
                            "tiprack_slot": "10",
                            "source_slot": src_slot,
                            "dest_slot": "1",
                            "tip_well": tip,
                            "source_well": "A1",
                            "dest_well": dest,
                            "volume": control_volume,
                            "cleaning_slot": "4",
                            "rinse_well": rinse,
                            **_rinse_kw,
                        },
                    }
                )

        # ---- Experiments (right pipette) ----
        steps.append(
            {
                "name": "Select right pipette",
                "kind": "use_pipette",
                "params": {"mount": "right"},
            }
        )

        dyes = [
            ("Red", "7", "A1", Vr, "A1"),
            ("Green", "8", "A2", Vg, "A2"),
            ("Blue", "9", "A3", Vb, "A3"),
        ]
        for label, src_slot, tip, vol, rinse in dyes:
            if vol >= 1.0:
                steps.append(
                    {
                        "name": f"Exp: {label} {vol:.0f}uL -> col{well_col}",
                        "kind": "transfer",
                        "params": {
                            "tiprack_slot": "11",
                            "source_slot": src_slot,
                            "dest_slot": "1",
                            "tip_well": tip,
                            "source_well": "A1",
                            "dest_well": f"A{well_col}",
                            "volume": vol,
                            "cleaning_slot": "4",
                            "rinse_well": rinse,
                            **_rinse_kw,
                        },
                    }
                )

        # ---- Mix ----
        steps.append(
            {
                "name": f"Mix col{well_col}",
                "kind": "mix",
                "params": {
                    "tiprack_slot": "11",
                    "plate_slot": "1",
                    "tip_well": "A4",
                    "mix_well": f"A{well_col}",
                    "cycles": mix_cycles,
                    "volume": mix_volume,
                    "cleaning_slot": "4",
                    "rinse_well": "A4",
                    **_rinse_kw,
                },
            }
        )

        # ---- Wait ----
        steps.append(
            {
                "name": f"Settle {settle_time}s",
                "kind": "wait",
                "params": {"seconds": settle_time},
            }
        )

        # ---- Capture ----
        steps.append(
            {
                "name": f"Capture col{well_col}",
                "kind": "capture",
                "params": {"label": f"col{well_col}_iter{iteration + 1}"},
            }
        )

        # ---- extract_rgb (placeholder -- loop wires real data post-hoc) ----
        steps.append(
            {
                "name": f"Extract RGB col{well_col}",
                "kind": "extract_rgb",
                "params": {
                    "image_path": "",
                    "col": col_idx,
                    "grid_path": grid_path,
                    "cached_reference": self._cached_reference,
                    "skip_controls": skip_controls,
                },
            }
        )

        # ---- fit_gp (placeholder -- loop wires real data post-hoc) ----
        steps.append(
            {
                "name": f"Fit GP iter{iteration + 1}",
                "kind": "fit_gp",
                "params": {
                    "volumes": list(volumes),
                    "observed_rgb": [],
                    "target_rgb": target_rgb or self.config["target_color"],
                    "distance_metric": distance_metric,
                    "convergence_threshold": convergence_threshold,
                    "iteration": iteration,
                    "all_X": list(self._all_X),
                    "all_Y": list(self._all_Y),
                    "all_dist": list(self._all_dist),
                },
            }
        )

        return steps
