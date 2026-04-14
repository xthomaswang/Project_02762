"""Plugin-driven active learning loop manager.

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

Integration layer only
----------------------
All task semantics (state, planning, run construction, result handling,
calibration post-processing) are delegated to a
:class:`~openot2.task_api.plugin.TaskPlugin`.  This module only handles
execution, the state machine, persistence, and generic run flow.

Step data is wired via the runner's ``$ref`` binding mechanism.
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

from openot2.control.models import (
    RunSequence,
    RunStep,
    RunStatus,
    StepStatus,
    TaskRun,
)

logger = logging.getLogger(__name__)


# ======================================================================
# ActiveLearningLoop
# ======================================================================

class ActiveLearningLoop:
    """Plugin-driven active learning loop manager.

    Delegates all task semantics to a :class:`TaskPlugin`.  This class
    handles run execution (via :class:`TaskRunner`), state machine
    (idle/running/paused/…), pause / resume / stop, persistence
    (sequences, analysis output), and generic calibration flow.

    Calibration artifacts (pure dye colours, gamut, etc.) are stored in
    ``_task_state["artifacts"]["calibration"]`` and owned by the plugin.
    """

    def __init__(self, runner, store, plugin, config, config_path: str):
        self.runner = runner
        self.store = store
        self.plugin = plugin
        self._config = config
        self.config_path = config_path
        self._reset_state()
        # Load persisted calibration via plugin
        if hasattr(self.plugin, "load_calibration_state"):
            cal = self.plugin.load_calibration_state(config_path)
            if cal:
                self._task_state.setdefault("artifacts", {})["calibration"] = cal

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
        self._active_run_id: Optional[str] = None
        self._live_progress: Optional[dict] = None

        # Threading primitives for pause/resume
        self._resume_event = threading.Event()

        # Plugin-managed task state (includes calibration artifacts)
        self._task_state: dict = {}

        # Plan (populated at start time)
        self._plan: Optional[dict] = None

    @property
    def task_cfg(self):
        """Return the task config (loaded by plugin)."""
        return self._config

    # -- calibration artifact helpers (read from task state) --

    def _cal(self) -> dict:
        """Return the calibration artifacts dict (may be empty)."""
        return self._task_state.get("artifacts", {}).get("calibration", {})

    @property
    def _calibration_done(self) -> bool:
        return self._cal().get("done", False)

    # ------------------------------------------------------------------
    # Plan enrichment
    # ------------------------------------------------------------------

    def _enrich_plan(self, plugin_plan: dict, mode: str) -> dict:
        """Enrich the plugin's plan with per-iteration UI details."""
        from src.ml import sample_simplex

        total = plugin_plan.get("total_iterations", 0)
        n_initial = plugin_plan.get("n_initial", 0)
        seed = getattr(self._config, "seed", 42)
        total_volume = float(getattr(self._config, "total_volume_ul", 200))
        calibration_done = self._calibration_done
        start_column = 1 if calibration_done else 0
        pre_calibrated = calibration_done

        if mode == "quick":
            mix_cycles = 2
            skip_controls_after_first = True
        else:
            exp = getattr(self._config, "experiment", None)
            mix_cycles = getattr(exp, "mix_cycles", 3) if exp else 3
            skip_controls_after_first = False

        # Pre-sample random volumes for display
        rng = np.random.default_rng(seed=seed)
        random_vols = sample_simplex(max(n_initial, 1), total_volume, d=3, rng=rng)

        target_rgb = list(getattr(self._config, "target_color", [0, 0, 0]))
        threshold = float(getattr(self._config, "convergence_threshold", 50))
        ml = getattr(self._config, "ml", None)
        distance_metric = getattr(ml, "distance_metric", "rgb_euclidean") if ml else "rgb_euclidean"

        # Adjust n_initial when pre-calibrated
        if pre_calibrated:
            n_initial = max(n_initial - 1, 0)
            total = n_initial + plugin_plan.get("n_optimization", 0)

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
            if phase == "random" and i < len(random_vols):
                v = random_vols[i]
                entry["volumes"] = [
                    round(float(v[0]), 1),
                    round(float(v[1]), 1),
                    round(float(v[2]), 1),
                ]
            else:
                entry["volumes"] = None
            iterations.append(entry)

        return {
            "mode": mode,
            "target_rgb": target_rgb,
            "convergence_threshold": threshold,
            "total_iterations": total,
            "n_initial": n_initial,
            "n_initial_configured": plugin_plan.get("n_initial", 0),
            "n_optimization": plugin_plan.get("n_optimization", 0),
            "mix_cycles": mix_cycles,
            "skip_controls_after_first": skip_controls_after_first,
            "pre_calibrated": pre_calibrated,
            "start_column": start_column,
            "distance_metric": distance_metric,
            "model_type": getattr(ml, "model", "correlated_gp") if ml else "correlated_gp",
            "acquisition": getattr(ml, "acquisition", "EI") if ml else "EI",
            "iterations": iterations,
        }

    # ------------------------------------------------------------------
    # Volume suggestion
    # ------------------------------------------------------------------

    def _prepare_volumes(self, iteration: int, mode: str) -> list:
        """Compute suggested volumes for an iteration and store in state.

        Returns the computed volumes list.
        """
        from src.ml import sample_simplex
        from src.web.handlers import handle_suggest_volumes

        n_initial = self._task_state.get("n_initial", 0)
        phase = "random" if iteration < n_initial else "bo"
        cache = self._task_state.get("cache", {})
        metrics = self._task_state.get("metrics", {})

        total_volume = float(getattr(self._config, "total_volume_ul", 200))
        seed = getattr(self._config, "seed", 42)

        if phase == "random":
            rng = np.random.default_rng(seed=seed)
            all_vols = sample_simplex(n_initial, total_volume, d=3, rng=rng)
            volumes = all_vols[iteration].tolist() if iteration < len(all_vols) else [0.0, 0.0, 0.0]
        else:
            target_rgb = metrics.get(
                "target_rgb",
                list(getattr(self._config, "target_color", [0, 0, 0])),
            )
            ml = getattr(self._config, "ml", None)

            class _Params:
                def __init__(self, params):
                    self.params = params

            suggest_result = handle_suggest_volumes(
                _Params({
                    "target_rgb": target_rgb,
                    "total_volume": total_volume,
                    "all_X": list(cache.get("all_X", [])),
                    "all_Y": list(cache.get("all_Y", [])),
                    "distance_metric": metrics.get("distance_metric", "rgb_euclidean"),
                    "acquisition": getattr(ml, "acquisition", "EI") if ml else "EI",
                    "model_type": getattr(ml, "model", "correlated_gp") if ml else "correlated_gp",
                    "bounds_min": list(getattr(self._config, "volume_bounds_min", [0, 0, 0])),
                    "bounds_max": list(getattr(self._config, "volume_bounds_max", [200, 200, 200])),
                    "phase": "bo",
                    "seed": seed,
                    "iteration": iteration,
                })
            )
            volumes = suggest_result["volumes"]
            if suggest_result.get("error"):
                logger.warning("GP error: %s", suggest_result["error"])

        self._task_state["_next_volumes"] = list(volumes)
        return volumes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, mode: str = "quick") -> str:
        """Start the active learning loop in a background thread.

        Returns the sequence ID for the new run.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Loop is already running")

        # Snapshot calibration artifacts from task state
        cal_snapshot = (self._task_state.get("artifacts", {}).get("calibration") or {}).copy()

        self._reset_state()
        self._mode = mode
        self._state = "running"

        # Initialise task state via plugin
        self._task_state = self.plugin.initial_state(self._config, mode)

        # Restore calibration artifacts
        if cal_snapshot:
            self._task_state.setdefault("artifacts", {})["calibration"] = cal_snapshot
            # Inject cached reference into cache for build_iteration_run
            cached_ref = cal_snapshot.get("cached_reference")
            if cached_ref:
                self._task_state.setdefault("cache", {})["cached_reference"] = cached_ref

        # Build plan via plugin, then enrich for UI
        plugin_plan = self.plugin.build_plan(self._config, self._task_state, mode)
        self._plan = self._enrich_plan(plugin_plan, mode)

        # Override target if user selected from gamut
        custom_target = self._cal().get("custom_target")
        if custom_target:
            self._task_state.setdefault("metrics", {})["target_rgb"] = list(custom_target)

        # Create persistent sequence
        seq = RunSequence(
            name=f"{self.plugin.name}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
        """Request a pause at the next safe-point."""
        if self._state == "running":
            self._state = "pausing"
            if self._active_run_id:
                try:
                    self.runner.request_pause(self._active_run_id)
                except Exception as exc:
                    logger.warning("request_pause failed: %s", exc)
            logger.info("Pause requested")

    def resume(self):
        """Resume from a paused state."""
        if self._state == "paused":
            self._state = "running"
            self._resume_event.set()
            logger.info("Resumed")

    def stop(self):
        """Emergency stop — works in ANY active state."""
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
        """Run a calibration procedure.

        Builds and executes a calibration run via the plugin, then
        hands the completed run to the plugin for post-processing.
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
        """Execute calibration synchronously.

        Generic flow: build → execute → hand to plugin → update state.
        """
        # Build calibration run via plugin
        cal_run = self.plugin.build_calibration_run(
            self._config, self._task_state or {},
        )

        # Create and execute run
        cal_run = self.store.create_run(cal_run)

        self._active_run_id = cal_run.id
        run_context = {"run_id": cal_run.id, "config_path": self.config_path}
        self.runner.run_until_pause_or_done(cal_run.id, context=run_context)
        self._active_run_id = None

        # Check if E-Stop was pressed during calibration
        if self._stop_requested:
            self._state = "stopped"
            return {"ok": False, "error": "Calibration stopped by user"}

        # Reload run to get outputs
        cal_run = self.store.load_run(cal_run.id)

        # Check run succeeded
        if cal_run.status != RunStatus.completed:
            self._state = "idle"
            failed_step = next((s for s in cal_run.steps if s.error), None)
            return {
                "ok": False,
                "error": f"Calibration run failed: "
                         f"{failed_step.error if failed_step else 'unknown'}",
            }

        # Hand completed run to plugin for task-specific post-processing
        if hasattr(self.plugin, "apply_calibration_result"):
            self._task_state, payload = self.plugin.apply_calibration_result(
                self._config, self._task_state or {}, cal_run, self.config_path,
            )
        else:
            payload = {"ok": True}

        self._state = "idle"
        return payload

    def tip_check(self) -> dict:
        """Run a tip pick-up / drop check for both pipettes."""
        if self._state not in ("idle", "stopped", "completed", "converged", "failed"):
            return {"ok": False, "error": f"Cannot run tip check in state '{self._state}'"}

        prev_state = self._state
        self._state = "calibrating"

        try:
            # Build tip check run via plugin
            tip_run = self.plugin.build_tip_check_run(self._config, self._task_state or {})

            # Create and execute the run synchronously
            tip_run = self.store.create_run(tip_run)

            self._active_run_id = tip_run.id
            self.runner.run_until_pause_or_done(tip_run.id)
            self._active_run_id = None

            # Check result
            tip_run = self.store.load_run(tip_run.id)
            if tip_run.status != RunStatus.completed:
                failed_step = next(
                    (s for s in tip_run.steps if s.status == StepStatus.failed), None
                )
                self._state = prev_state
                return {
                    "ok": False,
                    "error": (
                        failed_step.error
                        if failed_step and failed_step.error
                        else f"Tip check run ended with status {tip_run.status.value}"
                    ),
                    "run_id": tip_run.id,
                }

            self._state = prev_state
            logger.info("Tip check completed successfully")
            return {"ok": True, "run_id": tip_run.id}

        except Exception as exc:
            logger.error("Tip check failed: %s", exc, exc_info=True)
            self._state = prev_state
            return {"ok": False, "error": str(exc)}

    def reset(self) -> dict:
        """Reset the loop to idle state.  Only allowed when not running."""
        if self._state in ("running", "pausing", "paused", "stopping"):
            return {"ok": False, "error": "Cannot reset while protocol is active"}
        if self._active_run_id is not None:
            return {"ok": False, "error": "Cannot reset while an active run is still attached"}
        cal_snapshot = (self._task_state.get("artifacts", {}).get("calibration") or {}).copy()
        self._reset_state()
        if cal_snapshot:
            self._task_state.setdefault("artifacts", {})["calibration"] = cal_snapshot
        return {"ok": True}

    def status(self) -> dict:
        """Return a comprehensive snapshot of loop state."""
        plan = self._plan or self._build_default_plan()

        # Read task state fields
        cache = self._task_state.get("cache", {}) if self._task_state else {}
        history = self._task_state.get("history", []) if self._task_state else []
        iteration = self._task_state.get("iteration", 0) if self._task_state else 0

        all_dist = cache.get("all_dist", [])
        all_Y = cache.get("all_Y", [])
        best_idx = int(np.argmin(all_dist)) if all_dist else None

        active_run = self._active_run_snapshot()

        result = {
            "state": self._state,
            "sequence_id": self._sequence_id,
            "iteration": iteration,
            "max_iterations": plan.get("total_iterations", 0),
            "n_initial": plan.get("n_initial", 0),
            "n_optimization": plan.get("n_optimization", 0),
            "convergence_threshold": plan.get("convergence_threshold", 50.0),
            "target_rgb": plan.get("target_rgb", []),
            "distance_metric": plan.get("distance_metric", ""),
            "mode": self._mode,
            "best_distance": (
                all_dist[best_idx] if best_idx is not None else None
            ),
            "best_iteration": best_idx + 1 if best_idx is not None else None,
            "best_rgb": (
                all_Y[best_idx] if best_idx is not None else None
            ),
            "converged": self._task_state.get("terminal", False) if self._task_state else False,
            "error": self._error,
            "history": list(history),
            "plan": plan,
            "active_run": active_run,
        }

        # Merge task extension status (calibration fields, etc.)
        ext = self.plugin.web_extension(self._config)
        if ext is not None and hasattr(ext, "extra_status"):
            result.update(ext.extra_status(self._config, self._task_state or {}))

        return result

    def _build_default_plan(self) -> dict:
        """Build a default plan for status display when no loop is started."""
        plugin_plan = self.plugin.build_plan(
            self._config, self._task_state or {}, self._mode
        )
        return self._enrich_plan(plugin_plan, self._mode)

    def get_plan(self, mode: str = "quick") -> dict:
        """Build and return a plan without starting the loop."""
        plugin_plan = self.plugin.build_plan(
            self._config, self._task_state or {}, mode,
        )
        return self._enrich_plan(plugin_plan, mode)

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
        plan = self._plan
        target_rgb = np.array(plan["target_rgb"], dtype=float)
        # Override target if user selected from gamut
        custom_target = self._cal().get("custom_target")
        if custom_target:
            target_rgb = np.array(custom_target, dtype=float)
            logger.info("Using custom target from calibration: %s", list(custom_target))

        total = plan["total_iterations"]
        threshold = plan["convergence_threshold"]

        for iteration in range(total):
            # ---- Check stop before starting iteration ----
            if self._stop_requested:
                self._mark_remaining_skipped(iteration)
                self._state = "stopped"
                logger.info("Stopped before iteration %d", iteration + 1)
                return

            n_initial = self._task_state.get("n_initial", plan.get("n_initial", 0))
            col_idx = (iteration + plan.get("start_column", 0)) % 12
            phase = "random" if iteration < n_initial else "bo"

            # ---- Determine volumes ----
            volumes = self._prepare_volumes(iteration, mode)

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
            if iteration < len(plan.get("iterations", [])):
                plan["iterations"][iteration]["status"] = "running"
                if plan["iterations"][iteration]["volumes"] is None:
                    plan["iterations"][iteration]["volumes"] = [
                        round(Vr, 1),
                        round(Vg, 1),
                        round(Vb, 1),
                    ]

            # ---- Build run via plugin ----
            run = self.plugin.build_iteration_run(
                self._config, self._task_state, iteration, mode,
            )
            run.sequence_id = self._sequence_id

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
            run_context = {"run_id": run_id, "config_path": self.config_path}

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
                self._record_history_entry(
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
                self._record_history_entry(
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

            # ---- Fold results via plugin ----
            self._task_state = self.plugin.apply_run_result(
                self._config, self._task_state, run, mode,
            )

            # Read history entry added by apply_run_result
            history = self._task_state.get("history", [])
            new_entry = history[-1] if history else {}

            # Augment history entry with framework fields
            if new_entry and "run_id" not in new_entry:
                new_entry["run_id"] = run_id
                new_entry["status"] = "completed"
                new_entry["column"] = col_idx + 1
                # Map plugin field names to legacy names for UI
                if "mean_rgb" in new_entry and "observed_rgb" not in new_entry:
                    new_entry["observed_rgb"] = new_entry["mean_rgb"]

            observed_rgb = new_entry.get("observed_rgb") or new_entry.get("mean_rgb")
            dist = new_entry.get("distance")

            # Update plan
            if iteration < len(plan.get("iterations", [])):
                plan["iterations"][iteration]["status"] = "completed"

            if observed_rgb and dist is not None:
                logger.info(
                    "  Observed: (%.0f, %.0f, %.0f)  Distance: %.1f  Cal: %s",
                    observed_rgb[0],
                    observed_rgb[1],
                    observed_rgb[2],
                    dist,
                    new_entry.get("used_calibration", False),
                )

            # Clear active run id since this iteration is done
            self._active_run_id = None
            self._live_progress = None

            # ---- Check pause between iterations ----
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

            # ---- Check terminal (convergence or max iterations) ----
            if self._task_state.get("terminal", False):
                self._state = "converged"
                self._mark_remaining_skipped(iteration + 1)
                logger.info(
                    "*** CONVERGED at iteration %d, distance %.1f ***",
                    iteration + 1,
                    dist if dist is not None else float("inf"),
                )
                return

        # Finished all iterations without convergence
        self._state = "completed"
        cache = self._task_state.get("cache", {})
        all_dist = cache.get("all_dist", [])
        best_idx = (
            int(np.argmin(all_dist)) if all_dist else 0
        )
        logger.info(
            "=== Loop Complete ===  %d iterations, best dist=%.1f at iter %d",
            len(all_dist),
            all_dist[best_idx] if all_dist else float("inf"),
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
        cache = self._task_state.get("cache", {}) if self._task_state else {}
        history = self._task_state.get("history", []) if self._task_state else []
        all_dist = cache.get("all_dist", [])
        all_Y = cache.get("all_Y", [])
        best_idx = int(np.argmin(all_dist)) if all_dist else None
        iteration = self._task_state.get("iteration", 0) if self._task_state else 0

        # summary.json
        summary = {
            "sequence_id": self._sequence_id,
            "state": self._state,
            "target_rgb": plan.get("target_rgb"),
            "best_distance": (
                all_dist[best_idx] if best_idx is not None else None
            ),
            "best_iteration": best_idx + 1 if best_idx is not None else None,
            "best_rgb": (
                all_Y[best_idx] if best_idx is not None else None
            ),
            "converged": self._task_state.get("terminal", False) if self._task_state else False,
            "total_iterations_run": iteration,
            "distances": list(all_dist),
            "error": self._error,
        }
        try:
            with open(out_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Failed to write summary.json: %s", exc)

        # history.csv
        if history:
            try:
                fieldnames = list(history[0].keys())
                with open(out_dir / "history.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(history)
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
            return "image analysis"
        if step.kind == "fit_gp":
            return "GP update + convergence check"
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
        for j in range(from_iteration, len(self._plan.get("iterations", []))):
            self._plan["iterations"][j]["status"] = "skipped"

    # ------------------------------------------------------------------
    # History recording (fallback for error cases)
    # ------------------------------------------------------------------

    def _mark_plan_item(self, iteration: int, status: str):
        """Update the plan item status for a given iteration."""
        if self._plan and iteration < len(self._plan.get("iterations", [])):
            self._plan["iterations"][iteration]["status"] = status

    def _record_history_entry(
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
        """Record a history entry directly (used for error cases where
        apply_run_result cannot be called)."""
        entry: dict[str, Any] = {
            "iteration": iteration,
            "phase": phase,
            "col_idx": col_idx,
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
        self._task_state.setdefault("history", []).append(entry)
