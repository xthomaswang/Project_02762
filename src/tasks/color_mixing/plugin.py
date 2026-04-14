"""Color mixing TaskPlugin implementation.

Satisfies the :class:`openot2.task_api.plugin.TaskPlugin` protocol.
All color-mixing-specific logic is encapsulated here — the generic
framework (runner, CLI, web shell) programmes against this interface.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from openot2.control.models import RunStep, TaskRun
from openot2.task_api.models import make_state

from src.tasks.color_mixing.config import ColorMixingConfig, load_task_config
from src.tasks.color_mixing.steps import (
    build_iteration_steps,
    build_control_steps,
    build_tip_check_steps,
    steps_to_runner_dicts,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Web extension
# ======================================================================

class ColorMixingWebExtension:
    """Task-specific web routes and status fields for color mixing.

    Satisfies the :class:`~openot2.task_api.plugin.TaskWebExtension`
    protocol.  Provides ``/gamut`` and ``/set-target`` routes plus
    calibration status fields.
    """

    def __init__(self, config: ColorMixingConfig) -> None:
        self._config = config
        self._loop: Any = None  # bound later by the integration layer

    def bind(self, loop_manager: Any) -> None:
        """Bind the extension to a loop manager for state access."""
        self._loop = loop_manager

    # -- TaskWebExtension protocol -----------------------------------------

    def extra_routes(self) -> list | None:
        """Return ``(method, path, handler)`` tuples for task routes.

        Each tuple is ``(HTTP_METHOD, path, async_handler)``.  The
        integration layer adds them to the protocol app via
        ``app.add_api_route``.
        """
        if self._loop is None:
            return None

        loop = self._loop

        async def get_gamut():
            cal = loop._task_state.get("artifacts", {}).get("calibration", {})
            gamut = cal.get("gamut") or {}
            return {
                "calibration_done": cal.get("done", False),
                "pure_rgbs": cal.get("pure_rgbs"),
                "water_rgb": cal.get("water_rgb"),
                "suggested_targets": gamut.get("suggested_targets"),
                "samples_rgb": gamut.get("samples_rgb", []),
                "custom_target": cal.get("custom_target"),
            }

        async def set_target(body: dict):
            rgb = body.get("rgb")
            if not rgb or len(rgb) != 3:
                return {"ok": False, "error": "Must provide rgb: [R, G, B]"}
            target = [float(rgb[0]), float(rgb[1]), float(rgb[2])]
            loop._task_state.setdefault("artifacts", {}).setdefault(
                "calibration", {},
            )["custom_target"] = target
            logger.info("Custom target set: %s", target)
            return {"ok": True, "target_rgb": target}

        return [
            ("GET", "/gamut", get_gamut),
            ("POST", "/set-target", set_target),
        ]

    def ui_payload(self, config: Any, state: dict) -> dict:
        cal = state.get("artifacts", {}).get("calibration", {})
        return {
            "calibration_done": cal.get("done", False),
            "pure_rgbs": cal.get("pure_rgbs"),
            "water_rgb": cal.get("water_rgb"),
            "custom_target": cal.get("custom_target"),
        }

    def extra_status(self, config: Any, state: dict) -> dict:
        cal = state.get("artifacts", {}).get("calibration", {})
        gamut = cal.get("gamut") or {}
        return {
            "calibration_done": cal.get("done", False),
            "pure_rgbs": cal.get("pure_rgbs"),
            "water_rgb": cal.get("water_rgb"),
            "gamut_suggested_targets": gamut.get("suggested_targets"),
            "custom_target": cal.get("custom_target"),
        }


# ======================================================================
# Plugin
# ======================================================================

class ColorMixingPlugin:
    """Task plugin for active learning color mixing optimisation."""

    name: str = "color_mixing"

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def load_config(self, path: str) -> ColorMixingConfig:
        return load_task_config(path)

    def build_deck_config(self, config: ColorMixingConfig) -> dict:
        return config.build_deck_dict()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def initial_state(self, config: ColorMixingConfig, mode: str) -> dict:
        n_initial = config.experiment.n_initial
        n_opt = config.experiment.n_optimization

        cache: dict[str, Any] = {
            "cached_reference": None,
            "all_X": [],
            "all_Y": [],
            "all_dist": [],
        }

        if mode == "quick":
            cache["rinse_cycles"] = 1
            cache["mix_cycles"] = 2
            cache["skip_controls_after_first"] = True
        else:
            cache["rinse_cycles"] = config.cleaning_protocol.rinse_cycles
            cache["mix_cycles"] = config.experiment.mix_cycles
            cache["skip_controls_after_first"] = False

        return make_state(
            phase="random",
            iteration=0,
            history=[],
            metrics={
                "best_distance": None,
                "best_iteration": None,
                "target_rgb": list(config.target_color),
                "distance_metric": config.ml.distance_metric,
                "convergence_threshold": config.convergence_threshold,
            },
            artifacts={},
            cache=cache,
            terminal=False,
            n_initial=n_initial,
            n_optimization=n_opt,
            mode=mode,
        )

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def build_plan(
        self,
        config: ColorMixingConfig,
        state: dict,
        mode: str,
        pre_calibrated: bool = False,
    ) -> dict:
        n_initial_configured = state.get("n_initial", config.experiment.n_initial)
        n_opt = state.get("n_optimization", config.experiment.n_optimization)
        # Pre-calibrated runs consume one random iteration for calibration.
        n_initial = max(n_initial_configured - 1, 0) if pre_calibrated else n_initial_configured
        return {
            "total_iterations": n_initial + n_opt,
            "n_initial": n_initial,
            "n_initial_configured": n_initial_configured,
            "n_optimization": n_opt,
            "max_per_plate": config.experiment.max_per_plate,
            "target_color": list(config.target_color),
            "convergence_threshold": config.convergence_threshold,
            "mode": mode,
            "pre_calibrated": pre_calibrated,
        }

    # ------------------------------------------------------------------
    # Run construction
    # ------------------------------------------------------------------

    def build_iteration_run(
        self,
        config: ColorMixingConfig,
        state: dict,
        iteration: int,
        mode: str,
    ) -> TaskRun:
        """Build a full TaskRun for one column iteration.

        Includes liquid handling, wait, capture, extract_rgb, and fit_gp
        steps with $ref bindings.
        """
        cache = state.get("cache", {})
        n_initial = state.get("n_initial", config.experiment.n_initial)
        phase = "random" if iteration < n_initial else "bo"
        col_idx = iteration
        well_col = col_idx + 1
        settle_time = config.experiment.settle_time_seconds
        skip_controls = (
            cache.get("skip_controls_after_first", False) and iteration > 0
        )
        mix_cycles_override = cache.get("mix_cycles")

        # Volumes come from state (set by suggest_volumes or random sampling)
        volumes = state.get("_next_volumes")
        if volumes is None:
            volumes = [0.0, 0.0, 0.0]

        # --- Liquid-handling steps (canonical → runner dicts) ---
        canonical = build_iteration_steps(
            config, np.array(volumes), col_idx,
            skip_controls=skip_controls,
            mix_cycles_override=mix_cycles_override,
        )
        lh_dicts = steps_to_runner_dicts(
            canonical, config,
            rinse_cycles=cache.get("rinse_cycles"),
        )

        # --- Orchestration steps with $ref bindings ---
        step_dicts: List[dict] = list(lh_dicts)

        step_dicts.append({
            "name": f"Settle {settle_time}s",
            "kind": "wait",
            "params": {"seconds": settle_time},
        })

        step_dicts.append({
            "name": f"Capture col{well_col}",
            "kind": "capture",
            "key": "capture",
            "params": {"label": f"col{well_col}_iter{iteration + 1}"},
        })

        grid_path = config.resolve_path(config.calibration.grid_path)
        step_dicts.append({
            "name": f"Extract RGB col{well_col}",
            "kind": "extract_rgb",
            "key": "extract_rgb",
            "params": {
                "image_path": {"$ref": "capture.output.image_path"},
                "col": col_idx,
                "grid_path": grid_path,
                "cached_reference": cache.get("cached_reference"),
                "skip_controls": skip_controls,
                "is_quick": mode == "quick",
            },
        })

        step_dicts.append({
            "name": f"Fit GP iter{iteration + 1}",
            "kind": "fit_gp",
            "key": "fit_gp",
            "params": {
                "volumes": list(volumes),
                "observed_rgb": {"$ref": "extract_rgb.output.mean_rgb"},
                "target_rgb": list(config.target_color),
                "distance_metric": config.ml.distance_metric,
                "convergence_threshold": config.convergence_threshold,
                "iteration": iteration,
                "all_X": list(cache.get("all_X", [])),
                "all_Y": list(cache.get("all_Y", [])),
                "all_dist": list(cache.get("all_dist", [])),
            },
        })

        run_steps = [RunStep(**d) for d in step_dicts]
        return TaskRun(
            name=f"Iteration {iteration + 1} [{phase}] col {well_col}",
            steps=run_steps,
            metadata={
                "iteration": iteration,
                "phase": phase,
                "col_idx": col_idx,
                "volumes": list(volumes),
                "skip_controls": skip_controls,
            },
        )

    def build_calibration_run(
        self, config: ColorMixingConfig, state: dict,
    ) -> TaskRun:
        """Build a TaskRun for calibration (controls in column 1 + capture)."""
        cache = state.get("cache", {})
        canonical = build_control_steps(config)
        lh_dicts = steps_to_runner_dicts(
            canonical, config,
            rinse_cycles=cache.get("rinse_cycles"),
        )

        step_dicts: List[dict] = list(lh_dicts)
        settle_time = config.experiment.settle_time_seconds

        step_dicts.append({
            "name": "Home for imaging",
            "kind": "home",
            "params": {},
        })
        step_dicts.append({
            "name": f"Settle {settle_time}s",
            "kind": "wait",
            "params": {"seconds": settle_time},
        })
        step_dicts.append({
            "name": "Capture calibration",
            "kind": "capture",
            "key": "capture",
            "params": {"label": "calibration_col1"},
        })

        grid_path = config.resolve_path(config.calibration.grid_path)
        step_dicts.append({
            "name": "Extract calibration RGB",
            "kind": "extract_rgb",
            "key": "extract_rgb",
            "params": {
                "image_path": {"$ref": "capture.output.image_path"},
                "col": 0,
                "grid_path": grid_path,
                "cached_reference": None,
                "skip_controls": False,
                "is_quick": False,
            },
        })

        run_steps = [RunStep(**d) for d in step_dicts]
        return TaskRun(
            name="Calibration run (column 1)",
            steps=run_steps,
            metadata={"phase": "calibration"},
        )

    def build_tip_check_run(
        self, config: ColorMixingConfig, state: dict,
    ) -> TaskRun:
        """Build a TaskRun for a tip-presence check."""
        canonical = build_tip_check_steps(config)
        step_dicts = steps_to_runner_dicts(canonical, config)

        run_steps = [RunStep(**d) for d in step_dicts]
        return TaskRun(
            name="Tip check",
            steps=run_steps,
            metadata={"phase": "tip_check"},
        )

    # ------------------------------------------------------------------
    # Result handling
    # ------------------------------------------------------------------

    def apply_run_result(
        self,
        config: ColorMixingConfig,
        state: dict,
        run: TaskRun,
        mode: str,
    ) -> dict:
        """Fold a completed run's results back into state."""
        new_state = dict(state)
        cache = dict(new_state.get("cache", {}))
        history = list(new_state.get("history", []))
        metrics = dict(new_state.get("metrics", {}))

        # Extract outputs from keyed steps
        rgb_output = None
        gp_output = None
        for step in run.steps:
            if step.key == "extract_rgb" and step.output:
                rgb_output = step.output
            if step.key == "fit_gp" and step.output:
                gp_output = step.output

        iteration = run.metadata.get("iteration", new_state.get("iteration", 0))
        volumes = run.metadata.get("volumes", [0, 0, 0])

        entry: dict[str, Any] = {
            "iteration": iteration,
            "phase": run.metadata.get("phase", "unknown"),
            "col_idx": run.metadata.get("col_idx", iteration),
            "volumes": volumes,
        }

        if rgb_output:
            entry["mean_rgb"] = rgb_output.get("mean_rgb")
            entry["used_calibration"] = rgb_output.get("used_calibration", False)
            if rgb_output.get("runtime_reference") is not None:
                cache["cached_reference"] = rgb_output["runtime_reference"]

        if gp_output:
            entry["distance"] = gp_output.get("distance")
            entry["converged"] = gp_output.get("converged", False)
            cache["all_X"] = gp_output.get("all_X", cache.get("all_X", []))
            cache["all_Y"] = gp_output.get("all_Y", cache.get("all_Y", []))
            cache["all_dist"] = gp_output.get("all_dist", cache.get("all_dist", []))

            dist = gp_output.get("distance")
            best = metrics.get("best_distance")
            if dist is not None and (best is None or dist < best):
                metrics["best_distance"] = dist
                metrics["best_iteration"] = iteration + 1

            if gp_output.get("converged", False):
                new_state["terminal"] = True
                new_state["phase"] = "done"

        history.append(entry)
        new_state["history"] = history
        new_state["metrics"] = metrics
        new_state["cache"] = cache
        new_state["iteration"] = iteration + 1

        # Phase transition
        n_initial = new_state.get("n_initial", config.experiment.n_initial)
        if not new_state.get("terminal"):
            if new_state["iteration"] >= n_initial:
                new_state["phase"] = "bo"
            n_total = n_initial + new_state.get("n_optimization",
                                                 config.experiment.n_optimization)
            if new_state["iteration"] >= n_total:
                new_state["terminal"] = True
                new_state["phase"] = "done"

        return new_state

    # ------------------------------------------------------------------
    # Calibration result processing
    # ------------------------------------------------------------------

    def apply_calibration_result(
        self,
        config: ColorMixingConfig,
        state: dict,
        run: TaskRun,
        config_path: str,
    ) -> tuple[dict, dict]:
        """Process a completed calibration run.

        Extracts per-well RGBs, computes the reachable gamut, persists
        calibration state to disk, and returns ``(new_state, payload)``.

        *new_state* has calibration artifacts in
        ``state["artifacts"]["calibration"]``.
        *payload* is the dict to return to the API caller.
        """
        # Find step outputs
        rgb_output = None
        capture_output = None
        for step in run.steps:
            if getattr(step, "kind", None) == "extract_rgb" and getattr(step, "output", None):
                rgb_output = step.output
            if getattr(step, "kind", None) == "capture" and getattr(step, "output", None):
                capture_output = step.output

        if not rgb_output or rgb_output.get("error"):
            err = (rgb_output or {}).get("error", "No output")
            return state, {"ok": False, "error": f"RGB extraction failed: {err}"}

        image_path = None
        if capture_output:
            image_path = capture_output.get("image_path") or capture_output.get("path")
        if not image_path:
            return state, {"ok": False, "error": "No image path in capture output"}

        # Extract individual control well RGBs for gamut computation
        try:
            import cv2
            from src.vision.extraction import extract_well_rgb
            from src.vision.geometry import load_grid_calibration

            img = cv2.imread(image_path)
            grid_path = config.resolve_path(config.calibration.grid_path)
            grid = None
            if grid_path:
                try:
                    grid = load_grid_calibration(grid_path)
                except Exception:
                    pass

            row_to_idx = {chr(ord("A") + i): i for i in range(8)}
            pure_rgbs: dict[str, list] = {}
            water_rgb_val: list | None = None
            for row, role in config.well_roles.controls.items():
                row_idx = row_to_idx[row]
                rgb = extract_well_rgb(img, row_idx, 0, grid).tolist()
                if role == "water":
                    water_rgb_val = rgb
                else:
                    pure_rgbs[role] = rgb
        except Exception as exc:
            return state, {"ok": False, "error": f"Control well extraction failed: {exc}"}

        # Compute gamut
        from src.color.metrics import compute_reachable_gamut

        gamut = compute_reachable_gamut(pure_rgbs, np.array(water_rgb_val))

        # Cache the runtime reference
        cached_reference = rgb_output.get("runtime_reference")

        # Build calibration artifacts
        prev_cal = state.get("artifacts", {}).get("calibration", {})
        cal_artifacts: dict[str, Any] = {
            "done": True,
            "pure_rgbs": pure_rgbs,
            "water_rgb": water_rgb_val,
            "gamut": gamut,
            "cached_reference": cached_reference,
            "custom_target": prev_cal.get("custom_target"),
        }

        # Update state
        new_state = dict(state)
        new_state.setdefault("artifacts", {})["calibration"] = cal_artifacts
        if cached_reference:
            new_state.setdefault("cache", {})["cached_reference"] = cached_reference

        # Persist to disk
        try:
            self._save_calibration(config_path, cal_artifacts)
            logger.info("Saved color calibration")
        except Exception as exc:
            logger.warning("Could not save color calibration: %s", exc)

        logger.info(
            "Calibration complete:\n  Pure RGBs: %s\n  Water: %s",
            pure_rgbs, water_rgb_val,
        )

        return new_state, {
            "ok": True,
            "pure_rgbs": pure_rgbs,
            "water_rgb": water_rgb_val,
            "gamut": gamut,
        }

    def load_calibration_state(self, config_path: str) -> dict | None:
        """Load persisted calibration artifacts from disk.

        Returns a calibration artifacts dict suitable for storing in
        ``state["artifacts"]["calibration"]``, or *None* if no persisted
        calibration is found.
        """
        cal_path = (
            Path(config_path).resolve().parent
            / "calibrations" / "color_calibration" / "color_profile.json"
        )
        if not cal_path.exists():
            return None

        try:
            payload = json.loads(cal_path.read_text(encoding="utf-8"))
            pure_rgbs = payload.get("pure_rgbs")
            water_rgb = payload.get("water_rgb")
            cached_reference = payload.get("cached_reference")
            gamut = payload.get("gamut")

            if pure_rgbs is None or water_rgb is None or cached_reference is None:
                logger.warning(
                    "Ignoring incomplete color calibration at %s", cal_path,
                )
                return None

            if gamut is None:
                from src.color.metrics import compute_reachable_gamut

                gamut = compute_reachable_gamut(pure_rgbs, np.array(water_rgb))

            logger.info("Loaded color calibration from %s", cal_path)
            return {
                "done": True,
                "pure_rgbs": pure_rgbs,
                "water_rgb": water_rgb,
                "gamut": gamut,
                "cached_reference": cached_reference,
                "custom_target": None,
            }
        except Exception as exc:
            logger.warning(
                "Could not load color calibration from %s: %s", cal_path, exc,
            )
            return None

    def _save_calibration(self, config_path: str, cal: dict) -> None:
        """Persist calibration artifacts to disk."""
        cal_dir = (
            Path(config_path).resolve().parent
            / "calibrations" / "color_calibration"
        )
        cal_dir.mkdir(parents=True, exist_ok=True)
        path = cal_dir / "color_profile.json"
        payload = {
            "version": 1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "pure_rgbs": cal["pure_rgbs"],
            "water_rgb": cal["water_rgb"],
            "gamut": cal["gamut"],
            "cached_reference": cal["cached_reference"],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Calibration targets
    # ------------------------------------------------------------------

    def build_calibration_targets(self, config: ColorMixingConfig) -> list:
        return config.build_calibration_targets()

    # ------------------------------------------------------------------
    # Status & web
    # ------------------------------------------------------------------

    def status_payload(self, config: ColorMixingConfig, state: dict) -> dict:
        metrics = state.get("metrics", {})
        history = state.get("history", [])
        return {
            "task": self.name,
            "phase": state.get("phase", "idle"),
            "iteration": state.get("iteration", 0),
            "terminal": state.get("terminal", False),
            "target_color": list(config.target_color),
            "best_distance": metrics.get("best_distance"),
            "best_iteration": metrics.get("best_iteration"),
            "total_experiments": len(history),
            "convergence_threshold": config.convergence_threshold,
        }

    def web_extension(self, config: ColorMixingConfig) -> ColorMixingWebExtension:
        """Return the color mixing web extension."""
        return ColorMixingWebExtension(config)
