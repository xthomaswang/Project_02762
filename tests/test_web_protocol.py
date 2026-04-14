"""Tests for the web protocol integration (src/web/).

Covers: plugin-driven loop, plugin-driven server assembly, plugin-driven
CLI entry, status schema, state machine (pause/stop/fail), analysis
output, FastAPI routes, $ref-based step wiring, plugin-owned calibration,
task extension routes, and no task-specific protocol semantics in src/web.
"""

import csv
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from openot2.control.models import TaskRun, RunStep, RunSequence, RunStatus, StepStatus
from openot2.control.store import JsonRunStore
from openot2.control.runner import TaskRunner

from src.tasks.color_mixing.plugin import ColorMixingPlugin, ColorMixingWebExtension
from src.web.loop import ActiveLearningLoop
from src.web.handlers import handle_extract_rgb, handle_suggest_volumes, handle_fit_gp

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "color_mixing.yaml")


def _make_plugin():
    return ColorMixingPlugin()


def _load_config():
    plugin = _make_plugin()
    return plugin.load_config(CONFIG_PATH)


def _make_loop(store=None, register_dummy_handlers=False):
    """Create loop with temp store and plugin. Optionally register dummy handlers."""
    if store is None:
        tmpdir = tempfile.mkdtemp()
        store = JsonRunStore(base_dir=tmpdir)
    runner = TaskRunner(store=store)
    plugin = _make_plugin()
    config = plugin.load_config(CONFIG_PATH)

    if register_dummy_handlers:
        for kind in ("use_pipette", "transfer", "mix", "home", "wait"):
            runner.register_handler(kind, lambda step, ctx=None: {"ok": True})
        # Capture must return image_path for $ref resolution
        runner.register_handler("capture", lambda step, ctx=None: {
            "ok": True, "image_path": "/tmp/fake_capture.jpg",
        })
        # extract_rgb handler returns mock RGB results
        runner.register_handler("extract_rgb", lambda step, ctx=None: {
            "mean_rgb": [100, 50, 150], "std_rgb": [5, 5, 5],
            "raw_mean_rgb": [100, 50, 150], "used_calibration": False,
            "runtime_reference": None,
        })
        # fit_gp handler returns distance/convergence
        def _dummy_fit_gp(step, ctx=None):
            params = step.params if hasattr(step, "params") else step
            all_X = list(params.get("all_X", []))
            all_Y = list(params.get("all_Y", []))
            all_dist = list(params.get("all_dist", []))
            volumes = params.get("volumes", [0, 0, 0])
            observed_rgb = params.get("observed_rgb", [100, 50, 150])
            all_X.append(list(volumes))
            all_Y.append(list(observed_rgb))
            all_dist.append(42.0)
            return {
                "distance": 42.0, "converged": False,
                "best_distance": 42.0, "best_iteration": len(all_X),
                "all_X": all_X, "all_Y": all_Y, "all_dist": all_dist,
            }
        runner.register_handler("fit_gp", _dummy_fit_gp)

    return ActiveLearningLoop(
        runner=runner,
        store=store,
        plugin=plugin,
        config=config,
        config_path=CONFIG_PATH,
    )


def _inject_calibration(loop, pure_rgbs=None, water_rgb=None, gamut=None,
                         cached_reference=None, custom_target=None):
    """Inject calibration artifacts into loop task state."""
    cal = {
        "done": True,
        "pure_rgbs": pure_rgbs or {"red": [1, 0, 0], "green": [0, 1, 0], "blue": [0, 0, 1]},
        "water_rgb": water_rgb or [0.5, 0.5, 0.5],
        "gamut": gamut or {"samples_rgb": [[1, 0, 0]]},
        "cached_reference": cached_reference or {"source_column": 1},
        "custom_target": custom_target,
    }
    loop._task_state.setdefault("artifacts", {})["calibration"] = cal


# ======================================================================
# Plugin-driven loop
# ======================================================================

class TestPluginDrivenLoop(unittest.TestCase):
    """Verify that the loop delegates to the plugin."""

    def test_loop_has_plugin(self):
        loop = _make_loop()
        self.assertIsNotNone(loop.plugin)
        self.assertEqual(loop.plugin.name, "color_mixing")

    def test_loop_config_loaded_via_plugin(self):
        loop = _make_loop()
        self.assertIsNotNone(loop._config)
        self.assertTrue(hasattr(loop._config, "target_color"))
        self.assertTrue(hasattr(loop._config, "total_volume_ul"))

    def test_initial_state_from_plugin(self):
        loop = _make_loop()
        with mock.patch.object(threading.Thread, "start", return_value=None):
            loop.start(mode="quick")
        self.assertIn("phase", loop._task_state)
        self.assertIn("iteration", loop._task_state)
        self.assertIn("cache", loop._task_state)
        self.assertIn("history", loop._task_state)
        self.assertEqual(loop._task_state["phase"], "random")
        self.assertEqual(loop._task_state["iteration"], 0)

    def test_plan_from_plugin(self):
        loop = _make_loop()
        plan = loop.get_plan("quick")
        self.assertIn("total_iterations", plan)
        self.assertIn("n_initial", plan)
        self.assertIn("n_optimization", plan)
        self.assertIn("iterations", plan)
        self.assertEqual(plan["mode"], "quick")

    def test_plan_enriched_with_iterations(self):
        loop = _make_loop()
        plan = loop.get_plan("quick")
        self.assertTrue(len(plan["iterations"]) > 0)
        first = plan["iterations"][0]
        self.assertIn("iteration", first)
        self.assertIn("phase", first)
        self.assertIn("column", first)
        self.assertIn("status", first)

    def test_build_iteration_run_via_plugin(self):
        """Plugin builds iteration runs, not the loop."""
        loop = _make_loop()
        config = loop._config
        state = loop.plugin.initial_state(config, "quick")
        state["_next_volumes"] = [100, 50, 50]
        run = loop.plugin.build_iteration_run(config, state, 0, "quick")
        self.assertIsInstance(run, TaskRun)
        kinds = [s.kind for s in run.steps]
        self.assertIn("extract_rgb", kinds)
        self.assertIn("fit_gp", kinds)

    def test_build_calibration_run_via_plugin(self):
        """Plugin builds calibration runs, not the loop."""
        loop = _make_loop()
        config = loop._config
        state = loop.plugin.initial_state(config, "quick")
        run = loop.plugin.build_calibration_run(config, state)
        self.assertIsInstance(run, TaskRun)
        self.assertIn("calibration", run.name.lower())

    def test_build_tip_check_run_via_plugin(self):
        """Plugin builds tip check runs, not the loop."""
        loop = _make_loop()
        config = loop._config
        state = loop.plugin.initial_state(config, "quick")
        run = loop.plugin.build_tip_check_run(config, state)
        self.assertIsInstance(run, TaskRun)

    def test_apply_run_result_via_plugin(self):
        """Plugin folds run results into state."""
        loop = _make_loop()
        config = loop._config
        state = loop.plugin.initial_state(config, "quick")
        state["_next_volumes"] = [100, 50, 50]

        # Build and simulate a completed run
        run = loop.plugin.build_iteration_run(config, state, 0, "quick")
        for step in run.steps:
            step.status = StepStatus.succeeded
            if step.key == "extract_rgb":
                step.output = {
                    "mean_rgb": [100, 50, 150],
                    "used_calibration": False,
                    "runtime_reference": None,
                }
            elif step.key == "fit_gp":
                step.output = {
                    "distance": 42.0, "converged": False,
                    "best_distance": 42.0, "best_iteration": 1,
                    "all_X": [[100, 50, 50]], "all_Y": [[100, 50, 150]],
                    "all_dist": [42.0],
                }
        run.status = RunStatus.completed

        new_state = loop.plugin.apply_run_result(config, state, run, "quick")
        self.assertEqual(new_state["iteration"], 1)
        self.assertEqual(len(new_state["history"]), 1)
        self.assertIn("all_X", new_state["cache"])


# ======================================================================
# Status schema
# ======================================================================

class TestStatusSchema(unittest.TestCase):

    def test_initial_status_has_all_fields(self):
        loop = _make_loop()
        s = loop.status()
        required = [
            "state", "sequence_id", "iteration", "max_iterations",
            "n_initial", "n_optimization", "convergence_threshold",
            "target_rgb", "best_distance", "best_iteration", "best_rgb",
            "converged", "error", "history", "plan", "mode", "distance_metric",
            "active_run",
            # Calibration fields from extension extra_status
            "calibration_done", "pure_rgbs", "custom_target",
        ]
        for key in required:
            self.assertIn(key, s, f"Missing status key: {key}")
        self.assertEqual(s["state"], "idle")
        self.assertIsNone(s["error"])
        self.assertEqual(s["history"], [])
        self.assertIsNotNone(s["plan"])
        self.assertIsNone(s["active_run"])

    def test_status_exposes_active_run_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            loop = _make_loop(store=store)
            run = TaskRun(
                name="iter1",
                status=RunStatus.running,
                current_step_index=1,
                metadata={"iteration": 0, "phase": "random", "col_idx": 0, "volumes": [67.5, 65.6, 66.9]},
                steps=[
                    RunStep(name="Select left pipette", kind="use_pipette", status=StepStatus.succeeded),
                    RunStep(
                        name="Control: Red -> A1",
                        kind="transfer",
                        status=StepStatus.running,
                        params={"source_well": "A1", "dest_well": "A1", "volume": 200},
                    ),
                ],
            )
            run = store.create_run(run)
            loop._active_run_id = run.id
            s = loop.status()
            self.assertIsNotNone(s["active_run"])
            self.assertEqual(s["active_run"]["status"], "running")
            self.assertEqual(s["active_run"]["iteration"], 1)
            self.assertEqual(s["active_run"]["phase"], "random")
            self.assertEqual(s["active_run"]["column"], 1)
            self.assertEqual(s["active_run"]["current_step_number"], 2)
            self.assertEqual(s["active_run"]["total_steps"], 2)
            self.assertEqual(s["active_run"]["current_step"]["name"], "Control: Red -> A1")

    def test_status_exposes_live_substep_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            loop = _make_loop(store=store)
            run = TaskRun(
                name="iter1",
                status=RunStatus.running,
                current_step_index=0,
                metadata={"iteration": 0, "phase": "random", "col_idx": 0},
                steps=[
                    RunStep(
                        name="Exp: Red 68uL -> col1",
                        kind="transfer",
                        status=StepStatus.running,
                        params={"source_well": "A1", "dest_well": "A1", "volume": 67.5},
                    ),
                ],
            )
            run = store.create_run(run)
            loop._active_run_id = run.id
            loop.update_live_progress({
                "run_id": run.id,
                "step_id": run.steps[0].id,
                "step_name": run.steps[0].name,
                "step_kind": "transfer",
                "action": "aspirate",
                "detail": "A1 (67.5uL)",
            })
            s = loop.status()
            self.assertEqual(s["active_run"]["live_progress"]["action"], "aspirate")
            self.assertEqual(s["active_run"]["live_progress"]["detail"], "A1 (67.5uL)")


# ======================================================================
# $ref wiring: extract_rgb and fit_gp are real steps
# ======================================================================

class TestRefWiring(unittest.TestCase):
    """Verify that plugin-built runs use $ref bindings."""

    def test_iteration_run_has_ref_bindings(self):
        plugin = _make_plugin()
        config = plugin.load_config(CONFIG_PATH)
        state = plugin.initial_state(config, "quick")
        state["_next_volumes"] = [100, 50, 50]
        run = plugin.build_iteration_run(config, state, 0, "quick")

        capture = next(s for s in run.steps if s.kind == "capture")
        rgb = next(s for s in run.steps if s.kind == "extract_rgb")
        gp = next(s for s in run.steps if s.kind == "fit_gp")

        self.assertEqual(capture.key, "capture")
        self.assertEqual(rgb.key, "extract_rgb")
        self.assertEqual(gp.key, "fit_gp")
        self.assertIsInstance(rgb.params["image_path"], dict)
        self.assertEqual(rgb.params["image_path"]["$ref"], "capture.output.image_path")
        self.assertIsInstance(gp.params["observed_rgb"], dict)
        self.assertEqual(gp.params["observed_rgb"]["$ref"], "extract_rgb.output.mean_rgb")

    def test_ref_resolution_in_runner(self):
        """The runner should resolve $ref bindings during execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            runner = TaskRunner(store=store)

            runner.register_handler("capture", lambda step, ctx=None: {
                "image_path": "/tmp/test.jpg",
            })

            resolved_params = {}
            def capture_rgb_handler(step, ctx=None):
                resolved_params["image_path"] = step.params.get("image_path")
                return {"mean_rgb": [100, 50, 150]}
            runner.register_handler("extract_rgb", capture_rgb_handler)

            resolved_rgb = {}
            def capture_gp_handler(step, ctx=None):
                resolved_rgb["observed_rgb"] = step.params.get("observed_rgb")
                return {"distance": 10.0, "converged": False}
            runner.register_handler("fit_gp", capture_gp_handler)

            steps = [
                RunStep(name="Capture", kind="capture", key="capture", params={"label": "test"}),
                RunStep(name="Extract RGB", kind="extract_rgb", key="extract_rgb", params={
                    "image_path": {"$ref": "capture.output.image_path"}, "col": 0,
                }),
                RunStep(name="Fit GP", kind="fit_gp", key="fit_gp", params={
                    "observed_rgb": {"$ref": "extract_rgb.output.mean_rgb"},
                    "volumes": [100, 50, 50], "target_rgb": [128, 0, 255],
                }),
            ]
            run = TaskRun(name="test_ref", steps=steps)
            run = store.create_run(run)
            runner.run_until_pause_or_done(run.id)

            self.assertEqual(resolved_params["image_path"], "/tmp/test.jpg")
            self.assertEqual(resolved_rgb["observed_rgb"], [100, 50, 150])


# ======================================================================
# Handler contracts
# ======================================================================

class TestHandlerContracts(unittest.TestCase):

    def test_suggest_volumes_random(self):
        class P:
            def __init__(self, p): self.params = p
        result = handle_suggest_volumes(P({
            "target_rgb": [128, 0, 255], "total_volume": 200,
            "phase": "random", "seed": 42, "iteration": 0,
            "all_X": [], "all_Y": [], "distance_metric": "delta_e_lab",
            "acquisition": "EI", "model_type": "correlated_gp",
            "bounds_min": [0, 0, 0], "bounds_max": [200, 200, 200],
        }))
        self.assertEqual(len(result["volumes"]), 3)
        self.assertAlmostEqual(sum(result["volumes"]), 200, places=1)

    def test_fit_gp_distance_and_convergence(self):
        class P:
            def __init__(self, p): self.params = p
        result = handle_fit_gp(P({
            "volumes": [100, 50, 50], "observed_rgb": [128, 0, 255],
            "target_rgb": [128, 0, 255], "distance_metric": "delta_e_lab",
            "convergence_threshold": 50, "iteration": 0,
            "all_X": [], "all_Y": [], "all_dist": [],
        }))
        self.assertAlmostEqual(result["distance"], 0.0, places=1)
        self.assertTrue(result["converged"])
        self.assertEqual(len(result["all_X"]), 1)

    def test_extract_rgb_deferred_without_image_path(self):
        class P:
            def __init__(self, p): self.params = p
        result = handle_extract_rgb(P({
            "image_path": "", "col": 0, "grid_path": "",
            "config_path": "", "cached_reference": None, "skip_controls": False,
        }))
        self.assertTrue(result["deferred"])

    def test_fit_gp_deferred_without_observed_rgb(self):
        class P:
            def __init__(self, p): self.params = p
        result = handle_fit_gp(P({
            "volumes": [100, 50, 50], "observed_rgb": [],
            "target_rgb": [128, 0, 255],
            "all_X": [], "all_Y": [], "all_dist": [],
        }))
        self.assertTrue(result["deferred"])


# ======================================================================
# State machine
# ======================================================================

class TestStateMachine(unittest.TestCase):

    def test_fatal_error_sets_failed(self):
        loop = _make_loop()
        loop.start(mode="quick")
        loop._thread.join(timeout=15)
        s = loop.status()
        self.assertEqual(s["state"], "failed")
        self.assertIsNotNone(s["error"])

    def test_stop_prevents_new_iterations(self):
        loop = _make_loop(register_dummy_handlers=True)
        loop.start(mode="quick")
        time.sleep(0.3)
        loop.stop()
        loop._thread.join(timeout=15)
        s = loop.status()
        self.assertIn(s["state"], ("stopped", "failed", "completed", "converged"))

    def test_pause_state_is_pausing_not_paused(self):
        loop = _make_loop()
        loop._state = "running"
        loop._active_run_id = "fake-id"
        loop.pause()
        self.assertEqual(loop._state, "pausing")

    def test_resume_only_from_paused(self):
        loop = _make_loop()
        loop._state = "idle"
        loop.resume()
        self.assertEqual(loop._state, "idle")
        loop._state = "paused"
        loop.resume()
        self.assertEqual(loop._state, "running")

    def test_reset_clears_state(self):
        loop = _make_loop()
        loop._state = "completed"
        loop._task_state = {"history": [{"test": True}], "cache": {"all_dist": [1.0, 2.0]}}
        loop._error = "old error"
        result = loop.reset()
        self.assertTrue(result["ok"])
        s = loop.status()
        self.assertEqual(s["state"], "idle")
        self.assertEqual(s["history"], [])
        self.assertIsNone(s["error"])

    def test_reset_blocked_while_running(self):
        loop = _make_loop()
        loop._state = "running"
        result = loop.reset()
        self.assertFalse(result["ok"])

    def test_reset_blocked_while_paused(self):
        loop = _make_loop()
        loop._state = "paused"
        loop._active_run_id = "run-1"
        result = loop.reset()
        self.assertFalse(result["ok"])

    def test_home_blocked_while_running(self):
        loop = _make_loop()
        loop._state = "running"
        result = loop.home()
        self.assertFalse(result["ok"])

    def test_home_blocked_while_paused(self):
        loop = _make_loop()
        loop._state = "paused"
        loop._active_run_id = "run-1"
        result = loop.home()
        self.assertFalse(result["ok"])

    def test_home_reports_failed_manual_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            plugin = _make_plugin()
            config = plugin.load_config(CONFIG_PATH)
            runner = TaskRunner(store=store)
            loop = ActiveLearningLoop(
                runner=runner, store=store, plugin=plugin,
                config=config, config_path=CONFIG_PATH,
            )
            result = loop.home()
            self.assertFalse(result["ok"])
            self.assertIn("error", result)


# ======================================================================
# Sequence lifecycle
# ======================================================================

class TestSequenceLifecycle(unittest.TestCase):

    def test_sequence_created_on_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            loop = _make_loop(store=store)
            loop.start(mode="quick")
            loop._thread.join(timeout=15)
            seqs = store.list_sequences()
            self.assertGreaterEqual(len(seqs), 1)

    def test_sequence_status_updated_on_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            loop = _make_loop(store=store)
            loop.start(mode="quick")
            loop._thread.join(timeout=15)
            self.assertEqual(loop.status()["state"], "failed")
            if loop._sequence_id:
                seq = store.load_sequence(loop._sequence_id)
                self.assertEqual(seq.status, "failed")


# ======================================================================
# Plan item tracking
# ======================================================================

class TestPlanTracking(unittest.TestCase):

    def test_failed_plan_items(self):
        loop = _make_loop()
        loop.start(mode="quick")
        loop._thread.join(timeout=15)
        plan = loop.status()["plan"]
        statuses = [it["status"] for it in plan["iterations"]]
        self.assertIn("failed", statuses)


# ======================================================================
# Analysis output
# ======================================================================

class TestAnalysisOutput(unittest.TestCase):

    def test_analysis_saved_on_terminal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            loop = _make_loop(store=store, register_dummy_handlers=True)
            loop.start(mode="quick")
            loop._thread.join(timeout=30)
            s = loop.status()
            seq_id = s["sequence_id"]
            if seq_id and s["history"]:
                analysis_dir = os.path.join(tmpdir, "analysis", seq_id)
                self.assertTrue(os.path.isdir(analysis_dir))


# ======================================================================
# FastAPI routes
# ======================================================================

class TestFastAPIRoutes(unittest.TestCase):

    def _make_client(self):
        from starlette.testclient import TestClient
        from src.web.app import create_protocol_app
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            runner = TaskRunner(store=store)
            plugin = _make_plugin()
            config = plugin.load_config(CONFIG_PATH)
            loop = ActiveLearningLoop(
                runner=runner, store=store, plugin=plugin,
                config=config, config_path=CONFIG_PATH,
            )
            app = create_protocol_app(loop)
            return TestClient(app), loop

    def test_status_route(self):
        client, _ = self._make_client()
        r = client.get("/status")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["state"], "idle")

    def test_plan_route(self):
        client, _ = self._make_client()
        r = client.get("/plan?mode=quick")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["mode"], "quick")

    def test_home_route(self):
        client, _ = self._make_client()
        r = client.post("/home")
        self.assertEqual(r.status_code, 409)

    def test_reset_route(self):
        client, _ = self._make_client()
        r = client.post("/reset")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

    def test_reset_route_rejects_active_protocol(self):
        client, loop = self._make_client()
        loop._state = "paused"
        loop._active_run_id = "run-1"
        r = client.post("/reset")
        self.assertEqual(r.status_code, 409)

    def test_home_route_rejects_active_protocol(self):
        client, loop = self._make_client()
        loop._state = "paused"
        loop._active_run_id = "run-1"
        r = client.post("/home")
        self.assertEqual(r.status_code, 409)

    def test_protocol_page_renders(self):
        client, _ = self._make_client()
        r = client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("html", r.text.lower())


# ======================================================================
# Calibration flow
# ======================================================================

class TestCalibrationFlow(unittest.TestCase):

    def test_calibrate_blocked_while_running(self):
        loop = _make_loop()
        loop._state = "running"
        result = loop.calibrate()
        self.assertFalse(result["ok"])

    def test_calibration_status_fields(self):
        loop = _make_loop()
        s = loop.status()
        self.assertIn("calibration_done", s)
        self.assertIn("pure_rgbs", s)
        self.assertIn("custom_target", s)
        self.assertFalse(s["calibration_done"])

    def test_start_column_after_calibration(self):
        """After calibration, quick plan should start at column 2."""
        loop = _make_loop()
        _inject_calibration(loop)
        plan = loop.get_plan("quick")
        self.assertEqual(plan["iterations"][0]["column"], 2)
        self.assertEqual(plan["n_initial"], 4)
        self.assertTrue(plan["iterations"][0]["skip_controls"])

    def test_start_preserves_calibration_state(self):
        loop = _make_loop()
        _inject_calibration(
            loop,
            pure_rgbs={"red": [1.0, 0.0, 0.0], "green": [0.0, 1.0, 0.0], "blue": [0.0, 0.0, 1.0]},
            water_rgb=[0.5, 0.5, 0.5],
            gamut={"samples_rgb": [[1.0, 0.0, 0.0]]},
            cached_reference={"source_column": 1},
            custom_target=[100.0, 50.0, 200.0],
        )

        with mock.patch.object(threading.Thread, "start", return_value=None):
            loop.start(mode="quick")

        cal = loop._cal()
        self.assertTrue(cal.get("done"))
        self.assertEqual(cal["pure_rgbs"]["red"], [1.0, 0.0, 0.0])
        self.assertEqual(cal["water_rgb"], [0.5, 0.5, 0.5])
        self.assertEqual(cal["custom_target"], [100.0, 50.0, 200.0])
        self.assertEqual(cal["cached_reference"], {"source_column": 1})
        self.assertEqual(loop._plan["iterations"][0]["column"], 2)
        self.assertTrue(loop._plan["iterations"][0]["skip_controls"])
        self.assertEqual(loop._plan["n_initial"], 4)

    def test_calibration_delegates_post_processing_to_plugin(self):
        """Calibration must delegate post-processing to plugin.apply_calibration_result."""
        loop = _make_loop(register_dummy_handlers=True)

        with mock.patch.object(
            loop.plugin, "apply_calibration_result",
            wraps=loop.plugin.apply_calibration_result,
        ) as mock_apply:
            with mock.patch(
                "cv2.imread", return_value=np.zeros((8, 8, 3), dtype=np.uint8)
            ), mock.patch(
                "src.vision.geometry.load_grid_calibration", return_value=object()
            ), mock.patch(
                "src.vision.extraction.extract_well_rgb",
                side_effect=[
                    np.array([255.0, 0.0, 0.0]),
                    np.array([0.0, 255.0, 0.0]),
                    np.array([0.0, 0.0, 255.0]),
                    np.array([240.0, 240.0, 240.0]),
                ],
            ), mock.patch(
                "src.color.metrics.compute_reachable_gamut",
                return_value={"samples_rgb": []},
            ):
                result = loop.calibrate()

            self.assertTrue(result["ok"])
            mock_apply.assert_called_once()
            # Calibration artifacts should be in task state, not in loop fields
            cal = loop._cal()
            self.assertTrue(cal.get("done"))
            self.assertIn("pure_rgbs", cal)
            self.assertIn("water_rgb", cal)
            self.assertIn("gamut", cal)

    def test_loop_does_not_compute_gamut_itself(self):
        """loop.py must not import compute_reachable_gamut or extract_well_rgb."""
        import inspect
        from src.web import loop as loop_module
        source = inspect.getsource(loop_module)
        self.assertNotIn("compute_reachable_gamut", source)
        self.assertNotIn("extract_well_rgb", source)
        self.assertNotIn("import cv2", source)

    def test_loop_autoloads_saved_color_calibration(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with open(CONFIG_PATH) as f:
                raw = yaml.safe_load(f)
            config_path = tmpdir / "experiment.yaml"
            config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

            cal_dir = tmpdir / "calibrations" / "color_calibration"
            cal_dir.mkdir(parents=True)
            payload = {
                "version": 1,
                "pure_rgbs": {
                    "red": [255.0, 0.0, 0.0],
                    "green": [0.0, 255.0, 0.0],
                    "blue": [0.0, 0.0, 255.0],
                },
                "water_rgb": [240.0, 240.0, 240.0],
                "gamut": {"samples_rgb": [[255.0, 0.0, 0.0]]},
                "cached_reference": {"source_column": 1},
            }
            (cal_dir / "color_profile.json").write_text(json.dumps(payload), encoding="utf-8")

            store = JsonRunStore(base_dir=str(tmpdir / "run_data"))
            plugin = _make_plugin()
            config = plugin.load_config(str(config_path))
            runner = TaskRunner(store=store)
            loop = ActiveLearningLoop(
                runner=runner, store=store, plugin=plugin,
                config=config, config_path=str(config_path),
            )

            cal = loop._cal()
            self.assertTrue(cal.get("done"))
            self.assertEqual(cal["pure_rgbs"]["red"], [255.0, 0.0, 0.0])
            self.assertEqual(cal["water_rgb"], [240.0, 240.0, 240.0])
            self.assertEqual(cal["cached_reference"], {"source_column": 1})

    def test_robot_calibration_profile_route(self):
        from starlette.testclient import TestClient
        from src.web.app import create_protocol_app
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            runner = TaskRunner(store=store)
            plugin = _make_plugin()
            config = plugin.load_config(CONFIG_PATH)
            loop = ActiveLearningLoop(
                runner=runner, store=store, plugin=plugin,
                config=config, config_path=CONFIG_PATH,
            )
            app = create_protocol_app(loop)
            client = TestClient(app)
            r = client.get("/robot-calibration-profile")
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertIn("targets", data)

    def test_get_plan_uses_calibrated_start_column(self):
        loop = _make_loop()
        _inject_calibration(loop)
        plan = loop.get_plan("quick")
        self.assertEqual(plan["iterations"][0]["column"], 2)
        self.assertTrue(plan["iterations"][0]["skip_controls"])
        self.assertEqual(plan["n_initial"], 4)


# ======================================================================
# Task web extension routes (gamut, set-target)
# ======================================================================

class TestTaskExtensionRoutes(unittest.TestCase):
    """Verify /gamut and /set-target are served through the task extension."""

    def _make_client(self):
        from starlette.testclient import TestClient
        from src.web.app import create_protocol_app
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            runner = TaskRunner(store=store)
            plugin = _make_plugin()
            config = plugin.load_config(CONFIG_PATH)
            loop = ActiveLearningLoop(
                runner=runner, store=store, plugin=plugin,
                config=config, config_path=CONFIG_PATH,
            )
            app = create_protocol_app(loop)
            return TestClient(app), loop

    def test_gamut_route_served_via_extension(self):
        client, _ = self._make_client()
        r = client.get("/gamut")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["calibration_done"])

    def test_set_target_route_served_via_extension(self):
        client, loop = self._make_client()
        r = client.post("/set-target", json={"rgb": [100, 50, 200]})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])
        # Target should be stored in task state artifacts
        cal = loop._cal()
        self.assertEqual(cal["custom_target"], [100.0, 50.0, 200.0])

    def test_set_target_invalid(self):
        client, _ = self._make_client()
        r = client.post("/set-target", json={"rgb": [100, 50]})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["ok"])

    def test_gamut_reflects_calibration_artifacts(self):
        client, loop = self._make_client()
        _inject_calibration(
            loop,
            pure_rgbs={"red": [255, 0, 0], "green": [0, 255, 0], "blue": [0, 0, 255]},
            water_rgb=[240, 240, 240],
            gamut={"samples_rgb": [[128, 0, 128]], "suggested_targets": [[100, 50, 200]]},
        )
        r = client.get("/gamut")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["calibration_done"])
        self.assertEqual(data["pure_rgbs"]["red"], [255, 0, 0])
        self.assertEqual(data["water_rgb"], [240, 240, 240])
        self.assertEqual(data["suggested_targets"], [[100, 50, 200]])

    def test_app_source_has_no_gamut_or_set_target(self):
        """create_protocol_app must not define /gamut or /set-target itself."""
        import inspect
        from src.web.app import create_protocol_app
        source = inspect.getsource(create_protocol_app)
        # These should not appear as route definitions in the source
        self.assertNotIn("def get_gamut", source)
        self.assertNotIn("def set_target", source)
        self.assertNotIn("_gamut", source)
        self.assertNotIn("_pure_rgbs", source)
        self.assertNotIn("_water_rgb", source)
        self.assertNotIn("_custom_target", source)

    def test_plugin_returns_web_extension(self):
        """Plugin.web_extension() should return a ColorMixingWebExtension."""
        plugin = _make_plugin()
        config = plugin.load_config(CONFIG_PATH)
        ext = plugin.web_extension(config)
        self.assertIsNotNone(ext)
        self.assertIsInstance(ext, ColorMixingWebExtension)

    def test_extension_provides_extra_status(self):
        """Extension's extra_status should provide calibration fields."""
        plugin = _make_plugin()
        config = plugin.load_config(CONFIG_PATH)
        ext = plugin.web_extension(config)
        state = {"artifacts": {"calibration": {
            "done": True,
            "pure_rgbs": {"red": [1, 0, 0]},
            "water_rgb": [0.5, 0.5, 0.5],
            "gamut": {"suggested_targets": [[100, 50, 200]]},
            "custom_target": [100, 50, 200],
        }}}
        extra = ext.extra_status(config, state)
        self.assertTrue(extra["calibration_done"])
        self.assertEqual(extra["pure_rgbs"]["red"], [1, 0, 0])
        self.assertEqual(extra["custom_target"], [100, 50, 200])
        self.assertEqual(extra["gamut_suggested_targets"], [[100, 50, 200]])


# ======================================================================
# Real-step integration: extract_rgb/fit_gp execute in-run
# ======================================================================

class TestRealStepExecution(unittest.TestCase):

    def test_loop_reads_step_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            loop = _make_loop(store=store, register_dummy_handlers=True)
            loop.start(mode="quick")
            loop._thread.join(timeout=30)
            s = loop.status()
            if s["history"]:
                cache = loop._task_state.get("cache", {})
                self.assertTrue(len(cache.get("all_X", [])) > 0)
                self.assertTrue(len(cache.get("all_Y", [])) > 0)
                self.assertTrue(len(cache.get("all_dist", [])) > 0)

    def test_extract_rgb_and_fit_gp_steps_visible_in_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            loop = _make_loop(store=store, register_dummy_handlers=True)
            loop.start(mode="quick")
            loop._thread.join(timeout=30)
            s = loop.status()
            if s["history"]:
                run_id = s["history"][0].get("run_id")
                if run_id:
                    run = store.load_run(run_id)
                    kinds = [step.kind for step in run.steps]
                    self.assertIn("extract_rgb", kinds)
                    self.assertIn("fit_gp", kinds)
                    for step in run.steps:
                        if step.kind in ("extract_rgb", "fit_gp"):
                            self.assertEqual(step.status, StepStatus.succeeded)
                            self.assertIsNotNone(step.output)


# ======================================================================
# Plugin-driven architecture verification
# ======================================================================

class TestNoTaskSpecificSemanticsInLoop(unittest.TestCase):

    def test_loop_does_not_import_task_config(self):
        import inspect
        from src.web import loop
        source = inspect.getsource(loop)
        self.assertNotIn("from src.tasks.color_mixing.config", source)
        self.assertNotIn("from src.tasks.color_mixing.steps", source)

    def test_loop_has_no_build_column_steps(self):
        self.assertFalse(hasattr(ActiveLearningLoop, "_build_column_steps"))

    def test_loop_has_no_module_level_build_plan(self):
        import src.web.loop as loop_module
        self.assertIsNone(getattr(loop_module, "build_plan", None))

    def test_loop_uses_plugin_for_iteration_runs(self):
        import inspect
        source = inspect.getsource(ActiveLearningLoop._run_loop)
        self.assertIn("plugin.build_iteration_run", source)
        self.assertIn("plugin.apply_run_result", source)

    def test_loop_uses_plugin_for_calibration(self):
        import inspect
        source = inspect.getsource(ActiveLearningLoop._run_calibration)
        self.assertIn("plugin.build_calibration_run", source)
        self.assertIn("apply_calibration_result", source)

    def test_loop_uses_plugin_for_tip_check(self):
        import inspect
        source = inspect.getsource(ActiveLearningLoop.tip_check)
        self.assertIn("plugin.build_tip_check_run", source)

    def test_app_uses_plugin_for_calibration_targets(self):
        import inspect
        from src.web.app import create_protocol_app
        source = inspect.getsource(create_protocol_app)
        self.assertIn("plugin.build_calibration_targets", source)
        self.assertNotIn("cfg.build_calibration_targets", source)

    def test_loop_has_no_set_target_method(self):
        """set_target should be in the extension, not the loop."""
        self.assertFalse(hasattr(ActiveLearningLoop, "set_target"))

    def test_loop_has_no_private_calibration_fields(self):
        """Loop should not have _pure_rgbs, _water_rgb, _gamut, etc."""
        loop = _make_loop()
        self.assertFalse(hasattr(loop, "_pure_rgbs"))
        self.assertFalse(hasattr(loop, "_water_rgb"))
        self.assertFalse(hasattr(loop, "_gamut"))
        self.assertFalse(hasattr(loop, "_custom_target"))
        self.assertFalse(hasattr(loop, "_cached_reference"))


class TestServerPluginDriven(unittest.TestCase):

    def test_server_imports_plugin(self):
        server_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "server.py")
        with open(server_path) as f:
            src = f.read()
        self.assertIn("ColorMixingPlugin", src)
        self.assertIn("plugin.load_config", src)
        self.assertIn("plugin.build_deck_config", src)

    def test_server_creates_loop_with_plugin(self):
        server_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "server.py")
        with open(server_path) as f:
            src = f.read()
        self.assertIn("plugin=plugin", src)
        self.assertIn("config=cfg", src)

    def test_server_checks_web_extension(self):
        server_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "server.py")
        with open(server_path) as f:
            src = f.read()
        self.assertIn("web_extension", src)

    def test_server_no_labware_section_parsing(self):
        server_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "server.py")
        with open(server_path) as f:
            src = f.read()
        self.assertNotIn("labware_section", src)
        self.assertNotIn('get("labware"', src)

    def test_server_uses_canonical_webapp_import(self):
        server_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "server.py")
        with open(server_path) as f:
            src = f.read()
        self.assertIn("from openot2.webapp import WebApp", src)
        self.assertIn("from openot2.webapp.deck import DeckConfig", src)


class TestCLIPluginDriven(unittest.TestCase):

    def test_run_experiment_imports_plugin(self):
        path = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_experiment.py")
        with open(path) as f:
            src = f.read()
        self.assertIn("ColorMixingPlugin", src)
        self.assertIn("plugin.load_config", src)


class TestRinseConfig(unittest.TestCase):

    def test_server_passes_rinse_config(self):
        server_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "server.py")
        with open(server_path) as f:
            src = f.read()
        self.assertIn("rinse_volume", src)
        self.assertIn("rinse_cycles", src)
        self.assertIn("cleaning", src)


class TestCanonicalImports(unittest.TestCase):

    def test_handlers_uses_canonical_vision(self):
        handlers_path = os.path.join(
            os.path.dirname(__file__), "..", "OpenOT2", "openot2", "webapp", "handlers.py"
        )
        with open(handlers_path) as f:
            src = f.read()
        self.assertNotIn("from vision ", src)
        self.assertNotIn("from vision.", src)
        self.assertIn("from openot2.vision", src)

    def test_webapp_web_uses_canonical_vision(self):
        web_path = os.path.join(
            os.path.dirname(__file__), "..", "OpenOT2", "openot2", "webapp", "web.py"
        )
        with open(web_path) as f:
            src = f.read()
        self.assertNotIn("from vision ", src)
        self.assertNotIn("from vision.", src)

    def test_no_bare_webapp_imports_in_openot2(self):
        import glob as globmod
        ot2_dir = os.path.join(os.path.dirname(__file__), "..", "OpenOT2")
        for pyfile in globmod.glob(os.path.join(ot2_dir, "**", "*.py"), recursive=True):
            if "__pycache__" in pyfile:
                continue
            with open(pyfile) as f:
                for lineno, line in enumerate(f, 1):
                    stripped = line.strip()
                    if stripped.startswith("from webapp.") or stripped.startswith("from webapp "):
                        if not stripped.startswith("#"):
                            self.fail(f"Bare 'from webapp' import in {pyfile}:{lineno}")


class TestNoPathHacks(unittest.TestCase):

    def test_run_experiment_no_path_hack(self):
        path = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_experiment.py")
        with open(path) as f:
            src = f.read()
        self.assertNotIn("sys.path.insert", src)
        self.assertNotIn("sys.path.append", src)

    def test_server_no_path_hack(self):
        path = os.path.join(os.path.dirname(__file__), "..", "scripts", "server.py")
        with open(path) as f:
            src = f.read()
        self.assertNotIn("sys.path.insert", src)

    def test_test_web_protocol_no_path_hack(self):
        path = os.path.join(os.path.dirname(__file__), "test_web_protocol.py")
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("class ") or stripped.startswith("def "):
                break
            self.assertNotIn("sys.path.insert", stripped)


if __name__ == "__main__":
    unittest.main()
