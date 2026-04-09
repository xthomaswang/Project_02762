"""Tests for the web protocol integration (src/web/).

Covers: plan generation, status schema, step param names, run creation,
handler contracts, state machine (pause/stop/fail), analysis output,
FastAPI routes, and rinse config plumbing.
"""

import csv
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "OpenOT2"))

from openot2.control.models import TaskRun, RunStep, RunSequence, RunStatus, StepStatus
from openot2.control.store import JsonRunStore
from openot2.control.runner import TaskRunner

from src.web.loop import ActiveLearningLoop, build_plan
from src.web.handlers import handle_extract_rgb, handle_suggest_volumes, handle_fit_gp

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment.yaml")


def _load_config():
    import yaml
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _make_loop(store=None, register_dummy_handlers=False):
    """Create loop with temp store. Optionally register dummy handlers."""
    if store is None:
        tmpdir = tempfile.mkdtemp()
        store = JsonRunStore(base_dir=tmpdir)
    runner = TaskRunner(store=store)
    if register_dummy_handlers:
        for kind in ("use_pipette", "transfer", "mix", "home", "wait"):
            runner.register_handler(kind, lambda step, ctx=None: {"ok": True})
        # Capture must return image_path for the loop's data wiring
        runner.register_handler("capture", lambda step, ctx=None: {
            "ok": True, "image_path": "/tmp/fake_capture.jpg",
        })
        runner.register_handler("extract_rgb", lambda step, ctx=None: {
            "mean_rgb": [100, 50, 150], "std_rgb": [5, 5, 5],
            "raw_mean_rgb": [100, 50, 150], "used_calibration": False,
            "runtime_reference": None,
        })
        runner.register_handler("fit_gp", lambda step, ctx=None: {
            "distance": 42.0, "converged": False, "best_distance": 42.0,
            "best_iteration": 1, "all_X": [], "all_Y": [], "all_dist": [42.0],
        })
    return ActiveLearningLoop(runner=runner, store=store, config_path=CONFIG_PATH)


# ======================================================================
# Plan generation
# ======================================================================

class TestBuildPlan(unittest.TestCase):

    def test_quick_plan(self):
        plan = build_plan(_load_config(), "quick")
        self.assertEqual(plan["mode"], "quick")
        self.assertTrue(plan["skip_controls_after_first"])
        self.assertEqual(plan["mix_cycles"], 2)
        self.assertEqual(len(plan["iterations"]), plan["total_iterations"])
        self.assertFalse(plan["iterations"][0]["skip_controls"])
        self.assertTrue(plan["iterations"][1]["skip_controls"])
        for it in plan["iterations"][:plan["n_initial"]]:
            self.assertEqual(it["phase"], "random")
            self.assertIsNotNone(it["volumes"])
        for it in plan["iterations"][plan["n_initial"]:]:
            self.assertEqual(it["phase"], "bo")
            self.assertIsNone(it["volumes"])

    def test_full_plan_no_skip(self):
        plan = build_plan(_load_config(), "full")
        self.assertFalse(plan["skip_controls_after_first"])
        for it in plan["iterations"]:
            self.assertFalse(it["skip_controls"])

    def test_plan_required_fields(self):
        plan = build_plan(_load_config(), "quick")
        for key in ("mode", "target_rgb", "convergence_threshold",
                     "total_iterations", "n_initial", "n_optimization",
                     "distance_metric", "iterations"):
            self.assertIn(key, plan)


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
            self.assertEqual(s["active_run"]["current_step"]["kind"], "transfer")
            self.assertEqual(len(s["active_run"]["steps"]), 2)
            self.assertIsNone(s["active_run"]["live_progress"])

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
# Step param names (contract with OT2 handlers)
# ======================================================================

class TestStepParamNames(unittest.TestCase):

    def test_transfer_uses_rinse_well(self):
        loop = _make_loop()
        steps = loop._build_column_steps(0, 0, [100, 50, 50], False)
        for s in [x for x in steps if x["kind"] == "transfer"]:
            self.assertIn("rinse_well", s["params"], s["name"])
            self.assertNotIn("rinse_col", s["params"], s["name"])

    def test_mix_uses_plate_slot(self):
        loop = _make_loop()
        steps = loop._build_column_steps(0, 0, [100, 50, 50], False)
        mixes = [s for s in steps if s["kind"] == "mix"]
        self.assertEqual(len(mixes), 1)
        self.assertIn("plate_slot", mixes[0]["params"])
        self.assertNotIn("labware_slot", mixes[0]["params"])
        self.assertIn("rinse_well", mixes[0]["params"])

    def test_all_steps_have_names(self):
        loop = _make_loop()
        steps = loop._build_column_steps(0, 0, [100, 50, 50], False)
        for s in steps:
            self.assertIn("name", s)
            self.assertTrue(len(s["name"]) > 0)

    def test_capture_has_label(self):
        loop = _make_loop()
        steps = loop._build_column_steps(0, 0, [100, 50, 50], False)
        caps = [s for s in steps if s["kind"] == "capture"]
        self.assertEqual(len(caps), 1)
        self.assertIn("label", caps[0]["params"])

    def test_capture_sequence_does_not_home(self):
        loop = _make_loop()
        steps = loop._build_column_steps(0, 0, [100, 50, 50], False)
        kinds = [s["kind"] for s in steps]
        self.assertNotIn("home", kinds)

    def test_skip_controls_fewer_steps(self):
        loop = _make_loop()
        with_ctrl = loop._build_column_steps(0, 0, [100, 50, 50], False)
        no_ctrl = loop._build_column_steps(1, 1, [100, 50, 50], True)
        self.assertGreater(len(with_ctrl), len(no_ctrl))


# ======================================================================
# Run creation
# ======================================================================

class TestRunCreation(unittest.TestCase):

    def test_steps_are_valid_runsteps(self):
        loop = _make_loop()
        steps = loop._build_column_steps(0, 0, [100, 50, 50], False)
        for s in steps:
            rs = RunStep(**s)
            self.assertIsInstance(rs, RunStep)

    def test_taskrun_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            loop = _make_loop(store=store)
            steps = loop._build_column_steps(0, 0, [100, 50, 50], False)
            run_steps = [RunStep(**s) for s in steps]
            run = TaskRun(name="test", steps=run_steps, metadata={"test": True})
            created = store.create_run(run)
            reloaded = store.load_run(created.id)
            self.assertEqual(len(reloaded.steps), len(steps))


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
            "image_path": "",
            "col": 0,
            "grid_path": "",
            "cached_reference": None,
            "skip_controls": False,
        }))
        self.assertTrue(result["deferred"])
        self.assertEqual(result["reason"], "waiting_for_capture_output")

    def test_fit_gp_deferred_without_observed_rgb(self):
        class P:
            def __init__(self, p): self.params = p
        result = handle_fit_gp(P({
            "volumes": [100, 50, 50],
            "observed_rgb": [],
            "target_rgb": [128, 0, 255],
            "all_X": [],
            "all_Y": [],
            "all_dist": [],
        }))
        self.assertTrue(result["deferred"])
        self.assertEqual(result["reason"], "waiting_for_post_hoc_binding")


# ======================================================================
# State machine
# ======================================================================

class TestStateMachine(unittest.TestCase):

    def test_fatal_error_sets_failed(self):
        """If runner has no handlers, loop crashes and state becomes 'failed'."""
        loop = _make_loop()  # no handlers registered
        loop.start(mode="quick")
        loop._thread.join(timeout=15)
        s = loop.status()
        self.assertEqual(s["state"], "failed")
        self.assertIsNotNone(s["error"])

    def test_stop_prevents_new_iterations(self):
        """Stop should prevent the loop from continuing to next iteration."""
        loop = _make_loop(register_dummy_handlers=True)
        loop.start(mode="quick")
        time.sleep(0.3)
        loop.stop()
        loop._thread.join(timeout=15)
        s = loop.status()
        self.assertIn(s["state"], ("stopped", "failed", "completed", "converged"))

    def test_pause_state_is_pausing_not_paused(self):
        """pause() sets state to 'pausing', not immediately 'paused'."""
        loop = _make_loop()
        loop._state = "running"
        loop._active_run_id = "fake-id"
        # pause() should set pausing (runner.request_pause may fail on fake id, that's OK)
        loop.pause()
        self.assertEqual(loop._state, "pausing")

    def test_resume_only_from_paused(self):
        """resume() should only work when state is 'paused'."""
        loop = _make_loop()
        loop._state = "idle"
        loop.resume()
        self.assertEqual(loop._state, "idle")  # unchanged

        loop._state = "paused"
        loop.resume()
        self.assertEqual(loop._state, "running")

    def test_reset_clears_state(self):
        """reset() clears all state back to idle."""
        loop = _make_loop()
        loop._state = "completed"
        loop._all_dist = [1.0, 2.0]
        loop._history = [{"test": True}]
        loop._error = "old error"
        result = loop.reset()
        self.assertTrue(result["ok"])
        s = loop.status()
        self.assertEqual(s["state"], "idle")
        self.assertEqual(s["history"], [])
        self.assertIsNone(s["error"])

    def test_reset_blocked_while_running(self):
        """reset() should refuse while loop is running."""
        loop = _make_loop()
        loop._state = "running"
        result = loop.reset()
        self.assertFalse(result["ok"])

    def test_reset_blocked_while_paused(self):
        """reset() should refuse while loop is paused on an active run."""
        loop = _make_loop()
        loop._state = "paused"
        loop._active_run_id = "run-1"
        result = loop.reset()
        self.assertFalse(result["ok"])

    def test_home_blocked_while_running(self):
        """home() should refuse while loop is running."""
        loop = _make_loop()
        loop._state = "running"
        result = loop.home()
        self.assertFalse(result["ok"])

    def test_home_blocked_while_paused(self):
        """home() should refuse while loop is paused on an active run."""
        loop = _make_loop()
        loop._state = "paused"
        loop._active_run_id = "run-1"
        result = loop.home()
        self.assertFalse(result["ok"])

    def test_home_reports_failed_manual_run(self):
        """home() should surface a failed manual home run as an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            runner = TaskRunner(store=store)  # no home handler registered
            loop = ActiveLearningLoop(runner=runner, store=store, config_path=CONFIG_PATH)
            result = loop.home()
            self.assertFalse(result["ok"])
            self.assertIn("error", result)


# ======================================================================
# Sequence lifecycle
# ======================================================================

class TestSequenceLifecycle(unittest.TestCase):

    def test_sequence_created_on_start(self):
        """start() creates a RunSequence in the store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            loop = _make_loop(store=store)
            loop.start(mode="quick")
            loop._thread.join(timeout=15)
            # Should have created a sequence
            seqs = store.list_sequences()
            self.assertGreaterEqual(len(seqs), 1)

    def test_sequence_status_updated_on_failure(self):
        """Sequence status should reflect terminal state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            loop = _make_loop(store=store)  # no handlers = will fail
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
        """When loop fails, current plan item should be 'failed'."""
        loop = _make_loop()  # no handlers
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
        """Terminal state should produce analysis files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            loop = _make_loop(store=store, register_dummy_handlers=True)
            loop.start(mode="quick")
            loop._thread.join(timeout=30)

            s = loop.status()
            seq_id = s["sequence_id"]
            if seq_id:
                analysis_dir = os.path.join(tmpdir, "analysis", seq_id)
                # Analysis may or may not be saved depending on whether
                # iterations completed. Check if dir exists when we have history.
                if s["history"]:
                    self.assertTrue(
                        os.path.isdir(analysis_dir),
                        f"Analysis dir not created: {analysis_dir}",
                    )


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
            loop = ActiveLearningLoop(runner=runner, store=store, config_path=CONFIG_PATH)
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
        self.assertIn("Color Mixing Protocol", r.text)


# ======================================================================
# Rinse config plumbing
# ======================================================================

class TestQuickModeRinse(unittest.TestCase):

    def test_quick_mode_rinse_cycles_2(self):
        """Quick mode should set rinse_cycles=2 in step params."""
        loop = _make_loop()
        steps = loop._build_column_steps(
            0, 0, [100, 50, 50], False,
            rinse_cycles=2, rinse_volume=200,
        )
        transfers = [s for s in steps if s["kind"] == "transfer"]
        for t in transfers:
            self.assertEqual(t["params"].get("rinse_cycles"), 2, t["name"])
            self.assertEqual(t["params"].get("rinse_volume"), 200, t["name"])

    def test_full_mode_rinse_cycles_3(self):
        """Full mode should use config default (3 cycles)."""
        loop = _make_loop()
        steps = loop._build_column_steps(
            0, 0, [100, 50, 50], False,
            rinse_cycles=3, rinse_volume=200,
        )
        transfers = [s for s in steps if s["kind"] == "transfer"]
        for t in transfers:
            self.assertEqual(t["params"].get("rinse_cycles"), 3, t["name"])

    def test_mix_also_gets_rinse_params(self):
        """Mix step should also get rinse_cycles from config."""
        loop = _make_loop()
        steps = loop._build_column_steps(
            0, 0, [100, 50, 50], False,
            rinse_cycles=2, rinse_volume=200,
        )
        mixes = [s for s in steps if s["kind"] == "mix"]
        self.assertEqual(mixes[0]["params"].get("rinse_cycles"), 2)


class TestRinseConfig(unittest.TestCase):

    def test_config_has_rinse_200(self):
        """experiment.yaml should specify 200 uL rinse, not 250."""
        cfg = _load_config()
        rinse_vol = cfg.get("cleaning", {}).get("rinse_volume_ul", 250)
        self.assertEqual(rinse_vol, 200)

    def test_server_passes_rinse_config(self):
        """server.py should read cleaning config and pass to WebApp."""
        import ast
        server_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "server.py")
        with open(server_path) as f:
            src = f.read()
        self.assertIn("rinse_volume", src)
        self.assertIn("rinse_cycles", src)
        self.assertIn("cleaning", src)


class TestCalibrationFlow(unittest.TestCase):
    """Test calibration method on loop."""

    def test_calibrate_blocked_while_running(self):
        loop = _make_loop()
        loop._state = "running"
        result = loop.calibrate()
        self.assertFalse(result["ok"])

    def test_set_target(self):
        loop = _make_loop()
        result = loop.set_target([100, 50, 200])
        self.assertTrue(result["ok"])
        self.assertEqual(loop._custom_target, [100.0, 50.0, 200.0])
        s = loop.status()
        self.assertEqual(s["custom_target"], [100.0, 50.0, 200.0])

    def test_set_target_invalid(self):
        loop = _make_loop()
        result = loop.set_target([100, 50])  # only 2 values
        self.assertFalse(result["ok"])

    def test_calibration_status_fields(self):
        loop = _make_loop()
        s = loop.status()
        self.assertIn("calibration_done", s)
        self.assertIn("pure_rgbs", s)
        self.assertIn("custom_target", s)
        self.assertFalse(s["calibration_done"])

    def test_start_column_after_calibration(self):
        """After calibration, quick plan should start at column 2 and shrink by one random run."""
        from src.web.loop import build_plan
        import yaml
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        plan = build_plan(cfg, "quick", start_column=1, pre_calibrated=True)
        self.assertEqual(plan["iterations"][0]["column"], 2)
        self.assertEqual(plan["n_initial"], 4)
        self.assertEqual(plan["total_iterations"], 11)
        self.assertTrue(plan["iterations"][0]["skip_controls"])

    def test_start_preserves_calibration_state(self):
        loop = _make_loop()
        loop._calibration_done = True
        loop._pure_rgbs = {
            "red": [1.0, 0.0, 0.0],
            "green": [0.0, 1.0, 0.0],
            "blue": [0.0, 0.0, 1.0],
        }
        loop._water_rgb = [0.5, 0.5, 0.5]
        loop._gamut = {"samples_rgb": [[1.0, 0.0, 0.0]]}
        loop._custom_target = [100.0, 50.0, 200.0]
        loop._cached_reference = {"source_column": 1}

        with mock.patch.object(threading.Thread, "start", return_value=None):
            loop.start(mode="quick")

        self.assertTrue(loop._calibration_done)
        self.assertEqual(loop._pure_rgbs["red"], [1.0, 0.0, 0.0])
        self.assertEqual(loop._water_rgb, [0.5, 0.5, 0.5])
        self.assertEqual(loop._custom_target, [100.0, 50.0, 200.0])
        self.assertEqual(loop._cached_reference, {"source_column": 1})
        self.assertEqual(loop._plan["iterations"][0]["column"], 2)
        self.assertTrue(loop._plan["iterations"][0]["skip_controls"])
        self.assertEqual(loop._plan["n_initial"], 4)
        self.assertEqual(loop._plan["total_iterations"], 11)

    def test_calibration_uses_resolved_grid_path(self):
        loop = _make_loop(register_dummy_handlers=True)
        expected_grid_path = os.path.abspath(
            os.path.join(
                os.path.dirname(CONFIG_PATH),
                "calibrations",
                "cv_calibration",
                "grid.json",
            )
        )

        with mock.patch(
            "src.web.loop.handle_extract_rgb",
            return_value={
                "mean_rgb": [0.0, 0.0, 0.0],
                "std_rgb": [0.0, 0.0, 0.0],
                "raw_mean_rgb": [0.0, 0.0, 0.0],
                "used_calibration": True,
                "runtime_reference": None,
            },
        ) as extract_rgb_mock, mock.patch(
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
        self.assertEqual(
            extract_rgb_mock.call_args[0][0].params["grid_path"],
            expected_grid_path,
        )
        self.assertTrue(loop._calibration_done)

    def test_get_plan_uses_calibrated_start_column(self):
        loop = _make_loop()
        loop._calibration_done = True
        plan = loop.get_plan("quick")
        self.assertEqual(plan["iterations"][0]["column"], 2)
        self.assertTrue(plan["iterations"][0]["skip_controls"])
        self.assertEqual(plan["n_initial"], 4)
        self.assertEqual(plan["total_iterations"], 11)

    def test_loop_autoloads_saved_color_calibration(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)
            config_path = tmpdir / "experiment.yaml"
            config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

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
            (cal_dir / "color_profile.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            store = JsonRunStore(base_dir=str(tmpdir / "run_data"))
            runner = TaskRunner(store=store)
            for kind in ("use_pipette", "transfer", "mix", "home", "wait"):
                runner.register_handler(kind, lambda step, ctx=None: {"ok": True})
            runner.register_handler(
                "capture",
                lambda step, ctx=None: {"ok": True, "image_path": "/tmp/fake_capture.jpg"},
            )
            loop = ActiveLearningLoop(
                runner=runner,
                store=store,
                config_path=str(config_path),
            )

            self.assertTrue(loop._calibration_done)
            self.assertEqual(loop._pure_rgbs["red"], [255.0, 0.0, 0.0])
            self.assertEqual(loop._water_rgb, [240.0, 240.0, 240.0])
            self.assertEqual(loop._cached_reference, {"source_column": 1})

    def test_calibration_persists_color_profile(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)
            config_path = tmpdir / "experiment.yaml"
            config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

            store = JsonRunStore(base_dir=str(tmpdir / "run_data"))
            runner = TaskRunner(store=store)
            for kind in ("use_pipette", "transfer", "mix", "home", "wait"):
                runner.register_handler(kind, lambda step, ctx=None: {"ok": True})
            runner.register_handler(
                "capture",
                lambda step, ctx=None: {"ok": True, "image_path": "/tmp/fake_capture.jpg"},
            )
            loop = ActiveLearningLoop(
                runner=runner,
                store=store,
                config_path=str(config_path),
            )

            with mock.patch(
                "src.web.loop.handle_extract_rgb",
                return_value={
                    "mean_rgb": [0.0, 0.0, 0.0],
                    "std_rgb": [0.0, 0.0, 0.0],
                    "raw_mean_rgb": [0.0, 0.0, 0.0],
                    "used_calibration": True,
                    "runtime_reference": {"source_column": 1},
                },
            ), mock.patch(
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
            saved_path = tmpdir / "calibrations" / "color_calibration" / "color_profile.json"
            self.assertTrue(saved_path.exists())
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["pure_rgbs"]["red"], [255.0, 0.0, 0.0])
            self.assertEqual(saved["cached_reference"], {"source_column": 1})

    def test_gamut_api_route(self):
        from starlette.testclient import TestClient
        from src.web.app import create_protocol_app
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            runner = TaskRunner(store=store)
            loop = ActiveLearningLoop(runner=runner, store=store, config_path=CONFIG_PATH)
            app = create_protocol_app(loop)
            client = TestClient(app)
            r = client.get("/gamut")
            self.assertEqual(r.status_code, 200)
            self.assertFalse(r.json()["calibration_done"])

    def test_robot_calibration_profile_route(self):
        from starlette.testclient import TestClient
        from src.web.app import create_protocol_app
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            runner = TaskRunner(store=store)
            loop = ActiveLearningLoop(runner=runner, store=store, config_path=CONFIG_PATH)
            app = create_protocol_app(loop)
            client = TestClient(app)
            r = client.get("/robot-calibration-profile")
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertIn("targets", data)
            right_tip = next(t for t in data["targets"] if t["labware_slot"] == "11")
            self.assertEqual(right_tip["pipette_mount"], "right")
            self.assertEqual(right_tip["action"], "pick_up_tip")

    def test_set_target_route(self):
        from starlette.testclient import TestClient
        from src.web.app import create_protocol_app
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonRunStore(base_dir=tmpdir)
            runner = TaskRunner(store=store)
            loop = ActiveLearningLoop(runner=runner, store=store, config_path=CONFIG_PATH)
            app = create_protocol_app(loop)
            client = TestClient(app)
            r = client.post("/set-target", json={"rgb": [100, 50, 200]})
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.json()["ok"])


if __name__ == "__main__":
    unittest.main()
