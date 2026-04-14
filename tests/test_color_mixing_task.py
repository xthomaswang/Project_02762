"""Tests for the color mixing task module.

Covers: config loading, build_iteration_steps, analyze_capture,
fit_observation, build_robot_calibration_profile.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "color_mixing.yaml",
)


class TestLoadTaskConfig(unittest.TestCase):
    """Config loader should parse the canonical YAML into typed dataclass."""

    def test_loads_without_error(self):
        from src.tasks.color_mixing.config import load_task_config
        cfg = load_task_config(CONFIG_PATH)
        self.assertEqual(cfg.target_color, [128, 0, 255])
        self.assertEqual(cfg.total_volume_ul, 200.0)

    def test_reagent_roles(self):
        from src.tasks.color_mixing.config import load_task_config
        cfg = load_task_config(CONFIG_PATH)
        self.assertIn("red", cfg.reagents)
        self.assertIn("green", cfg.reagents)
        self.assertIn("blue", cfg.reagents)
        self.assertIn("water", cfg.reagents)
        self.assertEqual(cfg.reagents["red"].slot, "7")

    def test_well_roles(self):
        from src.tasks.color_mixing.config import load_task_config
        cfg = load_task_config(CONFIG_PATH)
        self.assertEqual(cfg.control_rows, {"A": "red", "B": "green",
                                            "C": "blue", "D": "water"})
        self.assertEqual(cfg.experiment_rows, ["E", "F", "G", "H"])

    def test_tipracks(self):
        from src.tasks.color_mixing.config import load_task_config
        cfg = load_task_config(CONFIG_PATH)
        self.assertIn("left", cfg.tipracks)
        self.assertIn("right", cfg.tipracks)
        self.assertEqual(cfg.tipracks["right"].slot, "11")
        self.assertEqual(cfg.tip_columns["red"], 1)
        self.assertEqual(cfg.tip_wells["water"], "D1")

    def test_plate(self):
        from src.tasks.color_mixing.config import load_task_config
        cfg = load_task_config(CONFIG_PATH)
        self.assertEqual(cfg.plate.slot, "1")

    def test_cleaning(self):
        from src.tasks.color_mixing.config import load_task_config
        cfg = load_task_config(CONFIG_PATH)
        self.assertEqual(cfg.cleaning.slot, "4")
        self.assertIn("mix", cfg.cleaning.rinse_wells)

    def test_ml_config(self):
        from src.tasks.color_mixing.config import load_task_config
        cfg = load_task_config(CONFIG_PATH)
        self.assertEqual(cfg.ml.model, "correlated_gp")
        self.assertEqual(cfg.ml.distance_metric, "delta_e_lab")

    def test_experiment_config(self):
        from src.tasks.color_mixing.config import load_task_config
        cfg = load_task_config(CONFIG_PATH)
        self.assertEqual(cfg.experiment.n_initial, 5)
        self.assertEqual(cfg.experiment.n_optimization, 7)
        self.assertEqual(cfg.experiment.mix_cycles, 3)

    def test_resolve_path(self):
        from src.tasks.color_mixing.config import load_task_config
        cfg = load_task_config(CONFIG_PATH)
        resolved = cfg.resolve_path("calibrations/cv_calibration/grid.json")
        self.assertTrue(os.path.isabs(resolved))

    def test_role_for_row(self):
        from src.tasks.color_mixing.config import load_task_config
        cfg = load_task_config(CONFIG_PATH)
        self.assertEqual(cfg.role_for_row("A"), "red")
        self.assertEqual(cfg.role_for_row("D"), "water")
        self.assertIsNone(cfg.role_for_row("E"))


class TestBuildIterationSteps(unittest.TestCase):
    """build_iteration_steps should produce role-driven steps."""

    def _cfg(self):
        from src.tasks.color_mixing.config import load_task_config
        return load_task_config(CONFIG_PATH)

    def test_full_mode_produces_all_steps(self):
        from src.tasks.color_mixing.steps import build_iteration_steps
        cfg = self._cfg()
        volumes = np.array([80.0, 60.0, 60.0])
        steps = build_iteration_steps(cfg, volumes, col_idx=0)

        # 4 controls + 3 dye transfers + 1 mix = 8
        self.assertEqual(len(steps), 8)
        phases = [s.phase for s in steps]
        self.assertEqual(phases.count("control"), 4)
        self.assertEqual(phases.count("experiment"), 4)

    def test_skip_controls(self):
        from src.tasks.color_mixing.steps import build_iteration_steps
        cfg = self._cfg()
        volumes = np.array([80.0, 60.0, 60.0])
        steps = build_iteration_steps(cfg, volumes, col_idx=0, skip_controls=True)

        # 3 dye transfers + 1 mix = 4
        self.assertEqual(len(steps), 4)
        self.assertTrue(all(s.phase == "experiment" for s in steps))

    def test_steps_reference_roles_not_slots(self):
        from src.tasks.color_mixing.steps import build_iteration_steps
        cfg = self._cfg()
        volumes = np.array([100.0, 50.0, 50.0])
        steps = build_iteration_steps(cfg, volumes, col_idx=2)

        for step in steps:
            # source_role should be a role name, not a slot
            if step.action == "transfer":
                self.assertIn(step.source_role,
                              ["red", "green", "blue", "water"])
            # dest_well should use correct column
            self.assertTrue(step.dest_well.endswith("3"),
                            f"Expected column 3, got {step.dest_well}")

    def test_control_steps_use_config_well_roles(self):
        from src.tasks.color_mixing.steps import build_iteration_steps
        cfg = self._cfg()
        volumes = np.array([100.0, 50.0, 50.0])
        steps = build_iteration_steps(cfg, volumes, col_idx=0)

        control_steps = [s for s in steps if s.phase == "control"]
        # Controls should match the well_roles.controls mapping
        dest_rows = {s.dest_well[0] for s in control_steps}
        expected_rows = set(cfg.control_rows.keys())
        self.assertEqual(dest_rows, expected_rows)

    def test_mix_step_present(self):
        from src.tasks.color_mixing.steps import build_iteration_steps
        cfg = self._cfg()
        volumes = np.array([100.0, 50.0, 50.0])
        steps = build_iteration_steps(cfg, volumes, col_idx=0)

        mix_steps = [s for s in steps if s.action == "mix"]
        self.assertEqual(len(mix_steps), 1)
        self.assertEqual(mix_steps[0].mix_cycles, cfg.experiment.mix_cycles)

    def test_mix_cycles_override(self):
        from src.tasks.color_mixing.steps import build_iteration_steps
        cfg = self._cfg()
        volumes = np.array([100.0, 50.0, 50.0])
        steps = build_iteration_steps(cfg, volumes, col_idx=0,
                                      mix_cycles_override=2)
        mix_step = [s for s in steps if s.action == "mix"][0]
        self.assertEqual(mix_step.mix_cycles, 2)

    def test_experiment_volumes_assigned_correctly(self):
        from src.tasks.color_mixing.steps import build_iteration_steps
        cfg = self._cfg()
        volumes = np.array([120.0, 30.0, 50.0])
        steps = build_iteration_steps(cfg, volumes, col_idx=0)

        exp_transfers = [s for s in steps
                         if s.phase == "experiment" and s.action == "transfer"]
        vol_map = {s.reagent: s.volume for s in exp_transfers}
        self.assertAlmostEqual(vol_map["red"], 120.0)
        self.assertAlmostEqual(vol_map["green"], 30.0)
        self.assertAlmostEqual(vol_map["blue"], 50.0)


class TestBuildRobotCalibrationProfile(unittest.TestCase):
    """build_robot_calibration_profile should resolve roles to labware IDs."""

    def test_profile_maps_all_roles(self):
        from src.tasks.color_mixing.config import load_task_config
        from src.tasks.color_mixing.deck import build_robot_calibration_profile

        cfg = load_task_config(CONFIG_PATH)

        # Mock robot
        robot = MagicMock()
        robot.load_pipette.return_value = "pip_id"
        robot.load_labware.side_effect = lambda name, slot: f"{name}@{slot}"

        profile = build_robot_calibration_profile(cfg, robot)

        # All reagents should be mapped
        for role in ("red", "green", "blue", "water"):
            self.assertIn(role, profile.reagent_ids)

        # Both tipracks
        self.assertIn("left", profile.tiprack_ids)
        self.assertIn("right", profile.tiprack_ids)

        # Plate and cleaning
        self.assertNotEqual(profile.plate_id, "")
        self.assertNotEqual(profile.cleaning_id, "")

    def test_labware_loads_use_config_slots(self):
        from src.tasks.color_mixing.config import load_task_config
        from src.tasks.color_mixing.deck import build_robot_calibration_profile

        cfg = load_task_config(CONFIG_PATH)
        robot = MagicMock()
        call_log = []

        def track_load(name, slot):
            call_log.append((name, slot))
            return f"{name}@{slot}"

        robot.load_labware.side_effect = track_load
        robot.load_pipette.return_value = "pip"

        build_robot_calibration_profile(cfg, robot)

        loaded_slots = {slot for _, slot in call_log}
        # Should include plate(1), cleaning(4), water(5), red(7), green(8), blue(9), tipracks(10,11)
        for expected in ("1", "4", "5", "7", "8", "9", "10", "11"):
            self.assertIn(expected, loaded_slots,
                          f"Slot {expected} not loaded")


class TestExecuteSteps(unittest.TestCase):
    """execute_steps should dispatch to OT2Operations correctly."""

    def test_transfer_calls_ops(self):
        from src.tasks.color_mixing.config import load_task_config
        from src.tasks.color_mixing.steps import build_iteration_steps, execute_steps
        from src.tasks.color_mixing.deck import DeckProfile

        cfg = load_task_config(CONFIG_PATH)
        volumes = np.array([80.0, 60.0, 60.0])
        steps = build_iteration_steps(cfg, volumes, col_idx=0, skip_controls=True)

        profile = DeckProfile(
            reagent_ids={"red": "r1", "green": "r2", "blue": "r3", "water": "r4"},
            tiprack_ids={"left": "tl", "right": "tr"},
            plate_id="plate",
            cleaning_id="clean",
        )

        ops = MagicMock()
        ops.client = MagicMock()

        execute_steps(steps, profile, ops)

        # 3 transfers (skipping if < 1.0 µL) + 1 mix
        transfer_calls = ops.transfer.call_count
        mix_calls = ops.mix.call_count
        self.assertGreaterEqual(transfer_calls, 1)
        self.assertEqual(mix_calls, 1)


class TestAnalyzeCapture(unittest.TestCase):
    """analyze_capture should return structured CaptureResult."""

    @patch("src.tasks.color_mixing.observation.extract_experiment_rgb")
    def test_returns_capture_result(self, mock_extract):
        from src.tasks.color_mixing.config import load_task_config
        from src.tasks.color_mixing.observation import analyze_capture

        mock_extract.return_value = {
            "experiment_rgb": np.array([
                [100.0, 60.0, 100.0],
                [105.0, 55.0, 95.0],
                [98.0, 62.0, 102.0],
                [102.0, 58.0, 98.0],
            ]),
            "experiment_mean": np.array([101.25, 58.75, 98.75]),
            "experiment_std": np.array([2.6, 2.6, 2.6]),
            "control_rgb": {
                "A": np.array([200.0, 20.0, 20.0]),
            },
        }

        cfg = load_task_config(CONFIG_PATH)

        # Patch cv2.imread to return None (skip calibration)
        with patch("cv2.imread", return_value=None):
            result = analyze_capture(
                image_path="/tmp/fake.jpg",
                cfg=cfg,
                col_idx=0,
            )

        self.assertIsNotNone(result.mean_rgb)
        self.assertEqual(result.mean_rgb.shape, (3,))
        self.assertEqual(result.image_path, "/tmp/fake.jpg")


class TestFitObservation(unittest.TestCase):
    """fit_observation should update the surrogate."""

    def test_incremental_update(self):
        from src.tasks.color_mixing.observation import fit_observation

        surrogate = MagicMock()
        volumes = np.array([100.0, 50.0, 50.0])
        rgb = np.array([128.0, 64.0, 200.0])

        fit_observation(surrogate, volumes, rgb)

        surrogate.update.assert_called_once()
        call_args = surrogate.update.call_args
        np.testing.assert_array_equal(call_args[0][0], volumes.reshape(1, 3))
        np.testing.assert_array_equal(call_args[0][1], rgb.reshape(1, 3))

    def test_full_refit(self):
        from src.tasks.color_mixing.observation import fit_observation

        surrogate = MagicMock()
        all_X = np.array([[100, 50, 50], [60, 80, 60]])
        all_Y = np.array([[128, 64, 200], [100, 150, 100]])

        fit_observation(surrogate, all_X[0], all_Y[0],
                        full_refit=True, all_X=all_X, all_Y=all_Y)

        surrogate.fit.assert_called_once()


class TestBuildDeckDict(unittest.TestCase):
    """build_deck_dict should produce a valid DeckConfig-compatible dict."""

    def test_has_all_slots(self):
        from src.tasks.color_mixing.config import load_task_config
        cfg = load_task_config(CONFIG_PATH)
        d = cfg.build_deck_dict()
        self.assertIn("pipettes", d)
        self.assertIn("labware", d)
        # Should have plate(1), cleaning(4), water(5), red(7), green(8), blue(9), tipracks(10,11)
        for slot in ("1", "4", "5", "7", "8", "9", "10", "11"):
            self.assertIn(slot, d["labware"], f"Missing slot {slot}")

    def test_pipettes(self):
        from src.tasks.color_mixing.config import load_task_config
        cfg = load_task_config(CONFIG_PATH)
        d = cfg.build_deck_dict()
        self.assertEqual(d["pipettes"]["left"], "p300_single_gen2")
        self.assertEqual(d["pipettes"]["right"], "p300_multi_gen2")


class TestBuildCalibrationTargets(unittest.TestCase):
    """build_calibration_targets should produce targets for robot calibration UI."""

    def test_has_targets(self):
        from src.tasks.color_mixing.config import load_task_config
        cfg = load_task_config(CONFIG_PATH)
        targets = cfg.build_calibration_targets()
        self.assertGreater(len(targets), 0)

    def test_targets_reference_config_slots(self):
        from src.tasks.color_mixing.config import load_task_config
        cfg = load_task_config(CONFIG_PATH)
        targets = cfg.build_calibration_targets()
        slots_used = {t["labware_slot"] for t in targets}
        # Must include tiprack and reagent slots from config
        self.assertIn(cfg.tipracks["right"].slot, slots_used)
        self.assertIn(cfg.plate.slot, slots_used)

    def test_right_tiprack_pick_up(self):
        from src.tasks.color_mixing.config import load_task_config
        cfg = load_task_config(CONFIG_PATH)
        targets = cfg.build_calibration_targets()
        right_tip = next(
            t for t in targets
            if t["labware_slot"] == cfg.tipracks["right"].slot
            and t["action"] == "pick_up_tip"
        )
        self.assertEqual(right_tip["pipette_mount"], "right")


class TestBuildRunnerSteps(unittest.TestCase):
    """build_runner_steps produces runner-ready step dicts."""

    def _cfg(self):
        from src.tasks.color_mixing.config import load_task_config
        return load_task_config(CONFIG_PATH)

    def test_returns_list_of_dicts(self):
        from src.tasks.color_mixing.steps import build_runner_steps
        cfg = self._cfg()
        steps = build_runner_steps(cfg, np.array([80, 60, 60]), 0)
        self.assertIsInstance(steps, list)
        for s in steps:
            self.assertIn("name", s)
            self.assertIn("kind", s)
            self.assertIn("params", s)

    def test_skip_controls_reduces_steps(self):
        from src.tasks.color_mixing.steps import build_runner_steps
        cfg = self._cfg()
        full = build_runner_steps(cfg, np.array([80, 60, 60]), 0)
        skip = build_runner_steps(cfg, np.array([80, 60, 60]), 0, skip_controls=True)
        self.assertGreater(len(full), len(skip))

    def test_rinse_override_passed_through(self):
        from src.tasks.color_mixing.steps import build_runner_steps
        cfg = self._cfg()
        steps = build_runner_steps(cfg, np.array([80, 60, 60]), 0, rinse_cycles=2)
        transfers = [s for s in steps if s["kind"] == "transfer"]
        for t in transfers:
            self.assertEqual(t["params"].get("rinse_cycles"), 2)

    def test_no_orchestration_steps(self):
        """Runner steps must NOT include wait, capture, extract_rgb, fit_gp."""
        from src.tasks.color_mixing.steps import build_runner_steps
        cfg = self._cfg()
        steps = build_runner_steps(cfg, np.array([80, 60, 60]), 0)
        kinds = {s["kind"] for s in steps}
        for forbidden in ("wait", "capture", "extract_rgb", "fit_gp"):
            self.assertNotIn(forbidden, kinds)


class TestBuildCalibrationRunnerSteps(unittest.TestCase):
    """build_calibration_runner_steps produces calibration step dicts."""

    def test_returns_control_transfers(self):
        from src.tasks.color_mixing.config import load_task_config
        from src.tasks.color_mixing.steps import build_calibration_runner_steps
        cfg = load_task_config(CONFIG_PATH)
        steps = build_calibration_runner_steps(cfg)
        transfers = [s for s in steps if s["kind"] == "transfer"]
        self.assertEqual(len(transfers), len(cfg.well_roles.controls))

    def test_all_dest_wells_are_column_1(self):
        from src.tasks.color_mixing.config import load_task_config
        from src.tasks.color_mixing.steps import build_calibration_runner_steps
        cfg = load_task_config(CONFIG_PATH)
        steps = build_calibration_runner_steps(cfg)
        for s in steps:
            if s["kind"] == "transfer":
                self.assertTrue(s["params"]["dest_well"].endswith("1"))


class TestBuildTipCheckRunnerSteps(unittest.TestCase):
    """build_tip_check_runner_steps produces tip-check step dicts."""

    def test_ends_with_home(self):
        from src.tasks.color_mixing.config import load_task_config
        from src.tasks.color_mixing.steps import build_tip_check_runner_steps
        cfg = load_task_config(CONFIG_PATH)
        steps = build_tip_check_runner_steps(cfg)
        self.assertEqual(steps[-1]["kind"], "home")

    def test_has_pick_up_and_drop(self):
        from src.tasks.color_mixing.config import load_task_config
        from src.tasks.color_mixing.steps import build_tip_check_runner_steps
        cfg = load_task_config(CONFIG_PATH)
        steps = build_tip_check_runner_steps(cfg)
        kinds = [s["kind"] for s in steps]
        self.assertIn("pick_up_tip", kinds)
        self.assertIn("drop_tip", kinds)


class TestNoHardcodedSlots(unittest.TestCase):
    """Task module code must not contain hardcoded slot numbers."""

    def test_steps_module_no_slot_literals(self):
        """steps.py should not contain hardcoded slot assignments."""
        import re
        steps_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "tasks", "color_mixing", "steps.py",
        )
        with open(steps_path) as f:
            code = f.read()
        # Should not contain slot = "N" patterns (hardcoded deck positions)
        self.assertIsNone(
            re.search(r'slot\s*[=:]\s*["\']?\d+', code),
            "steps.py must not hardcode slot numbers",
        )

    def test_observation_module_no_slot_literals(self):
        obs_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "tasks", "color_mixing", "observation.py",
        )
        with open(obs_path) as f:
            code = f.read()
        self.assertNotIn('slot', code.lower())


class TestPipelineIsThinkWrapper(unittest.TestCase):
    """pipeline.py should not contain deck semantics."""

    def test_no_hardcoded_control_rows(self):
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "pipeline.py",
        )
        with open(pipeline_path) as f:
            code = f.read()
        # Should not define DEFAULT_CONTROL_ROWS or hardcode A=red, etc.
        self.assertNotIn("DEFAULT_CONTROL_ROWS", code)
        self.assertNotIn("DEFAULT_EXPERIMENT_ROWS", code)

    def test_no_labware_slot_loading(self):
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "pipeline.py",
        )
        with open(pipeline_path) as f:
            code = f.read()
        # Should not have _load_all_labware or direct slot references
        self.assertNotIn("_load_all_labware", code)
        self.assertNotIn("red_source_id", code)
        self.assertNotIn("green_source_id", code)
        self.assertNotIn("blue_source_id", code)

    def test_imports_task_api(self):
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "pipeline.py",
        )
        with open(pipeline_path) as f:
            code = f.read()
        self.assertIn("from src.tasks.color_mixing", code)
        self.assertIn("build_iteration_steps", code)
        self.assertIn("analyze_capture", code)
        self.assertIn("build_robot_calibration_profile", code)


class TestColorMixingPlugin(unittest.TestCase):
    """ColorMixingPlugin must implement the TaskPlugin protocol."""

    def _plugin(self):
        from src.tasks.color_mixing.plugin import ColorMixingPlugin
        return ColorMixingPlugin()

    def _cfg(self):
        return self._plugin().load_config(CONFIG_PATH)

    def test_satisfies_task_plugin_protocol(self):
        from openot2.task_api.plugin import TaskPlugin
        plugin = self._plugin()
        self.assertIsInstance(plugin, TaskPlugin)

    def test_name(self):
        self.assertEqual(self._plugin().name, "color_mixing")

    def test_load_config(self):
        cfg = self._cfg()
        self.assertIsInstance(cfg, object)
        self.assertEqual(cfg.total_volume_ul, 200.0)

    def test_build_deck_config(self):
        plugin = self._plugin()
        cfg = self._cfg()
        deck = plugin.build_deck_config(cfg)
        self.assertIn("pipettes", deck)
        self.assertIn("labware", deck)

    def test_initial_state_full(self):
        plugin = self._plugin()
        cfg = self._cfg()
        state = plugin.initial_state(cfg, "full")
        self.assertEqual(state["phase"], "random")
        self.assertEqual(state["iteration"], 0)
        self.assertFalse(state["terminal"])
        self.assertIn("cache", state)
        self.assertIn("all_X", state["cache"])

    def test_initial_state_quick(self):
        plugin = self._plugin()
        cfg = self._cfg()
        state = plugin.initial_state(cfg, "quick")
        self.assertEqual(state["cache"]["rinse_cycles"], 1)
        self.assertEqual(state["cache"]["mix_cycles"], 2)
        self.assertTrue(state["cache"]["skip_controls_after_first"])

    def test_build_plan(self):
        plugin = self._plugin()
        cfg = self._cfg()
        state = plugin.initial_state(cfg, "full")
        plan = plugin.build_plan(cfg, state, "full")
        self.assertEqual(plan["n_initial"], cfg.experiment.n_initial)
        self.assertEqual(plan["n_optimization"], cfg.experiment.n_optimization)
        self.assertEqual(plan["total_iterations"],
                         cfg.experiment.n_initial + cfg.experiment.n_optimization)

    def test_build_calibration_targets(self):
        plugin = self._plugin()
        cfg = self._cfg()
        targets = plugin.build_calibration_targets(cfg)
        self.assertIsInstance(targets, list)
        self.assertGreater(len(targets), 0)

    def test_status_payload(self):
        plugin = self._plugin()
        cfg = self._cfg()
        state = plugin.initial_state(cfg, "full")
        payload = plugin.status_payload(cfg, state)
        self.assertEqual(payload["task"], "color_mixing")
        self.assertEqual(payload["phase"], "random")
        self.assertFalse(payload["terminal"])

    def test_web_extension_returns_extension(self):
        plugin = self._plugin()
        cfg = self._cfg()
        ext = plugin.web_extension(cfg)
        self.assertIsNotNone(ext)
        self.assertTrue(hasattr(ext, "extra_routes"))
        self.assertTrue(hasattr(ext, "extra_status"))
        self.assertTrue(hasattr(ext, "ui_payload"))


class TestBuildIterationRun(unittest.TestCase):
    """build_iteration_run should produce a full TaskRun with $ref bindings."""

    def _plugin_and_state(self, mode="full"):
        from src.tasks.color_mixing.plugin import ColorMixingPlugin
        plugin = ColorMixingPlugin()
        cfg = plugin.load_config(CONFIG_PATH)
        state = plugin.initial_state(cfg, mode)
        state["_next_volumes"] = [80.0, 60.0, 60.0]
        return plugin, cfg, state

    def test_returns_task_run(self):
        from openot2.control.models import TaskRun
        plugin, cfg, state = self._plugin_and_state()
        run = plugin.build_iteration_run(cfg, state, 0, "full")
        self.assertIsInstance(run, TaskRun)

    def test_includes_liquid_handling_steps(self):
        plugin, cfg, state = self._plugin_and_state()
        run = plugin.build_iteration_run(cfg, state, 0, "full")
        kinds = [s.kind for s in run.steps]
        self.assertIn("transfer", kinds)
        self.assertIn("mix", kinds)

    def test_includes_orchestration_steps(self):
        plugin, cfg, state = self._plugin_and_state()
        run = plugin.build_iteration_run(cfg, state, 0, "full")
        kinds = [s.kind for s in run.steps]
        self.assertIn("wait", kinds)
        self.assertIn("capture", kinds)
        self.assertIn("extract_rgb", kinds)
        self.assertIn("fit_gp", kinds)

    def test_capture_has_key(self):
        plugin, cfg, state = self._plugin_and_state()
        run = plugin.build_iteration_run(cfg, state, 0, "full")
        capture_step = next(s for s in run.steps if s.kind == "capture")
        self.assertEqual(capture_step.key, "capture")

    def test_extract_rgb_uses_ref_to_capture(self):
        plugin, cfg, state = self._plugin_and_state()
        run = plugin.build_iteration_run(cfg, state, 0, "full")
        extract_step = next(s for s in run.steps if s.kind == "extract_rgb")
        self.assertEqual(
            extract_step.params["image_path"],
            {"$ref": "capture.output.image_path"},
        )

    def test_fit_gp_uses_ref_to_extract(self):
        plugin, cfg, state = self._plugin_and_state()
        run = plugin.build_iteration_run(cfg, state, 0, "full")
        fit_step = next(s for s in run.steps if s.kind == "fit_gp")
        self.assertEqual(
            fit_step.params["observed_rgb"],
            {"$ref": "extract_rgb.output.mean_rgb"},
        )

    def test_metadata_includes_iteration(self):
        plugin, cfg, state = self._plugin_and_state()
        run = plugin.build_iteration_run(cfg, state, 3, "full")
        self.assertEqual(run.metadata["iteration"], 3)
        self.assertEqual(run.metadata["volumes"], [80.0, 60.0, 60.0])

    def test_quick_mode_skips_controls_after_first(self):
        plugin, cfg, state = self._plugin_and_state("quick")
        run0 = plugin.build_iteration_run(cfg, state, 0, "quick")
        run1 = plugin.build_iteration_run(cfg, state, 1, "quick")
        # First iteration should not skip controls
        self.assertFalse(run0.metadata["skip_controls"])
        # Second iteration should skip controls
        self.assertTrue(run1.metadata["skip_controls"])


class TestBuildCalibrationRun(unittest.TestCase):
    """build_calibration_run should produce a TaskRun with controls + capture."""

    def test_returns_task_run(self):
        from src.tasks.color_mixing.plugin import ColorMixingPlugin
        from openot2.control.models import TaskRun
        plugin = ColorMixingPlugin()
        cfg = plugin.load_config(CONFIG_PATH)
        state = plugin.initial_state(cfg, "full")
        run = plugin.build_calibration_run(cfg, state)
        self.assertIsInstance(run, TaskRun)

    def test_includes_control_transfers(self):
        from src.tasks.color_mixing.plugin import ColorMixingPlugin
        plugin = ColorMixingPlugin()
        cfg = plugin.load_config(CONFIG_PATH)
        state = plugin.initial_state(cfg, "full")
        run = plugin.build_calibration_run(cfg, state)
        transfers = [s for s in run.steps if s.kind == "transfer"]
        self.assertEqual(len(transfers), len(cfg.well_roles.controls))

    def test_includes_capture_and_extract(self):
        from src.tasks.color_mixing.plugin import ColorMixingPlugin
        plugin = ColorMixingPlugin()
        cfg = plugin.load_config(CONFIG_PATH)
        state = plugin.initial_state(cfg, "full")
        run = plugin.build_calibration_run(cfg, state)
        kinds = [s.kind for s in run.steps]
        self.assertIn("capture", kinds)
        self.assertIn("extract_rgb", kinds)

    def test_extract_rgb_refs_capture(self):
        from src.tasks.color_mixing.plugin import ColorMixingPlugin
        plugin = ColorMixingPlugin()
        cfg = plugin.load_config(CONFIG_PATH)
        state = plugin.initial_state(cfg, "full")
        run = plugin.build_calibration_run(cfg, state)
        extract_step = next(s for s in run.steps if s.kind == "extract_rgb")
        self.assertEqual(
            extract_step.params["image_path"],
            {"$ref": "capture.output.image_path"},
        )


class TestBuildTipCheckRun(unittest.TestCase):
    """build_tip_check_run should produce a TaskRun."""

    def test_returns_task_run(self):
        from src.tasks.color_mixing.plugin import ColorMixingPlugin
        from openot2.control.models import TaskRun
        plugin = ColorMixingPlugin()
        cfg = plugin.load_config(CONFIG_PATH)
        state = plugin.initial_state(cfg, "full")
        run = plugin.build_tip_check_run(cfg, state)
        self.assertIsInstance(run, TaskRun)

    def test_includes_pick_up_and_drop(self):
        from src.tasks.color_mixing.plugin import ColorMixingPlugin
        plugin = ColorMixingPlugin()
        cfg = plugin.load_config(CONFIG_PATH)
        state = plugin.initial_state(cfg, "full")
        run = plugin.build_tip_check_run(cfg, state)
        kinds = [s.kind for s in run.steps]
        self.assertIn("pick_up_tip", kinds)
        self.assertIn("drop_tip", kinds)

    def test_ends_with_home(self):
        from src.tasks.color_mixing.plugin import ColorMixingPlugin
        plugin = ColorMixingPlugin()
        cfg = plugin.load_config(CONFIG_PATH)
        state = plugin.initial_state(cfg, "full")
        run = plugin.build_tip_check_run(cfg, state)
        self.assertEqual(run.steps[-1].kind, "home")


class TestApplyRunResult(unittest.TestCase):
    """apply_run_result should fold results into state."""

    def _make_completed_run(self):
        from openot2.control.models import TaskRun, RunStep, RunStatus, StepStatus
        steps = [
            RunStep(name="transfer", kind="transfer", status=StepStatus.succeeded),
            RunStep(name="capture", kind="capture", key="capture",
                    status=StepStatus.succeeded,
                    output={"image_path": "/tmp/test.jpg"}),
            RunStep(name="extract", kind="extract_rgb", key="extract_rgb",
                    status=StepStatus.succeeded,
                    output={
                        "mean_rgb": [100.0, 50.0, 200.0],
                        "used_calibration": True,
                        "runtime_reference": {"roles": {"A1": "red"}},
                    }),
            RunStep(name="fit", kind="fit_gp", key="fit_gp",
                    status=StepStatus.succeeded,
                    output={
                        "distance": 25.0,
                        "converged": False,
                        "best_distance": 25.0,
                        "best_iteration": 1,
                        "all_X": [[80, 60, 60]],
                        "all_Y": [[100, 50, 200]],
                        "all_dist": [25.0],
                    }),
        ]
        return TaskRun(
            name="test run",
            status=RunStatus.completed,
            steps=steps,
            metadata={"iteration": 0, "phase": "random",
                      "col_idx": 0, "volumes": [80, 60, 60]},
        )

    def test_increments_iteration(self):
        from src.tasks.color_mixing.plugin import ColorMixingPlugin
        plugin = ColorMixingPlugin()
        cfg = plugin.load_config(CONFIG_PATH)
        state = plugin.initial_state(cfg, "full")
        run = self._make_completed_run()
        new_state = plugin.apply_run_result(cfg, state, run, "full")
        self.assertEqual(new_state["iteration"], 1)

    def test_appends_history(self):
        from src.tasks.color_mixing.plugin import ColorMixingPlugin
        plugin = ColorMixingPlugin()
        cfg = plugin.load_config(CONFIG_PATH)
        state = plugin.initial_state(cfg, "full")
        run = self._make_completed_run()
        new_state = plugin.apply_run_result(cfg, state, run, "full")
        self.assertEqual(len(new_state["history"]), 1)
        self.assertEqual(new_state["history"][0]["distance"], 25.0)

    def test_updates_metrics(self):
        from src.tasks.color_mixing.plugin import ColorMixingPlugin
        plugin = ColorMixingPlugin()
        cfg = plugin.load_config(CONFIG_PATH)
        state = plugin.initial_state(cfg, "full")
        run = self._make_completed_run()
        new_state = plugin.apply_run_result(cfg, state, run, "full")
        self.assertEqual(new_state["metrics"]["best_distance"], 25.0)

    def test_caches_reference(self):
        from src.tasks.color_mixing.plugin import ColorMixingPlugin
        plugin = ColorMixingPlugin()
        cfg = plugin.load_config(CONFIG_PATH)
        state = plugin.initial_state(cfg, "full")
        run = self._make_completed_run()
        new_state = plugin.apply_run_result(cfg, state, run, "full")
        self.assertIsNotNone(new_state["cache"]["cached_reference"])

    def test_convergence_sets_terminal(self):
        from src.tasks.color_mixing.plugin import ColorMixingPlugin
        from openot2.control.models import TaskRun, RunStep, RunStatus, StepStatus
        plugin = ColorMixingPlugin()
        cfg = plugin.load_config(CONFIG_PATH)
        state = plugin.initial_state(cfg, "full")
        steps = [
            RunStep(name="fit", kind="fit_gp", key="fit_gp",
                    status=StepStatus.succeeded,
                    output={
                        "distance": 10.0,
                        "converged": True,
                        "all_X": [], "all_Y": [], "all_dist": [],
                    }),
        ]
        run = TaskRun(name="converged", status=RunStatus.completed,
                      steps=steps, metadata={"iteration": 5, "phase": "bo",
                                              "volumes": [80, 60, 60]})
        new_state = plugin.apply_run_result(cfg, state, run, "full")
        self.assertTrue(new_state["terminal"])
        self.assertEqual(new_state["phase"], "done")


class TestSingleSourceOfTruth(unittest.TestCase):
    """Runner dicts must be derived from Step objects — not independent."""

    def _cfg(self):
        from src.tasks.color_mixing.config import load_task_config
        return load_task_config(CONFIG_PATH)

    def test_runner_steps_derived_from_iteration_steps(self):
        """build_runner_steps must produce the same transfers as
        steps_to_runner_dicts(build_iteration_steps(...))."""
        from src.tasks.color_mixing.steps import (
            build_iteration_steps, build_runner_steps, steps_to_runner_dicts,
        )
        cfg = self._cfg()
        volumes = np.array([80.0, 60.0, 60.0])

        # Direct call
        runner_direct = build_runner_steps(cfg, volumes, 0)

        # Manual: canonical → adapter
        canonical = build_iteration_steps(cfg, volumes, 0)
        runner_derived = steps_to_runner_dicts(canonical, cfg)

        # Must produce identical output
        self.assertEqual(len(runner_direct), len(runner_derived))
        for d, r in zip(runner_direct, runner_derived):
            self.assertEqual(d["kind"], r["kind"])
            self.assertEqual(d["params"], r["params"])

    def test_calibration_steps_derived_from_control_steps(self):
        """build_calibration_runner_steps must be derived from
        build_control_steps."""
        from src.tasks.color_mixing.steps import (
            build_control_steps, build_calibration_runner_steps,
            steps_to_runner_dicts,
        )
        cfg = self._cfg()

        runner_direct = build_calibration_runner_steps(cfg)
        canonical = build_control_steps(cfg)
        runner_derived = steps_to_runner_dicts(canonical, cfg)

        self.assertEqual(len(runner_direct), len(runner_derived))
        for d, r in zip(runner_direct, runner_derived):
            self.assertEqual(d["kind"], r["kind"])

    def test_tip_check_steps_derived(self):
        """build_tip_check_runner_steps must be derived from
        build_tip_check_steps."""
        from src.tasks.color_mixing.steps import (
            build_tip_check_steps, build_tip_check_runner_steps,
            steps_to_runner_dicts,
        )
        cfg = self._cfg()

        runner_direct = build_tip_check_runner_steps(cfg)
        canonical = build_tip_check_steps(cfg)
        runner_derived = steps_to_runner_dicts(canonical, cfg)

        self.assertEqual(len(runner_direct), len(runner_derived))
        for d, r in zip(runner_direct, runner_derived):
            self.assertEqual(d["kind"], r["kind"])


if __name__ == "__main__":
    unittest.main()
