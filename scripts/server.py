#!/usr/bin/env python3
"""Launch the automation_lab web server.

Plugin-driven assembly: loads a task plugin, derives config and deck
from it, then mounts the generic OpenOT2 web shell with the protocol
sub-app.

Usage::

    python -m scripts.server
    python -m scripts.server --port 8000
    python -m scripts.server --no-robot          # UI-only mode for development
    python -m scripts.server --config configs/color_mixing.yaml
"""

import argparse
import logging
from pathlib import Path

from openot2.webapp import WebApp
from openot2.webapp.deck import DeckConfig

from src.tasks.color_mixing.plugin import ColorMixingPlugin
from src.web.handlers import (
    handle_extract_rgb,
    handle_fit_gp,
    handle_suggest_volumes,
)
from src.web.loop import ActiveLearningLoop
from src.web.app import create_protocol_app

logger = logging.getLogger("automation_lab.server")


def _autoload_robot_calibration_profile(ot2_app, config_path: Path) -> None:
    """Load the saved robot calibration profile from configs, if present."""
    from openot2.webapp.calibration import CalibrationSession, load_profile

    profile_path = config_path.parent / "calibrations" / "robot_calibration" / "robot_profile.json"
    if not profile_path.exists():
        return

    try:
        profile = load_profile(profile_path)
        session = CalibrationSession(profile_id=profile.id, status="active")
        ot2_app.set_calibration_profile(profile, session=session)
        logger.info("Loaded robot calibration profile from %s", profile_path)
    except Exception as exc:
        logger.warning("Could not load robot calibration profile from %s: %s", profile_path, exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="automation_lab web server",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--config",
        default="configs/color_mixing.yaml",
        help="Path to experiment YAML (relative to project root)",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="UI-only mode — no robot connection",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
    )

    # ---- Load plugin ----
    plugin = ColorMixingPlugin()

    # ---- Load config through plugin ----
    root = Path(__file__).resolve().parent.parent
    config_path = root / args.config
    cfg = plugin.load_config(str(config_path))

    # ---- Build deck through plugin ----
    robot_ip = None if args.no_robot else cfg.robot_ip
    deck_dict = plugin.build_deck_config(cfg)
    deck = DeckConfig(pipettes=deck_dict["pipettes"], labware=deck_dict["labware"])

    # ---- Build OpenOT2 WebApp (generic shell) ----
    ot2_app = WebApp(
        deck=deck,
        robot_ip=robot_ip,
        plugin=plugin,
        config_path=str(config_path),
        rinse_cycles=cfg.cleaning_protocol.rinse_cycles,
        rinse_volume=float(cfg.cleaning_protocol.rinse_volume_ul),
    )
    _autoload_robot_calibration_profile(ot2_app, Path(str(config_path)))

    # ---- Register task step handlers ----
    ot2_app.register_handler("extract_rgb", handle_extract_rgb)
    ot2_app.register_handler("suggest_volumes", handle_suggest_volumes)
    ot2_app.register_handler("fit_gp", handle_fit_gp)

    # ---- Create the plugin-driven active-learning loop ----
    loop_manager = ActiveLearningLoop(
        runner=ot2_app.runner,
        store=ot2_app.store,
        plugin=plugin,
        config=cfg,
        config_path=str(config_path),
    )
    ot2_app.handlers.progress_callback = loop_manager.update_live_progress

    # ---- Mount protocol sub-app ----
    nav_link = {"title": "Protocol", "path": "/protocol/", "icon": "&#9883;"}
    ot2_app.add_nav_link(**nav_link)
    # Pass the same nav_links to the sub-app so it renders the sidebar correctly
    protocol_app = create_protocol_app(loop_manager, nav_links=ot2_app._nav_links)
    ot2_app.mount_app("/protocol", protocol_app)

    # ---- Serve ----
    logger.info(
        "Starting automation_lab server on http://%s:%d", args.host, args.port
    )
    logger.info("Protocol dashboard at http://localhost:%d/protocol/", args.port)
    ot2_app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
