#!/usr/bin/env python3
"""Launch the automation_lab web server.

Starts the OpenOT2 web dashboard with the color mixing protocol
page mounted at ``/protocol/``.

Usage::

    python scripts/server.py
    python scripts/server.py --port 8000
    python scripts/server.py --no-robot          # UI-only mode for development
    python scripts/server.py --config configs/experiment.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Path setup — allow running from the project root without ``pip install -e``
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "OpenOT2"))

# ---------------------------------------------------------------------------

logger = logging.getLogger("automation_lab.server")


def _build_deck_config(config: dict):
    """Translate *experiment.yaml* into an OpenOT2 ``DeckConfig``.

    Returns ``None`` when the config does not contain enough info.
    """
    from webapp.deck import DeckConfig  # OpenOT2 package

    labware_section = config.get("labware", {})
    pipette_section = config.get("pipettes", {})

    pipettes: dict[str, str] = {}
    for _key, pinfo in pipette_section.items():
        mount = pinfo.get("mount", _key)
        pipettes[mount] = pinfo["name"]

    labware: dict[str, str] = {}
    # Tipracks
    for tip in labware_section.get("tipracks", []):
        labware[str(tip["slot"])] = tip["name"]
    # Source reservoirs (red, green, blue)
    for _colour, src in labware_section.get("sources", {}).items():
        labware[str(src["slot"])] = src["name"]
    # Water reservoir
    water = labware_section.get("water", {})
    if water:
        labware[str(water["slot"])] = water["name"]
    # Cleaning reservoir
    cleaning = labware_section.get("cleaning", {})
    if cleaning:
        labware[str(cleaning["slot"])] = cleaning["name"]
    # Dispense plate (may also appear as "plate")
    for key in ("dispense", "plate"):
        section = labware_section.get(key, {})
        if section and str(section.get("slot", "")) not in labware:
            labware[str(section["slot"])] = section["name"]

    return DeckConfig(pipettes=pipettes, labware=labware)


def _autoload_robot_calibration_profile(ot2_app, config_path: Path) -> None:
    """Load the saved robot calibration profile from configs, if present."""
    from webapp.calibration import CalibrationSession, load_profile

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
        default="configs/experiment.yaml",
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

    # ---- Load experiment config ----
    config_path = ROOT / args.config
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    # ---- Build OpenOT2 WebApp ----
    from webapp import WebApp  # OpenOT2 package

    robot_ip = None if args.no_robot else config["robot"]["ip"]
    deck = _build_deck_config(config)

    clean_cfg = config.get("cleaning", {})
    ot2_app = WebApp(
        deck=deck,
        robot_ip=robot_ip,
        rinse_cycles=clean_cfg.get("rinse_cycles", 3),
        rinse_volume=float(clean_cfg.get("rinse_volume_ul", 200)),
    )
    _autoload_robot_calibration_profile(ot2_app, config_path)

    # ---- Register custom step handlers ----
    from src.web.handlers import (
        handle_extract_rgb,
        handle_fit_gp,
        handle_suggest_volumes,
    )

    ot2_app.register_handler("extract_rgb", handle_extract_rgb)
    ot2_app.register_handler("suggest_volumes", handle_suggest_volumes)
    ot2_app.register_handler("fit_gp", handle_fit_gp)

    # ---- Create the active-learning loop manager ----
    from src.web.loop import ActiveLearningLoop

    loop_manager = ActiveLearningLoop(
        runner=ot2_app.runner,
        store=ot2_app.store,
        config_path=str(config_path),
    )
    ot2_app.handlers.progress_callback = loop_manager.update_live_progress

    # ---- Mount protocol sub-app ----
    from src.web.app import create_protocol_app

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
