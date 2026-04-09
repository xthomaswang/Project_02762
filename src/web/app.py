"""FastAPI sub-application for the color mixing protocol page."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_OT2_TEMPLATES_DIR = (
    Path(__file__).resolve().parent.parent.parent / "OpenOT2" / "webapp" / "templates"
)


def create_protocol_app(loop_manager, nav_links: list = None) -> FastAPI:
    """Create the protocol monitoring FastAPI sub-app."""

    app = FastAPI(title="Color Mixing Protocol")
    _nav = nav_links

    templates = Jinja2Templates(
        directory=[str(_TEMPLATES_DIR), str(_OT2_TEMPLATES_DIR)]
    )

    def _get_nav():
        return _nav or []

    def _build_robot_calibration_profile() -> dict:
        cfg = loop_manager.config
        metadata = cfg.get("metadata", {})
        labware = cfg.get("labware", {})
        pipettes = cfg.get("pipettes", {})
        total_volume = float(cfg.get("total_volume_ul", 200))
        mix_volume = float(cfg.get("experiment", {}).get("mix_volume_ul", total_volume))

        tipracks = labware.get("tipracks", [])
        sources = labware.get("sources", {})
        water = labware.get("water", {})
        cleaning = labware.get("cleaning", {})
        cleaning_cols = cleaning.get("columns", {})
        plate = labware.get("dispense") or labware.get("plate") or {}

        def _tiprack_for_mount(mount: str) -> dict | None:
            for tiprack in tipracks:
                if str(tiprack.get("for") or tiprack.get("mount") or "") == mount:
                    return tiprack
            return None

        def _tip_well_for_mount(tiprack: dict, mount: str) -> str:
            if mount == "left":
                control_tips = tiprack.get("control_tips", {})
                if control_tips:
                    return next(iter(control_tips.values()))
                return "A1"

            tip_columns = tiprack.get("tip_columns", {})
            if tip_columns:
                first_col = next(iter(tip_columns.values()))
                if str(first_col).isdigit():
                    return f"A{first_col}"
            return "A1"

        targets: list[dict] = []

        def add_target(
            *,
            name: str,
            pipette_mount: str,
            labware_slot: str,
            well: str,
            action: str,
            volume: float | None = None,
        ) -> None:
            target = {
                "name": name,
                "pipette_mount": pipette_mount,
                "labware_slot": str(labware_slot),
                "well": well,
                "action": action,
            }
            if volume is not None:
                target["volume"] = float(volume)
            targets.append(target)

        for mount in ("left", "right"):
            if mount not in pipettes:
                continue

            tiprack = _tiprack_for_mount(mount)
            if tiprack:
                add_target(
                    name=f"{mount.title()} tiprack (slot {tiprack['slot']})",
                    pipette_mount=mount,
                    labware_slot=tiprack["slot"],
                    well=_tip_well_for_mount(tiprack, mount),
                    action="pick_up_tip",
                )

            source_volume = total_volume if mount == "left" else mix_volume
            for color in ("red", "green", "blue"):
                src = sources.get(color)
                if not src:
                    continue
                add_target(
                    name=f"{color.title()} reservoir ({mount})",
                    pipette_mount=mount,
                    labware_slot=src["slot"],
                    well=src.get("well", "A1"),
                    action="aspirate",
                    volume=source_volume,
                )

            if mount == "left" and water:
                add_target(
                    name="Water reservoir (left)",
                    pipette_mount="left",
                    labware_slot=water["slot"],
                    well=water.get("well", "A1"),
                    action="aspirate",
                    volume=total_volume,
                )

            if plate:
                add_target(
                    name=f"Plate dispense ({mount})",
                    pipette_mount=mount,
                    labware_slot=plate["slot"],
                    well="A1",
                    action="dispense",
                    volume=total_volume if mount == "left" else mix_volume,
                )

            if cleaning:
                if mount == "left":
                    rinse_well = cleaning_cols.get("control_rinse") or cleaning_cols.get("water_rinse") or "A1"
                else:
                    rinse_well = cleaning_cols.get("mix_rinse") or cleaning_cols.get("red_rinse") or "A1"
                add_target(
                    name=f"Cleaning reservoir ({mount})",
                    pipette_mount=mount,
                    labware_slot=cleaning["slot"],
                    well=rinse_well,
                    action="aspirate",
                    volume=source_volume,
                )

        profile_name = f"{metadata.get('name', 'protocol')}_robot_calibration"
        return {"name": profile_name, "targets": targets}

    @app.get("/")
    async def protocol_page(request: Request):
        status = loop_manager.status()
        return templates.TemplateResponse(
            request, "protocol.html",
            context={"status": status, "extra_nav": _get_nav()},
        )

    @app.post("/start")
    async def start_loop(request: Request):
        body = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
        mode = body.get("mode", "quick")
        try:
            seq_id = loop_manager.start(mode=mode)
            return {"status": "started", "sequence_id": seq_id}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/pause")
    async def pause_loop():
        loop_manager.pause()
        return {"status": "pause_requested"}

    @app.post("/resume")
    async def resume_loop():
        loop_manager.resume()
        return {"status": "resumed"}

    @app.post("/stop")
    async def stop_loop():
        loop_manager.stop()
        return {"status": "stop_requested"}

    @app.post("/home")
    async def home_robot():
        result = loop_manager.home()
        if not result.get("ok", False):
            raise HTTPException(status_code=409, detail=result.get("error", "Home failed"))
        return result

    @app.post("/reset")
    async def reset_loop():
        result = loop_manager.reset()
        if not result.get("ok", False):
            raise HTTPException(status_code=409, detail=result.get("error", "Reset failed"))
        return result

    @app.get("/status")
    async def loop_status():
        return loop_manager.status()

    @app.get("/plan")
    async def get_plan(mode: str = "quick"):
        return loop_manager.get_plan(mode)

    @app.get("/robot-calibration-profile")
    async def robot_calibration_profile():
        """Build a calibration profile from the active protocol config."""
        return _build_robot_calibration_profile()

    @app.post("/calibrate")
    async def calibrate():
        """Run calibration column to measure pure dye colors and compute gamut."""
        result = loop_manager.calibrate()
        return result



    @app.get("/gamut")
    async def get_gamut():
        """Return the computed gamut including all samples for the color picker."""
        s = loop_manager.status()
        # Include full samples_rgb for the canvas color picker
        gamut = loop_manager._gamut or {}
        return {
            "calibration_done": s.get("calibration_done", False),
            "pure_rgbs": s.get("pure_rgbs"),
            "water_rgb": s.get("water_rgb"),
            "suggested_targets": s.get("gamut_suggested_targets"),
            "samples_rgb": gamut.get("samples_rgb", []),
            "custom_target": s.get("custom_target"),
        }

    @app.post("/set-target")
    async def set_target(request: Request):
        """Set the target RGB from the gamut picker."""
        body = await request.json()
        rgb = body.get("rgb")
        if not rgb or len(rgb) != 3:
            return {"ok": False, "error": "Must provide rgb: [R, G, B]"}
        result = loop_manager.set_target(rgb)
        return result

    return app
