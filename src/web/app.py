"""FastAPI sub-application for the task protocol page.

Generic integration layer — task-specific semantics come from the
:class:`~openot2.task_api.plugin.TaskPlugin` and its
:class:`~openot2.task_api.plugin.TaskWebExtension`.

This module contains only framework-level routes (start, stop, pause,
resume, reset, status, plan, calibrate, home, robot-calibration-profile).
Task-specific routes (gamut, set-target, …) are provided by the
plugin's web extension and mounted automatically.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates


_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_OT2_TEMPLATES_DIR = (
    Path(__file__).resolve().parent.parent.parent / "OpenOT2" / "openot2" / "webapp" / "templates"
)


def create_protocol_app(loop_manager, nav_links: list = None) -> FastAPI:
    """Create the protocol monitoring FastAPI sub-app.

    The loop manager provides all task interaction.  The plugin attached
    to the loop is used for calibration targets and status payloads.
    Task-specific routes are mounted from the plugin's web extension.
    """

    app = FastAPI(title="Protocol")
    _nav = nav_links

    templates = Jinja2Templates(
        directory=[str(_TEMPLATES_DIR), str(_OT2_TEMPLATES_DIR)]
    )

    def _get_nav():
        return _nav or []

    def _build_robot_calibration_profile() -> dict:
        """Build a calibration profile from the plugin + config."""
        plugin = loop_manager.plugin
        cfg = loop_manager.task_cfg
        targets = plugin.build_calibration_targets(cfg)
        return {"name": f"{plugin.name}_robot_calibration", "targets": targets}

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
        """Build a calibration profile from the plugin and config."""
        return _build_robot_calibration_profile()

    @app.post("/calibrate")
    async def calibrate():
        """Run calibration via the loop manager."""
        result = loop_manager.calibrate()
        return result

    # ------------------------------------------------------------------
    # Mount task web extension routes (e.g. /gamut, /set-target)
    # ------------------------------------------------------------------
    ext = loop_manager.plugin.web_extension(loop_manager._config)
    if ext is not None:
        # Give the extension access to the loop for state management
        if hasattr(ext, "bind"):
            ext.bind(loop_manager)

        extra = ext.extra_routes()
        if isinstance(extra, (list, tuple)):
            for method, path, handler in extra:
                app.add_api_route(path, handler, methods=[method.upper()])

    return app
