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

    return app
