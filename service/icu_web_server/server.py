import os
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from agent_engine.agent_logger import AgentLogger

from core.icu_web.frontend_bridge import FrontendBridge
from core.icu_web.session_runner import run_session_loop


logger = AgentLogger(__name__)

# Project root is two levels up: service/icu_web_server/server.py -> project root
ROOT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT_DIR / "web" / "icu_agents"
ICU_DB_DIR = ROOT_DIR / "database" / "icu_raw"


app = FastAPI()


# API -----------------------------------------------------------------------
@app.get("/api/patients")
async def list_patients() -> JSONResponse:
    """Return patient IDs inferred from database/icu_raw/*.json files."""
    if not ICU_DB_DIR.exists():
        logger.warning("ICU database folder not found")
        return JSONResponse({"patients": []})
    ids: List[str] = []
    for p in sorted(ICU_DB_DIR.glob("*.json")):
        ids.append(p.stem)
    return JSONResponse({"patients": ids})


@app.post("/api/connect_patient")
async def connect_patient(payload: dict) -> JSONResponse:
    session_id: Optional[str] = payload.get("session_id")
    patient_id: Optional[str] = payload.get("patient_id")
    if not session_id or not patient_id:
        raise HTTPException(status_code=400, detail="session_id and patient_id are required")

    json_path = ICU_DB_DIR / f"{patient_id}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="patient not found")

    try:
        from agents.ICUDataIngestionAgent.agent import ICUDataIngestionAgent
    except Exception as e:  # pragma: no cover
        logger.error(f"Failed to import ICUDataIngestionAgent: {e}")
        raise HTTPException(status_code=500, detail="ICUDataIngestionAgent not available")

    ingestion = ICUDataIngestionAgent()
    # Load patient file
    ingestion.load_patient(str(json_path))

    # Prepare ICUMemoryAgent per session
    try:
        from agents.ICUMemoryAgent.agent import ICUMemoryAgent
    except Exception as e:  # pragma: no cover
        logger.error(f"Failed to import ICUMemoryAgent: {e}")
        raise HTTPException(status_code=500, detail="ICUMemoryAgent not available")

    memory = ICUMemoryAgent()

    bridge = FrontendBridge.get_instance()
    bridge.set_ingestion(session_id, ingestion)
    bridge.set_memory(session_id, memory)
    logger.info(f"Patient connected: session={session_id} patient={patient_id}")
    return JSONResponse({"ok": True, "patient_id": patient_id})


@app.post("/api/ingestion/update")
async def ingestion_update(payload: dict) -> JSONResponse:
    session_id: Optional[str] = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    bridge = FrontendBridge.get_instance()
    ingestion = bridge.get_ingestion(session_id)
    if ingestion is None:
        raise HTTPException(status_code=400, detail="ingestion is not initialized")

    try:
        batch = await ingestion.update()
    except Exception as e:  # pragma: no cover
        logger.error(f"ingestion.update failed: {e}")
        raise HTTPException(status_code=500, detail="ingestion update failed")

    written_count = len(batch or [])

    # Persist batch into memory if available
    memory = bridge.get_memory(session_id)
    if memory is not None and written_count > 0:
        patient_id = getattr(ingestion, 'patient_id', None)
        if patient_id:
            try:
                ids = await memory.add_events(patient_id, batch)
                written_count = len(ids or [])
            except Exception as e:  # pragma: no cover
                logger.error(f"memory.add_events failed: {e}")

    return JSONResponse({"ok": True, "written_count": written_count})


# WebSocket -----------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Expect session_id in query param
    session_id: Optional[str] = websocket.query_params.get("session_id")
    if not session_id:
        await websocket.send_text(json.dumps({"type": "error", "message": "session_id is required"}))
        await websocket.close()
        return

    bridge = FrontendBridge.get_instance()
    await bridge.register_websocket(session_id, websocket)
    # Ensure a background session loop is running per session
    loop_flag = f"_session_loop_{session_id}"
    if not hasattr(app.state, loop_flag):
        app.state.__dict__[loop_flag] = True
        import asyncio
        asyncio.create_task(run_session_loop(session_id))
    try:
        while True:
            data_text = await websocket.receive_text()
            try:
                data = json.loads(data_text)
            except Exception:
                continue
            msg_type = data.get("type")
            if msg_type == "user_message":
                text = data.get("text", "")
                if text:
                    await bridge.enqueue_user_input(session_id, text)
            # Reserved for future message types (e.g., typing, cancel)
    except WebSocketDisconnect:
        pass
    finally:
        await bridge.unregister_websocket(session_id, websocket)


# Static files (UI) ----------------------------------------------------------
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")
else:  # pragma: no cover
    logger.warning("UI directory not found; static hosting disabled")


def main() -> None:  # pragma: no cover
    import uvicorn
    uvicorn.run("service.icu_web_server.server:app", host="0.0.0.0", port=8080, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()


