import asyncio
import json
import uuid
from typing import Any, Dict, Optional, Set

from agent_engine.agent_logger import AgentLogger

try:
    from starlette.websockets import WebSocket
except Exception:  # pragma: no cover
    WebSocket = Any  # type: ignore


logger = AgentLogger(__name__)


class _SessionState:
    def __init__(self) -> None:
        self.websockets: Set[WebSocket] = set()
        self.input_queue: "asyncio.Queue[str]" = asyncio.Queue()
        self.ingestion: Optional[Any] = None
        self.memory: Optional[Any] = None


class FrontendBridge:
    _instance: Optional["FrontendBridge"] = None

    @classmethod
    def get_instance(cls) -> "FrontendBridge":
        if cls._instance is None:
            cls._instance = FrontendBridge()
        return cls._instance

    def __init__(self) -> None:
        self._sessions: Dict[str, _SessionState] = {}

    def _get_or_create_session(self, session_id: str) -> _SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = _SessionState()
        return self._sessions[session_id]

    async def register_websocket(self, session_id: str, websocket: WebSocket) -> None:
        state = self._get_or_create_session(session_id)
        state.websockets.add(websocket)
        logger.info(f"WebSocket registered for session={session_id}; total={len(state.websockets)}")

    async def unregister_websocket(self, session_id: str, websocket: WebSocket) -> None:
        state = self._get_or_create_session(session_id)
        if websocket in state.websockets:
            state.websockets.remove(websocket)
            logger.info(f"WebSocket unregistered for session={session_id}; total={len(state.websockets)}")

    def set_ingestion(self, session_id: str, ingestion: Any) -> None:
        state = self._get_or_create_session(session_id)
        state.ingestion = ingestion
        logger.info(f"Ingestion set for session={session_id}")

    def get_ingestion(self, session_id: str) -> Optional[Any]:
        state = self._get_or_create_session(session_id)
        return state.ingestion

    def set_memory(self, session_id: str, memory: Any) -> None:
        state = self._get_or_create_session(session_id)
        state.memory = memory
        logger.info(f"Memory set for session={session_id}")

    def get_memory(self, session_id: str) -> Optional[Any]:
        state = self._get_or_create_session(session_id)
        return state.memory

    async def enqueue_user_input(self, session_id: str, text: str) -> None:
        state = self._get_or_create_session(session_id)
        await state.input_queue.put(text)
        logger.info(f"User input enqueued for session={session_id}")

    async def get_user_input(self, session_id: str) -> str:
        state = self._get_or_create_session(session_id)
        text = await state.input_queue.get()
        return text

    async def create_message_block(self, session_id: str, header: str, block_type: str = "agent", metadata: Optional[Dict[str, Any]] = None) -> str:
        block_id = str(uuid.uuid4())
        payload = {
            "type": "block_create",
            "session_id": session_id,
            "block": {
                "id": block_id,
                "block_type": block_type,
                "header": header,
                "metadata": metadata or {},
            },
        }
        await self._broadcast(session_id, payload)
        return block_id

    async def update_message_block(self, session_id: str, block_id: str, llm_response: Any) -> None:
        payload = {
            "type": "block_update",
            "session_id": session_id,
            "block_id": block_id,
            "llm_response": llm_response,
        }
        await self._broadcast(session_id, payload)

    async def complete_message_block(self, session_id: str, block_id: str, result: Any) -> None:
        payload = {
            "type": "block_complete",
            "session_id": session_id,
            "block_id": block_id,
            "result": result,
        }
        await self._broadcast(session_id, payload)

    async def _broadcast(self, session_id: str, payload: Dict[str, Any]) -> None:
        state = self._get_or_create_session(session_id)
        if not state.websockets:
            logger.warning(f"No active WebSocket for session={session_id}; dropping message")
            return

        message_text = json.dumps(payload, ensure_ascii=False)
        dead: Set[WebSocket] = set()
        for ws in list(state.websockets):
            try:
                await ws.send_text(message_text)
            except Exception as e:  # pragma: no cover
                logger.error(f"Failed to send WS message: {e}")
                dead.add(ws)
        for ws in dead:
            try:
                state.websockets.remove(ws)
            except Exception:
                pass


