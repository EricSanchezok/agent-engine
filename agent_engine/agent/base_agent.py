from __future__ import annotations  # 支持Python3.8+的前向注解

import uvicorn
from typing import Optional

# A2A framework imports
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.apps import A2AStarletteApplication
from a2a.types import AgentCard

# Internal common logger
from ..agent_logger.agent_logger import AgentLogger


class BaseA2AAgent(AgentExecutor):
    def __init__(self, agent_card: AgentCard, task_store: Optional[InMemoryTaskStore] = None):
        self.agent_card: AgentCard = agent_card
        self.task_store: InMemoryTaskStore = task_store if task_store else InMemoryTaskStore()

        self.logger = AgentLogger(self.agent_card.name)

        super().__init__()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:  # noqa: D401
        raise NotImplementedError("Subclasses must implement the `execute` method.")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:  # noqa: D401
        msg = (
            f"Cancel operation is not supported by {self.agent_card.name}. "
            "Override `cancel` if you need this capability."
        )
        self.logger.warning(msg)
        raise NotImplementedError(msg)

    # ------------------------------------------------------------------
    # 便捷的服务启动封装（实例方法）
    # ------------------------------------------------------------------
    def run_server(self, *, host: Optional[str] = None, port: Optional[int] = None, log_level: str = "info") -> None:
        """启动符合 A2A 协议的 Starlette 服务。

        如果 ``host`` 或 ``port`` 未显式提供，则会从 ``self.agent_card.url`` 自动解析。
        """

        from urllib.parse import urlparse

        # 解析 host/port
        if host is None or port is None:
            parsed = urlparse(self.agent_card.url)
            auto_host = parsed.hostname or "0.0.0.0"
            auto_port = parsed.port or 8000
            host = host or auto_host
            port = port or auto_port

        server_logger = AgentLogger(f"{self.__class__.__name__}Server")

        server_logger.info(
            f"Launching {self.__class__.__name__} – Host: {host}, Port: {port}, URL: {self.agent_card.url}"
        )

        # 使用已有 task_store & 自身 executor
        request_handler = DefaultRequestHandler(
            agent_executor=self,
            task_store=self.task_store,
        )

        app_builder = A2AStarletteApplication(
            agent_card=self.agent_card,
            http_handler=request_handler,
        )

        uvicorn.run(app_builder.build(), host=host, port=port, log_level=log_level)


__all__ = ["BaseA2AAgent"]
