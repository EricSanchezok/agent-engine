from __future__ import annotations

import uvicorn
from typing import Optional
import datetime
import pytz
from holos_sdk.types import TaskPlan
from holos_sdk.plant_tracer import PlantTracer

# A2A framework imports
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue, event_queue
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.apps import A2AStarletteApplication
from a2a.types import AgentCard
from a2a.utils import (
    new_agent_text_message, new_agent_parts_message, new_artifact,
    new_data_artifact, new_task
)
from a2a.types import (
    Artifact, Message, Role, Task, TaskStatus, TaskState, 
    FilePart, Part, FileWithBytes, TextPart, MessageSendParams,
    TaskStatusUpdateEvent, TaskArtifactUpdateEvent
)

# Internal imports
from ..agent_logger.agent_logger import AgentLogger


class BaseA2AAgent(AgentExecutor):
    def __init__(self, agent_card: AgentCard, task_store: Optional[InMemoryTaskStore] = None):
        self.agent_card: AgentCard = agent_card
        self.task_store: InMemoryTaskStore = task_store if task_store else InMemoryTaskStore()
        # self.tracer = PlantTracer(
        #     creator_id=self.agent_card.name,
        #     base_url=self.agent_card.url
        # )

        self.logger = AgentLogger(self.agent_card.name)

        super().__init__()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:  # noqa: D401
        raise NotImplementedError("Subclasses must implement the `execute` method.")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:  # noqa: D401
        context_id = getattr(context, 'context_id', 'unknown')
        task_id = getattr(context, 'task_id', 'unknown')
        self.logger.warning(f"Cancel request received - Context ID: {context_id}, Task ID: {task_id}")
        error_info = f"Cancel operation is not supported by {self.agent_card.name}. This agent does not implement cancellation functionality."
        raise Exception(error_info)

    async def _task_failed(self, context: RequestContext, event_queue: EventQueue, error_info: str) -> None:
        task_id = getattr(context, 'task_id', 'unknown')
        context_id = getattr(context, 'context_id', 'unknown')
        user_input = context.get_user_input() if hasattr(context, 'get_user_input') else 'unknown'
        
        # Create a more detailed error message with context
        detailed_error = f"Task Execution Failed\nError: {error_info}\nContext:\n- Task ID: {task_id}\n- User Request: {user_input}\nPlease try again or contact support if the issue persists."
        
        self.logger.error(detailed_error)
        try:
            response_message = new_agent_text_message(detailed_error)
            task = Task(
                id=task_id,
                contextId=context_id,
                history=[response_message],
                status=TaskStatus(
                    state=TaskState.failed,
                    timestamp=datetime.datetime.now(pytz.timezone('Asia/Shanghai')).replace(microsecond=0).isoformat()
                )
            )
            await self._put_event(event_queue, task)
            self.logger.debug("Detailed error response message sent to event queue")
        except Exception as queue_error:
            self.logger.error(f"Failed to add error response to event queue: {queue_error}")

    async def _put_event(self, event_queue: EventQueue, task: Task) -> None:
        try:
            if hasattr(event_queue, 'enqueue_event') and callable(getattr(event_queue, 'enqueue_event')):
                await event_queue.enqueue_event(task)
                self.logger.debug("Event enqueued using enqueue_event method")
            else:
                await event_queue.put(task)
                self.logger.debug("Event put using put method")
        except Exception as queue_error:
            error_info = "Failed to add event to event queue. This may indicate a communication issue with the event system."
            self.logger.error(f"{error_info}: {queue_error}")
        
        await self.task_store.save(task)

    async def run_user_input(self, user_input: str) -> Optional[Task, Message, TaskStatusUpdateEvent, TaskArtifactUpdateEvent]:
        self.logger.info(f"Running user input: {user_input}")
        request = MessageSendParams(
            message=new_agent_text_message(user_input),
        )
        context = RequestContext(request=request)
        event_queue = EventQueue()
        await self.execute(context, event_queue)
        event = await event_queue.dequeue_event()
        await event_queue.close()
        return event

    async def run_message(self, message: Message) -> Optional[Task, Message, TaskStatusUpdateEvent, TaskArtifactUpdateEvent]:
        self.logger.info(f"Running message: {message}")
        request = MessageSendParams(
            message=message,
        )
        context = RequestContext(request=request)
        event_queue = EventQueue()
        await self.execute(context, event_queue)
        event = await event_queue.dequeue_event()
        await event_queue.close()
        return event

    def run_server(self, *, host: Optional[str] = None, port: Optional[int] = None, log_level: str = "info") -> None:
        from urllib.parse import urlparse

        if host is None or port is None:
            parsed = urlparse(self.agent_card.url)
            auto_host = parsed.hostname or "0.0.0.0"
            auto_port = parsed.port or 8000
            host = host or auto_host
            port = port or auto_port

        self.logger.info(
            f"Launching {self.agent_card.name} – Host: {host}, Port: {port}, URL: {self.agent_card.url}"
        )

        request_handler = DefaultRequestHandler(
            agent_executor=self,
            task_store=self.task_store,
        )

        app_builder = A2AStarletteApplication(
            agent_card=self.agent_card,
            http_handler=request_handler,
        )

        uvicorn.run(app_builder.build(), host=host, port=port, log_level=log_level)
