import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent_engine.agent_logger import AgentLogger

from agents.ICUMemoryAgent.collector import ICUMemoryCollector

from core.icu_web.frontend_bridge import FrontendBridge
from core.icu_web.icu_collector_adapter import CollectorToFrontendAdapter


logger = AgentLogger(__name__)


class WebICUMemoryCollector(ICUMemoryCollector):
    def __init__(self, adapter: CollectorToFrontendAdapter):
        super().__init__()
        self._adapter = adapter
        self._current_block_id = None

    async def build_plan(self, *args, **kwargs):  # proxy and push LLM
        response = await super().build_plan(*args, **kwargs)
        return response

    async def assess(self, *args, **kwargs):  # proxy and push LLM
        response = await super().assess(*args, **kwargs)
        return response

    # Override collect to instrument blocks
    async def collect_events_with_ui(self, patient_id: str, user_query: str, memory, max_iterations: int = 3) -> List[Dict[str, Any]]:
        total_events = memory.get_event_count(patient_id)
        event_time_range = memory.get_event_time_range(patient_id)
        current_time = event_time_range[1]

        aggregated_by_id: Dict[str, Dict[str, Any]] = {}
        reasons: str = ""
        result_summary: str = ""
        plan_history: List[Dict[str, Any]] = []

        for it in range(max_iterations):
            self._current_block_id = await self._adapter.on_plan_start(it + 1, max_iterations)
            plan = await super().build_plan(
                user_query=user_query,
                current_time=current_time,
                total_events=total_events,
                event_time_range=event_time_range,
                result_summary=result_summary or "",
                reasons=reasons,
            )
            await self._adapter.on_plan_result(self._current_block_id, plan)
            if not plan:
                break

            plan_history.append({"iteration": it + 1, "plan": plan})

            results_lists: List[List[Dict[str, Any]]] = []
            for call in plan:
                tool = call.get('tool_name') or call.get('tool')
                params = call.get('parameters', {})
                if tool == 'get_events_within_hours':
                    hours = int(params.get('hours', 0))
                    sub_types = params.get('sub_types')
                    res = memory.get_events_within_hours(patient_id, None, hours, False, sub_types)
                    results_lists.append(res)
                elif tool == 'get_recent_events':
                    n = int(params.get('n', 0))
                    sub_types = params.get('sub_types')
                    res = memory.get_recent_events(patient_id, n, False, sub_types)
                    results_lists.append(res)
                elif tool == 'get_events_between':
                    start_time = params.get('start_time')
                    end_time = params.get('end_time')
                    sub_types = params.get('sub_types')
                    res = memory.get_events_between(patient_id, start_time, end_time, False, sub_types)
                    results_lists.append(res)
            
            self._current_block_id = await self._adapter.on_assess_start()
            current_events_list = [*aggregated_by_id.values()]
            for events in results_lists:
                for ev in events:
                    ev_id = ev.get('id')
                    if not ev_id:
                        continue
                    aggregated_by_id[ev_id] = ev
            current_events_list = [*aggregated_by_id.values()]
            assessor = await super().assess(user_query, current_time, current_events_list)
            await self._adapter.on_assess_result(self._current_block_id, assessor)

            reasons = assessor.get('reasons', "")
            summary = assessor.get('summary', "")
            result_summary = {"summary": summary, "total_events_returned": len(current_events_list), "plan_history": plan_history}
            if assessor.get('is_sufficient', False):
                break

        def _to_datetime(ts: Optional[str]):
            if ts is None:
                return None
            s = str(ts).strip()
            if s.endswith('Z'):
                s = s[:-1] + '+00:00'
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(s)
            except Exception:
                try:
                    dt = datetime.fromisoformat(s.split('.')[0])
                except Exception:
                    return None
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        sorted_events = sorted(list(aggregated_by_id.values()), key=lambda e: _to_datetime(e.get('timestamp')) or datetime.min.replace(tzinfo=timezone.utc))

        final_block = await self._adapter.on_result_start()
        await self._adapter.on_result_complete(final_block, sorted_events)
        return sorted_events


async def run_session_loop(session_id: str) -> None:
    bridge = FrontendBridge.get_instance()
    adapter = CollectorToFrontendAdapter(session_id)
    logger.info(f"Session loop started: {session_id}")
    while True:
        user_text = await bridge.get_user_input(session_id)
        if not user_text:
            continue
        ingestion = bridge.get_ingestion(session_id)
        memory = bridge.get_memory(session_id)
        if ingestion is None or memory is None:
            continue
        patient_id = getattr(ingestion, 'patient_id', None)
        if not patient_id:
            continue
        collector = WebICUMemoryCollector(adapter)
        try:
            await collector.collect_events_with_ui(patient_id, user_text, memory)
        except Exception:
            logger.exception("Collector run failed")


