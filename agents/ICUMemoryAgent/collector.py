from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Tuple, Optional
import json
from datetime import datetime, timezone
import asyncio


# Agent Engine imports
from agent_engine.agent_logger import AgentLogger
from agent_engine.llm_client import AzureClient
from agent_engine.prompt import PromptLoader
from agent_engine.memory import ScalableMemory
from agent_engine.utils import get_relative_path_from_current_file

# Local imports
from agents.ICUMemoryAgent.utils import save_test_events
from agents.ICUMemoryAgent.agent import ICUMemoryAgent

logger = AgentLogger(__name__)

load_dotenv()

class ICUMemoryCollector:
    def __init__(self):
        self.llm_client = AzureClient(api_key=os.getenv('AZURE_API_KEY'))
        self.prompt_loader = PromptLoader(get_relative_path_from_current_file('prompts.yaml'))

    async def build_plan(
        self,
        user_query: str,
        total_events: int,
        event_time_range: Tuple[Optional[str], Optional[str]],
        result_summary: Optional[str] = None,
        reasons: str = "",
    ) -> List[Dict[str, Any]]:
        """Generate a JSON execution plan (list of tool calls) for data collection."""
        sub_type_descriptions = json.load(open(get_relative_path_from_current_file('sub_type_descriptions.json'), 'r', encoding='utf-8'))

        system_prompt = self.prompt_loader.get_prompt(
            section='collector',
            prompt_type='system',
            sub_type_descriptions=sub_type_descriptions
        )
        user_prompt = self.prompt_loader.get_prompt(
            section='collector',
            prompt_type='user',
            user_query=user_query,
            current_time=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            total_events=total_events,
            event_time_range=event_time_range,
            result_summary=(result_summary or ""),
            reasons=reasons,
        )

        try:
            response = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3-mini')
            response = json.loads(response)
            return response if isinstance(response, list) else []
        except Exception as e:
            logger.error(f"ICUMemoryCollector.build_plan error: {e}")
            return []

    async def assess(self, user_query: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess whether current coverage is sufficient. Returns JSON with keys: is_sufficient, reasons, summary."""
        system_prompt = self.prompt_loader.get_prompt(
            section='assessor',
            prompt_type='system'
        )
        user_prompt = self.prompt_loader.get_prompt(
            section='assessor',
            prompt_type='user',
            user_query=user_query,
            current_time=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            events=events,
        )
        try:
            response = await self.llm_client.chat(system_prompt, user_prompt, model_name='o3-mini')
            obj = json.loads(response)
            if not isinstance(obj, dict):
                raise ValueError("Assessor response is not a JSON object")
            # Normalize required keys
            obj.setdefault('is_sufficient', False)
            obj.setdefault('reasons', "")
            obj.setdefault('summary', "")
            return obj
        except Exception as e:
            logger.error(f"ICUMemoryCollector.assess error: {e}")
            return {"is_sufficient": False, "reasons": "assessor_error", "summary": ""}

    async def collect_events(
        self,
        patient_id: str,
        user_query: str,
        memory: ICUMemoryAgent,
        max_iterations: int = 3,
    ) -> List[Dict[str, Any]]:
        """Iteratively plan→execute→assess until coverage is sufficient or max_iterations is reached."""
        # Fetch coarse context once
        total_events = memory.get_event_count(patient_id)
        event_time_range = memory.get_event_time_range(patient_id)

        aggregated_by_id: Dict[str, Dict[str, Any]] = {}
        reasons: str = ""
        result_summary: str = ""
        plan_history: List[Dict[str, Any]] = []

        for it in range(max_iterations):
            logger.info(f"Iteration {it+1}/{max_iterations}: building plan")
            plan = await self.build_plan(
                user_query=user_query,
                total_events=total_events,
                event_time_range=event_time_range,
                result_summary=result_summary or "",
                reasons=reasons,
            )

            if not plan:
                logger.warning("Planner returned empty plan; stopping iterations")
                break

            logger.info(f"Plan: {plan}")

            # Record plan history
            plan_history.append({
                "iteration": it + 1,
                "plan": plan,
            })

            # Execute plan in parallel
            logger.info(f"Executing plan with {len(plan)} tool calls")
            results_lists = await self._execute_plan(patient_id, memory, plan)
            # Merge
            results_length = 0
            for events in results_lists:
                for ev in events:
                    ev_id = ev.get('id')
                    if not ev_id:
                        continue
                    results_length += 1
                    aggregated_by_id[ev_id] = ev
            
            logger.info(f"Result length: {results_length}, Aggregated length: {len(aggregated_by_id)}")

            # Assess on full events and update reasons/summary
            current_events_list = list(aggregated_by_id.values())
            assessor = await self.assess(user_query, current_events_list)
            reasons = assessor.get('reasons', "")
            assessor_summary = assessor.get('summary', "")
            # Combine assessor summary with additional fields
            result_summary = {
                "summary": assessor_summary,
                "total_events_returned": len(current_events_list),
                "plan_history": plan_history,
            }

            logger.info(f"Assessment: sufficient={assessor.get('is_sufficient', False)}")
            if assessor.get('is_sufficient', False):
                break

        # Return aggregated events sorted by timestamp asc
        sorted_events = sorted(
            list(aggregated_by_id.values()),
            key=lambda e: self._to_datetime(e.get('timestamp')) or datetime.min.replace(tzinfo=timezone.utc)
        )
        return sorted_events

    async def _execute_plan(
        self,
        patient_id: str,
        memory,
        plan: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        results_lists: List[List[Dict[str, Any]]] = []
        for call in plan:
            tool = call.get('tool_name') or call.get('tool')
            params = call.get('parameters', {})
            if tool == 'get_events_within_hours':
                hours = int(params.get('hours', 0))
                sub_types = params.get('sub_types')
                res = memory.get_events_within_hours(
                    patient_id,
                    None,
                    hours,
                    False,
                    sub_types,
                )
                results_lists.append(res)
            elif tool == 'get_recent_events':
                n = int(params.get('n', 0))
                sub_types = params.get('sub_types')
                res = memory.get_recent_events(
                    patient_id,
                    n,
                    False,
                    sub_types,
                )
                results_lists.append(res)
            elif tool == 'get_events_between':
                start_time = params.get('start_time')
                end_time = params.get('end_time')
                sub_types = params.get('sub_types')
                res = memory.get_events_between(
                    patient_id,
                    start_time,
                    end_time,
                    False,
                    sub_types,
                )
                results_lists.append(res)
            else:
                logger.warning(f"Unknown tool_name in plan: {tool}")
        return results_lists

    def _to_datetime(self, ts: Optional[str]) -> Optional[datetime]:
        if ts is None:
            return None
        s = str(ts).strip()
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            try:
                dt = datetime.fromisoformat(s.split('.')[0])
            except Exception:
                return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    # removed _build_result_summary; assessor now provides the authoritative summary

async def main():
    from agents.ICUDataIngestionAgent.agent import ICUDataIngestionAgent
    from agents.ICUMemoryAgent.agent import ICUMemoryAgent
    from pprint import pprint

    UPDATES = 100
    PATIENT_ID = "1125112810"
    user_query = "What was the most recent surgery the patient had?"

    ingestion = ICUDataIngestionAgent()
    patient_json_path = f"database/icu_raw/{PATIENT_ID}.json"
    ingestion.load_patient(patient_json_path)
    patient_id = ingestion.patient_id or PATIENT_ID
    memory = ICUMemoryAgent()
    memory.delete_patient_memory(ingestion.patient_id)

    total_written = 0
    for i in range(1, UPDATES + 1):
        batch = await ingestion.update()
        if not batch:
            logger.info("No more events from ingestion; stopping early at update %s", i)
            break
        ids = await memory.add_events(patient_id, batch)
        total_written += len(ids)
        logger.info(f"Update {i}: wrote {len(ids)} events (total={total_written})")

    collector = ICUMemoryCollector()
    result = await collector.collect_events(patient_id, user_query, memory)
    pprint(result)

if __name__ == "__main__":
    asyncio.run(main())