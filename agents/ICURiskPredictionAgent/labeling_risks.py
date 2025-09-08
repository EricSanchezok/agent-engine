import os
import json
import asyncio

# Agent Engine imports
from agent_engine.agent_logger import AgentLogger

# Local imports
from agents.ICURiskPredictionAgent.risk_diagnosis import RiskDiagnosis

logger = AgentLogger(__name__) 


MAX_CONCURRENT_TASKS = 16

async def process_file(file_path: str, save_path: str, selected_event_types: list, risk_diagnosis: RiskDiagnosis, semaphore: asyncio.Semaphore):
    async with semaphore:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for event in data['sequence']:
                if event['sub_type'] in selected_event_types:
                    event['risks'] = await risk_diagnosis.invoke(event)
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Processed {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

async def main():
    selected_event_types = [
        "exam", "blood_gas", "lab", "surgery"
    ]
    raw_dir = "database/icu_unlabeled"
    save_dir = "database/icu_patients"
    
    os.makedirs(save_dir, exist_ok=True)

    risk_diagnosis = RiskDiagnosis()
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    
    tasks = []
    
    for file in os.listdir(raw_dir):
        if file.endswith('.json'):
            file_path = os.path.join(raw_dir, file)
            save_path = os.path.join(save_dir, file)
            
            task = process_file(file_path, save_path, selected_event_types, risk_diagnosis, semaphore)
            tasks.append(task)
    
    logger.info(f"Starting to process {len(tasks)} files with max {MAX_CONCURRENT_TASKS} concurrent tasks")
    await asyncio.gather(*tasks)
    logger.info("All files processed successfully")


if __name__ == "__main__":
    asyncio.run(main())