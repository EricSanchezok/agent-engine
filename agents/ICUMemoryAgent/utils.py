from typing import List, Dict, Any
import json

# Agent Engine imports
from agent_engine.utils import get_relative_path_from_current_file

def save_test_events(events: List[Dict[str, Any]]) -> None:
    path = get_relative_path_from_current_file('test_events.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_events = []
    for event in events:
        _event = {
            'id': event['id'],
            'similarity': event['similarity'],
            'timestamp': event['timestamp'],
            'event_type': event['event_type'],
            'sub_type': event['sub_type'],
            'event_content': event['content'],
            'raw_content': event.get('metadata', {}).get('raw_content')
        }
        saved_events.append(_event)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(saved_events, f, ensure_ascii=False, indent=4)