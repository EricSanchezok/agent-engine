"""
Holos API client for interacting with agent services.
"""

import requests
import json
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin
import os


# AgentEngine imports
from agent_engine.agent_logger import AgentLogger

# Core imports
from core.holos.config import BASE_URL, PROXY_URL, USE_PROXY

logger = AgentLogger(__name__)


class HolosClient:
    """Client for interacting with Holos API services."""
    
    def __init__(self, base_url: str = "http://10.245.130.134:8000"):
        """
        Initialize the Holos client.
        
        Args:
            base_url: Base URL for the Holos API service
        """
        self.base_url = base_url.rstrip('/') if not USE_PROXY else PROXY_URL + base_url.rstrip('/')
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _deduplicate_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate list elements while preserving order.
        """
        if not isinstance(items, list):
            return []

        seen = set()
        unique_items: List[Dict[str, Any]] = []

        for item in items:
            try:
                key = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, dict) else str(item)
            except Exception:
                key = str(item)

            if key in seen:
                continue
            seen.add(key)
            unique_items.append(item)

        return unique_items

    def get_all_agents(self) -> List[Dict[str, Any]]:
        """
        Get all agents' agentcard information.
        
        Returns:
            List of agent dictionaries containing agentcard information
            
        Raises:
            requests.RequestException: If the API request fails and no fallback data is available
        """
        url = f"{self.base_url}/api/v1/holos/agents"
        logger.info(f"Getting all agents from: {url}")

        try:
            response = self.session.get(url, timeout=1)
            response.raise_for_status()
            
            # Parse JSON response
            agents_data = response.json()
            
            # Deduplicate items inside agents_data
            items = agents_data.get("data", {}).get("items", [])
            deduped_items = self._deduplicate_items(items)
            if "data" not in agents_data or not isinstance(agents_data.get("data"), dict):
                agents_data = {"data": {"items": deduped_items}}
            else:
                agents_data["data"]["items"] = deduped_items

            # Save agents_data to JSON file
            self._save_agents_data_to_file(agents_data)
            
            # Filter out excluded names for the return value
            excluded_names = {"Routing Agent", "User Assistant", "Planning Agent"}
            filtered_items = [
                item for item in deduped_items
                if not (isinstance(item, dict) and item.get("name") in excluded_names)
            ]
            return filtered_items
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching agents from API: {e}")
            logger.error("Attempting to load agents data from local file...")
            
            # Try to load from local file as fallback
            fallback_data = self._load_agents_data_from_file()
            if fallback_data:
                logger.info("Successfully loaded agents data from local file")
                excluded_names = {"Routing Agent", "User Assistant", "Planning Agent"}
                filtered_items = [
                    item for item in fallback_data
                    if not (isinstance(item, dict) and item.get("name") in excluded_names)
                ]
                return filtered_items
            else:
                logger.error("No fallback data available")
                raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.error("Attempting to load agents data from local file...")
            
            # Try to load from local file as fallback
            fallback_data = self._load_agents_data_from_file()
            if fallback_data:
                logger.info("Successfully loaded agents data from local file")
                excluded_names = {"Routing Agent", "User Assistant", "Planning Agent"}
                filtered_items = [
                    item for item in fallback_data
                    if not (isinstance(item, dict) and item.get("name") in excluded_names)
                ]
                return filtered_items
            else:
                logger.error("No fallback data available")
                raise
    
    def _save_agents_data_to_file(self, agents_data: Any) -> None:
        """
        Save agents_data to a JSON file.
        
        Args:
            agents_data: List of agent dictionaries to save
        """
        try:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, "agents_data.json")
            
            # Save data to JSON file with proper formatting
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(agents_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Agents data saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving agents data to file: {e}")
            # Don't raise the exception to avoid breaking the main functionality
    
    def _load_agents_data_from_file(self) -> Optional[List[Dict[str, Any]]]:
        """
        Load agents data from local JSON file as fallback.
        
        Returns:
            List of agent dictionaries if file exists and is valid, None otherwise
        """
        try:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, "agents_data.json")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"Fallback file not found: {file_path}")
                return None
            
            # Load data from JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                agents_data = json.load(f)
            
            logger.info(f"Loaded agents data from fallback file: {file_path}")
            items = agents_data.get("data", {}).get("items", [])
            return self._deduplicate_items(items)
            
        except Exception as e:
            logger.error(f"Error loading agents data from fallback file: {e}")
            return None
    
    def close(self):
        """Close the session and clean up resources."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def get_all_agent_cards(base_url: str = BASE_URL) -> List[Dict[str, Any]]:
    """
    Simple function to get all agents without creating a client instance.
    
    Args:
        base_url: Base URL for the Holos API service
        
    Returns:
        List of agent dictionaries containing agentcard information
    """
    with HolosClient(base_url) as client:
        return client.get_all_agents()


if __name__ == "__main__":
    import pprint
    # Example usage
    try:
        # Using the simple function
        agents = get_all_agent_cards()
        pprint.pprint(agents[:3])
            
    except Exception as e:
        logger.error(f"Error: {e}")
