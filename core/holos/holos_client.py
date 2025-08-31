"""
Holos API client for interacting with agent services.
"""

import requests
import json
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin
import os


class HolosClient:
    """Client for interacting with Holos API services."""
    
    def __init__(self, base_url: str = "http://10.245.130.134:8000"):
        """
        Initialize the Holos client.
        
        Args:
            base_url: Base URL for the Holos API service
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def get_all_agents(self) -> List[Dict[str, Any]]:
        """
        Get all agents' agentcard information.
        
        Returns:
            List of agent dictionaries containing agentcard information
            
        Raises:
            requests.RequestException: If the API request fails
        """
        url = urljoin(self.base_url, "/api/v1/holos/agents")
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            # Parse JSON response
            agents_data = response.json()
            
            # Save agents_data to JSON file
            self._save_agents_data_to_file(agents_data)
            
            return agents_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching agents: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
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
            
            print(f"Agents data saved to: {file_path}")
            
        except Exception as e:
            print(f"Error saving agents data to file: {e}")
            # Don't raise the exception to avoid breaking the main functionality
    
    def close(self):
        """Close the session and clean up resources."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def get_all_agent_cards(base_url: str = "http://10.245.130.134:8000") -> List[Dict[str, Any]]:
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
    # Example usage
    try:
        # Using the simple function
        agents = get_all_agent_cards()
            
    except Exception as e:
        print(f"Error: {e}")
