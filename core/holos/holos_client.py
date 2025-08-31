"""
Holos API client for interacting with agent services.
"""

import requests
import json
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin


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
            
            # Log the response for debugging
            print(f"Successfully retrieved {len(agents_data) if isinstance(agents_data, list) else 'unknown'} agents")
            
            return agents_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching agents: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            raise
    
    def close(self):
        """Close the session and clean up resources."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def get_all_agents_simple(base_url: str = "http://10.245.130.134:8000") -> List[Dict[str, Any]]:
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
        agents = get_all_agents_simple()
        print(f"Retrieved {len(agents)} agents")
        
        # Print first agent as example
        if agents:
            print("First agent:")
            print(json.dumps(agents[0], indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"Error: {e}")
