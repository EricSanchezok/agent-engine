"""
Test script for HolosClient functionality.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.holos.holos_client import HolosClient, get_all_agents_simple


def test_holos_client():
    """Test the HolosClient functionality."""
    
    print("Testing HolosClient...")
    
    # Test 1: Using the client class
    try:
        with HolosClient() as client:
            print("1. Testing client class method...")
            agents = client.get_all_agents()
            print(f"   Retrieved {len(agents) if isinstance(agents, list) else 'unknown'} agents")
            
            if agents and isinstance(agents, list):
                print(f"   First agent keys: {list(agents[0].keys()) if agents[0] else 'No agents'}")
                
    except Exception as e:
        print(f"   Error with client class: {e}")
    
    # Test 2: Using the simple function
    try:
        print("\n2. Testing simple function...")
        agents = get_all_agents_simple()
        print(f"   Retrieved {len(agents) if isinstance(agents, list) else 'unknown'} agents")
        
        if agents and isinstance(agents, list):
            print(f"   First agent keys: {list(agents[0].keys()) if agents[0] else 'No agents'}")
            
    except Exception as e:
        print(f"   Error with simple function: {e}")
    
    # Test 3: Test with custom base URL
    try:
        print("\n3. Testing with custom base URL...")
        custom_client = HolosClient("http://10.245.130.134:8000")
        agents = custom_client.get_all_agents()
        print(f"   Retrieved {len(agents) if isinstance(agents, list) else 'unknown'} agents")
        custom_client.close()
        
    except Exception as e:
        print(f"   Error with custom base URL: {e}")


if __name__ == "__main__":
    test_holos_client()
