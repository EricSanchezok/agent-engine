#!/usr/bin/env python3
"""
Test script for the Daily arXiv Paper Reports API
"""

import requests
import json
import time
import sys
from datetime import date

def test_api(base_url="http://localhost:5000"):
    """Test the API endpoints"""
    
    print("Testing Daily arXiv Paper Reports API")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Health check passed: {result['message']}")
        else:
            print(f"   ✗ Health check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Health check failed: {e}")
        return False
    
    # Test 2: Status check
    print("2. Testing status check...")
    try:
        response = requests.get(f"{base_url}/api/status", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Status check passed: {result['message']}")
            if result['data']:
                print(f"   Data: {json.dumps(result['data'], indent=2)}")
        else:
            print(f"   ✗ Status check failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ✗ Status check failed: {e}")
    
    # Test 3: Paper reports (test with recent date)
    print("3. Testing paper reports...")
    today = date.today()
    test_date = today.strftime("%Y%m%d")
    
    test_data = {
        "date_range": {
            "start_date": test_date,
            "end_date": test_date
        }
    }
    
    try:
        response = requests.post(f"{base_url}/api/paper-reports", 
                               json=test_data, 
                               timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Paper reports request successful: {result['message']}")
            print(f"   Found {len(result['data'])} papers")
            
            if result['data']:
                print("   Sample paper:")
                paper = result['data'][0]
                print(f"     Title: {paper['title']}")
                print(f"     Authors: {', '.join(paper['authors'])}")
                print(f"     Categories: {', '.join(paper['categories'])}")
                print(f"     PDF URL: {paper['pdf_url']}")
                print(f"     Report length: {len(paper['report'])} characters")
            else:
                print("   No papers found for today (this is normal if processing hasn't run yet)")
        else:
            print(f"   ✗ Paper reports request failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ✗ Paper reports request failed: {e}")
    
    # Test 4: Invalid date format
    print("4. Testing invalid date format...")
    invalid_data = {
        "date_range": {
            "start_date": "2025-13-45",  # Invalid date
            "end_date": "20250916"
        }
    }
    
    try:
        response = requests.post(f"{base_url}/api/paper-reports", 
                               json=invalid_data, 
                               timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result['code'] == 1:
                print(f"   ✓ Invalid date format properly handled: {result['message']}")
            else:
                print(f"   ✗ Expected error code 1, got {result['code']}")
        else:
            print(f"   ✗ Request failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ✗ Request failed: {e}")
    
    # Test 5: Missing request body
    print("5. Testing missing request body...")
    try:
        response = requests.post(f"{base_url}/api/paper-reports", 
                               timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result['code'] == 1:
                print(f"   ✓ Missing request body properly handled: {result['message']}")
            else:
                print(f"   ✗ Expected error code 1, got {result['code']}")
        else:
            print(f"   ✗ Request failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ✗ Request failed: {e}")
    
    print("\n" + "=" * 50)
    print("API testing completed!")
    return True

def main():
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:5000"
    
    print(f"Testing API at: {base_url}")
    print("Make sure the API server is running!")
    print("Press Ctrl+C to cancel...")
    
    try:
        time.sleep(2)  # Give user time to read
        test_api(base_url)
    except KeyboardInterrupt:
        print("\nTest cancelled by user")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    main()
