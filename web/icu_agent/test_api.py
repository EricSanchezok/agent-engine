#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
import sys
import os
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file
while project_root.parent != project_root:
    if (project_root / "pyproject.toml").exists():
        break
    project_root = project_root.parent
sys.path.insert(0, str(project_root))


ICU Agent API Test Script
Test the backend API endpoints to ensure they're working correctly.
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:5000/api"

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"Health Check: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
    except Exception as e:
        print(f"Health check failed: {e}")
    return False

def test_patient_data():
    """Test patient data endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/patient/1", timeout=5)
        print(f"Patient Data: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Patient: {data.get('name')} - {data.get('condition')}")
            return True
    except Exception as e:
        print(f"Patient data test failed: {e}")
    return False

def test_patient_risks():
    """Test patient risks endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/patient/1/risks", timeout=5)
        print(f"Patient Risks: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            risks = data.get('risks', [])
            print(f"Found {len(risks)} risk factors")
            for risk in risks[:2]:
                print(f"  - {risk['name']}: {risk['probability']}%")
            return True
    except Exception as e:
        print(f"Patient risks test failed: {e}")
    return False

def test_chat_api():
    """Test chat API endpoint"""
    try:
        payload = {
            "patient_id": 1,
            "message": "患者当前的血压情况如何？"
        }
        response = requests.post(
            f"{API_BASE_URL}/chat", 
            json=payload,
            timeout=10
        )
        print(f"Chat API: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Agent Response: {data.get('response')[:100]}...")
            return True
    except Exception as e:
        print(f"Chat API test failed: {e}")
    return False

def main():
    """Run all API tests"""
    print("ICU Agent API Test Suite")
    print("=" * 30)
    
    # Wait a moment for server to be ready
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health_check),
        ("Patient Data", test_patient_data),
        ("Patient Risks", test_patient_risks),
        ("Chat API", test_chat_api)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        if test_func():
            print(f"✓ {test_name} PASSED")
            passed += 1
        else:
            print(f"✗ {test_name} FAILED")
    
    print(f"\n--- Test Results ---")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("All tests passed! ICU Agent API is working correctly.")
    else:
        print("Some tests failed. Please check the server logs.")

if __name__ == "__main__":
    main()
