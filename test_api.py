#!/usr/bin/env python3
"""
Test script for the Local Transcription API service.
This script verifies the basic functionality of the service.
"""

import requests
import time
import os
from pathlib import Path

def test_api_health():
    """Test that the API is running and accessible."""
    print("Testing API health...")

    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print("✓ API is running and accessible")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ API health check failed with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Failed to connect to API - is it running?")
        return False

def test_transcribe_endpoint_exists():
    """Test that the transcribe endpoint exists."""
    print("Testing transcribe endpoint...")

    try:
        response = requests.post("http://localhost:8000/transcribe")
        # Should get a 400 error for missing file, not 404
        if response.status_code == 400:
            print("✓ Transcribe endpoint exists and is functional")
            return True
        elif response.status_code == 404:
            print("✗ Transcribe endpoint not found")
            return False
        else:
            print(f"✓ Transcribe endpoint exists (returned status {response.status_code})")
            return True
    except requests.exceptions.ConnectionError:
        print("✗ Failed to connect to API")
        return False

def test_detect_language_endpoint_exists():
    """Test that the detect_language endpoint exists."""
    print("Testing detect_language endpoint...")

    try:
        response = requests.post("http://localhost:8000/detect_language")
        # Should get a 400 error for missing file, not 404
        if response.status_code == 400:
            print("✓ Detect language endpoint exists and is functional")
            return True
        elif response.status_code == 404:
            print("✗ Detect language endpoint not found")
            return False
        else:
            print(f"✓ Detect language endpoint exists (returned status {response.status_code})")
            return True
    except requests.exceptions.ConnectionError:
        print("✗ Failed to connect to API")
        return False

def main():
    """Run all tests."""
    print("Running Local Transcription API Tests\n")

    tests = [
        test_api_health,
        test_transcribe_endpoint_exists,
        test_detect_language_endpoint_exists
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())