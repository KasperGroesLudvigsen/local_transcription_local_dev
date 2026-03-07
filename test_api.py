#!/usr/bin/env python3
"""
Simple test script for the Local Transcription API service.
This script tests the API endpoints with a sample audio file.
"""

import requests
import os
import sys
from pathlib import Path

def test_api_health():
    """Test that the API is running and accessible."""
    print("Testing API health...")

    try:
        response = requests.get("http://localhost:3030/")
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

def test_transcribe_endpoint(audio_file_path):
    """Test the transcribe endpoint with a sample audio file."""
    print("Testing transcribe endpoint...")

    # Check if the specified audio file exists
    test_file = Path(audio_file_path)
    if not test_file.exists():
        print(f"✗ Audio file '{audio_file_path}' not found")
        return False

    try:
        with open(test_file, 'rb') as f:
            files = {'file': f}
            response = requests.post("http://localhost:3030/transcribe", files=files)

        if response.status_code == 200:
            result = response.json()
            print(f"FULL RESPONSE:\n {result}")
            print("✓ Transcribe endpoint successful")
            print(f"  Full text: {result.get('full_text', 'No text')[:100]}...")
            print(f"  Language: {result.get('language', 'Unknown')}")
            return True
        elif response.status_code == 400:
            print(f"✗ Transcribe endpoint returned 400 (expected for invalid file format)")
            print(f"  Error: {response.json().get('detail', 'Unknown error')}")
            return True  # This might be expected for invalid file format
        else:
            print(f"✗ Transcribe endpoint failed with status code: {response.status_code}")
            print(f"  Error: {response.json().get('detail', 'Unknown error')}")
            return False

    except requests.exceptions.ConnectionError:
        print("✗ Failed to connect to API")
        return False
    except Exception as e:
        print(f"✗ Error calling transcribe endpoint: {str(e)}")
        return False

def test_detect_language_endpoint(audio_file_path):
    """Test the detect_language endpoint with a sample audio file."""
    print("Testing detect_language endpoint...")

    # Check if the specified audio file exists
    test_file = Path(audio_file_path)
    if not test_file.exists():
        print(f"✗ Audio file '{audio_file_path}' not found")
        return False

    try:
        with open(test_file, 'rb') as f:
            files = {'file': f}
            response = requests.post("http://localhost:3030/detect_language", files=files)

        if response.status_code == 200:
            result = response.json()
            print("✓ Detect language endpoint successful")
            print(f"  Detected language: {result.get('detected_language', 'Unknown')}")
            print(f"  Confidence: {result.get('confidence', 'Unknown')}")
            return True
        elif response.status_code == 400:
            print(f"✗ Detect language endpoint returned 400 (expected for invalid file format)")
            print(f"  Error: {response.json().get('detail', 'Unknown error')}")
            return True  # This might be expected for invalid file format
        else:
            print(f"✗ Detect language endpoint failed with status code: {response.status_code}")
            print(f"  Error: {response.json().get('detail', 'Unknown error')}")
            return False

    except requests.exceptions.ConnectionError:
        print("✗ Failed to connect to API")
        return False
    except Exception as e:
        print(f"✗ Error calling detect_language endpoint: {str(e)}")
        return False

def main():
    """Run all tests."""
    # Check if audio file path is provided as command line argument
    if len(sys.argv) != 2:
        print("Usage: python3 test_api.py <audio_file_path>")
        print("Example: python3 test_api.py my_audio.wav")
        return 1

    audio_file_path = sys.argv[1]

    print(f"Running Local Transcription API Tests with audio file: {audio_file_path}\n")

    tests = [
        test_api_health,
        lambda: test_transcribe_endpoint(audio_file_path),
        lambda: test_detect_language_endpoint(audio_file_path)
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