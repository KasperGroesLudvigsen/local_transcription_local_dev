"""
Unit tests for the transcription API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from main import app
import tempfile
import os

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Local Transcription API is running"

def test_transcribe_endpoint_missing_file():
    """Test transcribe endpoint with missing file."""
    response = client.post("/transcribe")
    assert response.status_code == 400

def test_detect_language_endpoint_missing_file():
    """Test detect_language endpoint with missing file."""
    response = client.post("/detect_language")
    assert response.status_code == 400

def test_transcribe_endpoint_invalid_format():
    """Test transcribe endpoint with invalid file format."""
    # Create a temporary file with invalid extension
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
        tmp_file.write(b"test content")
        tmp_file_path = tmp_file.name

    try:
        with open(tmp_file_path, "rb") as f:
            response = client.post("/transcribe", files={"file": f})
        assert response.status_code == 400
    finally:
        os.unlink(tmp_file_path)

def test_detect_language_endpoint_invalid_format():
    """Test detect_language endpoint with invalid file format."""
    # Create a temporary file with invalid extension
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
        tmp_file.write(b"test content")
        tmp_file_path = tmp_file.name

    try:
        with open(tmp_file_path, "rb") as f:
            response = client.post("/detect_language", files={"file": f})
        assert response.status_code == 400
    finally:
        os.unlink(tmp_file_path)