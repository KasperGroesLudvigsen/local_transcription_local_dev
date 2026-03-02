# Local Transcription API

A FastAPI-based service that exposes speech-to-text transcription capabilities using the Hugging Face "syvai/hviske-v3-conversation" model.

## Features

- RESTful API with FastAPI
- Speech-to-text transcription for Danish speech
- Support for multiple audio formats (wav, mp3, flac, m4a, aac)
- Concurrent request handling (up to 10 simultaneous requests)
- GPU-accelerated processing with CUDA support
- Docker containerization with Docker Compose
- Proper resource management and GPU memory cleanup

## Requirements

- NVIDIA GPU with CUDA 13.0 support
- NVIDIA driver 580.126.09 or higher
- 128GB unified VRAM (Fusionxpark environment)
- Docker and Docker Compose

## API Endpoints

### POST /transcribe
Transcribe an audio file to text.

**Request:**
- File upload (multipart/form-data)
- Supported formats: wav, mp3, flac, m4a, aac
- Maximum file size: 500MB

**Response:**
```json
{
  "full_text": "Transcribed text here...",
  "timestamped_transcriptions": [
    {
      "text": "Transcribed segment",
      "start": 0.0,
      "end": 5.0
    }
  ],
  "language": "da"
}
```

### POST /detect_language
Detect language of an audio file.

**Request:**
- File upload (multipart/form-data)
- Supported formats: wav, mp3, flac, m4a, aac
- Maximum file size: 500MB

**Response:**
```json
{
  "detected_language": "da",
  "confidence": 0.95
}
```

## Quick Start

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. The API will be available at http://localhost:3030

## Development

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- NVIDIA GPU with CUDA support

### Installation
```bash
pip install -r requirements.txt
```

### Running Locally
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Running Tests
```bash
# Run the API test script
python test_api.py

# Run unit tests (if pytest is installed)
pytest tests/
```

## Deployment

The service is designed to run in Docker containers with GPU support. The docker-compose.yml file is configured to:
- Use NVIDIA GPU drivers
- Map the required port (8000)
- Mount volume for data storage
- Set appropriate CUDA environment variables

## Configuration

The service is configured to work with:
- Hugging Face model: syvai/hviske-v3-conversation
- CUDA version: 13.0
- NVIDIA driver: 580.126.09
- Maximum concurrent requests: 10
- Maximum file size: 500MB

## Resource Management

The service includes:
- Semaphore-based concurrency control (max 10 simultaneous requests)
- GPU memory cleanup after processing
- Thread pool executor for asynchronous processing
- Proper error handling with resource cleanup