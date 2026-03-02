"""
Main FastAPI application for the transcription service.
"""

import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
from typing import Dict, Any

from transcriber import Transcriber
from models import TranscriptionResponse, LanguageDetectionResponse, ErrorResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Local Transcription API",
    description="API service for speech-to-text transcription using Hugging Face models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global transcriber instance
transcriber = None

@app.on_event("startup")
async def startup_event():
    """Initialize the transcriber when the application starts."""
    global transcriber
    try:
        transcriber = Transcriber()
        logger.info("Transcription service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize transcription service: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint to verify service is running."""
    return {"message": "Local Transcription API is running"}

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an audio file to text.

    Args:
        file: Audio file to transcribe

    Returns:
        Transcription results including full text and timestamped transcriptions
    """
    if not transcriber:
        raise HTTPException(status_code=500, detail="Service not initialized")

    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".flac", ".m4a", ".aac"}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported types: {allowed_extensions}"
        )

    # Validate file size (500MB limit)
    max_file_size = 500 * 1024 * 1024  # 500MB in bytes
    content = await file.read()

    if len(content) > max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {max_file_size / (1024*1024)} MB"
        )

    # Save file temporarily for processing
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Perform transcription
        result = transcriber.transcribe(tmp_file_path)

        # Clean up temporary file
        os.unlink(tmp_file_path)

        return result

    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/detect_language", response_model=LanguageDetectionResponse)
async def detect_language(file: UploadFile = File(...)):
    """
    Detect language of an audio file.

    Args:
        file: Audio file to analyze

    Returns:
        Detected language information
    """
    if not transcriber:
        raise HTTPException(status_code=500, detail="Service not initialized")

    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".flac", ".m4a", ".aac"}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported types: {allowed_extensions}"
        )

    # Validate file size (500MB limit)
    max_file_size = 500 * 1024 * 1024  # 500MB in bytes
    content = await file.read()

    if len(content) > max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {max_file_size / (1024*1024)} MB"
        )

    # Save file temporarily for processing
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Perform language detection
        result = transcriber.detect_language(tmp_file_path)

        # Clean up temporary file
        os.unlink(tmp_file_path)

        return result

    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        logger.error(f"Language detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)