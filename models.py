"""
Pydantic models for the transcription API endpoints.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class TranscriptionRequest(BaseModel):
    """Request model for transcription endpoint."""
    audio_file: str  # Path to audio file (in practice, this would be file upload)
    # Additional parameters can be added here if needed
    # For example: language, timestamps, etc.

class TranscriptionResponse(BaseModel):
    """Response model for transcription endpoint."""
    full_text: str
    timestamped_transcriptions: Optional[List[Dict[str, Any]]] = None
    language: Optional[str] = None

class LanguageDetectionRequest(BaseModel):
    """Request model for language detection endpoint."""
    audio_file: str

class LanguageDetectionResponse(BaseModel):
    """Response model for language detection endpoint."""
    detected_language: str
    confidence: float

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str