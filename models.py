"""
Pydantic models for the transcription API endpoints.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class TranscriptionResponse(BaseModel):
    """Response model for transcription endpoint."""
    full_text: str
    timestamped_transcriptions: Optional[List[Dict[str, Any]]] = None
    language: Optional[str] = None

class LanguageDetectionResponse(BaseModel):
    """Response model for language detection endpoint."""
    detected_language: str
    confidence: float

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str