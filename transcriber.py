"""
Transcription service module for handling speech-to-text conversion
using the Hugging Face syvai/hviske-v3-conversation model.
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Dict, Any, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)

class Transcriber:
    """Handles loading and using the Hugging Face speech transcription model."""

    def __init__(self, model_id: str = "syvai/hviske-v3-conversation"):
        """
        Initialize the transcriber with the specified model.

        Args:
            model_id: Hugging Face model identifier
        """
        self.model_id = model_id
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Initialize model, processor, and pipeline
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model, processor, and pipeline."""
        try:
            logger.info(f"Loading model {self.model_id} on device {self.device}")

            # Load model with optimized settings for GPU
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            self.model.to(self.device)

            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_id)

            # Create ASR pipeline
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=16,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Could not initialize transcription model: {str(e)}")

    def transcribe(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio file using the loaded model.

        Args:
            audio_file_path: Path to the audio file

        Returns:
            Dictionary containing transcription results
        """
        try:
            logger.info(f"Starting transcription for {audio_file_path}")

            # Perform transcription
            result = self.pipe(audio_file_path)

            # Extract text
            text = result.get("text", "")

            # Return structured response
            return {
                "full_text": text,
                "timestamped_transcriptions": result.get("chunks", []),
                "language": result.get("language", "unknown")
            }

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise RuntimeError(f"Transcription failed: {str(e)}")

    def detect_language(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Detect language of an audio file using the loaded model.

        Args:
            audio_file_path: Path to the audio file

        Returns:
            Dictionary containing language detection results
        """
        try:
            logger.info(f"Detecting language for {audio_file_path}")

            # Perform language detection (if supported by model)
            result = self.pipe(audio_file_path)

            # Return language information
            return {
                "detected_language": result.get("language", "unknown"),
                "confidence": result.get("language_confidence", 0.0)
            }

        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            raise RuntimeError(f"Language detection failed: {str(e)}")