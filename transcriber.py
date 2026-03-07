"""
Transcription service module for handling speech-to-text conversion
using the Hugging Face syvai/hviske-v3-conversation model.
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Dict, Any, Optional
import logging
import gc

# Set up logging
logger = logging.getLogger(__name__)

class Transcriber:
    """Handles loading and using the Hugging Face speech transcription model."""

    def __init__(self, model_id: str = "syvai/hviske-v3-conversation", token: Optional[str] = None):
        """
        Initialize the transcriber with the specified model.

        Args:
            model_id: Hugging Face model identifier
            token: Hugging Face token for accessing gated models
        """
        self.model_id = model_id
        self.token = token

        # Check for CUDA availability and set device accordingly
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.torch_dtype = torch.float16
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32

        print(f"Using device: {self.device}")  # Debug output

        # Initialize model, processor, and pipeline
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model, processor, and pipeline."""
        try:
            logger.info(f"Loading model {self.model_id} on device {self.device}")

            # Load model with optimized settings for GPU
            # For GPU, we don't need low_cpu_mem_usage as we're not dealing with large models
            # that would overflow CPU RAM
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
                "use_safetensors": True,
                "token": self.token
            }

            # Only use low_cpu_mem_usage for CPU, not GPU
            if self.device == "cpu":
                model_kwargs["low_cpu_mem_usage"] = True

            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            self.model.to(self.device)

            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_id, token=self.token)

            # Create ASR pipeline
            pipeline_kwargs = {
                "model": self.model,
                "tokenizer": self.processor.tokenizer,
                "feature_extractor": self.processor.feature_extractor,
                "max_new_tokens": 128,
                "chunk_length_s": 30,
                "torch_dtype": self.torch_dtype,
                "device": self.device,
                "token": self.token,
                "return_timestamps": True,
                "generate_kwargs": {
                    "pad_token_id": self.processor.tokenizer.pad_token_id,
                    "bos_token_id": self.processor.tokenizer.bos_token_id,
                    "eos_token_id": self.processor.tokenizer.eos_token_id
                }
            }

            self.pipe = pipeline(
                "automatic-speech-recognition",
                **pipeline_kwargs
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
            response = {
                "full_text": text,
                "timestamped_transcriptions": result.get("chunks", []),
                "language": result.get("language", "unknown")
            }

            # Force garbage collection to free up GPU memory
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

            return response

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            # Force garbage collection in case of error
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
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
            response = {
                "detected_language": result.get("language", "unknown"),
                "confidence": result.get("language_confidence", 0.0)
            }

            # Force garbage collection to free up GPU memory
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

            return response

        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            # Force garbage collection in case of error
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            raise RuntimeError(f"Language detection failed: {str(e)}")