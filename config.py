#!/usr/bin/env python3
"""
Configuration settings for Meeting Transcriber
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path("/Users/harsmis/Downloads/transcription")
AUDIO_DIR = BASE_DIR / "audio_files"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Model settings
WHISPER_MODEL = "large-v3"  # Options: tiny, base, small, medium, large, large-v2, large-v3
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Hugging Face token (replace with your actual token)
HUGGING_FACE_TOKEN = "<ENTER-YOUR-TOKEN-HERE>"

# Audio processing settings
SUPPORTED_FORMATS = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac')
SAMPLE_RATE = 16000  # Whisper's preferred sample rate

# Output settings
OUTPUT_FORMATS = {
    'transcript': True,     # Human-readable transcript
    'detailed_json': True,  # Detailed JSON with all metadata
    'analysis_csv': True,   # CSV for analysis
    'srt_subtitles': False, # SRT subtitle format
    'vtt_subtitles': False  # WebVTT subtitle format
}

# Processing settings
LANGUAGE = "en"  # Language code for Whisper
DEVICE = "auto"  # "auto", "cpu", or "cuda" for GPU
BATCH_SIZE = 16  # For faster processing on powerful machines

# Logging settings
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Quality settings
MIN_SPEAKER_DURATION = 1.0  # Minimum seconds for a speaker segment
CONFIDENCE_THRESHOLD = 0.5   # Minimum confidence for including segments
