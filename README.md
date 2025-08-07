# Meeting Transcription System

A comprehensive tool for transcribing meeting recordings with automatic speaker identification.

## Features
- High-quality transcription using OpenAI Whisper
- Automatic speaker diarization (Speaker 1, Speaker 2, etc.)
- Multiple output formats (text, JSON, CSV)
- Batch processing support
- Detailed logging and error handling

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get a Hugging Face token:
   - Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
   - Accept terms and get your token
   - Replace `YOUR_HUGGING_FACE_TOKEN` in `config.py`

3. Place audio files in `audio_files/` folder

4. Run transcription:
   ```bash
   python quick_transcribe.py audio_files/your_meeting.wav
   ```

## Output Files
- `output/transcripts/` - Human-readable transcripts
- `output/detailed/` - JSON files with timestamps and metadata
- `output/analysis/` - CSV files for data analysis

## Supported Audio Formats
WAV, MP3, M4A, FLAC, OGG, AAC
