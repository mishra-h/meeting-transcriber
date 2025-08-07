#!/bin/bash

# Setup script for meeting transcription system
# Run this in /Users/harsmis/Downloads/transcription

echo "🎙️  Setting up Meeting Transcription System..."
echo "================================================"

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p audio_files
mkdir -p output/{transcripts,detailed,analysis}
mkdir -p models
mkdir -p logs
mkdir -p utils

# Create .gitkeep files to preserve empty directories
touch audio_files/.gitkeep
touch output/.gitkeep
touch output/transcripts/.gitkeep
touch output/detailed/.gitkeep
touch output/analysis/.gitkeep
touch models/.gitkeep
touch logs/.gitkeep

# Create requirements.txt
echo "📦 Creating requirements.txt..."
cat > requirements.txt << EOF
torch>=2.0.0
torchaudio>=2.0.0
openai-whisper>=20231117
pyannote.audio>=3.1.0
librosa>=0.10.0
soundfile>=0.12.0
jupyter>=1.0.0
matplotlib>=3.7.0
pandas>=2.0.0
huggingface_hub>=0.16.0
pathlib
logging
EOF

# Create utils/__init__.py
echo "🔧 Creating utility modules..."
touch utils/__init__.py

# Create a simple README
echo "📝 Creating README.md..."
cat > README.md << 'EOF'
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
EOF

# Create simple command-line interface
echo "⚡ Creating quick transcribe script..."
cat > quick_transcribe.py << 'EOF'
#!/usr/bin/env python3
"""
Quick command-line interface for meeting transcription
Usage: python quick_transcribe.py <audio_file_path>
"""

import sys
import os
from pathlib import Path
from meeting_transcriber import MeetingTranscriber

def main():
    if len(sys.argv) != 2:
        print("Usage: python quick_transcribe.py <audio_file_path>")
        print("\nExample:")
        print("  python quick_transcribe.py audio_files/meeting.wav")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"❌ Error: File '{audio_path}' not found")
        sys.exit(1)
    
    print(f"🎙️  Starting transcription of: {audio_path}")
    
    try:
        transcriber = MeetingTranscriber()
        result = transcriber.transcribe_with_speakers(audio_path)
        
        print(f"✅ Successfully transcribed {len(result)} segments")
        print(f"📁 Output files saved in 'output/' directory")
        
        # Quick summary
        speakers = set(segment['speaker'] for segment in result)
        total_duration = max(segment['end'] for segment in result)
        
        print(f"\n📊 Summary:")
        print(f"   Duration: {total_duration:.1f} seconds")
        print(f"   Speakers: {len(speakers)} ({', '.join(sorted(speakers))})")
        
    except Exception as e:
        print(f"❌ Error during transcription: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Make scripts executable
chmod +x quick_transcribe.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Get HuggingFace token from: https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "3. Edit config.py and replace YOUR_HUGGING_FACE_TOKEN with your actual token"
echo "4. Place audio files in audio_files/ folder"
echo "5. Run: python quick_transcribe.py audio_files/your_meeting.wav"
echo ""
echo "🚀 You can also use the Jupyter notebook: transcription_notebook.ipynb"
