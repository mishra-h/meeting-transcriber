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
