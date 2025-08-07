#!/usr/bin/env python3
"""
Meeting Transcriber with Speaker Diarization
============================================

A comprehensive tool for transcribing meeting recordings with automatic
speaker identification using Whisper and pyannote.audio.

Author: Harshit Mishra
Created: 2025-August
"""

import whisper
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment
import librosa
import soundfile as sf
import os
import json
from datetime import timedelta
import pandas as pd
import logging
from pathlib import Path

class MeetingTranscriber:
    def __init__(self, config_file=None):
        """
        Initialize the Meeting Transcriber
        
        Args:
            config_file (str): Path to configuration file (optional)
        """
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Setup output directories
        self._setup_directories()
        
        # Load models
        self._load_models()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'transcription.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_file):
        """Load configuration settings"""
        try:
            # Import the config module directly
            import config
            
            # Map the config variables to our expected format
            loaded_config = {
                'whisper_model': getattr(config, 'WHISPER_MODEL', 'large-v3'),
                'hf_token': getattr(config, 'HUGGING_FACE_TOKEN', 'YOUR_HUGGING_FACE_TOKEN'),
                'output_base_dir': str(getattr(config, 'OUTPUT_DIR', '/Users/harsmis/Downloads/transcription/output')),
                'audio_dir': str(getattr(config, 'AUDIO_DIR', '/Users/harsmis/Downloads/transcription/audio_files'))
            }
            
            self.logger.info("Successfully loaded config.py")
            return loaded_config
            
        except ImportError:
            self.logger.warning("Could not import config.py, using defaults")
            # Fallback to default config
            default_config = {
                'whisper_model': 'large-v3',
                'hf_token': 'YOUR_HUGGING_FACE_TOKEN',
                'output_base_dir': '/Users/harsmis/Downloads/transcription/output',
                'audio_dir': '/Users/harsmis/Downloads/transcription/audio_files'
            }
            return default_config
    
    def _setup_directories(self):
        """Create output directory structure"""
        base_dir = Path(self.config['output_base_dir'])
        
        self.output_dirs = {
            'transcripts': base_dir / 'transcripts',
            'detailed': base_dir / 'detailed', 
            'analysis': base_dir / 'analysis'
        }
        
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _load_models(self):
        """Load Whisper and diarization models"""
        self.logger.info("Loading Whisper model...")
        self.whisper_model = whisper.load_model(self.config['whisper_model'])
        
        self.logger.info("Loading speaker diarization model...")
        
        # Debug: show first few characters of token
        token = self.config['hf_token']
        self.logger.info(f"Using token: {token[:10]}...")
        
        # Check if token is set
        if token == 'YOUR_HUGGING_FACE_TOKEN':
            raise ValueError(
                "Please set your Hugging Face token in config.py!\n"
                "1. Get token from: https://huggingface.co/settings/tokens\n"
                "2. Accept model terms: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "3. Replace 'YOUR_HUGGING_FACE_TOKEN' in config.py with your actual token"
            )
        
        try:
            # Load diarization pipeline
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )
            self.logger.info("Models loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to load diarization model: {str(e)}")
            raise ValueError(
                f"Failed to load speaker diarization model.\n"
                f"Error: {str(e)}\n\n"
                f"Please ensure:\n"
                f"1. Your HuggingFace token is valid and has READ access\n"
                f"2. You've accepted the model license at:\n"
                f"   https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                f"3. Your token in config.py starts with 'hf_'"
            )
    
    def _convert_audio_if_needed(self, audio_path):
        """Convert audio to WAV format if needed"""
        audio_path = Path(audio_path)
        
        # If already WAV, return as-is
        if audio_path.suffix.lower() == '.wav':
            return str(audio_path)
        
        # Create temp WAV file
        temp_wav = audio_path.parent / f"{audio_path.stem}_temp.wav"
        
        try:
            self.logger.info(f"Converting {audio_path.suffix} to WAV format...")
            
            # Load audio and convert to WAV
            audio_data, sample_rate = librosa.load(str(audio_path), sr=None)
            sf.write(str(temp_wav), audio_data, sample_rate)
            
            self.logger.info(f"Converted to: {temp_wav}")
            return str(temp_wav)
            
        except Exception as e:
            self.logger.error(f"Audio conversion failed: {str(e)}")
            # Try to use original file anyway
            return str(audio_path)

    def transcribe_with_speakers(self, audio_path, output_name=None):
        """
        Transcribe audio file with speaker diarization
        
        Args:
            audio_path (str): Path to audio file
            output_name (str): Custom output name (optional)
        """
        self.logger.info(f"Processing: {audio_path}")
        
        # Convert audio if needed
        processed_audio_path = self._convert_audio_if_needed(audio_path)
        temp_file_created = processed_audio_path != audio_path
        
        try:
            # Generate output name
            if output_name is None:
                output_name = Path(audio_path).stem
            
            # Step 1: Speaker Diarization
            self.logger.info("Performing speaker diarization...")
            diarization = self.diarization_pipeline(processed_audio_path)
            
            # Step 2: Transcribe entire audio
            self.logger.info("Transcribing audio...")
            result = self.whisper_model.transcribe(
                processed_audio_path,
                language="en",
                task="transcribe"
            )
            
            # Step 3: Align transcription with speakers
            self.logger.info("Aligning transcription with speakers...")
            segments_with_speakers = self._align_transcription_with_speakers(
                result["segments"], diarization
            )
            
            # Step 4: Generate outputs
            self._save_all_outputs(segments_with_speakers, output_name)
            
            self.logger.info(f"Transcription complete! Files saved with prefix '{output_name}'")
            return segments_with_speakers
            
        finally:
            # Clean up temporary file if created
            if temp_file_created and os.path.exists(processed_audio_path):
                try:
                    os.remove(processed_audio_path)
                    self.logger.info("Cleaned up temporary WAV file")
                except Exception as e:
                    self.logger.warning(f"Could not remove temp file: {e}")
    
    def _align_transcription_with_speakers(self, whisper_segments, diarization):
        """
        Align Whisper transcription segments with speaker diarization
        """
        aligned_segments = []
        
        for segment in whisper_segments:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()
            
            # Find the most overlapping speaker for this segment
            speaker = self._find_dominant_speaker(start_time, end_time, diarization)
            
            aligned_segments.append({
                "start": start_time,
                "end": end_time,
                "duration": end_time - start_time,
                "speaker": speaker,
                "text": text,
                "confidence": segment.get("avg_logprob", 0)
            })
        
        return aligned_segments
    
    def _find_dominant_speaker(self, start_time, end_time, diarization):
        """
        Find the speaker who talks the most during a given time segment
        """
        segment_duration = end_time - start_time
        speaker_durations = {}
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Calculate overlap between the transcription segment and speaker turn
            overlap_start = max(start_time, turn.start)
            overlap_end = min(end_time, turn.end)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                if speaker not in speaker_durations:
                    speaker_durations[speaker] = 0
                speaker_durations[speaker] += overlap_duration
        
        if not speaker_durations:
            return "Unknown"
        
        # Return speaker with most overlap
        dominant_speaker = max(speaker_durations, key=speaker_durations.get)
        return f"Speaker_{dominant_speaker.split('_')[-1]}"
    
    def _save_all_outputs(self, segments, output_name):
        """Save all output formats"""
        # Save detailed JSON
        json_path = self.output_dirs['detailed'] / f"{output_name}_detailed.json"
        self._save_detailed_json(segments, json_path)
        
        # Save readable transcript
        txt_path = self.output_dirs['transcripts'] / f"{output_name}_transcript.txt"
        self._save_readable_transcript(segments, txt_path)
        
        # Save CSV for analysis
        csv_path = self.output_dirs['analysis'] / f"{output_name}_analysis.csv"
        self._save_csv_analysis(segments, csv_path)
    
    def _save_detailed_json(self, segments, output_path):
        """Save detailed transcription with timestamps and speakers"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved detailed JSON: {output_path}")
    
    def _save_readable_transcript(self, segments, output_path):
        """Save human-readable transcript"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("MEETING TRANSCRIPT\n")
            f.write("=" * 50 + "\n\n")
            
            current_speaker = None
            for segment in segments:
                speaker = segment["speaker"]
                text = segment["text"]
                timestamp = str(timedelta(seconds=int(segment["start"])))
                
                if speaker != current_speaker:
                    f.write(f"\n[{timestamp}] {speaker}:\n")
                    current_speaker = speaker
                
                f.write(f"{text}\n")
        self.logger.info(f"Saved transcript: {output_path}")
    
    def _save_csv_analysis(self, segments, output_path):
        """Save CSV for further analysis"""
        df = pd.DataFrame(segments)
        df['start_formatted'] = df['start'].apply(lambda x: str(timedelta(seconds=int(x))))
        df['end_formatted'] = df['end'].apply(lambda x: str(timedelta(seconds=int(x))))
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved analysis CSV: {output_path}")
    
    def process_folder(self, folder_path, supported_formats=('.wav', '.mp3', '.m4a', '.flac')):
        """Process all audio files in a folder"""
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(supported_formats):
                audio_path = os.path.join(folder_path, filename)
                try:
                    self.transcribe_with_speakers(audio_path)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

# Usage Example
def main():
    # Initialize transcriber
    transcriber = MeetingTranscriber()
    
    # Process a single file
    audio_file = "/Users/harsmis/Downloads/transcription/audio_files/meeting.wav"
    if os.path.exists(audio_file):
        transcriber.transcribe_with_speakers(audio_file)
    
    # Or process all files in the audio_files folder
    audio_folder = "/Users/harsmis/Downloads/transcription/audio_files"
    if os.path.exists(audio_folder):
        transcriber.process_folder(audio_folder)

if __name__ == "__main__":
    main()
