#!/usr/bin/env bash
set -euo pipefail

AUDIO_DIR="audio_files"
PY="python3"
SCRIPT="quick_transcribe.py"

files=(
  "audio-file-01.m4a"
  "another-audio-02.m4a"
  "one-more-file-03.m4a"
)

for f in "${files[@]}"; do
  path="$AUDIO_DIR/$f"
  if [[ -f "$path" ]]; then
    echo "Transcribing: $path"
    $PY "$SCRIPT" "$path"
  else
    echo "Skip. Not found: $path" >&2
  fi
done

echo "All requested transcriptions finished."
