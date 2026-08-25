#!/bin/bash

# Download audio from YouTube to data/audio/pilot/
# Usage: ./download_audio.sh <youtube_url> <output_filename>

URL=$(wl-paste)

END="_opus.webm"
FILENAME=$1$END

if [ -z "$URL" ] || [ -z "$FILENAME" ]; then
    echo "Usage: $0 <youtube_url> <output_filename>"
    exit 1
fi

# Ensure directory exists
mkdir -p data/audio/pilot/

# Run yt-dlp with project-standard flags
yt-dlp -f "251/bestaudio" \
       --cookies-from-browser firefox \
       --extractor-args "youtube:remote_components=ejs:github" \
       --no-warnings \
       -o "data/audio/pilot/$FILENAME" \
       "$URL"
