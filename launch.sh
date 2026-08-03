#!/bin/bash

AUDIO_FILE="$(pwd)/Materials/sample_recording.wav"
TRANSCRIPTION_FILE="$(pwd)/Materials/sample_transcription.txt"

# URL-encode the file paths
ENCODED_AUDIO_FILE=$(python3 -c 'import urllib.parse; print(urllib.parse.quote("$AUDIO_FILE"))')
ENCODED_TRANSCRIPTION_FILE=$(python3 -c 'import urllib.parse; print(urllib.parse.quote("$TRANSCRIPTION_FILE"))')

# Construct the URL
URL="http://localhost:3000/?audio=file://${ENCODED_AUDIO_FILE}&transcription=file://${ENCODED_TRANSCRIPTION_FILE}"

# Open the URL in the default web browser
x-www-browser "$URL" || google-chrome "$URL" || firefox "$URL"
