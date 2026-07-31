# Hebrew Transcription App - Quick Start Guide

## Setup Instructions

### Step 1: Start the Development Server
1. Open PowerShell or Command Prompt in the `OmniVoice` folder.
2. Run: `pnpm install` (if you haven't already).
3. Run: `pnpm dev`.
4. Keep this terminal open. The server runs on `http://localhost:3000`.

### Step 2: Launch the App
1. Double-click `launch.bat` in the `OmniVoice` folder, or manually visit `http://localhost:3000`.
2. The app will open in your default browser.

### Step 3: Load Your Files (New!)
The app now uses a secure file upload interface to bypass browser security restrictions:

1. **Audio File**: Drag and drop your `.wav`, `.mp3`, `.ogg`, or `.flac` file into the blue "Audio File" box, or click it to browse.
2. **Transcription File**: Drag and drop your `.txt` file into the green "Transcription File" box, or click it to browse.

Sample files are located in the `Materials` folder:
- `Materials/sample_recording.wav`
- `Materials/sample_transcription.txt`

## Why the Change?
Modern browsers block the app from automatically reading files from your hard drive via the URL (CORS security). The new upload interface is the standard, secure way to load local files into a web application.

## Troubleshooting
- **"pnpm not recognized"**: Ensure pnpm is installed, or use `npm run dev`.
- **"Files won't load"**: Make sure you are using supported file types (.wav/.mp3 for audio, .txt for text).
- **"Port 3000 in use"**: The app might be running on a different port (check the terminal output).
