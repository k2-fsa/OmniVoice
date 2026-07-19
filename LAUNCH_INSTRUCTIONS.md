# Hebrew Transcription App - Launch Instructions

Welcome to the Hebrew Transcription App! This guide will help you set up and launch the application with sample Hebrew audio and transcription files.

## Project Structure

The project is organized as follows:

```
hebrew-transcription-app/
├── Materials/                      # Folder containing sample files
│   ├── sample_recording.wav       # Hebrew audio sample
│   └── sample_transcription.txt   # Hebrew transcription sample
├── launch.sh                       # Launch script for Linux/macOS
├── launch.bat                      # Launch script for Windows
├── client/                         # React frontend source code
├── server/                         # Backend server (placeholder)
├── package.json                    # Project dependencies
└── LAUNCH_INSTRUCTIONS.md         # This file
```

## Prerequisites

Before launching the application, ensure you have the following installed:

- **Node.js** (version 18 or higher)
- **pnpm** (package manager)
- **Web Browser** (Chrome, Firefox, Safari, or Edge)

## Setup Instructions

### Step 1: Install Dependencies

Navigate to the project directory and install all required dependencies:

```bash
cd hebrew-transcription-app
pnpm install
```

### Step 2: Start the Development Server

Run the development server:

```bash
pnpm dev
```

The server will start on `http://localhost:3000/`. You should see output similar to:

```
VITE v7.1.9  ready in 497 ms
➜  Local:   http://localhost:3000/
➜  Network: http://169.254.0.21:3000/
```

Keep this terminal window open while using the application.

## Launching with Sample Files

Once the development server is running, use one of the following launch scripts to automatically load the sample Hebrew audio and transcription.

### For Linux/macOS Users

Run the launch script:

```bash
./launch.sh
```

This script will:
1. Locate the sample audio file (`Materials/sample_recording.wav`)
2. Locate the sample transcription file (`Materials/sample_transcription.txt`)
3. Open your default web browser with the application
4. Automatically load both files into the interface

### For Windows Users

Double-click the batch file:

```
launch.bat
```

Alternatively, run it from Command Prompt:

```cmd
launch.bat
```

This script will:
1. Locate the sample audio file (`Materials\sample_recording.wav`)
2. Locate the sample transcription file (`Materials\sample_transcription.txt`)
3. Open your default web browser with the application
4. Automatically load both files into the interface

## Using Your Own Files

To use your own Hebrew audio and transcription files:

1. Replace the sample files in the `Materials/` folder with your own files:
   - `sample_recording.wav` → Replace with your audio file (`.wav`, `.mp3`, etc.)
   - `sample_transcription.txt` → Replace with your transcription file (`.txt`)

2. Run the appropriate launch script for your operating system.

Alternatively, you can manually pass file paths as URL parameters:

```
http://localhost:3000/?audio=file:///path/to/your/audio.wav&transcription=file:///path/to/your/transcription.txt
```

## Application Features

The Hebrew Transcription App provides the following features:

- **Audio Playback**: Play Hebrew audio recordings with standard browser controls (play, pause, volume, progress bar)
- **Transcription Display**: View the Hebrew transcription alongside the audio player
- **Automatic Loading**: Automatically load audio and transcription files via URL parameters
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## Troubleshooting

### Issue: Browser doesn't open automatically

**Solution**: Manually navigate to `http://localhost:3000/` in your web browser.

### Issue: Audio file doesn't load

**Solution**: Ensure the file path is correct and the file format is supported by your browser (`.wav`, `.mp3`, `.ogg`, `.flac`).

### Issue: Transcription text doesn't appear

**Solution**: Verify that the transcription file exists and is in `.txt` format with proper encoding (UTF-8 recommended for Hebrew text).

### Issue: Development server won't start

**Solution**: 
1. Ensure Node.js and pnpm are installed: `node --version` and `pnpm --version`
2. Delete `node_modules` and `pnpm-lock.yaml`, then run `pnpm install` again
3. Try a different port: `pnpm dev -- --port 3001`

## Sample Files

The project includes sample Hebrew files for testing:

- **sample_recording.wav**: A Hebrew audio recording with the text "שלום, זה קובץ תמלול לדוגמה בעברית..."
- **sample_transcription.txt**: The corresponding Hebrew transcription

These files are automatically loaded when you run the launch scripts.

## Development

For more information about the project structure and development workflow, refer to the main `README.md` file in the project root.

## Support

If you encounter any issues or have questions, please refer to the troubleshooting section above or consult the project documentation.

---

**Happy transcribing!** 🎙️
