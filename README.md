# ClearOps A/V Transcriber

Upload audio/video files, get full transcription with speaker diarization, a concise summary, meeting notes, and outcomes/to-dos.

## Features

- Speaker diarization (Person 1, Person 2, ... if unknown)
- Full transcript text + utterance-level timing
- Summary + Meeting Notes
- Outcomes / To-Dos (structured)
- Simple web UI (TailwindCSS) + JSON API

## Requirements

- Python 3.10+
- `OPENAI_API_KEY` (used for transcription and summarization)
- `ffmpeg` recommended for handling video containers (mp4/mov) via audio extraction fallback

## Quickstart

```bash
# From the project root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set environment variables (macOS/Linux bash)
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
# Optional
export APP_DEBUG=true
export KEEP_UPLOADS=true
export UPLOAD_DIR="uploads"

# Run the API
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 and upload your .mp3, .wav, .mp4, etc.

You can test with the sample `.mp3` files already present in this folder by uploading them via the UI.

## API

- `POST /api/process` with form-data `file`: returns JSON with `transcription` and `summary`.

## Notes on Security and Limits

- Do not commit your API keys.
- For very long transcripts, the app chunks and combines summaries, but extremely long media may exceed model limits.

## Dev Tips

- Code style: PEP8 (2-space indent in this project). Add linters/formatters as you wish (e.g., Black, Ruff).
- Error handling: Exceptions are returned as 400 responses with details.
