# Operational Instructions

This document captures configuration and operational guidance to avoid repeated mistakes.

## Environment

- `OPENAI_API_KEY`: Required for transcription and summarization.
- `APP_DEBUG` (optional): `true|false` for verbose traceback logging on server errors.
- `UPLOAD_DIR` (optional): Defaults to `uploads`.
- `KEEP_UPLOADS` (optional): Keep uploaded files after processing. Defaults to `true`.

## Error Handling

- Upload errors or provider failures return HTTP 400 with `detail` message.
- With `APP_DEBUG=true`, server logs include tracebacks for easier debugging.

## Process Flow

1. Upload file to `/api/process`.
2. File is saved under `UPLOAD_DIR`.
3. Transcription uses OpenAI (Whisper) with local speaker diarization via MFCC clustering.
4. Summarization leverages OpenAI (`gpt-4o-mini`) with strict JSON output.
5. Response includes: transcript text, diarized utterances, summary, meeting notes, outcomes.

## Customization

- Prompts for summarization are in `app/services/summarize_openai.py` (`_prompt_json_instruction`, `_build_user_prompt`).
- To keep uploads off disk, set `KEEP_UPLOADS=false`.

## Dependencies

- Ensure `pip install -r requirements.txt` is run in an activated virtualenv.
- Optional system package: `ffmpeg` recommended for extracting audio from video containers (mp4/mov) when needed.
- Open-source Python libs for diarization/audio: `numpy`, `scipy`, `scikit-learn`, `librosa`, `soundfile`, `audioread`.

## Execution

- Local run: `uvicorn app.main:app --reload --port 8000`.
- If port conflict occurs, change `--port`.

## Troubleshooting

- Missing API keys -> 400 errors. Set env vars and restart server.
- Large files -> consider increasing client timeout; server handles chunking for summarization, but extremely long transcripts may still be too big.
- Network/firewall issues can block API calls to providers.
