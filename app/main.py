"""FastAPI application entrypoint for A/V transcription and summarization."""

from __future__ import annotations

import traceback
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import Settings, ensure_directories
from app.models.schemas import ErrorResponse, ProcessResponse
from app.services.media_utils import save_upload_to_disk
from app.services.transcribe_openai import transcribe_file as openai_transcribe
from app.services.summarize_openai import summarize_transcript

settings = Settings.from_env()
ensure_directories(settings)

app = FastAPI(title="ClearOps A/V Transcriber", version="0.1.0")

# CORS (allow localhost by default)
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Static and templates
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/")
async def index(request: Request):
  return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/process", response_model=ProcessResponse, responses={400: {"model": ErrorResponse}})
async def process_file(file: UploadFile = File(...)):
  """Upload endpoint: saves file, transcribes with diarization, summarizes, and returns structured results."""
  if not settings.openai_api_key:
    raise HTTPException(status_code=400, detail="Missing OPENAI_API_KEY in environment")

  saved_path = save_upload_to_disk(file, settings.upload_dir)

  try:
    # 1) Transcribe with diarization (OpenAI Whisper + local clustering)
    tr = openai_transcribe(str(saved_path), settings.openai_api_key)

    # 2) Summarize and generate notes/outcomes
    sm = summarize_transcript(tr.text, tr.utterances, settings.openai_api_key)

    response = ProcessResponse(transcription=tr, summary=sm)
    return JSONResponse(content=response.model_dump())
  except Exception as e:
    if settings.debug:
      traceback.print_exc()
    raise HTTPException(status_code=400, detail=str(e))
  finally:
    if not settings.keep_uploads:
      try:
        saved_path.unlink(missing_ok=True)
      except Exception:
        pass
