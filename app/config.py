"""Application configuration utilities.
All environment variables are read here.
All comments and docstrings in this codebase are in English,
per project guidelines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
  """Holds application settings from environment variables."""

  debug: bool = False
  upload_dir: Path = Path("uploads")
  openai_api_key: str | None = None
  keep_uploads: bool = True

  @staticmethod
  def from_env() -> "Settings":
    debug = os.getenv("APP_DEBUG", "false").lower() in {"1", "true", "yes", "on"}
    upload_dir = Path(os.getenv("UPLOAD_DIR", "uploads"))
    keep_uploads = (
      os.getenv("KEEP_UPLOADS", "true").lower()
      in {"1", "true", "yes", "on"}
    )

    return Settings(
      debug=debug,
      upload_dir=upload_dir,
      openai_api_key=os.getenv("OPENAI_API_KEY"),
      keep_uploads=keep_uploads,
    )


def ensure_directories(settings: Settings) -> None:
  """Create required directories if they do not exist."""
  settings.upload_dir.mkdir(parents=True, exist_ok=True)
