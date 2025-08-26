"""Utilities for handling uploaded media files."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from fastapi import UploadFile


def save_upload_to_disk(upload: UploadFile, upload_dir: Path) -> Path:
  """Save an uploaded file to disk with a unique name.

  Args:
    upload: FastAPI UploadFile.
    upload_dir: Base directory to save files.

  Returns:
    Path to the saved file.
  """
  suffix = Path(upload.filename or "uploaded").suffix or ""
  unique_name = f"{uuid.uuid4().hex}{suffix}"
  dest = upload_dir / unique_name
  with dest.open("wb") as f:
    shutil.copyfileobj(upload.file, f)
  return dest
