"""Pydantic models (schemas) for API requests and responses."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Utterance(BaseModel):
  """Represents a speaker-labelled utterance with timing."""

  speaker: str = Field(..., description="Normalized speaker label e.g., 'Person 1'")
  start_ms: int = Field(..., description="Start time in milliseconds")
  end_ms: int = Field(..., description="End time in milliseconds")
  text: str = Field(..., description="Utterance text")


class TranscriptionResult(BaseModel):
  """Full transcription with diarization segments."""

  text: str
  utterances: List[Utterance]


class ActionItem(BaseModel):
  """Structured to-do/action item."""

  owner: Optional[str] = Field(None, description="Owner name or 'Unassigned'")
  task: str
  due: Optional[str] = Field(None, description="Due date or timeframe if present")
  priority: Optional[str] = Field(None, description="low|medium|high if inferred")


class SummaryResult(BaseModel):
  """Summarization and meeting notes."""

  summary: str
  meeting_notes: str
  outcomes: List[ActionItem] = Field(default_factory=list)


class ProcessResponse(BaseModel):
  """Response payload for /api/process."""

  transcription: TranscriptionResult
  summary: SummaryResult


class ErrorResponse(BaseModel):
  detail: str
