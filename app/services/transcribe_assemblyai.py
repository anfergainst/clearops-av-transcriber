"""AssemblyAI transcription service with speaker diarization."""

from __future__ import annotations

from typing import Dict, List

from assemblyai import Transcriber, TranscriptionConfig, settings as aai_settings

from app.models.schemas import TranscriptionResult, Utterance


def _normalize_speaker_labels(raw_speakers: List[str]) -> Dict[str, str]:
  """Create a stable mapping from raw speaker IDs to 'Person N'."""
  mapping: Dict[str, str] = {}
  counter = 1
  for spk in raw_speakers:
    if spk not in mapping:
      mapping[spk] = f"Person {counter}"
      counter += 1
  return mapping


def transcribe_file(file_path: str, api_key: str) -> TranscriptionResult:
  """Transcribe with diarization using AssemblyAI.

  Args:
    file_path: Path to media file supported by AssemblyAI.
    api_key: AssemblyAI API key.

  Returns:
    TranscriptionResult with text and utterances.
  """
  aai_settings.api_key = api_key

  config = TranscriptionConfig(
    speaker_labels=True,
    punctuate=True,
    format_text=True,
    disfluencies=False,
    language_detection=True,
  )

  transcriber = Transcriber()
  transcript = transcriber.transcribe(file_path, config=config)

  if transcript.error:
    raise RuntimeError(f"AssemblyAI transcription error: {transcript.error}")

  # Collect unique raw speaker IDs in order of appearance
  raw_speakers: List[str] = []
  utterances_out: List[Utterance] = []

  if transcript.utterances:
    for utt in transcript.utterances:
      raw = str(getattr(utt, "speaker", "Unknown"))
      if raw not in raw_speakers:
        raw_speakers.append(raw)

    mapping = _normalize_speaker_labels(raw_speakers)

    for utt in transcript.utterances:
      speaker_raw = str(getattr(utt, "speaker", "Unknown"))
      speaker = mapping.get(speaker_raw, speaker_raw or "Person")
      utterances_out.append(
        Utterance(
          speaker=speaker,
          start_ms=int(getattr(utt, "start", 0)),
          end_ms=int(getattr(utt, "end", 0)),
          text=getattr(utt, "text", ""),
        )
      )

  return TranscriptionResult(text=getattr(transcript, "text", ""), utterances=utterances_out)
