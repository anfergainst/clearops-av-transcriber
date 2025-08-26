"""OpenAI Whisper-based transcription with local speaker diarization via MFCC clustering.

This module uses the OpenAI Audio Transcriptions API to obtain transcript text and time-stamped
segments. It then loads the corresponding audio, computes MFCC-based embeddings for each segment,
and clusters them to assign stable speaker labels (Person 1, Person 2, ...).

Dependencies are all open-source: numpy, librosa, soundfile, scikit-learn. No paid services
besides OpenAI are used.
"""

from __future__ import annotations

import json
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from openai import OpenAI

from app.models.schemas import TranscriptionResult, Utterance


# Audio parameters
TARGET_SR = 16000


@dataclass
class WhisperSegment:
  start: float
  end: float
  text: str


def _seg_field(seg: object, name: str, default: object) -> object:
  """Safely extract a field from a segment that may be a dict or SDK object.

  Returns default if the field is missing or an error occurs.
  """
  try:
    if isinstance(seg, dict):
      return seg.get(name, default)
    return getattr(seg, name, default)
  except Exception:
    return default


def _openai_transcribe(file_path: str, api_key: str) -> Tuple[str, List[WhisperSegment]]:
  """Call OpenAI audio transcription API and return text + segments.

  Tries modern model first, falls back to whisper-1 if needed.
  """
  client = OpenAI(api_key=api_key)
  models_to_try = ["gpt-4o-mini-transcribe", "whisper-1"]

  last_err: Optional[Exception] = None
  for model in models_to_try:
    try:
      with open(file_path, "rb") as f:
        resp = client.audio.transcriptions.create(
          model=model,
          file=f,
          response_format="verbose_json",
          temperature=0,
        )
      # Parse response robustly across SDK versions
      text = getattr(resp, "text", None)
      segments_data = getattr(resp, "segments", None)
      if text is None or segments_data is None:
        # Fallback to dict/json payload
        try:
          data = resp.model_dump() if hasattr(resp, "model_dump") else json.loads(resp.model_dump_json())
        except Exception:
          # Final fallback: try .json() or str(resp)
          try:
            data = json.loads(resp.json())
          except Exception:
            data = json.loads(str(resp)) if str(resp).startswith("{") else {}
        text = data.get("text") or ""
        segments_data = data.get("segments") or []
      segments: List[WhisperSegment] = []
      for seg in segments_data:
        # Support both dict-like and object-like segment entries
        start_val = _seg_field(seg, "start", 0.0)
        start = float(start_val or 0.0)
        end_val = _seg_field(seg, "end", start)
        end = float(end_val if end_val is not None else start)
        seg_text_val = _seg_field(seg, "text", "")
        seg_text = str(seg_text_val or "")
        if end < start:
          end = start
        segments.append(WhisperSegment(start=start, end=end, text=seg_text))
      return text, segments
    except Exception as e:
      last_err = e
      continue

  raise RuntimeError(f"OpenAI transcription failed: {last_err}")


def _load_audio(file_path: str) -> Tuple[np.ndarray, int]:
  """Load audio as mono waveform using librosa. If direct load fails (e.g., mp4),
  try ffmpeg to convert to wav in a temp file.
  """
  try:
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    return y, sr
  except Exception:
    # Fallback via ffmpeg
    tmp_wav = Path(tempfile.gettempdir()) / f"transcode_{Path(file_path).stem}_{TARGET_SR}.wav"
    try:
      cmd = [
        "ffmpeg",
        "-y",
        "-i",
        file_path,
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        str(tmp_wav),
      ]
      subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      y, sr = librosa.load(str(tmp_wav), sr=TARGET_SR, mono=True)
      return y, sr
    except Exception as e:
      raise RuntimeError(
        "Failed to load audio. Ensure the file is an audio/video with a valid audio track. "
        "Installing ffmpeg may help for mp4/mov containers."
      ) from e


def _mfcc_features(y: np.ndarray, sr: int) -> np.ndarray:
  """Compute MFCC+delta mean/std feature vector for a waveform."""
  if y.size == 0:
    # 20 MFCC means + 20 MFCC stds + 20 delta means + 20 delta stds = 80
    return np.zeros(80, dtype=np.float32)
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
  delta = librosa.feature.delta(mfcc)
  feats = np.concatenate([
    mfcc.mean(axis=1), mfcc.std(axis=1),
  ])
  deltas = np.concatenate([
    delta.mean(axis=1), delta.std(axis=1),
  ])
  vec = np.concatenate([feats, deltas]).astype(np.float32)
  return vec


def _extract_segment_wave(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
  i0 = int(max(0.0, start) * sr)
  i1 = int(max(0.0, end) * sr)
  i1 = max(i1, i0 + 1)
  return y[i0:i1]


def _cluster_speakers(vectors: List[np.ndarray]) -> List[int]:
  """Cluster feature vectors and return speaker labels as integers 0..K-1.

  Uses silhouette score to pick K between 1 and min(8, n).
  """
  n = len(vectors)
  if n == 0:
    return []
  if n == 1:
    return [0]

  X = np.stack(vectors)

  best_k = 1
  best_score = -1.0
  max_k = min(8, n)
  labels_best = [0] * n

  for k in range(2, max_k + 1):
    try:
      km = KMeans(n_clusters=k, n_init=10, random_state=0)
      labels = km.fit_predict(X)
      if len(set(labels)) < 2:
        continue
      score = silhouette_score(X, labels)
      if score > best_score:
        best_score = score
        best_k = k
        labels_best = labels.tolist()
    except Exception:
      continue

  if best_k == 1:
    return [0] * n
  return labels_best


def _label_map_from_first_seen(labels: List[int]) -> Dict[int, str]:
  mapping: Dict[int, str] = {}
  counter = 1
  for lab in labels:
    if lab not in mapping:
      mapping[lab] = f"Person {counter}"
      counter += 1
  return mapping


def transcribe_file(file_path: str, api_key: str) -> TranscriptionResult:
  """Transcribe audio/video via OpenAI and assign speaker labels with local clustering.

  Args:
    file_path: path to media file
    api_key: OpenAI API key
  """
  text, segments = _openai_transcribe(file_path, api_key)

  # If no segments, return single-utterance transcript
  if not segments:
    utt = Utterance(speaker="Person 1", start_ms=0, end_ms=0, text=text)
    return TranscriptionResult(text=text, utterances=[utt])

  # Load audio and compute features per segment
  y, sr = _load_audio(file_path)
  vecs: List[np.ndarray] = []
  for seg in segments:
    yw = _extract_segment_wave(y, sr, seg.start, seg.end)
    vecs.append(_mfcc_features(yw, sr))

  labels = _cluster_speakers(vecs)
  label_map = _label_map_from_first_seen(labels)

  utterances: List[Utterance] = []
  for seg, lab in zip(segments, labels):
    speaker = label_map.get(lab, "Person 1")
    utterances.append(
      Utterance(
        speaker=speaker,
        start_ms=int(seg.start * 1000),
        end_ms=int(seg.end * 1000),
        text=seg.text.strip(),
      )
    )

  return TranscriptionResult(text=text, utterances=utterances)
