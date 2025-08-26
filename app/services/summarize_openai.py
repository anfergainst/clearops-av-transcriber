"""Summarization and meeting notes using OpenAI models."""

from __future__ import annotations

from typing import List

from openai import OpenAI

from app.models.schemas import ActionItem, SummaryResult, Utterance


MAX_CHARS_PER_CHUNK = 6000


def _chunks(text: str, size: int) -> List[str]:
  return [text[i : i + size] for i in range(0, len(text), size)]


def _prompt_json_instruction() -> str:
  return (
    "You are an expert meeting analyst. Read the transcript and output concise, "
    "high-signal results.\n"
    "Always return strict JSON with keys: summary (string), meeting_notes (string), "
    "outcomes (array of objects with keys: owner, task, due, priority).\n"
    "If it's not a meeting, still provide a relevant summary and put outcomes as an "
    "empty array.\n"
    "Do not include any text outside the JSON."
  )


def _build_user_prompt(full_text: str, utterances: List[Utterance]) -> str:
  """Build the user prompt with transcript and diarization context."""
  header = (
    "Transcript below with speaker labels. Provide: (1) a concise summary; (2) "
    "structured meeting notes (topics, decisions); (3) outcomes as to-do items "
    "if applicable.\n\n"
  )
  # Keep the prompt straightforward; the JSON shape is enforced by system msg.
  diarization_preview_lines = []
  for u in utterances[:20]:  # include only first 20 utterances to keep prompt lean
    diarization_preview_lines.append(f"{u.speaker}: {u.text}")
  diarization_preview = "\n".join(diarization_preview_lines)

  return (
    f"{header}Speaker preview (first 20):\n{diarization_preview}\n\nFull transcript:\n"
    f"{full_text}"
  )


def summarize_transcript(text: str, utterances: List[Utterance], api_key: str) -> SummaryResult:
  """Summarize the transcript and produce meeting notes/outcomes.

  Uses a simple chunking strategy to handle very long transcripts.
  """
  client = OpenAI(api_key=api_key)

  if len(text) <= MAX_CHARS_PER_CHUNK:
    user_prompt = _build_user_prompt(text, utterances)
    resp = client.chat.completions.create(
      model="gpt-4o-mini",
      temperature=0.2,
      messages=[
        {"role": "system", "content": _prompt_json_instruction()},
        {"role": "user", "content": user_prompt},
      ],
      response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
  else:
    # Multi-step: summarize chunks first, then combine.
    partial_summaries: List[str] = []
    for chunk in _chunks(text, MAX_CHARS_PER_CHUNK):
      user_prompt = _build_user_prompt(chunk, [])
      resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
          {"role": "system", "content": "Summarize this transcript chunk in 6-10 bullet points. Return plain text."},
          {"role": "user", "content": user_prompt},
        ],
      )
      partial_summaries.append(resp.choices[0].message.content or "")

    combined_text = "\n".join(partial_summaries)
    final_prompt = (
      "Combine the bullet-point summaries into a single coherent summary and meeting notes.\n"
      "Return strict JSON with keys: summary, meeting_notes, outcomes (array of {owner, task, due, priority})."
    )
    user_content = f"{final_prompt}\n\nSummaries to combine:\n{combined_text}"
    resp = client.chat.completions.create(
      model="gpt-4o-mini",
      temperature=0.2,
      messages=[
        {"role": "system", "content": _prompt_json_instruction()},
        {"role": "user", "content": user_content},
      ],
      response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content

  # Parse JSON result safely
  import json

  try:
    data = json.loads(content or "{}")
  except Exception:
    # Fallback: wrap raw content
    data = {"summary": content or "", "meeting_notes": "", "outcomes": []}

  outcomes = []
  for item in data.get("outcomes", []) or []:
    outcomes.append(
      ActionItem(
        owner=item.get("owner") or None,
        task=item.get("task") or "",
        due=item.get("due") or None,
        priority=item.get("priority") or None,
      )
    )

  return SummaryResult(
    summary=data.get("summary") or "",
    meeting_notes=data.get("meeting_notes") or "",
    outcomes=outcomes,
  )
