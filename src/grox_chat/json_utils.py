"""Shared JSON extraction utilities used by agents, rag, server, and librarian."""

import json
from typing import Optional


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from text."""
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _find_json_candidates(text: str) -> list[str]:
    """Find brace-balanced substrings, longest first."""
    candidates = []
    for i, ch in enumerate(text):
        if ch != '{':
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            if depth == 0:
                candidates.append(text[i:j + 1])
                break
    candidates.sort(key=len, reverse=True)
    return candidates


def extract_json_object(text: str) -> Optional[dict]:
    """Extract a JSON object from text, handling markdown fences and multiple candidates.

    Returns the first valid dict found, or None.
    """
    if not text:
        return None

    stripped = strip_markdown_fences(text)

    # Try direct parse first
    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Brace-balanced fallback: try all candidates from longest to shortest
    for candidate in _find_json_candidates(stripped):
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            continue

    return None
