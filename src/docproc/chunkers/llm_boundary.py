from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from dataclasses import dataclass
from typing import Iterable

from ..clean import normalize_text, simple_tokenize, is_probable_heading
from .base import Chunk, Page


def _split_into_units(text: str) -> list[str]:
    """Split into sentence/heading-like units.

    - Keeps probable headings as standalone units.
    - Splits other lines into sentences on punctuation.
    """
    import re

    text = normalize_text(text)
    if not text:
        return []

    units: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if is_probable_heading(line):
            units.append(line)
            continue

        parts = re.split(r"(?<=[.!?])\s+", line)
        for p in parts:
            p = p.strip()
            if p:
                units.append(p)

    return units


def _count_tokens(text: str) -> int:
    return len(simple_tokenize(text))


def _load_dotenv(path: str = ".env") -> None:
    """Very small .env loader (key=value). Safe to call even if file missing."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        # Never hard-fail on dotenv parsing.
        return


@dataclass
class OllamaClient:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"
    timeout_s: int = 120

    def generate(self, prompt: str) -> str:
        """Call Ollama /api/generate (non-streaming)."""
        url = self.base_url.rstrip("/") + "/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # Make outputs as deterministic as possible.
            "options": {
                "temperature": 0,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            obj = json.loads(raw)
            return str(obj.get("response", ""))
        except Exception:
            return raw


@dataclass
class LLMBoundaryChunker:
    """Chunker that uses an LLM (via Ollama) to pick better chunk boundaries.

    Algorithm:
      - Split pages into sentence/heading-like units.
      - Greedily grow a chunk until we would exceed max_tokens.
      - Ask the LLM to choose the best boundary near the token limit from a small candidate window.
      - Fallback to a heuristic boundary if LLM output is invalid.

    Notes:
      - This is intentionally dependency-free and should work with local Ollama.
      - Token counting uses simple_tokenize() (approximate, but consistent across experiments).
    """

    client: OllamaClient
    max_tokens: int = 450
    min_tokens: int = 80
    boundary_window: int = 6  # candidates: end-window .. end+window
    max_retries: int = 2
    retry_backoff_s: float = 0.8

    def _choose_boundary(self, units: list[str], start: int, soft_end: int) -> int:
        """Return an end index (exclusive) for the chunk."""
        # Candidate region
        lo = max(start + 1, soft_end - self.boundary_window)
        hi = min(len(units), soft_end + self.boundary_window)

        # Ensure we have room for min_tokens.
        def chunk_tokens(end_idx: int) -> int:
            return _count_tokens(" ".join(units[start:end_idx]))

        # Build candidates with token counts.
        candidates = []
        for end in range(lo, hi + 1):
            tok = chunk_tokens(end)
            if tok < self.min_tokens:
                continue
            candidates.append((end, tok))

        if not candidates:
            # Fallback: move forward until min_tokens reached.
            end = max(start + 1, soft_end)
            while end < len(units) and chunk_tokens(end) < self.min_tokens:
                end += 1
            return min(end, len(units))

        # Prefer candidates that are <= max_tokens but close to it.
        within = [c for c in candidates if c[1] <= self.max_tokens]
        if within:
            soft_pick = max(within, key=lambda x: x[1])[0]
        else:
            # all overflow; pick smallest overflow
            soft_pick = min(candidates, key=lambda x: x[1])[0]

        # Prepare prompt excerpt.
        excerpt_units = units[start : min(hi, start + 120)]  # safety bound
        # Provide candidate indices relative to start for easier reasoning.
        cand_rel = [{"end_rel": end - start, "tokens": tok} for end, tok in candidates]

        prompt = (
            "You are chunking a clinical guideline into coherent passages for retrieval.\n"
            "Pick the BEST chunk boundary (end index) so that:\n"
            " - the chunk is semantically coherent (do not cut mid-idea),\n"
            f" - token length is close to {self.max_tokens} but not too much above it,\n"
            f" - token length is at least {self.min_tokens}.\n\n"
            "Return ONLY valid JSON on one line with keys: end_rel (integer), rationale (string).\n"
            "Do NOT include markdown or extra text.\n\n"
            "UNITS (each entry is a sentence or heading):\n"
        )

        # Number the excerpt units.
        numbered = []
        for i, u in enumerate(excerpt_units, start=0):
            # Keep units short-ish for the prompt.
            uu = u.strip()
            if len(uu) > 240:
                uu = uu[:240] + "…"
            numbered.append(f"{i}: {uu}")
        prompt += "\n".join(numbered) + "\n\n"
        prompt += "CANDIDATES (end index relative to start, exclusive):\n"
        prompt += json.dumps(cand_rel, ensure_ascii=False) + "\n"
        prompt += f"SOFT_PICK_END_REL: {soft_pick - start}\n"

        # Call LLM with retries.
        for attempt in range(self.max_retries + 1):
            try:
                out = self.client.generate(prompt).strip()
                obj = json.loads(out)
                end_rel = int(obj["end_rel"])
                end_abs = start + end_rel
                if end_abs <= start:
                    raise ValueError("end_rel <= 0")
                if end_abs > len(units):
                    raise ValueError("end_abs out of bounds")
                # Validate token constraints loosely.
                tok = chunk_tokens(end_abs)
                if tok < self.min_tokens:
                    raise ValueError("below min_tokens")
                # Allow slight overflow.
                if tok > self.max_tokens * 1.35:
                    raise ValueError("too far above max_tokens")
                return end_abs
            except Exception:
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_s * (attempt + 1) + random.random() * 0.1)

        return soft_pick

    def chunk(self, pages: Iterable[Page]) -> list[Chunk]:
        units: list[str] = []
        unit_pages: list[int] = []

        for p in pages:
            u = _split_into_units(p.text)
            units.extend(u)
            unit_pages.extend([p.page] * len(u))

        chunks: list[Chunk] = []
        i = 0
        n = len(units)
        chunk_idx = 0

        while i < n:
            # Greedy soft end
            soft_end = i + 1
            while soft_end < n:
                tok = _count_tokens(" ".join(units[i:soft_end]))
                if tok >= self.max_tokens:
                    break
                soft_end += 1

            # If we never reached max_tokens, take the rest.
            if soft_end >= n:
                end = n
            else:
                end = self._choose_boundary(units, i, soft_end)

            text = " ".join(units[i:end]).strip()
            if not text:
                i = max(i + 1, end)
                continue

            page_start = int(unit_pages[i])
            page_end = int(unit_pages[end - 1])

            chunk_idx += 1
            chunks.append(
                Chunk(
                    chunk_id=f"llm_{chunk_idx:05d}",
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    section_title=None,
                    strategy="llm_boundary",
                    chunk_size=self.max_tokens,
                    overlap=0,
                )
            )

            i = end

        return chunks


def build_ollama_client_from_env() -> OllamaClient:
    _load_dotenv()
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    timeout_s = int(os.getenv("OLLAMA_TIMEOUT_S", "120"))
    return OllamaClient(base_url=base_url, model=model, timeout_s=timeout_s)
