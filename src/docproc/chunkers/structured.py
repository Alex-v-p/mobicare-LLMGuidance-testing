from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..clean import is_probable_heading, normalize_text, simple_tokenize, detokenize
from .base import Chunk, Page


@dataclass
class StructuredHeadingChunker:
    """Chunk by detected headings, with a fallback max token length.

    Many PDFs have inconsistent formatting, so this is heuristic.
    """

    max_tokens: int = 800
    min_tokens: int = 120

    def chunk(self, pages: Iterable[Page]) -> list[Chunk]:
        chunks: list[Chunk] = []

        current_title: str | None = None
        buffer_tokens: list[str] = []
        buffer_page_start: int | None = None
        buffer_page_end: int | None = None
        section_i = 0

        def flush() -> None:
            nonlocal buffer_tokens, buffer_page_start, buffer_page_end, section_i
            if not buffer_tokens or buffer_page_start is None or buffer_page_end is None:
                buffer_tokens = []
                buffer_page_start = None
                buffer_page_end = None
                return

            # If the section is huge, sub-chunk it (still “structured”).
            tokens = buffer_tokens
            start = 0
            part_i = 0
            while start < len(tokens):
                end = min(len(tokens), start + self.max_tokens)
                part = tokens[start:end]
                if len(part) < self.min_tokens and part_i > 0:
                    # attach small tail to previous chunk
                    break
                chunk_text = detokenize(part)
                cid = f"s{section_i:04d}_c{part_i:04d}"
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        text=chunk_text,
                        page_start=buffer_page_start,
                        page_end=buffer_page_end,
                        section_title=current_title,
                        strategy="structured_headings",
                    )
                )
                part_i += 1
                start += self.max_tokens

            section_i += 1
            buffer_tokens = []
            buffer_page_start = None
            buffer_page_end = None

        for p in pages:
            text = normalize_text(p.text)
            if not text:
                continue
            lines = [ln.strip() for ln in text.split("\n")]
            for ln in lines:
                if is_probable_heading(ln):
                    # flush previous section before starting new one
                    flush()
                    current_title = ln.strip().strip(":")
                    continue
                if not ln:
                    continue
                if buffer_page_start is None:
                    buffer_page_start = p.page
                buffer_page_end = p.page
                buffer_tokens.extend(simple_tokenize(ln + "\n"))

                # keep buffers bounded even without headings
                if len(buffer_tokens) >= self.max_tokens:
                    flush()

        flush()
        return chunks
