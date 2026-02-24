from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..clean import detokenize, normalize_text, simple_tokenize
from .base import Chunk, Page


@dataclass
class NaiveTokenChunker:
    chunk_size: int = 500
    overlap: int = 100

    def chunk(self, pages: Iterable[Page]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for p in pages:
            text = normalize_text(p.text)
            if not text:
                continue
            tokens = simple_tokenize(text)
            if not tokens:
                continue

            step = max(1, self.chunk_size - self.overlap)
            idx = 0
            local_i = 0
            while idx < len(tokens):
                window = tokens[idx : idx + self.chunk_size]
                if not window:
                    break
                chunk_text = detokenize(window)
                chunk_id = f"p{p.page:04d}_n{local_i:04d}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        page_start=p.page,
                        page_end=p.page,
                        section_title=None,
                        strategy="naive_tokens",
                        chunk_size=self.chunk_size,
                        overlap=self.overlap,
                    )
                )
                local_i += 1
                idx += step
        return chunks
