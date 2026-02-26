from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .base import Chunk, Page
from .page_index import PageIndexChunker


@dataclass
class LateChunkingChunker:
    """Index-time chunker for late chunking.

    Late chunking indexes larger units (here: full pages) and only produces
    sub-chunks at query time for the top retrieved pages.

    This chunker therefore returns 1 chunk per page, identical to PageIndexChunker,
    but marks the strategy as 'late_chunking' so experiments can track it.
    """

    def chunk(self, pages: Iterable[Page]) -> list[Chunk]:
        base = PageIndexChunker().chunk(pages)
        return [
            Chunk(
                chunk_id=c.chunk_id,
                text=c.text,
                page_start=c.page_start,
                page_end=c.page_end,
                section_title=c.section_title,
                strategy="late_chunking",
            )
            for c in base
        ]
