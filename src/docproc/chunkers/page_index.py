from __future__ import annotations

"""Page-index chunking.

Produces exactly **one chunk per page** (page_start == page_end).

This enables a simple "page-index" retrieval baseline: instead of RAG-style
sub-page chunks, you embed / index each full page and retrieve pages directly.
The existing evaluation harness already uses expected page numbers, so page-
level retrieval integrates cleanly.
"""

from typing import Iterable

from .base import Chunk, Page


class PageIndexChunker:
    """Return one chunk per page."""

    def chunk(self, pages: Iterable[Page]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for p in pages:
            chunks.append(
                Chunk(
                    chunk_id=f"page/{p.page}",
                    text=p.text,
                    page_start=p.page,
                    page_end=p.page,
                    section_title=None,
                    strategy="page_index",
                    chunk_size=None,
                    overlap=None,
                )
            )
        return chunks
