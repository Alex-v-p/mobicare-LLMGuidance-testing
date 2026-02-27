from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from ..clean import is_probable_heading, normalize_text, simple_tokenize
from ..index.embed_factory import STEmbedder
from .base import Chunk, Page


def _split_into_units(text: str) -> list[str]:
    """Split text into sentence-like units without external deps.

    We bias towards *not* merging headings into surrounding text.
    """
    text = normalize_text(text)
    if not text:
        return []

    units: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Keep headings as standalone units.
        if is_probable_heading(line):
            units.append(line)
            continue

        # Split non-heading lines into sentences.
        parts = re.split(r"(?<=[.!?])\s+", line)
        for p in parts:
            p = p.strip()
            if p:
                units.append(p)

    return units


def _count_tokens(text: str) -> int:
    return len(simple_tokenize(text))


@dataclass
class SemanticSimilarityChunker:
    """Embedding-based semantic chunker.

    Algorithm:
      1) Split pages into sentence-like units.
      2) Embed each unit.
      3) Start a new chunk when the similarity between consecutive units drops
         below `similarity_threshold` (and current chunk has at least
         `min_tokens`).
      4) Also enforce `max_tokens` as a hard upper bound.

    Note: assumes the embedder returns L2-normalized vectors.
    """

    embedder: STEmbedder
    max_tokens: int = 450
    min_tokens: int = 80
    similarity_threshold: float = 0.55
    batch_size: int = 128

    def chunk(self, pages: Iterable[Page]) -> list[Chunk]:
        # Flatten to units while keeping page attribution.
        unit_texts: list[str] = []
        unit_pages: list[int] = []

        for p in pages:
            units = _split_into_units(p.text)
            for u in units:
                unit_texts.append(u)
                unit_pages.append(p.page)

        if not unit_texts:
            return []

        # Embed units.
        vecs = self.embedder.transform(unit_texts)

        # Similarity between consecutive units (dot product = cosine when normalized).
        import numpy as np

        sims = np.sum(vecs[1:] * vecs[:-1], axis=1)

        chunks: list[Chunk] = []
        cur_units: list[str] = []
        cur_page_start = unit_pages[0]
        cur_page_end = unit_pages[0]
        cur_tokens = 0
        local_i = 0

        def flush():
            nonlocal cur_units, cur_page_start, cur_page_end, cur_tokens, local_i
            if not cur_units:
                return
            text = " ".join(cur_units).strip()
            if not text:
                cur_units = []
                cur_tokens = 0
                return
            chunk_id = f"s{local_i:04d}_p{cur_page_start:04d}-{cur_page_end:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    page_start=cur_page_start,
                    page_end=cur_page_end,
                    section_title=None,
                    strategy="semantic_similarity",
                    chunk_size=self.max_tokens,
                    overlap=None,
                )
            )
            local_i += 1
            cur_units = []
            cur_tokens = 0

        for i, u in enumerate(unit_texts):
            u_tokens = _count_tokens(u)

            # Start chunk if empty.
            if not cur_units:
                cur_units = [u]
                cur_page_start = unit_pages[i]
                cur_page_end = unit_pages[i]
                cur_tokens = u_tokens
                continue

            # Hard max size.
            if cur_tokens + u_tokens > self.max_tokens and cur_tokens >= self.min_tokens:
                flush()
                cur_units = [u]
                cur_page_start = unit_pages[i]
                cur_page_end = unit_pages[i]
                cur_tokens = u_tokens
                continue

            # Semantic boundary based on similarity to previous unit.
            sim_prev = float(sims[i - 1])  # sims is 1 shorter
            if sim_prev < self.similarity_threshold and cur_tokens >= self.min_tokens:
                flush()
                cur_units = [u]
                cur_page_start = unit_pages[i]
                cur_page_end = unit_pages[i]
                cur_tokens = u_tokens
                continue

            # Otherwise extend.
            cur_units.append(u)
            cur_page_end = unit_pages[i]
            cur_tokens += u_tokens

        flush()

        # If the last chunk is tiny, merge it into the previous one.
        if len(chunks) >= 2:
            last = chunks[-1]
            if _count_tokens(last.text) < max(20, self.min_tokens // 2):
                prev = chunks[-2]
                merged_text = (prev.text + " " + last.text).strip()
                chunks[-2] = Chunk(
                    chunk_id=prev.chunk_id,
                    text=merged_text,
                    page_start=prev.page_start,
                    page_end=last.page_end,
                    section_title=None,
                    strategy=prev.strategy,
                    chunk_size=prev.chunk_size,
                    overlap=prev.overlap,
                )
                chunks.pop()

        return chunks
