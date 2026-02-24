from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..clean import is_probable_heading, normalize_text, simple_tokenize, detokenize
from .base import Chunk, Page


@dataclass
class TreeParentChildChunker:
    """
    Builds:
      - parent chunks: one per heading-defined section (truncated to parent_max_tokens)
      - child chunks: token chunks inside each parent section (child_chunk_size/overlap)

    Output:
      (parents, children)
    """

    parent_max_tokens: int = 900
    child_chunk_size: int = 450
    child_overlap: int = 100

    def chunk(self, pages: Iterable[Page]) -> tuple[list[Chunk], list[Chunk]]:
        parents: list[Chunk] = []
        children: list[Chunk] = []

        current_title: str | None = None
        section_tokens: list[str] = []
        page_start: int | None = None
        page_end: int | None = None
        section_i = 0

        def flush_section() -> None:
            nonlocal section_tokens, page_start, page_end, section_i, current_title

            if not section_tokens or page_start is None or page_end is None:
                section_tokens = []
                page_start = None
                page_end = None
                return

            parent_id = f"p{section_i:04d}"
            # Parent text: truncate to keep parent vectors “focused”
            parent_text = detokenize(section_tokens[: self.parent_max_tokens])

            parents.append(
                Chunk(
                    chunk_id=parent_id,
                    text=parent_text,
                    page_start=page_start,
                    page_end=page_end,
                    section_title=current_title,
                    strategy="tree_parent",
                    chunk_size=None,
                    overlap=None,
                )
            )

            # Child chunks inside this section
            step = max(1, self.child_chunk_size - self.child_overlap)
            start = 0
            child_i = 0
            while start < len(section_tokens):
                end = min(len(section_tokens), start + self.child_chunk_size)
                part = section_tokens[start:end]
                if not part:
                    break

                child_id = f"{parent_id}/c{child_i:04d}"
                children.append(
                    Chunk(
                        chunk_id=child_id,
                        text=detokenize(part),
                        page_start=page_start,
                        page_end=page_end,
                        section_title=current_title,
                        strategy="tree_child",
                        chunk_size=self.child_chunk_size,
                        overlap=self.child_overlap,
                    )
                )

                child_i += 1
                start += step

            section_i += 1
            section_tokens = []
            page_start = None
            page_end = None

        for p in pages:
            text = normalize_text(p.text)
            if not text:
                continue
            lines = [ln.strip() for ln in text.split("\n")]
            for ln in lines:
                if is_probable_heading(ln):
                    # new heading => flush previous section
                    flush_section()
                    current_title = ln.strip().strip(":")
                    continue

                if not ln:
                    continue

                if page_start is None:
                    page_start = p.page
                page_end = p.page

                section_tokens.extend(simple_tokenize(ln + "\n"))

        flush_section()
        return parents, children