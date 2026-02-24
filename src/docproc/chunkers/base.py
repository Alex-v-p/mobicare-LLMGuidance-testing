from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


@dataclass
class Page:
    page: int
    text: str


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    section_title: str | None
    strategy: str
    chunk_size: int | None = None
    overlap: int | None = None


class Chunker(Protocol):
    def chunk(self, pages: Iterable[Page]) -> list[Chunk]: ...
