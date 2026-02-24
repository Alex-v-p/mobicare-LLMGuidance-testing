from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrievalQuestion:
    id: str
    question: str
    expected_pages: list[int]


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_retrieval_questions(path: Path) -> list[RetrievalQuestion]:
    raw = load_jsonl(path)
    out: list[RetrievalQuestion] = []
    for item in raw:
        out.append(
            RetrievalQuestion(
                id=str(item["id"]),
                question=str(item["question"]),
                expected_pages=list(item.get("expected_pages") or []),
            )
        )
    return out
