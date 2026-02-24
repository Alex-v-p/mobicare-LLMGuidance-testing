from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz  # PyMuPDF


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract text per page using PyMuPDF.

    Returns a list of dicts: {"page": int, "text": str}
    """
    doc = fitz.open(pdf_path)
    pages: list[dict] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text")
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def write_jsonl(items: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PDF pages -> JSONL")
    parser.add_argument("--pdf", required=True, help="Path to guideline PDF")
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path (e.g., data/processed/guidelines_pages.jsonl)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_path = Path(args.out)
    pages = extract_pdf_pages(pdf_path)
    write_jsonl(pages, out_path)
    print(f"Wrote {len(pages)} pages to {out_path}")


if __name__ == "__main__":
    main()
