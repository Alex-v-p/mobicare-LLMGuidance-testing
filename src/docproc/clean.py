from __future__ import annotations

import re


_re_hyphen_linebreak = re.compile(r"(\w)-\n(\w)")
_re_multispace = re.compile(r"[ \t\x0b\r\f]+")
_re_multi_newlines = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Light, deterministic cleanup suitable for guideline PDFs."""

    # Fix word breaks: "anti-\ncoagulant" -> "anticoagulant"
    text = _re_hyphen_linebreak.sub(r"\1\2", text)
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse repeated spaces
    text = _re_multispace.sub(" ", text)
    # Collapse excessive blank lines
    text = _re_multi_newlines.sub("\n\n", text)
    # Trim trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def is_probable_heading(line: str) -> bool:
    """Heuristic heading detector used by structured chunking."""
    s = line.strip()
    if not s:
        return False
    # Numbered headings: "1.", "2.3" etc.
    if re.match(r"^\d+(\.\d+)*\s+\S+", s):
        return True
    # All caps (allow short punctuation)
    letters = re.sub(r"[^A-Za-z]+", "", s)
    if len(letters) >= 6 and letters.isupper():
        return True
    # Colon headings: "Dosage:" "Contraindications:"
    if s.endswith(":") and len(s) <= 80:
        return True
    return False


def simple_tokenize(text: str) -> list[str]:
    """Tokenize without external deps (approx tokens ≈ words/punct)."""
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def detokenize(tokens: list[str]) -> str:
    """Best-effort detokenizer for the simple_tokenize output."""
    out: list[str] = []
    for t in tokens:
        if not out:
            out.append(t)
            continue
        if re.match(r"\w", t) and re.match(r"\w", out[-1]):
            out.append(" " + t)
        elif t in [".", ",", ":", ";", "?", "!", ")", "]", "}"]:
            out.append(t)
        elif out[-1] in ["(", "[", "{"]:
            out.append(t)
        else:
            out.append(" " + t)
    return "".join(out)
