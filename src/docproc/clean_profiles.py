from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_WS_RE = re.compile(r"[ \t]+")
_NONPRINT_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")  # dehyphenate across line breaks


@dataclass(frozen=True)
class CleaningProfile:
    name: str = "basic"

    # page-level cleaning
    strip_nonprinting: bool = True
    collapse_whitespace: bool = True
    dehyphenate: bool = False
    remove_headers_footers: bool = False  # optional (needs header/footer map)

    # chunk-level filtering (deep)
    drop_reference_like_chunks: bool = False
    drop_abbreviation_like_chunks: bool = False
    drop_low_signal_chunks: bool = False

    # thresholds for deep filtering
    min_alpha_ratio: float = 0.55
    min_token_count: int = 40
    max_digit_ratio: float = 0.25
    max_allcaps_ratio: float = 0.35
    max_equals_per_100_chars: float = 1.5


def profile_from_cfg(cfg: dict) -> CleaningProfile:
    # allow empty cfg
    cfg = cfg or {}
    return CleaningProfile(
        name=str(cfg.get("name", "basic")),
        strip_nonprinting=bool(cfg.get("strip_nonprinting", True)),
        collapse_whitespace=bool(cfg.get("collapse_whitespace", True)),
        dehyphenate=bool(cfg.get("dehyphenate", False)),
        remove_headers_footers=bool(cfg.get("remove_headers_footers", False)),
        drop_reference_like_chunks=bool(cfg.get("drop_reference_like_chunks", False)),
        drop_abbreviation_like_chunks=bool(cfg.get("drop_abbreviation_like_chunks", False)),
        drop_low_signal_chunks=bool(cfg.get("drop_low_signal_chunks", False)),
        min_alpha_ratio=float(cfg.get("min_alpha_ratio", 0.55)),
        min_token_count=int(cfg.get("min_token_count", 40)),
        max_digit_ratio=float(cfg.get("max_digit_ratio", 0.25)),
        max_allcaps_ratio=float(cfg.get("max_allcaps_ratio", 0.35)),
        max_equals_per_100_chars=float(cfg.get("max_equals_per_100_chars", 1.5)),
    )


def apply_cleaning(text: str, prof: CleaningProfile) -> str:
    """Page-level cleaning. Run before chunking."""
    if not text:
        return ""

    t = text

    if prof.strip_nonprinting:
        t = _NONPRINT_RE.sub("", t)

    # dehyphenate line-break hyphens (reno-\nvascular -> renovascular)
    if prof.dehyphenate:
        t = _HYPHEN_BREAK_RE.sub(r"\1\2", t)

    # normalize newlines first
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    if prof.collapse_whitespace:
        # collapse runs of spaces/tabs
        t = _WS_RE.sub(" ", t)
        # collapse excessive blank lines
        t = re.sub(r"\n{3,}", "\n\n", t)
        t = t.strip()

    return t


# ---- Chunk filtering heuristics ----

def _tokenize_quick(text: str) -> list[str]:
    return [tok for tok in re.split(r"\s+", text.strip()) if tok]


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(ch.isalpha() for ch in text)
    return alpha / max(1, len(text))


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    dig = sum(ch.isdigit() for ch in text)
    return dig / max(1, len(text))


def _allcaps_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    caps = 0
    for t in tokens:
        letters = [c for c in t if c.isalpha()]
        if letters and all(c.isupper() for c in letters):
            caps += 1
    return caps / max(1, len(tokens))


def _equals_per_100_chars(text: str) -> float:
    if not text:
        return 0.0
    return (text.count("=") / max(1, len(text))) * 100.0


def looks_like_reference_list(text: str) -> bool:
    """
    Heuristic: lots of years/citation patterns and separators.
    """
    t = text.lower()
    year_hits = len(re.findall(r"\b(19|20)\d{2}\b", t))
    # patterns common in references
    markers = sum(t.count(x) for x in ["et al", "doi", "journal", "vol.", "pp.", "http", "www."])
    semicolons = t.count(";")
    # if it looks like a bibliography chunk
    return (year_hits >= 3 and semicolons >= 3) or (markers >= 2 and year_hits >= 2)


def looks_like_abbreviation_list(text: str, tokens: list[str]) -> bool:
    """
    Heuristic: many ALLCAPS tokens + many '=' or ':' style definitions.
    """
    caps_r = _allcaps_ratio(tokens)
    eq100 = _equals_per_100_chars(text)
    # abbreviation lists often have "ABC = ..." / "ABC: ..."
    defs = len(re.findall(r"\b[A-Z]{2,}\b\s*(=|:)\s*", text))
    return (caps_r >= 0.40 and (eq100 >= 1.0 or defs >= 3)) or defs >= 6


def is_noise_chunk(text: str, prof: CleaningProfile) -> bool:
    """
    Chunk-level filter used only when deep profile toggles are on.
    """
    if not text:
        return True

    tokens = _tokenize_quick(text)

    # low signal checks
    if prof.drop_low_signal_chunks:
        if len(tokens) < prof.min_token_count:
            return True
        if _alpha_ratio(text) < prof.min_alpha_ratio:
            return True
        if _digit_ratio(text) > prof.max_digit_ratio:
            return True
        if _allcaps_ratio(tokens) > prof.max_allcaps_ratio:
            return True
        if _equals_per_100_chars(text) > prof.max_equals_per_100_chars:
            return True

    if prof.drop_reference_like_chunks and looks_like_reference_list(text):
        return True

    if prof.drop_abbreviation_like_chunks and looks_like_abbreviation_list(text, tokens):
        return True

    return False