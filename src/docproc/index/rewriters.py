from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@dataclass
class RewriteResult:
    original: str
    rewritten: str
    method: str


class QueryRewriter:
    def rewrite(self, q: str) -> RewriteResult:
        raise NotImplementedError


class NoopRewriter(QueryRewriter):
    def rewrite(self, q: str) -> RewriteResult:
        return RewriteResult(original=q, rewritten=q, method="none")


@dataclass
class HeuristicRewriter(QueryRewriter):
    """
    Cheap, offline baseline rewriter.
    It does NOT "invent" content; it just makes queries more retriever-friendly.
    """
    context_hint: str = "clinical guideline"

    def rewrite(self, q: str) -> RewriteResult:
        original = q

        # Normalize whitespace
        q2 = " ".join(q.split()).strip()

        # Light cleanup: remove trailing punctuation spam
        q2 = re.sub(r"[?!.]+$", "?", q2)

        # Add a context hint to help retrieval (BM25/dense) without changing semantics too much.
        # Only add if it's not already present.
        if self.context_hint and self.context_hint.lower() not in q2.lower():
            q2 = f"{q2} ({self.context_hint})"

        return RewriteResult(original=original, rewritten=q2, method="heuristic")


@dataclass
class HFSeq2SeqRewriter(QueryRewriter):
    model_name: str = "google/flan-t5-small"
    # device: None => auto-select cuda if available else cpu
    # allowed: None, "cpu", "cuda", "cuda:0"
    device: Optional[str] = None
    max_new_tokens: int = 64

    _tok: Optional[AutoTokenizer] = None
    _model: Optional[AutoModelForSeq2SeqLM] = None
    _device: Optional[str] = None

    def _ensure(self) -> None:
        if self._tok is not None and self._model is not None and self._device is not None:
            return

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        dev = self.device
        if dev is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"

        dev = dev.lower()
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"

        self._model.to(dev)
        self._model.eval()
        self._device = dev

    def rewrite(self, q: str) -> RewriteResult:
        self._ensure()

        prompt = (
            "Rewrite the question to be more specific and searchable for retrieving passages "
            "from a guideline document. Keep meaning the same. Do not add new facts.\n"
            f"Question: {q}\n"
            "Rewrite:"
        )

        inputs = self._tok(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

        rewritten = self._tok.decode(out_ids[0], skip_special_tokens=True).strip()
        if not rewritten:
            rewritten = q

        return RewriteResult(original=q, rewritten=rewritten, method=f"hf:{self.model_name}")


def _parse_device(value: object) -> Optional[str]:
    """
    Accepts: None, "auto", "cpu", "cuda", "cuda:0"
    Returns: None for auto-selection, or a valid torch device string.
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "none", "null", "auto"):
        return None
    if s == "gpu":
        return "cuda"
    if s.startswith("cuda") or s == "cpu":
        return s
    # Unknown => treat as auto to avoid crashing
    return None


def build_rewriter(cfg: dict | None) -> QueryRewriter:
    cfg = cfg or {}
    typ = str(cfg.get("type", "none")).lower()

    if typ in ("none", "noop"):
        return NoopRewriter()

    if typ in ("heuristic", "baseline"):
        return HeuristicRewriter(context_hint=str(cfg.get("context_hint", "clinical guideline")))

    if typ in ("hf", "huggingface", "seq2seq"):
        return HFSeq2SeqRewriter(
            model_name=str(cfg.get("model_name", "google/flan-t5-small")),
            device=_parse_device(cfg.get("device", None)),
            max_new_tokens=int(cfg.get("max_new_tokens", 32)),
        )

    raise ValueError(f"Unknown rewriter type: {typ}")