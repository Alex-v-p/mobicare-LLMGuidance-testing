from __future__ import annotations

"""Ollama LLM benchmark runner.

Purpose
-------
Compare local Ollama models on:
- Latency (total / load / prompt eval / generation)
- Throughput (generated tokens per second)
- Output length (prompt & completion token counts)
- Whether the model is running on CPU vs GPU (best-effort via `ollama ps`)

This is meant to be a lightweight, reproducible benchmark you can run alongside
your RAG experiments.

Notes
-----
- Metrics come from Ollama /api/generate response fields when available.
- GPU/CPU detection is best-effort and depends on `ollama ps` output.
"""

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..chunkers.llm_boundary import OllamaClient, build_ollama_client_from_env


@dataclass
class PromptResult:
    prompt_id: str
    ok: bool
    error: str | None
    total_s: float
    load_s: float
    prompt_eval_s: float
    eval_s: float
    prompt_tokens: int
    out_tokens: int
    out_tokens_per_s: float


@dataclass
class QualityResult:
    prompt_id: str
    repeat: int
    score: float
    passed: bool
    notes: dict[str, Any]


def _ensure_prompt_ids(prompts: list[dict[str, Any]]) -> None:
    """Ensure every prompt has a stable id field."""
    for i, p in enumerate(prompts):
        if not str(p.get("id") or "").strip():
            p["id"] = f"P{i+1}"

def _installed_ollama_models() -> set[str]:
    try:
        cp = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = cp.stdout.splitlines()
        models = set()
        for ln in lines[1:]:  # skip header
            parts = ln.strip().split()
            if parts:
                models.add(parts[0])
        return models
    except Exception:
        return set()
    
def _norm_id(x: Any) -> str:
    s = str(x or "")
    # Remove UTF-8 BOM if present
    s = s.lstrip("\ufeff")
    # Strip whitespace
    s = s.strip()
    # Optional: enforce case-insensitive ids (recommended)
    s = s.lower()
    return s


def _ensure_models_available(models: list[str]) -> None:
    installed = _installed_ollama_models()

    for m in models:
        if m not in installed:
            print(f"[INFO] Pulling missing model: {m}")
            subprocess.run(["ollama", "pull", m], check=True)
        else:
            print(f"[OK] Model already installed: {m}")


def _ns_to_s(x: Any) -> float:
    """Ollama durations are typically integers in nanoseconds."""
    try:
        v = float(x)
    except Exception:
        return -1.0
    if v < 0:
        return -1.0
    if v > 1e6:
        return v / 1e9
    return v


def _p50(values: list[float]) -> float:
    if not values:
        return -1.0
    return float(statistics.median(values))


def _mean(values: list[float]) -> float:
    if not values:
        return -1.0
    return float(sum(values) / len(values))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows



def _read_jsonl_with_errors(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read JSONL but keep going if a line is invalid.

    Returns (rows, errors). Each error contains line number, error string, and raw line.
    """
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.rstrip("\n")
            s = raw.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception as e:
                errors.append({"line": lineno, "error": str(e), "raw": raw})
    return rows, errors

def _index_gold(gold_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in gold_rows:
        pid = _norm_id(r.get("id"))
        if not pid:
            continue
        out[pid] = r
    return out


def _safe_lower(s: str) -> str:
    return (s or "").lower()


def _score_response(prompt_id: str, response: str, gold: dict[str, Any] | None) -> tuple[float, dict[str, Any]]:
    """Heuristic, RAG-independent quality scoring.

    The goal is *comparability* across models, not perfect correctness.
    Gold entries are optional; if missing, we return score=-1.

    Supported gold fields (all optional):
      - expected_contains: ["substring", ...]   (case-insensitive)
      - forbidden_contains: ["substring", ...]  (case-insensitive)
      - expected_regex: ["pattern", ...]        (re.IGNORECASE)
      - expect_json: bool                        (response must parse as JSON)
      - json_required_keys: ["k1", ...]
      - json_array_key_len: {"bullets": 3}      (key -> required len if it's a list)
      - expected_int_list: [1,2,3]               (extract ints from response; must match exactly)

    Returns (score, notes).
    """
    if not gold:
        return (-1.0, {"reason": "no_gold"})

    resp = response or ""
    resp_l = _safe_lower(resp)

    expected_contains = [str(x) for x in (gold.get("expected_contains") or [])]
    forbidden_contains = [str(x) for x in (gold.get("forbidden_contains") or [])]
    expected_regex = [str(x) for x in (gold.get("expected_regex") or [])]

    checks_total = 0
    checks_pass = 0
    notes: dict[str, Any] = {}

    if expected_contains:
        checks_total += len(expected_contains)
        hit = []
        for s in expected_contains:
            ok = _safe_lower(s) in resp_l
            hit.append({"text": s, "ok": ok})
            if ok:
                checks_pass += 1
        notes["expected_contains"] = hit

    if forbidden_contains:
        checks_total += len(forbidden_contains)
        hit = []
        for s in forbidden_contains:
            bad = _safe_lower(s) in resp_l
            hit.append({"text": s, "ok": (not bad)})
            if not bad:
                checks_pass += 1
        notes["forbidden_contains"] = hit

    if expected_regex:
        checks_total += len(expected_regex)
        hit = []
        for pat in expected_regex:
            try:
                ok = re.search(pat, resp, flags=re.IGNORECASE) is not None
            except re.error:
                ok = False
            hit.append({"pattern": pat, "ok": ok})
            if ok:
                checks_pass += 1
        notes["expected_regex"] = hit

    if bool(gold.get("expect_json")):
        checks_total += 1
        parsed = None
        try:
            parsed = json.loads(resp)
            checks_pass += 1
            notes["json_parse"] = {"ok": True}
        except Exception as e:
            notes["json_parse"] = {"ok": False, "error": repr(e)}

        if isinstance(parsed, dict):
            req_keys = [str(k) for k in (gold.get("json_required_keys") or [])]
            if req_keys:
                checks_total += len(req_keys)
                key_hits = []
                for k in req_keys:
                    ok = k in parsed
                    key_hits.append({"key": k, "ok": ok})
                    if ok:
                        checks_pass += 1
                notes["json_required_keys"] = key_hits

            arr_lens = gold.get("json_array_key_len") or {}
            if isinstance(arr_lens, dict) and arr_lens:
                for k, want_len in arr_lens.items():
                    checks_total += 1
                    ok = False
                    if k in parsed and isinstance(parsed[k], list):
                        try:
                            ok = len(parsed[k]) == int(want_len)
                        except Exception:
                            ok = False
                    notes.setdefault("json_array_key_len", []).append({"key": k, "want": want_len, "ok": ok})
                    if ok:
                        checks_pass += 1

        if gold.get("expected_int_list") is not None:
            checks_total += 1
            want = [int(x) for x in gold.get("expected_int_list")]

            # Prefer parsing a JSON list if the model returned one.
            try:
                parsed = json.loads(resp)
                if isinstance(parsed, list):
                    got = [int(x) for x in parsed]
                else:
                    got = [int(x) for x in re.findall(r"-?\d+", resp)]
            except Exception:
                got = [int(x) for x in re.findall(r"-?\d+", resp)]

            want_set = set(want)
            got_set = set(got)
            correct = len(want_set & got_set)

            precision = correct / len(got_set) if got_set else 0.0
            recall = correct / len(want_set) if want_set else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

            ok = f1 >= float(gold.get("pass_threshold", 0.8))

            notes["expected_int_list"] = {
                "ok": ok,
                "want": want,
                "got": got,
                "correct": sorted(list(want_set & got_set)),
                "missing": sorted(list(want_set - got_set)),
                "extra": sorted(list(got_set - want_set)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        if ok:
            checks_pass += 1

    if checks_total == 0:
        return (-1.0, {"reason": "gold_has_no_checks"})

    score = checks_pass / checks_total
    notes["checks_pass"] = checks_pass
    notes["checks_total"] = checks_total
    return (float(score), notes)


def _ollama_ps() -> str:
    """Return `ollama ps` output, or empty string if not available."""
    try:
        cp = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
        return out.strip()
    except Exception:
        return ""


def _detect_processor(model: str) -> tuple[str, str]:
    """Best-effort detection of CPU vs GPU from `ollama ps`."""
    ps = _ollama_ps()
    if not ps:
        return ("unknown", "")

    lines = [ln.strip() for ln in ps.splitlines() if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines):
        if re.search(r"\bPROCESSOR\b", ln, flags=re.I):
            header_idx = i
            break

    candidate_lines = lines[header_idx + 1 :] if header_idx is not None else lines
    for ln in candidate_lines:
        if model in ln:
            if re.search(r"\bGPU\b", ln, flags=re.I):
                return ("GPU", ln)
            if re.search(r"\bCPU\b", ln, flags=re.I):
                return ("CPU", ln)
            m = re.search(r"\b(GPU|CPU|METAL)\b", ln, flags=re.I)
            if m:
                return (m.group(1).upper(), ln)
            return ("unknown", ln)

    return ("unknown", "")


def _repo_root_from_here() -> Path:
    """Find repo root reliably, regardless of current working directory.

    We walk upwards until we find a folder that contains `src/`.
    """
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "src").is_dir():
            return parent
    # Fallback: current working directory
    return Path.cwd().resolve()


def _generate_raw(client: OllamaClient, prompt: str) -> dict[str, Any]:
    """Call Ollama and return the full JSON payload."""
    import urllib.request

    url = client.base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": client.model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=client.timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def bench_model(
    base_url: str,
    model: str,
    timeout_s: int,
    prompts: list[dict[str, Any]],
    gold_by_id: dict[str, dict[str, Any]] | None,
    repeats: int,
    warmup: int,
    run_dir: Path,
) -> dict[str, Any]:
    client = OllamaClient(base_url=base_url, model=model, timeout_s=timeout_s)

    # Warmup requests (helps load model; may also trigger GPU placement).
    warmup_results: list[dict[str, Any]] = []
    for i in range(max(0, warmup)):
        try:
            obj = _generate_raw(client, str(prompts[i % len(prompts)].get("prompt", "")))
            # also store the prompt + response for warmup
            warmup_results.append(
                {
                    "prompt_id": str(prompts[i % len(prompts)].get("id", f"warmup_{i}")),
                    "prompt": str(prompts[i % len(prompts)].get("prompt", "")),
                    "response": obj.get("response", ""),
                    "raw": obj,
                }
            )
        except Exception as e:
            warmup_results.append({"error": repr(e)})

    (run_dir / "warmup_raw.json").write_text(json.dumps(warmup_results, indent=2, ensure_ascii=False), encoding="utf-8")

    processor, ps_line = _detect_processor(model)

    per_prompt: list[PromptResult] = []
    raw_records: list[dict[str, Any]] = []
    qa_records: list[dict[str, Any]] = []  # explicit Q/A log (prompt + response)
    quality_records: list[QualityResult] = []

    for p in prompts:
        pid_raw = p.get("id", "prompt")
        pid = _norm_id(pid_raw)
        prompt_text = str(p.get("prompt", ""))

        for r in range(repeats):
            t0 = time.perf_counter()
            try:
                obj = _generate_raw(client, prompt_text)
                t1 = time.perf_counter()

                total_s = _ns_to_s(obj.get("total_duration"))
                if total_s < 0:
                    total_s = t1 - t0

                load_s = _ns_to_s(obj.get("load_duration"))
                prompt_eval_s = _ns_to_s(obj.get("prompt_eval_duration"))
                eval_s = _ns_to_s(obj.get("eval_duration"))
                prompt_tokens = int(obj.get("prompt_eval_count") or 0)
                out_tokens = int(obj.get("eval_count") or 0)

                out_tps = -1.0
                if eval_s and eval_s > 0 and out_tokens >= 0:
                    out_tps = float(out_tokens) / float(eval_s)

                per_prompt.append(
                    PromptResult(
                        prompt_id=pid,
                        ok=True,
                        error=None,
                        total_s=float(total_s),
                        load_s=float(load_s),
                        prompt_eval_s=float(prompt_eval_s),
                        eval_s=float(eval_s),
                        prompt_tokens=prompt_tokens,
                        out_tokens=out_tokens,
                        out_tokens_per_s=float(out_tps),
                    )
                )

                # Raw record (includes prompt + response so you can audit outputs)
                raw_records.append(
                    {
                        "prompt_id": pid,
                        "repeat": r,
                        "prompt": prompt_text,
                        "response": obj.get("response", ""),
                        "raw": obj,
                    }
                )

                # Quality scoring (heuristics; does NOT use retrieval)
                gold = (gold_by_id or {}).get(pid) if gold_by_id is not None else None
                q_score, q_notes = _score_response(pid, str(obj.get("response", "")), gold)
                passed = (q_score >= float(gold.get("pass_threshold", 0.8))) if (gold and q_score >= 0) else False
                quality_records.append(
                    QualityResult(prompt_id=pid, repeat=r, score=float(q_score), passed=bool(passed), notes=q_notes)
                )

                # Lightweight Q/A log (jsonl-friendly)
                qa_records.append(
                    {
                        "prompt_id": pid,
                        "repeat": r,
                        "model": model,
                        "prompt": prompt_text,
                        "response": obj.get("response", ""),
                    }
                )

            except Exception as e:
                t1 = time.perf_counter()
                per_prompt.append(
                    PromptResult(
                        prompt_id=pid,
                        ok=False,
                        error=repr(e),
                        total_s=float(t1 - t0),
                        load_s=-1.0,
                        prompt_eval_s=-1.0,
                        eval_s=-1.0,
                        prompt_tokens=0,
                        out_tokens=0,
                        out_tokens_per_s=-1.0,
                    )
                )
                raw_records.append({"prompt_id": pid, "repeat": r, "prompt": prompt_text, "error": repr(e)})
                qa_records.append(
                    {
                        "prompt_id": pid,
                        "repeat": r,
                        "model": model,
                        "prompt": prompt_text,
                        "error": repr(e),
                    }
                )

                # Still record quality as failed/unscored
                quality_records.append(
                    QualityResult(prompt_id=pid, repeat=r, score=-1.0, passed=False, notes={"error": repr(e)})
                )

    # Save detailed artifacts
    (run_dir / "per_prompt.json").write_text(
        json.dumps([asdict(x) for x in per_prompt], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "raw_records.json").write_text(json.dumps(raw_records, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save Q/A as JSONL (easy diff/grep)
    with (run_dir / "qa.jsonl").open("w", encoding="utf-8") as f:
        for rec in qa_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Save quality results (one row per prompt x repeat)
    (run_dir / "quality.json").write_text(
        json.dumps([asdict(x) for x in quality_records], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ok_rows = [x for x in per_prompt if x.ok]
    totals = [x.total_s for x in ok_rows]
    tps = [x.out_tokens_per_s for x in ok_rows if x.out_tokens_per_s >= 0]
    out_toks = [x.out_tokens for x in ok_rows]
    prompt_toks = [x.prompt_tokens for x in ok_rows]

    summary = {
        "model": model,
        "ollama_base_url": base_url,
        "processor": processor,
        "processor_ps_line": ps_line,
        "num_prompts": len(prompts),
        "repeats": repeats,
        "warmup": warmup,
        "ok": len(ok_rows),
        "errors": len(per_prompt) - len(ok_rows),
        "avg_total_s": _mean(totals),
        "p50_total_s": _p50(totals),
        "avg_out_tokens": _mean(out_toks),
        "avg_prompt_tokens": _mean(prompt_toks),
        "avg_out_tokens_per_s": _mean(tps),
    }

    # Aggregate quality metrics
    scored = [q for q in quality_records if q.score >= 0]
    q_scores = [q.score for q in scored]
    q_pass = [q.passed for q in scored]
    summary.update(
        {
            "quality_scored": len(scored),
            "quality_avg": _mean(q_scores),
            "quality_p50": _p50(q_scores),
            "quality_pass_rate": (sum(1 for x in q_pass if x) / len(q_pass)) if q_pass else -1.0,
        }
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="results", help="Results root (relative to repo root unless absolute)")
    ap.add_argument("--prompts", type=str, default="data/eval/llm_bench_prompts.jsonl")
    ap.add_argument(
        "--gold",
        type=str,
        default="data/eval/llm_bench_gold.jsonl",
        help="Optional gold file for heuristic quality scoring (JSONL). If missing, quality metrics are skipped.",
    )
    ap.add_argument("--models", type=str, default="", help="Comma-separated list of Ollama model names")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=int(os.getenv("OLLAMA_TIMEOUT_S", "120")))
    ap.add_argument("--base-url", type=str, default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    args = ap.parse_args()

    repo_root = _repo_root_from_here()

    prompts_path = Path(args.prompts)
    if not prompts_path.is_absolute():
        prompts_path = repo_root / prompts_path
    prompts = _read_jsonl(prompts_path)
    if not prompts:
        raise SystemExit(f"No prompts found in {prompts_path}")

    # Load optional gold for quality scoring (no retrieval involved)
    gold_by_id: dict[str, dict[str, Any]] | None = None
    gold_text = ""
    gold_parse_errors: list[dict[str, Any]] = []

    gold_path = Path(args.gold)
    if not gold_path.is_absolute():
        gold_path = repo_root / gold_path

    if gold_path.exists():
        gold_text = gold_path.read_text(encoding="utf-8")
        gold_rows, gold_parse_errors = _read_jsonl_with_errors(gold_path)
        # Index only the valid rows; keep parse errors for debugging.
        gold_by_id = _index_gold(gold_rows) if gold_rows else None
    else:
        gold_by_id = None  # no gold file -> skip quality scoring

    models: list[str]
    if args.models.strip():
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        client = build_ollama_client_from_env()
        models = [client.model]

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = repo_root / outdir

    runs_root = outdir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"llmbench_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy the prompts file used for the run (auditability)
    (run_dir / "prompts.jsonl").write_text(prompts_path.read_text(encoding="utf-8"), encoding="utf-8")
    if gold_text:
        (run_dir / "gold.jsonl").write_text(gold_text, encoding="utf-8")

    # If the gold file has invalid JSON lines, keep the valid ones but log parse errors.
    if gold_parse_errors:
        (run_dir / "gold_parse_errors.json").write_text(
            json.dumps(gold_parse_errors, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[WARN] Gold parse errors: {len(gold_parse_errors)} (see {run_dir / 'gold_parse_errors.json'})")

    if gold_by_id is not None and isinstance(gold_by_id, dict) and len(gold_by_id) == 0:
        print("[WARN] Gold file loaded but contained 0 valid/usable entries. Quality metrics will be skipped.")


    # Ensure stable prompt ids (prevents quality scoring from silently doing nothing)
    for i, p in enumerate(prompts):
        if not str(p.get("id") or "").strip():
            p["id"] = f"P{i+1}"

    # Save the ids actually used in this run (debugging)
    (run_dir / "prompt_ids.json").write_text(
        json.dumps([p["id"] for p in prompts], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if gold_by_id is not None:
        (run_dir / "gold_ids.json").write_text(
            json.dumps(sorted(list(gold_by_id.keys())), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    _ensure_models_available(models)

    if gold_by_id is not None:
        prompt_ids = {_norm_id(p.get("id")) for p in prompts}
        gold_ids = set(gold_by_id.keys())
        overlap = prompt_ids & gold_ids
        if not overlap:
            raise SystemExit(
                "Gold file loaded but no prompt ids matched.\n"
                f"Sample prompt ids: {sorted(list(prompt_ids))[:10]}\n"
                f"Sample gold ids:   {sorted(list(gold_ids))[:10]}\n"
            )

    rows: list[dict[str, Any]] = []
    for model in models:
        model_safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)
        mdir = run_dir / model_safe
        mdir.mkdir(parents=True, exist_ok=True)

        summary = bench_model(
            base_url=args.base_url,
            model=model,
            timeout_s=args.timeout,
            prompts=prompts,
            gold_by_id=gold_by_id,
            repeats=args.repeats,
            warmup=args.warmup,
            run_dir=mdir,
        )
        rows.append({"run_id": run_id, **summary, "run_dir": str(mdir)})

    leaderboard_dir = outdir / "summary"
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = leaderboard_dir / "leaderboard_llm.csv"
    fieldnames = list(rows[0].keys()) if rows else ["run_id", "model"]
    write_header = not leaderboard.exists()
    with leaderboard.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Repo root: {repo_root}")
    print(f"Run directory: {run_dir}")
    print(f"Leaderboard: {leaderboard}")


if __name__ == "__main__":
    main()