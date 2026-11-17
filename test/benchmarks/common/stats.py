"""Statistical methodology for SCRUB-VOICE benchmarking.

Provides:
- Non-parametric percentile bootstrapping for latency distributions.
- Wilson score intervals for binomial metrics (accuracy, OOD-FAR).
- Paired Wilcoxon signed-rank testing with Holm-Bonferroni correction.
- Word Error Rate (WER) with Levenshtein dynamic programming.
- Semantic slot matching with floating-point tolerance.
"""
from __future__ import annotations

import importlib
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

spstats = _try_import("scipy.stats")


def summarize(xs: Sequence[Any], boot: int = 1000, seed: int = 0) -> Dict[str, Any]:
    """Computes summary statistics with a 95% bootstrap CI of the mean."""
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return {"n": 0}
    lo = hi = float(a[0])
    if a.size > 1:
        rng = np.random.default_rng(seed)
        b = int(min(boot, max(200, 2_000_000 // a.size)))
        means = rng.choice(a, (b, a.size)).mean(axis=1)
        lo, hi = (float(v) for v in np.percentile(means, [2.5, 97.5]))
    p50, p90, p95, p99 = (float(v) for v in np.percentile(a, [50, 90, 95, 99]))
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "max": float(a.max()),
        "ci95_lo": lo,
        "ci95_hi": hi,
    }


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Computes Wilson score interval for binomial proportions (p, ci_lo, ci_hi)."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def holm(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction for multiple hypothesis comparisons."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def paired_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """Paired Wilcoxon signed-rank test on matched evaluation pairs."""
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a_arr) & np.isfinite(b_arr)
    a_arr, b_arr = a_arr[ok], b_arr[ok]
    if a_arr.size < 5 or spstats is None or np.allclose(a_arr, b_arr):
        return {"n": int(a_arr.size), "median_diff": float(np.median(a_arr - b_arr)) if a_arr.size else None, "p": None}
    try:
        p = float(spstats.wilcoxon(a_arr, b_arr).pvalue)
    except Exception:
        p = None
    return {"n": int(a_arr.size), "median_diff": float(np.median(a_arr - b_arr)), "p": p}


def norm_text(s: str) -> str:
    """Normalizes text for word error rate evaluation."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def word_errors(ref: str, hyp: str) -> Tuple[int, int]:
    """Calculates Levenshtein word distance (substitutions + deletions + insertions, ref_words)."""
    r, h = norm_text(ref).split(), norm_text(hyp).split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)], len(r)


def corpus_wer(errs: Sequence[int], refs: Sequence[int], boot: int = 1000) -> Dict[str, Any]:
    """Calculates corpus-level WER with a 1000-sample bootstrap confidence interval."""
    e, r = np.asarray(errs, float), np.asarray(refs, float)
    if r.sum() == 0:
        return {"n_utts": 0}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(e), (boot, len(e)))
    w = e[idx].sum(1) / np.maximum(r[idx].sum(1), 1)
    return {
        "n_utts": int(len(e)),
        "wer": float(e.sum() / r.sum()),
        "ci95_lo": float(np.percentile(w, 2.5)),
        "ci95_hi": float(np.percentile(w, 97.5)),
    }


def slots_match(gt: Dict[str, Any], pred: Dict[str, Any]) -> bool:
    """Checks if ground truth and predicted parameter slots match within tolerance."""
    for k, v in (gt or {}).items():
        pv = (pred or {}).get(k)
        if isinstance(v, (int, float)) and isinstance(pv, (int, float)):
            if not math.isclose(float(v), float(pv), rel_tol=1e-3, abs_tol=1e-3):
                return False
        elif str(v).lower() != str(pv).lower():
            return False
    return True


def intent_metrics(rows: List[dict]) -> Dict[str, Any]:
    """Calculates intent accuracy, OOD-FAR, and slot exact match with Wilson CIs."""
    n = len(rows)
    ood = [r for r in rows if r["gt_status"] == "OUT-OF-DOMAIN"]
    ind = [r for r in rows if r["gt_status"] != "OUT-OF-DOMAIN"]
    acc = wilson(sum(r["intent_ok"] for r in rows), n)
    far = wilson(sum(r["pred_status"] == "EXECUTE" for r in ood), len(ood))
    frr = wilson(sum(r["pred_status"] == "OUT-OF-DOMAIN" for r in ind), len(ind))
    slot_rows = [r for r in ind if r["slot_ok"] is not None]
    slot = wilson(sum(r["slot_ok"] for r in slot_rows), len(slot_rows))
    return {
        "n": n,
        "intent_acc": {"p": acc[0], "lo": acc[1], "hi": acc[2]},
        "ood_false_accept": {"p": far[0], "lo": far[1], "hi": far[2], "n": len(ood)},
        "in_domain_false_reject": {"p": frr[0], "lo": frr[1], "hi": frr[2], "n": len(ind)},
        "slot_exact": {"p": slot[0], "lo": slot[1], "hi": slot[2], "n": len(slot_rows)},
    }


