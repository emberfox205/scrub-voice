"""Scoring helpers: per-clip error counts, corpus WER and a cluster
bootstrap confidence interval.

Corpus WER = total errors / total reference words over all clips. Unlike the
mean of per-clip WERs, a one-word clip doesn't weigh as much as a seven-word
one.

The bootstrap resamples whole clusters (command groups from
vocab.command_group), not single clips: phrasings of the same command aren't
independent, so resampling clips would make the interval too narrow. The
same resample index matrix is used for every config, so two configs can be
compared pairwise on identical resamples.
"""

from __future__ import annotations

import jiwer
import numpy as np


def clip_errors(ref_normalized: str, hyp_normalized: str) -> tuple[int, int]:
    """(substitutions + deletions + insertions, reference word count)."""
    ref_words = len(ref_normalized.split())
    if ref_words == 0:
        return len(hyp_normalized.split()), 0
    out = jiwer.process_words(ref_normalized, hyp_normalized)
    return out.substitutions + out.deletions + out.insertions, ref_words


def corpus_wer(errors: np.ndarray, ref_words: np.ndarray) -> float:
    return float(errors.sum() / ref_words.sum())


def cluster_resamples(clusters: list[str], n: int = 1000, seed: int = 0) -> list[np.ndarray]:
    """n arrays of clip indices, each built by drawing clusters with
    replacement and taking every clip of each drawn cluster."""
    names = sorted(set(clusters))
    members = {c: np.array([i for i, k in enumerate(clusters) if k == c]) for c in names}
    rng = np.random.default_rng(seed)
    resamples = []
    for _ in range(n):
        drawn = rng.choice(len(names), size=len(names), replace=True)
        resamples.append(np.concatenate([members[names[d]] for d in drawn]))
    return resamples


def bootstrap_ci(errors: np.ndarray, ref_words: np.ndarray, resamples: list[np.ndarray]) -> tuple[float, float]:
    stats = [errors[idx].sum() / ref_words[idx].sum() for idx in resamples]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_diff_ci(
    errors_a: np.ndarray, errors_b: np.ndarray, ref_words: np.ndarray, resamples: list[np.ndarray]
) -> tuple[float, float, float]:
    """Corpus WER of A minus B: (point estimate, 2.5%, 97.5%). If the
    interval contains 0, the difference isn't distinguishable from noise."""
    diffs = [(errors_a[idx].sum() - errors_b[idx].sum()) / ref_words[idx].sum() for idx in resamples]
    point = (errors_a.sum() - errors_b.sum()) / ref_words.sum()
    return float(point), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
