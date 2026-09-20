"""Surprise-biased cell sampling — the argmax(S) <-> argmin(S) slider.

Rationale (measured, search_problem E31/E33/E62): novel subject matter concentrates on
high-sensitivity cells (boundaries/confluences) at a coherence price; interiors are the
reverse. A single scalar in [-1, +1] therefore spans a real aesthetic axis:
    -1  calm     -> sample argmin(S): coherent, on-prompt, repetitive
     0  uniform  -> ignore S entirely
    +1  surprise -> sample argmax(S): novel, hybrid, less well-formed

Sampling is softmax(surprise * z / TEMP) over z-scored sensitivities, without replacement.
Softmax rather than hard argsort so intermediate slider positions interpolate smoothly and
repeated draws vary; TEMP = 0.5 makes the extremes concentrate on the top/bottom decile,
matching the E33 "aimed" condition.
"""
from __future__ import annotations

import numpy as np

TEMP = 0.5


def surprise_sample(sensitivities, surprise: float, n: int, seed: int | None = None):
    """Pick `n` indices into `sensitivities`, biased by `surprise` in [-1, 1].

    `sensitivities` is a sequence of floats (NaN/None entries are excluded).
    Returns (indices, probabilities) — probabilities are the full distribution over the
    eligible entries, aligned with their original indices, for UI display.
    """
    s = np.array([np.nan if v is None else float(v) for v in sensitivities], dtype=float)
    ok = np.isfinite(s)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return [], {}
    z = s[idx]
    sd = z.std()
    z = (z - z.mean()) / sd if sd > 1e-12 else np.zeros_like(z)
    surprise = float(np.clip(surprise, -1.0, 1.0))
    logits = surprise * z / TEMP
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    rng = np.random.default_rng(seed)
    take = min(int(n), len(idx))
    chosen = rng.choice(idx, size=take, replace=False, p=p)
    return [int(c) for c in chosen], {int(i): float(q) for i, q in zip(idx, p)}
