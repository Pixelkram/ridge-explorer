"""Shared plumbing of the two k-independent surfaces: the Ridge microscope and the Mixing desk.

Both render arbitrary recipes through the Cascade's own path (`cascade.evaluate`: the same
pipeline, mixing, seed and image size as every chord and walk), and both revisit recipes they
have already paid for -- a microscope zoom at s/2 lands half its lattice on the parent's cells,
and a desk move along one prompt's line keeps that whole line. So the one thing shared here is a
per-run render cache keyed by (recipe, denoising steps): a recipe is rendered at most once per
fidelity per run, and only a render that actually happened is charged.

Deliberately thin. `cs.evaluate` is called through the module attribute, so the tests' stub of
it (tests/metro_test.py's pattern) reaches these surfaces too.
"""
import numpy as np

from backend.services import cascade as cs

# A recipe's cache key rounds to 1e-9: far below any spacing either surface probes at (the
# finest is a desk refine bracket of 0.05/8 = 0.006), far above the float noise of
# re-deriving the same point along two routes (centre + s*offset vs a zoom's own centre).
KEY_DIGITS = 9


def point_key(w, steps):
    """(rounded recipe, denoising steps): what makes two renders the same image."""
    return (tuple(round(float(v), KEY_DIGITS) for v in w), steps)


def render_cached(app, run, pool, weights, steps, label, ctl, on_arrival=None):
    """Render every weight vector not already rendered at these steps; returns (gis, n_new).

    gis    -- one global image index per input (None where a render never arrived), cache hits
              included, so callers read every point the same way;
    n_new  -- how many renders actually arrived in THIS call: the only images to charge.

    `steps` None means the run's full denoising steps (as in `cs.evaluate`), and is keyed as
    that number, so a full-fidelity render requested either way is the same cache entry.
    Duplicates inside one batch are rendered once. `on_arrival(i, gi)` fires for every input
    index, hits immediately and misses as their render lands, so a live view fills the same
    way whichever it was.
    """
    eff = int(run.steps) if steps is None else int(steps)
    cache = run.point_cache
    gis = [None] * len(weights)
    todo, todo_of = [], {}          # key -> position in todo; todo entry = (w, [input idx])
    for i, w in enumerate(weights):
        key = point_key(w, eff)
        gi = cache.get(key)
        if gi is not None and gi in run.embeddings:
            gis[i] = gi
            if on_arrival is not None:
                on_arrival(i, gi)
            continue
        if key in todo_of:
            todo[todo_of[key]][1].append(i)
        else:
            todo_of[key] = len(todo)
            todo.append((np.asarray(w, dtype=float), [i], key))
    if not todo:
        return gis, 0

    def _arrive(li, gi):
        for i in todo[li][1]:
            gis[i] = gi
            if on_arrival is not None:
                on_arrival(i, gi)

    got = cs.evaluate(app, run, pool, [t[0] for t in todo], run.seed, label, ctl=ctl,
                      on_arrival=_arrive, steps=steps)
    n_new = 0
    for (w, idxs, key), gi in zip(todo, got):
        if gi is None:
            continue
        n_new += 1
        cache[key] = gi
        for i in idxs:
            gis[i] = gi
    return gis, n_new


def legal_recipe(w, k, tol_sum=1e-3):
    """A weight vector over k prompts as a recipe, or (None, reason).

    The add-a-chain rule of routers/metro.py: right length, finite, non-negative (to 1e-9) and
    summing to 1 to 1e-3, then renormalised exactly -- a slider's round-off is forgiven, a vector
    that is simply not a recipe is not.
    """
    if w is None:
        return None, "no weights given"
    w = [float(v) for v in w]
    if len(w) != k:
        return None, f"{len(w)} weights for k={k} prompts"
    if any(v != v or v in (float("inf"), float("-inf")) for v in w):
        return None, "a weight is not finite"
    if min(w) < -1e-9 or abs(sum(w) - 1.0) > tol_sum:
        return None, (f"weights must be non-negative and sum to 1 "
                      f"(got min {min(w):.4g}, sum {sum(w):.6g})")
    return cs._clipn(np.asarray(w, dtype=float), k), None
