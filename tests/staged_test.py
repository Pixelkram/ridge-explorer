"""Offline checks of the staged early-readout probes (backend/services/staged.py, the Cascade's
and the mixing desk's probe_mode = "staged", routers/staged.py). No GPU, no model.

The GPU side is a FAKE POOL: submit(task) answers every StagedReadoutTask / StagedResumeTask /
StagedTraceTask / StagedParityTask at once into app.state.hike_inbox, as the result collector
would, from two analytic fields -- a FULL field (what a finished image's DINOv2 says) and an x̂0
field (what the step-t readout says: the full field's bands at lower contrast plus a SECOND set of
bands the full field does not have, so the readout flags some segments that are no crossing).
So the real readout / resume / _dispatch / LRU / ledger code runs; only the worker bodies are
faked (their bit-identity is tests/staged_parity_cpu.py's job). cascade.evaluate (bisection,
scoring, patches, refine) is stubbed with the full field, as tests/certify_test.py does.

  (i)   RULES -- flagging (>= theta, a missing readout is no evidence, skip_first), resume
        selection (both probes of every flagged segment, each once, finished ones skipped),
        the latent LRU (LRU order, byte cap, eviction, oversize refusal), the latent key, the
        latent bytes round trip, the cost ledger (measured and nominal image-eq), and the
        router's array encoding.
  (ii)  CASCADE -- a staged survey against a steps survey of the same field: the same crossings,
        found on exactly the segments the rule predicts, with only flagged segments' probes
        finished, crossing thumbnails from finished images, no chord probe through evaluate;
        with certify on, the rebracket round disappears (its brackets are already exact) while
        bisection still runs; the ledger's image-eq and resumed share; child rays never flag
        their first segment.
  (iii) DESK -- a staged position: one readout batch, the mix always finished, changes only
        between finished samples and bracketing the full field's crossing, stretch thumbnails
        from finished images, refine reusing the finished ends, the staged projection.
  (iv)  LRU IN THE LOOP -- a cache too small for the batch: evicted probes are resumed from
        noise (same label, counted from_scratch), the rest from their cached latents.
  (v)   ROUTERS through TestClient -- /staged-trace and /staged-parity end to end with the
        documented result format, refusals, failure reporting, /staged-cache; probe_mode on
        /api/cascade/start and /api/desk/start (bounds, status fields), and a live staged
        cascade through its router.

Run: python tests/staged_test.py
"""
import math
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services import staged as st              # noqa: E402
from backend.services import cascade as cs             # noqa: E402
from backend.services import desk as dk                # noqa: E402
from backend.services import gpu_pool as gp            # noqa: E402
from backend.services.cascade import COS_T             # noqa: E402

fails = []
S, T, THETA = 8, 4, 0.10
DIM = 64
STEP_S, DD_S = 0.10, 0.05          # fake worker timings: seconds per step, per decode+DINOv2


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


# ------------------------------------------------------------------------- analytic fields
def _dir(k, seed):
    v = np.random.default_rng(seed).standard_normal(k)
    v -= v.mean()
    return v / np.linalg.norm(v)


def _bands(k, period, dir_seed, off, n=20):
    nv = _dir(k, dir_seed)

    def f(w):
        b = int(math.floor(float(np.dot(np.asarray(w, dtype=float), nv)) / period))
        e = np.zeros(DIM)
        e[off + b % n] = 1.0
        e[off + (b + 1) % n] = 0.15
        return e / np.linalg.norm(e)
    return f


def fields(k):
    """(full, xhat): the full field's bands; the readout sees them at lower contrast, plus a
    second band set (period 0.09) that never reaches the finished image, plus a faint smooth
    drift. Neighbouring full bands are 0.85 apart (> COS_T); in the readout a full-band change
    is ~0.29 and a ghost-band change ~0.11, both >= THETA, and a within-band step << THETA."""
    full = _bands(k, 0.12, 7, 0)
    ghost = _bands(k, 0.09, 8, 20)
    drift = _dir(k, 9)

    def xhat(w):
        e = np.zeros(DIM)
        e[63] = 1.0
        e = e + 0.8 * full(w) + 0.5 * ghost(w)
        e[62] = 0.02 * math.sin(3.0 * float(np.dot(np.asarray(w, dtype=float), drift)))
        return e / np.linalg.norm(e)
    return full, xhat


# ------------------------------------------------------------------------- the fake pool
def _timing(steps, scratch=None):
    t = {"denoise_s": STEP_S * steps, "decode_dino_s": DD_S, "steps": steps}
    if scratch is not None:
        t["from_scratch"] = scratch
    return t


class FakePool:
    """Answers staged tasks synchronously into app.state.hike_inbox (the collector's routing)."""

    def __init__(self, app, full, xhat, n_gpus=2, fail=()):
        self.app, self.full, self.xhat = app, full, xhat
        self.n_gpus = n_gpus
        self.cancel_epoch = types.SimpleNamespace(value=0)
        self.fail = set(fail)
        self.tasks = []
        self.resumed = []            # (index, had_latent)
        self.read = []               # indices read out

    def submit(self, task):
        self.tasks.append(task)
        box = self.app.state.hike_inbox[task.job_id]

        def put(kind, idx, data, err=""):
            box.append(gp.StagedResult(job_id=task.job_id, gpu_id=0, kind=kind, index=idx,
                                       data=data, error=err))
        if isinstance(task, gp.StagedReadoutTask):
            for idx, w in task.points:
                self.read.append(idx)
                put("readout", idx, {"emb": self.xhat(w).astype(np.float32), "thumb": f"x{idx}".encode(),
                                     "latent": bytes(64), "shape": (1, 4, 8), "dtype": "bfloat16",
                                     "timing": _timing(task.t)})
        elif isinstance(task, gp.StagedResumeTask):
            for idx, (w, lat, _shape, _dt) in task.points:
                self.resumed.append((idx, lat is not None))
                scratch = lat is None
                put("resume", idx, {"emb": self.full(w).astype(np.float32), "thumb": f"full{idx}".encode(),
                                    "timing": _timing(task.steps if scratch else task.steps - task.t,
                                                      scratch)})
        elif isinstance(task, gp.StagedTraceTask):
            if task.onepass_points:
                n = len(task.onepass_points)
                put("onepass", -1, {"divs": [0.01 * i for i in range(n - 1)], "onepass_s": 0.2 * n, "n": n})
            for idx, w in task.points:
                if idx in self.fail:
                    put("trace", idx, {}, "RuntimeError: boom")
                    continue
                xh = np.stack([self.xhat(w)] * (task.steps - 1) + [self.full(w)])[:, :DIM]
                xh = np.pad(xh, ((0, 0), (0, 768 - DIM)))
                put("trace", idx, {"xhat": xh.astype(np.float32),
                                   "denoise_cum_s": [STEP_S * (i + 1) for i in range(task.steps)],
                                   "decode_dino_s": [DD_S] * task.steps,
                                   "fourstep": np.pad(self.full(w), (0, 768 - DIM)).astype(np.float32),
                                   "fourstep_s": 0.4, "gpu_id": 3})
        elif isinstance(task, gp.StagedParityTask):
            for idx, _w in task.points:
                put("parity", idx, {"cos_resumed_vs_direct": 1.0, "max_abs_latent_diff": 0.0,
                                    "latent_bit_identical": True, "pixels_identical": True,
                                    "cos_resumed_vs_pipe": 1.0, "max_abs_latent_diff_pipe": 0.0,
                                    "pixels_identical_pipe": True, "timing": {"direct_s": 1.0}, "gpu_id": 1})


def _app(cache=None):
    return types.SimpleNamespace(state=types.SimpleNamespace(
        hike_inbox={}, hike_task_errors={},
        staged_cache=cache if cache is not None else st.LatentCache()))


def _stub_eval(field, labels):
    """cascade.evaluate for the full-fidelity rounds (bisect / score / patches / refine)."""
    def stub(app, r, pool, weights, seed, label, ctl=None, on_arrival=None, steps=None):
        labels.append((label.split(".")[0], len(weights), steps))
        out = []
        for li, w in enumerate(weights):
            with r._lock:
                gi = r.next_idx
                r.next_idx += 1
            r.embeddings[gi] = field(w)
            r.generated += 1
            if r.thumbs is not None:
                r.thumbs[gi] = f"eval{gi}".encode()
            if on_arrival is not None:
                on_arrival(li, gi)
            out.append(gi)
        return out
    return stub


# ------------------------------------------------------------------------- (i) rules
def part1():
    print("\n(i) flagging, resume selection, LRU, key, bytes, ledger, encoding")
    a, b, c = np.eye(3)
    near = (a + 0.3 * b) / np.linalg.norm(a + 0.3 * b)          # 1 - cos(a, near) ~ 0.042
    embs = [a, a, near, b, None, b, c]
    divs = st.segment_divs(embs)
    fl = st.flag_segments(embs, THETA)
    check(fl == [False, False, True, False, False, True] and divs[4] is None,
          "a segment is flagged iff its readouts are >= theta apart; a missing readout flags nothing",
          f"{[None if d is None else round(d, 3) for d in divs]}")
    d_exact = st.segment_divs([a, near])[0]
    check(st.flag_segments([a, near], d_exact) == [True]
          and st.flag_segments([a, near], d_exact + 1e-12) == [False],
          "the threshold is inclusive (>= theta)")
    check(st.flag_segments([a, b, c], THETA, skip_first=True) == [False, True],
          "skip_first drops segment 0 (a child ray's parent crossing)")
    seqs = [[10, 11, 12, 13], [20, None, 22, 23], [12, 30]]
    flags = [[True, True, False], [True, False, True], [True]]
    sel = st.resume_selection(seqs, flags)
    check(sel == [10, 11, 12, 20, 22, 23, 30],
          "resume = both probes of every flagged segment, each once, in first-seen order; a "
          "missing probe is skipped", f"{sel}")
    check(st.resume_selection(seqs, flags, done={11, 23}) == [10, 12, 20, 22, 30],
          "probes that already carry a full-fidelity label are not resumed again")
    check(st.resume_selection(seqs, [[False] * 3, [False] * 3, [False]]) == [],
          "no flag, nothing finished")

    cache = st.LatentCache(cap_bytes=300)
    for i in range(3):
        cache.put(("k", i), bytes(100), (1, 2), "bfloat16")
    cache.get(("k", 0))                                   # refresh 0 -> 1 is now the oldest
    cache.put(("k", 3), bytes(100), (1, 2), "bfloat16")
    s1 = cache.stats()
    check(("k", 1) not in cache and ("k", 0) in cache and ("k", 3) in cache and s1["entries"] == 3
          and s1["bytes"] == 300 and s1["evictions"] == 1,
          "LRU: a get refreshes an entry, the least recently used one is evicted at the byte cap",
          f"{s1}")
    check(cache.get(("k", 1)) is None and cache.stats()["misses"] == 1 and cache.stats()["hits"] == 1,
          "a miss is counted (and returns None)")
    check(cache.put(("big",), bytes(301), (1,), "bfloat16") is False and ("big",) not in cache
          and cache.stats()["bytes"] == 300,
          "an entry larger than the whole cap is refused, nothing evicted for it")
    cache.put(("k", 0), bytes(50), (1,), "bfloat16")
    check(cache.stats()["bytes"] == 250, "re-putting a key replaces its bytes in the accounting")

    k0 = st.latent_key(["a", "b"], [0.25, 0.75], 42, 8, 4, 4.0, 512, 512)
    variants = [st.latent_key(["a", "c"], [0.25, 0.75], 42, 8, 4, 4.0, 512, 512),
                st.latent_key(["a", "b"], [0.25, 0.7500001], 42, 8, 4, 4.0, 512, 512),
                st.latent_key(["a", "b"], [0.25, 0.75], 43, 8, 4, 4.0, 512, 512),
                st.latent_key(["a", "b"], [0.25, 0.75], 42, 10, 4, 4.0, 512, 512),
                st.latent_key(["a", "b"], [0.25, 0.75], 42, 8, 3, 4.0, 512, 512),
                st.latent_key(["a", "b"], [0.25, 0.75], 42, 8, 4, 3.5, 512, 512),
                st.latent_key(["a", "b"], [0.25, 0.75], 42, 8, 4, 4.0, 256, 512)]
    check(all(v != k0 for v in variants)
          and st.latent_key(["a", "b"], [0.25 + 1e-12, 0.75], 42, 8, 4, 4.0, 512, 512) == k0,
          "the latent key changes with prompts, recipe, seed, S, t, guidance and size (recipe "
          "rounded to 1e-9)")

    import torch
    x = torch.randn(1, 6, 8, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    bb, shape, dt = st.latent_to_bytes(x)
    y = st.bytes_to_latent(bb, shape, dt, "cpu")
    x32 = torch.randn(2, 3)
    y32 = st.bytes_to_latent(*st.latent_to_bytes(x32), "cpu")
    check(torch.equal(x, y) and y.dtype == torch.bfloat16 and len(bb) == 2 * 48
          and torch.equal(x32, y32),
          "latent -> bytes -> latent is bit-exact (bf16 as raw int16; fp32 too)")

    lg = st.Ledger(S, T)
    c1 = lg.add("readout", _timing(T))
    c2 = lg.add("resume", _timing(S - T, False))
    c3 = lg.add("resume", _timing(S, True))
    unit = S * STEP_S + DD_S
    rep = lg.report()
    check(abs(c1 - (T * STEP_S + DD_S) / unit) < 1e-12 and abs(c2 - ((S - T) * STEP_S + DD_S) / unit) < 1e-12
          and abs(c3 - 1.0) < 1e-12 and abs(rep["image_eq"] - (c1 + c2 + c3)) < 1e-4
          and abs(rep["image_eq_nominal"] - (T / S + (S - T) / S + 1.0)) < 1e-12,
          "ledger: a readout costs its t steps + one decode over (S steps + one decode); a resume "
          "its S-t steps + one decode; a from-scratch resume one image; nominal t/S, (S-t)/S, 1",
          f"readout {c1:.4f}, resume {c2:.4f}, scratch {c3:.4f} image-eq")
    check(rep["readouts"] == 1 and rep["resumed"] == 1 and rep["from_scratch"] == 1
          and rep["resumed_share"] == 2.0 and abs(rep["decode_share"] - DD_S / unit) < 1e-4,
          "ledger report: counts, resumed share (resumes per readout), decode share of an image")
    lg0 = st.Ledger(S, T)
    check(abs(lg0.add("readout", {}) - T / S) < 1e-12, "without timings an op is charged nominally")

    from backend.routers import staged as rs
    arr = np.random.default_rng(1).standard_normal((3, 8, 5))
    e16, e32, el = (rs.encode_array(arr, e) for e in ("f16", "f32", "list"))
    check(np.allclose(rs.decode_array(e16), arr, atol=2e-3) and np.array_equal(rs.decode_array(e32), arr.astype("<f4"))
          and np.allclose(rs.decode_array(el), arr) and e16["shape"] == [3, 8, 5] and e16["dtype"] == "<f2",
          "array encoding: base64 little-endian f16 / f32 with shape, or nested lists; decode_array inverts")


# ------------------------------------------------------------------------- (ii) cascade
def _cascade(mode, certify, k=4, n_chords=6, seed=11, branch=0, depth=0, cache=None):
    full, xhat = fields(k)
    run = cs.CascadeRun(run_id="c" + mode, prompts=[f"p{i}" for i in range(k)], seed=seed, steps=S,
                        height=64, width=64, guidance_scale=4.0, n_chords=n_chords, n_patches=2,
                        certify=certify, probe_mode=mode, staged_t=T, staged_theta=THETA,
                        branch=branch, depth=depth, branch_top_pct=100)
    run.thumbs = {}
    app = _app(cache)
    pool = FakePool(app, full, xhat)
    labels = []
    orig = cs.evaluate
    cs.evaluate = _stub_eval(full, labels)
    try:
        cs.run_cascade(app, run, pool)
    finally:
        cs.evaluate = orig
    return run, pool, labels, full, xhat


def part2():
    print("\n(ii) the Cascade: staged vs steps on one field (fake pool, stubbed evaluate)")
    captured = []
    orig_rs = st.run_sequences

    def spy(app, run, pool, gis_seqs, label, ctl=None, skip_first=None, always=(), on_resume=None):
        captured.append([list(q) for q in gis_seqs])
        return orig_rs(app, run, pool, gis_seqs, label, ctl, skip_first, always, on_resume)
    st.run_sequences = spy
    try:
        run_s, pool, labels, full, xhat = _cascade("staged", certify=False)
    finally:
        st.run_sequences = orig_rs
    run_p, _pp, labels_p, _f, _x = _cascade("steps", certify=False)
    check(run_s.status == "complete" and run_p.status == "complete"
          and not any(lab.startswith("chords") for lab, _n, _s in labels)
          and all(lab.startswith("chords") for lab, _n, _s in labels_p),
          "staged chord probes never go through evaluate (the steps survey's do)",
          f"evaluate rounds staged {labels}")
    key = lambda x: (tuple(np.round(x.wa, 9)), tuple(np.round(x.wb, 9)))      # noqa: E731
    xs, xp = sorted(map(key, run_s.crossings)), sorted(map(key, run_p.crossings))
    check(len(xs) >= 3 and xs == xp,
          "the staged survey finds exactly the crossings the steps survey finds on the same field",
          f"{len(xs)} crossings")
    # independent replay of the rule over the sequences the survey handed to run_sequences
    readout_w = dict(run_s.staged_w)
    gis = sorted(readout_w)
    seqs = [q for batch in captured for q in batch]
    stride_ok = all(abs(np.linalg.norm(np.asarray(readout_w[g2]) - np.asarray(readout_w[g1])) - run_s.stride) < 1e-3
                    for q in seqs for g1, g2 in list(zip(q, q[1:]))[1:-1])
    check(len(seqs) == len(run_s.chords_geo) and sorted(g for q in seqs for g in q) == gis and stride_ok,
          "the sequences are the chords: one per chord, every readout once, consecutive probes one "
          "stride apart")
    n_flag = n_seg = n_true = n_true_flagged = 0
    want_resumed = set()
    ok_rule = True
    for seq in seqs:
        for g1, g2 in zip(seq, seq[1:]):
            n_seg += 1
            dx = 1 - float(np.dot(xhat(readout_w[g1]), xhat(readout_w[g2])))
            dfull = 1 - float(np.dot(full(readout_w[g1]), full(readout_w[g2])))
            flagged = dx >= THETA
            n_flag += flagged
            if flagged:
                want_resumed |= {g1, g2}
            if dfull > COS_T:
                n_true += 1
                n_true_flagged += flagged
            both = g1 in run_s.embeddings and g2 in run_s.embeddings
            ok_rule &= (not flagged) or both
    check(n_true >= 3 and n_true_flagged == n_true and n_flag > n_true,
          "precondition: every true crossing segment is flagged by the readout, and the readout "
          "also flags segments that are no crossing (ghost bands)",
          f"{n_true} true, {n_flag} flagged of {n_seg} segments")
    resumed = [i for i, _l in pool.resumed]
    check(set(resumed) == want_resumed and len(resumed) == len(set(resumed)) and ok_rule,
          "resumed = exactly both probes of every flagged segment, no duplicates; unflagged "
          "segments' probes are never finished",
          f"{len(resumed)} of {len(gis)} probes resumed")
    check(set(run_s.embeddings) == want_resumed | {g for g in run_s.embeddings if g not in readout_w},
          "only resumed probes carry a label (run.embeddings); the rest keep their x̂0 readout only")
    check(all(x.ea is not None and x.thumb in want_resumed and run_s.thumbs[x.thumb] == f"full{x.thumb}".encode()
              and x.thumb not in run_s.staged_preview and x.exact for x in run_s.crossings)
          and all(abs(1 - float(np.dot(x.ea, full(x.wa)))) < 1e-9 for x in run_s.crossings),
          "every crossing's sides are full-field labels and its thumbnail is a finished image")
    unres = [g for g in gis if g not in want_resumed]
    check(unres and all(run_s.thumbs[g] == f"x{g}".encode() and g in run_s.staged_preview for g in unres),
          "unfinished probes show their x̂0 preview, marked as such")
    rep = st.ledger(run_s).report()
    n_r, n_s = len(gis), len(resumed)
    unit = S * STEP_S + DD_S
    want_eq = n_r * (T * STEP_S + DD_S) / unit + n_s * ((S - T) * STEP_S + DD_S) / unit
    check(rep["readouts"] == n_r and rep["resumed"] == n_s and abs(rep["resumed_share"] - n_s / n_r) < 1e-12
          and abs(rep["image_eq"] - want_eq) < 1e-3
          and abs(rep["image_eq_nominal"] - (n_r * T / S + n_s * (S - T) / S)) < 1e-6
          and rep["segments"] == n_seg and rep["flagged"] == n_flag and run_s.generated == n_s,
          "ledger: readouts, resumed share, measured and nominal image-eq, segment counts; only "
          "finished images count as generated",
          f"{rep['image_eq']:.1f} image-eq for {n_r} probes ({100 * rep['resumed_share']:.0f} % resumed)"
          f" vs {n_r / 2:.1f} for 4-step probes")
    check(any("staged readout at step 4/8" in n for n in run_s.notes),
          "the run's notes summarise the staged phase")

    # certify on: rebracket disappears, bisection stays
    run_c, _pc, labels_c, *_ = _cascade("staged", certify=True)
    run_cp, _pp, labels_cp, *_ = _cascade("steps", certify=True)
    lab_c = [lab for lab, _n, _s in labels_c]
    lab_cp = [lab for lab, _n, _s in labels_cp]
    check("rebracket" not in lab_c and "rebracket" in lab_cp
          and any(lab.startswith("bisect") for lab in lab_c)
          and st.ledger(run_c).rebracket_skipped == len(run_c.crossings) > 0
          and any("rebracket skipped" in n for n in run_c.notes),
          "certify on: the staged run skips the full-fidelity rebracket (its bracket ends are "
          "resumed images) and goes straight to bisection; the steps run still rebrackets",
          f"staged rounds {sorted(set(lab_c))}")
    check(run_c.status == "complete" and all(x.b is not None for x in run_c.crossings)
          and len(run_c.crossings) == len(run_cp.crossings),
          "the certified staged run completes scoring with the same crossings")

    # child rays: their first segment is never flagged
    lg = st.Ledger(S, T)
    run = types.SimpleNamespace(staged_ledger=lg, staged_theta=THETA, embeddings={}, phase_total=0,
                                xhat={1: np.eye(3)[0], 2: np.eye(3)[1], 3: np.eye(3)[2]})
    called = []
    orig = st.resume
    st.resume = lambda app, r, pool, sel, label, ctl=None, on=None: called.append(list(sel))
    try:
        flags, sel = st.run_sequences(None, run, None, [[1, 2, 3], [1, 2, 3]], "x",
                                      skip_first=[True, False])
    finally:
        st.resume = orig
    check(flags == [[False, True], [True, True]] and sel == [2, 3, 1] and called == [[2, 3, 1]]
          and lg.segments == 3 and lg.flagged == 3,
          "run_sequences: skip_first sequences never flag segment 0 (nor count it); one resume "
          "call with the deduplicated selection")

    # steps mode never touches the staged path
    run_p2, pool_p2, *_ = _cascade("steps", certify=False)
    check(not pool_p2.tasks and run_p2.staged_ledger is None and not run_p2.xhat,
          "probe_mode 'steps' submits no staged task and keeps no staged state")


# ------------------------------------------------------------------------- (iii) desk
def _desk_fields(k, band=0.0005):
    rng = np.random.default_rng(2)
    p = rng.standard_normal(k)
    p -= p.mean()
    p /= np.linalg.norm(p)
    Q = np.linalg.qr(rng.standard_normal((DIM, 3)))[0]

    def full(w):
        g = float((np.asarray(w, dtype=float) - 1.0 / k) @ p)
        e = Q[:, 0] + 2.5 * math.tanh(g / band) * Q[:, 1] + 0.1 * math.sin(7 * g) * Q[:, 2]
        return e / np.linalg.norm(e)

    def xhat(w):
        g = float((np.asarray(w, dtype=float) - 1.0 / k) @ p)
        e = Q[:, 0] + 1.0 * math.tanh(g / 0.012) * Q[:, 1] + 0.1 * math.sin(7 * g) * Q[:, 2]
        return e / np.linalg.norm(e)
    return full, xhat, p


def _alpha_star(w0, i, p):
    k = len(w0)
    g0 = float((dk.line_point(w0, i, 0.0) - 1.0 / k) @ p)
    g1 = float((dk.line_point(w0, i, 1.0) - 1.0 / k) @ p)
    if (g0 > 0) == (g1 > 0):
        return None
    return g0 / (g0 - g1)


def part3():
    print("\n(iii) the mixing desk in staged mode (fake pool, stubbed evaluate)")
    k = 4
    full, xhat, p = _desk_fields(k)
    w0 = np.array([0.33, 0.21, 0.17, 0.29])      # every alpha* well inside a sample bracket
    run = dk.DeskRun(run_id="d", prompts=[f"p{i}" for i in range(k)], seed=3, steps=S, height=64,
                     width=64, guidance_scale=4.0, probe_steps=4, probe_mode="staged", staged_t=T,
                     staged_theta=THETA)
    app = _app()
    pool = FakePool(app, full, xhat)
    labels = []
    orig = cs.evaluate
    cs.evaluate = _stub_eval(full, labels)
    try:
        pos, _ = dk.add_position(run, w0, 0.05, False)
        dk.run_position(app, run, pool, pos)
        n_read = len(pool.read)
        check(pos.status == "complete" and n_read == k * 21 + 1 and not labels
              and len({tuple(np.round(run.staged_w[g], 9)) for g in pool.read}) == n_read,
              "one readout batch: k lines x 21 samples + the mix, each recipe once; nothing through "
              "evaluate", f"{n_read} readouts")
        check(pos.w0_image in run.embeddings,
              "the current mix is always finished (it is the image the desk shows)")
        ok, n_cross = True, 0
        for ln in pos.lines:
            a_star = _alpha_star(w0, ln["i"], p)
            imgs = ln["images"]
            for f in ln["flips"]:
                ok &= imgs[f["lo"]] in run.embeddings and imgs[f["hi"]] in run.embeddings
            ok &= len(ln["resumed"]) == 21 and len(ln["flagged"]) == 20
            ok &= all(ln["resumed"][t] and ln["resumed"][t + 1] for t, fl in enumerate(ln["flagged"]) if fl)
            if a_star is None:
                ok &= ln["flips"] == []
                continue
            n_cross += 1
            ok &= (len(ln["flips"]) == 1
                   and ln["alphas"][ln["flips"][0]["lo"]] <= a_star <= ln["alphas"][ln["flips"][0]["hi"]])
        check(ok and n_cross >= 2,
              "every crossing line shows exactly one change, between two FINISHED samples, "
              "bracketing the full field's alpha*; flagged segments have both ends finished",
              f"{n_cross}/{k} lines cross")
        resumed = {g for g, _l in pool.resumed}
        flagged_ends = {ln["images"][t + d] for ln in pos.lines for t, fl in enumerate(ln["flagged"]) if fl
                        for d in (0, 1)}
        check(resumed == flagged_ends | {pos.w0_image} and len(pool.resumed) == len(resumed)
              and len(resumed) < n_read / 2,
              "finished = the flagged segments' samples + the mix, each once -- a minority of the "
              "readouts", f"{len(resumed)} of {n_read}")
        thumbs = [s["thumb"] for ln in pos.lines for s in ln["segments"]]
        check(all(t in run.embeddings for t in thumbs) and all(t >= 0 for t in thumbs),
              "every stretch shows a finished image (a change-free line shows the mix)")
        lg = st.ledger(run)
        check(abs(pos.cost - lg.image_eq) < 1e-9 and abs(run.cost_image_eq - lg.image_eq) < 1e-9
              and run.n_probes == 0,
              "the position's cost is the ledger's measured image-eq", f"{pos.cost:.2f} image-eq")
        n0 = len(labels)
        pos_r, _ = dk.add_position(run, w0, 0.05, True)
        n_res0 = len(pool.resumed)
        dk.run_position(app, run, pool, pos_r)
        labs = [lab for lab, _n, _s in labels[n0:]]
        check(labs == ["desk1bis0", "desk1bis1", "desk1bis2"] and len(pool.resumed) == n_res0
              and all(f["refined"] for ln in pos_r.lines for f in ln["flips"]),
              "refine reuses the finished ends (no ends round, nothing re-resumed) and bisects",
              f"{labs}")
    finally:
        cs.evaluate = orig
    tot, rows = dk.projected_cost(4, 21, False, 4, probe_mode="staged", staged_t=T, steps=S)
    tot_r, _ = dk.projected_cost(4, 21, True, 4, probe_mode="staged", staged_t=T, steps=S)
    check(abs(tot - (4 * 21 * (T / S) + 4 * 21 * (S - T) / S + 1.0)) < 1e-9
          and abs(tot_r - tot - 4 * dk.REFINE_MAX_PER_LINE * dk.REFINE_ROUNDS) < 1e-9
          and dk.projected_cost(8, 21, True, 4, probe_mode="staged")[0] <= dk.MAX_REQUEST_IMAGE_EQ,
          "staged projection: readouts at t/S, worst-case resume of every sample, the mix finished, "
          "refine = its midpoints only (ends already finished); k = 8 with refine fits one request",
          f"{tot:.1f} / {tot_r:.1f} image-eq")


# ------------------------------------------------------------------------- (iv) LRU in the loop
def part4():
    print("\n(iv) the latent LRU inside readout -> resume")
    k = 3
    full, xhat = fields(k)
    run = dk.DeskRun(run_id="l", prompts=["a", "b", "c"], seed=1, steps=S, height=64, width=64,
                     guidance_scale=4.0, probe_mode="staged", staged_t=T, staged_theta=THETA)
    cache = st.LatentCache(cap_bytes=3 * 64)
    app = _app(cache)
    pool = FakePool(app, full, xhat, n_gpus=1)
    ctl = types.SimpleNamespace(status="running", notes=[])
    ws = [[1 - 0.1 * i, 0.1 * i, 0.0] for i in range(6)]
    gis = st.readout(app, run, pool, ws, "lru", ctl=ctl)
    check(gis == list(range(6)) and len(cache) == 3 and cache.stats()["evictions"] == 3,
          "six readouts into a three-latent cache: the oldest three are evicted")
    got = st.resume(app, run, pool, gis, "lru", ctl=ctl)
    had = dict(pool.resumed)
    rep = st.ledger(run).report()
    check(got == set(gis) and [had[g] for g in gis] == [False] * 3 + [True] * 3
          and rep["from_scratch"] == 3 and rep["cache_misses"] == 3 and rep["resumed"] == 3
          and all(abs(1 - float(np.dot(run.embeddings[g], full(ws[g])))) < 1e-9 for g in gis),
          "evicted probes are re-run from noise (no latent sent; same label; counted from_scratch "
          "and as cache misses), cached ones resume from their latent", f"{rep}")


# ------------------------------------------------------------------------- (v) routers
def _wait(c, url, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        j = c.get(url).json()
        if j["status"] not in ("running", "pending"):
            return j
        time.sleep(0.02)
    return j


def part5():
    print("\n(v) routers through TestClient (fake pool)")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.routers import staged as rs
    from backend.routers import cascade as rc
    from backend.routers import desk as rd
    from backend.cache.thumbnail_cache import ThumbnailCache

    k = 4
    full, xhat = fields(k)
    app = FastAPI()
    for r in (rs.router, rc.router, rd.router):
        app.include_router(r)
    tmp = tempfile.TemporaryDirectory()
    app.state.hike_inbox = {}
    app.state.hike_task_errors = {}
    app.state.staged_cache = st.LatentCache()
    app.state.cache = ThumbnailCache(Path(tmp.name))
    pool = FakePool(app, full, xhat, n_gpus=3)
    app.state.gpu_pool = pool
    P = ["a", "b", "c", "d"]
    chord = [[0.7 - 0.05 * i, 0.1, 0.1, 0.1 + 0.05 * i] for i in range(9)]
    orig_eval = cs.evaluate
    cs.evaluate = _stub_eval(full, [])
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post("/api/probe/staged-trace", json={"prompts": P, "weights": chord})
            jid = r.json().get("job_id", "")
            check(r.status_code == 200 and r.json()["total"] == 9 and jid.startswith("trace_"),
                  "/staged-trace answers with a job id", f"HTTP {r.status_code}")
            j = _wait(c, f"/api/probe/staged-trace/{jid}")
            res = j.get("result") or {}
            xh = rs.decode_array(res.get("xhat", {"b64": "", "dtype": "<f2", "shape": [0]}))
            fs = rs.decode_array(res["fourstep"]) if "fourstep" in res else None
            want_final = np.stack([np.pad(full(w), (0, 768 - DIM)) for w in chord])
            check(j["status"] == "done" and res["n"] == 9 and res["steps"] == 8 and res["dim"] == 768
                  and xh.shape == (9, 8, 768) and res["encoding"] == "f16"
                  and np.allclose(xh[:, 7, :], want_final, atol=2e-3)
                  and np.allclose(xh[:, 3, :DIM], np.stack([xhat(w) for w in chord]), atol=2e-3)
                  and fs is not None and fs.shape == (9, 768) and res["fourstep_steps"] == 4,
                  "trace result: xhat [n, S, 768] (rows 1..S-1 x̂0, row S the final image), "
                  "fourstep [n, 768], base64 float16 by default")
            tm = res["timing"]
            check(len(res["onepass_div"]) == 8 and len(tm["denoise_cum_s"]) == 9
                  and len(tm["denoise_cum_s"][0]) == 8 and len(tm["decode_dino_s"][0]) == 8
                  and len(tm["fourstep_s"]) == 9 and tm["gpu_id"] == [3] * 9
                  and abs(tm["onepass_s_per_point"] - tm["onepass_s_total"] / 9) < 1e-12
                  and res["weights"] == chord and res["prompts"] == P,
                  "trace result: onepass_div (n-1), per-point cumulative denoise and decode "
                  "timings, short-schedule and 1-pass timings, the recipe echoed")
            ops = [t for t in pool.tasks if isinstance(t, gp.StagedTraceTask)]
            check(sum(1 for t in ops if t.onepass_points) == 1
                  and [t.onepass_points for t in ops if t.onepass_points][0] == chord
                  and sum(len(t.points) for t in ops) == 9 and len([t for t in ops if t.points]) == 3,
                  "the 1-pass readout runs once over the WHOLE chord in order; the points are "
                  "sharded over the GPUs")
            r = c.post("/api/probe/staged-trace", json={"prompts": P, "weights": chord, "encoding": "list",
                                                        "fourstep_steps": 0, "onepass": False,
                                                        "shard": False})
            j2 = _wait(c, f"/api/probe/staged-trace/{r.json()['job_id']}")
            res2 = j2["result"]
            last2 = [t for t in pool.tasks if isinstance(t, gp.StagedTraceTask)][-2:]
            check(j2["status"] == "done" and isinstance(res2["xhat"], list)
                  and np.asarray(res2["xhat"]).shape == (9, 8, 768) and "fourstep" not in res2
                  and "onepass_div" not in res2 and all(t.fourstep_steps == 0 for t in last2)
                  and [len(t.points) for t in last2] == [8, 1],
                  "encoding 'list', the short-schedule arm and the 1-pass switched off, shard off "
                  "(rounds of 8 points on one worker)")
            pool.fail = {4}
            r = c.post("/api/probe/staged-trace", json={"prompts": P, "weights": chord})
            j3 = _wait(c, f"/api/probe/staged-trace/{r.json()['job_id']}")
            pool.fail = set()
            check(j3["status"] == "error" and "4" in (j3["error"] or "") and j3["result"] is None
                  and any("boom" in n for n in j3["notes"]),
                  "a point that fails on the worker leaves the job in error, naming it")
            for body, code, why in (({"prompts": P, "weights": [[0.5, 0.5]] * 3}, 400, "wrong length"),
                                    ({"prompts": P, "weights": [[0.9, 0.9, 0.1, 0.1]] * 3}, 400, "no recipe"),
                                    ({"prompts": P, "weights": [[1, 0, 0, 0]]}, 422, "one point"),
                                    ({"prompts": P, "weights": chord, "encoding": "f64"}, 422, "bad encoding")):
                r = c.post("/api/probe/staged-trace", json=body)
                check(r.status_code == code, f"/staged-trace with {why} -> {code}", f"HTTP {r.status_code}")
            check(c.get("/api/probe/staged-trace/nope").json()["status"] == "unknown",
                  "an unknown trace job answers 'unknown'")

            r = c.post("/api/probe/staged-parity", json={"prompts": P, "points": chord[:5], "t": 4})
            jp = _wait(c, f"/api/probe/staged-parity/{r.json()['job_id']}")
            rp = jp["result"]
            check(jp["status"] == "done" and len(rp["points"]) == 5 and rp["all_bit_identical"]
                  and rp["min_cos_resumed_vs_direct"] == 1.0 and rp["min_cos_resumed_vs_pipe"] == 1.0
                  and [p_["index"] for p_ in rp["points"]] == list(range(5))
                  and rp["points"][2]["weights"] == chord[2]
                  and all(t.t == 4 for t in pool.tasks if isinstance(t, gp.StagedParityTask)),
                  "/staged-parity: per point cos / latent diff / bit identity, and the summary")
            for t_bad in (0, 8, 9):
                r = c.post("/api/probe/staged-parity", json={"prompts": P, "points": chord[:2], "t": t_bad})
                check(r.status_code == 422, f"/staged-parity with t = {t_bad} (S = 8) -> 422",
                      f"HTTP {r.status_code}")
            check(c.get("/api/probe/staged-parity/nope").json()["status"] == "unknown"
                  and c.get(f"/api/probe/staged-trace/{jp['job_id']}").json()["status"] == "unknown",
                  "unknown parity jobs, and a parity id on the trace route, answer 'unknown'")
            sc = c.get("/api/probe/staged-cache").json()
            check(set(sc) >= {"entries", "bytes", "cap_bytes", "hits", "misses", "evictions"},
                  "/staged-cache reports the LRU's counters", f"{sc}")

            # probe_mode on the production start endpoints
            body = {"k": 4, "prompts": P, "n_chords": 5, "n_patches": 2, "probe_mode": "staged",
                    "staged_t": 4, "staged_theta": THETA}
            r = c.post("/api/cascade/start", json=body)
            rid = r.json().get("run_id", "")
            js = _wait(c, f"/api/cascade/{rid}/status")
            check(r.status_code == 200 and js["status"] == "complete" and js["probe_mode"] == "staged"
                  and js["staged_t"] == 4 and js["staged_theta"] == THETA
                  and js["staged"]["readouts"] > 0 and js["staged"]["resumed_share"] is not None
                  and js["staged"]["image_eq"] > 0,
                  "a live staged Cascade through its router: status reports the mode, t, theta and "
                  "the ledger (readouts, resumed share, image-eq)",
                  f"{ {kk: js['staged'][kk] for kk in ('readouts', 'resumed', 'image_eq')} }")
            run = app.state.cascades[rid]
            x0 = next(iter(run.crossings), None)
            if x0 is not None:
                img = c.get(f"/api/cascade/{rid}/{x0.thumb}.jpg")
                check(img.status_code == 200 and img.content == f"full{x0.thumb}".encode(),
                      "a crossing's thumbnail serves the finished image")
            unres = next(iter(sorted(run.staged_preview)), None)
            if unres is not None:
                pi = c.get(f"/api/cascade/{rid}/point/{unres}").json()
                check(pi["preview"] is True and pi["weights"] is not None,
                      "point info marks an unfinished probe's image as an x̂0 preview")
            jd = c.get(f"/api/cascade/{rid}/status").json()
            check(c.post("/api/cascade/start", json=dict(body, probe_mode="steps")).status_code == 200
                  and jd["certify"] is False,
                  "probe_mode 'steps' is still accepted")
            for bad, why in ((dict(body, staged_t=8), "staged_t = steps"),
                             (dict(body, staged_t=0), "staged_t = 0"),
                             (dict(body, staged_theta=0), "theta 0"),
                             (dict(body, probe_mode="fast"), "an unknown mode")):
                r = c.post("/api/cascade/start", json=bad)
                check(r.status_code == 422, f"/api/cascade/start with {why} -> 422", f"HTTP {r.status_code}")
            js0 = c.post("/api/cascade/start", json={"k": 4, "prompts": P, "n_chords": 4}).json()
            j0 = _wait(c, f"/api/cascade/{js0['run_id']}/status")
            check(j0["probe_mode"] == "steps" and j0["staged"] is None and j0["staged_t"] is None,
                  "a run started without probe_mode is a steps run (the default is unchanged)")

            r = c.post("/api/desk/start", json={"prompts": P, "probe_mode": "staged", "staged_t": 4,
                                                "staged_theta": THETA})
            did = r.json().get("desk_id", "")
            jdk = _wait(c, f"/api/desk/{did}/status")
            check(r.status_code == 200 and jdk["status"] == "complete" and jdk["probe_mode"] == "staged"
                  and jdk["staged"]["readouts"] == 4 * 21 + 1 - 4
                  and all(len(ln["resumed"]) == 21 and len(ln["flagged"]) == 20 for ln in jdk["lines"])
                  and abs(jdk["cost_image_eq"] - jdk["staged"]["image_eq"]) < 1e-3,
                  "a staged desk through its router: status carries the mode, the ledger, and per "
                  "line which samples were finished and which segments flagged",
                  f"{jdk['staged']['readouts']} readouts, {jdk['staged']['resumed']} finished")
            r = c.post("/api/desk/start", json={"prompts": P, "probe_mode": "staged", "staged_t": 8})
            check(r.status_code == 422, "/api/desk/start with staged_t = steps -> 422")
            jdef = _wait(c, f"/api/desk/{c.post('/api/desk/start', json={'prompts': P}).json()['desk_id']}/status")
            check(jdef["probe_mode"] == "steps" and jdef["staged"] is None
                  and all(ln["resumed"] == [] for ln in jdef["lines"]),
                  "a desk started without probe_mode is a steps desk")
    finally:
        cs.evaluate = orig_eval
        tmp.cleanup()


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
    part5()
    print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
