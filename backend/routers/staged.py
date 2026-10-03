"""Research endpoints of the staged early-readout probes (services/staged.py), built for the
pre-registered h27 study (search_problem/outputs/h27_staged_probes/PREREG.md). Both run on the
same manual CFG loop the production staged probes use.

  POST /api/probe/staged-trace          one chord per request -> {job_id}; poll the GET below
  GET  /api/probe/staged-trace/{job_id}
  POST /api/probe/staged-parity         points + t -> {job_id}; poll the GET below
  GET  /api/probe/staged-parity/{job_id}
  GET  /api/probe/staged-cache          the latent LRU's counters

TRACE result (`result` of a finished job), n = len(weights), S = steps:
  n, steps, dim (768), encoding, prompts, weights, seed, guidance_scale, height, width
  xhat        [n, S, dim]  row t-1 (t = 1..S-1): DINOv2 (ViT-B/14-reg CLS, L2-normalised -- the
                           tool's readout) of x̂0 = x_t − σ_t v̂_t predicted by the t-th CFG step of
                           the S-step run; row S-1: the FINAL image (t = S), decoded exactly as
                           pipe() decodes it
  fourstep    [n, dim]     DINOv2 of the complete `fourstep_steps`-step image through the Cascade
                           probe's own code path (gpu_pool._discover_point); absent when
                           fourstep_steps = 0
  onepass_div [n-1]        the Fast-Scan 1-pass readout (one transformer call at the first
                           timestep, no CFG, x̂0 = noise − velocity, flattened, L2-normalised,
                           batches of 8 in chord order), as consecutive-pair cosine distances
                           1 − cos(p_i, p_{i+1}); absent when onepass = false
  timing      denoise_cum_s [n][S]  cumulative denoising wall-clock through step t (decodes
                                    excluded); readout-to-t cost = denoise_cum_s[t-1], resume
                                    t -> S = denoise_cum_s[S-1] − denoise_cum_s[t-1]
              decode_dino_s [n][S]  decode + DINOv2 of the step-t readout (t = S: final image)
              fourstep_s    [n]     the whole short-schedule probe (pipe() incl. decode + DINOv2)
              onepass_s_total, onepass_s_per_point (= total / n; the chord runs as one batch job)
              gpu_id        [n]
  Arrays under encoding "f16" / "f32" are {"b64", "dtype": "<f2" | "<f4", "shape"}: base64 of the
  little-endian C-order buffer -- np.frombuffer(base64.b64decode(b64), dtype).reshape(shape).
  Under "list" they are nested lists of floats.

PARITY result: points [{index, weights, cos_resumed_vs_direct, max_abs_latent_diff,
  latent_bit_identical, pixels_identical, cos_resumed_vs_pipe, max_abs_latent_diff_pipe,
  pixels_identical_pipe, latent_bytes, latent_shape, latent_dtype, timing, gpu_id}],
  all_bit_identical, min_cos_resumed_vs_direct, min_cos_resumed_vs_pipe. The latent travels
  through the same bytes round trip the production cache uses.
"""
import asyncio
import base64
import math
import time
import types
import uuid

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from backend.models import StagedTraceRequest, StagedParityRequest, StagedJobStart, StagedJobStatus
from backend.services import staged as st
from backend.services.gpu_pool import StagedTraceTask, StagedParityTask

router = APIRouter(prefix="/api/probe", tags=["staged"])


def _jobs(app):
    app.state.staged_jobs = getattr(app.state, "staged_jobs", {})
    return app.state.staged_jobs


def _check_points(points, k, what):
    """Every point a finite k-vector of (near) non-negative weights summing to ~1 -- used verbatim."""
    for i, w in enumerate(points):
        if len(w) != k:
            raise HTTPException(status_code=400, detail=f"{what}[{i}] has {len(w)} weights for k={k}")
        if any(not math.isfinite(float(v)) for v in w):
            raise HTTPException(status_code=400, detail=f"{what}[{i}] is not finite")
        if min(w) < -1e-6 or abs(sum(w) - 1.0) > 0.05:
            raise HTTPException(status_code=400,
                                detail=f"{what}[{i}] is no recipe (min {min(w):.3g}, sum {sum(w):.4g})")


def encode_array(a, encoding):
    a = np.asarray(a, dtype=np.float64)
    if encoding == "list":
        return a.tolist()
    dt = "<f2" if encoding == "f16" else "<f4"
    return {"b64": base64.b64encode(np.ascontiguousarray(a.astype(dt)).tobytes()).decode("ascii"),
            "dtype": dt, "shape": list(a.shape)}


def decode_array(obj):
    """Inverse of encode_array (for clients and tests)."""
    if isinstance(obj, dict):
        return np.frombuffer(base64.b64decode(obj["b64"]), dtype=obj["dtype"]).reshape(obj["shape"])
    return np.asarray(obj, dtype=np.float64)


def _status(job):
    return StagedJobStatus(job_id=job["id"], kind=job["kind"], status=job["status"], done=job["done"],
                           total=job["total"], elapsed_s=round(time.time() - job["t0"], 2),
                           notes=list(job["notes"]), error=job.get("error"),
                           result=job.get("result") if job["status"] == "done" else None)


def _new_job(app, kind, total, req):
    jid = f"{kind}_{uuid.uuid4().hex[:8]}"
    job = {"id": jid, "kind": kind, "status": "running", "done": 0, "total": total, "t0": time.time(),
           "notes": [], "error": None, "result": None, "req": req}
    _jobs(app)[jid] = job
    return job


# ---------------------------------------------------------------------------- trace

def run_trace(app, job):
    req = job["req"]
    pool = app.state.gpu_pool
    ctl = types.SimpleNamespace(status="running", notes=job["notes"])
    n = len(req.weights)
    pts = [[float(v) for v in w] for w in req.weights]
    onepass = {}
    if req.onepass:
        def make_op(tid, _shard):
            return StagedTraceTask(job_id=tid, basis=list(req.prompts), points=[], seed=req.seed,
                                   height=req.height, width=req.width, steps=req.steps,
                                   guidance_scale=req.guidance_scale, fourstep_steps=0,
                                   onepass_points=pts)
        st._dispatch(app, pool, ctl, [(-1, None)], make_op, lambda r: onepass.update(r.data),
                     "onepass", job["id"], n_gpus=1)
    per = {}

    def make(tid, shard):
        return StagedTraceTask(job_id=tid, basis=list(req.prompts), points=shard, seed=req.seed,
                               height=req.height, width=req.width, steps=req.steps,
                               guidance_scale=req.guidance_scale, fourstep_steps=req.fourstep_steps)

    def on_result(r):
        per[r.index] = r.data
        job["done"] = len(per)

    if ctl.status == "running":
        st._dispatch(app, pool, ctl, list(enumerate(pts)), make, on_result, "trace", job["id"],
                     n_gpus=None if req.shard else 1)
    missing = [i for i in range(n) if i not in per]
    if req.onepass and "divs" not in onepass:
        missing.append("onepass")
    if missing or ctl.status != "running":
        job["status"] = "error"
        job["error"] = (f"incomplete: missing {missing[:12]}{'…' if len(missing) > 12 else ''}"
                        if missing else f"stopped ({ctl.status})")
        return
    S = req.steps
    enc = req.encoding
    res = {"n": n, "steps": S, "dim": int(per[0]["xhat"].shape[1]), "encoding": enc,
           "prompts": list(req.prompts), "weights": pts, "seed": req.seed,
           "guidance_scale": req.guidance_scale, "height": req.height, "width": req.width,
           "xhat": encode_array(np.stack([per[i]["xhat"] for i in range(n)]), enc),
           "timing": {"denoise_cum_s": [[float(x) for x in per[i]["denoise_cum_s"]] for i in range(n)],
                      "decode_dino_s": [[float(x) for x in per[i]["decode_dino_s"]] for i in range(n)],
                      "gpu_id": [int(per[i].get("gpu_id", -1)) for i in range(n)]}}
    if req.fourstep_steps:
        res["fourstep_steps"] = req.fourstep_steps
        res["fourstep"] = encode_array(np.stack([per[i]["fourstep"] for i in range(n)]), enc)
        res["timing"]["fourstep_s"] = [float(per[i]["fourstep_s"]) for i in range(n)]
    if req.onepass:
        res["onepass_div"] = [float(x) for x in onepass["divs"]]
        res["timing"]["onepass_s_total"] = float(onepass["onepass_s"])
        res["timing"]["onepass_s_per_point"] = float(onepass["onepass_s"]) / n
    job["result"] = res
    job["status"] = "done"


def _guard(app, job, fn):
    try:
        fn(app, job)
    except Exception as exc:                          # noqa: BLE001  a job must never stay "running"
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"


@router.post("/staged-trace", response_model=StagedJobStart)
async def staged_trace(req: StagedTraceRequest, request: Request):
    """One chord: per point the S-step run with x̂0 readouts after every step, the final image, the
    short-schedule arm and the 1-pass readout, with wall-clock per component. Poll the GET."""
    k = len(req.prompts)
    _check_points(req.weights, k, "weights")
    job = _new_job(request.app, "trace", len(req.weights), req)
    asyncio.create_task(asyncio.to_thread(_guard, request.app, job, run_trace))
    return StagedJobStart(job_id=job["id"], status="running", total=len(req.weights))


@router.get("/staged-trace/{job_id}", response_model=StagedJobStatus)
async def staged_trace_status(job_id: str, request: Request):
    job = _jobs(request.app).get(job_id)
    if job is None or job["kind"] != "trace":
        return StagedJobStatus(job_id=job_id, status="unknown", error="no such trace job")
    return _status(job)


# ---------------------------------------------------------------------------- parity

def run_parity(app, job):
    req = job["req"]
    pool = app.state.gpu_pool
    ctl = types.SimpleNamespace(status="running", notes=job["notes"])
    pts = [[float(v) for v in w] for w in req.points]
    per = {}

    def make(tid, shard):
        return StagedParityTask(job_id=tid, basis=list(req.prompts), points=shard, seed=req.seed,
                                height=req.height, width=req.width, steps=req.steps, t=req.t,
                                guidance_scale=req.guidance_scale, pipe_check=req.pipe_check)

    def on_result(r):
        per[r.index] = dict(r.data, index=int(r.index), weights=pts[r.index])
        job["done"] = len(per)

    st._dispatch(app, pool, ctl, list(enumerate(pts)), make, on_result, "parity", job["id"])
    missing = [i for i in range(len(pts)) if i not in per]
    if missing or ctl.status != "running":
        job["status"] = "error"
        job["error"] = f"incomplete: missing {missing[:12]}" if missing else f"stopped ({ctl.status})"
        return
    rows = [per[i] for i in range(len(pts))]
    res = {"t": req.t, "steps": req.steps, "points": rows,
           "all_bit_identical": all(r.get("latent_bit_identical") for r in rows),
           "min_cos_resumed_vs_direct": min(float(r["cos_resumed_vs_direct"]) for r in rows)}
    if req.pipe_check:
        res["min_cos_resumed_vs_pipe"] = min(float(r["cos_resumed_vs_pipe"]) for r in rows)
        res["all_bit_identical_pipe"] = all(r.get("max_abs_latent_diff_pipe") == 0.0 for r in rows)
    job["result"] = res
    job["status"] = "done"


@router.post("/staged-parity", response_model=StagedJobStart)
async def staged_parity(req: StagedParityRequest, request: Request):
    """V1: readout to t -> cached latent -> resume to S vs the direct S-step run, per point."""
    _check_points(req.points, len(req.prompts), "points")
    job = _new_job(request.app, "parity", len(req.points), req)
    asyncio.create_task(asyncio.to_thread(_guard, request.app, job, run_parity))
    return StagedJobStart(job_id=job["id"], status="running", total=len(req.points))


@router.get("/staged-parity/{job_id}", response_model=StagedJobStatus)
async def staged_parity_status(job_id: str, request: Request):
    job = _jobs(request.app).get(job_id)
    if job is None or job["kind"] != "parity":
        return StagedJobStatus(job_id=job_id, status="unknown", error="no such parity job")
    return _status(job)


@router.get("/staged-cache")
async def staged_cache(request: Request):
    """The process-wide latent LRU: entries, bytes against the cap, hits, misses, evictions."""
    return st.get_cache(request.app).stats()
