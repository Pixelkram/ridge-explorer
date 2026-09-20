"""Discovery endpoints: the measured procedure, exposed as a background job.

Mirrors the hike endpoints (start / status / cancel / image) so the frontend polling
pattern is unchanged. The work itself is deliberately simple -- draw prompts, draw weights,
generate, restart -- because that is all the ablations in services/discover.py justify.
"""

import asyncio
import io
import random
import uuid

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import Response

from backend import config
from backend.models import (DiscoverStartRequest, DiscoverStartResponse, DiscoverStatus,
                            DiscoverDefaults)
from backend.services import discover as dc
from backend.services.gpu_pool import DiscoverTask

router = APIRouter(prefix="/api/discover")


def _runs(app):
    app.state.discover = getattr(app.state, "discover", {})
    return app.state.discover


@router.get("/defaults", response_model=DiscoverDefaults)
async def defaults():
    """The measured defaults and the evidence behind each, so the UI can show WHY."""
    return {
        "k": dc.DEFAULT_K,
        "commitment": dc.COMMITMENT,
        "target_sim": dc.TARGET_SIM,
        "batch": dc.BATCH,
        "steps": config.DEFAULT_NUM_INFERENCE_STEPS,
        "provenance": dc.DEFAULTS_PROVENANCE,
        "alpha_table": {
            str(k): {f"{c:.2f}": round(dc.solve_alpha(k, c), 4)
                     for c in (0.45, 0.50, 0.55, 0.60, 0.65) if c > 1.0 / k}
            for k in (3, 4, 5, 6, 8)
        },
    }


@router.post("/start", response_model=DiscoverStartResponse)
async def start(req: DiscoverStartRequest, request: Request):
    if req.commitment <= 1.0 / req.k:
        return DiscoverStartResponse(
            run_id="", status="error", total=0,
            error=(f"commitment {req.commitment:.2f} is at or below the k={req.k} floor "
                   f"of {1.0/req.k:.3f} — a {req.k}-prompt mixture cannot be that balanced"))
    pool = list(req.prompt_pool) if req.prompt_pool else list(config.HIKE_PROMPT_POOL)
    if len(pool) < req.k:
        return DiscoverStartResponse(run_id="", status="error", total=0,
                                     error=f"pool has {len(pool)} prompts, need >= {req.k}")
    from backend.cache.thumbnail_cache import ThumbnailStore
    rid = uuid.uuid4().hex[:8]
    run = dc.DiscoverRun(
        run_id=rid, k=req.k, commitment=req.commitment, target_sim=req.target_sim,
        batch=req.batch, total=req.total, seed=req.seed, steps=req.steps,
        height=req.height, width=req.width, guidance_scale=req.guidance_scale, pool=pool,
        commit_band=tuple(req.commit_band) if req.commit_band else None,
        min_weight=req.min_weight, prompts_override=req.prompts,
        scout_gs=req.scout_gs, top_frac=req.top_frac, aim_jitter=req.aim_jitter,
        extrapolate=req.extrapolate)
    # Same store the grid jobs use: only the hash stays resident, bytes live on disk.
    run.thumbs = ThumbnailStore(request.app.state.cache)
    _runs(request.app)[rid] = run
    asyncio.create_task(asyncio.to_thread(_run, request.app, rid))
    return DiscoverStartResponse(run_id=rid, status="running", total=req.total)


@router.get("/{run_id}/status", response_model=DiscoverStatus)
async def status(run_id: str, request: Request):
    run = _runs(request.app).get(run_id)
    if run is None:
        return DiscoverStatus(run_id=run_id, status="unknown", generated=0, total=0,
                              simplices=[], error="no such run")
    per = run.per_simplex_diversity()
    return DiscoverStatus(
        run_id=run.run_id, status=run.status, generated=run.generated, total=run.total,
        k=run.k, commitment=run.commitment, diversity=_nan_to_none(run.diversity()),
        redundancy=_nan_to_none(run.redundancy()),
        error=run.error, notes=run.notes,
        simplices=[{
            "index": s.index, "prompts": s.prompts, "mean_sim": round(s.mean_sim, 3),
            "blend_frac": _nan_to_none(s.blend_frac),
            "alpha": round(s.alpha, 4), "n_points": s.n_points, "done": s.done,
            "diversity": _nan_to_none(per[i] if i < len(per) else float("nan")),
            "scout_n": s.scout_n, "aim_lo": s.aim_lo, "aim_hi": s.aim_hi,
            "draws_per_cell": _nan_to_none(s.draws_per_cell),
            "survey_ratio": _nan_to_none(s.survey_ratio),
        } for i, s in enumerate(run.simplices)])


@router.post("/{run_id}/cancel")
async def cancel(run_id: str, request: Request):
    run = _runs(request.app).get(run_id)
    if run is None:
        return {"status": "unknown"}
    run.status = "cancelled"
    return {"status": "cancelled"}


@router.get("/{run_id}/{index}.jpg")
async def image(run_id: str, index: int, request: Request):
    run = _runs(request.app).get(run_id)
    if run is None or run.thumbs is None:
        return Response(status_code=404)
    b = run.thumbs.get(index)
    if b is None:
        return Response(status_code=404)
    return Response(content=b, media_type="image/jpeg")


def _nan_to_none(x):
    return None if x is None or (isinstance(x, float) and x != x) else round(float(x), 4)


STALL_TIMEOUT = 15 * 60      # matches the hike drain: give up on a shard that goes quiet


def _submit_and_drain(app, run, s, points, pool_gpu, tids, label=""):
    """Generate `points` for simplex `s`, in small chunks, draining as results land.

    Chunked because there is no per-job cancel: workers only honour the pool-wide
    cancel_epoch, and bumping that would abort every other job in the pool. Keeping little
    queued is the only way to make a per-run stop responsive -- submitting a whole
    300-image simplex up front left that much still generating after "stop", and each new
    run queued behind it, so stop-then-start looked dead for minutes.
    """
    import time as _t
    ep = pool_gpu.cancel_epoch.value
    n_gpus = max(1, min(getattr(pool_gpu, "n_gpus", 1), len(points)))
    CHUNK = 8
    step = n_gpus * CHUNK
    for off in range(0, len(points), step):
        if run.status != "running":
            return
        block = points[off:off + step]
        shards = []
        for g in range(n_gpus):
            shard = block[g::n_gpus]
            if not shard:
                continue
            tid = f"discover:{run.run_id}:{s.index}:{label}{off}:{g}"
            app.state.hike_inbox.setdefault(tid, [])
            tids.append(tid)
            shards.append((tid, len(shard)))
            pool_gpu.submit(DiscoverTask(
                job_id=tid, basis=s.prompts, points=shard,
                seed=run.seed + s.index, height=run.height, width=run.width,
                steps=run.steps, guidance_scale=run.guidance_scale))
        want = sum(n for _, n in shards)
        got, last = 0, _t.time()
        while got < want and run.status == "running":
            moved = False
            for tid, _n in shards:
                ib = app.state.hike_inbox.get(tid) or []
                while ib:
                    r = ib.pop(0)
                    run.thumbs[r.row] = r.thumbnail_bytes
                    run.embeddings[r.row] = np.asarray(r.dino_embedding, dtype=np.float64)
                    run.generated += 1
                    s.done += 1
                    got += 1
                    moved = True
            if moved:
                last = _t.time()
                continue
            if pool_gpu.cancel_epoch.value != ep:
                run.notes.append("cancelled by a pool-wide cancel")
                run.status = "cancelled"
                break
            errs = pool_gpu.pop_task_errors()
            if errs:
                run.notes.append(f"worker error: {errs[-1]}")
                break
            if _t.time() - last > STALL_TIMEOUT:
                run.notes.append(f"simplex {s.index} stalled at {got}/{want}; moving on")
                break
            _t.sleep(0.25)
        for tid, _n in shards:
            app.state.hike_inbox.pop(tid, None)


def _run(app, run_id):
    """Draw prompts, sample, generate, restart. One batch per simplex, in order.

    Restarting is the whole mechanism: a fixed simplex exhausts (diversity -0.040 over 960
    images), and any restart fixes it (0.742 -> 0.913). So the loop does not try to be
    clever about where it goes next -- E35 showed a random restart matched a chained one.

    Results are taken from `app.state.hike_inbox`, NOT from pool.collect_results(). The
    result queue is shared by every job and is drained by the single collector in main.py,
    which routes each result to the job that owns it. A second drainer racing that
    collector wins some results and, because it cannot use another job's cells, drops them
    -- silently corrupting whatever grid or hike happened to be running alongside. The
    inbox is the supported route for jobs that are not grid entries.
    """
    import time as _t
    run = _runs(app)[run_id]
    tids = []
    try:
        # Everything that can raise lives inside the guard, including the lookups: an
        # AttributeError here previously escaped the try and left the run reporting
        # "running" forever with no error and no work, which is the worst failure mode
        # for a polled job.
        from backend.routers.grid import _clip_text_encoder
        pool_gpu = app.state.gpu_pool
        app.state.hike_inbox = getattr(app.state, "hike_inbox", {})
        text_enc = _clip_text_encoder(app)
        rng = random.Random(run.seed)
        base_index = 0

        while run.generated < run.total and run.status == "running":
            s = dc.plan_simplex(run, rng, text_enc)
            run.notes.append(
                f"simplex {s.index}: k={run.k} sim={s.mean_sim:.3f} alpha={s.alpha:.3f} "
                f"({s.n_points} images)")
            points = [(base_index + i, w) for i, w in enumerate(s.weights)]

            _submit_and_drain(app, run, s, points, pool_gpu, tids)
            # Advance by the FULL batch, not by what arrived: indices are handed to the
            # workers up-front, so a short leg must still not reuse them. Losing this line
            # in a refactor left base_index at 0, so the aiming phase below looked up
            # embeddings at NEGATIVE indices, found none and skipped silently -- the run
            # then scouted twice and produced 272 images for a requested 220.
            base_index += s.n_points

            # ---- ridge targeting, phase 2 ----------------------------------
            # The leg just generated was a SURVEY. Read its sensitivity field and spend the
            # rest of the leg near the steepest cells, where the model's own basins meet.
            # Measured over 10 prompt triplets, those cells gave redundancy 0.766 against
            # 0.912 for balanced random sampling of the same simplex, winning 10/10.
            if s.scout_n and not s.aimed and run.status == "running":
                emb = [run.embeddings.get(base_index - s.scout_n + m)
                       for m in range(s.scout_n)]
                have = [e for e in emb if e is not None]
                if len(have) >= 8:
                    E = np.stack([e if e is not None else np.zeros(768) for e in emb])
                    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
                    field = dc.sensitivity_field(E, s.scout_idx, run.scout_gs)
                    want = min(run.batch - s.scout_n, run.total - run.generated)
                    if field and want > 0:
                        rr = np.random.default_rng(abs(hash((run.seed, s.index, 1))) % (2**32))
                        Wa = dc.aim_points(field, s.scout_idx, run.scout_gs, want, rr,
                                           top_frac=run.top_frac,
                                           jitter_scale=run.aim_jitter,
                                           extrapolate=run.extrapolate)
                        # aim_points returns AT MOST one point per top-decile cell, so it
                        # may hand back fewer than asked. Take the length from the result,
                        # never from the request: padding the batch back up to `want` is
                        # exactly what collapsed selectivity in E30/E32 (9 draws per cell,
                        # no measurable benefit) and in the first version of this loop
                        # (6.5 draws per cell, modest). A smaller, better harvest is the
                        # point, so the run simply ends early.
                        n_aim = 0 if Wa is None else len(Wa)
                        if n_aim > 0:
                            n_top = max(1, int(round(len(field) * run.top_frac)))
                            s.aimed = True
                            s.n_points += n_aim
                            s.weights.extend([w.tolist() for w in Wa])
                            s.blend_frac = float((np.stack(
                                [np.asarray(w) for w in s.weights]).max(1) < 0.5).mean())
                            s.draws_per_cell = n_aim / n_top
                            s.survey_ratio = s.scout_n / n_aim
                            run.notes.append(
                                f"simplex {s.index}: aimed {n_aim} at the top {n_top} of "
                                f"{len(field)} cells — {s.survey_ratio:.1f}:1 survey ratio, "
                                f"{s.draws_per_cell:.2f} draws/cell")
                            s.aim_lo = base_index
                            s.aim_hi = base_index + n_aim
                            points = [(base_index + i, w) for i, w in enumerate(Wa)]
                            _submit_and_drain(app, run, s, points, pool_gpu, tids, label="aim")
                            base_index += n_aim

        if run.status == "running":
            run.status = "complete"
    except Exception as exc:                      # a failed run must not wedge the pool
        run.status = "error"
        run.error = f"{type(exc).__name__}: {exc}"
    finally:
        inbox = getattr(app.state, "hike_inbox", None)
        if inbox is not None:
            for tid in tids:
                inbox.pop(tid, None)
