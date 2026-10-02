"""AMR endpoints: start / status / cancel / images / points.json.

Mirrors the Cascade run pattern (routers/cascade.py): a plain sync worker on a thread via
asyncio.to_thread, cooperative cancel through run.status, state in its own never-evicted
app.state dict, images served by index from the content-hash thumbnail store, and every
worker-side state change inside one try/except so a failure cannot leave a run stuck
"running".

The one thing this router does that the Cascade's does not is REFUSE work. Adaptive
refinement is a k=4 tool (search_problem h25f): the fan-out is factor^(k-1), so k=5 already
misses the budget of record by ~8x and k=9 by five orders of magnitude. A request past
`amr.K_MAX`, or one whose finest lattice is larger than `amr.MAX_CELLS`, is answered with
the projected probe count rather than being queued -- the GPU pool is shared, and a run
nobody can afford is better refused with a number than accepted with a shrug.
"""
import asyncio
import math
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from backend.models import (AmrStartRequest, AmrStartResponse, AmrStatus, AmrLevelStat,
                            AmrExport, AmrPointExport, AmrEdgeExport)
from backend.services import amr
from backend.cache.thumbnail_cache import ThumbnailStore

router = APIRouter(prefix="/api/amr", tags=["amr"])


def _runs(app):
    app.state.amr_runs = getattr(app.state, "amr_runs", {})
    return app.state.amr_runs


def _refuse_k(k, req):
    """The 400 body for an unsupported k: the projected cost of the ladder that was asked
    for, plus the study's published figures for its own ladder, so the number is traceable."""
    probes, rows = amr.projected_probes(k, req.base, req.levels, req.factor)
    ladder = " -> ".join(str(r["level"]) for r in rows)
    return (
        f"adaptive refinement is supported at k <= {amr.K_MAX} only (you asked for k={k}). "
        f"The refinement fans out as factor^(k-1), so the ladder {ladder} projects to "
        f"{probes} probes = {probes * amr.probe_cost(req.probe_steps):.0f} image-eq at k={k}, "
        f"from the boundary fractions measured at k=4 in search_problem h25f. The study's own "
        f"5 -> 15 -> 30 -> 60 ladder projects {amr.H25F_K5_PROBES} probes at k=5 and "
        f"{amr.H25F_K9_PROBES} at k=9, against a {amr.H25F_BAR_IMAGE_EQ} image-eq bar. "
        f"Use the Cascade's chords at k >= {amr.K_MAX + 1}: they remain the only affordable "
        f"sampler there.")


@router.post("/start", response_model=AmrStartResponse)
async def start(req: AmrStartRequest, request: Request):
    app = request.app
    k = len(req.prompts)
    if k > amr.K_MAX:
        raise HTTPException(status_code=400, detail=_refuse_k(k, req))
    sched = amr.nested_schedule(req.base, req.levels, req.factor)
    # counted, never enumerated: base 10 / factor 3 / levels 5 is a legal request whose
    # finest lattice has 89 million cells, and building it to measure it is the one way to
    # take the server down with a 400-shaped mistake
    finest = math.comb(sched[-1] + k - 1, k - 1)
    if finest > amr.MAX_CELLS:
        probes, _rows = amr.projected_probes(k, req.base, req.levels, req.factor)
        raise HTTPException(
            status_code=400,
            detail=(f"the finest level of {' -> '.join(str(n) for n in sched)} has "
                    f"{finest} cells, past the {amr.MAX_CELLS} ceiling (k=4 at 60 points "
                    f"per edge, the finest ladder h25f validated); it would project "
                    f"{probes} probes. Lower `levels`, `base`, or `factor`."))
    rid = uuid.uuid4().hex[:8]
    run = amr.AmrRun(
        run_id=rid, prompts=list(req.prompts), seed=req.seed, steps=req.steps,
        height=req.height, width=req.width, guidance_scale=req.guidance_scale,
        base=req.base, levels=req.levels, factor=req.factor, r_ref=req.r_ref,
        probe_steps=req.probe_steps)
    # the ladder is known before any GPU work, and the map draws it as soon as the run
    # exists, so it is filled in here rather than at the top of run_amr
    run.schedule = list(sched)
    run.thumbs = ThumbnailStore(app.state.cache)
    _runs(app)[rid] = run
    asyncio.create_task(asyncio.to_thread(_run, app, rid))
    return AmrStartResponse(run_id=rid, status="running")


def _run(app, rid):
    run = None
    try:
        run = _runs(app)[rid]
        amr.run_amr(app, run, app.state.gpu_pool)
    except Exception as exc:                         # noqa: BLE001
        if run is not None:
            run.status = "error"
            run.error = f"{type(exc).__name__}: {exc}"


@router.get("/{run_id}/status", response_model=AmrStatus)
async def status(run_id: str, request: Request):
    run = _runs(request.app).get(run_id)
    if run is None:
        return AmrStatus(run_id=run_id, status="unknown", error="no such run")
    return AmrStatus(
        run_id=run.run_id, status=run.status, phase=run.phase, k=run.k,
        prompts=list(run.prompts), schedule=list(run.schedule),
        base=run.base, levels=run.levels, factor=run.factor, r_ref=run.r_ref,
        probe_steps=run.probe_steps, generated=run.generated,
        phase_done=run.phase_done, phase_total=run.phase_total,
        recent_thumbs=list(getattr(run, "recent_thumbs", [])),
        points=[p["weights"] for p in run.points],
        point_levels=[p["level"] for p in run.points],
        point_images=[p["image"] for p in run.points],
        point_divs=[p["div"] for p in run.points],
        edges=[[e["a"], e["b"]] for e in run.edges],
        edge_levels=[e["level"] for e in run.edges],
        edge_divs=[e["divergence"] for e in run.edges],
        levels_stats=[AmrLevelStat(**r) for r in run.level_stats],
        cost_image_eq=run.cost_image_eq,
        notes=list(run.notes), error=run.error)


@router.post("/{run_id}/cancel")
async def cancel(run_id: str, request: Request):
    run = _runs(request.app).get(run_id)
    if run is not None and run.status == "running":
        run.status = "cancelled"
    return {"ok": True}


@router.get("/{run_id}/points.json", response_model=AmrExport)
async def points_export(run_id: str, request: Request):
    """The survey as records, for reuse outside the map (404 for an unknown run)."""
    run = _runs(request.app).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="no such run")
    return AmrExport(
        run_id=run.run_id, status=run.status, k=run.k, prompts=list(run.prompts),
        schedule=list(run.schedule), r_ref=run.r_ref, seed=run.seed, steps=run.steps,
        probe_steps=run.probe_steps,
        points=[AmrPointExport(id=i, weights=p["weights"], level=p["level"],
                               image=p["image"], div=p["div"])
                for i, p in enumerate(run.points)],
        edges=[AmrEdgeExport(**e) for e in run.edges],
        levels_stats=[AmrLevelStat(**r) for r in run.level_stats],
        cost_image_eq=run.cost_image_eq, notes=list(run.notes))


@router.get("/{run_id}/image/{index}")
async def image(run_id: str, index: int, request: Request):
    """One probe's thumbnail, by the image index the points carry (the Cascade's store)."""
    run = _runs(request.app).get(run_id)
    if run is None:
        return Response(status_code=404)
    b = run.thumbs.get(index) if run.thumbs is not None else None
    if b is None:
        return Response(status_code=404)
    return Response(content=b, media_type="image/jpeg")
