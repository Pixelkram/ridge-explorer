"""Cascade endpoints: start / status / cancel / images.

Mirrors the Discover run pattern (routers/discover.py): a plain sync worker on a
thread via asyncio.to_thread, cooperative cancel through run.status, state in its own
never-evicted app.state dict, images served by index from the content-hash thumbnail
store. All worker-side state changes stay inside one try/except so a failure can never
leave a run stuck "running" (the discover incident).
"""
import asyncio
import uuid

import numpy as np

from fastapi import APIRouter, Request
from fastapi.responses import Response

from backend import config
from backend.models import (CascadeStartRequest, CascadeStartResponse, CascadeStatus,
                            CascadeCrossing, CascadePatch, CascadeChord,
                            WalkStartRequest, WalkStep, WalkStatus,
                            CascadePointInfo)
from backend.services import cascade as cs
from backend.cache.thumbnail_cache import ThumbnailStore

router = APIRouter(prefix="/api/cascade", tags=["cascade"])


def _runs(app):
    app.state.cascades = getattr(app.state, "cascades", {})
    return app.state.cascades


@router.post("/start", response_model=CascadeStartResponse)
async def start(req: CascadeStartRequest, request: Request):
    app = request.app
    if req.prompts is not None and len(req.prompts) != req.k:
        return CascadeStartResponse(
            run_id="", status="error",
            error=f"prompts has {len(req.prompts)} entries but k={req.k}")
    if req.focus is not None:
        if req.prompts is None:
            return CascadeStartResponse(
                run_id="", status="error",
                error="focus requires pinned prompts (the weights need a known basis)")
        if len(req.focus) != req.k:
            return CascadeStartResponse(
                run_id="", status="error",
                error=f"focus has {len(req.focus)} weights but k={req.k}")
    rid = uuid.uuid4().hex[:8]
    run = cs.CascadeRun(
        run_id=rid, prompts=list(req.prompts) if req.prompts else [],
        seed=req.seed, steps=req.steps, height=req.height, width=req.width,
        guidance_scale=req.guidance_scale, n_chords=req.n_chords,
        n_patches=req.n_patches, focus=req.focus, focus_radius=req.focus_radius,
        probe_steps=req.probe_steps)
    run.thumbs = ThumbnailStore(app.state.cache)
    run._target_sim = req.target_sim
    run._k_req = req.k
    _runs(app)[rid] = run
    asyncio.create_task(asyncio.to_thread(_run, app, rid))
    return CascadeStartResponse(run_id=rid, status="running")


def _run(app, rid):
    run = None
    try:
        run = _runs(app)[rid]
        if not run.prompts:
            # Draw prompts in the discover band (E10); lazy import avoids the
            # startup race on the shared CLIP text encoder.
            import random as _random
            from backend.routers.grid import _clip_text_encoder
            from backend.services.discover import choose_prompts
            text_enc = _clip_text_encoder(app)
            prompts, _ms = choose_prompts(
                config.HIKE_PROMPT_POOL, run._k_req, run._target_sim,
                _random.Random(run.seed), text_enc)
            run.prompts = prompts
        cs.run_cascade(app, run, app.state.gpu_pool)
    except Exception as exc:                     # noqa: BLE001
        if run is not None:
            run.status = "error"
            run.error = f"{type(exc).__name__}: {exc}"


@router.get("/{run_id}/status", response_model=CascadeStatus)
async def status(run_id: str, request: Request):
    run = _runs(request.app).get(run_id)
    if run is None:
        return CascadeStatus(run_id=run_id, status="unknown", error="no such run")
    cov = cs.coverage_stats(run.crossings)
    return CascadeStatus(
        run_id=run.run_id, status=run.status, phase=run.phase,
        generated=run.generated, phase_done=run.phase_done,
        phase_total=run.phase_total, prompts=run.prompts,
        recent_thumbs=list(getattr(run, "recent_thumbs", [])),
        points=list(getattr(run, "probe_geo", [])),
        point_divs=list(getattr(run, "probe_div", [])),
        chords=[CascadeChord(a=c[0], b=c[1])
                for c in getattr(run, "chords_geo", [])],
        crossings=[CascadeCrossing(
            cid=x.cid, weights=[float(v) for v in x.mid] if x.mid is not None
            else [float(v) for v in (x.wa + x.wb) / 2],
            b=x.b, significant=x.significant, thumb=x.thumb,
            ridge_group=(cov[3] or {}).get(x.cid),
            bracket_w=float(np.linalg.norm(x.wa - x.wb))
            if x.wa is not None and x.wb is not None else None)
            for x in run.crossings],
        bg_mean=run.bg_mean, bg_p95=run.bg_p95,
        distinct_ridges=cov[0], singleton_ridges=cov[1],
        unexplored_share=cov[2],
        patches=[CascadePatch(
            region=p.region, cid=p.cid, b=p.b, significant=p.significant,
            exploration=p.exploration, grid=p.grid,
            cols_with_crossing=p.cols_with_crossing) for p in run.patches],
        notes=list(run.notes), error=run.error)


@router.post("/{run_id}/walk", response_model=WalkStatus)
async def walk_start(run_id: str, req: WalkStartRequest, request: Request):
    app = request.app
    run = _runs(app).get(run_id)
    if run is None:
        return WalkStatus(walk_id="", status="error", error="no such run")
    if not any(x.cid == req.cid and x.mid is not None for x in run.crossings):
        return WalkStatus(walk_id="", status="error",
                          error="crossing not found (or run still bisecting)")
    wid = uuid.uuid4().hex[:8]
    walk = cs.Walk(wid=wid, cid=req.cid, direction=1 if req.direction >= 0 else -1,
                   n_steps=req.n_steps)
    run.walks = getattr(run, "walks", {})
    run.walks[wid] = walk

    def _go():
        try:
            cs.run_walk(app, run, app.state.gpu_pool, walk)
        except Exception as exc:                 # noqa: BLE001
            walk.status = "error"
            walk.error = f"{type(exc).__name__}: {exc}"

    asyncio.create_task(asyncio.to_thread(_go))
    return WalkStatus(walk_id=wid, status="running", cid=req.cid)


@router.get("/{run_id}/walk/{walk_id}", response_model=WalkStatus)
async def walk_status(run_id: str, walk_id: str, request: Request):
    run = _runs(request.app).get(run_id)
    walk = getattr(run, "walks", {}).get(walk_id) if run is not None else None
    if walk is None:
        return WalkStatus(walk_id=walk_id, status="unknown", error="no such walk")
    return WalkStatus(
        walk_id=walk.wid, status=walk.status, cid=walk.cid,
        steps=[WalkStep(**s) for s in walk.steps],
        segs=list(walk.segs),
        notes=list(walk.notes), error=walk.error)


@router.post("/{run_id}/walk/{walk_id}/cancel")
async def walk_cancel(run_id: str, walk_id: str, request: Request):
    run = _runs(request.app).get(run_id)
    walk = getattr(run, "walks", {}).get(walk_id) if run is not None else None
    if walk is not None and walk.status == "running":
        walk.status = "cancelled"
    return {"ok": True}


@router.post("/{run_id}/cancel")
async def cancel(run_id: str, request: Request):
    run = _runs(request.app).get(run_id)
    if run is not None and run.status == "running":
        run.status = "cancelled"
    return {"ok": True}


@router.get("/{run_id}/point/{index}", response_model=CascadePointInfo)
async def point_info(run_id: str, index: int, request: Request):
    """The recipe (weight vector) behind any generated thumbnail index."""
    run = _runs(request.app).get(run_id)
    if run is None:
        return CascadePointInfo()
    pos = getattr(run, "_geo_pos", {}).get(index)
    if pos is None:
        return CascadePointInfo()
    return CascadePointInfo(
        weights=[float(v) for v in run.probe_geo[pos]],
        div=run.probe_div[pos])


@router.get("/{run_id}/{index}.jpg")
async def image(run_id: str, index: int, request: Request):
    run = _runs(request.app).get(run_id)
    if run is None:
        return Response(status_code=404)
    b = run.thumbs.get(index)
    if b is None:
        return Response(status_code=404)
    return Response(content=b, media_type="image/jpeg")
