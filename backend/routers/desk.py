"""Mixing desk endpoints: start / move / status / cancel / image.

The Cascade run pattern (routers/metro.py): a plain sync worker on a thread via
asyncio.to_thread, state in its own never-evicted app.state dict, images by index from the
content-hash thumbnail store, every worker-side change inside one try/except. The unit of work
is a POSITION -- one current mix and its k lines -- and a newer move cancels a position still
computing, so dragging the markers around never queues more than one batch on the pool.

  * POST /start {prompts, w0, dalpha, refine, ...} -- the desk and its first position.
  * POST /{id}/move {w0} or {line, alpha} -- a new current mix: either outright, or the point at
    `alpha` on prompt `line`'s slider through the current mix (a released marker). Lines already
    measured come from the cache (a move along line i keeps line i whole), and a mix visited
    before is served as it stands, at no cost.

Both are refused past a ceiling with the projection, as Metro refuses: one request past
`desk.MAX_REQUEST_IMAGE_EQ`, or a desk past `desk.MAX_SESSION_IMAGE_EQ` in total.
"""
import asyncio
import uuid

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from backend.models import (DeskStartRequest, DeskMoveRequest, DeskStartResponse, DeskStatus,
                            DeskLine, DeskFlip, DeskSegment, DeskReadout, DeskFlipRef,
                            DeskPositionRef)
from backend.services import desk as dk
from backend.services import gridfree as gf
from backend.services.cascade import COS_T
from backend.cache.thumbnail_cache import ThumbnailStore

router = APIRouter(prefix="/api/desk", tags=["desk"])


def _runs(app):
    app.state.desk_runs = getattr(app.state, "desk_runs", {})
    return app.state.desk_runs


def _recipe(w, k):
    r, why = gf.legal_recipe(w, k)
    if r is None:
        raise HTTPException(status_code=400, detail=f"w0 is not a legal recipe: {why}")
    return r


def _refuse(total, rows, what, spent=None):
    parts = "; ".join(f"{r['what']} = {r['image_eq']:.1f} image-eq" for r in rows)
    if spent is None:
        return (f"{what} would cost up to {total:.1f} image-eq, past the "
                f"{dk.MAX_REQUEST_IMAGE_EQ} ceiling of one request: {parts}. A line is "
                f"1/dalpha + 1 cheap probes, so raise dalpha, turn refine off, or use fewer "
                f"prompts.")
    return (f"{what} would cost up to {total:.1f} image-eq on top of the {spent:.1f} this desk "
            f"has spent, past its {dk.MAX_SESSION_IMAGE_EQ} ceiling: {parts}. Start a new "
            f"desk.")


def _go(app, rid, pid):
    pos = None
    try:
        run = _runs(app)[rid]
        pos = run.positions[pid]
        dk.run_position(app, run, app.state.gpu_pool, pos)
    except Exception as exc:                         # noqa: BLE001
        if pos is not None:
            pos.status = "error"
            pos.error = f"{type(exc).__name__}: {exc}"


@router.post("/start", response_model=DeskStartResponse)
async def start(req: DeskStartRequest, request: Request):
    app = request.app
    k = len(req.prompts)
    w0 = np.full(k, 1.0 / k) if req.w0 is None else _recipe(req.w0, k)
    n = len(dk.alpha_grid(req.dalpha))
    total, rows = dk.projected_cost(k, n, req.refine, req.probe_steps)
    if total > dk.MAX_REQUEST_IMAGE_EQ:
        raise HTTPException(status_code=400, detail=_refuse(total, rows, "this desk"))
    rid = uuid.uuid4().hex[:8]
    run = dk.DeskRun(run_id=rid, prompts=list(req.prompts), seed=req.seed, steps=req.steps,
                     height=req.height, width=req.width, guidance_scale=req.guidance_scale,
                     probe_steps=req.probe_steps, dalpha=req.dalpha, refine=req.refine)
    run.thumbs = ThumbnailStore(app.state.cache)
    pos, _ = dk.add_position(run, w0, req.dalpha, req.refine)
    _runs(app)[rid] = run
    asyncio.create_task(asyncio.to_thread(_go, app, rid, pos.pid))
    return DeskStartResponse(desk_id=rid, status="running", position=pos.pid,
                             cost_image_eq=float(total), cached_lines=0)


@router.post("/{desk_id}/move", response_model=DeskStartResponse)
async def move(desk_id: str, req: DeskMoveRequest, request: Request):
    """A new current mix. Computed on release, never on drag: the panel posts once."""
    app = request.app
    run = _runs(app).get(desk_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no desk {desk_id}")
    k = run.k
    if req.w0 is not None:
        w0 = _recipe(req.w0, k)
    elif req.line is not None and req.alpha is not None:
        if req.line >= k:
            raise HTTPException(status_code=400,
                                detail=f"line {req.line} does not exist: this desk has k={k}")
        cur = run.positions[run.current]
        w0 = gf.legal_recipe(dk.line_point(cur.w0, req.line, req.alpha), k)[0]
    else:
        raise HTTPException(status_code=400,
                            detail="give w0, or line and alpha (a marker released on a slider)")
    dalpha = run.dalpha if req.dalpha is None else float(req.dalpha)
    refine = run.refine if req.refine is None else bool(req.refine)
    key = dk.position_key(w0, dalpha, refine)
    old = run.positions[run.position_index[key]] if key in run.position_index else None
    if old is not None and old.status == "complete":
        for p in run.positions:                       # a revisit needs nothing in flight
            if p.status in ("pending", "running"):
                p.status = "cancelled"
        dk.add_position(run, w0, dalpha, refine)
        return DeskStartResponse(desk_id=desk_id, status="complete", position=old.pid,
                                 cost_image_eq=0.0, cached_lines=k)
    n_cached = run.cached_lines(w0, dalpha, refine)
    total, rows = dk.projected_cost(k, len(dk.alpha_grid(dalpha)), refine, run.probe_steps,
                                    n_cached_lines=n_cached, w0_cached=run.w0_cached(w0))
    if total > dk.MAX_REQUEST_IMAGE_EQ:
        raise HTTPException(status_code=400, detail=_refuse(total, rows, "this move"))
    if run.cost_image_eq + total > dk.MAX_SESSION_IMAGE_EQ:
        raise HTTPException(status_code=400,
                            detail=_refuse(total, rows, "this move", run.cost_image_eq))
    for p in run.positions:
        if p.status in ("pending", "running"):
            p.status = "cancelled"                    # the newest release wins the pool
    run.dalpha, run.refine = dalpha, refine
    pos, _ = dk.add_position(run, w0, dalpha, refine)
    asyncio.create_task(asyncio.to_thread(_go, app, desk_id, pos.pid))
    return DeskStartResponse(desk_id=desk_id, status="running", position=pos.pid,
                             cost_image_eq=float(total), cached_lines=n_cached)


def _line(run, pos, ln, cached):
    return DeskLine(
        i=ln["i"], prompt=run.prompts[ln["i"]], alpha0=float(pos.w0[ln["i"]]),
        degenerate=bool(ln["degenerate"]), cached=bool(cached), status=ln["status"],
        alphas=list(ln["alphas"]), images=list(ln["images"]), divs=list(ln["divs"]),
        flips=[DeskFlip(**f) for f in ln["flips"]],
        segments=[DeskSegment(**s) for s in ln["segments"]])


@router.get("/{desk_id}/status", response_model=DeskStatus)
async def status(desk_id: str, request: Request):
    """The desk at its current mix: the k lines, the readout, and the mixes visited."""
    run = _runs(request.app).get(desk_id)
    if run is None:
        return DeskStatus(desk_id=desk_id, status="unknown", error="no such desk")
    pos = run.positions[run.current] if 0 <= run.current < len(run.positions) else None
    lines = []
    if pos is not None:
        lines = [_line(run, pos, ln, c) for ln, c in zip(pos.lines, pos.cached)]
    readout = [DeskReadout(i=r["i"], prompt=run.prompts[r["i"]],
                           up=DeskFlipRef(**r["up"]) if r["up"] else None,
                           down=DeskFlipRef(**r["down"]) if r["down"] else None,
                           nearest=r["nearest"]) for r in (pos.readout if pos else [])]
    return DeskStatus(
        desk_id=run.run_id, status=run.status, k=run.k, prompts=list(run.prompts),
        seed=run.seed, steps=run.steps, probe_steps=run.probe_steps,
        position=pos.pid if pos else -1, w0=list(pos.w0) if pos else [],
        w0_image=pos.w0_image if pos else -1, dalpha=pos.dalpha if pos else run.dalpha,
        refine=pos.refine if pos else run.refine, lines=lines, readout=readout,
        readout_label=dk.READOUT_LABEL,
        positions=[DeskPositionRef(pid=p.pid, w0=list(p.w0), status=p.status,
                                   cost_image_eq=float(p.cost)) for p in run.positions],
        cost_image_eq=run.cost_image_eq, position_cost=float(pos.cost) if pos else 0.0,
        max_request_image_eq=float(dk.MAX_REQUEST_IMAGE_EQ),
        max_session_image_eq=float(dk.MAX_SESSION_IMAGE_EQ), cos_t=COS_T,
        generated=run.generated,
        notes=list(run.notes) + (list(pos.notes) if pos else []),
        error=pos.error if pos else None)


@router.post("/{desk_id}/cancel")
async def cancel(desk_id: str, request: Request):
    run = _runs(request.app).get(desk_id)
    if run is not None:
        for p in run.positions:
            if p.status in ("pending", "running"):
                p.status = "cancelled"
    return {"ok": True}


@router.get("/{desk_id}/image/{index}")
async def image(desk_id: str, index: int, request: Request):
    run = _runs(request.app).get(desk_id)
    if run is None:
        return Response(status_code=404)
    b = run.thumbs.get(index) if run.thumbs is not None else None
    if b is None:
        return Response(status_code=404)
    return Response(content=b, media_type="image/jpeg")
