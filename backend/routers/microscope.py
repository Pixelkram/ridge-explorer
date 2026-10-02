"""Ridge microscope endpoints: plan / start / status / zoom / cancel / image.

The Cascade run pattern again (routers/metro.py): a plain sync worker on a thread via
asyncio.to_thread, state in its own never-evicted app.state dict, images served by index from
the content-hash thumbnail store, and every worker-side state change inside one try/except so a
failure cannot leave a level stuck "running". The unit of work here is a LEVEL (one lattice of
the zoom stack), so cancel stops the levels in flight and a later zoom still works.

  * POST /plan   -- the plane, the lattice (outside cells marked), the biplot and the cost,
                    with nothing rendered: the panel shows it before Start.
  * POST /start  -- refused past `microscope.MAX_IMAGE_EQ` with the reason, as Metro refuses.
  * POST /{id}/zoom {level, ia, ib} -- a new level centred on that cell at s/2, in the same
                    plane; cached by (centre, s), so a revisit costs nothing. Refused for an
                    outside cell, past the smallest half-width, or past the run's ceiling.

Crossing mode is validated against the live Cascade run it names the way Metro validates its
seed source -- unknown run 404, still running 409, unknown crossing 404, prompts that differ
400 -- because a plane "through a crossing" of some other field is a different picture.
"""
import asyncio
import uuid

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from backend.models import (MicroscopeRequest, MicroPlan, MicroStartResponse, MicroStatus,
                            MicroLevel, MicroCell, MicroEdge, MicroZoomRequest,
                            MicroZoomResponse)
from backend.services import microscope as ms
from backend.services import gridfree as gf
from backend.services import cascade as cs
from backend.services.cascade import COS_T
from backend.cache.thumbnail_cache import ThumbnailStore

router = APIRouter(prefix="/api/microscope", tags=["microscope"])


def _runs(app):
    app.state.microscope_runs = getattr(app.state, "microscope_runs", {})
    return app.state.microscope_runs


def _cascades(app):
    return getattr(app.state, "cascades", {}) or {}


def _centre(req_centre, k, fallback):
    """The lattice centre as a legal recipe (400 with the reason otherwise)."""
    if req_centre is None:
        return np.asarray(fallback, dtype=float)
    w, why = gf.legal_recipe(req_centre, k)
    if w is None:
        raise HTTPException(status_code=400, detail=f"centre is not a legal recipe: {why}")
    return w


def _resolve(app, req: MicroscopeRequest):
    """Everything the request names, resolved: prompts, image settings, centre and plane.

    Raises HTTPException with the reason for anything that cannot be honoured.
    """
    out = {"source_run": None, "source_cid": None, "notes": []}
    if req.mode == "crossing":
        if not req.cascade_run_id or req.cid is None:
            raise HTTPException(
                status_code=400,
                detail="mode 'crossing' needs cascade_run_id and cid (the crossing the plane "
                       "goes through); use 'random' or 'prompt-swap' for a free plane")
        src = _cascades(app).get(req.cascade_run_id)
        if src is None:
            raise HTTPException(status_code=404, detail=f"no cascade run {req.cascade_run_id}")
        if getattr(src, "status", "") == "running":
            raise HTTPException(
                status_code=409,
                detail=f"cascade run {req.cascade_run_id} is still running and its crossings "
                       f"can still move; open the microscope once it has finished")
        x = next((c for c in getattr(src, "crossings", None) or () if c.cid == req.cid), None)
        if x is None:
            raise HTTPException(status_code=404,
                                detail=f"cascade run {req.cascade_run_id} has no crossing "
                                       f"{req.cid}")
        prompts = list(src.prompts)
        if req.prompts is not None and list(req.prompts) != prompts:
            raise HTTPException(
                status_code=400,
                detail="crossing mode draws the lattice from the cascade run's own field, so "
                       "its prompts are used; the prompts sent differ from them")
        k = len(prompts)
        mid = x.mid if getattr(x, "mid", None) is not None else (x.wa + x.wb) / 2
        centre = _centre(req.centre, k, cs._clipn(np.asarray(mid, dtype=float), k))
        cloud = getattr(x, "cloud", None) or {}
        pseed = int(src.seed) if req.plane_seed is None else int(req.plane_seed)
        try:
            e1, e2, note = ms.plane_crossing(
                k, x.wa, x.wb, cloud_normal=cloud.get("normal"),
                cloud_pts=getattr(x, "cloud_pts", None), seed=pseed,
                redraw=req.e2_redraw, cid=int(req.cid))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        out.update(seed=int(src.seed), steps=int(src.steps), height=int(src.height),
                   width=int(src.width), guidance_scale=float(src.guidance_scale),
                   source_run=req.cascade_run_id, source_cid=int(req.cid))
        out["notes"].append(
            f"crossing {req.cid} of cascade run {req.cascade_run_id}: its prompts, seed "
            f"{src.seed} and {src.steps} steps are used, so the lattice is that run's field")
    else:
        if req.prompts is None:
            raise HTTPException(status_code=400,
                                detail=f"mode '{req.mode}' needs prompts (3 or more)")
        prompts = list(req.prompts)
        k = len(prompts)
        centre = _centre(req.centre, k, np.full(k, 1.0 / k))
        try:
            if req.mode == "prompt-swap":
                if req.swap_a is None or req.swap_b is None:
                    raise ValueError("mode 'prompt-swap' needs swap_a = [i, j] and "
                                     "swap_b = [p, q] (0-based prompt indices)")
                e1, e2, note = ms.plane_prompt_swap(k, tuple(req.swap_a), tuple(req.swap_b))
            else:
                pseed = int(req.seed) if req.plane_seed is None else int(req.plane_seed)
                e1, e2, note = ms.plane_random(k, pseed, req.e2_redraw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        out.update(seed=int(req.seed), steps=int(req.steps), height=int(req.height),
                   width=int(req.width), guidance_scale=float(req.guidance_scale))
    out.update(prompts=prompts, k=k, centre=centre, e1=e1, e2=e2, plane_note=note)
    return out


def _refuse(cost, spent=0.0):
    return (f"this lattice would render {cost:.0f} new images "
            + (f"on top of the {spent:.0f} this run has already spent " if spent else "")
            + f"-- past the {ms.MAX_IMAGE_EQ} image-eq ceiling of one microscope run. Every "
              f"in-simplex cell is one full-fidelity image; use a smaller grid, or start a "
              f"new run at this centre.")


@router.post("/plan", response_model=MicroPlan)
async def plan(req: MicroscopeRequest, request: Request):
    """The plane and lattice a /start with this body would render, and what it would cost.
    Geometry only -- nothing is rendered."""
    r = _resolve(request.app, req)
    cells = ms.lattice(r["centre"], r["e1"], r["e2"], req.s, req.grid)
    n_in = sum(1 for c in cells if c["inside"])
    notes = list(r["notes"])
    if n_in < len(cells):
        notes.append(f"{len(cells) - n_in} of {len(cells)} cells fall outside the simplex: "
                     f"shown empty, not rendered, not charged")
    return MicroPlan(
        k=r["k"], prompts=r["prompts"], mode=req.mode,
        centre=[float(v) for v in r["centre"]], s=float(req.s), grid=int(req.grid),
        e1=[float(v) for v in r["e1"]], e2=[float(v) for v in r["e2"]],
        plane_note=r["plane_note"], biplot=ms.biplot(r["e1"], r["e2"]),
        cells=[MicroCell(**c) for c in cells], n_inside=n_in,
        cost_image_eq=ms.projected_cost(cells), max_image_eq=float(ms.MAX_IMAGE_EQ),
        seed=r["seed"], steps=r["steps"], source_run=r["source_run"],
        source_cid=r["source_cid"], notes=notes)


@router.post("/start", response_model=MicroStartResponse)
async def start(req: MicroscopeRequest, request: Request):
    app = request.app
    r = _resolve(app, req)
    cells = ms.lattice(r["centre"], r["e1"], r["e2"], req.s, req.grid)
    cost = ms.projected_cost(cells)
    if cost > ms.MAX_IMAGE_EQ:
        raise HTTPException(status_code=400, detail=_refuse(cost))
    if not any(c["inside"] for c in cells):
        raise HTTPException(status_code=400,
                            detail="no lattice cell lies inside the simplex; move the centre "
                                   "inward or shrink s")
    rid = uuid.uuid4().hex[:8]
    run = ms.MicroRun(
        run_id=rid, prompts=r["prompts"], seed=r["seed"], steps=r["steps"],
        height=r["height"], width=r["width"], guidance_scale=r["guidance_scale"],
        mode=req.mode, e1=[float(v) for v in r["e1"]], e2=[float(v) for v in r["e2"]],
        grid=int(req.grid), plane_note=r["plane_note"], source_run=r["source_run"],
        source_cid=r["source_cid"], notes=list(r["notes"]) + [r["plane_note"]])
    run.thumbs = ThumbnailStore(app.state.cache)
    lv, _ = ms.add_level(run, r["centre"], req.s)
    _runs(app)[rid] = run
    asyncio.create_task(asyncio.to_thread(_render, app, rid, lv.level))
    return MicroStartResponse(run_id=rid, status="running", level=lv.level,
                              cost_image_eq=float(cost))


def _render(app, rid, level):
    lv = None
    try:
        run = _runs(app)[rid]
        lv = run.levels[level]
        ms.render_level(app, run, app.state.gpu_pool, lv)
    except Exception as exc:                         # noqa: BLE001
        if lv is not None:
            lv.status = "error"
            lv.error = f"{type(exc).__name__}: {exc}"


def _zoom_cost(run, lv, c):
    """New images a click on cell c would render: 0 when that level exists, None past S_MIN."""
    if not c["inside"]:
        return None
    centre, s2 = ms.zoom_target(lv.centre, lv.s, lv.grid, c["ia"], c["ib"], run.e1, run.e2)
    if s2 < ms.S_MIN:
        return None
    if ms.level_key(centre, s2) in run.level_index:
        return 0
    return ms.new_images(ms.lattice(centre, run.e1, run.e2, s2, lv.grid), run)


def _level(run, lv, costs):
    cells = [MicroCell(**c, zoom_cost=_zoom_cost(run, lv, c) if costs else None)
             for c in lv.cells]
    return MicroLevel(
        level=lv.level, parent=lv.parent, parent_cell=lv.parent_cell, centre=lv.centre,
        s=lv.s, grid=lv.grid, status=lv.status, cells=cells,
        edges=[MicroEdge(**e) for e in lv.edges],
        n_inside=sum(1 for c in lv.cells if c["inside"]), n_new=lv.n_new,
        n_reused=lv.n_reused, notes=list(lv.notes), error=lv.error)


@router.get("/{run_id}/status", response_model=MicroStatus)
async def status(run_id: str, request: Request, level: int | None = None):
    """The run with every level. Zoom costs are computed for `level` only (the one on screen;
    default: the newest) -- they are G^2 lattices each, so not for the whole stack per poll."""
    run = _runs(request.app).get(run_id)
    if run is None:
        return MicroStatus(run_id=run_id, status="unknown", error="no such run")
    show = len(run.levels) - 1 if level is None else int(level)
    return MicroStatus(
        run_id=run.run_id, status=run.status, k=run.k, prompts=list(run.prompts),
        mode=run.mode, e1=list(run.e1), e2=list(run.e2), plane_note=run.plane_note,
        biplot=ms.biplot(run.e1, run.e2), grid=run.grid, seed=run.seed, steps=run.steps,
        source_run=run.source_run, source_cid=run.source_cid,
        levels=[_level(run, lv, lv.level == show and lv.status == "complete")
                for lv in run.levels],
        cost_image_eq=run.cost_image_eq, max_image_eq=float(ms.MAX_IMAGE_EQ),
        s_min=ms.S_MIN, cos_t=COS_T, generated=run.generated, notes=list(run.notes),
        error=next((lv.error for lv in reversed(run.levels) if lv.error), None))


@router.post("/{run_id}/zoom", response_model=MicroZoomResponse)
async def zoom(run_id: str, req: MicroZoomRequest, request: Request):
    """Zoom 2x into one cell: a new level centred on it at s/2, in the same plane."""
    app = request.app
    run = _runs(app).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no microscope run {run_id}")
    if req.level >= len(run.levels):
        raise HTTPException(status_code=404,
                            detail=f"run {run_id} has no level {req.level} "
                                   f"(it has {len(run.levels)})")
    lv = run.levels[req.level]
    cell = next((c for c in lv.cells if c["ia"] == req.ia and c["ib"] == req.ib), None)
    if cell is None:
        raise HTTPException(status_code=404,
                            detail=f"level {req.level} is {lv.grid} x {lv.grid}; there is no "
                                   f"cell ({req.ia}, {req.ib})")
    if not cell["inside"]:
        raise HTTPException(status_code=400,
                            detail="that cell lies outside the simplex: it is not a recipe, so "
                                   "there is nothing to zoom into")
    centre, s2 = ms.zoom_target(lv.centre, lv.s, lv.grid, req.ia, req.ib, run.e1, run.e2)
    if s2 < ms.S_MIN:
        raise HTTPException(
            status_code=400,
            detail=f"a further zoom would make the half-width {s2:.4f}, below {ms.S_MIN}: the "
                   f"whole lattice would span less than a tenth of a Cascade stride, where a "
                   f"single seed shows no difference between cells")
    key = ms.level_key(centre, s2)
    if key in run.level_index:
        old = run.levels[run.level_index[key]]
        if old.status in ("cancelled", "error"):
            # a level whose render was stopped is re-rendered; its finished cells come back
            # out of the render cache, so only the missing ones are paid for
            old.status, old.error = "pending", None
            asyncio.create_task(asyncio.to_thread(_render, app, run_id, old.level))
            return MicroZoomResponse(level=old.level, cached=False, status="running",
                                     cost_image_eq=ms.projected_cost(old.cells, run))
        return MicroZoomResponse(level=old.level, cached=True, status=old.status)
    cells = ms.lattice(centre, run.e1, run.e2, s2, lv.grid)
    cost = ms.projected_cost(cells, run)
    if run.cost_image_eq + cost > ms.MAX_IMAGE_EQ:
        raise HTTPException(status_code=400, detail=_refuse(cost, run.cost_image_eq))
    new, _ = ms.add_level(run, centre, s2, parent=lv.level, parent_cell=[req.ia, req.ib])
    asyncio.create_task(asyncio.to_thread(_render, app, run_id, new.level))
    return MicroZoomResponse(level=new.level, cached=False, status="running",
                             cost_image_eq=float(cost))


@router.post("/{run_id}/cancel")
async def cancel(run_id: str, request: Request):
    """Stop every level still rendering. A later zoom still works (and re-renders a cancelled
    level on demand)."""
    run = _runs(request.app).get(run_id)
    if run is not None:
        for lv in run.levels:
            if lv.status in ("pending", "running"):
                lv.status = "cancelled"
    return {"ok": True}


@router.get("/{run_id}/image/{index}")
async def image(run_id: str, index: int, request: Request):
    run = _runs(request.app).get(run_id)
    if run is None:
        return Response(status_code=404)
    b = run.thumbs.get(index) if run.thumbs is not None else None
    if b is None:
        return Response(status_code=404)
    return Response(content=b, media_type="image/jpeg")
