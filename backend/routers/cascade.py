import time
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

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from backend import config
from backend.services.gpu_pool import ProbeTask
from backend.models import (
    ProbeStartResponse, CascadeProbeRequest,CascadeStartRequest, CascadeStartResponse, CascadeStatus,
                            CascadeCrossing, CascadePatch, CascadeChord, CascadeChordMeta,
                            WalkStartRequest, WalkStep, WalkStatus, RenderRequest,
                            CascadePointInfo, LocalSvChord, LocalSvMap)
from backend.services import cascade as cs
from backend.services import local_sv as lsv
from backend.cache.thumbnail_cache import ThumbnailStore

router = APIRouter(prefix="/api/cascade", tags=["cascade"])

# A run started with certify off stops after the chord phase: its crossings were never
# bisected or scored, so everything that reads a pinned `mid` or a `b`/`significant` verdict
# (walks, walk certification, the trace phase, the certified-only density map) has nothing to
# stand on and says so rather than quietly answering from bracket-precision guesses.
DETECTION_ONLY = "run was detection-only: no bisected crossings — re-run with certify on"


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
        probe_steps=req.probe_steps, stride=req.stride, certify=req.certify,
        branch=req.branch, depth=req.depth, branch_top_pct=req.branch_top_pct,
        hires=req.hires, hires_top_pct=req.hires_top_pct, hires_factor=req.hires_factor,
        hires_mode=req.hires_mode, hires_cloud_n=req.hires_cloud_n,
        hires_cloud_r=req.hires_cloud_r,
        chord_seed=req.chord_seed,
        trace=req.trace, trace_steps=req.trace_steps, trace_certify=req.trace_certify)
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
    traced = bool(getattr(run, "traces", None))
    # Ridge IDENTITIES (the map's grouping) come from every crossing; the Good-Turing
    # CERTIFICATE is quoted on generation 0 alone when branching was on, because a child ray
    # starts on a boundary -- a preferential sample, not a fair sample of boundary area.
    roots_only = int(getattr(run, "branch", 0)) > 0
    fair = ([x for x in run.crossings if getattr(x, "gen", 0) == 0] if roots_only
            else run.crossings)
    grp = cs.coverage_stats(run.crossings)
    grp_t = (cs.coverage_stats(run.crossings, links=run.trace_links) if traced
             else (None, None, None, None))
    cov = cs.coverage_stats(fair) if roots_only else grp
    cov_t = ((cs.coverage_stats(fair, links=run.trace_links) if roots_only else grp_t)
             if traced else (None, None, None, None))
    return CascadeStatus(
        run_id=run.run_id, status=run.status, phase=run.phase,
        generated=run.generated, phase_done=run.phase_done,
        phase_total=run.phase_total, prompts=run.prompts,
        recent_thumbs=list(getattr(run, "recent_thumbs", [])),
        points=list(getattr(run, "probe_geo", [])),
        point_divs=list(getattr(run, "probe_div", [])),
        chords=[CascadeChord(a=c[0], b=c[1])
                for c in getattr(run, "chords_geo", [])],
        chords_meta=[CascadeChordMeta(**m) for m in getattr(run, "chords_meta", [])],
        stride=float(getattr(run, "stride", cs.STRIDE)),
        branch=int(getattr(run, "branch", 0)), depth=int(getattr(run, "depth", 0)),
        branch_top_pct=int(getattr(run, "branch_top_pct", 20)),
        hires=bool(getattr(run, "hires", False)),
        hires_top_pct=int(getattr(run, "hires_top_pct", 20)),
        hires_factor=int(getattr(run, "hires_factor", 4)),
        hires_mode=str(getattr(run, "hires_mode", "bracket")),
        hires_cloud_n=int(getattr(run, "hires_cloud_n", 12)),
        hires_cloud_r=float(getattr(run, "hires_cloud_r", 0.05)),
        certify=bool(getattr(run, "certify", True)),
        crossings=[CascadeCrossing(
            cid=x.cid, weights=[float(v) for v in x.mid] if x.mid is not None
            else [float(v) for v in (x.wa + x.wb) / 2],
            b=x.b, significant=x.significant, thumb=x.thumb,
            gen=int(getattr(x, "gen", 0)),
            ridge_group=(grp[3] or {}).get(x.cid),
            traced_group=(grp_t[3] or {}).get(x.cid),
            hires=bool(getattr(x, "hires", False)),
            hires_width=getattr(x, "hires_width", None),
            split_from=getattr(x, "split_from", None),
            hires_mode=getattr(x, "hires_mode", None),
            cloud=getattr(x, "cloud", None),
            cloud_pts=getattr(x, "cloud_pts", None),
            bracket_w=float(np.linalg.norm(x.wa - x.wb))
            if x.wa is not None and x.wb is not None else None)
            for x in run.crossings],
        bg_mean=run.bg_mean, bg_p95=run.bg_p95,
        distinct_ridges=cov[0], singleton_ridges=cov[1],
        unexplored_share=cov[2], stats_scope="roots" if roots_only else "all",
        traces=list(getattr(run, "traces", [])),
        trace_links=[list(l) for l in getattr(run, "trace_links", [])],
        traced_ridges=cov_t[0], traced_singletons=cov_t[1], traced_unexplored_share=cov_t[2],
        patches=[CascadePatch(
            region=p.region, cid=p.cid, b=p.b, significant=p.significant,
            exploration=p.exploration, grid=p.grid,
            cols_with_crossing=p.cols_with_crossing) for p in run.patches],
        notes=list(run.notes), error=run.error)


@router.get("/{run_id}/local-sv", response_model=LocalSvMap)
async def local_sv_map(run_id: str, request: Request, h: float = lsv.H_DEFAULT,
                       mode: str = "all", cloud: int = 0):
    """Local boundary density over this run's own chords (services/local_sv.py).

    Reads a finished survey's geometry -- no images, no GPU. `mode=all` (the default) is the
    estimator of record; `mode=certified` filters the numerator only and is a diagnostic,
    not a cleaner map. Ranking is trustworthy from ~20 chords; the values are calibrated
    only once `calibrated_ok` (>= 80 chords, and the calibration itself is k=4 only) --
    evidence in search_problem/outputs/h23_local_sv_kde.

    Branching runs feed ALL chords in here, children included: the ratio estimator needs the
    chord DIRECTIONS to be isotropic, not their positions to be uniform, and the children's
    are (_iso_dir). Its `s_global` is the one number that does read as a global average, so
    with branching on it is biased upward -- the fair-area figure is the status certificate,
    which is quoted on the roots alone.
    """
    run = _runs(request.app).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="no such run")
    if mode not in ("all", "certified"):
        raise HTTPException(status_code=400, detail="mode must be 'all' or 'certified'")
    if mode == "certified" and not getattr(run, "certify", True):
        # mode=all is the estimator of record and reads the same chords either way; the
        # certified filter would silently return an all-zero numerator instead
        raise HTTPException(status_code=400, detail=DETECTION_ONLY)
    if not (0.01 <= h <= 1.0):
        raise HTTPException(status_code=400, detail="h must lie in [0.01, 1.0]")
    chords, crossings, k = lsv.from_run(run)
    if not chords or k < 3:
        # nothing laid down yet -- the survey has not reached a map
        raise HTTPException(status_code=404, detail="run has no chords yet")
    if not crossings:
        # a finished survey CAN find nothing (run_cascade says so in its notes); that is a
        # real, empty answer about the space, not a server error
        return LocalSvMap(run_id=run_id, k=k, h=h, mode=mode, n_chords=len(chords),
                          calibrated_ok=len(chords) >= lsv.N_CALIBRATED,
                          error="no boundary crossings in this run")
    m = lsv.local_sv(chords, crossings, k, h=h, mode=mode, cloud=max(0, min(cloud, 20000)))
    st, xs, cl = m["stations"], m["crossings"], m["cloud"]
    sel = [st["chord"] == ci for ci in range(len(chords))]
    by_chord = [LocalSvChord(
        points=[[float(v) for v in w] for w in st["weights"][s]],
        values=[float(v) for v in st["values"][s]],
        ranks=[float(v) for v in st["ranks"][s]],
        uncovered=[bool(v) for v in st["uncovered"][s]]) for s in sel]
    return LocalSvMap(
        run_id=run_id, k=k, h=m["h"], mode=m["mode"], delta=m["delta"], c_d=m["c_d"],
        n_chords=m["n_chords"], n_crossings=m["n_crossings"],
        calibrated_ok=m["calibrated_ok"], s_global=m["s_global"], chords=by_chord,
        crossing_cids=[int(x.cid) for x in run.crossings],
        crossing_values=[float(v) for v in xs["values"]],
        crossing_ranks=[float(v) for v in xs["ranks"]],
        crossing_uncovered=[bool(v) for v in xs["uncovered"]],
        cloud=[[float(v) for v in w] for w in cl["weights"]],
        cloud_values=[float(v) for v in cl["values"]],
        cloud_ranks=[float(v) for v in cl["ranks"]],
        cloud_uncovered=[bool(v) for v in cl["uncovered"]])


@router.post("/{run_id}/walk", response_model=WalkStatus)
async def walk_start(run_id: str, req: WalkStartRequest, request: Request):
    app = request.app
    run = _runs(app).get(run_id)
    if run is None:
        return WalkStatus(walk_id="", status="error", error="no such run")
    if not getattr(run, "certify", True):
        raise HTTPException(status_code=409, detail=DETECTION_ONLY)
    if not any(x.cid == req.cid and x.mid is not None for x in run.crossings):
        return WalkStatus(walk_id="", status="error",
                          error="crossing not found (or run still bisecting)")
    wid = uuid.uuid4().hex[:8]
    walk = cs.Walk(wid=wid, cid=req.cid, direction=1 if req.direction >= 0 else -1,
                   n_steps=req.n_steps, use_jvp=bool(req.use_jvp), sig_mode=req.sig_mode, mode=req.mode,
                   cont_gamma=req.cont_gamma)
    run.walks = getattr(run, "walks", {})
    run.walks[wid] = walk

    def _go():
        try:
            cs.run_walk(app, run, app.state.gpu_pool, walk)
        except Exception as exc:                 # noqa: BLE001
            walk.status = "error"
            walk.error = f"{type(exc).__name__}: {exc}"

    asyncio.create_task(asyncio.to_thread(_go))
    return WalkStatus(walk_id=wid, status="running", cid=req.cid, mode=req.mode)


def _walk_status(walk):
    return WalkStatus(
        walk_id=walk.wid, status=walk.status, cid=walk.cid, mode=walk.mode, images=walk.images,
        plane=walk.plane, cert=walk.cert, trace=list(walk.trace), jvp=getattr(walk, "jvp", None),
        steps=[WalkStep(**s) for s in walk.steps],
        segs=list(walk.segs),
        notes=list(walk.notes), error=walk.error)


@router.get("/{run_id}/walk/{walk_id}", response_model=WalkStatus)
async def walk_status(run_id: str, walk_id: str, request: Request):
    run = _runs(request.app).get(run_id)
    walk = getattr(run, "walks", {}).get(walk_id) if run is not None else None
    if walk is None:
        return WalkStatus(walk_id=walk_id, status="unknown", error="no such walk")
    return _walk_status(walk)


@router.post("/{run_id}/walk/{walk_id}/certify", response_model=WalkStatus)
async def walk_certify(run_id: str, walk_id: str, request: Request):
    """Re-measure a finished walk's stations on held-out seeds (cs.certify_walk); poll the walk for `cert`."""
    app = request.app
    run = _runs(app).get(run_id)
    if run is not None and not getattr(run, "certify", True):
        # no walk can exist on such a run (walk_start refuses), and certification reads the
        # run's measured background, which the score phase never produced
        raise HTTPException(status_code=409, detail=DETECTION_ONLY)
    walk = getattr(run, "walks", {}).get(walk_id) if run is not None else None
    if walk is None:
        return WalkStatus(walk_id=walk_id, status="unknown", error="no such walk")
    if walk.status == "running":
        return WalkStatus(walk_id=walk_id, status="running", error="walk still running")
    walk.cert = {"status": "running"}

    def _go():
        try:
            cs.certify_walk(app, run, app.state.gpu_pool, walk)
        except Exception as exc:                 # noqa: BLE001
            walk.cert = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    asyncio.create_task(asyncio.to_thread(_go))
    return _walk_status(walk)


@router.post("/{run_id}/render")
async def render_start(run_id: str, req: RenderRequest, request: Request):
    """Research tool (cs.render_points): render weight vectors with this run's pipeline/seed/steps/size, save the
    DINOv2 embeddings to tests/gt/<name>.npz; poll GET /{run_id}/render/{render_id}."""
    from pathlib import Path
    app = request.app
    run = _runs(app).get(run_id)
    if run is None:
        return {"status": "error", "error": "no such run"}
    if not req.weights or any(len(w) != run.k for w in req.weights):
        return {"status": "error", "error": f"every weight vector needs k={run.k} entries"}
    out = Path(__file__).resolve().parents[2] / "tests" / "gt"
    out.mkdir(parents=True, exist_ok=True)
    rid = uuid.uuid4().hex[:8]
    job = {"id": rid, "status": "running", "done": 0, "total": len(req.weights), "path": str(out / f"{req.name}.npz")}
    run.renders = getattr(run, "renders", {})
    run.renders[rid] = job

    def _go():
        try:
            cs.render_points(app, run, app.state.gpu_pool, req.weights,
                             run.seed if req.seed is None else req.seed, job)
        except Exception as exc:                 # noqa: BLE001
            job["status"], job["error"] = "error", f"{type(exc).__name__}: {exc}"

    asyncio.create_task(asyncio.to_thread(_go))
    return {k: v for k, v in job.items() if k != "notes"}


@router.get("/{run_id}/render/{render_id}")
async def render_status(run_id: str, render_id: str, request: Request):
    run = _runs(request.app).get(run_id)
    job = getattr(run, "renders", {}).get(render_id) if run is not None else None
    if job is None:
        return {"status": "unknown"}
    return {k: v for k, v in job.items() if k != "notes"}


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


@router.post("/{run_id}/jvp-probe", response_model=ProbeStartResponse)
async def cascade_jvp_probe(run_id: str, req: CascadeProbeRequest, request: Request):
    """Exact-JVP ridge normal at a crossing (`cid`) or at arbitrary `weights` of this run's k-prompt basis.
    Returns the whitened k-vector normal, rank-1 share (front) and participation ratio (corner). Mixing is
    linear, as in the cascade's own evaluate(); steps/guidance/size follow the run. Poll
    GET /api/grid/jvp-probe/{probe_id}. Evidence: search_problem/outputs/h08_vector_uses/RESULTS.md (h08-1)."""
    run = _runs(request.app).get(run_id)
    if run is None or not run.prompts:
        return ProbeStartResponse(probe_id="", status="error")
    k = len(run.prompts)
    weights, cid = None, None
    if req.cid is not None:
        x = next((c for c in run.crossings if c.cid == req.cid), None)
        if x is None:
            return ProbeStartResponse(probe_id="", status="error")
        w = x.mid if x.mid is not None else (x.wa + x.wb) / 2
        weights, cid = [float(v) for v in w], req.cid
    elif req.weights is not None and len(req.weights) == k:
        weights = [float(v) for v in req.weights]
    if weights is None:
        return ProbeStartResponse(probe_id="", status="error")
    tot = sum(weights)
    if abs(tot - 1.0) > 1e-3 and tot > 0:
        weights = [v / tot for v in weights]
    probe_id = f"jvp_{uuid.uuid4().hex[:8]}"
    request.app.state.jobs[probe_id] = {"type": "jvp_probe", "status": "running", "kind": "jvp", "job_id": run_id,
                                        "weights": weights, "cid": cid, "k": k, "names": [f"P{i + 1}" for i in range(k)],
                                        "result": None, "error": "", "started_at": time.time()}
    request.app.state.gpu_pool.submit(ProbeTask(
        probe_id=probe_id, job_id=run_id, prompts=list(run.prompts), weights=weights,
        seed=req.seed if req.seed is not None else int(run.seed), height=run.height, width=run.width,
        steps=run.steps, guidance_scale=run.guidance_scale, use_slerp=False))
    return ProbeStartResponse(probe_id=probe_id, status="running")
