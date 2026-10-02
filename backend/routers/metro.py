"""Metropolis sampler endpoints: start / status / cancel / images / samples.json.

Mirrors the AMR run pattern (routers/amr.py), which mirrors the Cascade's: a plain sync
worker on a thread via asyncio.to_thread, cooperative cancel through run.status, state in
its own never-evicted app.state dict, images served by index from the content-hash
thumbnail store, and every worker-side state change inside one try/except so a failure
cannot leave a run stuck "running".

Two things this router does beyond that:

  * it REFUSES a configuration whose projected cost passes `metro.MAX_IMAGE_EQ`, with the
    projection broken down by phase. The chain is one batched round per MCMC step, so the
    bill is (1 + m) x chains x steps probes and easy to type past the shared pool's day;
  * it validates the CASCADE seed mode against the live run it names -- unknown id, still
    running, wrong k, or no crossings -- because "seed at the crossings" of a run that has
    none would silently become uniform sampling, which is a different experiment.

`GET /api/metro/cascade-runs` lists the runs that mode can use, so the panel can offer a
selector instead of asking the user to remember an 8-hex id.
"""
import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from backend.models import (MetroStartRequest, MetroStartResponse, MetroStatus, MetroSample,
                            MetroChainStat, MetroSummary, MetroExport, MetroCascadeRun)
from backend.services import metro
from backend.services.cascade import COS_T
from backend.cache.thumbnail_cache import ThumbnailStore

router = APIRouter(prefix="/api/metro", tags=["metro"])


def _runs(app):
    app.state.metro_runs = getattr(app.state, "metro_runs", {})
    return app.state.metro_runs


def _cascades(app):
    return getattr(app.state, "cascades", {}) or {}


def _n_crossings(src):
    return len(getattr(src, "crossings", None) or ())


@router.get("/cascade-runs", response_model=list[MetroCascadeRun])
async def cascade_runs(request: Request, k: int | None = None):
    """Finished Cascade runs whose crossings this sampler can be seeded from.

    `k` filters to one arity, which is what the panel asks for: a run's crossings are weight
    vectors over ITS prompts, so they only seed a chain over the same number of prompts.
    Runs with no crossings are listed with n_crossings 0 rather than hidden -- the panel
    greys them out, and a silent omission looks like the run is gone.
    """
    out = []
    for rid, src in _cascades(request.app).items():
        if getattr(src, "status", "") == "running":
            continue
        kk = len(getattr(src, "prompts", None) or ())
        if k is not None and kk != k:
            continue
        out.append(MetroCascadeRun(
            run_id=rid, status=str(getattr(src, "status", "")), k=kk,
            prompts=list(getattr(src, "prompts", None) or ()),
            n_crossings=_n_crossings(src),
            certify=bool(getattr(src, "certify", False))))
    return sorted(out, key=lambda r: (-r.n_crossings, r.run_id))


def _refuse_budget(req, k):
    """The 400 body for a configuration past the image-eq ceiling: the projection, by phase."""
    total, rows = metro.projected_cost(
        k, req.m, req.chains, req.chain_steps, req.n_seed_chords, req.seed_mode,
        req.probe_steps, req.render_full)
    parts = "; ".join(f"{r['what']} {r['probes']} probes = {r['image_eq']:.0f} image-eq"
                      for r in rows)
    return (f"this configuration projects {total:.0f} image-eq, past the "
            f"{metro.MAX_IMAGE_EQ} ceiling: {parts}. Each MCMC step is one batched round of "
            f"(1 + m) probes per chain, so the bill scales as (1 + {req.m}) x {req.chains} "
            f"chains x {req.chain_steps} steps. Lower chains, steps or m (h25a's design of "
            f"record is 60 x 50 with m = 2, about "
            f"{metro.projected_cost(k, 2, 60, 50, 20, 'iur', 4)[0]:.0f} image-eq).")


@router.post("/start", response_model=MetroStartResponse)
async def start(req: MetroStartRequest, request: Request):
    app = request.app
    k = len(req.prompts)
    if req.seed_mode == "cascade":
        if not req.cascade_run_id:
            raise HTTPException(
                status_code=400,
                detail="seed_mode 'cascade' needs cascade_run_id (the run whose crossings "
                       "the chains start on); use seed_mode 'iur' to draw fresh chords")
        src = _cascades(app).get(req.cascade_run_id)
        if src is None:
            raise HTTPException(status_code=404,
                                detail=f"no cascade run {req.cascade_run_id}")
        if getattr(src, "status", "") == "running":
            raise HTTPException(
                status_code=409,
                detail=f"cascade run {req.cascade_run_id} is still running; seed from it "
                       f"once it has finished, or use seed_mode 'iur'")
        src_k = len(getattr(src, "prompts", None) or ())
        if src_k != k:
            raise HTTPException(
                status_code=400,
                detail=f"cascade run {req.cascade_run_id} has k={src_k} but this request has "
                       f"k={k}: its crossings are weight vectors over its own prompts")
        if not _n_crossings(src):
            raise HTTPException(
                status_code=400,
                detail=f"cascade run {req.cascade_run_id} found no crossings, so there is "
                       f"nothing to seed the chains at (seeding uniformly instead would be a "
                       f"different experiment -- use seed_mode 'iur')")
    total, _rows = metro.projected_cost(k, req.m, req.chains, req.chain_steps,
                                        req.n_seed_chords, req.seed_mode, req.probe_steps,
                                        req.render_full)
    if total > metro.MAX_IMAGE_EQ:
        raise HTTPException(status_code=400, detail=_refuse_budget(req, k))
    rid = uuid.uuid4().hex[:8]
    run = metro.MetroRun(
        run_id=rid, prompts=list(req.prompts), seed=req.seed, steps=req.steps,
        height=req.height, width=req.width, guidance_scale=req.guidance_scale,
        beta=req.beta, sigma=req.sigma, m=req.m, chains=req.chains,
        chain_steps=req.chain_steps, seed_mode=req.seed_mode,
        cascade_run_id=req.cascade_run_id, n_seed_chords=req.n_seed_chords,
        probe_steps=req.probe_steps, render_full=req.render_full)
    if req.beta > metro.BETA_WARN:
        # not a refusal: 1.5 was never measured either way, and the run is legal. But the
        # reason beta = 2 failed (estimator noise, squared) applies here too, so it is on
        # the record the run carries rather than only in a tooltip.
        run.notes.append(
            f"beta {req.beta} is above the {metro.BETA_WARN} of record: under single-seed "
            f"energies h25a measured KS 0.19-0.28 at beta 2 (vs 0.07-0.13 at beta 1), "
            f"because squaring an m-direction estimate of S amplifies the estimator noise")
    run.thumbs = ThumbnailStore(app.state.cache)
    _runs(app)[rid] = run
    asyncio.create_task(asyncio.to_thread(_run, app, rid))
    return MetroStartResponse(run_id=rid, status="running")


def _run(app, rid):
    run = None
    try:
        run = _runs(app)[rid]
        metro.run_metro(app, run, app.state.gpu_pool)
    except Exception as exc:                         # noqa: BLE001
        if run is not None:
            run.status = "error"
            run.error = f"{type(exc).__name__}: {exc}"


def _samples(run):
    return [MetroSample(**x) for x in run.samples]


def _chain_stats(run):
    return [MetroChainStat(**c) for c in run.chains_state]


@router.get("/{run_id}/status", response_model=MetroStatus)
async def status(run_id: str, request: Request):
    run = _runs(request.app).get(run_id)
    if run is None:
        return MetroStatus(run_id=run_id, status="unknown", error="no such run")
    return MetroStatus(
        run_id=run.run_id, status=run.status, phase=run.phase, k=run.k,
        prompts=list(run.prompts), beta=run.beta, sigma=run.sigma, m=run.m,
        chains=run.chains, chain_steps=run.chain_steps, delta=run.delta,
        seed_mode=run.seed_mode, cascade_run_id=run.cascade_run_id,
        n_seed_chords=run.n_seed_chords, probe_steps=run.probe_steps,
        render_full=run.render_full, seed=run.seed, steps=run.steps,
        generated=run.generated, phase_done=run.phase_done, phase_total=run.phase_total,
        round_done=run.round_done,
        recent_thumbs=list(getattr(run, "recent_thumbs", [])),
        seeds=[list(x["weights"]) for x in run.seed_crossings],
        seed_divs=[x["divergence"] for x in run.seed_crossings],
        chains_stats=_chain_stats(run), samples=_samples(run),
        summary=MetroSummary(**run.summary()),
        notes=list(run.notes), error=run.error)


@router.post("/{run_id}/cancel")
async def cancel(run_id: str, request: Request):
    run = _runs(request.app).get(run_id)
    if run is not None and run.status == "running":
        run.status = "cancelled"
    return {"ok": True}


@router.get("/{run_id}/samples.json", response_model=MetroExport)
async def samples_export(run_id: str, request: Request):
    """The accepted states as records, for reuse outside the map (404 for an unknown run)."""
    run = _runs(request.app).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="no such run")
    return MetroExport(
        run_id=run.run_id, status=run.status, k=run.k, prompts=list(run.prompts),
        beta=run.beta, sigma=run.sigma, m=run.m, delta=run.delta, chains=run.chains,
        chain_steps=run.chain_steps, seed_mode=run.seed_mode,
        cascade_run_id=run.cascade_run_id, seed=run.seed, steps=run.steps,
        probe_steps=run.probe_steps, cos_t=COS_T,
        seeds=[list(x["weights"]) for x in run.seed_crossings],
        samples=_samples(run), chains_stats=_chain_stats(run),
        summary=MetroSummary(**run.summary()), notes=list(run.notes))


@router.get("/{run_id}/image/{index}")
async def image(run_id: str, index: int, request: Request):
    """One probe's thumbnail, by the image index the samples carry (the Cascade's store)."""
    run = _runs(request.app).get(run_id)
    if run is None:
        return Response(status_code=404)
    b = run.thumbs.get(index) if run.thumbs is not None else None
    if b is None:
        return Response(status_code=404)
    return Response(content=b, media_type="image/jpeg")
