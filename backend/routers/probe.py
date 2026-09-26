"""Generic exact-JVP probe: any prompt basis, any barycentric point. For Discover runs and ad-hoc use.
Poll GET /api/grid/jvp-probe/{probe_id}. Evidence: search_problem/outputs/h07_chain_SUMMARY.md, h08_vector_uses/RESULTS.md."""
import time
import uuid

from fastapi import APIRouter, Request

from backend.models import GenericProbeRequest, ProbeStartResponse
from backend.services.gpu_pool import ProbeTask

router = APIRouter(prefix="/api/probe", tags=["probe"])


@router.post("/jvp", response_model=ProbeStartResponse)
async def generic_jvp_probe(req: GenericProbeRequest, request: Request):
    k = len(req.prompts)
    if len(req.weights) != k:
        return ProbeStartResponse(probe_id="", status="error")
    weights = [float(v) for v in req.weights]
    tot = sum(weights)
    if abs(tot - 1.0) > 1e-3 and tot > 0:
        weights = [v / tot for v in weights]
    probe_id = f"jvp_{uuid.uuid4().hex[:8]}"
    request.app.state.jobs[probe_id] = {"type": "jvp_probe", "status": "running", "kind": "jvp", "job_id": "",
                                        "weights": weights, "k": k, "names": [f"P{i + 1}" for i in range(k)],
                                        "result": None, "error": "", "started_at": time.time()}
    request.app.state.gpu_pool.submit(ProbeTask(
        probe_id=probe_id, job_id="", prompts=list(req.prompts), weights=weights, seed=req.seed,
        height=req.height, width=req.width, steps=req.steps, guidance_scale=req.guidance_scale, use_slerp=req.use_slerp))
    return ProbeStartResponse(probe_id=probe_id, status="running")
