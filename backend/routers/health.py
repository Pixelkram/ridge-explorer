from fastapi import APIRouter, Request
from backend.models import HealthResponse
from backend import config

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    pool = request.app.state.gpu_pool
    # Liveness, not a high-water mark: ready_count only ever increments, so a pool that
    # lost every worker still reported "ok" with the full count. Workers now os._exit(1)
    # on a lost CUDA context, which makes is_alive() a real signal.
    live = sum(1 for w in pool.workers if w.is_alive())
    return HealthResponse(
        status="ok" if live >= config.N_GPUS else ("degraded" if live else "down"),
        n_gpus=config.N_GPUS,
        workers_ready=live,
        model=config.MODEL_ID,
        recent_errors=[f"gpu {e.get('gpu_id')}: {e.get('error', '')}"
                       for e in pool.recent_errors[-5:]],
    )
