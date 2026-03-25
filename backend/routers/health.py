from fastapi import APIRouter, Request
from backend.models import HealthResponse
from backend import config

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    pool = request.app.state.gpu_pool
    return HealthResponse(
        status="ok",
        n_gpus=config.N_GPUS,
        workers_ready=pool.ready_count,
        model=config.MODEL_ID,
    )
