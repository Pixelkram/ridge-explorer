from pydantic import BaseModel


class GridStartRequest(BaseModel):
    prompt_a: str
    prompt_b: str
    prompt_c: str = ""  # optional 3rd prompt for triangular plane
    grid_size: int = 15
    seed: int = 42
    seed_count: int = 1  # number of seeds (1=single, >1=multi-seed evaluation)
    height: int = 256
    width: int = 256
    steps: int = 4
    guidance_scale: float = 4.0


class GridStartResponse(BaseModel):
    job_id: str
    total_cells: int
    status: str


class RenderHQRequest(BaseModel):
    tau: float = 1.5  # re-render cells above median*tau at HQ


class SeedProbeRequest(BaseModel):
    alpha: float
    beta: float
    seed_start: int = 0
    seed_end: int = 19


class SeedProbeResponse(BaseModel):
    probe_id: str
    total: int
    status: str


class SeedProbeStatus(BaseModel):
    probe_id: str
    alpha: float
    beta: float
    seeds: list[int]
    images: list[str | None]  # thumbnail URLs per seed
    complete: bool


class RefineRequest(BaseModel):
    tau: float = 1.5       # threshold: refine cells with sensitivity >= median*tau
    multiplier: int = 4    # resolution multiplier (e.g., 4 means 4x4 sub-grid per cell)


class RefineResponse(BaseModel):
    refine_job_id: str
    parent_job_id: str
    total_cells: int
    status: str


class CellStatus(BaseModel):
    row: int
    col: int
    alpha: float
    beta: float
    status: str  # "pending", "generated", "hq"
    sensitivity: float | None = None
    cluster: int | None = None
    thumbnail_url: str | None = None
    hq_url: str | None = None
    span: int = 1  # how many grid cells this tile covers (1=normal, N=parent tile)


class GridStatusResponse(BaseModel):
    job_id: str
    status: str  # "running", "complete", "failed"
    phase: str   # "generating", "analyzing", "complete"
    grid_size: int
    cells_generated: int
    cells_total: int
    cells: list[CellStatus]
    seed_cells: dict[str, list[CellStatus]] | None = None  # per-seed cells for multi-seed
    seeds: list[int] = []
    prompt_a: str
    prompt_b: str
    prompt_c: str
    heatmap_url: str | None = None
    overlay_url: str | None = None
    cluster_url: str | None = None
    image_grid_url: str | None = None


class HealthResponse(BaseModel):
    status: str
    n_gpus: int
    workers_ready: int
    model: str
