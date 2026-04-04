from pydantic import BaseModel


class GridStartRequest(BaseModel):
    prompt_a: str
    prompt_b: str
    prompt_c: str = ""
    prompt_d: str = ""  # 4th prompt for 3D mode
    dimensions: int = 2  # 2 or 3
    grid_size: int = 15
    seed: int = 42
    seed_count: int = 1
    height: int = 256
    width: int = 256
    steps: int = 4
    guidance_scale: float = 4.0


class GridStartResponse(BaseModel):
    job_id: str
    total_cells: int
    dimensions: int
    status: str


class RenderHQRequest(BaseModel):
    tau: float = 1.5


class SeedProbeRequest(BaseModel):
    alpha: float
    beta: float
    gamma: float = 0.0
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
    images: list[str | None]
    complete: bool


class RefineRequest(BaseModel):
    tau: float = 1.5
    multiplier: int = 4
    extra_positions: list[tuple[int, int]] = []


class RefineResponse(BaseModel):
    refine_job_id: str
    parent_job_id: str
    total_cells: int
    status: str


class CellStatus(BaseModel):
    row: int
    col: int
    depth: int = 0  # z-index for 3D grids
    alpha: float
    beta: float
    gamma: float = 0.0
    status: str
    sensitivity: float | None = None
    cluster: int | None = None
    thumbnail_url: str | None = None
    hq_url: str | None = None
    span: int = 1


class GridStatusResponse(BaseModel):
    job_id: str
    status: str
    phase: str
    dimensions: int = 2
    grid_size: int
    cells_generated: int
    cells_total: int
    cells: list[CellStatus]
    seed_cells: dict[str, list[CellStatus]] | None = None
    seeds: list[int] = []
    prompt_a: str
    prompt_b: str
    prompt_c: str
    prompt_d: str = ""
    heatmap_url: str | None = None
    overlay_url: str | None = None
    cluster_url: str | None = None
    image_grid_url: str | None = None
    ridge_mesh_url: str | None = None  # 3D: marching cubes mesh as JSON


class FastScanRequest(BaseModel):
    prompt_a: str
    prompt_b: str
    prompt_c: str = ""
    prompt_d: str = ""  # 4th prompt for 3D mode
    dimensions: int = 2  # 2 or 3
    grid_size: int = 50
    seed: int = 42
    height: int = 256
    width: int = 256
    guidance_scale: float = 1.0  # 1.0 = single forward pass (no CFG), fastest


class FastScanResponse(BaseModel):
    job_id: str
    total_cells: int
    status: str


class GenerateSelectedRequest(BaseModel):
    tau: float = 1.5
    height: int = 256
    width: int = 256
    steps: int = 4
    guidance_scale: float = 4.0


class GenerateSelectedResponse(BaseModel):
    status: str
    total_cells: int = 0


class HealthResponse(BaseModel):
    status: str
    n_gpus: int
    workers_ready: int
    model: str
