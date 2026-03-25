"""
LatentPoint storage — UMAP-ready collection of explored points.
"""
import threading
from dataclasses import dataclass, field


@dataclass
class LatentPoint:
    id: str
    job_id: str
    row: int
    col: int
    alpha: float
    beta: float
    sensitivity: float
    thumbnail_hash: str
    prompt_a: str = ""
    prompt_b: str = ""
    # Future UMAP fields
    umap_2d: tuple[float, float] | None = None
    umap_3d: tuple[float, float, float] | None = None


class PointStore:
    """Thread-safe collection of explored latent points."""

    def __init__(self):
        self._points: dict[str, LatentPoint] = {}
        self._lock = threading.Lock()

    def add(self, point: LatentPoint):
        with self._lock:
            self._points[point.id] = point

    def get(self, point_id: str) -> LatentPoint | None:
        with self._lock:
            return self._points.get(point_id)

    def get_by_job(self, job_id: str) -> list[LatentPoint]:
        with self._lock:
            return [p for p in self._points.values() if p.job_id == job_id]

    def count(self) -> int:
        with self._lock:
            return len(self._points)
