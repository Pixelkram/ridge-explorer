"""
Disk-based thumbnail cache keyed by content hash.
"""
from pathlib import Path
from backend import config


class ThumbnailCache:
    def __init__(self, cache_dir: Path = config.THUMBNAILS_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, hash_key: str) -> Path:
        bucket = hash_key[:2]
        return self.cache_dir / bucket / f"{hash_key}.jpg"

    def has(self, hash_key: str) -> bool:
        return self._path(hash_key).exists()

    def save(self, hash_key: str, data: bytes):
        path = self._path(hash_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get_path(self, hash_key: str) -> Path | None:
        path = self._path(hash_key)
        return path if path.exists() else None

    def url(self, hash_key: str) -> str:
        return f"/cache/thumbnails/{hash_key[:2]}/{hash_key}.jpg"
