"""
Disk-based thumbnail cache keyed by content hash.
"""
import hashlib
from collections.abc import MutableMapping
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


class ThumbnailStore(MutableMapping):
    """A job's cell-key -> JPEG-bytes mapping that keeps only the hash resident.

    Every rendered image is already written to the on-disk cache, so holding a second
    full-resolution copy per cell in app.state.jobs was pure duplication — a 100x100
    grid of 256 px renders is on the order of a gigabyte of process memory that nothing
    ever freed, and app.state.jobs is only cleared by /cancel. Behaves as a plain dict
    of bytes for every existing caller (`.get`, `in`, iteration, assignment); assigning
    bytes hashes and persists them, and reads come back off disk.
    """

    def __init__(self, cache: ThumbnailCache | None = None, hashes: dict | None = None):
        self._cache = cache or ThumbnailCache()
        self._hashes = dict(hashes or {})

    def __setitem__(self, key, value):
        if isinstance(value, str):          # already-cached hash
            self._hashes[key] = value
            return
        h = hashlib.md5(value).hexdigest()
        if not self._cache.has(h):
            self._cache.save(h, value)
        self._hashes[key] = h

    def __getitem__(self, key) -> bytes:
        path = self._cache.get_path(self._hashes[key])
        if path is None:
            # the cache file was removed under us; treat as absent rather than
            # returning None, which every caller would paint as a black tile
            raise KeyError(key)
        return path.read_bytes()

    def __delitem__(self, key):
        del self._hashes[key]

    def __iter__(self):
        return iter(self._hashes)

    def __len__(self):
        return len(self._hashes)

    @property
    def hashes(self) -> dict:
        return self._hashes
