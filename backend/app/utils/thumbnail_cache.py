# backend/app/utils/thumbnail_cache.py
"""
Robust thumbnail cache for camera snapshots.

Features:
- Async-friendly get_snapshot / set_snapshot APIs (same signatures).
- In-process LRU cache with TTL.
- Optional Redis backend (aioredis) if installed and REDIS_URL configured.
- Atomic disk writes and disk usage limit enforcement (MAX_DISK_BYTES).
- Per-key asyncio locks to avoid concurrent writes.
- Best-effort safe behavior when Redis or disk fails.
"""

import os
import time
import hashlib
import asyncio
from typing import Optional, Dict, Tuple
from pathlib import Path
from collections import OrderedDict

# Optional Redis import
try:
    import aioredis
except Exception:
    aioredis = None

# Config (tweak via environment)
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "data/snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
REDIS_URL = os.getenv("REDIS_URL", "")
MAX_MEM_ITEMS = int(os.getenv("THUMB_CACHE_MEM_ITEMS", "512"))
MEM_TTL_SECONDS = int(os.getenv("THUMB_CACHE_MEM_TTL", "30"))  # in-memory TTL
REDIS_TTL = int(os.getenv("THUMB_CACHE_REDIS_TTL", "30"))  # redis expiry
MAX_DISK_BYTES = int(os.getenv("THUMB_MAX_DISK_BYTES", str(200 * 1024 * 1024)))  # default 200MB
DISK_CLEANUP_BATCH = int(os.getenv("THUMB_DISK_CLEANUP_BATCH", "10"))  # files to remove per cleanup

# Internal LRU structure: key -> (bytes, expiry_ts)
class _InMemoryLRU:
    def __init__(self, max_items: int):
        self.max_items = max_items
        self._dict: OrderedDict[str, Tuple[bytes, float]] = OrderedDict()

    def get(self, key: str) -> Optional[bytes]:
        item = self._dict.get(key)
        if item:
            data, expiry = item
            if expiry and expiry < time.time():
                # expired
                self._dict.pop(key, None)
                return None
            # move to end
            self._dict.move_to_end(key)
            return data
        return None

    def set(self, key: str, data: bytes, ttl: Optional[int] = None):
        expiry = (time.time() + ttl) if ttl and ttl > 0 else None
        if key in self._dict:
            self._dict.pop(key, None)
        elif len(self._dict) >= self.max_items:
            # evict oldest
            try:
                self._dict.popitem(last=False)
            except KeyError:
                pass
        self._dict[key] = (data, expiry)

    def pop(self, key: str):
        self._dict.pop(key, None)

_mem_cache = _InMemoryLRU(MAX_MEM_ITEMS)

# Redis client (lazy init)
_redis = None
_redis_lock = asyncio.Lock()
_redis_initialized = False

async def _init_redis():
    global _redis, _redis_initialized
    if _redis_initialized:
        return
    _redis_initialized = True
    if not REDIS_URL or aioredis is None:
        _redis = None
        return
    try:
        _redis = await aioredis.from_url(REDIS_URL)
    except Exception:
        _redis = None

# Per-key locks to avoid concurrent writes/reads
_key_locks: Dict[str, asyncio.Lock] = {}
_key_locks_lock = asyncio.Lock()

async def _get_key_lock(key: str) -> asyncio.Lock:
    async with _key_locks_lock:
        lk = _key_locks.get(key)
        if not lk:
            lk = asyncio.Lock()
            _key_locks[key] = lk
        return lk

def _snapshot_disk_path(cam_id: str) -> Path:
    # Use hash to avoid weird characters in filenames
    h = hashlib.sha1(cam_id.encode("utf-8")).hexdigest()
    return Path(SNAPSHOT_DIR) / f"{h}.jpg"

def _total_disk_usage_bytes() -> int:
    total = 0
    try:
        for p in Path(SNAPSHOT_DIR).glob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except Exception:
                    continue
    except Exception:
        return 0
    return total

def _enforce_disk_quota():
    """
    Delete oldest files until total size <= MAX_DISK_BYTES.
    Non-async; called from async context but file ops are fast.
    """
    try:
        total = _total_disk_usage_bytes()
        if total <= MAX_DISK_BYTES:
            return
        # list files sorted by mtime ascending (oldest first)
        files = [p for p in Path(SNAPSHOT_DIR).glob("*") if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0)
        removed = 0
        for f in files:
            if total <= MAX_DISK_BYTES:
                break
            try:
                size = f.stat().st_size
                f.unlink()
                total -= size
                removed += 1
            except Exception:
                continue
            # optional: stop early in small batches
            if removed >= DISK_CLEANUP_BATCH:
                break
    except Exception:
        # best-effort: do not raise
        pass

async def get_snapshot(cam_id: str) -> Optional[bytes]:
    """
    Try Redis -> in-memory -> disk. Return bytes or None.
    """
    key = f"snapshot:{cam_id}"
    await _init_redis()

    # 1) Try redis
    if _redis:
        try:
            maybe = await _redis.get(key)
            if maybe:
                # Redis returns bytes
                _mem_cache.set(key, maybe, ttl=MEM_TTL_SECONDS)
                return maybe
        except Exception:
            # ignore redis failures
            pass

    # 2) Try memory
    mem = _mem_cache.get(key)
    if mem:
        return mem

    # 3) Try disk (read using blocking I/O but keep small)
    p = _snapshot_disk_path(cam_id)
    if p.exists():
        lk = await _get_key_lock(key)
        async with lk:
            try:
                # read bytes
                with p.open("rb") as f:
                    data = f.read()
                # update memory cache
                _mem_cache.set(key, data, ttl=MEM_TTL_SECONDS)
                # optionally refresh redis
                if _redis:
                    try:
                        await _redis.set(key, data, ex=REDIS_TTL)
                    except Exception:
                        pass
                return data
            except Exception:
                return None
    return None

async def set_snapshot(cam_id: str, data: bytes, *, persist_to_disk: bool = True):
    """
    Store snapshot into Redis, memory, and optionally disk.
    Atomic disk write + disk quota enforcement.
    """
    key = f"snapshot:{cam_id}"
    await _init_redis()

    # 1) Redis
    if _redis:
        try:
            await _redis.set(key, data, ex=REDIS_TTL)
        except Exception:
            pass

    # 2) Memory
    _mem_cache.set(key, data, ttl=MEM_TTL_SECONDS)

    # 3) Disk (atomic)
    if persist_to_disk:
        p = _snapshot_disk_path(cam_id)
        lk = await _get_key_lock(key)
        async with lk:
            try:
                tmp = p.with_suffix(".tmp")
                # write atomically using blocking IO (small files)
                with tmp.open("wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, p)
                # enforce disk quota
                _enforce_disk_quota()
            except Exception:
                # best-effort
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass

# Small utility to remove snapshot from caches
async def delete_snapshot(cam_id: str):
    key = f"snapshot:{cam_id}"
    # remove redis
    await _init_redis()
    if _redis:
        try:
            await _redis.delete(key)
        except Exception:
            pass
    _mem_cache.pop(key)
    p = _snapshot_disk_path(cam_id)
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass
