# backend/app/utils/thumbnail_cache.py
import os
import time
import hashlib
import asyncio
from typing import Optional
from pathlib import Path

try:
    import aioredis
except Exception:
    aioredis = None  # optional

# Configuration
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "/app/backend/data/snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Simple in-process LRU cache (small)
class SimpleLRU:
    def __init__(self, max_items=256):
        self.max_items = max_items
        self.cache = {}
        self.access = []

    def get(self, key):
        item = self.cache.get(key)
        if item is not None:
            # move to end (most recent)
            try:
                self.access.remove(key)
            except ValueError:
                pass
            self.access.append(key)
        return item

    def set(self, key, value):
        if key in self.cache:
            try:
                self.access.remove(key)
            except ValueError:
                pass
        elif len(self.cache) >= self.max_items:
            # evict oldest
            old = self.access.pop(0)
            self.cache.pop(old, None)
        self.cache[key] = value
        self.access.append(key)

_lru = SimpleLRU(max_items=512)
_redis = None
_redis_initialized = False

async def _maybe_init_redis():
    global _redis, _redis_initialized
    if _redis_initialized:
        return
    _redis_initialized = True
    if aioredis is None:
        return
    try:
        _redis = await aioredis.from_url(REDIS_URL)
    except Exception:
        _redis = None

def _snapshot_disk_path(cam_id: str) -> str:
    h = hashlib.sha1(cam_id.encode("utf-8")).hexdigest()
    return os.path.join(SNAPSHOT_DIR, f"{h}.jpg")

async def get_snapshot(cam_id: str) -> Optional[bytes]:
    """
    Try redis -> in-memory -> disk
    Returns raw bytes or None.
    """
    await _maybe_init_redis()
    key = f"snapshot:{cam_id}"
    # try redis
    if _redis:
        try:
            data = await _redis.get(key)
            if data:
                return data
        except Exception:
            pass
    # try memory
    mem = _lru.get(key)
    if mem:
        return mem
    # try disk
    path = _snapshot_disk_path(cam_id)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = f.read()
            _lru.set(key, data)
            return data
        except Exception:
            return None
    return None

async def set_snapshot(cam_id: str, data: bytes):
    await _maybe_init_redis()
    key = f"snapshot:{cam_id}"
    # set redis
    if _redis:
        try:
            await _redis.set(key, data, ex=30)  # expire after 30s by default
        except Exception:
            pass
    # set memory
    _lru.set(key, data)
    # write disk (best-effort)
    path = _snapshot_disk_path(cam_id)
    try:
        with open(path, "wb") as f:
            f.write(data)
    except Exception:
        pass
