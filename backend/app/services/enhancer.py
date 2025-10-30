# backend/app/services/enhancer.py
"""
Lightweight image enhancer for camera snapshots.

This module provides a safe, CPU-friendly enhancement pipeline intended
for dev/demo use on CPU-only machines (e.g., laptops). It performs:
  - optional bicubic upscaling
  - optional mild sharpening
  - re-encoding to JPEG with configurable quality

Design goals:
- Minimal dependencies (Pillow). If Pillow is missing, enhancer is a no-op.
- Runs heavy work in threads via asyncio.to_thread to avoid blocking the event loop.
- Writes enhanced image back to thumbnail cache (set_snapshot) for fast subsequent reads.
- Configurable via environment variables.
"""

from __future__ import annotations

import os
import io
import asyncio
from typing import Optional
from pathlib import Path

from app.utils.logger import get_logger
logger = get_logger("app.services.enhancer")

# Config (tweak via env)
UPSCALE_FACTOR = float(os.getenv("ENHANCER_UPSCALE_FACTOR", "2"))
MAX_PIXELS = int(os.getenv("ENHANCER_MAX_PIXELS", str(4_000_000)))  # avoid huge images
JPEG_QUALITY = int(os.getenv("ENHANCER_JPEG_QUALITY", "85"))
SHARPEN = os.getenv("ENHANCER_SHARPEN", "true").lower() in ("1", "true", "yes")

# Try import Pillow (PIL)
try:
    from PIL import Image, ImageFilter
except Exception:
    Image = None
    ImageFilter = None

# Import thumbnail cache hooks
try:
    from app.utils.thumbnail_cache import get_snapshot, set_snapshot
except Exception:
    get_snapshot = None
    set_snapshot = None

def _safe_resize_params(width: int, height: int, upscale_factor: float):
    """
    Given original width/height and upscale factor, compute new size while
    enforcing MAX_PIXELS. Returns (new_w, new_h).
    """
    new_w = int(round(width * upscale_factor))
    new_h = int(round(height * upscale_factor))
    if new_w * new_h > MAX_PIXELS:
        # scale down upscale to meet MAX_PIXELS constraint
        scale = (MAX_PIXELS / (width * height)) ** 0.5
        # if scale < 1 then we cannot upscale to requested factor; use scale
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
    return new_w, new_h

def _do_enhance_bytes(src_bytes: bytes) -> Optional[bytes]:
    """
    Synchronous worker that performs enhancement on raw image bytes.
    Returns enhanced JPEG bytes or None on failure.
    """
    if Image is None:
        logger.warning("Pillow (PIL) not installed — enhancer is disabled")
        return None
    try:
        with Image.open(io.BytesIO(src_bytes)) as im:
            # Convert to RGB if needed
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")
            # Determine resize/upscale
            w, h = im.size
            new_w, new_h = _safe_resize_params(w, h, UPSCALE_FACTOR)
            if new_w > w or new_h > h:
                # upscale using bicubic interpolation
                im = im.resize((new_w, new_h), resample=Image.BICUBIC)
            # Optional sharpen
            if SHARPEN and ImageFilter is not None:
                im = im.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
            # Encode to JPEG bytes
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
            return out.getvalue()
    except Exception as e:
        logger.exception("Enhancer worker failed: %s", e)
        return None

async def enhance_snapshot(cam_id: str, src_path: Optional[Path | str] = None) -> Optional[bytes]:
    """
    Enhance the snapshot for 'cam_id'.

    - src_path: optional Path or str pointing to a source image file (preferred for disk-based enhancement).
                If not provided, the function will attempt to read bytes from get_snapshot(cam_id).
    - Returns enhanced bytes on success (and writes them to thumbnail cache), otherwise returns None.

    This function is async-friendly: the CPU-bound work runs inside asyncio.to_thread.
    """
    # If Pillow not available or thumbnail cache not present, no-op
    if Image is None:
        logger.info("enhance_snapshot skipped: Pillow not available")
        return None
    if get_snapshot is None or set_snapshot is None:
        logger.info("enhance_snapshot skipped: thumbnail_cache not available")
        return None

    try:
        src_bytes = None
        # Prefer reading from provided src_path (if exists)
        if src_path:
            try:
                p = Path(src_path)
                if p.exists():
                    # read in thread to avoid blocking
                    def _read_file():
                        with p.open("rb") as f:
                            return f.read()
                    src_bytes = await asyncio.to_thread(_read_file)
            except Exception:
                logger.exception("Failed to read src_path %s for enhancer (falling back to cache)", src_path)

        # fallback to cache
        if src_bytes is None:
            try:
                src_bytes = await get_snapshot(cam_id)
            except Exception:
                # in case get_snapshot is blocking or errors, run in thread
                try:
                    src_bytes = await asyncio.to_thread(lambda: None)
                except Exception:
                    src_bytes = None

        if not src_bytes:
            logger.debug("No source bytes available to enhance for cam %s", cam_id)
            return None

        # run enhancement in thread
        enhanced = await asyncio.to_thread(_do_enhance_bytes, src_bytes)
        if not enhanced:
            logger.debug("Enhancer produced no output for cam %s", cam_id)
            return None

        # Write back into thumbnail cache (best-effort)
        try:
            await set_snapshot(cam_id, enhanced)
        except Exception:
            # set_snapshot may be synchronous; try thread fallback
            try:
                await asyncio.to_thread(lambda: set_snapshot(cam_id, enhanced))
            except Exception:
                logger.exception("Failed to write enhanced snapshot to cache for cam %s", cam_id)

        logger.info("Enhanced snapshot saved for cam %s (size=%d bytes)", cam_id, len(enhanced))
        return enhanced
    except Exception as e:
        logger.exception("enhance_snapshot failed for cam %s: %s", cam_id, e)
        return None
