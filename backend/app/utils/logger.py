# backend/app/utils/logger.py
"""
Advanced logging utilities for the Face Recognition Platform.

Features:
- Idempotent setup_logger() that configures console + rotating file handlers.
- Separate audit logger (append-only, strict permissions) for detection/alert events.
- Optional JSON-line logging for production via LOG_FORMAT_JSON env var.
- Redaction helper to avoid logging embeddings/images/raw binaries.
- audit_event() convenience wrapper to emit audit logs and optionally persist to DB.
- get_logger(name) helper for consistent logger retrieval.

Environment variables (defaults):
- LOG_DIR=logs
- LOG_LEVEL=INFO
- LOG_FORMAT_JSON=false
- LOG_MAX_BYTES=10485760 (10MB)
- LOG_BACKUP_COUNT=5
- AUDIT_LOG_FILE=audit.log
- AUDIT_TO_DB=false  (attempt to save audit events to Mongo via app.utils.db.get_db())
"""
from __future__ import annotations

import os
import sys
import json
import stat
import logging
from logging import Logger
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Any, Dict, Iterable, Optional

# Optional DB hook for audit persistence (non-fatal if unavailable)
try:
    from app.utils import db as db_utils  # type: ignore
    _HAS_DB = True
except Exception:
    db_utils = None
    _HAS_DB = False

# Environment-configurable settings
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT_JSON = os.getenv("LOG_FORMAT_JSON", "false").lower() in ("1", "true", "yes")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB default
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
AUDIT_LOG_FILE = os.path.join(LOG_DIR, os.getenv("AUDIT_LOG_FILE", "audit.log"))
APP_LOG_FILE = os.path.join(LOG_DIR, os.getenv("APP_LOG_FILE", "app.log"))
AUDIT_TO_DB = os.getenv("AUDIT_TO_DB", "false").lower() in ("1", "true", "yes")

# Sensitive keys that should be redacted in logs
_DEFAULT_REDACT_KEYS = {"embedding", "image", "raw_image", "face_image", "frame", "bytes"}

# Ensure log files exist with safe permissions
def _ensure_file_with_mode(path: str, mode: int = 0o600):
    try:
        if not os.path.exists(path):
            # create empty file safely
            with open(path, "w", encoding="utf-8"):
                pass
        # set restrictive permissions
        os.chmod(path, mode)
    except Exception:
        # Best-effort; do not crash the app if chmod fails on some filesystems
        pass

_ensure_file_with_mode(APP_LOG_FILE)
_ensure_file_with_mode(AUDIT_LOG_FILE)

# JSON formatter (simple)
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # include extra fields if present
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        # include exception info if present
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

# Plain text formatter fallback
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Internal state to avoid duplicate handler registration
_LOG_SETUP_DONE = False

def redact_sensitive(obj: Any, keys: Optional[Iterable[str]] = None) -> Any:
    """
    Recursively redact values for keys found in dicts (useful for logs).
    Returns a copy with sensitive fields replaced by "<REDACTED>".
    """
    if keys is None:
        keys = _DEFAULT_REDACT_KEYS
    try:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in keys:
                    out[k] = "<REDACTED>"
                else:
                    out[k] = redact_sensitive(v, keys)
            return out
        elif isinstance(obj, list):
            return [redact_sensitive(i, keys) for i in obj]
        else:
            # primitives
            return obj
    except Exception:
        # On unexpected types, return a safe string
        return "<REDACTED>"

def setup_logger(level: Optional[str | int] = None, *, json_format: Optional[bool] = None):
    """
    Configure root logger and audit logger.

    - level: overrides LOG_LEVEL if provided.
    - json_format: if True, use JSON-line logs (overrides LOG_FORMAT_JSON).
    Idempotent: safe to call multiple times.
    """
    global _LOG_SETUP_DONE
    if level is None:
        level = LOG_LEVEL
    if json_format is None:
        json_format = LOG_FORMAT_JSON

    root_logger = logging.getLogger()
    # Idempotent setup: clear only if we haven't configured yet or explicitly want reset
    if getattr(root_logger, "_frp_configured", False):
        # Already configured: ensure level is updated and return
        root_logger.setLevel(level)
        return

    # Remove any existing handlers to avoid duplicate lines on reload
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    if json_format:
        console.setFormatter(JSONFormatter())
    else:
        console.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATEFMT))
    root_logger.addHandler(console)

    # Rotating file handler for app logs
    try:
        file_handler = RotatingFileHandler(APP_LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATEFMT))
        root_logger.addHandler(file_handler)
    except Exception:
        # If file handler cannot be created (permissions/readonly fs), continue with console only
        pass

    # Configure audit logger: append-only file, one JSON object per line
    audit_logger = logging.getLogger("app.audit")
    # remove existing audit handlers
    for h in list(audit_logger.handlers):
        audit_logger.removeHandler(h)
    audit_logger.setLevel(logging.INFO)
    try:
        audit_fh = RotatingFileHandler(AUDIT_LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
        # audit logger should always write JSON lines for easier ingestion
        audit_fh.setFormatter(JSONFormatter())
        audit_logger.addHandler(audit_fh)
        # Prevent audit logs from propagating to root handlers (avoid duplicate output)
        audit_logger.propagate = False
    except Exception:
        # best-effort fallback: attach console only
        audit_logger.addHandler(console)
        audit_logger.propagate = False

    # Set root level
    root_logger.setLevel(level)

    # Silence verbose third-party loggers by default (can be overridden via env)
    logging.getLogger("uvicorn.access").setLevel(os.getenv("UVICORN_ACCESS_LOG_LEVEL", "WARNING"))
    logging.getLogger("socketio.server").setLevel(os.getenv("SOCKETIO_LOG_LEVEL", "WARNING"))
    logging.getLogger("engineio.server").setLevel(os.getenv("ENGINEIO_LOG_LEVEL", "WARNING"))

    # Mark configured
    root_logger._frp_configured = True
    _LOG_SETUP_DONE = True

def get_logger(name: Optional[str] = None) -> Logger:
    """
    Return a configured logger (root must be set up first via setup_logger()).
    Use get_logger(__name__) in modules.
    """
    if not _LOG_SETUP_DONE:
        # Try to auto-configure with defaults if user didn't call setup_logger
        try:
            setup_logger()
        except Exception:
            # fallback: basic config
            logging.basicConfig(level=LOG_LEVEL)
    return logging.getLogger(name)

def audit_event(event_type: str, payload: Dict[str, Any], redact_keys: Optional[Iterable[str]] = None, persist_to_db: Optional[bool] = None):
    """
    Emit an audit event.
    - event_type: short string (e.g., "detection", "alert", "deepfake")
    - payload: dict containing event details (will be JSON-serialized)
    - redact_keys: keys to redact (defaults to sensible set)
    - persist_to_db: if True attempt to save to Mongo 'logs'/'deepfakes' collections; default controlled by AUDIT_TO_DB env var

    This function is best-effort and will not raise if DB persistence fails.
    """
    if redact_keys is None:
        redact_keys = _DEFAULT_REDACT_KEYS

    safe_payload = redact_sensitive(payload, keys=set(redact_keys))
    entry = {
        "type": event_type,
        "payload": safe_payload,
        "timestamp": datetime_utc_iso(),
    }

    # Log to audit logger (one JSON object per line)
    audit_logger = get_logger("app.audit")
    try:
        # Using logger.info so message gets formatted by JSONFormatter
        # Attach entry as extra to allow formatters to pick it up
        audit_logger.info(json.dumps(entry, ensure_ascii=False), extra={"extra": {}})
    except Exception:
        # Fallback: log plain text
        try:
            audit_logger.info(str(entry))
        except Exception:
            pass

    # Optionally persist to DB (non-fatal)
    if persist_to_db is None:
        persist_to_db = AUDIT_TO_DB

    if persist_to_db and _HAS_DB:
        try:
            # Decide collection by event_type heuristics
            if event_type.lower() in ("deepfake", "deep_fake", "deep-fake"):
                coll = db_utils.get_collection("deepfakes")
            else:
                coll = db_utils.get_collection("logs")
            # Store the raw payload (non-redacted) for forensic purposes if needed.
            doc = {"event_type": event_type, "payload": payload, "timestamp": datetime_utc_iso()}
            coll.insert_one(doc)
        except Exception:
            # Do not escalate DB errors from logging
            get_logger(__name__).exception("Failed to persist audit_event to DB (ignored)")

def datetime_utc_iso():
    """Return current UTC time as ISO string (used in audit events)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

# Convenience: small test/demo function that shows a usage pattern
def _demo():
    setup_logger()
    lg = get_logger(__name__)
    lg.info("Logger demo - startup")
    audit_event("detection", {"camera": 1, "target": "John Doe", "embedding": [0.1, 0.2, 0.3]})

# Only run demo when executed directly (not on import)
if __name__ == "__main__":
    _demo()
