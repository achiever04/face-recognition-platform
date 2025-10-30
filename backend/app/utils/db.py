# backend/app/utils/db.py
"""
Database utilities for the Face Recognition Platform.

Goals and features:
- Robust MongoDB connection with retries and sensible defaults.
- Safe encryption key handling for embeddings (Fernet).
- Safe file-based logging helpers (append JSON/text).
- Convenience helpers to get DB/collection objects, with optional persistence hooks.
- Backwards-compatible API: all previous function names/signatures retained.
"""

import os
import time
import json
import base64
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

from pymongo import MongoClient, ASCENDING, DESCENDING, errors as pymongo_errors
from bson import ObjectId

# Optional guarded imports
try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None  # will fail later if encryption is required

# dotenv load (safe to call again)
from dotenv import load_dotenv

load_dotenv()  # best-effort; existing code did this as well

# Setup module logger (keeps prior prints but routes through logging)
logger = logging.getLogger("app.utils.db")
if not logger.handlers:
    # Basic console handler as default; your main app logging may reconfigure this
    ch = logging.StreamHandler()
    formatter = logging.Formatter("[DB] %(asctime)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
logger.setLevel(os.getenv("DB_LOG_LEVEL", "INFO").upper())

# -------------------------------
# MongoDB connection (singleton)
# -------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "face_recognition_db")
_MONGO_CONNECT_RETRIES = int(os.getenv("MONGO_CONNECT_RETRIES", "3"))
_MONGO_CONNECT_BACKOFF = float(os.getenv("MONGO_CONNECT_BACKOFF", "2.0"))  # seconds base

_client: Optional[MongoClient] = None


def get_mongo_client(max_retries: int = _MONGO_CONNECT_RETRIES) -> MongoClient:
    """
    Return a global MongoClient, connecting with retries.
    This function reuses a module-level client to avoid creating multiple clients.
    """
    global _client
    if _client is not None:
        return _client

    logger.info("Connecting to MongoDB: %s", MONGO_URI.split("@")[-1])
    attempt = 0
    last_exc = None
    while attempt < max_retries:
        attempt += 1
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            # Ping to verify connection
            client.admin.command("ping")
            _client = client
            logger.info("MongoDB connection successful (db=%s)", MONGO_DB_NAME)
            # Ensure indexes are created (idempotent)
            ensure_indexes(client[MONGO_DB_NAME])
            return _client
        except Exception as e:
            last_exc = e
            logger.warning(
                "MongoDB connection attempt %d/%d failed: %s",
                attempt,
                max_retries,
                str(e),
            )
            if attempt < max_retries:
                sleep_time = _MONGO_CONNECT_BACKOFF * attempt
                logger.info("Retrying in %.1f seconds...", sleep_time)
                time.sleep(sleep_time)
    logger.critical("MongoDB connection failed after %d attempts: %s", max_retries, last_exc)
    raise last_exc


def close_mongo_client():
    """Close and drop the module-level Mongo client (useful on shutdown)."""
    global _client
    try:
        if _client:
            _client.close()
            _client = None
            logger.info("MongoDB client closed.")
    except Exception:
        logger.exception("Error closing MongoDB client (ignored)")


def get_db():
    """Convenience: return the configured database handle (ensures connection)."""
    client = get_mongo_client()
    return client[MONGO_DB_NAME]


def get_collection(name: str):
    """Return a collection handle for the configured DB."""
    db = get_db()
    return db[name]


# -------------------------------
# Collections & directories (setup)
# -------------------------------
db = get_db()
faces_collection = db["faces"]
logs_collection = db["logs"]
deepfake_collection = db["deepfakes"]
tracking_collection = db["tracking"]
config_collection = db["config"]

# File-based logs paths (ensure they exist)
LOGS_DIR = os.getenv("LOGS_DIR", "logs")
DEEPFAKE_LOGS_DIR = os.getenv("DEEPFAKE_LOGS_DIR", "data/deepfake_logs")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DEEPFAKE_LOGS_DIR, exist_ok=True)

# -------------------------------
# Encryption key handling
# -------------------------------
ENCRYPTION_KEY_PATH = os.getenv("ENCRYPTION_KEY_PATH", "data/.encryption_key")
DISABLE_ENCRYPTION = os.getenv("DISABLE_ENCRYPTION", "false").lower() in ("1", "true", "yes")
KEY_FILE = Path(ENCRYPTION_KEY_PATH)
KEY_FILE.parent.mkdir(parents=True, exist_ok=True)

ENCRYPTION_KEY: Optional[bytes] = None
fernet: Optional["Fernet"] = None  # type: ignore

if DISABLE_ENCRYPTION:
    logger.info("Embedding encryption disabled via DISABLE_ENCRYPTION=true")
else:
    if KEY_FILE.exists():
        try:
            ENCRYPTION_KEY = KEY_FILE.read_bytes()
            logger.info("Loaded existing encryption key from %s", KEY_FILE.resolve())
        except Exception:
            logger.exception("Failed to read encryption key file - attempting to generate a new key")
            ENCRYPTION_KEY = None

    if ENCRYPTION_KEY is None:
        # Generate and save key (safe file permissions)
        if Fernet is None:
            logger.critical("cryptography.Fernet not available but encryption is required. Set DISABLE_ENCRYPTION=true to continue.")
            raise RuntimeError("cryptography is required for embedding encryption")
        ENCRYPTION_KEY = Fernet.generate_key()
        try:
            KEY_FILE.write_bytes(ENCRYPTION_KEY)
            KEY_FILE.chmod(0o600)
            logger.info("Generated and stored new encryption key at %s", KEY_FILE.resolve())
        except Exception:
            logger.exception("Failed to persist generated encryption key (attempting in-memory only)")

    # Create Fernet instance
    try:
        if ENCRYPTION_KEY:
            fernet = Fernet(ENCRYPTION_KEY)  # type: ignore
    except Exception as e:
        logger.exception("Failed to initialize Fernet encryption: %s", e)
        raise SystemExit("Invalid encryption key - check ENCRYPTION_KEY_PATH or set DISABLE_ENCRYPTION=true")

# -------------------------------
# Utility helpers
# -------------------------------
def iso_now() -> str:
    """Return timezone-naive ISO string for now (consistent format used in DB entries)."""
    return datetime.utcnow().isoformat()  # UTC timestamps

def json_serialize(doc: dict) -> dict:
    """Convert MongoDB ObjectId to strings and handle nested docs safely."""
    if not doc:
        return {}
    new_doc = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            new_doc[k] = str(v)
        else:
            try:
                json.dumps(v)
                new_doc[k] = v
            except TypeError:
                # fallback: convert to str
                new_doc[k] = str(v)
    return new_doc

# -------------------------------
# Embedding encryption helpers (backwards-compatible)
# -------------------------------
def encrypt_embedding(embedding: List[float]) -> str:
    """
    Encrypt a face embedding and return base64 encoded encrypted blob.
    If encryption is disabled, return JSON string of the embedding.
    """
    try:
        if DISABLE_ENCRYPTION or fernet is None:
            return json.dumps(embedding)
        raw = json.dumps(embedding).encode("utf-8")
        encrypted = fernet.encrypt(raw)  # type: ignore
        return base64.b64encode(encrypted).decode("utf-8")
    except Exception:
        logger.exception("encrypt_embedding failed - returning plaintext fallback")
        return json.dumps(embedding)


def decrypt_embedding(encrypted_str: str) -> List[float]:
    """
    Decrypt a stored embedding (base64 encrypted) or parse plaintext JSON if encryption disabled.
    Returns list of floats (or empty list on error).
    """
    try:
        if DISABLE_ENCRYPTION or fernet is None:
            return json.loads(encrypted_str)
        encrypted = base64.b64decode(encrypted_str)
        decrypted = fernet.decrypt(encrypted)  # type: ignore
        return json.loads(decrypted.decode("utf-8"))
    except Exception:
        logger.exception("decrypt_embedding failed for an entry (returning empty list)")
        return []

# -------------------------------
# Index creation (idempotent)
# -------------------------------
def ensure_indexes(db_handle):
    """Create indexes in an idempotent way."""
    try:
        db_handle["logs"].create_index([("target", ASCENDING), ("camera_id", ASCENDING), ("timestamp", ASCENDING)])
        db_handle["deepfakes"].create_index([("camera_id", ASCENDING), ("timestamp", ASCENDING)])
        db_handle["tracking"].create_index([("person", ASCENDING), ("timestamp", DESCENDING)])
        db_handle["config"].create_index("name", unique=True)
        logger.info("Ensured database indexes.")
    except Exception:
        logger.exception("Failed to ensure indexes (non-fatal)")

# call ensure_indexes once (safe)
try:
    ensure_indexes(db)
except Exception:
    logger.exception("Index creation encountered an error at startup")

# -------------------------------
# File logging helpers (robust)
# -------------------------------
def append_log_text(target: str, message: str):
    filename = os.path.join(LOGS_DIR, f"{target}.txt")
    try:
        # Atomic append
        with open(filename, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        logger.exception("Error writing text log for %s", target)


def append_log_json(target: str, entry: dict):
    filename = os.path.join(LOGS_DIR, f"{target}.json")
    try:
        existing = []
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
            except Exception:
                logger.warning("Could not decode existing JSON log for %s - starting fresh", target)
                existing = []

        existing.append(json_serialize(entry))
        # Write atomically
        tmpf = filename + ".tmp"
        with open(tmpf, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        os.replace(tmpf, filename)
    except Exception:
        logger.exception("Error writing JSON log for %s", target)


def create_target_log_files(target: str):
    txt_file = os.path.join(LOGS_DIR, f"{target}.txt")
    json_file = os.path.join(LOGS_DIR, f"{target}.json")
    try:
        if not os.path.exists(txt_file):
            with open(txt_file, "w", encoding="utf-8"):
                pass
        if not os.path.exists(json_file):
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)
    except Exception:
        logger.exception("Error creating log files for %s", target)

# -------------------------------
# DB insert wrapper with retries (used by log functions)
# -------------------------------
def _safe_insert(collection, doc: dict, max_retries: int = 2) -> Optional[Any]:
    for attempt in range(1, max_retries + 1):
        try:
            res = collection.insert_one(doc)
            return res
        except pymongo_errors.AutoReconnect as e:
            logger.warning("AutoReconnect on insert, attempt %d/%d: %s", attempt, max_retries, e)
            time.sleep(0.5 * attempt)
        except Exception:
            logger.exception("Insert failed unexpectedly (non-fatal)")
            break
    return None

# -------------------------------
# Log recognition event (Alerts) - original behavior preserved
# -------------------------------
def log_alert(camera_id: int, camera_name: str, geo: str, target: str, distance: float, cooldown: int = 10):
    now = datetime.utcnow()
    cutoff_time = now - timedelta(seconds=cooldown)
    try:
        duplicate = logs_collection.find_one({
            "target": target,
            "camera_id": camera_id,
            "timestamp": {"$gte": cutoff_time.isoformat()}
        })
        if duplicate:
            return False

        log_entry = {
            "camera_id": camera_id,
            "camera_name": camera_name,
            "geo": geo,
            "target": target,
            "distance": distance,
            "timestamp": now.isoformat()
        }

        res = _safe_insert(logs_collection, log_entry)
        if res is None:
            logger.warning("MongoDB insert returned None for alert log (falling back to file logs)")

        # File logs
        text_message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Detected at Camera {camera_id} ({camera_name}, Geo={geo}), distance={distance:.2f}"
        append_log_text(target, text_message)
        append_log_json(target, json_serialize(log_entry))

        # --- ADDITION: Emit audit log (non-fatal) ---
        try:
            from app.utils.logger import audit_event  # local import to avoid cycles
            audit_payload = {
                "camera_id": camera_id,
                "camera_name": camera_name,
                "geo": geo,
                "target": target,
                "distance": distance,
                "timestamp": now.isoformat()
            }
            # Use audit_event (it redacts sensitive fields by default)
            audit_event("detection", audit_payload, persist_to_db=False)
        except Exception:
            logger.exception("audit_event call in log_alert failed (ignored)")

        return True
    except Exception:
        logger.exception("Failed to insert alert log (returning False)")
        return False


# -------------------------------
# Log DeepFake Event (preserves original behavior)
# -------------------------------
def log_deepfake(camera_id: int, camera_name: str, geo: str, detection: dict, frame_id: int):
    now = datetime.utcnow()
    log_entry = {
        "camera_id": camera_id,
        "camera_name": camera_name,
        "geo": geo,
        "frame_id": frame_id,
        "is_fake": detection.get("is_fake"),
        "confidence": detection.get("confidence"),
        "bbox": detection.get("bbox"),
        "timestamp": now.isoformat()
    }
    try:
        res = _safe_insert(deepfake_collection, log_entry)
        # Persist to deepfake file logs as well
        log_file = os.path.join(DEEPFAKE_LOGS_DIR, "deepfake_events.json")
        data = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                logger.warning("Could not decode deepfake log file - starting fresh")
                data = []
        data.append(json_serialize(log_entry))
        tmpf = log_file + ".tmp"
        with open(tmpf, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmpf, log_file)

        # --- ADDITION: Emit audit log for deepfake event ---
        try:
            from app.utils.logger import audit_event  # local import to avoid cycles
            audit_payload = {
                "camera_id": camera_id,
                "camera_name": camera_name,
                "geo": geo,
                "frame_id": frame_id,
                "detection": {
                    "is_fake": detection.get("is_fake"),
                    "confidence": detection.get("confidence"),
                    "bbox": detection.get("bbox")
                },
                "timestamp": now.isoformat()
            }
            audit_event("deepfake", audit_payload, persist_to_db=False)
        except Exception:
            logger.exception("audit_event call in log_deepfake failed (ignored)")

        return True
    except Exception:
        logger.exception("Failed to log deepfake event")
        return False


# -------------------------------
# Store / Retrieve Embeddings (preserve API)
# -------------------------------
def store_embedding(target: str, embedding: List[float]) -> bool:
    try:
        encrypted = encrypt_embedding(embedding)
        faces_collection.update_one(
            {"target": target},
            {"$set": {"embedding": encrypted, "updated_at": iso_now()}},
            upsert=True
        )
        return True
    except Exception:
        logger.exception("Failed to store embedding for %s", target)
        return False


def retrieve_embedding(target: str) -> List[float]:
    try:
        doc = faces_collection.find_one({"target": target})
        if doc and "embedding" in doc:
            return decrypt_embedding(doc["embedding"])
        return []
    except Exception:
        logger.exception("Failed to retrieve embedding for %s", target)
        return []

def retrieve_all_embeddings() -> List[Dict[str, Any]]:
    try:
        cursor = faces_collection.find({}, {"_id": 0, "target": 1, "embedding": 1})
        return list(cursor)
    except Exception:
        logger.exception("Failed to retrieve all embeddings")
        return []

# -------------------------------
# Watchlist & Geofence persistence (preserve API)
# -------------------------------
def save_watchlist_db(watchlist: List[str]):
    try:
        config_collection.update_one(
            {"name": "watchlist"},
            {"$set": {"data": {"items": watchlist}, "updated_at": iso_now()}},
            upsert=True
        )
    except Exception:
        logger.exception("Failed to save watchlist")

def load_watchlist_db() -> List[str]:
    try:
        doc = config_collection.find_one({"name": "watchlist"})
        return doc.get("data", {}).get("items", []) if doc else []
    except Exception:
        logger.exception("Failed to load watchlist")
        return []

def save_geofence_db(geofences: Dict[str, Any]):
    try:
        config_collection.update_one(
            {"name": "geofences"},
            {"$set": {"data": {"zones": geofences}, "updated_at": iso_now()}},
            upsert=True
        )
    except Exception:
        logger.exception("Failed to save geofences")

def load_geofences_db() -> Dict[str, Any]:
    try:
        doc = config_collection.find_one({"name": "geofences"})
        return doc.get("data", {}).get("zones", {}) if doc else {}
    except Exception:
        logger.exception("Failed to load geofences")
        return {}

# -------------------------------
# Tracking persistence (preserve API)
# -------------------------------
def save_detection_to_db(detection: Dict[str, Any]):
    try:
        if isinstance(detection.get("geo"), tuple):
            detection["geo"] = list(detection["geo"])
        # augment timestamp if missing
        if "timestamp" not in detection:
            detection["timestamp"] = iso_now()
        _safe_insert(tracking_collection, detection)
    except Exception:
        logger.exception("Failed to save tracking detection")

def load_tracking_history_db(limit_per_person: int = 100) -> Dict[str, List[Dict[str, Any]]]:
    try:
        pipeline = [
            {"$sort": {"timestamp": DESCENDING}},
            {"$group": {"_id": "$person", "history": {"$push": "$$ROOT"}}},
            {"$project": {"_id": 1, "history": {"$slice": ["$history", limit_per_person]}}}
        ]
        result_cursor = tracking_collection.aggregate(pipeline)
        all_history = {}
        for doc in result_cursor:
            person = doc["_id"]
            history_list = [json_serialize(item) for item in reversed(doc.get("history", []))]
            all_history[person] = history_list
        return all_history
    except Exception:
        logger.exception("Failed to load tracking history")
        return {}

def clear_history_in_db(person_name: Optional[str] = None):
    try:
        if person_name:
            logger.info("Deleting tracking history for %s", person_name)
            result = tracking_collection.delete_many({"person": person_name})
            logger.info("Deleted %d tracking records for %s", result.deleted_count, person_name)
        else:
            logger.info("Deleting ALL tracking history")
            result = tracking_collection.delete_many({})
            logger.info("Deleted %d total tracking records", result.deleted_count)
    except Exception:
        logger.exception("Failed to clear tracking history")
        raise

# -------------------------------
# Graceful shutdown helper
# -------------------------------
def shutdown():
    """Call at application shutdown to close DB client."""
    try:
        close_mongo_client()
    except Exception:
        logger.exception("Error during DB shutdown (ignored)")

# -------------------------------
# End of module
# -------------------------------
