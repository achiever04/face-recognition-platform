# --- NEW IMPORTS ---
import os
from dotenv import load_dotenv
# --- END NEW IMPORTS ---

from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime, timedelta
# import os # Already imported above
import json
import base64
from pathlib import Path
from cryptography.fernet import Fernet
from bson import ObjectId
from typing import List, Dict, Optional, Any

# --- NEW: Load environment variables from .env file ---
# This should be called early, before accessing environment variables.
load_dotenv()
print("[DB] Loaded environment variables from .env")

# -------------------------------
# MongoDB Connection with retry logic
# --- UPGRADED ---
# -------------------------------
def get_mongo_client(max_retries=3):
    """Create MongoDB client with connection retry, using .env variables."""
    # --- IMPROVEMENT: Read URI from environment variable, provide default ---
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    print(f"[DB] Connecting to MongoDB at: {mongo_uri.split('@')[-1]}") # Log URI without credentials

    for attempt in range(max_retries):
        try:
            client = MongoClient(
                mongo_uri, # Use the URI from .env
                serverSelectionTimeoutMS=5000
            )
            # Test connection
            client.admin.command('ping')
            print("[DB] MongoDB connection successful.")
            return client
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"MongoDB connection attempt {attempt + 1} failed, retrying...")
                import time
                time.sleep(2)
            else:
                print(f"MongoDB connection failed after {max_retries} attempts: {e}")
                raise

client = get_mongo_client()

# Database
# --- IMPROVEMENT: Read DB Name from environment variable ---
db_name = os.getenv("MONGO_DB_NAME", "face_recognition_db")
db = client[db_name]
print(f"[DB] Using database: {db_name}")

# Collections
faces_collection = db["faces"]
logs_collection = db["logs"]          # For high-priority Alerts
deepfake_collection = db["deepfakes"]
tracking_collection = db["tracking"]  # For every single detection (Movement Log)
config_collection = db["config"]      # For persistent watchlist & geofences

# Ensure indexes (No changes needed here)
# ... (indexes remain the same) ...
logs_collection.create_index(
    [("target", ASCENDING), ("camera_id", ASCENDING), ("timestamp", ASCENDING)]
)
deepfake_collection.create_index(
    [("camera_id", ASCENDING), ("timestamp", ASCENDING)]
)
tracking_collection.create_index(
    [("person", ASCENDING), ("timestamp", DESCENDING)]
)
config_collection.create_index(
    "name", unique=True
)

# -------------------------------
# Ensure directories exist (No changes needed here)
# ... (directory creation remains the same) ...
LOGS_DIR = "logs"
DEEPFAKE_LOGS_DIR = "data/deepfake_logs"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DEEPFAKE_LOGS_DIR, exist_ok=True)

# -------------------------------
# ✅ FIX: Persistent Encryption Key
# --- UPGRADED ---
# -------------------------------
# --- IMPROVEMENT: Read key path from environment variable ---
key_path_str = os.getenv("ENCRYPTION_KEY_PATH", "data/.encryption_key")
KEY_FILE = Path(key_path_str)
print(f"[DB] Using encryption key path: {KEY_FILE.resolve()}")

# Ensure parent directory exists
KEY_FILE.parent.mkdir(parents=True, exist_ok=True) # Added parents=True

if KEY_FILE.exists():
    ENCRYPTION_KEY = KEY_FILE.read_bytes()
    print("[DB] Loaded existing encryption key.")
else:
    print("[DB] Generating new encryption key...")
    ENCRYPTION_KEY = Fernet.generate_key()
    KEY_FILE.write_bytes(ENCRYPTION_KEY)
    KEY_FILE.chmod(0o600)  # Secure permissions
    print(f"[DB] New encryption key saved to {KEY_FILE.resolve()}")

try:
    fernet = Fernet(ENCRYPTION_KEY)
except Exception as e:
    print(f"[DB] CRITICAL: Failed to initialize Fernet with encryption key: {e}")
    # Decide how to handle this - maybe raise an exception to stop startup?
    raise SystemExit(f"Invalid encryption key found at {KEY_FILE.resolve()}. Please check or delete the file.")

def encrypt_embedding(embedding: list) -> str:
    """Encrypt a face embedding and return base64 string."""
    json_bytes = json.dumps(embedding).encode("utf-8")
    encrypted = fernet.encrypt(json_bytes)
    return base64.b64encode(encrypted).decode("utf-8")

def decrypt_embedding(encrypted_str: str) -> list:
    """Decrypt a base64 encrypted embedding string."""
    try:
        encrypted_bytes = base64.b64decode(encrypted_str)
        decrypted_bytes = fernet.decrypt(encrypted_bytes)
        return json.loads(decrypted_bytes.decode("utf-8"))
    except Exception as e:
        # Avoid printing sensitive info in logs if decryption fails
        # print(f"[DB] Decryption failed: {e}")
        print(f"[DB] Decryption failed for a stored embedding.")
        return []

# --- Rest of the file remains the same ---
# JSON Serialization Helper, File Logging Helpers, log_alert, log_deepfake,
# store/retrieve embeddings, retrieve_all_embeddings,
# save/load watchlist, save/load geofences, save/load tracking, clear history
# ... (all these functions stay exactly as they were in the previous version) ...

# -------------------------------
# JSON Serialization Helper
# -------------------------------
def json_serialize(doc: dict) -> dict:
    """Convert MongoDB ObjectId to strings."""
    if not doc:
        return {}
    new_doc = doc.copy() # Avoid modifying original
    for key, value in new_doc.items():
        if isinstance(value, ObjectId):
            new_doc[key] = str(value)
    return new_doc

# -------------------------------
# File Logging Helpers
# -------------------------------
def append_log_text(target: str, message: str):
    """Append plain text log entry."""
    filename = os.path.join(LOGS_DIR, f"{target}.txt")
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception as e:
        print(f"[FileLog] Error writing text log for {target}: {e}")

def append_log_json(target: str, entry: dict):
    """Append structured JSON log entry."""
    filename = os.path.join(LOGS_DIR, f"{target}.json")
    data = []
    try:
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"[FileLog] Warning: Could not decode existing JSON log for {target}. Starting fresh.")
                data = []

        data.append(json_serialize(entry)) # Serialize before appending

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[FileLog] Error writing JSON log for {target}: {e}")

def create_target_log_files(target: str):
    """Ensure log files exist for target."""
    txt_file = os.path.join(LOGS_DIR, f"{target}.txt")
    json_file = os.path.join(LOGS_DIR, f"{target}.json")
    try:
        if not os.path.exists(txt_file):
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write("")
        if not os.path.exists(json_file):
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)
    except Exception as e:
        print(f"[FileLog] Error creating log files for {target}: {e}")

# -------------------------------
# Log Recognition Event (Alerts)
# -------------------------------
def log_alert(camera_id: int, camera_name: str, geo: str, target: str, distance: float, cooldown: int = 10):
    """Insert recognition event with duplicate prevention."""
    now = datetime.now()
    cutoff_time = now - timedelta(seconds=cooldown)

    try:
        # Check for duplicate
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

        result = logs_collection.insert_one(log_entry)
        log_entry_serialized = json_serialize(log_entry) # Serialize before file logging

        # File logs
        text_message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Detected at Camera {camera_id} ({camera_name}, Geo={geo}), distance={distance:.2f}"
        append_log_text(target, text_message)
        append_log_json(target, log_entry_serialized)

        return True
    except Exception as e:
        print(f"[MongoDB] Failed to insert alert log: {e}")
        return False

# -------------------------------
# Log DeepFake Event
# -------------------------------
def log_deepfake(camera_id: int, camera_name: str, geo: str, detection: dict, frame_id: int):
    """Insert DeepFake detection event."""
    now = datetime.now()

    log_entry = {
        "camera_id": camera_id,
        "camera_name": camera_name,
        "geo": geo,
        "frame_id": frame_id,
        "is_fake": detection["is_fake"],
        "confidence": detection["confidence"],
        "bbox": detection["bbox"],
        "timestamp": now.isoformat()
    }

    try:
        result = deepfake_collection.insert_one(log_entry)
        log_entry_serialized = json_serialize(log_entry) # Serialize before file logging

        log_file = os.path.join(DEEPFAKE_LOGS_DIR, "deepfake_events.json")

        data = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"[FileLog] Warning: Could not decode deepfake log file. Starting fresh.")
                data = []

        data.append(log_entry_serialized)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return True
    except Exception as e:
        print(f"[MongoDB] Failed to insert deepfake log: {e}")
        return False

# -------------------------------
# Store/Retrieve Embeddings
# -------------------------------
def store_embedding(target: str, embedding: list):
    """Store encrypted face embedding."""
    try:
        encrypted = encrypt_embedding(embedding)
        faces_collection.update_one(
            {"target": target},
            {"$set": {"embedding": encrypted, "updated_at": datetime.now().isoformat()}},
            upsert=True
        )
        return True
    except Exception as e:
        print(f"[DB] Failed to store embedding for {target}: {e}")
        return False

def retrieve_embedding(target: str) -> list:
    """Retrieve decrypted embedding."""
    try:
        doc = faces_collection.find_one({"target": target})
        if doc and "embedding" in doc:
            return decrypt_embedding(doc["embedding"])
        return []
    except Exception as e:
        print(f"[DB] Failed to retrieve embedding for {target}: {e}")
        return []

# -------------------------------
# NEW: Retrieve All Embeddings (for face_service persistence)
# -------------------------------
def retrieve_all_embeddings() -> List[Dict[str, Any]]:
    """Retrieve all targets and their embeddings from the DB."""
    try:
        cursor = faces_collection.find({}, {"_id": 0, "target": 1, "embedding": 1}) # Exclude _id
        return list(cursor)
    except Exception as e:
        print(f"[DB] Failed to retrieve all embeddings: {e}")
        return []

# -------------------------------
# NEW: Persistence for Alert Service (Watchlist & Geofences)
# -------------------------------
def save_watchlist_db(watchlist: List[str]):
    """Save the entire watchlist to the config collection."""
    try:
        config_collection.update_one(
            {"name": "watchlist"},
            {"$set": {"data": {"items": watchlist}, "updated_at": datetime.now().isoformat()}}, # Embed in 'data'
            upsert=True
        )
    except Exception as e:
        print(f"[DB] Failed to save watchlist: {e}")

def load_watchlist_db() -> List[str]:
    """Load the watchlist from the config collection."""
    try:
        doc = config_collection.find_one({"name": "watchlist"})
        return doc.get("data", {}).get("items", []) if doc else []
    except Exception as e:
        print(f"[DB] Failed to load watchlist: {e}")
        return []

def save_geofence_db(geofences: Dict[str, Any]):
    """Save all geofence zones to the config collection."""
    try:
        config_collection.update_one(
            {"name": "geofences"},
            {"$set": {"data": {"zones": geofences}, "updated_at": datetime.now().isoformat()}}, # Embed in 'data'
            upsert=True
        )
    except Exception as e:
        print(f"[DB] Failed to save geofences: {e}")

def load_geofences_db() -> Dict[str, Any]:
    """Load all geofence zones from the config collection."""
    try:
        doc = config_collection.find_one({"name": "geofences"})
        return doc.get("data", {}).get("zones", {}) if doc else {}
    except Exception as e:
        print(f"[DB] Failed to load geofences: {e}")
        return {}

# -------------------------------
# NEW: Persistence for Tracking Service (Movement History)
# -------------------------------
def save_detection_to_db(detection: Dict[str, Any]):
    """
    Save a single detection event to the tracking collection.
    This creates the persistent audit trail / movement log.
    """
    try:
        # Ensure geo is stored consistently (e.g., as list if tuple)
        if isinstance(detection.get("geo"), tuple):
            detection["geo"] = list(detection["geo"])
        tracking_collection.insert_one(detection)
    except Exception as e:
        print(f"[DB] Failed to save tracking detection: {e}")

def load_tracking_history_db(limit_per_person: int = 100) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load the last N detection records for ALL persons from the DB.
    Uses an aggregation pipeline for efficiency.
    """
    try:
        pipeline = [
            {
                "$sort": {"timestamp": DESCENDING} # Get newest first
            },
            {
                "$group": {
                    "_id": "$person", # Group by person
                    "history": {"$push": "$$ROOT"} # Push all records into an array
                }
            },
            {
                "$project": {
                    "_id": 1, # Keep person name
                    "history": {"$slice": ["$history", limit_per_person]} # Keep only the last N
                }
            }
        ]

        result_cursor = tracking_collection.aggregate(pipeline)

        # Reformat the data for the service
        all_history = {}
        for doc in result_cursor:
            person = doc["_id"]
            # Records are newest-to-oldest, so reverse them and serialize ObjectId
            history_list = [json_serialize(item) for item in reversed(doc["history"])]
            all_history[person] = history_list

        return all_history

    except Exception as e:
        print(f"[DB] Failed to load tracking history: {e}")
        return {}

def clear_history_in_db(person_name: Optional[str] = None):
    """
    Delete tracking records from the database.
    If person_name is None, deletes ALL tracking history.
    """
    try:
        if person_name:
            print(f"[DB] Deleting tracking history for {person_name}...")
            result = tracking_collection.delete_many({"person": person_name})
            print(f"[DB] Deleted {result.deleted_count} tracking records for {person_name}.")
        else:
            print("[DB] Deleting ALL tracking history...")
            result = tracking_collection.delete_many({})
            print(f"[DB] Deleted {result.deleted_count} total tracking records.")
    except Exception as e:
        print(f"[DB] Failed to clear tracking history: {e}")
        # Re-raise the exception to be handled by the service layer
        raise e