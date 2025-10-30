# backend/app/routes/federated.py
"""
Federated learning endpoints.

Improvements:
 - concurrency protection (RLock)
 - atomic disk writes
 - robust numpy handling and shape/dtype checks
 - JSON-serializable responses (convert sets -> lists)
 - clearer logging and error handling
"""

from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import os
import json
import numpy as np
from datetime import datetime
import logging
from collections import defaultdict
from io import BytesIO
import tempfile
import threading

from app.state import FL_WEIGHTS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/face/fl", tags=["Federated Learning"])

# Ensure directory exists
FL_DIR = os.getenv("FL_DIR", "data/fl_weights")
os.makedirs(FL_DIR, exist_ok=True)

# ---------- Shared FL state (protected by lock) ----------
_lock = threading.RLock()

fl_state = {
    "current_round": 0,
    "total_clients": 0,
    "active_clients": set(),
    "training_status": "idle",  # idle, training, aggregating
    "last_aggregation": None,
    "global_model_version": 0
}

client_registry: Dict[str, Dict[str, Any]] = {}  # client_id -> metadata
aggregation_history: List[Dict[str, Any]] = []  # recent aggregation events

client_metrics = defaultdict(lambda: {
    "total_updates": 0,
    "total_rounds_participated": 0,
    "average_weights_size": 0.0,
    "last_contribution": None,
    "contribution_quality": []
})

# -------------------------------
# Pydantic models
# -------------------------------
class FLWeightsUpload(BaseModel):
    target: str = Field(..., min_length=1, max_length=100, description="Client ID")
    weights: Dict[str, list] = Field(..., description="Model weights as dict of layer -> values")
    round_number: Optional[int] = Field(None, ge=0, description="Training round number")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")

    @validator("weights")
    def validate_weights(cls, v):
        if not v:
            raise ValueError("Weights cannot be empty")
        if not isinstance(v, dict):
            raise ValueError("Weights must be a dict of layer -> list")
        for layer_name, layer_weights in v.items():
            if not isinstance(layer_weights, (list, tuple)):
                raise ValueError(f"Layer '{layer_name}' weights must be a list/tuple")
            if len(layer_weights) == 0:
                raise ValueError(f"Layer '{layer_name}' weights cannot be empty")
        return v


class AggregationConfig(BaseModel):
    algorithm: str = Field(default="fedavg", description="Aggregation algorithm (fedavg/weighted)")
    min_clients: int = Field(default=2, ge=1, le=100, description="Minimum clients required")
    client_selection: Optional[List[str]] = Field(None, description="Specific clients to aggregate (None = all)")
    weights_strategy: str = Field(default="equal", description="Weighting strategy (equal/contribution)")


class ClientConfig(BaseModel):
    client_id: str = Field(..., min_length=1, max_length=100, description="Unique client ID")
    client_name: Optional[str] = Field(None, description="Human-readable name")
    metadata: Optional[dict] = Field(default_factory=dict, description="Client metadata")


# ---------- Helper utilities ----------
def _now_iso() -> str:
    return datetime.now().isoformat()


def _atomic_write_json(path: str, data: dict):
    """
    Write JSON to file atomically by writing to a temp file then replacing.
    """
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, prefix=".tmp_fl_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # Ensure tmp file removed if replace fails
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise


def _np_from_list_safe(lst):
    """
    Convert list-like to numpy array safely, coerce dtype to float64 for stability.
    """
    arr = np.asarray(lst, dtype=np.float64)
    return arr


def _make_serializable(obj):
    """Convert sets inside fl_state to lists for JSON responses."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, set):
                out[k] = list(v)
            elif isinstance(v, dict):
                out[k] = _make_serializable(v)
            else:
                out[k] = v
        return out
    return obj


# -------------------------------
# POST /face/fl/upload_weights
# -------------------------------
@router.post("/upload_weights")
async def upload_fl_weights(payload: FLWeightsUpload):
    try:
        target = payload.target
        weights_dict = payload.weights
        round_number = payload.round_number if payload.round_number is not None else fl_state["current_round"]

        logger.info("Receiving FL weights from client '%s' for round %s", target, round_number)

        # Validate weights presence
        if not weights_dict:
            raise HTTPException(status_code=400, detail="Weights cannot be empty")

        # Convert to numpy arrays and validate values
        weights_np = {}
        for layer_name, layer_vals in weights_dict.items():
            arr = _np_from_list_safe(layer_vals)
            if arr.size == 0:
                raise HTTPException(status_code=400, detail=f"Layer '{layer_name}' has empty weights")
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                raise HTTPException(status_code=400, detail=f"Layer '{layer_name}' contains invalid values (NaN or Inf)")
            weights_np[layer_name] = arr

        with _lock:
            # Auto-register unregistered clients (preserve original behavior)
            if target not in client_registry:
                logger.warning("Unregistered client '%s' attempting upload. Auto-registering.", target)
                client_registry[target] = {
                    "registered_at": _now_iso(),
                    "status": "active",
                    "last_update": None,
                    "round": 0,
                    "contribution_count": 0
                }

            # If client already has stored layers, check for layer name consistency (warning)
            if target in FL_WEIGHTS:
                try:
                    existing_layers = set(FL_WEIGHTS[target].keys())
                except Exception:
                    existing_layers = set()
                new_layers = set(weights_np.keys())
                if existing_layers and new_layers != existing_layers:
                    logger.warning("Layer structure changed for client '%s'. Previous: %s, New: %s", target, existing_layers, new_layers)

            # Persist in-memory
            FL_WEIGHTS[target] = weights_np

            # Prepare save payload (use original list representation for portability)
            save_path = os.path.join(FL_DIR, f"{target}.json")
            save_data = {
                "weights": {k: v.tolist() for k, v in weights_np.items()},
                "round": round_number,
                "timestamp": _now_iso(),
                "metadata": payload.metadata or {},
                "version": fl_state["global_model_version"]
            }

            try:
                _atomic_write_json(save_path, save_data)
            except Exception as e:
                logger.exception("Failed to persist weights to disk for client %s: %s", target, e)
                # Disk persistence failure is logged but we keep in-memory copy.

            # Update registry and metrics (protected by lock)
            client_info = client_registry.setdefault(target, {
                "registered_at": _now_iso(),
                "status": "active",
                "last_update": None,
                "round": 0,
                "contribution_count": 0
            })

            client_info["last_update"] = _now_iso()
            client_info["round"] = int(round_number)
            client_info["contribution_count"] = client_info.get("contribution_count", 0) + 1
            client_info["status"] = "active"

            fl_state["active_clients"].add(target)
            fl_state["total_clients"] = len(client_registry)

            # Update metrics
            metrics = client_metrics[target]
            metrics["total_updates"] = metrics.get("total_updates", 0) + 1
            metrics["total_rounds_participated"] = max(metrics.get("total_rounds_participated", 0), int(round_number) + 1)
            metrics["last_contribution"] = _now_iso()

            total_weights = sum(int(v.size) for v in weights_np.values())
            prev_updates = metrics["total_updates"]
            # recompute average safely
            prev_avg = float(metrics.get("average_weights_size", 0.0))
            if prev_updates <= 1:
                metrics["average_weights_size"] = float(total_weights)
            else:
                # previous average * (n-1) + new / n
                metrics["average_weights_size"] = (prev_avg * (prev_updates - 1) + total_weights) / prev_updates

        logger.info("Successfully stored FL weights for client '%s' (round %s)", target, round_number)

        return JSONResponse({
            "status": "success",
            "message": f"Federated weights received for client '{target}'",
            "client_id": target,
            "round": int(round_number),
            "layers": list(weights_dict.keys()),
            "total_parameters": total_weights,
            "contribution_count": client_info["contribution_count"],
            "global_model_version": fl_state["global_model_version"],
            "timestamp": _now_iso()
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error uploading FL weights: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to upload FL weights: {str(e)}")


# -------------------------------
# GET /status (returns weights for a client)
# -------------------------------
@router.get("/status")
async def get_fl_status(client_id: str = Query(..., description="Client ID to query")):
    try:
        logger.debug("Fetching FL status for client: %s", client_id)

        with _lock:
            # In-memory first
            if client_id in FL_WEIGHTS:
                weights_np = FL_WEIGHTS[client_id]
                weights_json = {k: v.tolist() for k, v in weights_np.items()}
                client_info = client_registry.get(client_id, {})
                metrics = client_metrics.get(client_id, {})
                return JSONResponse({
                    "status": "success",
                    "client_id": client_id,
                    "weights": weights_json,
                    "layers": list(weights_json.keys()),
                    "client_info": {
                        "last_update": client_info.get("last_update"),
                        "round": client_info.get("round", 0),
                        "contribution_count": client_info.get("contribution_count", 0),
                        "status": client_info.get("status", "unknown")
                    },
                    "metrics": {
                        "total_updates": metrics.get("total_updates", 0),
                        "rounds_participated": metrics.get("total_rounds_participated", 0)
                    },
                    "global_model_version": fl_state["global_model_version"]
                })

        # Try disk (no lock needed for read)
        disk_path = os.path.join(FL_DIR, f"{client_id}.json")
        if os.path.exists(disk_path):
            try:
                with open(disk_path, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)
            except Exception as e:
                logger.exception("Failed to read disk weights for %s: %s", client_id, e)
                raise HTTPException(500, f"Failed to read saved weights: {e}")

            if isinstance(saved_data, dict) and "weights" in saved_data:
                weights_json = saved_data["weights"]
                metadata = {
                    "round": saved_data.get("round", 0),
                    "timestamp": saved_data.get("timestamp"),
                    "version": saved_data.get("version", 0)
                }
            else:
                weights_json = saved_data
                metadata = {}

            # Load into memory for quicker next access
            with _lock:
                FL_WEIGHTS[client_id] = {k: _np_from_list_safe(v) for k, v in weights_json.items()}

            return JSONResponse({
                "status": "success",
                "client_id": client_id,
                "weights": weights_json,
                "layers": list(weights_json.keys()),
                "metadata": metadata,
                "source": "disk"
            })

        logger.warning("No weights found for client: %s", client_id)
        return JSONResponse({
            "status": "success",
            "client_id": client_id,
            "weights": {},
            "message": "No weights found for this client",
            "suggestion": "Client needs to upload weights first"
        })

    except Exception as e:
        logger.exception("Error fetching FL status: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch FL status: {str(e)}")


# -------------------------------
# GET /get_weights (compat)
# -------------------------------
@router.get("/get_weights")
async def get_fl_weights(target: str = Query(..., description="Target client ID")):
    return await get_fl_status(client_id=target)


# -------------------------------
# DELETE /weights/{client_id}
# -------------------------------
@router.delete("/weights/{client_id}")
async def delete_fl_weights(
    client_id: str,
    remove_from_registry: bool = Query(False, description="Also remove from client registry")
):
    try:
        logger.info("Deleting FL weights for client: %s", client_id)
        removed_from_memory = False
        removed_from_disk = False
        removed_from_registry_flag = False

        with _lock:
            if client_id in FL_WEIGHTS:
                del FL_WEIGHTS[client_id]
                removed_from_memory = True

            # Remove from active clients
            if client_id in fl_state["active_clients"]:
                fl_state["active_clients"].remove(client_id)

            if remove_from_registry and client_id in client_registry:
                del client_registry[client_id]
                client_metrics.pop(client_id, None)
                removed_from_registry_flag = True
                fl_state["total_clients"] = len(client_registry)

        disk_path = os.path.join(FL_DIR, f"{client_id}.json")
        if os.path.exists(disk_path):
            try:
                os.remove(disk_path)
                removed_from_disk = True
            except Exception:
                logger.exception("Failed to remove disk file for client %s", client_id)

        if not removed_from_memory and not removed_from_disk:
            raise HTTPException(status_code=404, detail=f"No weights found for client '{client_id}'")

        logger.info("Successfully deleted FL weights for client: %s", client_id)
        return JSONResponse({
            "status": "success",
            "message": f"Weights deleted for client '{client_id}'",
            "client_id": client_id,
            "removed_from_memory": removed_from_memory,
            "removed_from_disk": removed_from_disk,
            "removed_from_registry": removed_from_registry_flag
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deleting FL weights: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to delete weights: {str(e)}")


# -------------------------------
# GET /list
# -------------------------------
@router.get("/list")
async def list_fl_clients(
    include_metrics: bool = Query(False, description="Include performance metrics"),
    status_filter: Optional[str] = Query(None, description="Filter by status (active/inactive)")
):
    try:
        logger.debug("Listing FL clients")

        with _lock:
            memory_clients = set(FL_WEIGHTS.keys())
        disk_clients = set()
        if os.path.exists(FL_DIR):
            for filename in os.listdir(FL_DIR):
                if filename.endswith(".json"):
                    disk_clients.add(filename[:-5])

        all_clients = memory_clients.union(disk_clients)
        clients_data = []

        with _lock:
            for client_id in sorted(all_clients):
                client_data: Dict[str, Any] = {
                    "client_id": client_id,
                    "in_memory": client_id in memory_clients,
                    "on_disk": client_id in disk_clients
                }

                if client_id in client_registry:
                    client_info = client_registry[client_id]
                    client_data["status"] = client_info.get("status", "unknown")
                    client_data["last_update"] = client_info.get("last_update")
                    client_data["contribution_count"] = client_info.get("contribution_count", 0)
                    client_data["round"] = client_info.get("round", 0)
                else:
                    client_data["status"] = "unregistered"

                if include_metrics and client_id in client_metrics:
                    metrics = client_metrics[client_id]
                    client_data["metrics"] = {
                        "total_updates": metrics.get("total_updates", 0),
                        "rounds_participated": metrics.get("total_rounds_participated", 0),
                        "average_weights_size": int(metrics.get("average_weights_size", 0)),
                        "last_contribution": metrics.get("last_contribution")
                    }

                # status filter
                if status_filter:
                    if status_filter == "active" and client_data.get("status") != "active":
                        continue
                    if status_filter == "inactive" and client_data.get("status") == "active":
                        continue

                clients_data.append(client_data)

        return JSONResponse({
            "status": "success",
            "total_clients": len(clients_data),
            "registered_clients": len(client_registry),
            "active_clients": len(fl_state["active_clients"]),
            "clients": clients_data,
            "global_model_version": fl_state["global_model_version"],
            "current_round": fl_state["current_round"]
        })

    except Exception as e:
        logger.exception("Error listing FL clients: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to list clients: {str(e)}")


# -------------------------------
# POST /register
# -------------------------------
@router.post("/register")
async def register_client(config: ClientConfig):
    try:
        client_id = config.client_id
        logger.info("Registering FL client: %s", client_id)

        with _lock:
            if client_id in client_registry:
                logger.warning("Client '%s' already registered", client_id)
                return JSONResponse({
                    "status": "success",
                    "message": f"Client '{client_id}' was already registered",
                    "client_id": client_id,
                    "already_registered": True,
                    "registration_date": client_registry[client_id].get("registered_at")
                })

            client_registry[client_id] = {
                "registered_at": _now_iso(),
                "client_name": config.client_name,
                "metadata": config.metadata or {},
                "status": "registered",
                "last_update": None,
                "round": 0,
                "contribution_count": 0
            }
            fl_state["total_clients"] = len(client_registry)

        logger.info("Successfully registered client: %s", client_id)
        return JSONResponse({
            "status": "success",
            "message": f"Client '{client_id}' registered successfully",
            "client_id": client_id,
            "registered_at": client_registry[client_id]["registered_at"],
            "total_clients": fl_state["total_clients"]
        })

    except Exception as e:
        logger.exception("Error registering client: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to register client: {str(e)}")


# -------------------------------
# DELETE /unregister/{client_id}
# -------------------------------
@router.delete("/unregister/{client_id}")
async def unregister_client(client_id: str, delete_weights: bool = Query(False)):
    try:
        logger.info("Unregistering client: %s", client_id)
        with _lock:
            if client_id not in client_registry:
                raise HTTPException(status_code=404, detail=f"Client '{client_id}' not registered")

            del client_registry[client_id]
            if client_id in fl_state["active_clients"]:
                fl_state["active_clients"].remove(client_id)
            fl_state["total_clients"] = len(client_registry)

            weights_deleted = False
            if delete_weights:
                if client_id in FL_WEIGHTS:
                    del FL_WEIGHTS[client_id]
                disk_path = os.path.join(FL_DIR, f"{client_id}.json")
                if os.path.exists(disk_path):
                    try:
                        os.remove(disk_path)
                        weights_deleted = True
                    except Exception:
                        logger.exception("Failed deleting disk weights for %s", client_id)

        logger.info("Successfully unregistered client: %s", client_id)
        return JSONResponse({
            "status": "success",
            "message": f"Client '{client_id}' unregistered",
            "client_id": client_id,
            "weights_deleted": weights_deleted
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error unregistering client: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to unregister client: {str(e)}")


# -------------------------------
# POST /aggregate
# -------------------------------
@router.post("/aggregate")
async def aggregate_weights(config: AggregationConfig = Body(...)):
    try:
        logger.info("Starting model aggregation with algorithm=%s", config.algorithm)
        with _lock:
            fl_state["training_status"] = "aggregating"

            if config.client_selection:
                clients_to_aggregate = [c for c in config.client_selection if c in FL_WEIGHTS]
            else:
                clients_to_aggregate = list(FL_WEIGHTS.keys())

            if len(clients_to_aggregate) < config.min_clients:
                fl_state["training_status"] = "idle"
                raise HTTPException(status_code=400, detail=f"Insufficient clients for aggregation. Required: {config.min_clients}, Available: {len(clients_to_aggregate)}")

            logger.info("Aggregating weights from %d clients: %s", len(clients_to_aggregate), clients_to_aggregate)

            client_weights_list = [FL_WEIGHTS[c] for c in clients_to_aggregate]

            # layer consistency check
            layer_names = set(client_weights_list[0].keys())
            for cw in client_weights_list[1:]:
                if set(cw.keys()) != layer_names:
                    fl_state["training_status"] = "idle"
                    raise HTTPException(status_code=400, detail="Inconsistent layer structure across clients. All clients must have the same model architecture.")

            # determine aggregation weights
            if config.weights_strategy == "equal":
                aggregation_weights = [1.0 / len(clients_to_aggregate)] * len(clients_to_aggregate)
            elif config.weights_strategy == "contribution":
                contributions = [client_registry.get(c, {}).get("contribution_count", 1) for c in clients_to_aggregate]
                total_contributions = sum(contributions) or 1
                aggregation_weights = [c / total_contributions for c in contributions]
            else:
                aggregation_weights = [1.0 / len(clients_to_aggregate)] * len(clients_to_aggregate)

            logger.debug("Aggregation weights: %s", aggregation_weights)

            # perform FedAvg with careful dtype/shape handling
            aggregated_weights: Dict[str, np.ndarray] = {}
            for layer_name in layer_names:
                # initialize accumulator with zeros of same shape as first client
                first_arr = _np_from_list_safe(client_weights_list[0][layer_name])
                accumulator = np.zeros_like(first_arr, dtype=np.float64)

                for client_w, w in zip(client_weights_list, aggregation_weights):
                    arr = _np_from_list_safe(client_w[layer_name])
                    if arr.shape != accumulator.shape:
                        fl_state["training_status"] = "idle"
                        raise HTTPException(status_code=400, detail=f"Shape mismatch for layer '{layer_name}': expected {accumulator.shape}, got {arr.shape}")
                    accumulator += arr * float(w)

                aggregated_weights[layer_name] = accumulator

            # store aggregated model in memory and persist to disk
            global_model_id = f"global_model_v{fl_state['global_model_version'] + 1}"
            FL_WEIGHTS[global_model_id] = aggregated_weights

            save_path = os.path.join(FL_DIR, f"{global_model_id}.json")
            save_data = {
                "weights": {k: v.tolist() for k, v in aggregated_weights.items()},
                "version": fl_state["global_model_version"] + 1,
                "round": fl_state["current_round"],
                "clients_aggregated": clients_to_aggregate,
                "aggregation_algorithm": config.algorithm,
                "weights_strategy": config.weights_strategy,
                "timestamp": _now_iso()
            }
            try:
                _atomic_write_json(save_path, save_data)
            except Exception:
                logger.exception("Failed to persist aggregated model to disk (ignored)")

            # update fl_state and history
            fl_state["global_model_version"] += 1
            fl_state["current_round"] += 1
            fl_state["last_aggregation"] = _now_iso()
            fl_state["training_status"] = "idle"

            aggregation_event = {
                "version": fl_state["global_model_version"],
                "round": fl_state["current_round"] - 1,
                "timestamp": _now_iso(),
                "clients_count": len(clients_to_aggregate),
                "clients": clients_to_aggregate,
                "algorithm": config.algorithm,
                "weights_strategy": config.weights_strategy
            }
            aggregation_history.append(aggregation_event)
            if len(aggregation_history) > 100:
                aggregation_history.pop(0)

        logger.info("Aggregation complete. New global model version: %s", fl_state["global_model_version"])

        return JSONResponse({
            "status": "success",
            "message": "Model aggregation completed successfully",
            "global_model": {
                "id": global_model_id,
                "version": fl_state["global_model_version"],
                "round": fl_state["current_round"] - 1,
                "layers": list(aggregated_weights.keys()),
                "total_parameters": int(sum(int(w.size) for w in aggregated_weights.values()))
            },
            "aggregation_details": {
                "clients_aggregated": len(clients_to_aggregate),
                "client_ids": clients_to_aggregate,
                "algorithm": config.algorithm,
                "weights_strategy": config.weights_strategy,
                "aggregation_weights": aggregation_weights
            },
            "timestamp": _now_iso()
        })

    except HTTPException:
        with _lock:
            fl_state["training_status"] = "idle"
        raise
    except Exception as e:
        with _lock:
            fl_state["training_status"] = "idle"
        logger.exception("Error in model aggregation: %s", e)
        raise HTTPException(status_code=500, detail=f"Aggregation failed: {str(e)}")


# -------------------------------
# GET /global_model
# -------------------------------
@router.get("/global_model")
async def get_global_model(version: Optional[int] = Query(None, description="Specific version (None = latest)")):
    try:
        with _lock:
            if version is None:
                version = fl_state["global_model_version"]

            if version == 0:
                return JSONResponse({
                    "status": "success",
                    "message": "No global model available yet",
                    "global_model_version": 0,
                    "suggestion": "Aggregate client weights first using /aggregate endpoint"
                })

            global_model_id = f"global_model_v{version}"
            if global_model_id in FL_WEIGHTS:
                weights_np = FL_WEIGHTS[global_model_id]
                weights_json = {k: v.tolist() for k, v in weights_np.items()}
                return JSONResponse({
                    "status": "success",
                    "global_model": {
                        "id": global_model_id,
                        "version": version,
                        "weights": weights_json,
                        "layers": list(weights_json.keys()),
                        "total_parameters": int(sum(len(v) for v in weights_json.values()))
                    },
                    "timestamp": _now_iso()
                })

        # Try disk
        disk_path = os.path.join(FL_DIR, f"{global_model_id}.json")
        if os.path.exists(disk_path):
            with open(disk_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            weights_json = saved_data.get("weights", {})
            with _lock:
                FL_WEIGHTS[global_model_id] = {k: _np_from_list_safe(v) for k, v in weights_json.items()}

            return JSONResponse({
                "status": "success",
                "global_model": {
                    "id": global_model_id,
                    "version": version,
                    "weights": weights_json,
                    "layers": list(weights_json.keys()),
                    "total_parameters": int(sum(len(v) for v in weights_json.values())),
                    "metadata": {
                        "round": saved_data.get("round"),
                        "timestamp": saved_data.get("timestamp"),
                        "clients_aggregated": saved_data.get("clients_aggregated", [])
                    }
                },
                "source": "disk"
            })

        raise HTTPException(status_code=404, detail=f"Global model version {version} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fetching global model: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get global model: {str(e)}")


# -------------------------------
# GET /aggregation/history
# -------------------------------
@router.get("/aggregation/history")
async def get_aggregation_history(limit: int = Query(50, ge=1, le=1000)):
    try:
        logger.debug("Fetching aggregation history (limit=%d)", limit)
        with _lock:
            history = list(aggregation_history[-limit:])
        history.reverse()
        return JSONResponse({
            "status": "success",
            "total_aggregations": len(aggregation_history),
            "returned": len(history),
            "history": history,
            "current_version": fl_state["global_model_version"]
        })
    except Exception as e:
        logger.exception("Error fetching aggregation history: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


# -------------------------------
# GET /stats
# -------------------------------
@router.get("/stats")
async def get_fl_statistics():
    try:
        logger.debug("Fetching FL statistics")
        with _lock:
            total_clients = len(client_registry)
            active_clients_count = len(fl_state["active_clients"])
            total_weights_stored = len(FL_WEIGHTS)

            contributions = [info.get("contribution_count", 0) for info in client_registry.values()]
            if contributions:
                avg_contributions = sum(contributions) / len(contributions)
                max_contributions = max(contributions)
                min_contributions = min(contributions)
            else:
                avg_contributions = max_contributions = min_contributions = 0

            total_params = 0
            if fl_state["global_model_version"] > 0:
                global_model_id = f"global_model_v{fl_state['global_model_version']}"
                if global_model_id in FL_WEIGHTS:
                    total_params = int(sum(w.size for w in FL_WEIGHTS[global_model_id].values()))

            recent_updates = []
            for client_id, info in client_registry.items():
                if info.get("last_update"):
                    try:
                        last_update_time = datetime.fromisoformat(info["last_update"])
                        time_since = (datetime.now() - last_update_time).total_seconds()
                        if time_since < 3600:
                            recent_updates.append(client_id)
                    except Exception:
                        continue

            model_layers = 0
            if fl_state["global_model_version"] > 0:
                global_model_id = f"global_model_v{fl_state['global_model_version']}"
                if global_model_id in FL_WEIGHTS:
                    model_layers = len(FL_WEIGHTS[global_model_id].keys())

            stats = {
                "federated_learning": {
                    "current_round": fl_state["current_round"],
                    "global_model_version": fl_state["global_model_version"],
                    "training_status": fl_state["training_status"],
                    "last_aggregation": fl_state["last_aggregation"],
                    "total_aggregations": len(aggregation_history)
                },
                "clients": {
                    "total_registered": total_clients,
                    "active": active_clients_count,
                    "inactive": total_clients - active_clients_count,
                    "with_weights": total_weights_stored
                },
                "contributions": {
                    "average_per_client": round(avg_contributions, 2),
                    "max_contributions": max_contributions,
                    "min_contributions": min_contributions,
                    "total_updates": sum(contributions)
                },
                "model": {
                    "total_parameters": total_params,
                    "layers": model_layers
                },
                "activity": {
                    "recent_updates_last_hour": len(recent_updates),
                    "active_clients": list(fl_state["active_clients"])
                }
            }

        return JSONResponse({
            "status": "success",
            "statistics": stats,
            "timestamp": _now_iso()
        })
    except Exception as e:
        logger.exception("Error fetching FL statistics: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


# -------------------------------
# GET /client/{client_id}/metrics
# -------------------------------
@router.get("/client/{client_id}/metrics")
async def get_client_metrics(client_id: str):
    try:
        logger.debug("Fetching metrics for client: %s", client_id)
        with _lock:
            if client_id not in client_registry:
                raise HTTPException(status_code=404, detail=f"Client '{client_id}' not registered")
            client_info = client_registry[client_id]
            metrics = client_metrics.get(client_id, {})

            total_updates = sum(m.get("total_updates", 0) for m in client_metrics.values())
            contribution_percentage = (metrics.get("total_updates", 0) / total_updates * 100) if total_updates > 0 else 0.0
            has_weights = client_id in FL_WEIGHTS

            return JSONResponse({
                "status": "success",
                "client_id": client_id,
                "client_info": {
                    "name": client_info.get("client_name"),
                    "registered_at": client_info.get("registered_at"),
                    "status": client_info.get("status"),
                    "last_update": client_info.get("last_update"),
                    "current_round": client_info.get("round", 0),
                    "has_weights": has_weights
                },
                "metrics": {
                    "total_updates": metrics.get("total_updates", 0),
                    "rounds_participated": metrics.get("total_rounds_participated", 0),
                    "average_weights_size": int(metrics.get("average_weights_size", 0)),
                    "last_contribution": metrics.get("last_contribution"),
                    "contribution_percentage": round(contribution_percentage, 2)
                },
                "contribution_count": client_info.get("contribution_count", 0)
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fetching client metrics: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get client metrics: {str(e)}")


# -------------------------------
# POST /reset
# -------------------------------
@router.post("/reset")
async def reset_fl_state(
    confirmation: str = Query(..., description="Must be 'CONFIRM_RESET'"),
    keep_clients: bool = Query(True, description="Keep client registry"),
    keep_weights: bool = Query(False, description="Keep stored weights")
):
    try:
        if confirmation != "CONFIRM_RESET":
            raise HTTPException(status_code=400, detail="Invalid confirmation. Must provide confirmation=CONFIRM_RESET")

        logger.warning("RESETTING FL STATE - confirmed")

        with _lock:
            old_round = fl_state["current_round"]
            old_version = fl_state["global_model_version"]

            fl_state["current_round"] = 0
            fl_state["global_model_version"] = 0
            fl_state["training_status"] = "idle"
            fl_state["last_aggregation"] = None
            fl_state["active_clients"].clear()
            aggregation_history.clear()

            clients_cleared = 0
            if not keep_clients:
                clients_cleared = len(client_registry)
                client_registry.clear()
                client_metrics.clear()
                fl_state["total_clients"] = 0

            weights_cleared = 0
            if not keep_weights:
                weights_cleared = len(FL_WEIGHTS)
                FL_WEIGHTS.clear()
                if os.path.exists(FL_DIR):
                    for filename in os.listdir(FL_DIR):
                        if filename.endswith(".json"):
                            try:
                                os.remove(os.path.join(FL_DIR, filename))
                            except Exception:
                                logger.debug("Failed remove file during reset (ignored)")

        logger.warning("FL state reset complete. Previous round=%s, version=%s", old_round, old_version)
        return JSONResponse({
            "status": "success",
            "message": "FL state reset successfully",
            "previous_state": {"round": old_round, "version": old_version},
            "cleared": {"clients": clients_cleared if not keep_clients else 0, "weights": weights_cleared if not keep_weights else 0},
            "kept": {"clients": keep_clients, "weights": keep_weights}
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error resetting FL state: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to reset FL state: {str(e)}")


# -------------------------------
# GET /export
# -------------------------------
@router.get("/export")
async def export_fl_data(
    include_weights: bool = Query(False, description="Include model weights"),
    format: str = Query("json", description="Export format (json/csv)")
):
    try:
        logger.info("Exporting FL data (format=%s, include_weights=%s)", format, include_weights)
        with _lock:
            export_data = {
                "export_time": _now_iso(),
                "fl_state": {
                    "current_round": fl_state["current_round"],
                    "global_model_version": fl_state["global_model_version"],
                    "training_status": fl_state["training_status"],
                    "last_aggregation": fl_state["last_aggregation"],
                    "total_clients": fl_state["total_clients"],
                    "active_clients": list(fl_state["active_clients"])
                },
                "clients": [
                    {
                        "client_id": client_id,
                        **info,
                        "metrics": client_metrics.get(client_id, {})
                    }
                    for client_id, info in client_registry.items()
                ],
                "aggregation_history": aggregation_history
            }

            if include_weights:
                export_data["weights"] = {
                    client_id: {k: v.tolist() for k, v in weights.items()}
                    for client_id, weights in FL_WEIGHTS.items()
                }

        filename = f"fl_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if format == "json":
            json_bytes = json.dumps(export_data, indent=2).encode("utf-8")
            file_stream = BytesIO(json_bytes)
            return StreamingResponse(file_stream, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename}.json"})
        elif format == "csv":
            import csv
            from io import StringIO
            csv_buffer = StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["Client ID", "Status", "Contribution Count", "Last Update", "Round"])
            with _lock:
                items = list(client_registry.items())
            for client_id, info in items:
                writer.writerow([client_id, info.get("status", "unknown"), info.get("contribution_count", 0), info.get("last_update", "N/A"), info.get("round", 0)])
            csv_bytes = csv_buffer.getvalue().encode("utf-8")
            file_stream = BytesIO(csv_bytes)
            return StreamingResponse(file_stream, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}.csv"})
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use 'json' or 'csv'")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error exporting FL data: %s", e)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# -------------------------------
# GET /health
# -------------------------------
@router.get("/health")
async def fl_health_check():
    try:
        weights_dir_accessible = os.path.exists(FL_DIR) and os.access(FL_DIR, os.W_OK)
        weights_loaded = len(FL_WEIGHTS) > 0
        health_status = "healthy" if weights_dir_accessible else "degraded"
        with _lock:
            resp = {
                "status": health_status,
                "service": "federated_learning",
                "components": {
                    "weights_directory": "accessible" if weights_dir_accessible else "unavailable",
                    "weights_loaded": weights_loaded,
                    "client_registry": "operational"
                },
                "statistics": {
                    "total_clients": fl_state["total_clients"],
                    "active_clients": len(fl_state["active_clients"]),
                    "global_model_version": fl_state["global_model_version"],
                    "current_round": fl_state["current_round"],
                    "training_status": fl_state["training_status"]
                },
                "timestamp": _now_iso()
            }
        return JSONResponse(resp)
    except Exception as e:
        logger.exception("Health check failed: %s", e)
        return JSONResponse({"status": "error", "service": "federated_learning", "error": str(e)}, status_code=503)


# -------------------------------
# POST /round/start
# -------------------------------
@router.post("/round/start")
async def start_training_round():
    try:
        with _lock:
            if fl_state["training_status"] != "idle":
                raise HTTPException(status_code=400, detail=f"Cannot start new round. Current status: {fl_state['training_status']}")
            logger.info("Starting training round %s", fl_state["current_round"])
            fl_state["training_status"] = "training"
            fl_state["active_clients"].clear()
            return JSONResponse({
                "status": "success",
                "message": f"Training round {fl_state['current_round']} started",
                "round": fl_state["current_round"],
                "total_clients": fl_state["total_clients"],
                "timestamp": _now_iso()
            })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error starting training round: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to start round: {str(e)}")


# -------------------------------
# GET /round/status
# -------------------------------
@router.get("/round/status")
async def get_training_status():
    try:
        with _lock:
            total_clients = fl_state["total_clients"]
            active = len(fl_state["active_clients"])
        participation_rate = (active / total_clients * 100) if total_clients > 0 else 0.0
        return JSONResponse({
            "status": "success",
            "training": {
                "current_round": fl_state["current_round"],
                "status": fl_state["training_status"],
                "total_clients": total_clients,
                "active_clients": active,
                "participation_rate": round(participation_rate, 2)
            },
            "global_model": {
                "version": fl_state["global_model_version"],
                "last_aggregation": fl_state["last_aggregation"]
            },
            "timestamp": _now_iso()
        })
    except Exception as e:
        logger.exception("Error fetching training status: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


# -------------------------------
# POST /validate
# -------------------------------
@router.post("/validate")
async def validate_weights(payload: FLWeightsUpload):
    try:
        logger.info("Validating weights structure for client: %s", payload.target)
        validation_results = {"valid": True, "issues": [], "warnings": [], "structure": {}}
        total_params = 0
        for layer_name, layer_weights in payload.weights.items():
            arr = _np_from_list_safe(layer_weights)
            if arr.size == 0:
                validation_results["valid"] = False
                validation_results["issues"].append(f"Layer '{layer_name}' has empty weights")
            if np.any(np.isnan(arr)):
                validation_results["valid"] = False
                validation_results["issues"].append(f"Layer '{layer_name}' contains NaN values")
            if np.any(np.isinf(arr)):
                validation_results["valid"] = False
                validation_results["issues"].append(f"Layer '{layer_name}' contains Inf values")
            if np.all(arr == 0):
                validation_results["warnings"].append(f"Layer '{layer_name}' contains all zeros")
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            if std == 0:
                validation_results["warnings"].append(f"Layer '{layer_name}' has zero variance")
            validation_results["structure"][layer_name] = {
                "shape": arr.shape,
                "size": int(arr.size),
                "dtype": str(arr.dtype),
                "mean": mean,
                "std": std,
                "min": float(np.min(arr)),
                "max": float(np.max(arr))
            }
            total_params += int(arr.size)
        validation_results["total_parameters"] = total_params
        validation_results["total_layers"] = len(payload.weights)
        logger.info("Validation complete for %s: %s", payload.target, "VALID" if validation_results["valid"] else "INVALID")
        return JSONResponse({"status": "success", "client_id": payload.target, "validation": validation_results})
    except Exception as e:
        logger.exception("Error validating weights: %s", e)
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
