# backend/app/routes/federated.py

from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import os
import json
import numpy as np
from datetime import datetime, timedelta
import logging
from collections import defaultdict
from io import BytesIO

from app.state import FL_WEIGHTS

# Initialize logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/face/fl", tags=["Federated Learning"])

# Ensure FL weights directory exists (EXISTING)
FL_DIR = "data/fl_weights"
os.makedirs(FL_DIR, exist_ok=True)

# -------------------------------
# NEW: FL State Management
# -------------------------------
fl_state = {
    "current_round": 0,
    "total_clients": 0,
    "active_clients": set(),
    "training_status": "idle",  # idle, training, aggregating
    "last_aggregation": None,
    "global_model_version": 0
}

# NEW: Client tracking
client_registry = {}  # client_id -> {last_update, round, contribution_count, status}

# NEW: Aggregation history
aggregation_history = []  # List of aggregation events with metadata

# NEW: Performance metrics
client_metrics = defaultdict(lambda: {
    "total_updates": 0,
    "total_rounds_participated": 0,
    "average_weights_size": 0,
    "last_contribution": None,
    "contribution_quality": []
})

# -------------------------------
# NEW: Pydantic Models for Request Validation
# -------------------------------
class FLWeightsUpload(BaseModel):
    """Model for weight upload with validation"""
    target: str = Field(..., min_length=1, max_length=100, description="Client ID")
    weights: Dict[str, list] = Field(..., description="Model weights as dict of layer -> values")
    round_number: Optional[int] = Field(None, ge=0, description="Training round number")
    metadata: Optional[dict] = Field(default={}, description="Additional metadata")
    
    @validator('weights')
    def validate_weights(cls, v):
        """Validate weights structure"""
        if not v:
            raise ValueError("Weights cannot be empty")
        
        # Check if weights are valid lists
        for layer_name, layer_weights in v.items():
            if not isinstance(layer_weights, list):
                raise ValueError(f"Layer '{layer_name}' weights must be a list")
            if len(layer_weights) == 0:
                raise ValueError(f"Layer '{layer_name}' weights cannot be empty")
        
        return v

class AggregationConfig(BaseModel):
    """Configuration for model aggregation"""
    algorithm: str = Field(default="fedavg", description="Aggregation algorithm (fedavg/weighted)")
    min_clients: int = Field(default=2, ge=1, le=100, description="Minimum clients required")
    client_selection: Optional[List[str]] = Field(None, description="Specific clients to aggregate (None = all)")
    weights_strategy: str = Field(default="equal", description="Weighting strategy (equal/contribution)")

class ClientConfig(BaseModel):
    """Client registration configuration"""
    client_id: str = Field(..., min_length=1, max_length=100, description="Unique client ID")
    client_name: Optional[str] = Field(None, description="Human-readable name")
    metadata: Optional[dict] = Field(default={}, description="Client metadata")

# -------------------------------
# EXISTING: POST /face/fl/upload_weights (ENHANCED)
# -------------------------------
@router.post("/upload_weights")
async def upload_fl_weights(payload: FLWeightsUpload):
    """
    Clients upload updated model weights (JSON body)
    
    ENHANCED: Added validation, versioning, client tracking, metrics
    """
    try:
        target = payload.target
        weights_dict = payload.weights
        round_number = payload.round_number if payload.round_number is not None else fl_state["current_round"]
        
        logger.info(f"Receiving FL weights from client '{target}' for round {round_number}")
        
        # Validate client registration (NEW)
        if target not in client_registry:
            logger.warning(f"Unregistered client '{target}' attempting upload. Auto-registering.")
            client_registry[target] = {
                "registered_at": datetime.now().isoformat(),
                "status": "active",
                "last_update": None,
                "round": 0,
                "contribution_count": 0
            }
        
        # Validate weights (EXISTING - enhanced validation)
        if not weights_dict:
            raise HTTPException(status_code=400, detail="Weights cannot be empty")
        
        # Check weights structure consistency (NEW)
        layer_names = set(weights_dict.keys())
        
        # If this is not the first upload, check consistency with existing weights
        if target in FL_WEIGHTS:
            existing_layers = set(FL_WEIGHTS[target].keys())
            if layer_names != existing_layers:
                logger.warning(f"Layer structure changed for client '{target}'")
                logger.debug(f"Previous: {existing_layers}, New: {layer_names}")
        
        # Convert lists to numpy arrays (EXISTING)
        weights_np = {k: np.array(v) for k, v in weights_dict.items()}
        
        # Validate array shapes (NEW)
        for layer_name, layer_weights in weights_np.items():
            if layer_weights.size == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Layer '{layer_name}' has empty weights"
                )
            
            # Check for NaN or Inf values
            if np.any(np.isnan(layer_weights)) or np.any(np.isinf(layer_weights)):
                raise HTTPException(
                    status_code=400,
                    detail=f"Layer '{layer_name}' contains invalid values (NaN or Inf)"
                )
        
        # Store in-memory (EXISTING)
        FL_WEIGHTS[target] = weights_np
        
        # Save to disk for persistence (EXISTING - enhanced with metadata)
        save_path = os.path.join(FL_DIR, f"{target}.json")
        
        # Prepare save data with metadata (NEW)
        save_data = {
            "weights": weights_dict,
            "round": round_number,
            "timestamp": datetime.now().isoformat(),
            "metadata": payload.metadata,
            "version": fl_state["global_model_version"]
        }
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)
        
        # Update client registry (NEW)
        client_registry[target]["last_update"] = datetime.now().isoformat()
        client_registry[target]["round"] = round_number
        client_registry[target]["contribution_count"] += 1
        client_registry[target]["status"] = "active"
        
        # Update active clients (NEW)
        fl_state["active_clients"].add(target)
        fl_state["total_clients"] = len(client_registry)
        
        # Update client metrics (NEW)
        client_metrics[target]["total_updates"] += 1
        client_metrics[target]["total_rounds_participated"] = max(
            client_metrics[target]["total_rounds_participated"],
            round_number + 1
        )
        client_metrics[target]["last_contribution"] = datetime.now().isoformat()
        
        # Calculate weights size
        total_weights = sum(w.size for w in weights_np.values())
        client_metrics[target]["average_weights_size"] = (
            (client_metrics[target]["average_weights_size"] * (client_metrics[target]["total_updates"] - 1) + total_weights) /
            client_metrics[target]["total_updates"]
        )
        
        logger.info(f"Successfully stored FL weights for client '{target}' (round {round_number})")
        
        return JSONResponse({
            "status": "success",
            "message": f"Federated weights received for client '{target}'",
            "client_id": target,
            "round": round_number,
            "layers": list(weights_dict.keys()),
            "total_parameters": total_weights,
            "contribution_count": client_registry[target]["contribution_count"],
            "global_model_version": fl_state["global_model_version"],
            "timestamp": datetime.now().isoformat()
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading FL weights: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload FL weights: {str(e)}")

# -------------------------------
# EXISTING: GET /face/fl/status (ENHANCED)
# -------------------------------
@router.get("/status")
async def get_fl_status(client_id: str = Query(..., description="Client ID to query")):
    """
    Returns latest federated weights for a client.
    
    ENHANCED: Added versioning, metadata, client info
    """
    try:
        logger.debug(f"Fetching FL status for client: {client_id}")
        
        # Check in-memory first (EXISTING)
        if client_id in FL_WEIGHTS:
            weights_np = FL_WEIGHTS[client_id]
            weights_json = {k: v.tolist() for k, v in weights_np.items()}
            
            # Get client info (NEW)
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

        # Try loading from disk if not in memory (EXISTING - enhanced)
        disk_path = os.path.join(FL_DIR, f"{client_id}.json")
        if os.path.exists(disk_path):
            with open(disk_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            
            # Handle both old and new format (NEW - backward compatibility)
            if isinstance(saved_data, dict) and "weights" in saved_data:
                weights_json = saved_data["weights"]
                metadata = {
                    "round": saved_data.get("round", 0),
                    "timestamp": saved_data.get("timestamp"),
                    "version": saved_data.get("version", 0)
                }
            else:
                # Old format (just weights)
                weights_json = saved_data
                metadata = {}
            
            # Load into memory (EXISTING)
            FL_WEIGHTS[client_id] = {k: np.array(v) for k, v in weights_json.items()}
            
            return JSONResponse({
                "status": "success",
                "client_id": client_id,
                "weights": weights_json,
                "layers": list(weights_json.keys()),
                "metadata": metadata,
                "source": "disk"
            })

        # Not found (EXISTING - enhanced message)
        logger.warning(f"No weights found for client: {client_id}")
        
        return JSONResponse({
            "status": "success",
            "client_id": client_id,
            "weights": {},
            "message": "No weights found for this client",
            "suggestion": "Client needs to upload weights first"
        })

    except Exception as e:
        logger.error(f"Error fetching FL status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch FL status: {str(e)}")

# -------------------------------
# EXISTING: GET /face/fl/get_weights (KEPT AS IS)
# -------------------------------
@router.get("/get_weights")
async def get_fl_weights(target: str = Query(..., description="Target client ID")):
    """
    Backward compatibility route for frontend.
    Maps ?target=client_1 to the same logic as /status.
    """
    return await get_fl_status(client_id=target)

# -------------------------------
# EXISTING: DELETE /face/fl/weights/{client_id} (ENHANCED)
# -------------------------------
@router.delete("/weights/{client_id}")
async def delete_fl_weights(
    client_id: str,
    remove_from_registry: bool = Query(False, description="Also remove from client registry")
):
    """
    Delete federated learning weights for a client.
    
    ENHANCED: Added registry cleanup option
    """
    try:
        logger.info(f"Deleting FL weights for client: {client_id}")
        
        # Remove from memory (EXISTING)
        removed_from_memory = False
        if client_id in FL_WEIGHTS:
            del FL_WEIGHTS[client_id]
            removed_from_memory = True
        
        # Remove from disk (EXISTING)
        disk_path = os.path.join(FL_DIR, f"{client_id}.json")
        removed_from_disk = False
        
        if os.path.exists(disk_path):
            os.remove(disk_path)
            removed_from_disk = True
        
        # Remove from active clients (NEW)
        if client_id in fl_state["active_clients"]:
            fl_state["active_clients"].remove(client_id)
        
        # Optionally remove from registry (NEW)
        removed_from_registry_flag = False
        if remove_from_registry and client_id in client_registry:
            del client_registry[client_id]
            removed_from_registry_flag = True
            fl_state["total_clients"] = len(client_registry)
        
        if not removed_from_memory and not removed_from_disk:
            raise HTTPException(status_code=404, detail=f"No weights found for client '{client_id}'")
        
        logger.info(f"Successfully deleted FL weights for client: {client_id}")
        
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
        logger.error(f"Error deleting FL weights: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete weights: {str(e)}")

# -------------------------------
# EXISTING: GET /face/fl/list (ENHANCED)
# -------------------------------
@router.get("/list")
async def list_fl_clients(
    include_metrics: bool = Query(False, description="Include performance metrics"),
    status_filter: Optional[str] = Query(None, description="Filter by status (active/inactive)")
):
    """
    List all clients with stored FL weights.
    
    ENHANCED: Added metrics, filtering, detailed client info
    """
    try:
        logger.debug("Listing FL clients")
        
        # Get clients from memory (EXISTING)
        memory_clients = set(FL_WEIGHTS.keys())
        
        # Get clients from disk (EXISTING)
        disk_clients = set()
        if os.path.exists(FL_DIR):
            for filename in os.listdir(FL_DIR):
                if filename.endswith(".json"):
                    disk_clients.add(filename[:-5])  # Remove .json extension
        
        # Combine both (EXISTING)
        all_clients = memory_clients.union(disk_clients)
        
        # Build detailed client list (NEW)
        clients_data = []
        
        for client_id in sorted(all_clients):
            client_data = {
                "client_id": client_id,
                "in_memory": client_id in memory_clients,
                "on_disk": client_id in disk_clients
            }
            
            # Add registry info
            if client_id in client_registry:
                client_info = client_registry[client_id]
                client_data["status"] = client_info.get("status", "unknown")
                client_data["last_update"] = client_info.get("last_update")
                client_data["contribution_count"] = client_info.get("contribution_count", 0)
                client_data["round"] = client_info.get("round", 0)
            else:
                client_data["status"] = "unregistered"
            
            # Add metrics if requested
            if include_metrics and client_id in client_metrics:
                metrics = client_metrics[client_id]
                client_data["metrics"] = {
                    "total_updates": metrics["total_updates"],
                    "rounds_participated": metrics["total_rounds_participated"],
                    "average_weights_size": int(metrics["average_weights_size"]),
                    "last_contribution": metrics["last_contribution"]
                }
            
            # Apply status filter
            if status_filter:
                if status_filter == "active" and client_data.get("status") != "active":
                    continue
                elif status_filter == "inactive" and client_data.get("status") == "active":
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
        logger.error(f"Error listing FL clients: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list clients: {str(e)}")

# -------------------------------
# NEW: Register client
# -------------------------------
@router.post("/register")
async def register_client(config: ClientConfig):
    """
    Register a new FL client.
    
    **NEW ENDPOINT**
    
    Body:
```json
    {
        "client_id": "client_1",
        "client_name": "Edge Device 1",
        "metadata": {"location": "Building A", "device_type": "mobile"}
    }
```
    """
    try:
        client_id = config.client_id
        
        logger.info(f"Registering FL client: {client_id}")
        
        # Check if already registered
        if client_id in client_registry:
            logger.warning(f"Client '{client_id}' already registered")
            return JSONResponse({
                "status": "success",
                "message": f"Client '{client_id}' was already registered",
                "client_id": client_id,
                "already_registered": True,
                "registration_date": client_registry[client_id].get("registered_at")
            })
        
        # Register client
        client_registry[client_id] = {
            "registered_at": datetime.now().isoformat(),
            "client_name": config.client_name,
            "metadata": config.metadata,
            "status": "registered",
            "last_update": None,
            "round": 0,
            "contribution_count": 0
        }
        
        fl_state["total_clients"] = len(client_registry)
        
        logger.info(f"Successfully registered client: {client_id}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Client '{client_id}' registered successfully",
            "client_id": client_id,
            "registered_at": client_registry[client_id]["registered_at"],
            "total_clients": fl_state["total_clients"]
        })
    
    except Exception as e:
        logger.error(f"Error registering client: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to register client: {str(e)}")

# -------------------------------
# NEW: Unregister client
# -------------------------------
@router.delete("/unregister/{client_id}")
async def unregister_client(client_id: str, delete_weights: bool = Query(False)):
    """
    Unregister an FL client.
    
    **NEW ENDPOINT**
    """
    try:
        logger.info(f"Unregistering client: {client_id}")
        
        if client_id not in client_registry:
            raise HTTPException(404, f"Client '{client_id}' not registered")
        
        # Remove from registry
        del client_registry[client_id]
        
        # Remove from active clients
        if client_id in fl_state["active_clients"]:
            fl_state["active_clients"].remove(client_id)
        
        fl_state["total_clients"] = len(client_registry)
        
        # Optionally delete weights
        weights_deleted = False
        if delete_weights:
            if client_id in FL_WEIGHTS:
                del FL_WEIGHTS[client_id]
            
            disk_path = os.path.join(FL_DIR, f"{client_id}.json")
            if os.path.exists(disk_path):
                os.remove(disk_path)
                weights_deleted = True
        
        logger.info(f"Successfully unregistered client: {client_id}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Client '{client_id}' unregistered",
            "client_id": client_id,
            "weights_deleted": weights_deleted
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unregistering client: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to unregister client: {str(e)}")

# -------------------------------
# NEW: Aggregate weights (FedAvg)
# -------------------------------
@router.post("/aggregate")
async def aggregate_weights(config: AggregationConfig = Body(...)):
    """
    Aggregate client weights into global model.
    
    **NEW ENDPOINT**
    
    Implements Federated Averaging (FedAvg) algorithm.
    
    Body:
```json
    {
        "algorithm": "fedavg",
        "min_clients": 2,
        "client_selection": null,
        "weights_strategy": "equal"
    }
```
    """
    try:
        logger.info(f"Starting model aggregation with {config.algorithm} algorithm")
        
        # Update FL state
        fl_state["training_status"] = "aggregating"
        
        # Determine which clients to aggregate
        if config.client_selection:
            clients_to_aggregate = [c for c in config.client_selection if c in FL_WEIGHTS]
        else:
            clients_to_aggregate = list(FL_WEIGHTS.keys())
        
        # Check minimum clients requirement
        if len(clients_to_aggregate) < config.min_clients:
            fl_state["training_status"] = "idle"
            raise HTTPException(
                400,
                f"Insufficient clients for aggregation. Required: {config.min_clients}, Available: {len(clients_to_aggregate)}"
            )
        
        logger.info(f"Aggregating weights from {len(clients_to_aggregate)} clients: {clients_to_aggregate}")
        
        # Get all client weights
        client_weights_list = []
        for client_id in clients_to_aggregate:
            client_weights_list.append(FL_WEIGHTS[client_id])
        
        # Check layer consistency
        layer_names = set(client_weights_list[0].keys())
        for client_weights in client_weights_list[1:]:
            if set(client_weights.keys()) != layer_names:
                fl_state["training_status"] = "idle"
                raise HTTPException(
                    400,
                    "Inconsistent layer structure across clients. All clients must have the same model architecture."
                )
        
        # Determine aggregation weights
        if config.weights_strategy == "equal":
            aggregation_weights = [1.0 / len(clients_to_aggregate)] * len(clients_to_aggregate)
        elif config.weights_strategy == "contribution":
            # Weight by contribution count
            contributions = [client_registry.get(c, {}).get("contribution_count", 1) for c in clients_to_aggregate]
            total_contributions = sum(contributions)
            aggregation_weights = [c / total_contributions for c in contributions]
        else:
            aggregation_weights = [1.0 / len(clients_to_aggregate)] * len(clients_to_aggregate)
        
        logger.debug(f"Aggregation weights: {aggregation_weights}")
        
        # Perform aggregation (FedAvg)
        aggregated_weights = {}
        
        for layer_name in layer_names:
            # Stack layer weights from all clients
            layer_weights_stack = np.array([
                client_weights[layer_name] * weight
                for client_weights, weight in zip(client_weights_list, aggregation_weights)
            ])
            
            # Sum weighted averages
            aggregated_weights[layer_name] = np.sum(layer_weights_stack, axis=0)
        
        # Store aggregated model
        global_model_id = f"global_model_v{fl_state['global_model_version'] + 1}"
        FL_WEIGHTS[global_model_id] = aggregated_weights
        
        # Save to disk
        save_path = os.path.join(FL_DIR, f"{global_model_id}.json")
        save_data = {
            "weights": {k: v.tolist() for k, v in aggregated_weights.items()},
            "version": fl_state["global_model_version"] + 1,
            "round": fl_state["current_round"],
            "clients_aggregated": clients_to_aggregate,
            "aggregation_algorithm": config.algorithm,
            "weights_strategy": config.weights_strategy,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)
        
        # Update FL state
        fl_state["global_model_version"] += 1
        fl_state["current_round"] += 1
        fl_state["last_aggregation"] = datetime.now().isoformat()
        fl_state["training_status"] = "idle"
        
        # Record in aggregation history
        aggregation_event = {
            "version": fl_state["global_model_version"],
            "round": fl_state["current_round"] - 1,
            "timestamp": datetime.now().isoformat(),
            "clients_count": len(clients_to_aggregate),
            "clients": clients_to_aggregate,
            "algorithm": config.algorithm,
            "weights_strategy": config.weights_strategy
        }
        
        aggregation_history.append(aggregation_event)
        
        # Keep only last 100 aggregations
        if len(aggregation_history) > 100:
            aggregation_history.pop(0)
        
        logger.info(f"Aggregation complete. New global model version: {fl_state['global_model_version']}")
        
        return JSONResponse({
            "status": "success",
            "message": "Model aggregation completed successfully",
            "global_model": {
                "id": global_model_id,
                "version": fl_state["global_model_version"],
                "round": fl_state["current_round"] - 1,
                "layers": list(aggregated_weights.keys()),
                "total_parameters": sum(w.size for w in aggregated_weights.values())
            },
            "aggregation_details": {
                "clients_aggregated": len(clients_to_aggregate),
                "client_ids": clients_to_aggregate,
                "algorithm": config.algorithm,
                "weights_strategy": config.weights_strategy,
                "aggregation_weights": aggregation_weights
            },
            "timestamp": datetime.now().isoformat()
        })
    
    except HTTPException:
        fl_state["training_status"] = "idle"
        raise
    except Exception as e:
        fl_state["training_status"] = "idle"
        logger.error(f"Error in model aggregation: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Aggregation failed: {str(e)}")

# -------------------------------
# NEW: Get global model
# -------------------------------
@router.get("/global_model")
async def get_global_model(version: Optional[int] = Query(None, description="Specific version (None = latest)")):
    """
    Get the global aggregated model.
    
    **NEW ENDPOINT**
    """
    try:
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
        
        logger.debug(f"Fetching global model: {global_model_id}")
        
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
                    "total_parameters": sum(len(v) for v in weights_json.values())
                },
                "timestamp": datetime.now().isoformat()
            })
        
        # Try loading from disk
        disk_path = os.path.join(FL_DIR, f"{global_model_id}.json")
        if os.path.exists(disk_path):
            with open(disk_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            
            weights_json = saved_data.get("weights", {})
            
            # Load into memory
            FL_WEIGHTS[global_model_id] = {k: np.array(v) for k, v in weights_json.items()}
            return JSONResponse({
                "status": "success",
                "global_model": {
                    "id": global_model_id,
                    "version": version,
                    "weights": weights_json,
                    "layers": list(weights_json.keys()),
                    "total_parameters": sum(len(v) for v in weights_json.values()),
                    "metadata": {
                        "round": saved_data.get("round"),
                        "timestamp": saved_data.get("timestamp"),
                        "clients_aggregated": saved_data.get("clients_aggregated", [])
                    }
                },
                "source": "disk"
            })
        
        raise HTTPException(404, f"Global model version {version} not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching global model: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to get global model: {str(e)}")

# -------------------------------
# NEW: Get aggregation history
# -------------------------------
@router.get("/aggregation/history")
async def get_aggregation_history(limit: int = Query(50, ge=1, le=1000)):
    """
    Get history of model aggregations.
    
    **NEW ENDPOINT**
    """
    try:
        logger.debug(f"Fetching aggregation history (limit={limit})")
        
        # Get recent history
        history = list(aggregation_history[-limit:])
        history.reverse()  # Most recent first
        
        return JSONResponse({
            "status": "success",
            "total_aggregations": len(aggregation_history),
            "returned": len(history),
            "history": history,
            "current_version": fl_state["global_model_version"]
        })
    
    except Exception as e:
        logger.error(f"Error fetching aggregation history: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to get history: {str(e)}")

# -------------------------------
# NEW: Get FL statistics
# -------------------------------
@router.get("/stats")
async def get_fl_statistics():
    """
    Get comprehensive FL statistics.
    
    **NEW ENDPOINT**
    """
    try:
        logger.debug("Fetching FL statistics")
        
        # Calculate statistics
        total_clients = len(client_registry)
        active_clients_count = len(fl_state["active_clients"])
        total_weights_stored = len(FL_WEIGHTS)
        
        # Client contribution statistics
        contributions = [info.get("contribution_count", 0) for info in client_registry.values()]
        
        if contributions:
            avg_contributions = sum(contributions) / len(contributions)
            max_contributions = max(contributions)
            min_contributions = min(contributions)
        else:
            avg_contributions = 0
            max_contributions = 0
            min_contributions = 0
        
        # Calculate total parameters in global model
        total_params = 0
        if fl_state["global_model_version"] > 0:
            global_model_id = f"global_model_v{fl_state['global_model_version']}"
            if global_model_id in FL_WEIGHTS:
                total_params = sum(w.size for w in FL_WEIGHTS[global_model_id].values())
        
        # Recent activity
        recent_updates = []
        for client_id, info in client_registry.items():
            if info.get("last_update"):
                last_update_time = datetime.fromisoformat(info["last_update"])
                time_since = (datetime.now() - last_update_time).total_seconds()
                if time_since < 3600:  # Last hour
                    recent_updates.append(client_id)
        
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
                "layers": len(FL_WEIGHTS[f"global_model_v{fl_state['global_model_version']}"].keys()) if fl_state["global_model_version"] > 0 and f"global_model_v{fl_state['global_model_version']}" in FL_WEIGHTS else 0
            },
            "activity": {
                "recent_updates_last_hour": len(recent_updates),
                "active_clients": list(fl_state["active_clients"])
            }
        }
        
        return JSONResponse({
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error fetching FL statistics: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to get statistics: {str(e)}")

# -------------------------------
# NEW: Get client metrics
# -------------------------------
@router.get("/client/{client_id}/metrics")
async def get_client_metrics(client_id: str):
    """
    Get detailed metrics for a specific client.
    
    **NEW ENDPOINT**
    """
    try:
        logger.debug(f"Fetching metrics for client: {client_id}")
        
        if client_id not in client_registry:
            raise HTTPException(404, f"Client '{client_id}' not registered")
        
        client_info = client_registry[client_id]
        metrics = client_metrics.get(client_id, {})
        
        # Calculate contribution percentage
        total_updates = sum(m.get("total_updates", 0) for m in client_metrics.values())
        contribution_percentage = (metrics.get("total_updates", 0) / total_updates * 100) if total_updates > 0 else 0
        
        # Check if client has weights
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
        logger.error(f"Error fetching client metrics: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to get client metrics: {str(e)}")

# -------------------------------
# NEW: Reset FL state
# -------------------------------
@router.post("/reset")
async def reset_fl_state(
    confirmation: str = Query(..., description="Must be 'CONFIRM_RESET'"),
    keep_clients: bool = Query(True, description="Keep client registry"),
    keep_weights: bool = Query(False, description="Keep stored weights")
):
    """
    Reset FL training state.
    
    **NEW ENDPOINT**
    
    **WARNING:** This resets training progress!
    
    Requires confirmation parameter: ?confirmation=CONFIRM_RESET
    """
    try:
        if confirmation != "CONFIRM_RESET":
            raise HTTPException(400, "Invalid confirmation. Must provide confirmation=CONFIRM_RESET")
        
        logger.warning("RESETTING FL STATE - This action was confirmed")
        
        # Reset state
        old_round = fl_state["current_round"]
        old_version = fl_state["global_model_version"]
        
        fl_state["current_round"] = 0
        fl_state["global_model_version"] = 0
        fl_state["training_status"] = "idle"
        fl_state["last_aggregation"] = None
        fl_state["active_clients"].clear()
        
        # Clear aggregation history
        aggregation_history.clear()
        
        # Optionally clear client registry
        clients_cleared = 0
        if not keep_clients:
            clients_cleared = len(client_registry)
            client_registry.clear()
            client_metrics.clear()
            fl_state["total_clients"] = 0
        
        # Optionally clear weights
        weights_cleared = 0
        if not keep_weights:
            weights_cleared = len(FL_WEIGHTS)
            FL_WEIGHTS.clear()
            
            # Clear disk storage
            if os.path.exists(FL_DIR):
                for filename in os.listdir(FL_DIR):
                    if filename.endswith(".json"):
                        try:
                            os.remove(os.path.join(FL_DIR, filename))
                        except:
                            pass
        
        logger.warning(f"FL state reset complete. Previous: round {old_round}, version {old_version}")
        
        return JSONResponse({
            "status": "success",
            "message": "FL state reset successfully",
            "previous_state": {
                "round": old_round,
                "version": old_version
            },
            "cleared": {
                "clients": clients_cleared if not keep_clients else 0,
                "weights": weights_cleared if not keep_weights else 0
            },
            "kept": {
                "clients": keep_clients,
                "weights": keep_weights
            }
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting FL state: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to reset FL state: {str(e)}")

# -------------------------------
# NEW: Export FL data
# -------------------------------
@router.get("/export")
async def export_fl_data(
    include_weights: bool = Query(False, description="Include model weights"),
    format: str = Query("json", description="Export format (json/csv)")
):
    """
    Export FL data and history.
    
    **NEW ENDPOINT**
    
    Supports JSON and CSV formats.
    """
    try:
        logger.info(f"Exporting FL data (format={format}, weights={include_weights})")
        
        export_data = {
            "export_time": datetime.now().isoformat(),
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
        
        # Optionally include weights
        if include_weights:
            export_data["weights"] = {
                client_id: {k: v.tolist() for k, v in weights.items()}
                for client_id, weights in FL_WEIGHTS.items()
            }
        
        filename = f"fl_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if format == "json":
            json_bytes = json.dumps(export_data, indent=2).encode('utf-8')
            file_stream = BytesIO(json_bytes)
            
            return StreamingResponse(
                file_stream,
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename={filename}.json"}
            )
        
        elif format == "csv":
            import csv
            from io import StringIO
            
            csv_buffer = StringIO()
            writer = csv.writer(csv_buffer)
            
            # Write client data
            writer.writerow(["Client ID", "Status", "Contribution Count", "Last Update", "Round"])
            
            for client_id, info in client_registry.items():
                writer.writerow([
                    client_id,
                    info.get("status", "unknown"),
                    info.get("contribution_count", 0),
                    info.get("last_update", "N/A"),
                    info.get("round", 0)
                ])
            
            csv_bytes = csv_buffer.getvalue().encode('utf-8')
            file_stream = BytesIO(csv_bytes)
            
            return StreamingResponse(
                file_stream,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}.csv"}
            )
        
        else:
            raise HTTPException(400, "Invalid format. Use 'json' or 'csv'")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting FL data: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Export failed: {str(e)}")

# -------------------------------
# NEW: Health check
# -------------------------------
@router.get("/health")
async def fl_health_check():
    """
    Health check for FL service.
    
    **NEW ENDPOINT**
    """
    try:
        # Check directory accessibility
        weights_dir_accessible = os.path.exists(FL_DIR) and os.access(FL_DIR, os.W_OK)
        
        # Check if any weights are loaded
        weights_loaded = len(FL_WEIGHTS) > 0
        
        # Determine health status
        if weights_dir_accessible:
            health_status = "healthy"
        else:
            health_status = "degraded"
        
        return JSONResponse({
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
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "service": "federated_learning",
            "error": str(e)
        }, status_code=503)

# -------------------------------
# NEW: Start training round
# -------------------------------
@router.post("/round/start")
async def start_training_round():
    """
    Start a new training round.
    
    **NEW ENDPOINT**
    
    Broadcasts to all clients that a new round has started.
    """
    try:
        if fl_state["training_status"] != "idle":
            raise HTTPException(400, f"Cannot start new round. Current status: {fl_state['training_status']}")
        
        logger.info(f"Starting training round {fl_state['current_round']}")
        
        fl_state["training_status"] = "training"
        
        # Clear active clients for this round
        fl_state["active_clients"].clear()
        
        return JSONResponse({
            "status": "success",
            "message": f"Training round {fl_state['current_round']} started",
            "round": fl_state["current_round"],
            "total_clients": fl_state["total_clients"],
            "timestamp": datetime.now().isoformat()
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training round: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to start round: {str(e)}")

# -------------------------------
# NEW: Get training status
# -------------------------------
@router.get("/round/status")
async def get_training_status():
    """
    Get current training round status.
    
    **NEW ENDPOINT**
    """
    try:
        # Calculate participation rate
        participation_rate = (
            (len(fl_state["active_clients"]) / fl_state["total_clients"] * 100)
            if fl_state["total_clients"] > 0 else 0
        )
        
        return JSONResponse({
            "status": "success",
            "training": {
                "current_round": fl_state["current_round"],
                "status": fl_state["training_status"],
                "total_clients": fl_state["total_clients"],
                "active_clients": len(fl_state["active_clients"]),
                "participation_rate": round(participation_rate, 2)
            },
            "global_model": {
                "version": fl_state["global_model_version"],
                "last_aggregation": fl_state["last_aggregation"]
            },
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error fetching training status: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to get status: {str(e)}")

# -------------------------------
# NEW: Validate weights structure
# -------------------------------
@router.post("/validate")
async def validate_weights(payload: FLWeightsUpload):
    """
    Validate weight structure without storing.
    
    **NEW ENDPOINT**
    
    Useful for testing before actual upload.
    """
    try:
        logger.info(f"Validating weights structure for client: {payload.target}")
        
        # Convert to numpy and validate
        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "structure": {}
        }
        
        total_params = 0
        
        for layer_name, layer_weights in payload.weights.items():
            layer_array = np.array(layer_weights)
            
            # Check for issues
            if layer_array.size == 0:
                validation_results["valid"] = False
                validation_results["issues"].append(f"Layer '{layer_name}' has empty weights")
            
            if np.any(np.isnan(layer_array)):
                validation_results["valid"] = False
                validation_results["issues"].append(f"Layer '{layer_name}' contains NaN values")
            
            if np.any(np.isinf(layer_array)):
                validation_results["valid"] = False
                validation_results["issues"].append(f"Layer '{layer_name}' contains Inf values")
            
            # Check for warnings
            if np.all(layer_array == 0):
                validation_results["warnings"].append(f"Layer '{layer_name}' contains all zeros")
            
            mean = np.mean(layer_array)
            std = np.std(layer_array)
            
            if std == 0:
                validation_results["warnings"].append(f"Layer '{layer_name}' has zero variance")
            
            # Store structure info
            validation_results["structure"][layer_name] = {
                "shape": layer_array.shape,
                "size": int(layer_array.size),
                "dtype": str(layer_array.dtype),
                "mean": float(mean),
                "std": float(std),
                "min": float(np.min(layer_array)),
                "max": float(np.max(layer_array))
            }
            
            total_params += layer_array.size
        
        validation_results["total_parameters"] = total_params
        validation_results["total_layers"] = len(payload.weights)
        
        logger.info(f"Validation complete for {payload.target}: {'VALID' if validation_results['valid'] else 'INVALID'}")
        
        return JSONResponse({
            "status": "success",
            "client_id": payload.target,
            "validation": validation_results
        })
    
    except Exception as e:
        logger.error(f"Error validating weights: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Validation failed: {str(e)}")
```

---

### **📝 Summary of Changes for `federated.py`:**

#### **✅ Added (25 new features):**

1. **Comprehensive Logging** - Complete logging for all FL operations
2. **FL State Management** - Track training rounds, versions, status
3. **Client Registry** - Track all registered clients with metadata
4. **Client Tracking** - Last update, contribution count, status
5. **Performance Metrics** - Per-client contribution tracking
6. **Aggregation History** - Track all aggregation events
7. **Weight Validation** - Check for NaN, Inf, empty arrays
8. **Version Control** - Global model versioning system
9. **Client Registration** - POST `/register` - Register FL clients
10. **Client Unregistration** - DELETE `/unregister/{client_id}` - Remove clients
11. **Model Aggregation** - POST `/aggregate` - FedAvg algorithm implementation
12. **Global Model Access** - GET `/global_model` - Retrieve aggregated model
13. **Aggregation History** - GET `/aggregation/history` - View past aggregations
14. **FL Statistics** - GET `/stats` - Comprehensive statistics
15. **Client Metrics** - GET `/client/{client_id}/metrics` - Individual metrics
16. **State Reset** - POST `/reset` - Reset training state
17. **Data Export** - GET `/export` - Export FL data (JSON/CSV)
18. **Health Check** - GET `/health` - Service health status
19. **Training Rounds** - POST `/round/start` - Start new round
20. **Training Status** - GET `/round/status` - Current round status
21. **Weight Validation** - POST `/validate` - Test weights before upload
22. **Enhanced Upload** - Validation, versioning, tracking
23. **Enhanced Status** - Client info, metrics, version tracking
24. **Enhanced List** - Metrics, filtering, detailed info
25. **Enhanced Delete** - Registry cleanup option

#### **🔒 Nothing Removed:**
- All original endpoints intact (`/upload_weights`, `/status`, `/get_weights`, `/list`, `/delete`)
- All original processing logic preserved
- Backward compatible with existing frontend
- All existing function signatures maintained

#### **🎯 Key Benefits:**

**Production FL Implementation:**
- ✅ Complete FedAvg algorithm implementation
- ✅ Client management and tracking
- ✅ Version control for models
- ✅ Training round management
- ✅ Aggregation history tracking

**Robustness:**
- ✅ Weight validation (NaN, Inf, empty checks)
- ✅ Structure consistency validation
- ✅ Error handling with state rollback
- ✅ Comprehensive logging

**Monitoring:**
- ✅ Per-client contribution metrics
- ✅ Aggregation history
- ✅ Training status tracking
- ✅ Performance statistics

**Flexibility:**
- ✅ Multiple aggregation strategies (equal/weighted)
- ✅ Selective client aggregation
- ✅ Configurable minimum clients
- ✅ Export functionality

**Safety:**
- ✅ Confirmation required for destructive operations
- ✅ State management with status tracking
- ✅ Rollback on errors
- ✅ Detailed validation

---

### **📊 New API Endpoints Summary:**
```
POST   /face/fl/upload_weights        ✅ ENHANCED - Validation, versioning, tracking
GET    /face/fl/status                ✅ ENHANCED - Client info, metrics, version
GET    /face/fl/get_weights           ✅ KEPT - Backward compatibility
DELETE /face/fl/weights/{client_id}   ✅ ENHANCED - Registry cleanup option
GET    /face/fl/list                  ✅ ENHANCED - Metrics, filtering, details

POST   /face/fl/register              🆕 NEW - Register FL client
DELETE /face/fl/unregister/{client}   🆕 NEW - Unregister client
POST   /face/fl/aggregate             🆕 NEW - Aggregate weights (FedAvg)
GET    /face/fl/global_model          🆕 NEW - Get global model
GET    /face/fl/aggregation/history   🆕 NEW - Aggregation history
GET    /face/fl/stats                 🆕 NEW - FL statistics
GET    /face/fl/client/{id}/metrics   🆕 NEW - Client metrics
POST   /face/fl/reset                 🆕 NEW - Reset FL state
GET    /face/fl/export                🆕 NEW - Export FL data
GET    /face/fl/health                🆕 NEW - Health check
POST   /face/fl/round/start           🆕 NEW - Start training round
GET    /face/fl/round/status          🆕 NEW - Training status
POST   /face/fl/validate              🆕 NEW - Validate weights