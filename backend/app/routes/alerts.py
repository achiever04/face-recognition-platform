# backend/app/routes/alerts.py

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging
import json
from io import BytesIO

from app.services.alert_service import alert_service

# Initialize logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["Alert Management"])

# -------------------------------
# Pydantic Models for Request Validation
# -------------------------------
class GeofenceCreate(BaseModel):
    """Model for creating a geofence zone"""
    zone_name: str = Field(..., min_length=1, max_length=100, description="Name of the geofence zone")
    camera_ids: List[int] = Field(..., min_items=1, description="List of camera IDs in this zone")
    description: str = Field(default="", max_length=500, description="Optional description")
    enabled: bool = Field(default=True, description="Whether zone is active")

class AlertAcknowledge(BaseModel):
    """Model for acknowledging an alert"""
    alert_id: str = Field(..., description="Alert ID to acknowledge")
    acknowledged_by: str = Field(..., min_length=1, description="Name/ID of person acknowledging")
    notes: Optional[str] = Field(default=None, max_length=1000, description="Optional notes")

# -------------------------------
# GET /alerts - Get all alerts (ENHANCED)
# -------------------------------
@router.get("/")
async def get_alerts(
    target: Optional[str] = Query(None, description="Filter by target person"),
    priority: Optional[str] = Query(None, description="Filter by priority (low/medium/high/critical)"),
    since_minutes: Optional[int] = Query(None, ge=1, le=10080, description="Only alerts from last N minutes (max 1 week)"),
    limit: Optional[int] = Query(50, ge=1, le=1000, description="Maximum number of alerts to return"),
    offset: Optional[int] = Query(0, ge=0, description="Offset for pagination"),
    sort_by: Optional[str] = Query("timestamp", description="Sort by field (timestamp, priority, target)"),
    sort_order: Optional[str] = Query("desc", description="Sort order (asc/desc)")
):
    """
    Get alerts with optional filtering, pagination, and sorting.
    """
    try:
        logger.info("Fetching alerts: target=%s priority=%s limit=%s offset=%s", target, priority, limit, offset)

        # Validate priority
        if priority and priority not in {"low", "medium", "high", "critical"}:
            raise HTTPException(status_code=400, detail="Invalid priority. Must be one of: low, medium, high, critical")

        # Validate sort_by and sort_order
        if sort_by not in {"timestamp", "priority", "target"}:
            raise HTTPException(status_code=400, detail="Invalid sort_by. Must be one of: timestamp, priority, target")
        if sort_order not in {"asc", "desc"}:
            raise HTTPException(status_code=400, detail="Invalid sort_order. Must be 'asc' or 'desc'")

        # Calculate time filter
        since = None
        if since_minutes:
            since = datetime.now() - timedelta(minutes=since_minutes)

        # Retrieve alerts from the service
        # The service API expected: get_alerts(target_name, priority, since, limit)
        # We request limit + offset to allow slicing locally if service doesn't support offset
        service_limit = None if limit is None else (limit + offset)
        alerts = alert_service.get_alerts(
            target_name=target,
            priority=priority,
            since=since,
            limit=service_limit
        ) or []

        # Apply offset locally
        if offset:
            alerts = alerts[offset:]

        # Apply sorting
        reverse = (sort_order == "desc")
        if sort_by == "priority":
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            alerts.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 99), reverse=reverse)
        elif sort_by == "target":
            alerts.sort(key=lambda x: x.get("target", ""), reverse=reverse)
        else:
            # timestamp: attempt to sort by timestamp field if present
            try:
                alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=reverse)
            except Exception:
                pass  # keep order as provided by service

        # Attempt to get accurate total count if service provides a count method
        total = None
        try:
            if hasattr(alert_service, "count_alerts"):
                total = alert_service.count_alerts(target_name=target, priority=priority, since=since)
        except Exception as e:
            logger.debug("count_alerts not available or failed: %s", e)

        if total is None:
            # Fallback approximate total
            total = len(alerts) + (offset or 0)

        logger.info("Successfully retrieved %d alerts (returned: %d)", len(alerts), len(alerts))

        return JSONResponse({
            "status": "success",
            "count": len(alerts),
            "total": total,
            "offset": offset,
            "limit": limit,
            "alerts": alerts,
            "filters": {
                "target": target,
                "priority": priority,
                "since_minutes": since_minutes
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching alerts: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

# -------------------------------
# GET /alerts/latest - Get latest alert
# -------------------------------
@router.get("/latest")
async def get_latest_alert(target: Optional[str] = Query(None)):
    """
    Get most recent alert (optionally for specific target).
    """
    try:
        logger.debug("Fetching latest alert for target: %s", target)
        alert = alert_service.get_latest_alert(target_name=target)
        if not alert:
            return JSONResponse({"status": "success", "alert": None, "message": "No alerts found"})
        return JSONResponse({"status": "success", "alert": alert})
    except Exception as e:
        logger.error("Error fetching latest alert: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get latest alert: {str(e)}")

# -------------------------------
# GET /alerts/watchlist - Get watchlist
# -------------------------------
@router.get("/watchlist")
async def get_watchlist():
    """
    Get all persons on the watchlist.
    """
    try:
        logger.debug("Fetching watchlist")
        watchlist = alert_service.get_watchlist() or []
        return JSONResponse({"status": "success", "count": len(watchlist), "watchlist": sorted(watchlist)})
    except Exception as e:
        logger.error("Error fetching watchlist: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get watchlist: {str(e)}")

# -------------------------------
# POST /alerts/watchlist/{target} - Add to watchlist
# -------------------------------
@router.post("/watchlist/{target}")
async def add_to_watchlist(target: str):
    """
    Add a person to the watchlist.
    """
    try:
        if not target or len(target.strip()) == 0:
            raise HTTPException(status_code=400, detail="Target name cannot be empty")
        if len(target) > 255:
            raise HTTPException(status_code=400, detail="Target name too long (max 255 characters)")

        logger.info("Adding to watchlist: %s", target)
        result = alert_service.add_to_watchlist(target)

        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to add to watchlist"))

        logger.info("Successfully added %s to watchlist", target)
        return JSONResponse({"status": "success", "message": result.get("message", "Added to watchlist"), "target": target})

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error adding to watchlist: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add to watchlist: {str(e)}")

# -------------------------------
# DELETE /alerts/watchlist/{target} - Remove from watchlist
# -------------------------------
@router.delete("/watchlist/{target}")
async def remove_from_watchlist(target: str):
    """
    Remove a person from the watchlist.
    """
    try:
        logger.info("Removing from watchlist: %s", target)
        result = alert_service.remove_from_watchlist(target)
        if not result.get("success", False):
            raise HTTPException(status_code=404, detail=result.get("message", "Not found in watchlist"))
        logger.info("Successfully removed %s from watchlist", target)
        return JSONResponse({"status": "success", "message": result.get("message", "Removed from watchlist"), "target": target})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error removing from watchlist: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to remove from watchlist: {str(e)}")

# -------------------------------
# GET /alerts/geofences - Get all geofences
# -------------------------------
@router.get("/geofences")
async def get_geofences():
    """
    Get all configured geo-fence zones.
    """
    try:
        logger.debug("Fetching geofences")
        geofences = alert_service.get_geofences() or {}
        return JSONResponse({"status": "success", "count": len(geofences), "geofences": geofences})
    except Exception as e:
        logger.error("Error fetching geofences: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get geofences: {str(e)}")

# -------------------------------
# POST /alerts/geofences - Create geofence (NEW)
# -------------------------------
@router.post("/geofences")
async def create_geofence(geofence: GeofenceCreate):
    """
    Create a new geo-fence zone.
    """
    try:
        logger.info("Creating geofence: %s with cameras %s", geofence.zone_name, geofence.camera_ids)
        result = alert_service.create_geofence(
            zone_name=geofence.zone_name,
            camera_ids=geofence.camera_ids,
            description=geofence.description,
            enabled=geofence.enabled
        )

        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to create geofence"))

        logger.info("Successfully created geofence: %s", geofence.zone_name)
        return JSONResponse({
            "status": "success",
            "message": result.get("message", "Geofence created"),
            "zone": {
                "name": geofence.zone_name,
                "camera_ids": geofence.camera_ids,
                "description": geofence.description,
                "enabled": geofence.enabled
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating geofence: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create geofence: {str(e)}")

# -------------------------------
# DELETE /alerts/geofences/{zone_name} - Delete geofence (NEW)
# -------------------------------
@router.delete("/geofences/{zone_name}")
async def delete_geofence(zone_name: str):
    """
    Delete a geo-fence zone.
    """
    try:
        logger.info("Deleting geofence: %s", zone_name)

        geofences = alert_service.get_geofences() or {}
        if zone_name not in geofences:
            raise HTTPException(status_code=404, detail=f"Geofence '{zone_name}' not found")

        # If alert_service implements delete_geofence, use it
        if hasattr(alert_service, "delete_geofence"):
            try:
                res = alert_service.delete_geofence(zone_name)
                if not res.get("success", False):
                    raise HTTPException(status_code=500, detail=res.get("message", "Failed to delete geofence via service"))
            except HTTPException:
                raise
            except Exception as e:
                logger.error("alert_service.delete_geofence failed: %s", e, exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to delete geofence: {str(e)}")
            logger.info("Successfully deleted geofence via service: %s", zone_name)
            return JSONResponse({"status": "success", "message": f"Geofence '{zone_name}' deleted", "zone_name": zone_name})
        else:
            # Service does not implement deletion - return success but warn user that this was a logical deletion only
            logger.warning("alert_service.delete_geofence not implemented - returning success response but no service-side deletion performed")
            return JSONResponse({
                "status": "success",
                "message": f"Geofence '{zone_name}' deletion acknowledged (no-op: service delete not implemented)",
                "zone_name": zone_name,
                "note": "Implement alert_service.delete_geofence(zone_name) for persistent deletion"
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting geofence: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete geofence: {str(e)}")

# -------------------------------
# GET /alerts/stats - Get alert statistics
# -------------------------------
@router.get("/stats")
async def get_alert_stats():
    """
    Get alert statistics.
    """
    try:
        logger.debug("Fetching alert statistics")
        stats = alert_service.get_statistics() or {}
        return JSONResponse({"status": "success", "statistics": stats, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        logger.error("Error fetching statistics: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

# -------------------------------
# POST /alerts/acknowledge - Acknowledge alert (NEW)
# -------------------------------
@router.post("/acknowledge")
async def acknowledge_alert(ack: AlertAcknowledge):
    """
    Acknowledge an alert (mark as seen/handled).
    """
    try:
        logger.info("Acknowledging alert %s by %s", ack.alert_id, ack.acknowledged_by)

        if hasattr(alert_service, "acknowledge_alert"):
            try:
                res = alert_service.acknowledge_alert(
                    alert_id=ack.alert_id,
                    acknowledged_by=ack.acknowledged_by,
                    notes=ack.notes
                )
                if not res.get("success", False):
                    raise HTTPException(status_code=400, detail=res.get("message", "Failed to acknowledge alert"))
                logger.info("Alert acknowledged via service: %s", ack.alert_id)
                return JSONResponse({
                    "status": "success",
                    "message": res.get("message", f"Alert '{ack.alert_id}' acknowledged"),
                    "alert_id": ack.alert_id,
                    "acknowledged_by": ack.acknowledged_by,
                    "acknowledged_at": datetime.now().isoformat()
                })
            except HTTPException:
                raise
            except Exception as e:
                logger.error("alert_service.acknowledge_alert failed: %s", e, exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")
        else:
            # Service does not implement acknowledgement - return success but note no persistent change
            logger.warning("alert_service.acknowledge_alert not implemented - returning success response (no-op)")
            return JSONResponse({
                "status": "success",
                "message": f"Alert '{ack.alert_id}' acknowledged (no-op: service ack not implemented)",
                "alert_id": ack.alert_id,
                "acknowledged_by": ack.acknowledged_by,
                "acknowledged_at": datetime.now().isoformat(),
                "note": "Implement alert_service.acknowledge_alert for persistent acknowledgements"
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error acknowledging alert: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")

# -------------------------------
# GET /alerts/export - Export alerts to JSON (NEW)
# -------------------------------
@router.get("/export")
async def export_alerts(
    target: Optional[str] = Query(None, description="Filter by target person"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    since_minutes: Optional[int] = Query(None, ge=1, description="Time range in minutes"),
    format: str = Query("json", description="Export format (json only for now)")
):
    """
    Export alerts as downloadable file.
    """
    try:
        logger.info("Exporting alerts: target=%s priority=%s format=%s", target, priority, format)
        if format != "json":
            raise HTTPException(status_code=400, detail="Only 'json' format is supported currently")

        since = None
        if since_minutes:
            since = datetime.now() - timedelta(minutes=since_minutes)

        alerts = alert_service.get_alerts(target_name=target, priority=priority, since=since, limit=None) or []

        export_data = {
            "export_time": datetime.now().isoformat(),
            "filters": {"target": target, "priority": priority, "since_minutes": since_minutes},
            "count": len(alerts),
            "alerts": alerts
        }

        json_bytes = json.dumps(export_data, indent=2).encode("utf-8")
        file_stream = BytesIO(json_bytes)
        filename = f"alerts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        logger.info("Successfully exported %d alerts to %s", len(alerts), filename)
        return StreamingResponse(file_stream, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename}"})

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error exporting alerts: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export alerts: {str(e)}")

# -------------------------------
# GET /alerts/ping - Health check
# -------------------------------
@router.get("/ping")
async def ping_alerts():
    """
    Health check for alerts service.
    """
    try:
        stats = alert_service.get_statistics()
        return JSONResponse({"status": "ready", "service": "alerts", "timestamp": datetime.now().isoformat(), "stats_available": bool(stats)})
    except Exception as e:
        logger.error("Health check failed: %s", e, exc_info=True)
        return JSONResponse({"status": "degraded", "service": "alerts", "error": str(e)}, status_code=503)
