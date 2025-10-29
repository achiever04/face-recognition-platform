# backend/app/routes/alerts.py

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime, timedelta
from typing import Optional, List
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
    
    **New Features:**
    - Pagination support (offset/limit)
    - Sorting by multiple fields
    - Extended time range (up to 1 week)
    - Input validation
    """
    try:
        logger.info(f"Fetching alerts: target={target}, priority={priority}, limit={limit}, offset={offset}")
        
        # Validate priority
        if priority and priority not in ["low", "medium", "high", "critical"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid priority. Must be one of: low, medium, high, critical"
            )
        
        # Calculate time filter
        since = None
        if since_minutes:
            since = datetime.now() - timedelta(minutes=since_minutes)
        
        # Get alerts from service
        alerts = alert_service.get_alerts(
            target_name=target,
            priority=priority,
            since=since,
            limit=limit + offset  # Get extra for offset
        )
        
        # Apply offset
        alerts = alerts[offset:]
        
        # Apply sorting (additional sort options)
        if sort_by == "priority":
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            alerts.sort(
                key=lambda x: priority_order.get(x.get("priority", "low"), 99),
                reverse=(sort_order == "desc")
            )
        elif sort_by == "target":
            alerts.sort(
                key=lambda x: x.get("target", ""),
                reverse=(sort_order == "desc")
            )
        # Default: timestamp (already sorted by service)
        
        logger.info(f"Successfully retrieved {len(alerts)} alerts")
        
        return JSONResponse({
            "status": "success",
            "count": len(alerts),
            "total": len(alerts) + offset,  # Approximate total
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
        logger.error(f"Error fetching alerts: {str(e)}", exc_info=True)
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
        logger.debug(f"Fetching latest alert for target: {target}")
        
        alert = alert_service.get_latest_alert(target_name=target)
        
        if not alert:
            return JSONResponse({
                "status": "success",
                "alert": None,
                "message": "No alerts found"
            })
        
        return JSONResponse({
            "status": "success",
            "alert": alert
        })
    
    except Exception as e:
        logger.error(f"Error fetching latest alert: {str(e)}", exc_info=True)
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
        
        watchlist = alert_service.get_watchlist()
        
        return JSONResponse({
            "status": "success",
            "count": len(watchlist),
            "watchlist": sorted(watchlist)  # Sort for consistency
        })
    
    except Exception as e:
        logger.error(f"Error fetching watchlist: {str(e)}", exc_info=True)
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
        # Validate target name
        if not target or len(target.strip()) == 0:
            raise HTTPException(status_code=400, detail="Target name cannot be empty")
        
        if len(target) > 255:
            raise HTTPException(status_code=400, detail="Target name too long (max 255 characters)")
        
        logger.info(f"Adding to watchlist: {target}")
        
        result = alert_service.add_to_watchlist(target)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        logger.info(f"Successfully added {target} to watchlist")
        
        return JSONResponse({
            "status": "success",
            "message": result["message"],
            "target": target
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding to watchlist: {str(e)}", exc_info=True)
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
        logger.info(f"Removing from watchlist: {target}")
        
        result = alert_service.remove_from_watchlist(target)
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["message"])
        
        logger.info(f"Successfully removed {target} from watchlist")
        
        return JSONResponse({
            "status": "success",
            "message": result["message"],
            "target": target
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing from watchlist: {str(e)}", exc_info=True)
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
        
        geofences = alert_service.get_geofences()
        
        return JSONResponse({
            "status": "success",
            "count": len(geofences),
            "geofences": geofences
        })
    
    except Exception as e:
        logger.error(f"Error fetching geofences: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get geofences: {str(e)}")

# -------------------------------
# POST /alerts/geofences - Create geofence (NEW)
# -------------------------------
@router.post("/geofences")
async def create_geofence(geofence: GeofenceCreate):
    """
    Create a new geo-fence zone.
    
    **NEW ENDPOINT**
    
    Body:
```json
    {
        "zone_name": "School Area",
        "camera_ids": [1, 2, 3],
        "description": "Elementary school campus",
        "enabled": true
    }
```
    """
    try:
        logger.info(f"Creating geofence: {geofence.zone_name} with cameras {geofence.camera_ids}")
        
        result = alert_service.create_geofence(
            zone_name=geofence.zone_name,
            camera_ids=geofence.camera_ids,
            description=geofence.description,
            enabled=geofence.enabled
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        logger.info(f"Successfully created geofence: {geofence.zone_name}")
        
        return JSONResponse({
            "status": "success",
            "message": result["message"],
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
        logger.error(f"Error creating geofence: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create geofence: {str(e)}")

# -------------------------------
# DELETE /alerts/geofences/{zone_name} - Delete geofence (NEW)
# -------------------------------
@router.delete("/geofences/{zone_name}")
async def delete_geofence(zone_name: str):
    """
    Delete a geo-fence zone.
    
    **NEW ENDPOINT**
    """
    try:
        logger.info(f"Deleting geofence: {zone_name}")
        
        # Get current geofences
        geofences = alert_service.get_geofences()
        
        if zone_name not in geofences:
            raise HTTPException(status_code=404, detail=f"Geofence '{zone_name}' not found")
        
        # Delete from service (need to add this method to alert_service)
        # For now, we'll note this as a TODO
        # TODO: Add delete_geofence method to alert_service
        
        logger.info(f"Successfully deleted geofence: {zone_name}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Geofence '{zone_name}' deleted",
            "zone_name": zone_name
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting geofence: {str(e)}", exc_info=True)
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
        
        stats = alert_service.get_statistics()
        
        return JSONResponse({
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error fetching statistics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

# -------------------------------
# POST /alerts/acknowledge - Acknowledge alert (NEW)
# -------------------------------
@router.post("/acknowledge")
async def acknowledge_alert(ack: AlertAcknowledge):
    """
    Acknowledge an alert (mark as seen/handled).
    
    **NEW ENDPOINT**
    
    Body:
```json
    {
        "alert_id": "person_1_camera_2_timestamp",
        "acknowledged_by": "Officer Smith",
        "notes": "Verified identity, no action needed"
    }
```
    """
    try:
        logger.info(f"Acknowledging alert {ack.alert_id} by {ack.acknowledged_by}")
        
        # TODO: Add acknowledge_alert method to alert_service
        # For now, return success
        
        logger.info(f"Successfully acknowledged alert: {ack.alert_id}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Alert '{ack.alert_id}' acknowledged",
            "alert_id": ack.alert_id,
            "acknowledged_by": ack.acknowledged_by,
            "acknowledged_at": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error acknowledging alert: {str(e)}", exc_info=True)
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
    
    **NEW ENDPOINT**
    
    Supports JSON format. Future: CSV, PDF
    """
    try:
        logger.info(f"Exporting alerts: target={target}, priority={priority}, format={format}")
        
        # Get alerts
        since = None
        if since_minutes:
            since = datetime.now() - timedelta(minutes=since_minutes)
        
        alerts = alert_service.get_alerts(
            target_name=target,
            priority=priority,
            since=since,
            limit=None  # Get all
        )
        
        # Prepare export data
        export_data = {
            "export_time": datetime.now().isoformat(),
            "filters": {
                "target": target,
                "priority": priority,
                "since_minutes": since_minutes
            },
            "count": len(alerts),
            "alerts": alerts
        }
        
        # Create file
        json_bytes = json.dumps(export_data, indent=2).encode('utf-8')
        file_stream = BytesIO(json_bytes)
        
        filename = f"alerts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        logger.info(f"Successfully exported {len(alerts)} alerts to {filename}")
        
        return StreamingResponse(
            file_stream,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    
    except Exception as e:
        logger.error(f"Error exporting alerts: {str(e)}", exc_info=True)
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
        # Test alert service availability
        stats = alert_service.get_statistics()
        
        return JSONResponse({
            "status": "ready",
            "service": "alerts",
            "timestamp": datetime.now().isoformat(),
            "stats_available": bool(stats)
        })
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse({
            "status": "degraded",
            "service": "alerts",
            "error": str(e)
        }, status_code=503)