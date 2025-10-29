# --- NEW IMPORTS ---
import os
from dotenv import load_dotenv
# --- END NEW IMPORTS ---

import threading
from typing import Dict, List, Optional, Set, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- Optional Twilio import ---
# from twilio.rest import Client

from app.state import CAMERA_METADATA
from app.utils.db import (
    log_alert,
    load_watchlist_db,
    save_watchlist_db,
    load_geofences_db,
    save_geofence_db
)

# --- NEW: Load environment variables ---
load_dotenv()
print("[AlertService] Loaded environment variables from .env")

class AlertService:
    """Service for managing alerts and notifications"""

    def __init__(self):
        # --- IMPROVEMENT: Initialize configs from .env ---
        self._load_config_from_env()

        # Alert queue: list of pending alerts
        self.alert_queue = []

        # Alert history: target -> deque of last 200 alerts
        self.alert_history = defaultdict(lambda: deque(maxlen=200))

        # Watchlist: set of target names that trigger alerts
        self.watchlist: Set[str] = set()

        # Geo-fenced zones: {zone_name: {cameras: [...], enabled: bool}}
        self.geofence_zones = {}

        # Alert subscribers: {target -> list of callback functions}
        self.subscribers = defaultdict(list)

        # Thread safety
        self._lock = threading.Lock()

        # Statistics
        self.stats = {
            "total_alerts": 0,
            "email_sent": 0,
            "sms_sent": 0,
            "failed_notifications": 0
        }

        # Load persistent state from DB on startup
        self._init_from_db()

    # -------------------------------
    # NEW: Load Config from .env
    # -------------------------------
    def _load_config_from_env(self):
        """Load notification settings from environment variables."""
        print("[AlertService] Loading notification config from environment variables...")

        # Email Config
        self.email_config = {
            "enabled": os.getenv("EMAIL_ENABLED", 'False').lower() == 'true',
            "smtp_server": os.getenv("SMTP_SERVER", ""),
            "smtp_port": int(os.getenv("SMTP_PORT", 587)), # Convert port to int
            "sender_email": os.getenv("SENDER_EMAIL", ""),
            "sender_password": os.getenv("SENDER_PASSWORD", ""),
            # Split comma-separated string into list, remove whitespace
            "recipients": [email.strip() for email in os.getenv("EMAIL_RECIPIENTS", "").split(',') if email.strip()]
        }
        if self.email_config["enabled"]:
             print(f"[AlertService] Email notifications ENABLED for {len(self.email_config['recipients'])} recipients.")
        else:
             print("[AlertService] Email notifications DISABLED.")

        # SMS Config (Example for Twilio)
        self.sms_config = {
            "enabled": os.getenv("SMS_ENABLED", 'False').lower() == 'true',
            "api_key": os.getenv("TWILIO_ACCOUNT_SID", ""), # Using Twilio var names
            "api_secret": os.getenv("TWILIO_AUTH_TOKEN", ""),
            "sender_phone": os.getenv("TWILIO_SENDER_PHONE", ""),
            "recipients": [phone.strip() for phone in os.getenv("SMS_RECIPIENTS", "").split(',') if phone.strip()]
        }
        if self.sms_config["enabled"]:
             print(f"[AlertService] SMS notifications ENABLED for {len(self.sms_config['recipients'])} recipients.")
        else:
             print("[AlertService] SMS notifications DISABLED.")

    # -------------------------------
    # Load persistent state
    # -------------------------------
    def _init_from_db(self):
        """Load watchlist and geofences from database on startup."""
        print("[AlertService] Initializing watchlist/geofences from database...")
        try:
            with self._lock:
                self.watchlist = set(load_watchlist_db())
                self.geofence_zones = load_geofences_db()
            print(f"[AlertService] Loaded {len(self.watchlist)} watchlist targets.")
            print(f"[AlertService] Loaded {len(self.geofence_zones)} geo-fence zones.")
        except Exception as e:
            print(f"[AlertService] CRITICAL: Failed to load from DB: {e}")
            print("[AlertService] Running with empty in-memory state.")

    # --- Rest of the file remains the same ---
    # Watchlist Management (add, remove, get, is_watchlisted),
    # Geo-Fencing (create, delete, toggle, check, get),
    # Alert Generation (generate_alert), Alert Retrieval (get_alerts, get_latest_alert),
    # Notification Configuration (configure_email, configure_sms - these now OVERRIDE .env),
    # Send Notifications (_send_notifications_task, _send_email_alert, _send_sms_alert),
    # Subscriber Pattern, Statistics, Helpers (_get_confidence_level)
    # ... (all these methods stay exactly as they were in the previous version) ...

    # -------------------------------
    # Watchlist Management
    # -------------------------------
    def add_to_watchlist(self, target_name: str) -> Dict:
        """
        Add a person to watchlist (triggers alerts on detection). Persists to database.
        """
        with self._lock:
            if target_name in self.watchlist:
                return {"success": False, "message": f"'{target_name}' already on watchlist"}
            self.watchlist.add(target_name)
            try:
                save_watchlist_db(list(self.watchlist))
            except Exception as e:
                print(f"DB Error: Failed to save watchlist: {e}")
            return {"success": True, "message": f"'{target_name}' added to watchlist"}

    def remove_from_watchlist(self, target_name: str) -> Dict:
        """Remove person from watchlist. Persists to database."""
        with self._lock:
            if target_name not in self.watchlist:
                return {"success": False, "message": f"'{target_name}' not on watchlist"}
            self.watchlist.remove(target_name)
            try:
                save_watchlist_db(list(self.watchlist))
            except Exception as e:
                print(f"DB Error: Failed to save watchlist: {e}")
            return {"success": True, "message": f"'{target_name}' removed from watchlist"}

    def get_watchlist(self) -> List[str]:
        """Get all watchlisted persons"""
        with self._lock:
            return list(self.watchlist)

    def is_watchlisted(self, target_name: str) -> bool:
        """Check if person is on watchlist"""
        with self._lock:
            return target_name in self.watchlist

    # -------------------------------
    # Geo-Fencing
    # -------------------------------
    def create_geofence(self, zone_name: str, camera_ids: List[int], description: str = "", enabled: bool = True) -> Dict:
        """Create a geo-fenced zone. Persists to database."""
        with self._lock:
            if zone_name in self.geofence_zones:
                return {"success": False, "message": f"Zone '{zone_name}' already exists"}
            invalid_cameras = [cam for cam in camera_ids if cam not in CAMERA_METADATA]
            if invalid_cameras:
                return {"success": False, "message": f"Invalid camera IDs: {invalid_cameras}"}
            self.geofence_zones[zone_name] = {
                "camera_ids": camera_ids,
                "description": description,
                "enabled": enabled,
                "created_at": datetime.now().isoformat()
            }
            try:
                save_geofence_db(self.geofence_zones)
            except Exception as e:
                print(f"DB Error: Failed to save geofences: {e}")
            return {"success": True, "message": f"Geo-fence zone '{zone_name}' created with {len(camera_ids)} cameras"}

    def delete_geofence(self, zone_name: str) -> Dict:
        """Delete a geo-fenced zone."""
        with self._lock:
            if zone_name not in self.geofence_zones:
                return {"success": False, "message": f"Zone '{zone_name}' not found"}
            del self.geofence_zones[zone_name]
            try:
                save_geofence_db(self.geofence_zones)
            except Exception as e:
                print(f"DB Error: Failed to save geofences: {e}")
            return {"success": True, "message": f"Zone '{zone_name}' deleted"}

    def toggle_geofence_enabled(self, zone_name: str, enabled: bool) -> Dict:
        """Enable or disable a geo-fenced zone."""
        with self._lock:
            if zone_name not in self.geofence_zones:
                return {"success": False, "message": f"Zone '{zone_name}' not found"}
            self.geofence_zones[zone_name]["enabled"] = enabled
            try:
                save_geofence_db(self.geofence_zones)
            except Exception as e:
                print(f"DB Error: Failed to save geofences: {e}")
            status = "enabled" if enabled else "disabled"
            return {"success": True, "message": f"Zone '{zone_name}' is now {status}"}

    def check_geofence(self, camera_id: int) -> List[str]:
        """Check which geo-fenced zones contain this camera."""
        with self._lock:
            matching_zones = [
                zone_name for zone_name, zone_data in self.geofence_zones.items()
                if zone_data.get("enabled", False) and camera_id in zone_data.get("camera_ids", [])
            ]
            return matching_zones

    def get_geofences(self) -> Dict:
        """Get all geo-fence zones"""
        with self._lock:
            return dict(self.geofence_zones) # Return a copy

    # -------------------------------
    # Alert Generation
    # -------------------------------
    def generate_alert(self, target_name: str, camera_id: int, distance: float, timestamp: Optional[datetime] = None, metadata: Optional[Dict] = None) -> Dict:
        """Generate an alert for a detection event. Triggers notifications asynchronously."""
        if timestamp is None:
            timestamp = datetime.now()
        alert_result = {}
        with self._lock:
            camera_info = CAMERA_METADATA.get(camera_id, {})
            camera_name = camera_info.get("name", f"Camera {camera_id}")
            geo_tuple = camera_info.get("geo", (0.0, 0.0))
            # Ensure geo is stored as string for log_alert compatibility
            geo_str = str(geo_tuple)

            geofence_zones = self.check_geofence(camera_id)
            is_watchlisted = target_name in self.watchlist
            in_geofence = bool(geofence_zones)
            high_confidence = distance < 0.4

            if is_watchlisted and in_geofence: priority = "critical"
            elif is_watchlisted or in_geofence: priority = "high"
            elif high_confidence: priority = "medium"
            else: priority = "low"

            alert_id = f"{target_name}_{camera_id}_{timestamp.timestamp()}_{priority}" # Added priority to ID
            alert = {
                "alert_id": alert_id, "target": target_name, "camera_id": camera_id,
                "camera_name": camera_name, "geo": geo_tuple, # Store tuple in memory/JSON
                "distance": round(distance, 4), # Round distance
                "confidence": self._get_confidence_level(distance),
                "priority": priority, "geofence_zones": geofence_zones,
                "is_watchlisted": is_watchlisted, "timestamp": timestamp.isoformat(),
                "metadata": metadata or {}
            }

            self.alert_queue.append(alert)
            self.alert_history[target_name].append(alert)
            self.stats["total_alerts"] += 1

            log_alert( # This function uses geo_str
                camera_id=camera_id, camera_name=camera_name, geo=geo_str,
                target=target_name, distance=distance
            )

            if priority in ["high", "critical"]:
                threading.Thread(target=self._send_notifications_task, args=(alert,)).start()

            self._notify_subscribers(target_name, alert)

            alert_result = {
                "alert_id": alert_id, "triggered": True, "priority": priority,
                "geofence_zones": geofence_zones,
                "notification_sent": priority in ["high", "critical"]
            }
        return alert_result

    # -------------------------------
    # Alert Retrieval
    # -------------------------------
    def get_alerts(self, target_name: Optional[str] = None, priority: Optional[str] = None, since: Optional[datetime] = None, limit: Optional[int] = None) -> List[Dict]:
        """Get alerts with optional filtering."""
        with self._lock:
            all_alerts_flat = [a for history_deque in self.alert_history.values() for a in history_deque]
            if not all_alerts_flat: return []

            # Apply filters
            filtered = all_alerts_flat
            if target_name:
                filtered = [a for a in filtered if a["target"] == target_name]
            if priority:
                filtered = [a for a in filtered if a["priority"] == priority]
            if since:
                filtered = [a for a in filtered if datetime.fromisoformat(a["timestamp"]) > since]

            # Sort by timestamp (newest first) AFTER filtering
            filtered.sort(key=lambda x: x["timestamp"], reverse=True)

            # Apply limit
            if limit:
                filtered = filtered[:limit]

            return filtered

    def get_latest_alert(self, target_name: Optional[str] = None) -> Optional[Dict]:
        """Get most recent alert"""
        alerts = self.get_alerts(target_name=target_name, limit=1)
        return alerts[0] if alerts else None

    # -------------------------------
    # Notification Configuration
    # -------------------------------
    def configure_email(self, smtp_server: str, smtp_port: int, sender_email: str, sender_password: str, recipients: List[str], enabled: bool = True) -> Dict:
        """Configure email notifications (overrides .env settings for current session)."""
        with self._lock:
            self.email_config = {
                "enabled": enabled, "smtp_server": smtp_server, "smtp_port": smtp_port,
                "sender_email": sender_email, "sender_password": sender_password, "recipients": recipients
            }
            status = "enabled" if enabled else "disabled"
            print(f"[AlertService] Email config updated via API: Status={status}")
            # Note: This does NOT save back to the .env file
            return {"success": True, "message": f"Email configuration updated (runtime only). Status: {status}"}

    def configure_sms(self, api_key: str, api_secret: str, sender_phone: str, recipients: List[str], enabled: bool = True) -> Dict:
        """Configure SMS notifications (overrides .env settings for current session)."""
        with self._lock:
            self.sms_config = {
                "enabled": enabled, "api_key": api_key, "api_secret": api_secret,
                "sender_phone": sender_phone, "recipients": recipients
            }
            status = "enabled" if enabled else "disabled"
            print(f"[AlertService] SMS config updated via API: Status={status}")
             # Note: This does NOT save back to the .env file
            return {"success": True, "message": f"SMS configuration updated (runtime only). Status: {status}"}

    # -------------------------------
    # Send Notifications (Internal)
    # -------------------------------
    def _send_notifications_task(self, alert: Dict):
        """Send email/SMS notifications for an alert in a background thread."""
        email_sent_in_task = False
        sms_sent_in_task = False
        email_failed = False
        sms_failed = False

        if self.email_config.get("enabled"):
            try:
                self._send_email_alert(alert)
                email_sent_in_task = True
            except Exception as e:
                print(f"Failed to send email: {e}")
                email_failed = True

        if self.sms_config.get("enabled"):
            try:
                self._send_sms_alert(alert)
                sms_sent_in_task = True
            except Exception as e:
                print(f"Failed to send SMS: {e}")
                sms_failed = True

        # Update stats safely after attempts
        with self._lock:
            if email_sent_in_task: self.stats["email_sent"] += 1
            if sms_sent_in_task: self.stats["sms_sent"] += 1
            if email_failed or sms_failed: self.stats["failed_notifications"] += 1

    def _send_email_alert(self, alert: Dict):
        """Send email notification"""
        # --- Use the config loaded/updated in memory ---
        config = self.email_config
        if not config.get("sender_email") or not config.get("recipients"):
             print("[AlertService] Email send skipped: Sender or recipients not configured.")
             return

        subject = f"[{alert['priority'].upper()}] Detection Alert: {alert['target']}"
        body = f"""
        FACE RECOGNITION ALERT

        Priority: {alert['priority'].upper()}
        Target: {alert['target']}
        Location: {alert['camera_name']} (Camera {alert['camera_id']})
        Coordinates: {alert['geo']}
        Confidence: {alert['confidence'].upper()}
        Time: {alert['timestamp']}

        {'Watchlist Match' if alert['is_watchlisted'] else ''}
        {'Geo-Fence Breach: ' + ', '.join(alert['geofence_zones']) if alert['geofence_zones'] else ''}

        ---
        Automated alert from Face Recognition Platform.
        """

        msg = MIMEMultipart()
        msg['From'] = config['sender_email']
        msg['To'] = ", ".join(config['recipients'])
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        print(f"[AlertService] Attempting to send email alert to {len(config['recipients'])} recipients via {config['smtp_server']}...")
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()
            server.login(config['sender_email'], config['sender_password'])
            server.send_message(msg)
        print("[AlertService] Email alert sent successfully.")

    def _send_sms_alert(self, alert: Dict):
        """Send SMS notification (Example using Twilio)."""
        # --- Use the config loaded/updated in memory ---
        config = self.sms_config
        if not config.get("api_key") or not config.get("sender_phone") or not config.get("recipients"):
            print("[AlertService] SMS send skipped: API key, sender phone, or recipients not configured.")
            return

        message = f"[{alert['priority'].upper()}] {alert['target']} detected at {alert['camera_name']} @ {alert['timestamp']}"

        print(f"[AlertService] Attempting to send SMS alert to {len(config['recipients'])} recipients...")
        # --- TWILIO EXAMPLE (Uncomment and install twilio to use) ---
        # try:
        #     from twilio.rest import Client # Import here to avoid making it a hard dependency
        #     client = Client(config['api_key'], config['api_secret'])
        #     for recipient in config['recipients']:
        #         message_instance = client.messages.create(
        #             body=message,
        #             from_=config['sender_phone'],
        #             to=recipient
        #         )
        #         print(f"[AlertService] SMS sent to {recipient} (SID: {message_instance.sid})")
        #     print("[AlertService] SMS alerts sent successfully via Twilio.")
        # except ImportError:
        #     print("[AlertService] Twilio library not installed. Cannot send SMS. Run 'pip install twilio'.")
        # except Exception as e:
        #     print(f"[AlertService] Twilio SMS failed: {e}")
        #     raise e # Re-raise to be caught by _send_notifications_task
        # --- END TWILIO EXAMPLE ---

        # Fallback print statement (if Twilio is not used/installed)
        if 'Client' not in locals(): # Check if Twilio was imported
             print(f"SMS (TODO - Install Twilio): To {config['recipients']}: {message}")


    # -------------------------------
    # Subscriber Pattern
    # -------------------------------
    def subscribe(self, target_name: str, callback: Callable):
        """Subscribe to alerts for a specific target."""
        with self._lock:
            self.subscribers[target_name].append(callback)

    def _notify_subscribers(self, target_name: str, alert: Dict):
        """Notify all subscribers for a target"""
        callbacks_to_run = []
        with self._lock:
             # Get list of callbacks under lock
             callbacks_to_run = list(self.subscribers.get(target_name, []))

        # Run callbacks outside the lock to avoid deadlocks if a callback is slow
        for callback in callbacks_to_run:
            try:
                callback(alert)
            except Exception as e:
                print(f"Subscriber callback failed: {e}")

    # -------------------------------
    # Statistics
    # -------------------------------
    def get_statistics(self) -> Dict:
        """Get alert statistics"""
        with self._lock:
            # Return a copy to prevent modification outside the service
            return {
                **self.stats,
                "watchlist_size": len(self.watchlist),
                "geofence_zones": len(self.geofence_zones),
                "pending_alerts": len(self.alert_queue)
            }

    # -------------------------------
    # Helpers
    # -------------------------------
    def _get_confidence_level(self, distance: float) -> str:
        """Convert distance to confidence level"""
        if distance < 0.4: return "high"
        elif distance < 0.6: return "medium"
        else: return "low"


# -------------------------------
# Singleton instance
# -------------------------------
alert_service = AlertService()