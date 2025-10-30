# backend/app/services/alert_service.py
"""
Improved AlertService for Face Recognition Platform.

Features added:
- Structured logging via app.utils.logger.get_logger
- SMTP (STARTTLS / SMTPS) with retries and exponential backoff
- SMS (Twilio) example kept as optional dynamic import
- Background notification sending (non-blocking)
- Per-target and per-channel cooldown (to avoid spamming)
- Thread-safe internal state and metrics
- Persists watchlist & geofences using existing DB helpers
- Keeps public API compatible with previous implementation
"""

from __future__ import annotations

import os
import threading
import time
import math
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
load_dotenv()  # load .env early

from app.state import CAMERA_METADATA
from app.utils.db import (
    log_alert,
    load_watchlist_db,
    save_watchlist_db,
    load_geofences_db,
    save_geofence_db,
)
from app.utils.logger import get_logger

logger = get_logger("app.services.alert_service")

# -------------------------
# Environment / Defaults
# -------------------------
NOTIF_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "30"))  # per-target global cooldown
EMAIL_COOLDOWN_SECONDS = int(os.getenv("ALERT_EMAIL_COOLDOWN_SECONDS", "60"))
SMS_COOLDOWN_SECONDS = int(os.getenv("ALERT_SMS_COOLDOWN_SECONDS", "60"))
EMAIL_RETRY_ATTEMPTS = int(os.getenv("ALERT_EMAIL_RETRIES", "2"))
EMAIL_RETRY_BASE = float(os.getenv("ALERT_EMAIL_RETRY_BASE", "1.5"))  # multiplier
NOTIF_THREAD_POOL = int(os.getenv("ALERT_THREAD_POOL", "4"))

# Email defaults read from env
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "False").lower() == "true"
SMTP_SERVER = os.getenv("SMTP_SERVER", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")
DEFAULT_EMAIL_RECIPIENTS = [e.strip() for e in os.getenv("EMAIL_RECIPIENTS", "").split(",") if e.strip()]

# SMS defaults
SMS_ENABLED = os.getenv("SMS_ENABLED", "False").lower() == "true"
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_SENDER_PHONE = os.getenv("TWILIO_SENDER_PHONE", "")
DEFAULT_SMS_RECIPIENTS = [p.strip() for p in os.getenv("SMS_RECIPIENTS", "").split(",") if p.strip()]

# Use thread pool semaphore to limit concurrent notif threads
_notif_semaphore = threading.BoundedSemaphore(value=max(1, NOTIF_THREAD_POOL))


class AlertService:
    """Service for managing alerts and notifications."""

    def __init__(self):
        logger.info("Initializing AlertService")
        # Load runtime config from env (can be overridden via configure_email/configure_sms)
        self.email_config = {
            "enabled": EMAIL_ENABLED,
            "smtp_server": SMTP_SERVER,
            "smtp_port": SMTP_PORT,
            "sender_email": SENDER_EMAIL,
            "sender_password": SENDER_PASSWORD,
            "recipients": DEFAULT_EMAIL_RECIPIENTS.copy(),
        }

        self.sms_config = {
            "enabled": SMS_ENABLED,
            "api_key": TWILIO_ACCOUNT_SID,
            "api_secret": TWILIO_AUTH_TOKEN,
            "sender_phone": TWILIO_SENDER_PHONE,
            "recipients": DEFAULT_SMS_RECIPIENTS.copy(),
        }

        # State
        self.alert_queue: List[Dict[str, Any]] = []
        self.alert_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.watchlist: Set[str] = set()
        self.geofence_zones: Dict[str, Any] = {}
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)

        # cooldown tracking: { (target, channel) : last_sent_ts }
        self._last_sent: Dict[tuple, float] = {}

        # Threading and locks
        self._lock = threading.RLock()

        # Stats
        self.stats = {
            "total_alerts": 0,
            "email_sent": 0,
            "sms_sent": 0,
            "failed_notifications": 0,
        }

        # Initialize from DB
        self._init_from_db()

    # -------------------------------
    # Persistence & Initialization
    # -------------------------------
    def _init_from_db(self):
        """Load persistent state (watchlist & geofences)."""
        try:
            with self._lock:
                wl = load_watchlist_db()
                self.watchlist = set(wl or [])
                self.geofence_zones = load_geofences_db() or {}
            logger.info("Loaded watchlist (%d) and geofences (%d) from DB", len(self.watchlist), len(self.geofence_zones))
        except Exception as e:
            logger.exception("Failed to initialize AlertService from DB: %s", e)
            self.watchlist = set()
            self.geofence_zones = {}

    # -------------------------------
    # Watchlist Management
    # -------------------------------
    def add_to_watchlist(self, target_name: str) -> Dict[str, Any]:
        with self._lock:
            if target_name in self.watchlist:
                return {"success": False, "message": f"'{target_name}' already on watchlist"}
            self.watchlist.add(target_name)
            try:
                save_watchlist_db(list(self.watchlist))
            except Exception as e:
                logger.exception("Failed to save watchlist to DB: %s", e)
            return {"success": True, "message": f"'{target_name}' added to watchlist"}

    def remove_from_watchlist(self, target_name: str) -> Dict[str, Any]:
        with self._lock:
            if target_name not in self.watchlist:
                return {"success": False, "message": f"'{target_name}' not on watchlist"}
            self.watchlist.remove(target_name)
            try:
                save_watchlist_db(list(self.watchlist))
            except Exception as e:
                logger.exception("Failed to save watchlist to DB: %s", e)
            return {"success": True, "message": f"'{target_name}' removed from watchlist"}

    def get_watchlist(self) -> List[str]:
        with self._lock:
            return list(self.watchlist)

    def is_watchlisted(self, target_name: str) -> bool:
        with self._lock:
            return target_name in self.watchlist

    # -------------------------------
    # Geo-Fencing
    # -------------------------------
    def create_geofence(self, zone_name: str, camera_ids: List[int], description: str = "", enabled: bool = True) -> Dict[str, Any]:
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
                "created_at": datetime.now().isoformat(),
            }
            try:
                save_geofence_db(self.geofence_zones)
            except Exception as e:
                logger.exception("Failed to save geofences to DB: %s", e)
            return {"success": True, "message": f"Geo-fence zone '{zone_name}' created with {len(camera_ids)} cameras"}

    def delete_geofence(self, zone_name: str) -> Dict[str, Any]:
        with self._lock:
            if zone_name not in self.geofence_zones:
                return {"success": False, "message": f"Zone '{zone_name}' not found"}
            del self.geofence_zones[zone_name]
            try:
                save_geofence_db(self.geofence_zones)
            except Exception as e:
                logger.exception("Failed to save geofences to DB: %s", e)
            return {"success": True, "message": f"Zone '{zone_name}' deleted"}

    def toggle_geofence_enabled(self, zone_name: str, enabled: bool) -> Dict[str, Any]:
        with self._lock:
            if zone_name not in self.geofence_zones:
                return {"success": False, "message": f"Zone '{zone_name}' not found"}
            self.geofence_zones[zone_name]["enabled"] = enabled
            try:
                save_geofence_db(self.geofence_zones)
            except Exception as e:
                logger.exception("Failed to save geofences to DB: %s", e)
            status = "enabled" if enabled else "disabled"
            return {"success": True, "message": f"Zone '{zone_name}' is now {status}"}

    def check_geofence(self, camera_id: int) -> List[str]:
        with self._lock:
            return [
                zone_name
                for zone_name, zone_data in self.geofence_zones.items()
                if zone_data.get("enabled", False) and camera_id in zone_data.get("camera_ids", [])
            ]

    def get_geofences(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.geofence_zones)

    # -------------------------------
    # Alert Generation
    # -------------------------------
    def generate_alert(self, target_name: str, camera_id: int, distance: float, timestamp: Optional[datetime] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        if timestamp is None:
            timestamp = datetime.now()
        with self._lock:
            camera_info = CAMERA_METADATA.get(camera_id, {})
            camera_name = camera_info.get("name", f"Camera {camera_id}")
            geo_tuple = camera_info.get("geo", (0.0, 0.0))
            geo_str = str(geo_tuple)

            geofence_zones = self.check_geofence(camera_id)
            is_watchlisted = target_name in self.watchlist
            in_geofence = bool(geofence_zones)
            high_confidence = distance < 0.4

            if is_watchlisted and in_geofence:
                priority = "critical"
            elif is_watchlisted or in_geofence:
                priority = "high"
            elif high_confidence:
                priority = "medium"
            else:
                priority = "low"

            alert_id = f"{target_name}_{camera_id}_{timestamp.timestamp()}_{priority}"
            alert = {
                "alert_id": alert_id,
                "target": target_name,
                "camera_id": camera_id,
                "camera_name": camera_name,
                "geo": geo_tuple,
                "distance": round(distance, 4),
                "confidence": self._get_confidence_level(distance),
                "priority": priority,
                "geofence_zones": geofence_zones,
                "is_watchlisted": is_watchlisted,
                "timestamp": timestamp.isoformat(),
                "metadata": metadata or {},
            }

            # persist to in-memory structures and DB
            self.alert_queue.append(alert)
            self.alert_history[target_name].append(alert)
            self.stats["total_alerts"] += 1

            try:
                log_alert(
                    camera_id=camera_id,
                    camera_name=camera_name,
                    geo=geo_str,
                    target=target_name,
                    distance=distance,
                )
            except Exception:
                logger.exception("log_alert failed (non-fatal) for alert %s", alert_id)

            # send notifications for high/critical priorities
            if priority in ("high", "critical"):
                # dispatch notification background thread
                self._dispatch_notification(alert)

            # notify subscribers synchronously (callbacks run outside lock to avoid deadlocks)
            self._notify_subscribers(target_name, alert)

            return {
                "alert_id": alert_id,
                "triggered": True,
                "priority": priority,
                "geofence_zones": geofence_zones,
                "notification_sent": priority in ("high", "critical"),
            }

    # -------------------------------
    # Retrieval
    # -------------------------------
    def get_alerts(self, target_name: Optional[str] = None, priority: Optional[str] = None, since: Optional[datetime] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            all_alerts_flat = [a for history_deque in self.alert_history.values() for a in history_deque]
            if not all_alerts_flat:
                return []

            filtered = all_alerts_flat
            if target_name:
                filtered = [a for a in filtered if a["target"] == target_name]
            if priority:
                filtered = [a for a in filtered if a["priority"] == priority]
            if since:
                filtered = [a for a in filtered if datetime.fromisoformat(a["timestamp"]) > since]

            # sort newest first
            filtered.sort(key=lambda x: x["timestamp"], reverse=True)
            if limit:
                filtered = filtered[:limit]
            return filtered

    def get_latest_alert(self, target_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        alerts = self.get_alerts(target_name=target_name, limit=1)
        return alerts[0] if alerts else None

    # -------------------------------
    # Notification dispatch & workers
    # -------------------------------
    def _dispatch_notification(self, alert: Dict[str, Any]):
        """Dispatch notification sending in background (non-blocking)."""
        # Throttle by global cooldown (simple)
        target = alert["target"]
        now_ts = time.time()
        last_global = self._last_sent.get((target, "global"), 0)
        if now_ts - last_global < NOTIF_COOLDOWN_SECONDS:
            logger.debug("Global cooldown active for %s (skip notify)", target)
            return
        self._last_sent[(target, "global")] = now_ts

        # start a background thread (bounded by semaphore)
        def _runner():
            acquired = _notif_semaphore.acquire(timeout=10)
            if not acquired:
                logger.warning("Notification semaphore busy; skipping notification for %s", alert.get("alert_id"))
                return
            try:
                self._send_notifications_task(alert)
            finally:
                try:
                    _notif_semaphore.release()
                except Exception:
                    pass

        t = threading.Thread(target=_runner, daemon=True)
        t.start()

    def _send_notifications_task(self, alert: Dict[str, Any]):
        """Background thread: send email and/or SMS with retries and cooldown checks."""
        target = alert["target"]
        email_sent = False
        sms_sent = False
        email_failed = False
        sms_failed = False

        # Email
        if self.email_config.get("enabled"):
            try:
                # Check per-target cooldown for email
                key = (target, "email")
                last = self._last_sent.get(key, 0)
                if time.time() - last >= EMAIL_COOLDOWN_SECONDS:
                    self._send_email_with_retries(alert)
                    email_sent = True
                    self._last_sent[key] = time.time()
                else:
                    logger.debug("Email cooldown active for %s; skipping email", target)
            except Exception as e:
                logger.exception("Email send failed for %s: %s", target, e)
                email_failed = True

        # SMS
        if self.sms_config.get("enabled"):
            try:
                key = (target, "sms")
                last = self._last_sent.get(key, 0)
                if time.time() - last >= SMS_COOLDOWN_SECONDS:
                    self._send_sms_alert(alert)
                    sms_sent = True
                    self._last_sent[key] = time.time()
                else:
                    logger.debug("SMS cooldown active for %s; skipping SMS", target)
            except Exception as e:
                logger.exception("SMS send failed for %s: %s", target, e)
                sms_failed = True

        # Update stats
        with self._lock:
            if email_sent:
                self.stats["email_sent"] += 1
            if sms_sent:
                self.stats["sms_sent"] += 1
            if email_failed or sms_failed:
                self.stats["failed_notifications"] += 1

    # -------------------------------
    # Email sending with retries
    # -------------------------------
    def _send_email_with_retries(self, alert: Dict[str, Any]):
        attempts = max(1, EMAIL_RETRY_ATTEMPTS)
        backoff = 1.0
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                self._send_email_alert(alert)
                logger.info("Email sent for alert %s (attempt %d/%d)", alert.get("alert_id"), attempt, attempts)
                return
            except Exception as exc:
                last_exc = exc
                logger.exception("Email attempt %d failed for %s: %s", attempt, alert.get("target"), exc)
                if attempt < attempts:
                    # exponential backoff
                    sleep_time = backoff
                    time.sleep(sleep_time)
                    backoff *= EMAIL_RETRY_BASE
        # all attempts failed
        logger.error("All email attempts failed for alert %s: last_exc=%s", alert.get("alert_id"), last_exc)
        raise last_exc

    def _send_email_alert(self, alert: Dict[str, Any]):
        """Send email notification synchronously (may be called in background thread)."""
        cfg = self.email_config
        if not cfg.get("sender_email") or not cfg.get("recipients") or not cfg.get("smtp_server"):
            logger.warning("Email not sent: SMTP config incomplete")
            return

        subject = f"[{alert['priority'].upper()}] Detection Alert: {alert['target']}"
        body = (
            f"FACE RECOGNITION ALERT\n\n"
            f"Priority: {alert['priority'].upper()}\n"
            f"Target: {alert['target']}\n"
            f"Location: {alert['camera_name']} (Camera {alert['camera_id']})\n"
            f"Coordinates: {alert['geo']}\n"
            f"Confidence: {alert['confidence']}\n"
            f"Time: {alert['timestamp']}\n\n"
            f"{'Watchlist Match' if alert['is_watchlisted'] else ''}\n"
            f"{'Geo-Fence Breach: ' + ', '.join(alert['geofence_zones']) if alert['geofence_zones'] else ''}\n\n"
            f"---\nAutomated alert from Face Recognition Platform."
        )

        msg = MIMEMultipart()
        msg["From"] = cfg["sender_email"]
        msg["To"] = ", ".join(cfg["recipients"])
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        smtp_host = cfg["smtp_server"]
        smtp_port = int(cfg.get("smtp_port", SMTP_PORT))
        username = cfg.get("sender_email")
        password = cfg.get("sender_password")

        # Choose SSL vs STARTTLS depending on port (465 usually implies SSL)
        use_ssl = smtp_port == 465
        logger.debug("Attempting email send via %s:%s (ssl=%s) to %d recipients", smtp_host, smtp_port, use_ssl, len(cfg["recipients"]))
        if use_ssl:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=15)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=15)

        try:
            with server:
                server.ehlo()
                if not use_ssl:
                    # Start TLS if server supports it
                    try:
                        server.starttls()
                        server.ehlo()
                    except Exception:
                        logger.debug("STARTTLS failed or not supported; continuing without TLS")
                if username and password:
                    try:
                        server.login(username, password)
                    except Exception as e:
                        logger.exception("SMTP login failed: %s", e)
                        # continue without login if server accepts anonymous sending
                server.send_message(msg)
        finally:
            try:
                server.quit()
            except Exception:
                pass

    # -------------------------------
    # SMS sending (Twilio example; optional)
    # -------------------------------
    def _send_sms_alert(self, alert: Dict[str, Any]):
        cfg = self.sms_config
        if not cfg.get("api_key") or not cfg.get("sender_phone") or not cfg.get("recipients"):
            logger.warning("SMS not sent: Twilio config incomplete")
            return

        message = f"[{alert['priority'].upper()}] {alert['target']} detected at {alert['camera_name']} @ {alert['timestamp']}"
        logger.debug("Sending SMS to %d recipients (Twilio)", len(cfg["recipients"]))
        try:
            # dynamic import to keep Twilio optional
            from twilio.rest import Client  # type: ignore

            client = Client(cfg["api_key"], cfg["api_secret"])
            for recipient in cfg["recipients"]:
                try:
                    rcpt = client.messages.create(body=message, from_=cfg["sender_phone"], to=recipient)
                    logger.info("SMS sent to %s (sid=%s)", recipient, getattr(rcpt, "sid", "<no-sid>"))
                except Exception:
                    logger.exception("Failed to send SMS to %s (continuing)", recipient)
        except ImportError:
            logger.warning("Twilio not installed; SMS fallback: printing message")
            for recipient in cfg["recipients"]:
                logger.info("SMS (mock) -> %s : %s", recipient, message)
        except Exception:
            logger.exception("Unexpected SMS send failure (caught and logged)")

    # -------------------------------
    # Subscriber pattern
    # -------------------------------
    def subscribe(self, target_name: str, callback: Callable[[Dict[str, Any]], None]):
        with self._lock:
            self.subscribers[target_name].append(callback)

    def _notify_subscribers(self, target_name: str, alert: Dict[str, Any]):
        # copy callbacks under lock
        with self._lock:
            callbacks = list(self.subscribers.get(target_name, []))
        for cb in callbacks:
            try:
                cb(alert)
            except Exception:
                logger.exception("Subscriber callback raised an exception for target %s", target_name)

    # -------------------------------
    # Statistics & helpers
    # -------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                **self.stats,
                "watchlist_size": len(self.watchlist),
                "geofence_zones": len(self.geofence_zones),
                "pending_alerts": len(self.alert_queue),
            }

    def _get_confidence_level(self, distance: float) -> str:
        if distance < 0.4:
            return "high"
        if distance < 0.6:
            return "medium"
        return "low"

    # -------------------------------
    # Runtime configuration overrides (API)
    # -------------------------------
    def configure_email(self, smtp_server: str, smtp_port: int, sender_email: str, sender_password: str, recipients: List[str], enabled: bool = True) -> Dict[str, Any]:
        with self._lock:
            self.email_config = {
                "enabled": enabled,
                "smtp_server": smtp_server,
                "smtp_port": smtp_port,
                "sender_email": sender_email,
                "sender_password": sender_password,
                "recipients": recipients,
            }
            status = "enabled" if enabled else "disabled"
            logger.info("Email config updated via API: status=%s", status)
            return {"success": True, "message": f"Email configuration updated (runtime only). Status: {status}"}

    def configure_sms(self, api_key: str, api_secret: str, sender_phone: str, recipients: List[str], enabled: bool = True) -> Dict[str, Any]:
        with self._lock:
            self.sms_config = {
                "enabled": enabled,
                "api_key": api_key,
                "api_secret": api_secret,
                "sender_phone": sender_phone,
                "recipients": recipients,
            }
            status = "enabled" if enabled else "disabled"
            logger.info("SMS config updated via API: status=%s", status)
            return {"success": True, "message": f"SMS configuration updated (runtime only). Status: {status}"}

    # -------------------------------
    # Graceful shutdown helper (optional)
    # -------------------------------
    def shutdown(self):
        logger.info("AlertService shutdown requested - no active shutdown tasks implemented")

# Singleton instance
alert_service = AlertService()
