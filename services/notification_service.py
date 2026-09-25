import sqlite3
import datetime
import uuid
import logging
from typing import Dict, Any, List
from database.database import get_connection

logger = logging.getLogger(__name__)

class NotificationService:
    def __init__(self, db_path: str = None):
        self.db_path = db_path

    def _get_connection(self):
        return get_connection(self.db_path)

    def create_notification(self, alert_id: str, user_role: str = "admin") -> str:
        notif_id = f"NOTIF-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.datetime.utcnow().isoformat()
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("INSERT INTO notifications (notification_id, alert_id, user_role, status, created_at) VALUES (?, ?, ?, ?, ?)", (notif_id, alert_id, user_role, "unread", now))
            return notif_id
        finally:
            conn.close()

    def get_notifications(self, user_role: str = "admin", status: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        query = "SELECT n.notification_id, n.status as notif_status, n.created_at, n.read_at, a.alert_id, a.type, a.severity, a.title, a.message, a.source, a.reference_id, a.occurrence_count FROM notifications n JOIN alerts a ON n.alert_id = a.alert_id WHERE n.user_role IN (?, 'all')"
        params = [user_role]
        if status: query += " AND n.status = ?"; params.append(status)
        query += " ORDER BY n.created_at DESC LIMIT ?"; params.append(limit)
        
        results = []
        conn = self._get_connection()
        try:
            with conn:
                for row in conn.execute(query, params): results.append(dict(row))
        except Exception as e:
            logger.error(f"Error fetching notifications: {e}")
        finally:
            conn.close()
        return results

    def get_unread_count(self, user_role: str = "admin") -> int:
        conn = self._get_connection()
        try:
            with conn:
                row = conn.execute("SELECT COUNT(*) as c FROM notifications WHERE user_role IN (?, 'all') AND status = 'unread'", (user_role,)).fetchone()
                return row["c"] if row else 0
        finally:
            conn.close()

    def mark_as_read(self, notification_id: str, user_role: str = "admin") -> bool:
        now = datetime.datetime.utcnow().isoformat()
        conn = self._get_connection()
        try:
            with conn:
                res = conn.execute("UPDATE notifications SET status = 'read', read_at = ? WHERE notification_id = ? AND user_role IN (?, 'all')", (now, notification_id, user_role))
                return res.rowcount > 0
        finally:
            conn.close()

    def mark_all_as_read(self, user_role: str = "admin") -> int:
        now = datetime.datetime.utcnow().isoformat()
        conn = self._get_connection()
        try:
            with conn:
                res = conn.execute("UPDATE notifications SET status = 'read', read_at = ? WHERE user_role IN (?, 'all') AND status = 'unread'", (now, user_role))
                return res.rowcount
        finally:
            conn.close()
