import sqlite3
import datetime
import uuid
import logging
from typing import Dict, Any, Optional
from services.notification_service import NotificationService
from database.database import get_connection

logger = logging.getLogger(__name__)

class AlertEngine:
    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.notifier = NotificationService(db_path)
        self.HIGH_RISK_THRESHOLD = 75.0
        self.MIN_USABLE_AGENTS = 4

    def _get_connection(self):
        return get_connection(self.db_path)

    def process_event(self, event: Dict[str, Any]) -> Optional[str]:
        event_type = event.get("event_type")
        source = event.get("source", "system")
        dedup_key = event.get("dedup_key")
        if not dedup_key: dedup_key = f"{event_type}:{source}:{event.get('reference_id', 'none')}"
        now = datetime.datetime.utcnow().isoformat()
        
        created_alert_id = None
        user_role_to_notify = None
        
        conn = self._get_connection()
        try:
            with conn:
                row = conn.execute("SELECT alert_id, occurrence_count FROM alerts WHERE dedup_key = ? AND status = 'active'", (dedup_key,)).fetchone()
                if row:
                    alert_id = row["alert_id"]
                    new_count = row["occurrence_count"] + 1
                    conn.execute("UPDATE alerts SET occurrence_count = ?, last_seen_at = ? WHERE alert_id = ?", (new_count, now, alert_id))
                    return alert_id
                else:
                    alert_id = f"ALR-{uuid.uuid4().hex[:8].upper()}"
                    conn.execute(
                        "INSERT INTO alerts (alert_id, type, severity, title, message, source, reference_id, dedup_key, status, occurrence_count, first_seen_at, last_seen_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (alert_id, event_type, event.get("severity", "info"), event.get("title", "Unknown Alert"), event.get("message", ""), source, event.get("reference_id"), dedup_key, "active", 1, now, now)
                    )
                    created_alert_id = alert_id
                    user_role_to_notify = "admin" if event_type in ["agent_failure", "database_failure", "api_failure"] else "all"
        except Exception as e:
            logger.error(f"AlertEngine Error processing event: {e}")
            return None
        finally:
            conn.close()

        # Dispatch notification OUTSIDE the transaction to avoid SQLite deadlock
        if created_alert_id:
            try:
                self.notifier.create_notification(created_alert_id, user_role_to_notify)
                logger.info(f"AlertEngine: Created new alert {created_alert_id} for {event_type}")
            except Exception as e:
                logger.error(f"AlertEngine Error dispatching notification: {e}")
                
        return created_alert_id

    def resolve_alert(self, dedup_key: str) -> None:
        now = datetime.datetime.utcnow().isoformat()
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("UPDATE alerts SET status = 'resolved', resolved_at = ? WHERE dedup_key = ? AND status = 'active'", (now, dedup_key))
        except Exception as e:
            logger.error(f"AlertEngine Error resolving alert: {e}")
        finally:
            conn.close()

    def evaluate_investigation(self, analysis: Dict[str, Any]) -> None:
        analysis_id = analysis.get("analysis_id")
        risk_score = analysis.get("final_assessment", {}).get("risk_score", 0)
        verdict = analysis.get("final_assessment", {}).get("verdict", "unknown")
        usable = analysis.get("consensus", {}).get("available_agents", 0)
        conflict = analysis.get("consensus", {}).get("conflict_detected", False)
        
        if risk_score >= self.HIGH_RISK_THRESHOLD and verdict in ["phishing", "suspicious"]:
            self.process_event({"event_type": "high_risk_investigation", "severity": "critical" if risk_score >= 90 else "high", "title": f"High-Risk Investigation ({risk_score:.1f})", "message": f"Target classified as {verdict.title()}.", "source": "fusion_engine", "reference_id": analysis_id, "dedup_key": f"high_risk:{analysis_id}"})
        elif verdict == "unknown":
            self.process_event({"event_type": "unknown_verdict", "severity": "medium", "title": "Insufficient Evidence", "message": f"Investigation completed with 'unknown' verdict. Coverage: {usable}/6", "source": "fusion_engine", "reference_id": analysis_id, "dedup_key": f"unknown:{analysis_id}"})
        elif usable < self.MIN_USABLE_AGENTS:
            self.process_event({"event_type": "partial_analysis", "severity": "medium", "title": "Partial Analysis", "message": f"Investigation completed with limited coverage ({usable}/6 agents).", "source": "fusion_engine", "reference_id": analysis_id, "dedup_key": f"partial:{analysis_id}"})
        elif conflict:
            self.process_event({"event_type": "fusion_conflict", "severity": "medium", "title": "Fusion Conflict", "message": "Agents strongly disagreed on the threat assessment.", "source": "fusion_engine", "reference_id": analysis_id, "dedup_key": f"conflict:{analysis_id}"})

    def evaluate_system_health(self, health_data: Dict[str, Any]) -> None:
        db_status = health_data.get("database", {}).get("status")
        db_key = "sys_health:database"
        if db_status != "connected":
            self.process_event({"event_type": "database_failure", "severity": "critical", "title": "Database Unavailable", "message": "Database connection failed.", "source": "system", "dedup_key": db_key})
        else:
            self.resolve_alert(db_key)
            
        agents = health_data.get("agents", {})
        for agent_name, agent_data in agents.items():
            agent_key = f"sys_health:agent:{agent_name}"
            if agent_data.get("status", "unknown") != "healthy":
                self.process_event({"event_type": "agent_failure", "severity": "high", "title": f"Agent Failure: {agent_data.get('name', agent_name)}", "message": f"{agent_data.get('name', agent_name)} is currently unavailable.", "source": agent_name, "dedup_key": agent_key})
            else:
                self.resolve_alert(agent_key)
