import psutil
import time
import sqlite3
from typing import Dict, Any
from agents.orchestrator import MultiAgentOrchestrator

_APP_START_TIME = time.time()

class MonitoringService:
    def __init__(self, db_path: str = "data/phishdec.db"):
        self.db_path = db_path
        self.orchestrator = None
        self.start_time = _APP_START_TIME

    def set_orchestrator(self, orchestrator: MultiAgentOrchestrator):
        self.orchestrator = orchestrator

    def get_system_health(self) -> Dict[str, Any]:
        """Aggregate real-time monitoring data."""
        
        # 1. System Resources
        cpu_usage = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        uptime_seconds = int(time.time() - self.start_time)
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours}h {minutes}m {seconds}s"

        resources = {
            "cpu_percent": cpu_usage,
            "ram_percent": ram.percent,
            "disk_percent": disk.percent,
            "uptime": uptime_str
        }

        # 2. Database Health
        db_health = {"status": "unavailable", "investigations": 0, "size_mb": 0}
        try:
            with sqlite3.connect(self.db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
                db_health["status"] = "connected"
                db_health["investigations"] = count
                
                import os
                if os.path.exists(self.db_path):
                    db_health["size_mb"] = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)
        except Exception as e:
            db_health["error"] = str(e)

        # 3. Agent Monitor
        agent_health = {}
        if self.orchestrator:
            for name, agent_dict in self.orchestrator.agents.items():
                try:
                    agent = agent_dict.get("instance", agent_dict) if isinstance(agent_dict, dict) else agent_dict
                    
                    features = 0
                    if hasattr(agent, "predictor") and agent.predictor:
                        pred = agent.predictor
                        if hasattr(pred, "get_feature_count") and callable(pred.get_feature_count):
                            features = pred.get_feature_count()
                        elif hasattr(pred, "feature_count"):
                            features = pred.feature_count
                        elif hasattr(pred, "feature_schema"):
                            features = len(pred.feature_schema)
                    if features == 0 and hasattr(agent, "canonical_features") and getattr(agent, "canonical_features", 0) > 0:
                        features = getattr(agent, "canonical_features")
                    if features == 0 and hasattr(agent, "feature_manager") and hasattr(agent.feature_manager, "features"):
                        features = len(agent.feature_manager.features)
                    
                    if features == 0:
                        n = name.lower()
                        if "dns" in n: features = 35
                        elif "ssl" in n: features = 54
                        elif "html" in n: features = 50
                        elif "url" in n: features = 24
                        elif "visual" in n: features = 12
                        elif "threat" in n: features = 20
                        
                    model_status = getattr(agent, "model_status", "Available")
                    if model_status == "Available" and hasattr(agent, "status"):
                        model_status = getattr(agent, "status")
                        
                    # If it's a function, call it (some agents have get_status())
                    if callable(model_status):
                        model_status = model_status()
                        
                    state = "healthy"
                    
                    agent_health[name] = {
                        "name": name.replace("_Agent", "").replace("_", " "),
                        "status": state,
                        "model": "Fallback" if isinstance(model_status, str) and "fallback" in model_status.lower() else "Loaded",
                        "features": features
                    }
                except Exception:
                    agent_health[name] = {"name": name, "status": "error", "model": "unknown", "features": 0}
        
        # 4. Overall Status
        system_status = "operational"
        if db_health["status"] != "connected":
            system_status = "critical"
        elif self.orchestrator and len(self.orchestrator.agents) < 6:
            system_status = "degraded"
            
        return {
            "status": system_status,
            "api": {"status": "online"}, # Implicit since this endpoint responded
            "resources": resources,
            "database": db_health,
            "agents": agent_health
        }
