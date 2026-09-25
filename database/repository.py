"""
database/repository.py — Persistence & Query Repository
========================================================

Day 16 — Database & Investigation History

Implements the repository pattern to isolate SQL queries from the API
and application layer. Guarantees:
1. Atomic transactions across the 4 relational tables.
2. Zero risk recalculation — preserves Fusion as the single source of truth.
3. Strict NULL preservation for unavailable agent risk scores.
4. Full parameterized SQL to prevent SQL injection.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from .database import get_connection

logger = logging.getLogger(__name__)


class AnalysisRepository:
    """
    Repository handling persistence and retrieval for PhishDec investigations.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def save_analysis(self, analysis_input: Any) -> bool:
        """
        Persist a complete investigation atomically across all 4 tables.

        Accepts either an AnalysisResult instance or a dictionary from to_dict().
        Preserves exact Fusion risk scores and NULL states.
        """
        # Normalize input to dictionary
        if hasattr(analysis_input, "to_dict"):
            data = analysis_input.to_dict()
        elif isinstance(analysis_input, dict):
            data = analysis_input
        else:
            raise TypeError(f"Unsupported analysis input type: {type(analysis_input)}")

        analysis_id = data.get("analysis_id")
        if not analysis_id:
            raise ValueError("Analysis input is missing required 'analysis_id'")

        target = data.get("target") or {}
        final_assessment = data.get("final_assessment") or {}
        consensus = data.get("consensus") or {}
        agents = data.get("agents") or {}
        explanation = data.get("explanation") or {}
        meta = data.get("meta") or {}

        url = target.get("url") or data.get("target_url") or ""
        domain = target.get("domain") or data.get("target_domain")
        created_at = data.get("timestamp") or ""
        status = data.get("status") or "success"

        final_verdict = final_assessment.get("verdict") or "unknown"
        risk_score = final_assessment.get("risk_score")  # Can be None
        risk_level = final_assessment.get("risk_level") or "unknown"
        confidence = final_assessment.get("confidence")

        usable_count = consensus.get("available_agents", 0)
        total_count = consensus.get("total_agents", 6)
        coverage = consensus.get("coverage", 0.0)
        consensus_available = 1 if consensus.get("consensus_available", False) else 0
        consensus_satisfied = 1 if consensus.get("consensus_satisfied", False) else 0
        conflict_detected = 1 if consensus.get("conflict_detected", False) else 0
        limited_evidence = 1 if consensus.get("limited_evidence", False) else 0
        evidence_state = consensus.get("evidence_state") or "unknown"

        decision_basis = meta.get("decision_basis") or data.get("decision_basis") or {}
        decision_basis_json = json.dumps(decision_basis) if decision_basis else None

        risk_calibration = data.get("risk_calibration") or decision_basis.get("risk_calibration") or {}
        risk_calibration_json = json.dumps(risk_calibration) if risk_calibration else None

        execution_time_ms = meta.get("execution_time_ms")

        conn = get_connection(self.db_path)
        try:
            with conn:
                # 1. Insert into analyses
                conn.execute(
                    """
                    INSERT INTO analyses (
                        analysis_id, url, domain, created_at, status,
                        final_verdict, risk_score, risk_level, confidence,
                        usable_agent_count, total_agent_count, coverage,
                        consensus_available, consensus_satisfied,
                        conflict_detected, limited_evidence, evidence_state,
                        decision_basis_json, risk_calibration_json, execution_time_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        analysis_id, url, domain, created_at, status,
                        final_verdict, risk_score, risk_level, confidence,
                        usable_count, total_count, coverage,
                        consensus_available, consensus_satisfied,
                        conflict_detected, limited_evidence, evidence_state,
                        decision_basis_json, risk_calibration_json, execution_time_ms,
                    ),
                )

                # 2. Insert into agent_results (up to 6 agents)
                for display_key, agent_data in agents.items():
                    if not isinstance(agent_data, dict):
                        continue

                    agent_name = agent_data.get("agent") or display_key
                    agent_status = agent_data.get("status") or "unavailable"
                    signal_avail = 1 if agent_data.get("signal", False) else 0
                    prediction = agent_data.get("prediction")
                    
                    # Explicit NULL preservation: None is preserved as None (SQL NULL)
                    agent_risk = agent_data.get("risk_score")
                    raw_risk = agent_data.get("raw_risk_score")
                    agent_level = agent_data.get("risk_level")
                    agent_conf = agent_data.get("confidence")
                    reason = agent_data.get("reason")
                    curve = agent_data.get("calibration_curve")
                    meaning = agent_data.get("operational_meaning")
                    agent_time = agent_data.get("execution_time_ms")

                    conn.execute(
                        """
                        INSERT INTO agent_results (
                            analysis_id, agent_name, display_key, status,
                            signal_available, prediction, risk_score,
                            raw_risk_score, risk_level, confidence,
                            reason, calibration_curve, operational_meaning, execution_time_ms
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            analysis_id, agent_name, display_key, agent_status,
                            signal_avail, prediction, agent_risk,
                            raw_risk, agent_level, agent_conf,
                            reason, curve, meaning, agent_time,
                        ),
                    )

                # 3. Insert into risk_factors
                risk_factors = explanation.get("risk_factors") or []
                for rf in risk_factors:
                    if isinstance(rf, str):
                        factor_text = rf
                        rf_agent = None
                        rf_sev = None
                    elif isinstance(rf, dict):
                        factor_text = rf.get("factor") or rf.get("description") or str(rf)
                        rf_agent = rf.get("agent") or rf.get("agent_name")
                        rf_sev = rf.get("severity")
                    else:
                        factor_text = str(rf)
                        rf_agent = None
                        rf_sev = None

                    conn.execute(
                        """
                        INSERT INTO risk_factors (
                            analysis_id, agent_name, factor, severity
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (analysis_id, rf_agent, factor_text, rf_sev),
                    )

                # 4. Insert into evidence
                evidence_items = explanation.get("evidence") or []
                for ev in evidence_items:
                    if isinstance(ev, str):
                        ev_desc = ev
                        ev_agent = None
                        ev_type = None
                        ev_sev = None
                    elif isinstance(ev, dict):
                        ev_desc = ev.get("description") or str(ev)
                        ev_agent = ev.get("agent") or ev.get("agent_name")
                        ev_type = ev.get("evidence_type") or ev.get("type")
                        ev_sev = ev.get("severity")
                    else:
                        ev_desc = str(ev)
                        ev_agent = None
                        ev_type = None
                        ev_sev = None

                    conn.execute(
                        """
                        INSERT INTO evidence (
                            analysis_id, agent_name, evidence_type,
                            description, severity
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (analysis_id, ev_agent, ev_type, ev_desc, ev_sev),
                    )

            logger.info("Successfully persisted analysis %s for URL: %s", analysis_id, url)
            return True

        except Exception as exc:
            logger.error("Failed to persist analysis %s: %s", analysis_id, exc, exc_info=True)
            raise
        finally:
            conn.close()

    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a complete investigation by analysis_id.
        Reconstructs the full nested dictionary structure.
        """
        conn = get_connection(self.db_path)
        try:
            row = conn.execute(
                "SELECT * FROM analyses WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()

            if not row:
                return None

            # Fetch child records
            agent_rows = conn.execute(
                "SELECT * FROM agent_results WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchall()

            risk_factor_rows = conn.execute(
                "SELECT * FROM risk_factors WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchall()

            evidence_rows = conn.execute(
                "SELECT * FROM evidence WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchall()

            # Reconstruct agents dict
            agents_dict = {}
            for ar in agent_rows:
                key = ar["display_key"]
                agents_dict[key] = {
                    "agent": ar["agent_name"],
                    "status": ar["status"],
                    "signal": bool(ar["signal_available"]),
                    "prediction": ar["prediction"],
                    "risk_score": ar["risk_score"],
                    "raw_risk_score": ar["raw_risk_score"],
                    "risk_level": ar["risk_level"],
                    "confidence": ar["confidence"],
                    "reason": ar["reason"],
                    "calibration_curve": ar["calibration_curve"],
                    "operational_meaning": ar["operational_meaning"],
                    "risk_score_meaning": "operational_severity",
                    "confidence_meaning": "class_certainty",
                }

            # Reconstruct explanation
            risk_factors_list = [
                {
                    "factor": rf["factor"],
                    "agent": rf["agent_name"],
                    "severity": rf["severity"],
                }
                for rf in risk_factor_rows
            ]

            evidence_list = [
                {
                    "description": ev["description"],
                    "agent": ev["agent_name"],
                    "evidence_type": ev["evidence_type"],
                    "severity": ev["severity"],
                }
                for ev in evidence_rows
            ]

            # Reconstruct decision basis & calibration
            db_json = row["decision_basis_json"]
            decision_basis = json.loads(db_json) if db_json else {}

            rc_json = row["risk_calibration_json"]
            risk_calibration = json.loads(rc_json) if rc_json else {}

            return {
                "status": row["status"],
                "analysis_id": row["analysis_id"],
                "timestamp": row["created_at"],
                "target": {
                    "url": row["url"],
                    "domain": row["domain"],
                    "ip": None,
                },
                "final_assessment": {
                    "verdict": row["final_verdict"],
                    "risk_score": row["risk_score"],
                    "risk_level": row["risk_level"],
                    "confidence": row["confidence"],
                },
                "consensus": {
                    "available_agents": row["usable_agent_count"],
                    "total_agents": row["total_agent_count"],
                    "coverage": row["coverage"],
                    "consensus_available": bool(row["consensus_available"]),
                    "consensus_satisfied": bool(row["consensus_satisfied"]),
                    "conflict_detected": bool(row["conflict_detected"]),
                    "evidence_state": row["evidence_state"],
                    "limited_evidence": bool(row["limited_evidence"]),
                },
                "risk_calibration": risk_calibration,
                "agents": agents_dict,
                "explanation": {
                    "risk_factors": risk_factors_list,
                    "evidence": evidence_list,
                },
                                "meta": {
                    "execution_time_ms": row["execution_time_ms"],
                    "decision_basis": decision_basis,
                },
                "notes": row["notes"],
                "tags": row["tags"],
                "is_saved": bool(row["is_saved"]),
            }

        finally:
            conn.close()

    def get_recent_analyses(self, limit: int = 50, offset: int = 0,
                            search: str = None, verdict: str = None,
                            risk_level: str = None, status: str = None,
                            saved_only: bool = False, date_filter: str = None,
                            sort_by: str = 'newest') -> dict:
        """Retrieve a paginated, filtered, and sorted list of previous investigations."""
        conn = get_connection(self.db_path)
        try:
            query_conditions = []
            params = []
            
            if search:
                query_conditions.append("(url LIKE ? OR domain LIKE ? OR analysis_id LIKE ? OR tags LIKE ?)")
                search_term = f"%{search}%"
                params.extend([search_term, search_term, search_term, search_term])
                
            if verdict and verdict.lower() != 'all':
                query_conditions.append("final_verdict = ?")
                params.append(verdict.lower())
                
            if risk_level and risk_level.lower() != 'all':
                query_conditions.append("risk_level = ?")
                params.append(risk_level.lower())
                
            if status and status.lower() != 'all':
                query_conditions.append("status = ?")
                params.append(status.lower())
                
            if saved_only:
                query_conditions.append("is_saved = 1")
                
            if date_filter and date_filter.lower() != 'all':
                if date_filter == 'today':
                    query_conditions.append("date(created_at) = date('now')")
                elif date_filter == 'yesterday':
                    query_conditions.append("date(created_at) = date('now', '-1 day')")
                elif date_filter == 'last7days':
                    query_conditions.append("date(created_at) >= date('now', '-7 days')")
                elif date_filter == 'last30days':
                    query_conditions.append("date(created_at) >= date('now', '-30 days')")
                    
            where_clause = " WHERE " + " AND ".join(query_conditions) if query_conditions else ""
            
            # Count total matching
            count_query = f"SELECT COUNT(*) FROM analyses{where_clause}"
            total_count = conn.execute(count_query, params).fetchone()[0]
            
            # Sorting logic
            order_clause = "ORDER BY created_at DESC"
            if sort_by == 'oldest':
                order_clause = "ORDER BY created_at ASC"
            elif sort_by == 'risk_high':
                order_clause = "ORDER BY risk_score DESC NULLS LAST"
            elif sort_by == 'risk_low':
                order_clause = "ORDER BY risk_score ASC NULLS LAST"
            elif sort_by == 'conf_high':
                order_clause = "ORDER BY confidence DESC NULLS LAST"
            elif sort_by == 'conf_low':
                order_clause = "ORDER BY confidence ASC NULLS LAST"

            query = f"""
                SELECT analysis_id, url, domain, created_at, status,
                       final_verdict, risk_score, risk_level, confidence,
                       usable_agent_count, total_agent_count, conflict_detected,
                       is_saved, tags
                FROM analyses
                {where_clause}
                {order_clause}
                LIMIT ? OFFSET ?
            """
            
            rows = conn.execute(query, params + [limit, offset]).fetchall()

            analyses = [
                {
                    "analysis_id": r["analysis_id"],
                    "url": r["url"],
                    "domain": r["domain"],
                    "created_at": r["created_at"],
                    "status": r["status"],
                    "final_verdict": r["final_verdict"],
                    "risk_score": r["risk_score"],
                    "risk_level": r["risk_level"],
                    "confidence": r["confidence"],
                    "available_agents": r["usable_agent_count"],
                    "total_agents": r["total_agent_count"],
                    "conflict_detected": bool(r["conflict_detected"]),
                    "is_saved": bool(r["is_saved"]),
                    "tags": r["tags"]
                }
                for r in rows
            ]
            
            # Also get overall stats
            stats_row = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) as high_risk,
                    SUM(CASE WHEN risk_level = 'critical' THEN 1 ELSE 0 END) as critical,
                    SUM(CASE WHEN date(created_at) = date('now') THEN 1 ELSE 0 END) as today
                FROM analyses
            """).fetchone()

            return {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "count": len(analyses),
                "analyses": analyses,
                "stats": {
                    "total_investigations": stats_row["total"] or 0,
                    "high_risk": stats_row["high_risk"] or 0,
                    "critical": stats_row["critical"] or 0,
                    "today": stats_row["today"] or 0
                }
            }

        finally:
            conn.close()

    
    def update_investigation(self, analysis_id: str, notes: Optional[str] = None, tags: Optional[str] = None, is_saved: Optional[bool] = None) -> bool:
        """Update notes, tags, and saved status for a specific investigation (Day 19/20)."""
        conn = get_connection(self.db_path)
        try:
            with conn:
                # Build dynamic query
                updates = []
                params = []
                
                if notes is not None:
                    updates.append("notes = ?")
                    params.append(notes)
                if tags is not None:
                    updates.append("tags = ?")
                    params.append(tags)
                if is_saved is not None:
                    updates.append("is_saved = ?")
                    params.append(1 if is_saved else 0)
                    
                if not updates:
                    return True
                    
                query = f"UPDATE analyses SET {', '.join(updates)} WHERE analysis_id = ?"
                params.append(analysis_id)
                
                cursor = conn.execute(query, params)
                return cursor.rowcount > 0
        except Exception as e:
            logger.error("Failed to update investigation %s: %s", analysis_id, e)
            return False
        finally:
            conn.close()
            
    def delete_analysis(self, analysis_id: str) -> bool:
        """
        Delete an analysis by analysis_id. Foreign keys cascade to child tables.
        Returns True if a record was deleted, False otherwise.
        """
        conn = get_connection(self.db_path)
        try:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM analyses WHERE analysis_id = ?",
                    (analysis_id,),
                )
                return cursor.rowcount > 0
        finally:
            conn.close()
