import sqlite3
import datetime
import io
import csv
from typing import Dict, Any, List

class AnalyticsService:
    def __init__(self, db_path: str = "data/phishdec.db"):
        self.db_path = db_path

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_dashboard_analytics(self, period_days: int = 30, filters: Dict[str, str] = None) -> Dict[str, Any]:
        """Aggregate all metrics for the Admin Analytics Dashboard with filters."""
        cutoff_date = (datetime.datetime.utcnow() - datetime.timedelta(days=period_days)).isoformat()
        filters = filters or {}
        
        # Base filter conditions
        conditions = ["created_at >= ?"]
        params = [cutoff_date]
        
        if filters.get("verdict"):
            conditions.append("final_verdict = ?")
            params.append(filters["verdict"])
        if filters.get("risk"):
            conditions.append("risk_level = ?")
            params.append(filters["risk"])
            
        where_clause = " AND ".join(conditions)

        with self._get_connection() as conn:
            # 1. Total & Today Analyses
            today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
            total_analyses = conn.execute(f"SELECT COUNT(*) as c FROM analyses WHERE {where_clause}", params).fetchone()["c"]
            today_analyses = conn.execute(f"SELECT COUNT(*) as c FROM analyses WHERE {where_clause} AND created_at >= ?", params + [today]).fetchone()["c"]
            
            # 2. Verdict Distribution
            verdicts = {"phishing": 0, "suspicious": 0, "legitimate": 0, "unknown": 0}
            for row in conn.execute(f"SELECT final_verdict, COUNT(*) as c FROM analyses WHERE {where_clause} GROUP BY final_verdict", params):
                v = row["final_verdict"].lower()
                if v in verdicts:
                    verdicts[v] = row["c"]
                else:
                    verdicts["unknown"] += row["c"]
                    
            # 3. Risk Distribution
            risks = {"critical": 0, "high": 0, "medium": 0, "low": 0, "unknown": 0}
            for row in conn.execute(f"SELECT risk_level, COUNT(*) as c FROM analyses WHERE {where_clause} GROUP BY risk_level", params):
                r = (row["risk_level"] or "unknown").lower()
                if r in risks:
                    risks[r] = row["c"]
                else:
                    risks["unknown"] += row["c"]
                    
            # 4. Coverage (How many agents succeeded per analysis)
            coverage = {"6/6": 0, "5/6": 0, "4/6": 0, "3/6": 0, "2/6": 0, "1/6": 0}
            avg_coverage_sum = 0
            for row in conn.execute(f"SELECT usable_agent_count, COUNT(*) as c FROM analyses WHERE {where_clause}", params):
                count = row["usable_agent_count"] or 0
                c_total = row["c"]
                avg_coverage_sum += (count * c_total)
                label = f"{min(6, count)}/6"
                if label in coverage:
                    coverage[label] += c_total
            
            avg_coverage = round(avg_coverage_sum / total_analyses, 1) if total_analyses > 0 else 0

            # 5. Fusion Statistics
            fusion = {
                "decisions": total_analyses,
                "consensus_available": 0,
                "consensus_satisfied": 0,
                "limited_evidence": 0,
                "conflicts": 0,
                "unknown_verdicts": verdicts["unknown"]
            }
            fusion_row = conn.execute(f"""
                SELECT 
                    SUM(CASE WHEN consensus_available = 1 THEN 1 ELSE 0 END) as c_avail,
                    SUM(CASE WHEN consensus_satisfied = 1 THEN 1 ELSE 0 END) as c_sat,
                    SUM(CASE WHEN limited_evidence = 1 THEN 1 ELSE 0 END) as lim,
                    SUM(CASE WHEN conflict_detected = 1 THEN 1 ELSE 0 END) as conf
                FROM analyses WHERE {where_clause}
            """, params).fetchone()
            
            if fusion_row and fusion_row["c_avail"] is not None:
                fusion["consensus_available"] = fusion_row["c_avail"]
                fusion["consensus_satisfied"] = fusion_row["c_sat"]
                fusion["limited_evidence"] = fusion_row["lim"]
                fusion["conflicts"] = fusion_row["conf"]

            # 6. Six-Agent Analytics
            agents_metrics = {}
            agent_filter_sql = ""
            if filters.get("agent"):
                agent_filter_sql = " AND display_key = ?"
                params_agent = params + [filters["agent"]]
            else:
                agent_filter_sql = ""
                params_agent = params

            for row in conn.execute(f"""
                SELECT 
                    display_key,
                    COUNT(*) as executions,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes,
                    SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) as errors,
                    AVG(execution_time_ms) as avg_latency
                FROM agent_results 
                WHERE analysis_id IN (SELECT analysis_id FROM analyses WHERE {where_clause})
                {agent_filter_sql}
                GROUP BY display_key
            """, params_agent):
                name = row["display_key"]
                execs = row["executions"]
                succ = row["successes"]
                avail = round((succ / execs * 100), 1) if execs > 0 else 0
                
                agents_metrics[name] = {
                    "name": name.replace("_Agent", "").replace("_", " "),
                    "executions": execs,
                    "success": succ,
                    "errors": row["errors"],
                    "availability": avail,
                    "avg_latency": round(row["avg_latency"] or 0, 1)
                }

            # 7. Volume over time
            volume_trend = []
            for row in conn.execute(f"""
                SELECT substr(created_at, 1, 10) as day, COUNT(*) as c 
                FROM analyses 
                WHERE {where_clause}
                GROUP BY day ORDER BY day ASC
            """, params):
                volume_trend.append({"date": row["day"], "count": row["c"]})

            return {
                "overview": {
                    "total_analyses": total_analyses,
                    "today": today_analyses,
                    "reports_generated": total_analyses,
                    "avg_coverage": avg_coverage
                },
                "volume_trend": volume_trend,
                "verdicts": verdicts,
                "risks": risks,
                "coverage": coverage,
                "fusion": fusion,
                "agents": agents_metrics
            }

    def export_analytics_csv(self, period_days: int = 30) -> str:
        """Export basic analytics aggregates to CSV format."""
        data = self.get_dashboard_analytics(period_days)
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Analyses", data["overview"]["total_analyses"]])
        writer.writerow(["Analyses Today", data["overview"]["today"]])
        writer.writerow(["Average Coverage", data["overview"]["avg_coverage"]])
        writer.writerow([])
        
        writer.writerow(["Verdict", "Count"])
        for k, v in data["verdicts"].items():
            writer.writerow([k.title(), v])
        writer.writerow([])
        
        writer.writerow(["Agent", "Executions", "Success", "Errors", "Availability (%)", "Avg Latency (ms)"])
        for agent in data["agents"].values():
            writer.writerow([agent["name"].title(), agent["executions"], agent["success"], agent["errors"], agent["availability"], agent["avg_latency"]])
            
        return output.getvalue()
