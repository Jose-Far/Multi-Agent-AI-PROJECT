import json
import logging
import io
import datetime
from typing import Dict, Any, Optional
from database.repository import AnalysisRepository, get_connection

try:
    from xhtml2pdf import pisa
    from jinja2 import Environment, FileSystemLoader
    HAS_PDF_LIBS = True
except ImportError:
    HAS_PDF_LIBS = False

logger = logging.getLogger(__name__)

class ReportService:
    def __init__(self, db_path: str = "data/phishdec.db"):
        self.repo = AnalysisRepository(db_path)
        if HAS_PDF_LIBS:
            self.jinja_env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
        
    def generate_report_data(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Gather all investigation data and build a structured ReportData dictionary."""
        investigation = self.repo.get_analysis(analysis_id)
        if not investigation:
            return None
            
        target_dict = investigation.get("target", {})
        final_assessment = investigation.get("final_assessment", {})
        consensus = investigation.get("consensus", {})
        explanation = investigation.get("explanation", {})
            
        report = {
            "metadata": {
                "analysis_id": investigation.get("analysis_id"),
                "target": target_dict.get("domain") or target_dict.get("url"),
                "url": target_dict.get("url"),
                "date": investigation.get("timestamp"),
                "status": investigation.get("status")
            },
            "summary": {
                "verdict": final_assessment.get("verdict", "unknown").upper() if final_assessment.get("verdict") else "UNKNOWN",
                "risk_score": final_assessment.get("risk_score"),
                "risk_level": final_assessment.get("risk_level", "UNKNOWN") or "UNKNOWN",
                "confidence": final_assessment.get("confidence", 0),
                "coverage": f"{consensus.get('available_agents', 0)} / {consensus.get('total_agents', 6)}",
                "is_partial": investigation.get("status") == "limited_evidence"
            },
            "agents": investigation.get("agents", {}),
            "evidence": {},
            "fusion": consensus,
            "notes": investigation.get("notes"),
            "tags": [],
            "technical": {
                "fusion_engine": "Decision Fusion Engine",
                "database": "SQLite",
                "execution_time_ms": investigation.get("meta", {}).get("execution_time_ms", 0)
            },
            "risk_factors": explanation.get("risk_factors", []),
            "explainability": {
                "risk_factors": explanation.get("risk_factors", []),
                "decision_basis": investigation.get("meta", {}).get("decision_basis", {})
            },
            "timeline": {
                "created_at": investigation.get("timestamp"),
                "events": ["Investigation started", "Agents completed", "Fusion consensus reached"]
            }
        }

        # Parse tags
        try:
            raw_tags = investigation.get("tags")
            if raw_tags:
                report["tags"] = json.loads(raw_tags) if isinstance(raw_tags, str) else raw_tags
        except Exception:
            report["tags"] = []

        # Build clean evidence structure
        ev_list_data = explanation.get("evidence", [])
        for ev in ev_list_data:
            agent_raw = ev.get("agent")
            agent = (agent_raw if agent_raw else "System Telemetry").title().replace("_", " ")
            
            if agent not in report["evidence"]:
                report["evidence"][agent] = []
                
            desc_raw = ev.get('description', '')
            
            # Attempt to parse dictionary string for cleaner output
            if desc_raw.startswith('{') and desc_raw.endswith('}'):
                try:
                    import ast
                    desc_dict = ast.literal_eval(desc_raw)
                    if 'type' in desc_dict and 'target' in desc_dict:
                        desc_raw = f"Analysis Target: {desc_dict['target']}"
                    elif 'type' in desc_dict and 'model_status' in desc_dict:
                        desc_raw = f"Model: {desc_dict.get('model_type', 'Unknown')} ({desc_dict['model_status']})"
                    elif 'type' in desc_dict and 'risk_score' in desc_dict:
                        desc_raw = f"Risk Assessment: Score {desc_dict['risk_score']} [{desc_dict.get('risk_level', '').upper()}]"
                    else:
                        parts = [f"{k}: {v}" for k, v in desc_dict.items() if k != 'type']
                        desc_raw = ", ".join(parts)
                except:
                    pass
                    
            report["evidence"][agent].append(f"[{(ev.get('severity') or 'info').upper()}] {desc_raw}")

        return report

    def generate_pdf_report(self, analysis_id: str) -> Optional[bytes]:
        if not HAS_PDF_LIBS:
            raise RuntimeError("PDF dependencies (xhtml2pdf, jinja2) are not installed.")
            
        data = self.generate_report_data(analysis_id)
        if not data:
            return None
            
        template = self.jinja_env.get_template("report_template.html")
        html_out = template.render(report=data)
        
        pdf_io = io.BytesIO()
        pisa_status = pisa.CreatePDF(html_out, dest=pdf_io, encoding='utf-8')
        
        if pisa_status.err:
            logger.error("Failed to generate PDF for %s: %s", analysis_id, pisa_status.err)
            raise RuntimeError("PDF Generation failed")
            
        return pdf_io.getvalue()
