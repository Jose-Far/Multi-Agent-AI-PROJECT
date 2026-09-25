"""
api/app.py â€” PhishDec Flask API
================================

Day 15 â€” Backend Integration Layer

Exposes the full 6-agent cybersecurity analysis pipeline through two
clean HTTP endpoints:

    GET  /health     â€” Liveness / readiness check
    POST /analyze    â€” Main phishing analysis endpoint

Pipeline:
    POST /analyze
          â†“
    Input Validation
          â†“
    MultiAgentOrchestrator.analyze(url)
          â†“
    6 AI Agents (URL + HTML + SSL + DNS + Visual + Threat Intel)
          â†“
    Decision Fusion Engine v17
          â†“
    AnalysisResult
          â†“
    JSON response

The UI never needs to know how the individual agents work.
"""

import os
from dotenv import load_dotenv
load_dotenv()
DB_PATH = "data/phishdec.db"
import sys


# ============================================================================
# GET /admin/system
# ============================================================================


# ============================================================================
# PROJECT ROOT PATH
# ============================================================================
# Allows api/app.py to import sibling packages such as:
# agents, Collector, feature_extraction, etc.
#
# This fixes:
# ModuleNotFoundError: No module named 'agents'
# when running:
#     python api/app.py
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import logging
from performance.config import PerformanceConfig
import threading
import re
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from flask import Flask, jsonify, request, g
from api.auth_middleware import login_required, roles_required, permission_required, get_current_user
from services.auth_service import AuthService
from services.alert_engine import AlertEngine
from services.notification_service import NotificationService

from agents.orchestrator import MultiAgentOrchestrator
from agents.analysis_result import AnalysisResult
from database.database import initialize_database
from database.repository import AnalysisRepository


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================================
# FLASK APPLICATION
# ============================================================================

app = Flask(__name__)

# ----------------------------------------------------------------------------
# CONCURRENCY CONTROL (DAY 27)
# ----------------------------------------------------------------------------
analysis_semaphore = threading.Semaphore(PerformanceConfig.MAX_CONCURRENT_ANALYSIS)

from functools import wraps

def limit_concurrency(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        acquired = analysis_semaphore.acquire(timeout=PerformanceConfig.REQUEST_TIMEOUT)
        if not acquired:
            return jsonify({
                "status": "error",
                "error": "TOO_MANY_REQUESTS",
                "message": "Maximum concurrent analyses reached. Please try again later."
            }), 429
        try:
            return f(*args, **kwargs)
        finally:
            analysis_semaphore.release()
    return decorated



app.config["JSON_SORT_KEYS"] = False


# ============================================================================
# SECURITY HEADERS, CORS & RATE LIMITING
# ============================================================================

from security.config import SecurityConfig
from security.rate_limiter import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    max_requests=SecurityConfig.RATE_LIMIT_REQUESTS,
    window_sec=SecurityConfig.RATE_LIMIT_WINDOW_SEC
)

@app.after_request
def add_security_and_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,X-Session-Token,X-User-Role"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
    
    # Inject security hardening headers
    for header_name, header_val in SecurityConfig.SECURITY_HEADERS.items():
        response.headers[header_name] = header_val
        
    return response

@app.after_request
def audit_log_user_activity(response):
    """Log user activities, changes, and searches."""
    try:
        if request.path.startswith("/health") or "monitoring" in request.path or "analytics" in request.path or request.method == "OPTIONS":
            return response
            
        user = getattr(g, "current_user", None)
        if not user:
            return response

        method = request.method
        if method in ["POST", "PATCH", "DELETE", "PUT"]:
            if "login" in request.path or "logout" in request.path:
                return response
                
            details = f"Performed {method} on {request.path}"
            event_type = "USER_ACTIVITY"
            
            if "analyze" in request.path:
                try:
                    j = request.json or {}
                    target_url = j.get("url", j.get("target_url", "Unknown"))
                    details = f"Submitted new URL for analysis: {target_url}"
                except:
                    pass
            elif "users" in request.path:
                details = f"Admin modification applied to users at {request.path}"
            elif "analyses" in request.path:
                details = f"Modified analysis record at {request.path}"

            from services.auth_service import AuthService
            AuthService(db_path=DB_PATH).log_audit_event(
                event_type=event_type,
                user_id=user["user_id"],
                username=user["username"],
                ip_address=request.remote_addr,
                details=details
            )
        elif method == "GET" and dict(request.args):
            if request.path.startswith("/analyses") or request.path.startswith("/api/admin/audit"):
                args = dict(request.args)
                if "q" in args or "search" in args or "status" in args:
                    details = f"Performed search/query on {request.path}. Parameters: {args}"
                    from services.auth_service import AuthService
                    AuthService(db_path=DB_PATH).log_audit_event(
                        event_type="USER_SEARCH",
                        user_id=user["user_id"],
                        username=user["username"],
                        ip_address=request.remote_addr,
                        details=details
                    )
    except Exception as e:
        logger.error(f"Audit log middleware error: {e}")
    return response


@app.before_request
def handle_options_and_rate_limiting():
    if request.method == "OPTIONS":
        return "", 204

    # Skip rate limiting on health check endpoint
    if request.path in ("/health",):
        return None

    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
    allowed, retry_after = rate_limiter.is_allowed(client_ip)
    if not allowed:
        return jsonify({
            "status": "error",
            "error": "RATE_LIMIT_EXCEEDED",
            "message": f"Rate limit exceeded. Try again in {retry_after} seconds."
        }), 429



# ============================================================================
# GLOBAL SYSTEM COMPONENTS (Orchestrator, Collector, Feature Manager, Database)
# ============================================================================

logger.info("Initializing system components...")

_orchestrator: Optional[MultiAgentOrchestrator] = None
_collection_manager: Any = None
_feature_manager: Any = None
_repository: Optional[AnalysisRepository] = None


def _get_orchestrator() -> MultiAgentOrchestrator:
    """Lazily initialize and cache the orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator()
        logger.info("MultiAgentOrchestrator ready.")
    return _orchestrator


def _get_collector():
    """Lazily initialize and cache the CollectionManager."""
    global _collection_manager
    if _collection_manager is None:
        try:
            from Collector.manager import CollectionManager
            _collection_manager = CollectionManager()
            logger.info("CollectionManager ready.")
        except Exception as exc:
            logger.warning("Could not initialize CollectionManager: %s", exc)
    return _collection_manager


def _get_feature_manager():
    """Lazily initialize and cache the FeatureManager."""
    global _feature_manager
    if _feature_manager is None:
        try:
            from feature_extraction.manager import FeatureManager
            _feature_manager = FeatureManager()
            logger.info("FeatureManager ready.")
        except Exception as exc:
            logger.warning("Could not initialize FeatureManager: %s", exc)
    return _feature_manager


def _get_repository() -> AnalysisRepository:
    """Lazily initialize and cache the AnalysisRepository."""
    global _repository
    if _repository is None:
        try:
            initialize_database()
            _repository = AnalysisRepository()
            logger.info("AnalysisRepository ready.")
        except Exception as exc:
            logger.error("Could not initialize AnalysisRepository: %s", exc, exc_info=True)
            raise
    return _repository


# Pre-warm on startup so the first request is not slow.
try:
    _get_orchestrator()
    _get_collector()
    _get_feature_manager()
    _get_repository()
except Exception as _exc:
    logger.error("Component initialization failed at startup: %s", _exc)


# ============================================================================
# INPUT VALIDATION
# ============================================================================

_ALLOWED_SCHEMES = {"http", "https"}

_DOMAIN_RE = re.compile(
    r"^[a-zA-Z0-9]"          # starts with alnum
    r"([a-zA-Z0-9\-]*"       # body
    r"[a-zA-Z0-9])?"         # end with alnum (optional, single-label domains)
    r"(\.[a-zA-Z]{2,})+$",   # at least one dot-extension
)


from security.validator import SSRFValidator

def _validate_url(url):
    if not isinstance(url, str):
        return False, "", "INVALID_URL"
    url = url.strip()
    if not url:
        return False, "", "EMPTY_URL"
    if "://" not in url:
        url = "https://" + url

    # Use central SSRF validator
    is_safe, clean_url, err_msg = SSRFValidator.validate_url(url)
    if not is_safe:
        return False, "", err_msg

    return True, clean_url, ""


# ============================================================================
# ERROR HELPER
# ============================================================================

def _error(code: str, message: str, http_status: int) -> Any:
    return jsonify(
        {
            "status":  "error",
            "error":   code,
            "message": message,
        }
    ), http_status


# ============================================================================
# GET /health
# ============================================================================

@app.route("/health", methods=["GET"])
def health() -> Any:
    """
    Liveness / readiness check.

    Returns information about the orchestrator and Fusion Engine so that
    monitoring systems can verify the full pipeline is loaded.
    """

    try:
        orch = _get_orchestrator()
        hc   = orch.health_check()
        st   = orch.get_status()

        return jsonify(
            {
                "status":           "healthy",
                "orchestrator":     st.get("orchestrator_name") or st.get("orchestrator", "Multi-Agent Cybersecurity Orchestrator"),
                "orchestrator_version": st.get("orchestrator_version"),
                "active_agents":    st.get("active_agent_count", 6),
                "fusion_version":   st.get("fusion_engine", {}).get("engine_version"),
                "fusion_ready":     hc.get("fusion_ready", False),
                "configuration_consistent": hc.get("configuration_consistent", False),
            }
        ), 200

    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return jsonify(
            {
                "status":  "unhealthy",
                "error":   str(exc),
            }
        ), 500


# ============================================================================
# POST /analyze
# ============================================================================

@app.route("/analyze", methods=["POST"])
@app.route("/api/analyze", methods=["POST"])
@roles_required("super_admin", "admin", "analyst")
@limit_concurrency
def analyze() -> Any:
    """
    Main phishing analysis endpoint.

    Request body (JSON):
        {"url": "https://example.com"}

    Response:
        AnalysisResult JSON
    """

    # â”€â”€ 1. Parse body â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return _error(
            "MISSING_PAYLOAD",
            "Request body must be a JSON object with a 'url' field.",
            400,
        )

    if "url" not in payload:
        return _error(
            "MISSING_URL",
            "Missing required field 'url' in request body.",
            400,
        )

    # â”€â”€ 2. Validate URL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    ok, clean_url, err_code = _validate_url(payload.get("url"))

    if not ok:
        messages = {
            "EMPTY_URL":             "The 'url' field must contain a non-empty string.",
            "INVALID_URL":           "The provided URL is not valid. Please supply a complete URL (e.g. https://example.com).",
            "UNSUPPORTED_PROTOCOL":  "Only http:// and https:// URLs are supported.",
        }
        return _error(err_code, messages.get(err_code, "Invalid URL."), 400)

    # â”€â”€ 3. Run pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    logger.info("Analysis requested for: %s", clean_url)

    # Determine input mode:
    # - caller can pass pre-extracted feature blocks directly in payload (e.g. "html_features", etc.)
    # - caller can pass "live": true to force full data collection + feature extraction
    # - or "url_only": true to run in lightweight URL-only mode
    has_custom_features = any(
        k in payload for k in (
            "url_features", "html_features", "ssl_features",
            "dns_features", "visual_features", "threat_features"
        )
    )
    is_live = payload.get("live", False)
    is_url_only = payload.get("url_only", False)

    try:

        orch = _get_orchestrator()

        if has_custom_features:
            # Caller supplied pre-extracted feature blocks
            vector = dict(payload)
            vector["url"] = clean_url
            orch_result = orch.analyze(vector)

        elif is_url_only:
            # Explicit fast test mode: pass URL directly to orchestrator without telemetry
            logger.info("Executing URL-only mode for: %s", clean_url)
            orch_result = orch.analyze(clean_url)

        else:
            # Standard production mode (default): Full live scanning pipeline
            # Collection -> Feature Extraction -> Multi-Agent Orchestrator
            logger.info("Executing Live Telemetry Collection for: %s", clean_url)
            collector = _get_collector()
            feature_mgr = _get_feature_manager()

            if not collector:
                raise RuntimeError(
                    "CollectionManager failed to initialize at startup. "
                    "Check logs for the import error. "
                    "You can use 'url_only': true to bypass live collection."
                )

            if not feature_mgr:
                raise RuntimeError(
                    "FeatureManager failed to initialize at startup. "
                    "Check logs for the import error. "
                    "You can use 'url_only': true to bypass live collection."
                )

            col_res = collector.execute_collection(clean_url)

            if col_res.get("status") == "error":
                raise RuntimeError(
                    "Live data collection failed: "
                    + str(col_res.get("message", "unknown error"))
                )

            vector = feature_mgr.process_and_store(col_res)

            if not vector or not isinstance(vector, dict):
                raise RuntimeError(
                    "FeatureManager returned an empty feature vector. "
                    "All 6 collectors succeeded but feature extraction produced nothing."
                )

            logger.info(
                "Feature vector ready â€” blocks: %s",
                {k: len(v) for k, v in vector.items()
                 if k.endswith("_features") and isinstance(v, dict)}
            )

            orch_result = orch.analyze(vector)

        if not isinstance(orch_result, dict):
            raise RuntimeError("Orchestrator returned a non-dictionary result.")

    except Exception as exc:

        logger.error("Orchestrator execution failed for %s: %s", clean_url, exc, exc_info=True)

        return jsonify(
            {
                "status":      "error",
                "error":       "PIPELINE_ERROR",
                "message":     f"Analysis pipeline failed: {exc}",
                "target": {
                    "url": clean_url,
                },
            }
        ), 500

    # â”€â”€ 4. Build AnalysisResult â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    try:
        result = AnalysisResult.from_orchestrator_output(orch_result, clean_url)
        response_dict = result.to_dict()
    except Exception as exc:
        logger.error("AnalysisResult construction failed: %s", exc, exc_info=True)
        # Fallback â€” return raw orchestrator output so the caller gets something
        response_dict = {
            "status":      "error",
            "error":       "RESULT_BUILD_ERROR",
            "message":     f"Result serialization failed: {exc}",
            "raw":         orch_result,
        }

    # â”€â”€ 5. Persist to Database (Day 16) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    saved = False
    if "analysis_id" in response_dict and response_dict.get("status") != "error":
        try:
            repo = _get_repository()
            saved = repo.save_analysis(response_dict)
        except Exception as db_exc:
            logger.error(
                "Database persistence failed for %s: %s",
                response_dict.get("analysis_id"),
                db_exc,
                exc_info=True,
            )
            saved = False
            
    # Phase 24: Alert Engine Evaluation
    if "analysis_id" in response_dict and response_dict.get("status") != "error":
        try:
            AlertEngine(db_path=DB_PATH).evaluate_investigation(response_dict)
        except Exception as alert_exc:
            logger.error(f"Alert Engine evaluation failed: {alert_exc}")

    response_dict["persistence"] = {
        "saved": saved,
        "database": "sqlite",
    }

    logger.info(
        "Analysis complete for %s | verdict=%s | agents=%s/%s | persisted=%s",
        clean_url,
        response_dict.get("final_assessment", {}).get("verdict"),
        response_dict.get("consensus", {}).get("available_agents"),
        response_dict.get("consensus", {}).get("total_agents"),
        saved,
    )

    return jsonify(response_dict), 200


# ============================================================================
# INVESTIGATION HISTORY ENDPOINTS (Day 16)
# ============================================================================

@app.route("/analyses", methods=["GET"])
@login_required
def list_analyses() -> Any:
    """
    Return paginated, filtered, and sorted list of previous investigations for the history UI.
    """
    try:
        limit = min(int(request.args.get("limit", 50)), 200)
        offset = max(int(request.args.get("offset", 0)), 0)
    except (TypeError, ValueError):
        return _error("INVALID_PAGINATION", "The 'limit' and 'offset' parameters must be integers.", 400)

    try:
        repo = _get_repository()
        
        search = request.args.get("search")
        verdict = request.args.get("verdict")
        risk_level = request.args.get("risk_level")
        status = request.args.get("status")
        saved_only = request.args.get("saved_only", "false").lower() == "true"
        date_filter = request.args.get("date_filter")
        sort_by = request.args.get("sort", "newest")
        
        result = repo.get_recent_analyses(
            limit=limit,
            offset=offset,
            search=search,
            verdict=verdict,
            risk_level=risk_level,
            status=status,
            saved_only=saved_only,
            date_filter=date_filter,
            sort_by=sort_by
        )
        return jsonify({
            "status": "success",
            **result
        }), 200
    except Exception as exc:
        logger.error("Failed to list analyses: %s", exc, exc_info=True)
        return _error("DATABASE_ERROR", f"Failed to retrieve investigations: {exc}", 500)


@app.route("/analyses/<analysis_id>", methods=["GET"])
@login_required
def get_analysis_detail(analysis_id: str) -> Any:
    """
    Return complete investigation details by analysis_id.
    """
    analysis_id = (analysis_id or "").strip()
    if not analysis_id:
        return _error("INVALID_ID", "analysis_id cannot be empty.", 400)

    try:
        repo = _get_repository()
        investigation = repo.get_analysis(analysis_id)
        if not investigation:
            return _error("ANALYSIS_NOT_FOUND", f"Analysis '{analysis_id}' was not found.", 404)
        return jsonify(investigation), 200
    except Exception as exc:
        logger.error("Failed to retrieve analysis %s: %s", analysis_id, exc, exc_info=True)
        return _error("DATABASE_ERROR", f"Failed to retrieve investigation: {exc}", 500)


@app.route("/analyses/<analysis_id>", methods=["DELETE"])
@roles_required("super_admin", "admin")
def delete_analysis_record(analysis_id: str) -> Any:
    """
    Delete an investigation by analysis_id (cascades to all child records).
    """
    analysis_id = (analysis_id or "").strip()
    if not analysis_id:
        return _error("INVALID_ID", "analysis_id cannot be empty.", 400)

    try:
        repo = _get_repository()
        deleted = repo.delete_analysis(analysis_id)
        if not deleted:
            return _error("ANALYSIS_NOT_FOUND", f"Analysis '{analysis_id}' was not found.", 404)
        return jsonify({
            "status": "success",
            "deleted": True,
            "analysis_id": analysis_id,
        }), 200
    except Exception as exc:
        logger.error("Failed to delete analysis %s: %s", analysis_id, exc, exc_info=True)
        return _error("DATABASE_ERROR", f"Failed to delete investigation: {exc}", 500)


# ============================================================================
# GLOBAL ERROR HANDLERS
# ============================================================================

@app.route("/analyses/<analysis_id>", methods=["PATCH"])
@roles_required("super_admin", "admin", "analyst")
def update_analysis_record(analysis_id: str) -> Any:
    """Update notes, tags, and saved status for a specific investigation."""
    analysis_id = (analysis_id or "").strip()
    if not analysis_id:
        return _error("INVALID_ID", "analysis_id cannot be empty.", 400)

    if not request.is_json:
        return _error("INVALID_CONTENT_TYPE", "Request must be application/json", 415)

    try:
        data = request.get_json()
        notes = data.get("notes")
        tags = data.get("tags")
        if isinstance(tags, list):
            import json
            tags = json.dumps(tags)
            
        is_saved = data.get("is_saved")
        
        repo = _get_repository()
        success = repo.update_investigation(analysis_id, notes=notes, tags=tags, is_saved=is_saved)
        
        if not success:
            return _error("UPDATE_FAILED", "Failed to update investigation or investigation not found.", 404)
            
        return jsonify({"status": "success", "message": "Investigation updated successfully."}), 200
        
    except Exception as exc:
        logger.error("Failed to update analysis: %s", exc, exc_info=True)
        return _error("INTERNAL_ERROR", "Failed to update investigation.", 500)


# ============================================================================
# REPORT GENERATION (Day 21)
# ============================================================================

from services.report_service import ReportService
from services.analytics_service import AnalyticsService
from services.monitoring_service import MonitoringService
from services.alert_engine import AlertEngine
from services.notification_service import NotificationService

@app.route("/analyses/<analysis_id>/report", methods=["GET"])
@login_required
def get_report_data(analysis_id: str) -> Any:
    """Retrieve structured report data for frontend preview."""
    analysis_id = (analysis_id or "").strip()
    if not analysis_id:
        return _error("INVALID_ID", "analysis_id cannot be empty.", 400)
        
    try:
        report_service = ReportService()
        data = report_service.generate_report_data(analysis_id)
        if not data:
            return _error("NOT_FOUND", "Investigation not found.", 404)
            
        return jsonify({"status": "success", "report": data}), 200
    except Exception as exc:
        logger.error("Failed to generate report data: %s", exc, exc_info=True)
        return _error("REPORT_ERROR", "Failed to build report.", 500)

@app.route("/analyses/<analysis_id>/report/pdf", methods=["GET"])
@login_required
def download_pdf_report(analysis_id: str) -> Any:
    """Generate and return PDF report binary."""
    analysis_id = (analysis_id or "").strip()
    if not analysis_id:
        return _error("INVALID_ID", "analysis_id cannot be empty.", 400)
        
    try:
        from flask import send_file
        import io
        report_service = ReportService()
        pdf_bytes = report_service.generate_pdf_report(analysis_id)
        
        if not pdf_bytes:
            return _error("NOT_FOUND", "Investigation not found.", 404)
            
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"PhishDec_Report_{analysis_id}.pdf"
        )
    except RuntimeError as rexc:
        logger.error("PDF generation not supported or failed: %s", rexc)
        return _error("PDF_UNAVAILABLE", str(rexc), 501)
    except Exception as exc:
        logger.error("Failed to generate PDF: %s", exc, exc_info=True)
        return _error("PDF_ERROR", "Failed to generate PDF.", 500)

@app.errorhandler(404)
def not_found(error: Any) -> Any:
    return _error("NOT_FOUND", "The requested endpoint does not exist.", 404)


@app.errorhandler(405)
def method_not_allowed(error: Any) -> Any:
    return _error("METHOD_NOT_ALLOWED", "HTTP method not allowed on this endpoint.", 405)


@app.errorhandler(500)
def internal_server_error(error: Any) -> Any:
    logger.error("Unhandled 500: %s", error)
    return _error("INTERNAL_ERROR", "Internal server error.", 500)


# ============================================================================
# ENTRY POINT
# ============================================================================


    logger.info("Starting PhishDec API server...")
# ============================================================================
# GET /admin/system
# ============================================================================
@app.route("/admin/system", methods=["GET", "OPTIONS"])
@roles_required("super_admin", "admin")
def admin_system_status() -> Any:
    """Return detailed system and agent health for the Admin Panel."""
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    try:
        orch = _get_orchestrator()
        hc = orch.health_check()
        st = orch.get_status()
        
        repo = _get_repository()
        import sqlite3
        db_path = getattr(repo, "db_path", None) or "data/phishdec.db"
        investigations = 0
        try:
            with sqlite3.connect(db_path) as c:
                row = c.execute("SELECT COUNT(*) FROM analyses").fetchone()
                investigations = row[0] if row else 0
        except Exception:
            investigations = 0

        agent_list = []
        for name, agent_dict in orch.agents.items():
            agent_instance = agent_dict.get("instance", agent_dict) if isinstance(agent_dict, dict) else agent_dict
            
            agent_status = "Available"
            if hasattr(agent_instance, "model_status"): agent_status = agent_instance.model_status
            elif hasattr(agent_instance, "status"): agent_status = agent_instance.status
            
            features = 0
            if hasattr(agent_instance, "predictor") and agent_instance.predictor:
                pred = agent_instance.predictor
                if hasattr(pred, "get_feature_count") and callable(pred.get_feature_count):
                    features = pred.get_feature_count()
                elif hasattr(pred, "feature_count"):
                    features = pred.feature_count
                elif hasattr(pred, "feature_schema"):
                    features = len(pred.feature_schema)
            if features == 0 and hasattr(agent_instance, "canonical_features") and getattr(agent_instance, "canonical_features", 0) > 0:
                features = getattr(agent_instance, "canonical_features")
            if features == 0 and hasattr(agent_instance, "feature_manager") and hasattr(agent_instance.feature_manager, "features"):
                features = len(agent_instance.feature_manager.features)
                
            agent_name_lower = name.lower()
            
            # Hard fallback for feature counts if reflection fails
            if features == 0:
                if "dns" in agent_name_lower: features = 35
                elif "ssl" in agent_name_lower: features = 54
                elif "html" in agent_name_lower: features = 50
                elif "url" in agent_name_lower: features = 24
                elif "visual" in agent_name_lower: features = 12
                elif "threat" in agent_name_lower: features = 20

            details = {
                "algorithm": "Unknown",
                "purpose": "General Analysis",
                "latency_profile": "Standard (< 200ms)",
                "data_sources": "Internal Vectors",
                "dependencies": "Core Orchestrator"
            }

            if "url" in agent_name_lower:
                prediction_source = "XGBoost"
                details = {
                    "algorithm": "XGBoost Classifier (Tree Booster)",
                    "purpose": "Analyzes lexical properties, entropy, and domain obfuscation.",
                    "latency_profile": "Ultra-fast (< 50ms)",
                    "data_sources": "Lexical extraction, regex parsing",
                    "dependencies": "URLFeatureSchema"
                }
            elif "html" in agent_name_lower:
                prediction_source = "Random Forest"
                details = {
                    "algorithm": "Random Forest Ensemble",
                    "purpose": "Evaluates DOM structure, hidden iframes, and script density.",
                    "latency_profile": "Fast (< 150ms)",
                    "data_sources": "HTML DOM Parser, BeautifulSoup",
                    "dependencies": "HTMLFeatureSchema"
                }
            elif "ssl" in agent_name_lower:
                prediction_source = "Random Forest"
                details = {
                    "algorithm": "Random Forest Classifier",
                    "purpose": "Checks certificate validity, issuer trust, and SAN configuration.",
                    "latency_profile": "Variable (Network Dependent)",
                    "data_sources": "OpenSSL, TLS Handshake",
                    "dependencies": "Network Outbound (Port 443)"
                }
            elif "dns" in agent_name_lower:
                prediction_source = "Random Forest"
                details = {
                    "algorithm": "Random Forest Classifier",
                    "purpose": "Assesses A, MX, TXT records and detects fast-flux behavior.",
                    "latency_profile": "Variable (Network Dependent)",
                    "data_sources": "DNS Queries, WHOIS lookups",
                    "dependencies": "DNS Resolvers"
                }
            elif "visual" in agent_name_lower:
                prediction_source = "ResNet50"
                details = {
                    "algorithm": "ResNet50 Convolutional Neural Network",
                    "purpose": "Detects brand logos, layout spoofing, and credential forms.",
                    "latency_profile": "Intensive (< 800ms)",
                    "data_sources": "Rendered DOM Screenshots",
                    "dependencies": "PyTorch, Torchvision"
                }
            elif "threat" in agent_name_lower:
                prediction_source = "External APIs"
                details = {
                    "algorithm": "Heuristic Aggregation",
                    "purpose": "Cross-references indicators against global threat feeds.",
                    "latency_profile": "Variable (< 1000ms)",
                    "data_sources": "VirusTotal, OpenPhish, PhishTank",
                    "dependencies": "API Keys, Rate Limiting"
                }
            else:
                prediction_source = "Unknown Model"
                
            if "fallback" in agent_status.lower() or agent_status == "no_model_fallback":
                prediction_source += " (Fallback)"

            agent_list.append({
                "name": name.replace("_Agent", "").replace("_", " "),
                "health": "Healthy",
                "model": agent_status.replace("_", " ").title(),
                "features": features,
                "version": getattr(agent_instance, "version", "2.0.0"),
                "prediction_source": prediction_source,
                "last_check": "Just now",
                "details": details
            })

        return jsonify({
            "system": {"status": "healthy" if hc.get("healthy") else "degraded"},
            "api": {"status": "online"},
            "database": {"status": "connected", "engine": "SQLite", "investigations": investigations, "reports": investigations},
            "agents": {"total": len(agent_list), "available": len([a for a in agent_list if a["health"] == "Healthy"]), "list": agent_list},
            "fusion": {"status": "operational" if hc.get("fusion_ready") else "offline", "active_agents": len(agent_list), "minimum_consensus": 2, "version": st.get("fusion_engine", {}).get("engine_version", "17.0.0")}
        }), 200
    except Exception as exc:
        logger.error("Admin system status failed: %s", exc, exc_info=True)
        return _error("ADMIN_ERROR", str(exc), 500)



@app.route("/api/admin/analytics/export", methods=["GET"])
def export_analytics():
    # Phase 24 Security Check Placeholder
    if request.headers.get("X-User-Role") == "user":
        return jsonify({"error": "Unauthorized"}), 403

    period = request.args.get("period", "30")
    try:
        period_days = int(period)
    except ValueError:
        period_days = 30
        
    try:
        from flask import Response
        csv_data = AnalyticsService().export_analytics_csv(period_days=period_days)
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename=phishdec_analytics_{period_days}d.csv"}
        )
    except Exception as e:
        logger.error(f"Analytics Export Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/analytics", methods=["GET"])
@roles_required("super_admin", "admin")
def get_analytics():
    # Phase 24 Security Check Placeholder
    if request.headers.get("X-User-Role") == "user":
        return jsonify({"error": "Unauthorized"}), 403

    period = request.args.get("period", "30")
    try:
        period_days = int(period)
    except ValueError:
        period_days = 30
        
    filters = {
        "agent": request.args.get("agent"),
        "verdict": request.args.get("verdict"),
        "risk": request.args.get("risk")
    }
    
    try:
        data = AnalyticsService().get_dashboard_analytics(period_days=period_days, filters=filters)
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Analytics API Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
@app.route("/api/admin/monitoring", methods=["GET"])
@roles_required("super_admin", "admin")
def get_monitoring():
    try:
        data = mon_srv = MonitoringService()
        mon_srv.set_orchestrator(_get_orchestrator())
        data = mon_srv.get_system_health()
        
        # Phase 24: Alert Engine System Evaluation
        try:
            AlertEngine(db_path=DB_PATH).evaluate_system_health(data)
        except Exception as alert_exc:
            logger.error(f"Alert Engine system evaluation failed: {alert_exc}")
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Monitoring API Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/notifications", methods=["GET"])
@login_required
def get_notifications():
    user_role = g.current_user.get("role", "viewer")
    status = request.args.get("status")
    try:
        ns = NotificationService(db_path=DB_PATH)
        return jsonify(ns.get_notifications(user_role=user_role, status=status)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/notifications/unread", methods=["GET"])
@login_required
def get_unread_count():
    user_role = g.current_user.get("role", "viewer")
    try:
        ns = NotificationService(db_path=DB_PATH)
        return jsonify({"count": ns.get_unread_count(user_role=user_role)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/notifications/<notif_id>/read", methods=["PATCH"])
@login_required
def mark_read(notif_id):
    user_role = g.current_user.get("role", "viewer")
    try:
        ns = NotificationService(db_path=DB_PATH)
        success = ns.mark_as_read(notif_id, user_role=user_role)
        return jsonify({"success": success}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/notifications/read-all", methods=["PATCH"])
@login_required
def mark_all_read():
    user_role = g.current_user.get("role", "viewer")
    try:
        ns = NotificationService(db_path=DB_PATH)
        count = ns.mark_all_as_read(user_role=user_role)
        return jsonify({"success": True, "count": count}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# DAY 25: AUTHENTICATION & AUTHORIZATION ENDPOINTS
# ============================================================================

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({
            "status": "error",
            "error": "BAD_REQUEST",
            "message": "Username and password are required."
        }), 400

    auth_svc = AuthService(db_path=DB_PATH)
    ip_address = request.remote_addr
    success, token_or_msg, safe_user = auth_svc.authenticate(username, password, ip_address=ip_address)

    if not success:
        return jsonify({
            "status": "error",
            "error": "UNAUTHORIZED",
            "message": token_or_msg
        }), 401

    return jsonify({
        "status": "success",
        "token": token_or_msg,
        "user": safe_user
    }), 200


@app.route("/api/auth/logout", methods=["POST"])
@login_required
def auth_logout():
    token = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    elif auth_header:
        token = auth_header.strip()
    if not token:
        token = request.headers.get("X-Session-Token")

    auth_svc = AuthService(db_path=DB_PATH)
    auth_svc.terminate_session(token, ip_address=request.remote_addr)
    return jsonify({
        "status": "success",
        "message": "Logged out successfully."
    }), 200


@app.route("/api/auth/me", methods=["GET"])
@login_required
def auth_me():
    return jsonify({
        "status": "success",
        "authenticated": True,
        "user": g.current_user
    }), 200


@app.route("/api/auth/change-password", methods=["POST"])
@login_required
def auth_change_password():
    data = request.get_json(silent=True) or {}
    new_password = data.get("new_password", "")
    if not new_password or len(new_password) < 6:
        return jsonify({"status": "error", "message": "Password must be at least 6 characters long."}), 400

    auth_svc = AuthService(db_path=DB_PATH)
    success, msg = auth_svc.reset_password(g.current_user["user_id"], new_password)
    if not success:
        return jsonify({"status": "error", "message": msg}), 400
    return jsonify({"status": "success", "message": msg}), 200


# ============================================================================
# DAY 25: ADMIN USER MANAGEMENT ENDPOINTS
# ============================================================================

@app.route("/api/admin/users", methods=["GET"])
@roles_required("super_admin", "admin")
def admin_get_users():
    auth_svc = AuthService(db_path=DB_PATH)
    users = auth_svc.get_all_users()
    return jsonify({"status": "success", "users": users}), 200


@app.route("/api/admin/users", methods=["POST"])
@roles_required("super_admin", "admin")
def admin_create_user():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")
    role = data.get("role", "analyst").strip().lower()

    auth_svc = AuthService(db_path=DB_PATH)
    success, msg, user = auth_svc.create_user(username, email, password, role, creator_id=g.current_user.get("user_id"))
    if not success:
        return jsonify({"status": "error", "message": msg}), 400
    return jsonify({"status": "success", "message": msg, "user": user}), 201


@app.route("/api/admin/users/<user_id>", methods=["PATCH", "DELETE"])
@roles_required("super_admin", "admin")
def admin_update_user(user_id):
    auth_svc = AuthService(db_path=DB_PATH)
    admin_id = g.current_user.get("user_id")

    if request.method == "DELETE":
        # Enforce super_admin inside the handler since @roles_required allows "admin" for PATCH
        if g.current_user.get("role") != "super_admin":
            return jsonify({"status": "error", "message": "Only Super Admins can permanently delete user accounts."}), 403
            
        success, msg = auth_svc.delete_user(user_id, admin_id=admin_id)
        if not success:
            return jsonify({"status": "error", "message": msg}), 400
        return jsonify({"status": "success", "message": "User permanently deleted."}), 200

    # PATCH logic
    data = request.get_json(silent=True) or {}
    
    if "role" in data:
        success, msg = auth_svc.update_user_role(user_id, data["role"], admin_id=admin_id)
        if not success:
            return jsonify({"status": "error", "message": msg}), 400

    if "status" in data:
        success, msg = auth_svc.toggle_user_status(user_id, data["status"], admin_id=admin_id)
        if not success:
            return jsonify({"status": "error", "message": msg}), 400

    return jsonify({"status": "success", "message": "User updated successfully."}), 200



@app.route("/api/admin/users/<user_id>/reset-password", methods=["POST"])
@roles_required("super_admin", "admin")
def admin_reset_user_password(user_id):
    data = request.get_json(silent=True) or {}
    new_password = data.get("new_password", "")
    if not new_password or len(new_password) < 6:
        return jsonify({"status": "error", "message": "Password must be at least 6 characters long."}), 400

    auth_svc = AuthService(db_path=DB_PATH)
    success, msg = auth_svc.reset_password(user_id, new_password, admin_id=g.current_user.get("user_id"))
    if not success:
        return jsonify({"status": "error", "message": msg}), 400
    return jsonify({"status": "success", "message": msg}), 200


@app.route("/api/admin/audit-logs", methods=["GET"])
@roles_required("super_admin")
def admin_audit_logs():
    limit = int(request.args.get("limit", 50))
    auth_svc = AuthService(db_path=DB_PATH)
    logs = auth_svc.get_audit_logs(limit=limit)
    return jsonify({"status": "success", "audit_logs": logs}), 200


if __name__ == "__main__":

    logger.info("Starting PhishDec API server...")
    app.run(debug=True, port=5050)

