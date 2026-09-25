"""
AnalysisResult
==============

Day 15 — Multi-Agent AI Cybersecurity Analyst

Defines the standardized final output contract consumed by the UI and API.

The UI never needs to know how individual agents work.
It sends one URL and receives one AnalysisResult.

Structure:
    AnalysisResult
    ├── status              "success" | "limited_evidence" | "error"
    ├── analysis_id         Unique per-analysis ID (e.g. "PHISH-20260916-A7F31")
    ├── timestamp           ISO-8601
    ├── target              url / domain / ip
    ├── final_assessment    verdict / risk_score / risk_level / confidence
    ├── consensus           agent coverage and agreement
    ├── agents              per-agent status summary
    └── explanation         risk_factors / evidence / summary
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ============================================================================
# ANALYSIS ID GENERATOR
# ============================================================================


# ============================================================================
# ANALYSIS ID GENERATOR
# ============================================================================

def _generate_analysis_id() -> str:
    """
    Generate a unique analysis ID.

    Format:  PHISH-YYYYMMDD-XXXXXXXX
    Example: PHISH-20260916-A7F31C2B
    """
    date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
    uid_part = uuid.uuid4().hex[:8].upper()
    return f"PHISH-{date_part}-{uid_part}"


# ============================================================================
# DOMAIN EXTRACTION HELPER
# ============================================================================

def _extract_domain(url: str) -> str:
    """Extract domain from URL string."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc or url
    except Exception:
        return url


# ============================================================================
# AGENT NAME → DISPLAY KEY MAPPING
# ============================================================================

_AGENT_KEY_MAP: Dict[str, str] = {
    "URL_AI_Agent":        "url",
    "HTML_AI_Agent":       "html",
    "SSL_AI_Agent":        "ssl",
    "DNS_AI_Agent":        "dns",
    "Visual_AI_Agent":     "visual",
    "Threat_Intel_Agent":  "threat_intelligence",
}


SIGNAL_FIELD_SEMANTICS: Dict[str, str] = {
    "prediction": (
        "Classifier label for this agent: legitimate, phishing, "
        "suspicious, or unknown. It is independent of risk_score."
    ),
    "confidence": (
        "Model certainty in the predicted class, in [0, 1]. "
        "This is not phishing probability and not a risk score. "
        "0 means confidence was unavailable."
    ),
    "risk_score": (
        "Operational security-risk severity on 0-100, derived from "
        "model probability plus agent-specific evidence. A legitimate "
        "prediction may still have an elevated risk_score when "
        "infrastructure evidence is concerning."
    ),
    "risk_level": (
        "Bucket of risk_score: low (0-29), medium (30-59), "
        "high (60-79), critical (80-100)."
    ),
}


# ============================================================================
# ANALYSIS RESULT
# ============================================================================

class AnalysisResult:
    """
    Standardized final output for the PhishDec multi-agent analysis pipeline.

    Build from orchestrator output:
        result = AnalysisResult.from_orchestrator_output(orch_result, url)

    Serialize to dict:
        result.to_dict()
    """

    def __init__(
        self,
        *,
        analysis_id: str,
        timestamp: str,
        status: str,
        target_url: str,
        target_domain: str,
        target_ip: Optional[str],
        verdict: str,
        risk_score: Optional[float],
        risk_level: str,
        confidence: float,
        usable_agent_count: int,
        total_agent_count: int,
        coverage: float,
        consensus_available: bool,
        consensus_satisfied: bool,
        conflict_detected: bool,
        evidence_state: str,
        limited_evidence: bool,
        agents: Dict[str, Any],
        risk_factors: List[Any],
        evidence: List[Any],
        explanation_summary: str,
        decision_basis: Dict[str, Any],
        execution_time_ms: Optional[float],
    ) -> None:

        self.analysis_id       = analysis_id
        self.timestamp         = timestamp
        self.status            = status

        self.target_url        = target_url
        self.target_domain     = target_domain
        self.target_ip         = target_ip

        self.verdict           = verdict
        self.risk_score        = risk_score
        self.risk_level        = risk_level
        self.confidence        = confidence

        self.usable_agent_count  = usable_agent_count
        self.total_agent_count   = total_agent_count
        self.coverage            = coverage
        self.consensus_available = consensus_available
        self.consensus_satisfied = consensus_satisfied
        self.conflict_detected   = conflict_detected
        self.evidence_state      = evidence_state
        self.limited_evidence    = limited_evidence

        self.agents              = agents

        self.risk_factors        = risk_factors
        self.evidence            = evidence
        self.explanation_summary = explanation_summary
        self.decision_basis      = decision_basis

        self.execution_time_ms   = execution_time_ms

    # =========================================================================
    # FACTORY
    # =========================================================================

    @classmethod
    def from_orchestrator_output(
        cls,
        orch: Dict[str, Any],
        target_url: str = "",
    ) -> "AnalysisResult":
        """
        Build an AnalysisResult from MultiAgentOrchestrator.analyze() output.
        """

        # Resolve target URL
        url = (
            target_url
            or orch.get("url")
            or ""
        )
        domain = _extract_domain(url)

        # Core decision fields
        verdict           = str(orch.get("final_verdict") or "unknown")
        risk_score_raw    = orch.get("final_risk_score")
        risk_score        = float(risk_score_raw) if risk_score_raw is not None else None
        risk_level        = str(orch.get("risk_level") or "unknown").lower()
        # Ensure risk_level strictly adheres to the 4 documented operational severity buckets:
        # low (0-29), medium (30-59), high (60-79), critical (80-100)
        if risk_score is not None:
            if risk_score >= 80.0:
                risk_level = "critical"
            elif risk_score >= 60.0:
                risk_level = "high"
            elif risk_score >= 30.0:
                risk_level = "medium"
            else:
                risk_level = "low"
        elif risk_level == "suspicious":
            risk_level = "medium"
        confidence        = float(orch.get("confidence") or 0.0)

        # Agent counts
        usable_count  = int(orch.get("usable_agent_count") or 0)
        total_count   = 6
        coverage      = round(usable_count / total_count, 4)

        # Consensus & conflict
        consensus_available = bool(orch.get("consensus_available", False))
        conflict_detected   = bool(orch.get("conflict_detected", False))
        evidence_state      = str(orch.get("evidence_state") or "unavailable")
        limited_evidence    = (
            str(orch.get("analysis_status") or "").lower() == "limited_evidence"
            or usable_count < 2
        )
        consensus_satisfied = (
            usable_count >= 2
            and verdict not in {"unknown", "error"}
        )

        # API-facing status
        analysis_status = str(orch.get("analysis_status") or "unknown").lower()
        if analysis_status in {"complete", "success", "critical"} or (usable_count >= 2 and verdict not in {"unknown", "error"}):
            status = "success"
        elif analysis_status == "limited_evidence" or limited_evidence:
            status = "limited_evidence"
        else:
            status = "error"

        # Per-agent summary
        fusion_result = orch.get("fusion_result") or orch.get("fusion") or {}
        calibrated_decisions = (
            orch.get("calibrated_decisions")
            or fusion_result.get("calibrated_decisions")
            or []
        )
        agent_results_raw: List[Dict[str, Any]] = (
            orch.get("agent_results")
            or calibrated_decisions
            or []
        )
        calibrations_by_agent = {
            d.get("agent"): d
            for d in calibrated_decisions
            if isinstance(d, dict) and d.get("agent")
        }
        agents: Dict[str, Any] = {}

        for raw in agent_results_raw:
            if not isinstance(raw, dict):
                continue
            internal_name = (
                raw.get("agent_name")
                or raw.get("agent")
                or ""
            )
            display_key = _AGENT_KEY_MAP.get(internal_name, internal_name.lower())

            c_info = calibrations_by_agent.get(internal_name) or {}
            c_meta = c_info.get("calibration") or raw.get("calibration") or {}
            raw_rs = c_info.get("raw_risk_score", raw.get("raw_risk_score", raw.get("risk_score")))
            cal_rs = c_info.get("calibrated_risk_score", raw.get("calibrated_risk_score", raw.get("risk_score")))

            agents[display_key] = {
                "agent":          internal_name,
                "status":         str(raw.get("analysis_status") or raw.get("status") or "unavailable"),
                "signal":         bool(raw.get("signal_available", False)),
                "prediction":     str(raw.get("prediction") or raw.get("verdict") or "unknown"),
                "risk_score":     cal_rs,
                "execution_time_ms": raw.get("execution_time_ms"),
                "raw_risk_score": raw_rs,
                "risk_level":     str(raw.get("risk_level") or "unknown"),
                "confidence":     _extract_agent_confidence(raw),
                "reason":         _extract_agent_reason(raw),
                "calibration_curve": c_meta.get("calibration_curve", "identity"),
                "operational_meaning": c_meta.get("operational_meaning", "operational_severity"),
                "risk_score_meaning": "operational_severity",
                "confidence_meaning": "class_certainty",
            }

        # Ensure all 6 slots exist even if agent didn't run
        for display_key in _AGENT_KEY_MAP.values():
            agents.setdefault(display_key, {
                "agent":      "",
                "status":     "unavailable",
                "signal":     False,
                "prediction": "unknown",
                "risk_score": None,
                "raw_risk_score": None,
                "risk_level": "unknown",
                "confidence": 0.0,
                "reason":     None,
                "calibration_curve": "none",
                "operational_meaning": "none",
                "risk_score_meaning": "operational_severity",
                "confidence_meaning": "class_certainty",
            })

        # Explanation
        fusion_result = orch.get("fusion_result") or {}
        db = orch.get("decision_basis") or fusion_result.get("decision_basis") or {}
        risk_factors  = []
        evidence_list = []

        for ar in agent_results_raw:
            if not isinstance(ar, dict):
                continue
            for rf in ar.get("risk_factors") or []:
                risk_factors.append(rf)
            for ev in ar.get("evidence") or []:
                evidence_list.append(ev)

        reasons = db.get("reasons") or []
        explanation_summary = (
            "; ".join(str(r) for r in reasons)
            if reasons
            else _build_summary(verdict, usable_count, conflict_detected, limited_evidence)
        )

        return cls(
            analysis_id         = _generate_analysis_id(),
            timestamp           = datetime.now(timezone.utc).isoformat(),
            status              = status,
            target_url          = url,
            target_domain       = domain,
            target_ip           = None,
            verdict             = verdict,
            risk_score          = risk_score,
            risk_level          = risk_level,
            confidence          = confidence,
            usable_agent_count  = usable_count,
            total_agent_count   = total_count,
            coverage            = coverage,
            consensus_available = consensus_available,
            consensus_satisfied = consensus_satisfied,
            conflict_detected   = conflict_detected,
            evidence_state      = evidence_state,
            limited_evidence    = limited_evidence,
            agents              = agents,
            risk_factors        = risk_factors,
            evidence            = evidence_list,
            explanation_summary = explanation_summary,
            decision_basis      = db,
            execution_time_ms   = orch.get("execution_time_ms"),
        )

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a UI-ready JSON-compatible dict."""

        return {
            "status":       self.status,
            "analysis_id":  self.analysis_id,
            "timestamp":    self.timestamp,

            "target": {
                "url":    self.target_url,
                "domain": self.target_domain,
                "ip":     self.target_ip,
            },

            "final_assessment": {
                "verdict":    self.verdict,
                "risk_score": self.risk_score,
                "risk_level": self.risk_level,
                "confidence": round(self.confidence, 6),
            },

            "consensus": {
                "available_agents":    self.usable_agent_count,
                "total_agents":        self.total_agent_count,
                "coverage":            self.coverage,
                "consensus_available": self.consensus_available,
                "consensus_satisfied": self.consensus_satisfied,
                "conflict_detected":   self.conflict_detected,
                "evidence_state":      self.evidence_state,
                "limited_evidence":    self.limited_evidence,
            },

            "risk_calibration": self.decision_basis.get("risk_calibration", {
                "status": "applied",
                "calibration_standard": "Operational Security Risk Harmonization (OSRH-v2)",
            }),

            "agents": self.agents,

            "signal_semantics": SIGNAL_FIELD_SEMANTICS,

            "explanation": {
                "risk_factors": self.risk_factors,
                "evidence":     self.evidence,
                "summary":      self.explanation_summary,
            },

            "meta": {
                "execution_time_ms": self.execution_time_ms,
                "decision_basis":    self.decision_basis,
            },
        }


# ============================================================================
# AGENT FIELD EXTRACTORS
# ============================================================================

def _as_unit_interval(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if numeric != numeric:
        return None

    if numeric > 1.0 and numeric <= 100.0:
        numeric = numeric / 100.0

    if numeric < 0.0 or numeric > 1.0:
        return None

    return numeric


def _extract_agent_confidence(raw: Dict[str, Any]) -> float:
    """
    Extract class-certainty. 0 means unavailable, not a sure vote.
    """

    for key in ("confidence", "confidence_score", "final_confidence"):
        numeric = _as_unit_interval(raw.get(key))
        if numeric is not None and numeric > 0.0:
            return round(numeric, 6)

    class_probabilities = raw.get("class_probabilities")
    if isinstance(class_probabilities, dict) and class_probabilities:
        values = [
            value
            for value in (
                _as_unit_interval(item)
                for item in class_probabilities.values()
            )
            if value is not None
        ]
        if values:
            return round(max(values), 6)

    for key in ("threat_probability", "phishing_probability", "probability"):
        numeric = _as_unit_interval(raw.get(key))
        if numeric is not None:
            derived = max(numeric, 1.0 - numeric)
            if derived > 0.0:
                return round(derived, 6)

    return 0.0


def _extract_agent_reason(raw: Dict[str, Any]) -> Optional[str]:
    """
    Prefer a string reason, then explanation.summary, then error text.
    """

    candidates = [
        raw.get("reason"),
        raw.get("error"),
        raw.get("error_message"),
    ]

    explanation = raw.get("explanation")
    if isinstance(explanation, str):
        candidates.append(explanation)
    elif isinstance(explanation, dict):
        candidates.append(explanation.get("summary"))
        candidates.append(explanation.get("reason"))

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    return None


# ============================================================================
# SUMMARY BUILDER
# ============================================================================

def _build_summary(
    verdict: str,
    usable_count: int,
    conflict_detected: bool,
    limited_evidence: bool,
) -> str:
    """Build a human-readable summary string from the analysis outcome."""

    if limited_evidence:
        return (
            f"Insufficient evidence — only {usable_count} of 6 agents produced usable signals. "
            "Verdict suppressed; minimum consensus of 2 agents required."
        )
    if usable_count == 0:
        return "No usable signals produced by any agent. Verdict is unknown."
    if conflict_detected:
        return (
            f"Conflict detected across {usable_count} agents. "
            f"Weighted fusion produced verdict: {verdict}."
        )
    return (
        f"Consensus verdict from {usable_count} agents: {verdict}."
    )
