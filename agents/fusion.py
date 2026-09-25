"""
Decision Fusion Engine
======================

Day 14 - Multi-Agent AI Cybersecurity Analyst

Purpose
-------
Combines standardized results from the six active cybersecurity agents:

1. URL_AI_Agent
2. HTML_AI_Agent
3. SSL_AI_Agent
4. DNS_AI_Agent
5. Visual_AI_Agent
6. Threat_Intel_Agent

Design principles
-----------------
- AgentResult-first compatibility
- Six-agent scope only
- Weighted evidence fusion
- Confidence-aware decisions
- Conflict detection
- Threat Intelligence critical override
- DNS critical evidence preservation
- Partial-agent support
- Minimum-consensus protection
- UI-ready output
- Stable application-facing contract

Interpretation rules
--------------------
- Majority voting and weighted fusion are separate signals. The weighted
  result is authoritative when enough agents provide usable evidence.
- Final risk is the weighted average of agent risk scores, not the maximum
  individual score.
- Decisiveness is the distance of the weighted risk from the ambiguous
  midpoint: ``abs(weighted_risk - 50) / 50``.
- Critical indicators contribute to risk, but only explicitly normalized
  critical malicious evidence activates the critical override.

Important Day 14 rule
---------------------
Normal fusion requires at least MINIMUM_AGENTS_FOR_CONSENSUS usable
agents.

If fewer than that are usable, the engine MUST NOT pretend that a
normal consensus decision exists.

Therefore:

    0 usable agents -> unknown / no_signal
    1 usable agent   -> unknown / limited_evidence
    2+ usable       -> normal weighted fusion

Critical malicious evidence is evaluated before the minimum-consensus
check because explicitly configured critical evidence can legitimately
override normal weighted fusion.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


# ============================================================================
# ENGINE CONSTANTS
# ============================================================================

ENGINE_NAME = "Decision Fusion Engine"
ENGINE_VERSION = "17.0.0"


# ============================================================================
# ACTIVE AGENTS
# ============================================================================

URL_AGENT_NAME = "URL_AI_Agent"
HTML_AGENT_NAME = "HTML_AI_Agent"
SSL_AGENT_NAME = "SSL_AI_Agent"
DNS_AGENT_NAME = "DNS_AI_Agent"
VISUAL_AGENT_NAME = "Visual_AI_Agent"
THREAT_INTEL_AGENT_NAME = "Threat_Intel_Agent"


ACTIVE_AGENTS = {
    URL_AGENT_NAME,
    HTML_AGENT_NAME,
    SSL_AGENT_NAME,
    DNS_AGENT_NAME,
    VISUAL_AGENT_NAME,
    THREAT_INTEL_AGENT_NAME,
}


# ============================================================================
# AGENT WEIGHTS
# ============================================================================

AGENT_WEIGHTS = {
    URL_AGENT_NAME: 0.15,
    HTML_AGENT_NAME: 0.15,
    SSL_AGENT_NAME: 0.05,
    DNS_AGENT_NAME: 0.05,
    VISUAL_AGENT_NAME: 0.15,
    THREAT_INTEL_AGENT_NAME: 0.15,

    # Intentionally inactive.
    "Reputation_AI_Agent": 0.0,
    "Malware_AI_Agent": 0.0,
}


# ============================================================================
# FEATURE MAPPING
# ============================================================================

AGENT_FEATURE_MAPPING = {
    URL_AGENT_NAME: "url_features",
    HTML_AGENT_NAME: "html_features",
    SSL_AGENT_NAME: "ssl_features",
    DNS_AGENT_NAME: "dns_features",
    VISUAL_AGENT_NAME: "visual_features",
    THREAT_INTEL_AGENT_NAME: "threat_features",

    "Reputation_AI_Agent": "reputation_features",
    "Malware_AI_Agent": "malware_features",
}


# ============================================================================
# HEURISTIC RELIABILITY
# ============================================================================

HEURISTIC_RELIABILITY = {
    URL_AGENT_NAME: 0.50,
    HTML_AGENT_NAME: 0.50,
    SSL_AGENT_NAME: 0.35,
    DNS_AGENT_NAME: 0.35,
    VISUAL_AGENT_NAME: 0.25,
    THREAT_INTEL_AGENT_NAME: 0.50,
}


# ============================================================================
# DECISION THRESHOLDS
# ============================================================================

MINIMUM_AGENTS_FOR_CONSENSUS = 2

# ── Risk level output labels (risk_level field: operational severity bucket) ──
# These define the displayed risk_level string only. They do NOT affect scoring.
# 80–100 → critical
# 60–79  → high
# 30–59  → medium
#  0–29  → low
RISK_LEVEL_CRITICAL_THRESHOLD = 80.0
HIGH_RISK_THRESHOLD            = 60.0
SUSPICIOUS_THRESHOLD           = 30.0

# ── Verdict boundaries (final verdict string: the classification decision) ────
# 70–100 → phishing
# 40–69  → suspicious
#  0–39  → legitimate
VERDICT_PHISHING_THRESHOLD    = 70.0
VERDICT_SUSPICIOUS_THRESHOLD  = 40.0

# ── Conflict detection thresholds ─────────────────────────────────────────────
# Used ONLY in _detect_conflict() to decide if evidence spans "high" and "low"
# zones simultaneously. These must stay at the original 70/30 values so that
# conflict is flagged only when agents genuinely contradict each other (one
# very high-risk vs. one very low-risk), not on every moderate disagreement.
CONFLICT_HIGH_RISK_FLOOR = 70.0
CONFLICT_LOW_RISK_CEILING = 30.0


# ============================================================================
# CRITICAL EVIDENCE POLICY
# ============================================================================

CRITICAL_ML_PHISHING_PROBABILITY = 0.95
STRONG_ML_PHISHING_PROBABILITY = 0.90

CRITICAL_THREAT_SCORE = 90.0
CRITICAL_MALICIOUS_ENGINE_COUNT = 5

DNS_MEDIUM_RISK_FLOOR = 40.0
DNS_HIGH_RISK_FLOOR = 60.0
DNS_CRITICAL_RISK_FLOOR = 70.0


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def _safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    """
    Safely convert a value to float.
    """

    if value is None:
        return default

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(number):
        return default

    return number


def _clamp(
    value: Any,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    """
    Clamp numeric value to a range.
    """

    number = _safe_float(value, minimum)

    if number is None:
        number = minimum

    return max(
        minimum,
        min(
            maximum,
            number,
        ),
    )


def _clamp_probability(
    value: Any,
) -> Optional[float]:
    """
    Clamp probability to [0, 1].
    """

    number = _safe_float(value)

    if number is None:
        return None

    return max(
        0.0,
        min(
            1.0,
            number,
        ),
    )


def _round_or_none(
    value: Any,
    digits: int = 6,
) -> Optional[float]:
    """
    Round numeric values while preserving None.
    """

    number = _safe_float(value)

    if number is None:
        return None

    return round(
        number,
        digits,
    )


def _safe_copy(
    value: Any,
) -> Any:
    """
    Safely deep-copy evidence structures.
    """

    try:
        return copy.deepcopy(value)
    except Exception:
        return value


# ============================================================================
# OBJECT -> DICTIONARY NORMALIZATION
# ============================================================================

def _object_to_dict(
    value: Any,
) -> Dict[str, Any]:
    """
    Convert dictionaries, dataclasses and AgentResult-like objects into
    dictionaries.

    This keeps the fusion engine compatible with the Day 13 AgentResult
    implementation without requiring the fusion engine to import AgentResult
    directly.
    """

    if isinstance(value, Mapping):
        return dict(value)

    if is_dataclass(value):
        try:
            return asdict(value)
        except Exception:
            pass

    if hasattr(value, "to_dict"):
        try:
            converted = value.to_dict()

            if isinstance(converted, Mapping):
                return dict(converted)
        except Exception:
            pass

    if hasattr(value, "model_dump"):
        try:
            converted = value.model_dump()

            if isinstance(converted, Mapping):
                return dict(converted)
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return dict(vars(value))
        except Exception:
            pass

    return {}


# ============================================================================
# FIELD EXTRACTION
# ============================================================================

def _extract_agent_name(
    decision: Any,
) -> str:
    """
    Extract agent name from a normalized or raw decision.
    """

    if isinstance(decision, Mapping):
        value = (
            decision.get("agent")
            or decision.get("agent_name")
            or decision.get("name")
        )

        return str(
            value
            if value is not None
            else "unknown"
        )

    return "unknown"


def _extract_status(
    decision: Mapping[str, Any],
) -> str:
    """
    Extract normalized status.
    """

    value = decision.get("status")
    if value is None:
        value = decision.get("analysis_status")
    if value is None:
        value = decision.get("model_status")
    if value is None:
        value = "unknown"

    return str(
        value
        if value is not None
        else "unknown"
    ).strip().lower()


def _extract_prediction(
    decision: Mapping[str, Any],
) -> str:
    """
    Extract and normalize prediction.
    """

    value = (
        decision.get("prediction")
        or decision.get("verdict")
        or decision.get("final_verdict")
        or "unknown"
    )

    text = str(value).strip().lower()

    aliases = {
        "malicious": "phishing",
        "malicious_threat": "phishing",
        "phishing_site": "phishing",
        "phishing-url": "phishing",
        "phish": "phishing",

        "safe": "legitimate",
        "benign": "legitimate",
        "clean": "legitimate",
        "clean_reputation": "legitimate",

        "medium": "suspicious",
        "warning": "suspicious",

        "": "unknown",
        "none": "unknown",
        "null": "unknown",
    }

    return aliases.get(
        text,
        text,
    )


def _extract_probability(
    decision: Mapping[str, Any],
) -> Optional[float]:
    """
    Extract phishing probability.
    """

    candidates = [
        decision.get("phishing_probability"),
        decision.get("probability"),
        decision.get("threat_probability"),
        decision.get("final_phishing_probability"),
        decision.get("fused_probability"),
    ]

    for value in candidates:

        probability = _clamp_probability(
            value
        )

        if probability is not None:
            return probability

    return None


def _extract_confidence(
    decision: Mapping[str, Any],
) -> float:
    """
    Extract confidence in [0, 1].

    Confidence is class certainty, not risk_score.
    If the explicit field is missing/zero, recover it from class
    probabilities or phishing probability.
    """

    candidates = [
        decision.get("confidence"),
        decision.get("final_confidence"),
        decision.get("confidence_score"),
    ]

    for value in candidates:

        confidence = _clamp_probability(
            value
        )

        if confidence is not None and confidence > 0.0:
            return confidence

    class_probabilities = decision.get(
        "class_probabilities"
    )

    if isinstance(class_probabilities, Mapping):

        values = []

        for value in class_probabilities.values():

            numeric = _clamp_probability(
                value
            )

            if numeric is not None:
                values.append(numeric)

        if values:
            return max(values)

    probability = _extract_probability(
        decision
    )

    if probability is not None:
        return max(
            probability,
            1.0 - probability,
        )

    return 0.0


def _extract_risk_score(
    decision: Mapping[str, Any],
) -> float:
    """
    Extract risk score in [0, 100].
    """

    candidates = [
        decision.get("risk_score"),
        decision.get("final_risk_score"),
    ]

    for value in candidates:

        score = _safe_float(
            value
        )

        if score is not None:
            return _clamp(
                score
            )

    probability = _extract_probability(
        decision
    )

    if probability is not None:
        return round(
            probability * 100.0,
            6,
        )

    return 0.0


def _extract_risk_level(
    decision: Mapping[str, Any],
    risk_score: Optional[float] = None,
) -> str:
    """
    Extract or derive risk level.
    """

    value = (
        decision.get("risk_level")
        or decision.get("final_risk_level")
    )

    if value is not None:

        text = str(
            value
        ).strip().lower()

        if text not in {
            "",
            "unknown",
            "none",
            "null",
        }:
            return text

    score = (
        _clamp(
            risk_score
            if risk_score is not None
            else _extract_risk_score(decision)
        )
    )

    if score >= RISK_LEVEL_CRITICAL_THRESHOLD:
        return "critical"

    if score >= HIGH_RISK_THRESHOLD:
        return "high"

    if score >= SUSPICIOUS_THRESHOLD:
        return "medium"

    return "low"


# ============================================================================
# AGENT RISK CALIBRATION LAYER
# ============================================================================

def _calibrate_agent_risk(
    agent: str,
    raw_risk: float,
    prediction: str,
    confidence: float,
    evidence_type: str = "model",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calibrate an agent's raw risk score into a standardized operational
    security severity score on [0, 100].

    Heterogeneous agents derive scores using fundamentally different mechanisms:
    - URL: Lexical probability from XGBoost / Random Forest
    - DNS: XGBoost ML probability + infrastructure indicator floors (TTL, fast flux, SPF)
    - HTML: XGBoost ML probability + DOM form/script heuristic indicators
    - SSL: XGBoost ML probability + X.509 validity/CA heuristic adjustments
    - Visual: Screenshot layout & visual complexity ML probability
    - Threat Intelligence: Deterministic vendor consensus & reputation scoring

    Without calibration, raw scores like 'DNS risk 70' (an infrastructure anomaly on
    a legitimate CDN) and 'URL risk 70' (a 70% direct phishing attack certainty)
    are treated identically, causing severe false-positive or false-negative skew.

    This function harmonizes raw scores onto a single operational scale:
      0-29: Low operational severity
      30-59: Medium operational severity
      60-79: High operational severity
      80-100: Critical operational severity
    """
    raw_score = _clamp(float(raw_risk), 0.0, 100.0)
    pred = (prediction or "").lower().strip()
    conf = _clamp_probability(confidence) or 0.5
    calibrated_score = raw_score
    curve_applied = "identity"
    operational_meaning = "Direct operational severity mapping"

    if agent == DNS_AGENT_NAME:
        if pred in {"legitimate", "benign", "safe"}:
            if raw_score >= 60.0:
                # Infrastructure anomaly (low TTL, fast-flux) under legitimate prediction
                # represents network operational volatility / CDN behavior, not phishing attack.
                # Normalized to Medium operational caution band (30-45).
                calibrated_score = round(30.0 + (raw_score - 60.0) * 0.375, 4)
                calibrated_score = min(45.0, calibrated_score)
                curve_applied = "dns_infrastructure_posture_calibration"
                operational_meaning = "Infrastructure volatility normalized to operational caution under legitimate prediction"
            else:
                calibrated_score = raw_score
                curve_applied = "dns_legitimate_baseline"
                operational_meaning = "Normal legitimate DNS infrastructure posture"
        elif pred in {"phishing", "malicious", "suspicious"}:
            calibrated_score = raw_score
            curve_applied = "dns_malicious_direct_mapping"
            operational_meaning = "Direct malicious DNS infrastructure attack indicator"

    elif agent == VISUAL_AGENT_NAME:
        if pred in {"phishing", "malicious"}:
            if conf < 0.65:
                # Dampen uncorroborated low-confidence visual signals on complex responsive layouts
                damping_factor = 0.50 + 0.50 * conf
                calibrated_score = round(raw_score * damping_factor, 4)
                curve_applied = "visual_complexity_damping"
                operational_meaning = "Visual layout entropy dampening applied for low-to-medium confidence"
            else:
                calibrated_score = raw_score
                curve_applied = "visual_high_confidence_direct_mapping"
                operational_meaning = "High-confidence visual brand/layout phishing indicator"
        else:
            calibrated_score = raw_score
            curve_applied = "visual_legitimate_direct_mapping"
            operational_meaning = "Visual appearance consistent with legitimate layout"

    elif agent == THREAT_INTEL_AGENT_NAME:
        if raw_score >= 70.0:
            calibrated_score = raw_score
            curve_applied = "threat_intel_critical_consensus"
            operational_meaning = "High-authority vendor consensus or global blacklist detection"
        elif raw_score >= 40.0 and conf < 0.50:
            calibrated_score = round(35.0 + (raw_score - 40.0) * 0.60, 4)
            curve_applied = "threat_intel_single_vendor_smoothing"
            operational_meaning = "Single-engine or low-consensus threat signal scaled to medium operational caution"
        else:
            calibrated_score = raw_score
            curve_applied = "threat_intel_standard_mapping"
            operational_meaning = "Reputation and threat intelligence scoring"

    elif agent == URL_AGENT_NAME:
        calibrated_score = raw_score
        curve_applied = "url_lexical_probability_mapping"
        operational_meaning = "Direct lexical attack probability mapped to operational severity"

    elif agent == HTML_AGENT_NAME:
        if pred in {"legitimate", "benign", "safe"} and raw_score > 40.0:
            calibrated_score = round(25.0 + (raw_score - 40.0) * 0.33, 4)
            calibrated_score = min(38.0, calibrated_score)
            curve_applied = "html_credential_portal_normalization"
            operational_meaning = "Legitimate login/credential form complexity normalized"
        else:
            calibrated_score = raw_score
            curve_applied = "html_dom_structure_mapping"
            operational_meaning = "DOM structure and form analysis operational severity"

    elif agent == SSL_AGENT_NAME:
        if pred in {"legitimate", "benign", "safe"} and raw_score > 40.0:
            calibrated_score = round(25.0 + (raw_score - 40.0) * 0.35, 4)
            calibrated_score = min(38.0, calibrated_score)
            curve_applied = "ssl_certificate_lifespan_normalization"
            operational_meaning = "Short-lived / automated certificate lifespan normalized under legitimate CA"
        else:
            calibrated_score = raw_score
            curve_applied = "ssl_cryptographic_mapping"
            operational_meaning = "SSL/TLS cryptographic validation operational severity"

    calibrated_score = round(_clamp(calibrated_score, 0.0, 100.0), 4)

    return calibrated_score, {
        "raw_risk_score": round(raw_score, 4),
        "calibrated_risk_score": calibrated_score,
        "calibration_curve": curve_applied,
        "operational_meaning": operational_meaning,
        "raw_to_calibrated_delta": round(calibrated_score - raw_score, 4),
    }


def _extract_prediction_source(
    decision: Mapping[str, Any],
) -> str:
    """
    Extract prediction source.
    """

    value = (
        decision.get("prediction_source")
        or decision.get("source")
        or "unknown"
    )

    return str(
        value
    ).strip().lower()


def _is_model_based(
    decision: Mapping[str, Any],
) -> bool:
    """
    Determine whether the decision appears to originate from a trained model.
    """

    source = _extract_prediction_source(
        decision
    )

    return (
        source in {
            "trained_ml",
            "xgboost",
            "ml",
            "model",
            "machine_learning",
        }
        or bool(
            decision.get("model_based", False)
        )
    )


# ============================================================================
# SIGNAL VALIDATION
# ============================================================================

def _is_signal_available(
    decision: Mapping[str, Any],
) -> bool:
    """
    Determine whether an agent produced a usable signal.
    """

    status = _extract_status(
        decision
    )

    if status in {
        "failed",
        "error",
        "unavailable",
        "timeout",
        "skipped",
        "unknown",
    }:
        return False

    explicit_signal = decision.get(
        "signal_available"
    )

    if explicit_signal is not None:
        return bool(
            explicit_signal
        )

    prediction = _extract_prediction(
        decision
    )

    if prediction == "unknown":
        return False

    return True


def _is_usable_decision(
    decision: Mapping[str, Any],
) -> bool:
    """
    Determine whether a decision can participate in normal fusion.
    """

    agent = _extract_agent_name(
        decision
    )

    if agent not in ACTIVE_AGENTS:
        return False

    if not _is_signal_available(
        decision
    ):
        return False

    prediction = _extract_prediction(
        decision
    )

    if prediction not in {
        "legitimate",
        "phishing",
        "suspicious",
    }:
        return False

    return True


# ============================================================================
# NORMALIZATION
# ============================================================================

def _normalize_input(
    agent_decisions: Any,
) -> List[Dict[str, Any]]:
    """
    Normalize all supported input forms into dictionaries.

    Supported:
    - list of dictionaries
    - tuple
    - AgentResult objects
    - dataclass instances
    - a single dictionary
    - a single AgentResult-like object
    """

    if agent_decisions is None:
        return []

    if isinstance(
        agent_decisions,
        Mapping,
    ):
        # API callers commonly provide a mapping keyed by agent name:
        # {"DNS_AI_Agent": {...}, "URL_AI_Agent": {...}}. Treat that as a
        # collection, while retaining support for one result dictionary.
        if (
            "agent" not in agent_decisions
            and "agent_name" not in agent_decisions
            and all(
                isinstance(value, Mapping)
                for value in agent_decisions.values()
            )
        ):
            raw_items = []
            for key, value in agent_decisions.items():
                item = dict(value)
                item.setdefault("agent", key)
                raw_items.append(item)
        else:
            raw_items = [
                agent_decisions
            ]

    elif isinstance(
        agent_decisions,
        (list, tuple, set),
    ):
        raw_items = list(
            agent_decisions
        )

    else:
        raw_items = [
            agent_decisions
        ]

    normalized = []

    for raw in raw_items:

        item = _object_to_dict(
            raw
        )

        if not item:
            continue

        normalized.append(
            item
        )

    return normalized


# ============================================================================
# PREPARE DECISIONS
# ============================================================================

def _prepare_decisions(
    decisions: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Prepare usable decisions for fusion.

    This function intentionally performs normalization of fields so all
    downstream calculations use a consistent internal representation.
    """

    prepared = []

    for raw in decisions:

        item = dict(
            raw
        )

        agent = _extract_agent_name(
            item
        )

        if not _is_usable_decision(
            item
        ):
            continue

        probability = _extract_probability(
            item
        )

        risk_score = _extract_risk_score(
            item
        )

        confidence = _extract_confidence(
            item
        )

        prediction = _extract_prediction(
            item
        )

        source = _extract_prediction_source(
            item
        )

        if probability is None:
            probability = (
                risk_score / 100.0
            )

        probability = (
            _clamp_probability(
                probability
            )
            or 0.0
        )

        risk_score = _clamp(
            risk_score
        )

        # If the agent says legitimate but supplies a very low phishing
        # probability, preserve that probability.
        #
        # If the agent says phishing but supplies no useful probability,
        # risk_score provides the fallback.
        if (
            prediction == "phishing"
            and probability <= 0.0
            and risk_score > 0.0
        ):
            probability = (
                risk_score / 100.0
            )

        if (
            prediction == "legitimate"
            and probability >= 1.0
            and risk_score < 50.0
        ):
            probability = (
                risk_score / 100.0
            )

        evidence_type = (
            item.get(
                "evidence_type"
            )
            or (
                "model"
                if _is_model_based(item)
                else "heuristic"
            )
        )

        quality_multiplier = 1.0

        if source in {
            "heuristic",
            "fallback",
        }:
            quality_multiplier = HEURISTIC_RELIABILITY.get(
                agent,
                0.5,
            )

        calibrated_risk, calibration_details = _calibrate_agent_risk(
            agent=agent,
            raw_risk=risk_score,
            prediction=prediction,
            confidence=confidence,
            evidence_type=evidence_type,
            metadata=item,
        )

        prepared_item = {
            **item,

            "agent":
                agent,

            "status":
                _extract_status(
                    item
                ),

            "prediction":
                prediction,

            "probability":
                probability,

            "phishing_probability":
                probability,

            "confidence":
                confidence,

            "raw_risk_score":
                risk_score,

            "calibrated_risk_score":
                calibrated_risk,

            "risk_score":
                calibrated_risk,

            "risk_level":
                _extract_risk_level(
                    item,
                    calibrated_risk,
                ),

            "calibration":
                calibration_details,

            "prediction_source":
                source,

            "model_based":
                _is_model_based(
                    item
                ),

            "evidence_type":
                evidence_type,

            "quality_multiplier":
                quality_multiplier,

            "signal_available":
                True,
        }

        prepared.append(
            prepared_item
        )

    return prepared


# ============================================================================
# THREAT INTELLIGENCE EXTRACTION
# ============================================================================

def _extract_threat_intel(
    decision: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Extract Threat Intelligence evidence from an agent result.
    """

    candidate = (
        decision.get(
            "threat_intelligence"
        )
    )

    if isinstance(
        candidate,
        Mapping,
    ):
        return dict(
            candidate
        )

    # Some agent outputs may expose threat fields at top level.
    threat_fields = {
        "threat_score",
        "overall_threat_score",
        "malicious_engines_count",
        "blacklisted",
        "is_listed",
        "vendors",
    }

    if any(
        field in decision
        for field in threat_fields
    ):

        return {
            field:
                decision.get(field)
            for field
            in threat_fields
            if field in decision
        }

    return None


def _extract_dns_evidence(
    decision: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Extract DNS evidence.
    """

    candidates = [
        decision.get(
            "dns_critical_evidence"
        ),
        decision.get(
            "dns_evidence"
        ),
        decision.get(
            "risk_assessment"
        ),
    ]

    for candidate in candidates:

        if isinstance(
            candidate,
            Mapping,
        ):
            evidence = dict(candidate)
            indicator_summary = evidence.get("indicator_summary")
            if (
                isinstance(indicator_summary, Mapping)
                and "critical_count" not in evidence
                and "critical" in indicator_summary
            ):
                evidence["critical_count"] = indicator_summary["critical"]
            if (
                "critical" not in evidence
                and _extract_agent_name(decision) == DNS_AGENT_NAME
                and evidence.get("critical_count", 0)
            ):
                evidence["critical"] = True
            return evidence

    if _extract_agent_name(
        decision
    ) == DNS_AGENT_NAME:

        fields = {
            "risk_score",
            "risk_level",
            "triggered_indicators",
            "indicator_summary",
            "indicator_adjustment",
            "posture_breakdown",
            "critical_count",
            "high_count",
            "medium_count",
            "critical_indicators",
            "critical",
            "dns_critical",
            "dns_risk_score",
            "dns_risk_level",
        }

        evidence = {
            field:
                decision.get(field)
            for field
            in fields
            if field in decision
        }

        # Older DNS responses expose the critical flag and score with
        # dns_* names. Normalize those aliases for the policy evaluator.
        if "dns_critical" in evidence and "critical" not in evidence:
            evidence["critical"] = evidence["dns_critical"]
        if "dns_risk_score" in evidence and "risk_score" not in evidence:
            evidence["risk_score"] = evidence["dns_risk_score"]
        if "dns_risk_level" in evidence and "risk_level" not in evidence:
            evidence["risk_level"] = evidence["dns_risk_level"]

        indicator_summary = evidence.get("indicator_summary")
        if (
            isinstance(indicator_summary, Mapping)
            and "critical_count" not in evidence
            and "critical" in indicator_summary
        ):
            evidence["critical_count"] = indicator_summary["critical"]

        return evidence

    return None


# ============================================================================
# CRITICAL EVIDENCE EVALUATION
# ============================================================================

def _evaluate_critical_evidence(
    usable_decisions: List[Dict[str, Any]],
    threat_intel: Optional[Dict[str, Any]],
    dns_evidence: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate explicitly configured critical malicious evidence.
    """

    reasons = []
    sources = []

    threat_intel_critical = False
    dns_critical = False

    # ------------------------------------------------------------------------
    # Threat Intelligence
    # ------------------------------------------------------------------------

    if threat_intel is not None:

        threat_score = _safe_float(
            threat_intel.get(
                "overall_threat_score",
                threat_intel.get(
                    "threat_score"
                ),
            ),
            0.0,
        ) or 0.0

        malicious_engines = int(
            _safe_float(
                threat_intel.get(
                    "malicious_engines_count",
                    0,
                ),
                0.0,
            )
            or 0
        )

        blacklisted = bool(
            threat_intel.get(
                "blacklisted",
                threat_intel.get(
                    "is_listed",
                    False,
                ),
            )
        )

        if blacklisted:

            threat_intel_critical = True

            reasons.append(
                "Threat Intelligence reports the target as blacklisted."
            )

        if (
            threat_score
            >= CRITICAL_THREAT_SCORE
        ):

            threat_intel_critical = True

            reasons.append(
                "Threat Intelligence produced a critical threat score "
                f"({threat_score:.1f})."
            )

        if (
            malicious_engines
            >= CRITICAL_MALICIOUS_ENGINE_COUNT
        ):

            threat_intel_critical = True

            reasons.append(
                "Threat Intelligence reported "
                f"{malicious_engines} malicious security engines."
            )

        if threat_intel_critical:

            sources.append(
                THREAT_INTEL_AGENT_NAME
            )

    # ------------------------------------------------------------------------
    # DNS
    # ------------------------------------------------------------------------

    if dns_evidence is not None:

        critical_count = int(
            _safe_float(
                dns_evidence.get(
                    "critical_count",
                    0,
                ),
                0.0,
            )
            or 0
        )

        explicit_critical = bool(
            dns_evidence.get(
                "critical",
                False,
            )
        )

        dns_pred = str(
            dns_evidence.get(
                "prediction",
                dns_evidence.get("verdict", ""),
            )
        ).lower()

        if (
            explicit_critical
            or critical_count > 0
        ):

            dns_critical = True

            sources.append(
                DNS_AGENT_NAME
            )

            reasons.append(
                "DNS analysis produced critical malicious evidence."
            )

    critical = (
        threat_intel_critical
        or dns_critical
    )

    if critical:

        reasons.append(
            "Critical override takes precedence over normal weighted "
            "fusion because high-confidence malicious evidence was detected."
        )

    return {
        "critical":
            critical,

        "critical_sources":
            list(
                dict.fromkeys(
                    sources
                )
            ),

        "reasons":
            reasons,

        "threat_intel_critical":
            threat_intel_critical,

        "dns_critical":
            dns_critical,
    }


# ============================================================================
# CONSENSUS
# ============================================================================

def _calculate_consensus(
    usable_decisions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate categorical agent consensus.
    """

    verdicts = [
        _extract_prediction(
            item
        )
        for item
        in usable_decisions
    ]

    verdicts = [
        verdict
        for verdict
        in verdicts
        if verdict
        in {
            "legitimate",
            "phishing",
            "suspicious",
        }
    ]

    total_votes = len(
        verdicts
    )

    if not verdicts:

        return {
            "majority_agent_verdict":
                "unknown",

            "majority_vote_count":
                0,

            "total_votes":
                0,

            "agreement_ratio":
                0.0,

            "vote_counts":
                {},
        }

    counts = Counter(
        verdicts
    )

    top_verdict, top_count = (
        counts.most_common(1)[0]
    )

    agreement_ratio = (
        top_count
        / total_votes
        if total_votes
        else 0.0
    )

    tied_verdicts = sorted(
        verdict
        for verdict, count in counts.items()
        if count == top_count
    )
    is_tie = len(tied_verdicts) > 1
    consensus_verdict = "tie" if is_tie else top_verdict

    return {
        "majority_agent_verdict":
            consensus_verdict,

        "majority_vote_count":
            top_count,

        "plurality_agent_verdict":
            top_verdict,

        "tied_verdicts":
            tied_verdicts,

        "is_tie":
            is_tie,

        "total_votes":
            total_votes,

        "agreement_ratio":
            agreement_ratio,

        "vote_counts":
            dict(
                counts
            ),
    }


# ============================================================================
# CONFLICT DETECTION
# ============================================================================

def _detect_conflict(
    usable_decisions: List[Dict[str, Any]],
    consensus: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Detect categorical and risk-level disagreement.
    """

    if consensus is None:
        consensus = _calculate_consensus(
            usable_decisions
        )

    verdicts = [
        _extract_prediction(
            item
        )
        for item
        in usable_decisions
    ]

    verdicts = [
        value
        for value
        in verdicts
        if value
        in {
            "legitimate",
            "phishing",
            "suspicious",
        }
    ]

    unique_verdicts = set(
        verdicts
    )

    conflict_reasons = []

    if len(
        unique_verdicts
    ) > 1:

        conflict_reasons.append(
            "Agents produced different verdict categories."
        )

    risk_scores = [
        item.get(
            "calibrated_risk_score",
            _extract_risk_score(item),
        )
        for item
        in usable_decisions
    ]

    # Use dedicated conflict thresholds (70/30), NOT the risk_level bucket
    # thresholds (60/30), so risk-level-label changes don't affect conflict logic.
    high_risk = any(
        score >= CONFLICT_HIGH_RISK_FLOOR
        for score
        in risk_scores
    )

    low_risk = any(
        score < CONFLICT_LOW_RISK_CEILING
        for score
        in risk_scores
    )

    if high_risk and low_risk:

        conflict_reasons.append(
            "High-risk and low-risk agent evidence coexist."
        )

    conflict_detected = bool(
        conflict_reasons
    )

    if not conflict_detected:

        conflict_level = "none"

    elif len(
        unique_verdicts
    ) >= 3:

        conflict_level = "high"

    else:

        conflict_level = "medium"

    conflicting_agents = []

    if conflict_detected:

        majority_verdict = consensus.get(
            "majority_agent_verdict",
            "unknown",
        )

        for item in usable_decisions:

            prediction = _extract_prediction(
                item
            )

            if prediction != majority_verdict:

                conflicting_agents.append(
                    _extract_agent_name(
                        item
                    )
                )

    return {
        "conflict_detected":
            conflict_detected,

        "conflict_level":
            conflict_level,

        "conflicting_agents":
            conflicting_agents,

        "conflict_reasons":
            conflict_reasons,
    }


# ============================================================================
# WEIGHT CALCULATION
# ============================================================================

def _calculate_weights(
    usable_decisions: List[Dict[str, Any]],
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    float,
]:
    """
    Calculate effective and normalized weights.

    Quality multipliers are used for heuristic/fallback evidence, while the
    configured agent weight remains visible separately.
    """

    effective_weights = {}

    for item in usable_decisions:

        agent = item[
            "agent"
        ]

        configured_weight = _safe_float(
            AGENT_WEIGHTS.get(
                agent,
                0.0,
            ),
            0.0,
        ) or 0.0

        quality_multiplier = _safe_float(
            item.get(
                "quality_multiplier",
                1.0,
            ),
            1.0,
        ) or 1.0

        quality_multiplier = max(
            0.0,
            min(
                1.0,
                quality_multiplier,
            ),
        )

        effective_weights[
            agent
        ] = (
            configured_weight
            * quality_multiplier
        )

    total_effective_weight = sum(
        effective_weights.values()
    )

    if total_effective_weight <= 0:

        fallback_weight = (
            1.0
            / len(usable_decisions)
            if usable_decisions
            else 0.0
        )

        normalized_weights = {
            item["agent"]:
                fallback_weight
            for item
            in usable_decisions
        }

    else:

        normalized_weights = {
            agent:
                weight
                / total_effective_weight
            for agent, weight
            in effective_weights.items()
        }

    return (
        normalized_weights,
        effective_weights,
        total_effective_weight,
    )


# ============================================================================
# WEIGHTED RISK
# ============================================================================

def _calculate_weighted_risk(
    usable_decisions: List[Dict[str, Any]],
    normalized_weights: Dict[str, float],
) -> float:
    """
    Calculate weighted phishing risk score.
    """

    total = 0.0

    for item in usable_decisions:

        agent = item[
            "agent"
        ]

        weight = normalized_weights.get(
            agent,
            0.0,
        )

        risk_score = _clamp(
            item.get(
                "risk_score",
                0.0,
            )
        )

        total += (
            risk_score
            * weight
        )

    return round(
        _clamp(
            total
        ),
        4,
    )


# ============================================================================
# WEIGHTED PROBABILITY
# ============================================================================

def _calculate_weighted_probability(
    usable_decisions: List[Dict[str, Any]],
    normalized_weights: Dict[str, float],
) -> float:
    """
    Calculate weighted phishing probability.
    """

    total = 0.0
    weight = 0.0

    for item in usable_decisions:

        agent = item[
            "agent"
        ]

        item_weight = normalized_weights.get(
            agent,
            0.0,
        )

        probability = _extract_probability(
            item
        )

        if probability is None:

            probability = (
                _extract_risk_score(
                    item
                )
                / 100.0
            )

        total += (
            probability
            * item_weight
        )

        weight += item_weight

    if weight <= 0:
        return 0.0

    return (
        _clamp_probability(
            total / weight
        )
        or 0.0
    )


# ============================================================================
# AGENT CONTRIBUTIONS
# ============================================================================

def _build_contributions(
    usable_decisions: List[Dict[str, Any]],
    normalized_weights: Dict[str, float],
    effective_weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Build UI-friendly agent contribution information.
    """

    contributions = []

    for item in usable_decisions:

        agent = item[
            "agent"
        ]

        normalized_weight = normalized_weights.get(
            agent,
            0.0,
        )

        risk_score = _clamp(
            item.get(
                "risk_score",
                0.0,
            )
        )

        probability = _extract_probability(
            item
        )

        if probability is None:
            probability = (
                risk_score
                / 100.0
            )

        risk_contribution = (
            risk_score
            * normalized_weight
        )

        contributions.append(
            {
                "agent":
                    agent,

                "configured_weight":
                    round(
                        AGENT_WEIGHTS.get(
                            agent,
                            0.0,
                        ),
                        6,
                    ),

                "effective_weight":
                    round(
                        effective_weights.get(
                            agent,
                            0.0,
                        ),
                        6,
                    ),

                "normalized_weight":
                    round(
                        normalized_weight,
                        6,
                    ),

                "phishing_probability":
                    round(
                        probability,
                        6,
                    ),

                "prediction":
                    item.get(
                        "prediction",
                        "unknown",
                    ),

                "confidence":
                    round(
                        _extract_confidence(
                            item
                        ),
                        6,
                    ),

                "quality_multiplier":
                    round(
                        item.get(
                            "quality_multiplier",
                            1.0,
                        ),
                        6,
                    ),

                "evidence_type":
                    item.get(
                        "evidence_type",
                        "unspecified",
                    ),

                "model_based":
                    item.get(
                        "model_based",
                        False,
                    ),

                "raw_risk_score":
                    item.get(
                        "raw_risk_score",
                        risk_score,
                    ),

                "calibrated_risk_score":
                    risk_score,

                "calibration":
                    item.get(
                        "calibration",
                        {},
                    ),

                "risk_score":
                    risk_score,

                "risk_contribution":
                    round(
                        risk_contribution,
                        4,
                    ),

                "weighted_contribution":
                    round(
                        risk_contribution,
                        4,
                    ),
            }
        )

    return contributions


# ============================================================================
# CONFIDENCE
# ============================================================================

def _calculate_confidence(
    usable_decisions: List[Dict[str, Any]],
    consensus: Dict[str, Any],
    conflict: Dict[str, Any],
    coverage: float,
    decisiveness: float,
) -> Dict[str, float]:
    """
    Calculate confidence from:

    - agent confidence
    - agreement
    - coverage
    - decisiveness
    - conflict penalty
    """

    if not usable_decisions:

        return {
            "agent_confidence":
                0.0,

            "agreement":
                0.0,

            "coverage":
                coverage,

            "decisiveness":
                0.0,

            "conflict_penalty":
                0.0,
        }

    agent_confidence = sum(
        _extract_confidence(
            item
        )
        for item
        in usable_decisions
    ) / len(
        usable_decisions
    )

    agreement = _clamp_probability(
        consensus.get(
            "agreement_ratio",
            0.0,
        )
    ) or 0.0

    conflict_penalty = (
        0.15
        if conflict.get(
            "conflict_detected",
            False,
        )
        else 0.0
    )

    return {
        "agent_confidence":
            agent_confidence,

        "agreement":
            agreement,

        "coverage":
            coverage,

        "decisiveness":
            decisiveness,

        "conflict_penalty":
            conflict_penalty,
    }


def _combine_confidence(
    confidence_model: Dict[str, float],
) -> float:
    """
    Combine confidence components into a stable [0,1] score.
    """

    agent_confidence = confidence_model.get(
        "agent_confidence",
        0.0,
    )

    agreement = confidence_model.get(
        "agreement",
        0.0,
    )

    coverage = confidence_model.get(
        "coverage",
        0.0,
    )

    decisiveness = confidence_model.get(
        "decisiveness",
        0.0,
    )

    conflict_penalty = confidence_model.get(
        "conflict_penalty",
        0.0,
    )

    score = (
        0.35 * agent_confidence
        + 0.25 * agreement
        + 0.15 * coverage
        + 0.25 * decisiveness
        - conflict_penalty * 0.20
    )

    return round(
        _clamp(
            score,
            0.0,
            1.0,
        ),
        6,
    )


# ============================================================================
# RISK LEVEL
# ============================================================================

def _risk_level_from_score(
    risk_score: float,
) -> str:
    """
    Convert a 0-100 risk score into the four documented risk level buckets.

    Contract:
        80-100  → critical
        60-79   → high
        30-59   → medium
        0-29    → low

    This is intentionally separate from the final verdict
    (legitimate / suspicious / phishing).
    """

    score = _clamp(
        risk_score
    )

    if score >= RISK_LEVEL_CRITICAL_THRESHOLD:
        return "critical"

    if score >= HIGH_RISK_THRESHOLD:
        return "high"

    if score >= SUSPICIOUS_THRESHOLD:
        return "medium"

    return "low"


# ============================================================================
# FINAL VERDICT FROM RISK
# ============================================================================

def _verdict_from_risk(
    risk_score: float,
) -> str:
    """
    Convert weighted risk into the final system verdict.

    Contract:
        70-100  → phishing
        40-69   → suspicious
        0-39    → legitimate

    This is intentionally separate from risk_level.
    verdict  = the classification decision (legitimate / suspicious / phishing)
    risk_level = the operational severity bucket (low / medium / high / critical)
    """

    score = _clamp(
        risk_score
    )

    if score >= VERDICT_PHISHING_THRESHOLD:
        return "phishing"

    if score >= VERDICT_SUSPICIOUS_THRESHOLD:
        return "suspicious"

    return "legitimate"


# ============================================================================
# DECISION BASIS
# ============================================================================

def _build_decision_summary(
    final_verdict: str,
    consensus: Dict[str, Any],
    conflict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a clear explanation separating categorical majority from weighted
    fusion.

    This avoids ambiguous output such as:

        Final Verdict: phishing
        Agent consensus: legitimate.

    Both values are intentionally preserved because they represent different
    concepts.
    """

    majority_verdict = consensus.get(
        "majority_agent_verdict",
        "unknown",
    )

    weighted_verdict = final_verdict

    majority_count = consensus.get(
        "majority_vote_count",
        0,
    )

    total_votes = consensus.get(
        "total_votes",
        0,
    )

    return {
        "majority_agent_verdict":
            majority_verdict,

        "majority_agent_vote_count":
            majority_count,

        "total_agent_votes":
            total_votes,

        "majority_agreement_ratio":
            consensus.get(
                "agreement_ratio",
                0.0,
            ),

        "plurality_agent_verdict":
            consensus.get(
                "plurality_agent_verdict",
                consensus.get("majority_agent_verdict", "unknown"),
            ),

        "tied_verdicts":
            consensus.get("tied_verdicts", []),

        "is_tie":
            consensus.get("is_tie", False),

        "weighted_fusion_verdict":
            weighted_verdict,

        "verdicts_agree":
            not conflict.get("conflict_detected", False),

        "majority_agrees_with_weighted":
            majority_verdict == weighted_verdict,

        "conflict_detected":
            conflict.get(
                "conflict_detected",
                False,
            ),

        "conflict_level":
            conflict.get(
                "conflict_level",
                "none",
            ),
    }


# ============================================================================
# ENGINE
# ============================================================================

class DecisionFusionEngine:
    """
    Evidence-aware Decision Fusion Engine.
    """

    def __init__(
        self,
        agent_weights: Optional[
            Mapping[str, float]
        ] = None,
    ) -> None:

        self.engine_name = ENGINE_NAME
        self.engine_version = ENGINE_VERSION

        self.weights = dict(
            agent_weights
            if agent_weights is not None
            else AGENT_WEIGHTS
        )

        # Validate every active agent.
        for agent in ACTIVE_AGENTS:

            if agent not in self.weights:

                raise ValueError(
                    "Missing fusion weight for active agent: "
                    f"{agent}"
                )

            weight = _safe_float(
                self.weights[agent]
            )

            if (
                weight is None
                or weight < 0
            ):

                raise ValueError(
                    "Invalid fusion weight for "
                    f"{agent}: {self.weights[agent]}"
                )

        self.active_agents = sorted(
            ACTIVE_AGENTS
        )

        self.agent_feature_mapping = dict(
            AGENT_FEATURE_MAPPING
        )

        self.minimum_agents_for_consensus = (
            MINIMUM_AGENTS_FOR_CONSENSUS
        )

        self.high_risk_threshold = (
            HIGH_RISK_THRESHOLD
        )

        self.suspicious_threshold = (
            SUSPICIOUS_THRESHOLD
        )

        self.evidence_policy = {

            "critical_ml_phishing_probability":
                CRITICAL_ML_PHISHING_PROBABILITY,

            "strong_ml_phishing_probability":
                STRONG_ML_PHISHING_PROBABILITY,

            "critical_threat_score":
                CRITICAL_THREAT_SCORE,

            "critical_malicious_engine_count":
                CRITICAL_MALICIOUS_ENGINE_COUNT,

            "dns_medium_risk_floor":
                DNS_MEDIUM_RISK_FLOOR,

            "dns_high_risk_floor":
                DNS_HIGH_RISK_FLOOR,

            "dns_critical_risk_floor":
                DNS_CRITICAL_RISK_FLOOR,

            "dns_evidence_enabled":
                True,

            "critical_override_requires_explicit_evidence":
                True,

            "critical_override_does_not_follow_risk_level_alone":
                True,

            "heuristic_fallbacks":
                dict(
                    HEURISTIC_RELIABILITY
                ),
        }

    # ========================================================================
    # STATUS
    # ========================================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return stable public engine status.
        """

        configured_weight = sum(
            self.weights.get(
                agent,
                0.0,
            )
            for agent
            in ACTIVE_AGENTS
        )

        return {

            "engine_name":
                self.engine_name,

            "engine_version":
                self.engine_version,

            "status":
                "ready",

            "active_agents":
                list(
                    self.active_agents
                ),

            "active_agent_count":
                len(
                    self.active_agents
                ),

            "weights":
                dict(
                    self.weights
                ),

            "agent_feature_mapping":
                dict(
                    self.agent_feature_mapping
                ),

            "configured_weight":
                round(
                    configured_weight,
                    6,
                ),

            "minimum_agents_for_consensus":
                self.minimum_agents_for_consensus,

            "high_risk_threshold":
                self.high_risk_threshold,

            "suspicious_threshold":
                self.suspicious_threshold,

            "html_agent":
                {
                    "enabled":
                        HTML_AGENT_NAME
                        in ACTIVE_AGENTS,

                    "feature_key":
                        "html_features",

                    "feature_count":
                        50,

                    "model_type":
                        "XGBoost",
                },

            "dns_agent":
                {
                    "enabled":
                        DNS_AGENT_NAME
                        in ACTIVE_AGENTS,

                    "feature_key":
                        "dns_features",

                    "feature_count":
                        35,

                    "model_type":
                        "XGBoost",

                    "model_path":
                        "data/models/dns_agent/dns_xgb_model.json",

                    "fallback_supported":
                        True,

                    "label_definition":
                        {
                            "0":
                                "legitimate",

                            "1":
                                "phishing",
                        },

                    "ml_prediction_sources":
                        [
                            "xgboost",
                            "trained_ml",
                        ],
                },

            "threat_intel":
                {
                    "enabled":
                        THREAT_INTEL_AGENT_NAME
                        in ACTIVE_AGENTS,

                    "active":
                        THREAT_INTEL_AGENT_NAME
                        in ACTIVE_AGENTS,

                    "conditional":
                        False,

                    "feature_key":
                        "threat_features",
                },

            "day14_fusion":
                {
                    "agent_result_first":
                        True,

                    "confidence_model":
                        "agent confidence + agreement + "
                        "coverage + decisiveness - conflict",

                    "conflict_detection":
                        True,

                    "ui_ready_output":
                        True,

                    "minimum_consensus_enforced":
                        True,
                },

            "evidence_policy":
                copy.deepcopy(
                    self.evidence_policy
                ),
        }

    def status(self) -> Dict[str, Any]:
        """Backward-compatible alias for callers using the short name."""

        return self.get_status()

    # ========================================================================
    # NO SIGNAL RESULT
    # ========================================================================

    def _build_no_signal_result(
        self,
        excluded_agents: List[str],
        threat_intel: Optional[Dict[str, Any]],
        dns_evidence: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Return result when zero agents provide usable signals.
        """

        return {

            "engine_name":
                self.engine_name,

            "engine_version":
                self.engine_version,

            "status":
                "no_signal",

            "analysis_status":
                "no_signal",

            "signal_available":
                False,

            "final_verdict":
                "unknown",

            "verdict":
                "unknown",

            "risk_level":
                "unknown",

            "final_risk_level":
                "unknown",

            "final_risk_score":
                None,

            "risk_score":
                None,

            "confidence":
                0.0,

            "phishing_probability":
                None,

            "final_phishing_probability":
                None,

            "fused_probability":
                None,

            "legitimate_probability":
                None,

            "usable_agent_count":
                0,

            "usable_agents":
                [],

            "excluded_agents":
                list(
                    excluded_agents
                ),

            "consensus_available":
                False,

            "consensus":
                {},

            "agent_contributions":
                [],

            "html_agent_contribution":
                None,

            "threat_intelligence":
                _safe_copy(
                    threat_intel
                ),

            "dns_critical_evidence":
                _safe_copy(
                    dns_evidence
                ),

            "critical_evidence_override":
                False,

            "critical_evidence":
                {
                    "critical":
                        False,

                    "critical_sources":
                        [],

                    "reasons":
                        [],

                    "threat_intel_critical":
                        False,

                    "dns_critical":
                        False,
                },

            "conflict_detected":
                False,

            "conflict_level":
                "none",

            "conflicting_agents":
                [],

            "conflict_reasons":
                [],

            "evidence_state":
                "unavailable",

            "majority_agent_verdict":
                "unknown",

            "weighted_fusion_verdict":
                "unknown",

            "verdicts_agree":
                False,

            "decision_basis":
                {
                    "type":
                        "no_signal",

                    "critical_override":
                        False,

                    "majority_agent_verdict":
                        "unknown",

                    "weighted_fusion_verdict":
                        "unknown",

                    "verdicts_agree":
                        False,

                    "reasons":
                        [
                            "No usable agent signal was available."
                        ],
                },

            "weighting":
                {},

            "reason":
                "No usable agent signal available.",
        }

    # ========================================================================
    # LIMITED EVIDENCE RESULT
    # ========================================================================

    def _build_limited_evidence_result(
        self,
        usable_decisions: List[Dict[str, Any]],
        excluded_agents: List[str],
        threat_intel: Optional[Dict[str, Any]],
        dns_evidence: Optional[Dict[str, Any]],
        critical_evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Return a safe result when usable agent count is below the configured
        minimum consensus requirement.

        IMPORTANT:

        This method intentionally DOES NOT return the prediction of the
        single usable agent as the final verdict.

        The system must communicate uncertainty rather than presenting
        one-agent evidence as a multi-agent consensus decision.
        """

        usable_count = len(
            usable_decisions
        )

        coverage = (
            usable_count
            / len(
                self.active_agents
            )
            if self.active_agents
            else 0.0
        )

        usable_names = [
            item[
                "agent"
            ]
            for item
            in usable_decisions
        ]

        # Preserve the single agent's raw evidence for the UI/audit trail.
        limited_evidence = []

        for item in usable_decisions:

            limited_evidence.append(
                {
                    "agent":
                        item.get(
                            "agent",
                            "unknown",
                        ),

                    "prediction":
                        item.get(
                            "prediction",
                            "unknown",
                        ),

                    "risk_score":
                        _clamp(
                            item.get(
                                "risk_score",
                                0.0,
                            )
                        ),

                    "confidence":
                        _clamp_probability(
                            item.get(
                                "confidence",
                                0.0,
                            )
                        )
                        or 0.0,

                    "prediction_source":
                        item.get(
                            "prediction_source",
                            "unknown",
                        ),
                }
            )

        reasons = [
            (
                "Insufficient agent coverage for a normal fusion decision."
            ),

            (
                f"Only {usable_count} of "
                f"{len(self.active_agents)} active agents produced "
                "usable evidence."
            ),

            (
                "Minimum consensus requirement is "
                f"{self.minimum_agents_for_consensus} agents."
            ),

            (
                "The available evidence is preserved, but it is not "
                "promoted to a final phishing/legitimate decision."
            ),
        ]

        return {

            "engine_name":
                self.engine_name,

            "engine_version":
                self.engine_version,

            "status":
                "limited_evidence",

            "analysis_status":
                "limited_evidence",

            "signal_available":
                True,

            # ---------------------------------------------------------------
            # IMPORTANT FIX
            # ---------------------------------------------------------------

            "final_verdict":
                "unknown",

            "verdict":
                "unknown",

            "risk_level":
                "unknown",

            "final_risk_level":
                "unknown",

            "final_risk_score":
                None,

            "risk_score":
                None,

            "confidence":
                0.0,

            "phishing_probability":
                None,

            "final_phishing_probability":
                None,

            "fused_probability":
                None,

            "legitimate_probability":
                None,

            # ---------------------------------------------------------------
            # Agent information
            # ---------------------------------------------------------------

            "usable_agent_count":
                usable_count,

            "usable_agents":
                list(
                    usable_names
                ),

            "excluded_agents":
                list(
                    excluded_agents
                ),

            "consensus_available":
                False,

            "consensus":
                {},

            "agent_contributions":
                limited_evidence,

            "html_agent_contribution":
                next(
                    (
                        item
                        for item
                        in limited_evidence
                        if item.get(
                            "agent"
                        )
                        == HTML_AGENT_NAME
                    ),
                    None,
                ),

            # ---------------------------------------------------------------
            # Preserved evidence
            # ---------------------------------------------------------------

            "threat_intelligence":
                _safe_copy(
                    threat_intel
                ),

            "dns_critical_evidence":
                _safe_copy(
                    dns_evidence
                ),

            "critical_evidence_override":
                False,

            "critical_evidence":
                _safe_copy(
                    critical_evidence
                ),

            # ---------------------------------------------------------------
            # Conflict
            # ---------------------------------------------------------------

            "conflict_detected":
                False,

            "conflict_level":
                "none",

            "conflicting_agents":
                [],

            "conflict_reasons":
                [],

            "evidence_state":
                "limited",

            "majority_agent_verdict":
                "unknown",

            "weighted_fusion_verdict":
                "unknown",

            "verdicts_agree":
                False,

            # ---------------------------------------------------------------
            # Explicit coverage information
            # ---------------------------------------------------------------

            "coverage":
                round(
                    coverage,
                    6,
                ),

            "minimum_agents_for_consensus":
                self.minimum_agents_for_consensus,

            "consensus_required":
                True,

            "consensus_satisfied":
                False,

            "limited_evidence":
                True,

            "limited_evidence_agents":
                limited_evidence,

            # ---------------------------------------------------------------
            # Decision basis
            # ---------------------------------------------------------------

            "decision_basis":
                {
                    "type":
                        "limited_evidence",

                    "critical_override":
                        False,

                    "reasons":
                        reasons,

                    "usable_agent_count":
                        usable_count,

                    "required_agent_count":
                        self.minimum_agents_for_consensus,

                    "active_agent_count":
                        len(
                            self.active_agents
                        ),

                    "coverage":
                        round(
                            coverage,
                            6,
                        ),

                    "available_evidence_preserved":
                        True,

                    "final_decision_suppressed":
                        True,

                    "final_verdict_policy":
                        (
                            "unknown when usable agent count is below "
                            "minimum consensus"
                        ),
                },

            "weighting":
                {},

            "reason":
                (
                    "Insufficient agent consensus. "
                    "Final decision suppressed."
                ),
        }

    # ========================================================================
    # CRITICAL RESULT
    # ========================================================================

    def _build_critical_result(
        self,
        usable_decisions: List[Dict[str, Any]],
        excluded_agents: List[str],
        threat_intel: Optional[Dict[str, Any]],
        dns_evidence: Optional[Dict[str, Any]],
        critical_evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build result for explicitly detected critical malicious evidence.
        """

        consensus = _calculate_consensus(
            usable_decisions
        )

        (
            normalized_weights,
            effective_weights,
            total_effective_weight,
        ) = _calculate_weights(
            usable_decisions
        )

        contributions = _build_contributions(
            usable_decisions,
            normalized_weights,
            effective_weights,
        )

        conflict = _detect_conflict(
            usable_decisions,
            consensus,
        )

        reasons = list(
            critical_evidence.get(
                "reasons",
                [],
            )
        )

        if not reasons:

            reasons = [
                "Critical malicious evidence was detected."
            ]

        reasons.insert(
            0,
            (
                "Critical malicious evidence overrides normal weighted "
                "fusion."
            ),
        )

        decision_summary = _build_decision_summary(
            "phishing",
            consensus,
            conflict,
        )

        threat_score = None

        if threat_intel is not None:

            threat_score = _safe_float(
                threat_intel.get(
                    "overall_threat_score",
                    threat_intel.get(
                        "threat_score"
                    ),
                )
            )

        if threat_score is None:

            threat_score = 95.0

        final_risk_score = _clamp(
            threat_score
        )

        decision_basis = {

            "type":
                "critical_evidence_override",

            "critical_override":
                True,

            "reasons":
                reasons,

            "majority_agent_verdict":
                consensus.get(
                    "majority_agent_verdict",
                    "unknown",
                ),

            "weighted_fusion_verdict":
                "phishing",

            "verdicts_agree":
                decision_summary[
                    "verdicts_agree"
                ],

            "override_policy":
                (
                    "Critical malicious evidence overrides normal "
                    "weighted fusion."
                ),

            "risk_score_source":
                "critical_evidence_override",

            "critical_evidence_sources":
                list(
                    critical_evidence.get(
                        "critical_sources",
                        [],
                    )
                ),

            "dns_critical":
                critical_evidence.get(
                    "dns_critical",
                    False,
                ),

            "dns_evidence_preserved":
                dns_evidence is not None,

            "dns_critical_evidence":
                _safe_copy(
                    dns_evidence
                ),

            "threat_intel_critical":
                critical_evidence.get(
                    "threat_intel_critical",
                    False,
                ),

            "threat_intelligence_evidence_preserved":
                threat_intel is not None,

            "threat_intelligence_evidence":
                _safe_copy(
                    threat_intel
                ),

            "usable_agent_count":
                len(
                    usable_decisions
                ),
        }

        return {

            "engine_name":
                self.engine_name,

            "engine_version":
                self.engine_version,

            "status":
                "critical_override",

            "analysis_status":
                "critical",

            "signal_available":
                True,

            "final_verdict":
                "phishing",

            "verdict":
                "phishing",

            "risk_level":
                "critical",

            "final_risk_level":
                "critical",

            "final_risk_score":
                round(
                    final_risk_score,
                    4,
                ),

            "risk_score":
                round(
                    final_risk_score,
                    4,
                ),

            "confidence":
                0.95,

            "phishing_probability":
                0.95,

            "final_phishing_probability":
                0.95,

            "fused_probability":
                0.95,

            "legitimate_probability":
                0.05,

            "usable_agent_count":
                len(
                    usable_decisions
                ),

            "usable_agents":
                [
                    item["agent"]
                    for item
                    in usable_decisions
                ],

            "excluded_agents":
                list(
                    excluded_agents
                ),

            "consensus_available":
                bool(
                    usable_decisions
                ),

            "consensus":
                consensus,

            "agent_contributions":
                contributions,

            "html_agent_contribution":
                next(
                    (
                        item
                        for item
                        in contributions
                        if item["agent"]
                        == HTML_AGENT_NAME
                    ),
                    None,
                ),

            "threat_intelligence":
                _safe_copy(
                    threat_intel
                ),

            "dns_critical_evidence":
                _safe_copy(
                    dns_evidence
                ),

            "critical_evidence_override":
                True,

            "critical_evidence":
                _safe_copy(
                    critical_evidence
                ),

            "conflict_detected":
                conflict.get(
                    "conflict_detected",
                    False,
                ),

            "conflict_level":
                conflict.get(
                    "conflict_level",
                    "none",
                ),

            "conflicting_agents":
                conflict.get(
                    "conflicting_agents",
                    [],
                ),

            "conflict_reasons":
                conflict.get(
                    "conflict_reasons",
                    [],
                ),

            "evidence_state":
                "critical",

            "majority_agent_verdict":
                consensus.get(
                    "majority_agent_verdict",
                    "unknown",
                ),

            "weighted_fusion_verdict":
                "phishing",

            "verdicts_agree":
                decision_summary[
                    "verdicts_agree"
                ],

            "decision_basis":
                decision_basis,

            "weighting":
                {
                    "configured":
                        dict(
                            self.weights
                        ),

                    "normalized":
                        normalized_weights,

                    "effective":
                        effective_weights,

                    "total_effective_weight":
                        total_effective_weight,
                },

            "reason":
                (
                    "Critical malicious evidence triggered an override."
                ),
        }

    # ========================================================================
    # NORMAL RESULT
    # ========================================================================

    def _build_normal_result(
        self,
        usable_decisions: List[Dict[str, Any]],
        excluded_agents: List[str],
        threat_intel: Optional[Dict[str, Any]],
        dns_evidence: Optional[Dict[str, Any]],
        critical_evidence: Dict[str, Any],
        normalized_weights: Dict[str, float],
        effective_weights: Dict[str, float],
        total_effective_weight: float,
    ) -> Dict[str, Any]:
        """
        Build normal weighted fusion result.
        """

        usable_count = len(
            usable_decisions
        )

        coverage = (
            usable_count
            / len(
                self.active_agents
            )
            if self.active_agents
            else 0.0
        )

        consensus = _calculate_consensus(
            usable_decisions
        )

        weighted_risk = _calculate_weighted_risk(
            usable_decisions,
            normalized_weights,
        )

        weighted_probability = (
            _calculate_weighted_probability(
                usable_decisions,
                normalized_weights,
            )
        )

        final_verdict = _verdict_from_risk(
            weighted_risk
        )

        risk_level = _risk_level_from_score(
            weighted_risk
        )

        # --------------------------------------------------------------------
        # Decisiveness
        #
        # Measures how far the weighted risk is from the ambiguous center.
        # 0 = highly ambiguous
        # 1 = strongly decisive
        # --------------------------------------------------------------------

        decisiveness = abs(
            weighted_risk
            - 50.0
        ) / 50.0

        decisiveness = _clamp(
            decisiveness,
            0.0,
            1.0,
        )

        conflict = _detect_conflict(
            usable_decisions,
            consensus,
        )

        confidence_model = _calculate_confidence(
            usable_decisions=usable_decisions,
            consensus=consensus,
            conflict=conflict,
            coverage=coverage,
            decisiveness=decisiveness,
        )

        final_confidence = _combine_confidence(
            confidence_model
        )

        decision_summary = _build_decision_summary(
            final_verdict,
            consensus,
            conflict,
        )

        reasons = []

        if final_verdict == "phishing":

            reasons.append(
                "Weighted evidence indicates malicious/phishing risk."
            )

        elif final_verdict == "suspicious":

            reasons.append(
                "Weighted evidence indicates suspicious phishing risk."
            )

        else:

            reasons.append(
                "Weighted evidence indicates legitimate/low phishing risk."
            )

        majority_verdict = consensus.get(
            "majority_agent_verdict",
            "unknown",
        )

        majority_count = consensus.get(
            "majority_vote_count",
            0,
        )

        if majority_verdict != "unknown":

            reasons.append(
                "Categorical consensus: "
                f"{majority_verdict}."
            )

            reasons.append(
                "Weighted fusion verdict: "
                f"{final_verdict}."
            )

            if (
                majority_verdict
                != final_verdict
            ):

                reasons.append(
                    (
                        "The majority agent verdict differs from the "
                        "weighted fusion verdict; weighted evidence "
                        "determines the final decision."
                    )
                )

        if conflict.get(
            "conflict_detected",
            False,
        ):

            reasons.extend(
                conflict.get(
                    "conflict_reasons",
                    [],
                )
            )

        decision_basis = {

            "type":
                "weighted_fusion",

            "critical_override":
                False,

            "reasons":
                reasons,

            "majority_agent_verdict":
                majority_verdict,

            "majority_agent_vote_count":
                majority_count,

            "plurality_agent_verdict":
                consensus.get(
                    "plurality_agent_verdict",
                    "unknown",
                ),

            "tied_verdicts":
                consensus.get(
                    "tied_verdicts",
                    [],
                ),

            "is_tie":
                consensus.get(
                    "is_tie",
                    False,
                ),

            "total_agent_votes":
                consensus.get(
                    "total_votes",
                    0,
                ),

            "majority_agreement_ratio":
                consensus.get(
                    "agreement_ratio",
                    0.0,
                ),

            "weighted_fusion_verdict":
                final_verdict,

            "verdicts_agree":
                decision_summary[
                    "verdicts_agree"
                ],

            "risk_score_source":
                "weighted_agent_fusion",

            "confidence_model":
                confidence_model,

            "usable_agent_count":
                usable_count,

            "coverage":
                coverage,

            "minimum_agents_for_consensus":
                self.minimum_agents_for_consensus,

            "consensus_satisfied":
                usable_count
                >= self.minimum_agents_for_consensus,

            "risk_calibration": {
                "status": "applied",
                "calibration_standard": "Operational Security Risk Harmonization (OSRH-v2)",
                "calibration_version": "2.0.0",
                "rationale": "Harmonizes heterogeneous agent scoring paradigms (ML probability, DOM heuristics, network telemetry, threat intelligence) onto a common operational security severity scale (0-100).",
                "agent_calibrations": {
                    item["agent"]: item.get("calibration", {})
                    for item in usable_decisions
                },
            },
        }

        contributions = _build_contributions(
            usable_decisions,
            normalized_weights,
            effective_weights,
        )

        return {

            "engine_name":
                self.engine_name,

            "engine_version":
                self.engine_version,

            "status":
                "success",

            "analysis_status":
                "complete",

            "signal_available":
                True,

            "final_verdict":
                final_verdict,

            "verdict":
                final_verdict,

            "risk_level":
                risk_level,

            "final_risk_level":
                risk_level,

            "final_risk_score":
                round(
                    weighted_risk,
                    4,
                ),

            "risk_score":
                round(
                    weighted_risk,
                    4,
                ),

            "risk_calibration":
                decision_basis["risk_calibration"],

            "calibrated_decisions":
                usable_decisions,

            "confidence":
                final_confidence,

            "phishing_probability":
                round(
                    weighted_probability,
                    6,
                ),

            "final_phishing_probability":
                round(
                    weighted_probability,
                    6,
                ),

            "fused_probability":
                round(
                    weighted_probability,
                    6,
                ),

            "legitimate_probability":
                round(
                    1.0
                    - weighted_probability,
                    6,
                ),

            "usable_agent_count":
                usable_count,

            "usable_agents":
                [
                    item["agent"]
                    for item
                    in usable_decisions
                ],

            "excluded_agents":
                list(
                    excluded_agents
                ),

            "consensus_available":
                True,

            "consensus":
                consensus,

            "agent_contributions":
                contributions,

            "html_agent_contribution":
                next(
                    (
                        item
                        for item
                        in contributions
                        if item["agent"]
                        == HTML_AGENT_NAME
                    ),
                    None,
                ),

            "threat_intelligence":
                _safe_copy(
                    threat_intel
                ),

            "dns_critical_evidence":
                _safe_copy(
                    dns_evidence
                ),

            "critical_evidence_override":
                False,

            "critical_evidence":
                _safe_copy(
                    critical_evidence
                ),

            "conflict_detected":
                conflict.get(
                    "conflict_detected",
                    False,
                ),

            "conflict_level":
                conflict.get(
                    "conflict_level",
                    "none",
                ),

            "conflicting_agents":
                conflict.get(
                    "conflicting_agents",
                    [],
                ),

            "conflict_reasons":
                conflict.get(
                    "conflict_reasons",
                    [],
                ),

            "evidence_state":
                (
                    "conflicted"
                    if conflict.get(
                        "conflict_detected",
                        False,
                    )
                    else "consistent"
                ),

            "majority_agent_verdict":
                majority_verdict,

            "weighted_fusion_verdict":
                final_verdict,

            "verdicts_agree":
                decision_summary[
                    "verdicts_agree"
                ],

            "decision_basis":
                decision_basis,

            "weighting":
                {
                    "configured":
                        dict(
                            self.weights
                        ),

                    "normalized":
                        {
                            key:
                                round(
                                    value,
                                    6,
                                )
                            for key, value
                            in normalized_weights.items()
                        },

                    "effective":
                        {
                            key:
                                round(
                                    value,
                                    6,
                                )
                            for key, value
                            in effective_weights.items()
                        },

                    "total_effective_weight":
                        round(
                            total_effective_weight,
                            6,
                        ),
                },

            "reason":
                (
                    "Weighted multi-agent fusion completed."
                ),
        }

    # ========================================================================
    # FUSE
    # ========================================================================

    def fuse(
        self,
        agent_decisions: Any,
    ) -> Dict[str, Any]:
        """
        Execute the complete fusion pipeline.

        Pipeline:

        1. Normalize input.
        2. Prepare usable decisions.
        3. Determine excluded agents.
        4. Preserve Threat Intelligence evidence.
        5. Preserve DNS evidence.
        6. Evaluate critical evidence.
        7. Apply critical override if necessary.
        8. Handle zero usable agents.
        9. Handle below-minimum consensus.
        10. Execute normal weighted fusion.
        """

        decisions = _normalize_input(
            agent_decisions
        )

        usable_decisions = (
            _prepare_decisions(
                decisions
            )
        )

        usable_names = {
            item[
                "agent"
            ]
            for item
            in usable_decisions
        }

        excluded_agents = [
            agent
            for agent
            in self.active_agents
            if agent
            not in usable_names
        ]

        # --------------------------------------------------------------------
        # Preserve Threat Intelligence evidence BEFORE filtering.
        # --------------------------------------------------------------------

        threat_intel = None

        for decision in decisions:

            candidate = _extract_threat_intel(
                decision
            )

            if candidate is None:
                continue

            if (
                _extract_agent_name(
                    decision
                )
                == THREAT_INTEL_AGENT_NAME
            ):

                threat_intel = candidate
                break

            if threat_intel is None:

                threat_intel = candidate

        # --------------------------------------------------------------------
        # Preserve DNS evidence BEFORE filtering.
        # --------------------------------------------------------------------

        dns_evidence = None

        for decision in decisions:

            candidate = _extract_dns_evidence(
                decision
            )

            if candidate is not None:

                dns_evidence = candidate
                break

        # --------------------------------------------------------------------
        # Critical evidence evaluation.
        # --------------------------------------------------------------------

        critical_evidence = (
            _evaluate_critical_evidence(
                usable_decisions,
                threat_intel,
                dns_evidence,
            )
        )

        # --------------------------------------------------------------------
        # CRITICAL OVERRIDE
        # --------------------------------------------------------------------
        #
        # Explicitly configured critical malicious evidence is allowed to
        # override ordinary weighted fusion.
        #
        # This is evaluated before the minimum-consensus branch.
        # --------------------------------------------------------------------

        if critical_evidence[
            "critical"
        ]:

            return self._build_critical_result(
                usable_decisions,
                excluded_agents,
                threat_intel,
                dns_evidence,
                critical_evidence,
            )

        # --------------------------------------------------------------------
        # ZERO USABLE SIGNAL
        # --------------------------------------------------------------------

        if not usable_decisions:

            return self._build_no_signal_result(
                excluded_agents,
                threat_intel,
                dns_evidence,
            )

        # --------------------------------------------------------------------
        # BELOW MINIMUM CONSENSUS
        # --------------------------------------------------------------------
        #
        # THIS IS THE IMPORTANT STEP 4.7.4 FIX.
        #
        # One usable agent is NOT enough for a final phishing/legitimate
        # decision.
        #
        # We preserve the agent's evidence but suppress the final verdict.
        # --------------------------------------------------------------------

        if (
            len(
                usable_decisions
            )
            < self.minimum_agents_for_consensus
        ):

            return self._build_limited_evidence_result(
                usable_decisions=usable_decisions,
                excluded_agents=excluded_agents,
                threat_intel=threat_intel,
                dns_evidence=dns_evidence,
                critical_evidence=critical_evidence,
            )

        # --------------------------------------------------------------------
        # NORMAL WEIGHTED FUSION
        # --------------------------------------------------------------------

        (
            normalized_weights,
            effective_weights,
            total_effective_weight,
        ) = _calculate_weights(
            usable_decisions
        )

        return self._build_normal_result(
            usable_decisions,
            excluded_agents,
            threat_intel,
            dns_evidence,
            critical_evidence,
            normalized_weights,
            effective_weights,
            total_effective_weight,
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def fuse_agent_decisions(
    agent_decisions: Any,
) -> Dict[str, Any]:
    """
    Convenience wrapper around DecisionFusionEngine.
    """

    engine = DecisionFusionEngine()

    return engine.fuse(
        agent_decisions
    )


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [

    "DecisionFusionEngine",

    "fuse_agent_decisions",

    "ACTIVE_AGENTS",

    "AGENT_WEIGHTS",

    "AGENT_FEATURE_MAPPING",

    "HEURISTIC_RELIABILITY",

    "MINIMUM_AGENTS_FOR_CONSENSUS",

    "HIGH_RISK_THRESHOLD",

    "SUSPICIOUS_THRESHOLD",

    "CRITICAL_ML_PHISHING_PROBABILITY",

    "STRONG_ML_PHISHING_PROBABILITY",

    "CRITICAL_THREAT_SCORE",

    "CRITICAL_MALICIOUS_ENGINE_COUNT",

    "DNS_MEDIUM_RISK_FLOOR",

    "DNS_HIGH_RISK_FLOOR",

    "DNS_CRITICAL_RISK_FLOOR",

    "ENGINE_NAME",

    "ENGINE_VERSION",

    # Day 14 compatibility helpers.
    "_normalize_input",

    "_prepare_decisions",

    "_calculate_consensus",

    "_detect_conflict",

    "_calculate_weights",

    "_calculate_weighted_risk",

    "_calculate_weighted_probability",

    "_calculate_confidence",

    "_evaluate_critical_evidence",
]