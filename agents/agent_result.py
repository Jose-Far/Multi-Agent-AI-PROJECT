"""
Standard Agent Result
=====================

Day 13 - Multi-Agent AI Cybersecurity Analyst

This module defines the common result contract used by all cybersecurity
agents.

Active agents currently supported:

    1. URL AI Agent
    2. HTML AI Agent
    3. SSL AI Agent
    4. DNS AI Agent
    5. Visual AI Agent
    6. Threat Intelligence AI Agent

Architecture:

    Agent
       |
       v
    AgentResult
       |
       v
    Orchestrator
       |
       v
    Fusion
       |
       v
    UI / Report

Security principles:

    - unavailable != legitimate
    - error != legitimate
    - unknown prediction is used when analysis is unavailable
    - missing probability is represented by None
    - missing risk score is represented by None
    - risk scores are always normalized to 0-100
    - risk levels use one common scale
    - ML and heuristic predictions are explicitly identified
    - old orchestrator fields remain available for compatibility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional


# ============================================================================
# SCHEMA VERSION
# ============================================================================

AGENT_RESULT_SCHEMA_VERSION = "1.0.0"


# ============================================================================
# STATUS VALUES
# ============================================================================

STATUS_SUCCESS = "success"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ERROR = "error"

VALID_STATUSES = {
    STATUS_SUCCESS,
    STATUS_PARTIAL,
    STATUS_UNAVAILABLE,
    STATUS_ERROR,
}


# ============================================================================
# PREDICTION VALUES
# ============================================================================

PREDICTION_LEGITIMATE = "legitimate"
PREDICTION_PHISHING = "phishing"
PREDICTION_SUSPICIOUS = "suspicious"
PREDICTION_MALICIOUS = "malicious"
PREDICTION_UNKNOWN = "unknown"

VALID_PREDICTIONS = {
    PREDICTION_LEGITIMATE,
    PREDICTION_PHISHING,
    PREDICTION_SUSPICIOUS,
    PREDICTION_MALICIOUS,
    PREDICTION_UNKNOWN,
}


# ============================================================================
# PREDICTION SOURCE
# ============================================================================

PREDICTION_SOURCE_TRAINED_ML = "trained_ml"
PREDICTION_SOURCE_HEURISTIC = "heuristic"
PREDICTION_SOURCE_FALLBACK_HEURISTIC = "fallback_heuristic"
PREDICTION_SOURCE_UNAVAILABLE = "unavailable"
PREDICTION_SOURCE_ERROR = "error"

VALID_PREDICTION_SOURCES = {
    PREDICTION_SOURCE_TRAINED_ML,
    PREDICTION_SOURCE_HEURISTIC,
    PREDICTION_SOURCE_FALLBACK_HEURISTIC,
    PREDICTION_SOURCE_UNAVAILABLE,
    PREDICTION_SOURCE_ERROR,
}


# ============================================================================
# MODEL STATUS
# ============================================================================

MODEL_STATUS_AVAILABLE = "available"
MODEL_STATUS_FALLBACK = "fallback"
MODEL_STATUS_UNAVAILABLE = "unavailable"
MODEL_STATUS_ERROR = "error"

VALID_MODEL_STATUSES = {
    MODEL_STATUS_AVAILABLE,
    MODEL_STATUS_FALLBACK,
    MODEL_STATUS_UNAVAILABLE,
    MODEL_STATUS_ERROR,
}


# ============================================================================
# RISK LEVELS
# ============================================================================

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
RISK_CRITICAL = "critical"
RISK_UNKNOWN = "unknown"

VALID_RISK_LEVELS = {
    RISK_LOW,
    RISK_MEDIUM,
    RISK_HIGH,
    RISK_CRITICAL,
    RISK_UNKNOWN,
}


# ============================================================================
# LIMITS
# ============================================================================

MIN_RISK_SCORE = 0
MAX_RISK_SCORE = 100

MIN_PROBABILITY = 0.0
MAX_PROBABILITY = 1.0

MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0


# ============================================================================
# RISK LEVEL CALCULATOR
# ============================================================================

def get_risk_level(
    risk_score: Optional[float],
) -> str:
    """
    Convert a risk score into the standard project risk level.

    0-29   -> low
    30-59  -> medium
    60-79  -> high
    80-100 -> critical

    None -> unknown
    """

    if risk_score is None:
        return RISK_UNKNOWN

    try:
        score = float(risk_score)
    except (TypeError, ValueError):
        return RISK_UNKNOWN

    score = max(
        MIN_RISK_SCORE,
        min(
            MAX_RISK_SCORE,
            score,
        ),
    )

    if score < 30:
        return RISK_LOW

    if score < 60:
        return RISK_MEDIUM

    if score < 80:
        return RISK_HIGH

    return RISK_CRITICAL


# ============================================================================
# NORMALIZE RISK SCORE
# ============================================================================

def normalize_risk_score(
    risk_score: Optional[float],
) -> Optional[float]:
    """
    Normalize risk score to 0-100.

    None remains None.
    """

    if risk_score is None:
        return None

    try:
        score = float(risk_score)
    except (TypeError, ValueError):
        return None

    score = max(
        MIN_RISK_SCORE,
        min(
            MAX_RISK_SCORE,
            score,
        ),
    )

    if score.is_integer():
        return int(score)

    return round(score, 4)


# ============================================================================
# NORMALIZE PROBABILITY
# ============================================================================

def normalize_probability(
    probability: Optional[float],
) -> Optional[float]:
    """
    Normalize probability to 0.0-1.0.
    """

    if probability is None:
        return None

    try:
        value = float(probability)
    except (TypeError, ValueError):
        return None

    value = max(
        MIN_PROBABILITY,
        min(
            MAX_PROBABILITY,
            value,
        ),
    )

    return round(value, 6)


# ============================================================================
# NORMALIZE CONFIDENCE
# ============================================================================

def normalize_confidence(
    confidence: Optional[float],
) -> float:
    """
    Normalize confidence to 0.0-1.0.
    """

    if confidence is None:
        return 0.0

    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return 0.0

    value = max(
        MIN_CONFIDENCE,
        min(
            MAX_CONFIDENCE,
            value,
        ),
    )

    return round(value, 6)


# ============================================================================
# AGENT RESULT
# ============================================================================

@dataclass
class AgentResult:
    """
    Standard result object for every cybersecurity agent.
    """

    # ========================================================================
    # SCHEMA
    # ========================================================================

    schema_version: ClassVar[str] = (
        AGENT_RESULT_SCHEMA_VERSION
    )

    # ========================================================================
    # IDENTITY
    # ========================================================================

    agent: str

    # ========================================================================
    # EXECUTION
    # ========================================================================

    status: str = STATUS_SUCCESS

    # ========================================================================
    # PREDICTION
    # ========================================================================

    prediction: str = PREDICTION_UNKNOWN

    probability: Optional[float] = None

    confidence: float = 0.0

    # ========================================================================
    # RISK
    # ========================================================================

    risk_score: Optional[float] = None

    risk_level: str = RISK_UNKNOWN

    # ========================================================================
    # MODEL
    # ========================================================================

    prediction_source: str = (
        PREDICTION_SOURCE_UNAVAILABLE
    )

    model_status: str = (
        MODEL_STATUS_UNAVAILABLE
    )

    # ========================================================================
    # EXPLAINABILITY
    # ========================================================================

    risk_factors: List[Any] = field(
        default_factory=list
    )

    evidence: List[Any] = field(
        default_factory=list
    )

    explanation: Dict[str, Any] = field(
        default_factory=dict
    )

    # ========================================================================
    # OPTIONAL EXECUTION INFORMATION
    # ========================================================================

    execution_time_ms: Optional[float] = None

    # ========================================================================
    # OPTIONAL ERROR INFORMATION
    # ========================================================================

    error: Optional[str] = None

    error_type: Optional[str] = None

    # ========================================================================
    # FEATURE INFORMATION
    # ========================================================================

    feature_key: Optional[str] = None

    # ========================================================================
    # EXTENSIBLE METADATA
    # ========================================================================

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


    # ========================================================================
    # POST INITIALIZATION
    # ========================================================================

    def __post_init__(self) -> None:

        # --------------------------------------------------------------------
        # Agent
        # --------------------------------------------------------------------

        if not isinstance(
            self.agent,
            str,
        ):
            self.agent = str(
                self.agent
            )

        self.agent = self.agent.strip()

        if not self.agent:
            self.agent = "Unknown Agent"

        # --------------------------------------------------------------------
        # Status
        # --------------------------------------------------------------------

        self.status = self._normalize_status(
            self.status
        )

        # --------------------------------------------------------------------
        # Prediction
        # --------------------------------------------------------------------

        self.prediction = self._normalize_prediction(
            self.prediction
        )

        # --------------------------------------------------------------------
        # Probability
        # --------------------------------------------------------------------

        self.probability = normalize_probability(
            self.probability
        )

        # --------------------------------------------------------------------
        # Confidence
        # --------------------------------------------------------------------

        self.confidence = normalize_confidence(
            self.confidence
        )

        # --------------------------------------------------------------------
        # Risk score
        # --------------------------------------------------------------------

        self.risk_score = normalize_risk_score(
            self.risk_score
        )

        # --------------------------------------------------------------------
        # Risk level
        # --------------------------------------------------------------------

        self.risk_level = self._normalize_risk_level(
            self.risk_level
        )

        if self.risk_score is not None:

            self.risk_level = get_risk_level(
                self.risk_score
            )

        # --------------------------------------------------------------------
        # Prediction source
        # --------------------------------------------------------------------

        self.prediction_source = (
            self._normalize_prediction_source(
                self.prediction_source
            )
        )

        # --------------------------------------------------------------------
        # Model status
        # --------------------------------------------------------------------

        self.model_status = (
            self._normalize_model_status(
                self.model_status
            )
        )

        # --------------------------------------------------------------------
        # Risk factors
        # --------------------------------------------------------------------

        if self.risk_factors is None:

            self.risk_factors = []

        elif not isinstance(
            self.risk_factors,
            list,
        ):

            self.risk_factors = [
                self.risk_factors
            ]

        # --------------------------------------------------------------------
        # Evidence
        # --------------------------------------------------------------------

        if self.evidence is None:

            self.evidence = []

        elif not isinstance(
            self.evidence,
            list,
        ):

            self.evidence = [
                self.evidence
            ]

        # --------------------------------------------------------------------
        # Explanation
        # --------------------------------------------------------------------

        if self.explanation is None:

            self.explanation = {}

        elif not isinstance(
            self.explanation,
            dict,
        ):

            self.explanation = {
                "summary": str(
                    self.explanation
                )
            }

        # --------------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------------

        if self.metadata is None:

            self.metadata = {}

        elif not isinstance(
            self.metadata,
            dict,
        ):

            self.metadata = {
                "value": self.metadata
            }

        # --------------------------------------------------------------------
        # Execution time
        # --------------------------------------------------------------------

        if self.execution_time_ms is not None:

            try:

                self.execution_time_ms = round(
                    max(
                        0.0,
                        float(
                            self.execution_time_ms
                        ),
                    ),
                    3,
                )

            except (
                TypeError,
                ValueError,
            ):

                self.execution_time_ms = None

        # --------------------------------------------------------------------
        # Security consistency rules
        # --------------------------------------------------------------------

        self._apply_safety_rules()


    # ========================================================================
    # STATUS NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_status(
        status: Any,
    ) -> str:

        if status is None:
            return STATUS_ERROR

        value = str(
            status
        ).strip().lower()

        aliases = {

            "ok":
                STATUS_SUCCESS,

            "successful":
                STATUS_SUCCESS,

            "completed":
                STATUS_SUCCESS,

            "partial_success":
                STATUS_PARTIAL,

            "failed":
                STATUS_ERROR,

            "failure":
                STATUS_ERROR,

            "not_available":
                STATUS_UNAVAILABLE,

            "missing":
                STATUS_UNAVAILABLE,
        }

        value = aliases.get(
            value,
            value,
        )

        if value not in VALID_STATUSES:
            return STATUS_ERROR

        return value


    # ========================================================================
    # PREDICTION NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_prediction(
        prediction: Any,
    ) -> str:

        if prediction is None:
            return PREDICTION_UNKNOWN

        value = str(
            prediction
        ).strip().lower()

        aliases = {

            "benign":
                PREDICTION_LEGITIMATE,

            "safe":
                PREDICTION_LEGITIMATE,

            "legit":
                PREDICTION_LEGITIMATE,

            "clean":
                PREDICTION_LEGITIMATE,

            "clean_reputation":
                PREDICTION_LEGITIMATE,

            "phish":
                PREDICTION_PHISHING,

            "malicious_threat":
                PREDICTION_PHISHING,

            "threat":
                PREDICTION_PHISHING,

            "malware":
                PREDICTION_MALICIOUS,

            "none":
                PREDICTION_UNKNOWN,
        }

        value = aliases.get(
            value,
            value,
        )

        if value not in VALID_PREDICTIONS:
            return PREDICTION_UNKNOWN

        return value


    # ========================================================================
    # RISK LEVEL NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_risk_level(
        risk_level: Any,
    ) -> str:

        if risk_level is None:
            return RISK_UNKNOWN

        value = str(
            risk_level
        ).strip().lower()

        aliases = {

            "safe":
                RISK_LOW,

            "normal":
                RISK_LOW,

            "moderate":
                RISK_MEDIUM,

            "medium-risk":
                RISK_MEDIUM,

            "high-risk":
                RISK_HIGH,

            "severe":
                RISK_CRITICAL,

            "critical-risk":
                RISK_CRITICAL,

            "none":
                RISK_UNKNOWN,

            "unknown":
                RISK_UNKNOWN,
        }

        value = aliases.get(
            value,
            value,
        )

        if value not in VALID_RISK_LEVELS:
            return RISK_UNKNOWN

        return value


    # ========================================================================
    # PREDICTION SOURCE NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_prediction_source(
        source: Any,
    ) -> str:

        if source is None:
            return PREDICTION_SOURCE_UNAVAILABLE

        value = str(
            source
        ).strip().lower()

        aliases = {

            "xgboost":
                PREDICTION_SOURCE_TRAINED_ML,

            "ml":
                PREDICTION_SOURCE_TRAINED_ML,

            "machine_learning":
                PREDICTION_SOURCE_TRAINED_ML,

            "trained":
                PREDICTION_SOURCE_TRAINED_ML,

            "rule_based":
                PREDICTION_SOURCE_HEURISTIC,

            "rules":
                PREDICTION_SOURCE_HEURISTIC,

            "fallback":
                PREDICTION_SOURCE_FALLBACK_HEURISTIC,

            "fallback_rule_based":
                PREDICTION_SOURCE_FALLBACK_HEURISTIC,

            "none":
                PREDICTION_SOURCE_UNAVAILABLE,
        }

        value = aliases.get(
            value,
            value,
        )

        if value not in VALID_PREDICTION_SOURCES:
            return PREDICTION_SOURCE_UNAVAILABLE

        return value


    # ========================================================================
    # MODEL STATUS NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_model_status(
        model_status: Any,
    ) -> str:

        if model_status is None:
            return MODEL_STATUS_UNAVAILABLE

        value = str(
            model_status
        ).strip().lower()

        aliases = {

            "loaded":
                MODEL_STATUS_AVAILABLE,

            "ready":
                MODEL_STATUS_AVAILABLE,

            "active":
                MODEL_STATUS_AVAILABLE,

            "loaded_ml_model":
                MODEL_STATUS_AVAILABLE,

            "trained":
                MODEL_STATUS_AVAILABLE,

            "fallback_mode":
                MODEL_STATUS_FALLBACK,

            "fallback_heuristic":
                MODEL_STATUS_FALLBACK,

            "not_loaded":
                MODEL_STATUS_UNAVAILABLE,

            "failed":
                MODEL_STATUS_ERROR,
        }

        value = aliases.get(
            value,
            value,
        )

        if value not in VALID_MODEL_STATUSES:
            return MODEL_STATUS_UNAVAILABLE

        return value


    # ========================================================================
    # SECURITY SAFETY RULES
    # ========================================================================

    def _apply_safety_rules(
        self,
    ) -> None:
        """
        Ensure unavailable/error states cannot accidentally become
        legitimate/low-risk results.
        """

        # --------------------------------------------------------------------
        # UNAVAILABLE
        # --------------------------------------------------------------------

        if self.status == STATUS_UNAVAILABLE:

            self.prediction = (
                PREDICTION_UNKNOWN
            )

            self.probability = None

            self.confidence = 0.0

            self.risk_score = None

            self.risk_level = (
                RISK_UNKNOWN
            )

            self.prediction_source = (
                PREDICTION_SOURCE_UNAVAILABLE
            )

            self.model_status = (
                MODEL_STATUS_UNAVAILABLE
            )

        # --------------------------------------------------------------------
        # ERROR
        # --------------------------------------------------------------------

        elif self.status == STATUS_ERROR:

            self.prediction = (
                PREDICTION_UNKNOWN
            )

            self.probability = None

            self.confidence = 0.0

            self.risk_score = None

            self.risk_level = (
                RISK_UNKNOWN
            )

            self.prediction_source = (
                PREDICTION_SOURCE_ERROR
            )

            self.model_status = (
                MODEL_STATUS_ERROR
            )

        # --------------------------------------------------------------------
        # UNKNOWN PREDICTION
        # --------------------------------------------------------------------

        if self.prediction == PREDICTION_UNKNOWN:

            if self.status in {
                STATUS_UNAVAILABLE,
                STATUS_ERROR,
            }:

                self.probability = None

        # --------------------------------------------------------------------
        # No risk score
        # --------------------------------------------------------------------

        if self.risk_score is None:

            if self.status in {
                STATUS_UNAVAILABLE,
                STATUS_ERROR,
            }:

                self.risk_level = (
                    RISK_UNKNOWN
                )


    # ========================================================================
    # SUCCESS FACTORY
    # ========================================================================

    @classmethod
    def success(
        cls,
        agent: str,
        prediction: str,
        probability: Optional[float] = None,
        confidence: float = 0.0,
        risk_score: Optional[float] = None,
        prediction_source: str = (
            PREDICTION_SOURCE_TRAINED_ML
        ),
        model_status: str = (
            MODEL_STATUS_AVAILABLE
        ),
        risk_factors: Optional[List[Any]] = None,
        evidence: Optional[List[Any]] = None,
        explanation: Optional[Dict[str, Any]] = None,
        execution_time_ms: Optional[float] = None,
        feature_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentResult":

        return cls(

            agent=agent,

            status=STATUS_SUCCESS,

            prediction=prediction,

            probability=probability,

            confidence=confidence,

            risk_score=risk_score,

            risk_level=get_risk_level(
                risk_score
            ),

            prediction_source=prediction_source,

            model_status=model_status,

            risk_factors=(
                risk_factors
                if risk_factors is not None
                else []
            ),

            evidence=(
                evidence
                if evidence is not None
                else []
            ),

            explanation=(
                explanation
                if explanation is not None
                else {}
            ),

            execution_time_ms=(
                execution_time_ms
            ),

            feature_key=feature_key,

            metadata=(
                metadata
                if metadata is not None
                else {}
            ),
        )


    # ========================================================================
    # PARTIAL FACTORY
    # ========================================================================

    @classmethod
    def partial(
        cls,
        agent: str,
        prediction: str = PREDICTION_UNKNOWN,
        probability: Optional[float] = None,
        confidence: float = 0.0,
        risk_score: Optional[float] = None,
        prediction_source: str = (
            PREDICTION_SOURCE_HEURISTIC
        ),
        model_status: str = (
            MODEL_STATUS_FALLBACK
        ),
        risk_factors: Optional[List[Any]] = None,
        evidence: Optional[List[Any]] = None,
        explanation: Optional[Dict[str, Any]] = None,
        execution_time_ms: Optional[float] = None,
        feature_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentResult":

        return cls(

            agent=agent,

            status=STATUS_PARTIAL,

            prediction=prediction,

            probability=probability,

            confidence=confidence,

            risk_score=risk_score,

            risk_level=get_risk_level(
                risk_score
            ),

            prediction_source=prediction_source,

            model_status=model_status,

            risk_factors=(
                risk_factors
                if risk_factors is not None
                else []
            ),

            evidence=(
                evidence
                if evidence is not None
                else []
            ),

            explanation=(
                explanation
                if explanation is not None
                else {}
            ),

            execution_time_ms=(
                execution_time_ms
            ),

            feature_key=feature_key,

            metadata=(
                metadata
                if metadata is not None
                else {}
            ),
        )


    # ========================================================================
    # UNAVAILABLE FACTORY
    # ========================================================================

    @classmethod
    def unavailable(
        cls,
        agent: str,
        reason: Optional[str] = None,
        feature_key: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentResult":

        explanation = {}

        if reason:

            explanation[
                "summary"
            ] = reason

        return cls(

            agent=agent,

            status=STATUS_UNAVAILABLE,

            prediction=PREDICTION_UNKNOWN,

            probability=None,

            confidence=0.0,

            risk_score=None,

            risk_level=RISK_UNKNOWN,

            prediction_source=(
                PREDICTION_SOURCE_UNAVAILABLE
            ),

            model_status=(
                MODEL_STATUS_UNAVAILABLE
            ),

            risk_factors=[],

            evidence=[],

            explanation=explanation,

            execution_time_ms=(
                execution_time_ms
            ),

            feature_key=feature_key,

            metadata=(
                metadata
                if metadata is not None
                else {}
            ),
        )


    # ========================================================================
    # ERROR FACTORY
    # ========================================================================

    @classmethod
    def error_result(
        cls,
        agent: str,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        feature_key: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentResult":

        return cls(

            agent=agent,

            status=STATUS_ERROR,

            prediction=PREDICTION_UNKNOWN,

            probability=None,

            confidence=0.0,

            risk_score=None,

            risk_level=RISK_UNKNOWN,

            prediction_source=(
                PREDICTION_SOURCE_ERROR
            ),

            model_status=(
                MODEL_STATUS_ERROR
            ),

            risk_factors=[],

            evidence=[],

            explanation={},

            execution_time_ms=(
                execution_time_ms
            ),

            error=error,

            error_type=error_type,

            feature_key=feature_key,

            metadata=(
                metadata
                if metadata is not None
                else {}
            ),
        )


    # ========================================================================
    # SIGNAL AVAILABILITY
    # ========================================================================

    def is_signal_available(
        self,
    ) -> bool:
        """
        Return True when the agent produced usable analysis.
        """

        return self.status in {
            STATUS_SUCCESS,
            STATUS_PARTIAL,
        }


    # ========================================================================
    # FUSION USABILITY
    # ========================================================================

    def is_fusion_usable(
        self,
    ) -> bool:
        """
        Return True when this result can participate in mathematical fusion.
        """

        return (
            self.is_signal_available()
            and
            self.prediction
            != PREDICTION_UNKNOWN
        )


    # ========================================================================
    # CONVERT TO DICTIONARY
    # ========================================================================

    def to_dict(
        self,
        include_none: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert AgentResult into a dictionary.

        Both the new Day-13 fields and the existing orchestrator compatibility
        fields are returned.
        """

        result = {

            # ----------------------------------------------------------------
            # Schema
            # ----------------------------------------------------------------

            "schema_version":
                self.schema_version,

            # ----------------------------------------------------------------
            # New standard identity
            # ----------------------------------------------------------------

            "agent":
                self.agent,

            # ----------------------------------------------------------------
            # Existing orchestrator compatibility
            # ----------------------------------------------------------------

            "agent_name":
                self.agent,

            # ----------------------------------------------------------------
            # New status
            # ----------------------------------------------------------------

            "status":
                self.status,

            # ----------------------------------------------------------------
            # Existing orchestrator compatibility
            # ----------------------------------------------------------------

            "analysis_status":
                self.status,

            # ----------------------------------------------------------------
            # Existing orchestrator signal field
            # ----------------------------------------------------------------

            "signal_available":
                self.is_signal_available(),

            # ----------------------------------------------------------------
            # Prediction
            # ----------------------------------------------------------------

            "prediction":
                self.prediction,

            "probability":
                self.probability,

            "confidence":
                self.confidence,

            # ----------------------------------------------------------------
            # Risk
            # ----------------------------------------------------------------

            "risk_score":
                self.risk_score,

            "risk_level":
                self.risk_level,

            # ----------------------------------------------------------------
            # Model
            # ----------------------------------------------------------------

            "prediction_source":
                self.prediction_source,

            "model_status":
                self.model_status,

            # ----------------------------------------------------------------
            # Explainability
            # ----------------------------------------------------------------

            "risk_factors":
                list(
                    self.risk_factors
                ),

            "evidence":
                list(
                    self.evidence
                ),

            "explanation":
                dict(
                    self.explanation
                ),

            "reason":
                (
                    str(self.explanation.get("summary")).strip()
                    if isinstance(self.explanation, dict)
                    and self.explanation.get("summary")
                    else None
                ),

            # ----------------------------------------------------------------
            # Additional information
            # ----------------------------------------------------------------

            "execution_time_ms":
                self.execution_time_ms,

            "feature_key":
                self.feature_key,

            "error":
                self.error,

            "error_type":
                self.error_type,

            "metadata":
                dict(
                    self.metadata
                ),
        }

        if not include_none:

            result = {
                key: value
                for key, value in result.items()
                if value is not None
            }

        return result


    # ========================================================================
    # DICTIONARY ALIAS
    # ========================================================================

    def as_dict(
        self,
        include_none: bool = True,
    ) -> Dict[str, Any]:

        return self.to_dict(
            include_none=include_none
        )


    # ========================================================================
    # FROM DICTIONARY
    # ========================================================================

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        default_agent: str = "Unknown Agent",
    ) -> "AgentResult":
        """
        Convert an existing agent dictionary into AgentResult.

        This is important during Day 13 because existing agents still use
        different output formats.
        """

        if not isinstance(
            data,
            dict,
        ):

            return cls.error_result(

                agent=default_agent,

                error=(
                    "Agent result must be a dictionary."
                ),

                error_type=(
                    "InvalidAgentResultType"
                ),
            )

        # --------------------------------------------------------------------
        # Identity
        # --------------------------------------------------------------------

        agent = (
            data.get(
                "agent"
            )
            or data.get(
                "agent_name"
            )
            or default_agent
        )

        # --------------------------------------------------------------------
        # Status
        # --------------------------------------------------------------------

        status = (
            data.get(
                "status"
            )
            or data.get(
                "analysis_status"
            )
            or STATUS_SUCCESS
        )

        # --------------------------------------------------------------------
        # Prediction
        # --------------------------------------------------------------------

        prediction = (
            data.get(
                "prediction"
            )
            or data.get(
                "final_verdict"
            )
            or data.get(
                "verdict"
            )
            or PREDICTION_UNKNOWN
        )

        # --------------------------------------------------------------------
        # Probability
        # --------------------------------------------------------------------

        probability = data.get(
            "probability"
        )

        if probability is None:

            probability = data.get(
                "prediction_probability"
            )

        if probability is None:

            probability = data.get(
                "phishing_probability"
            )

        # --------------------------------------------------------------------
        # Confidence
        # --------------------------------------------------------------------

        confidence = data.get(
            "confidence",
            0.0,
        )

        # --------------------------------------------------------------------
        # Risk score
        # --------------------------------------------------------------------

        risk_score = data.get(
            "risk_score"
        )

        if risk_score is None:

            risk_score = data.get(
                "final_risk_score"
            )

        # --------------------------------------------------------------------
        # Risk level
        # --------------------------------------------------------------------

        risk_level = data.get(
            "risk_level"
        )

        # --------------------------------------------------------------------
        # Prediction source
        # --------------------------------------------------------------------

        prediction_source = (
            data.get(
                "prediction_source"
            )
            or data.get(
                "source"
            )
            or PREDICTION_SOURCE_UNAVAILABLE
        )

        # --------------------------------------------------------------------
        # Model status
        # --------------------------------------------------------------------

        model_status = data.get(
            "model_status"
        )

        if model_status is None:

            if (
                prediction_source
                == PREDICTION_SOURCE_FALLBACK_HEURISTIC
            ):

                model_status = (
                    MODEL_STATUS_FALLBACK
                )

            elif (
                prediction_source
                == PREDICTION_SOURCE_TRAINED_ML
            ):

                model_status = (
                    MODEL_STATUS_AVAILABLE
                )

            else:

                model_status = (
                    MODEL_STATUS_UNAVAILABLE
                )

        # --------------------------------------------------------------------
        # Risk factors
        # --------------------------------------------------------------------

        risk_factors = (
            data.get(
                "risk_factors"
            )
            or []
        )

        # --------------------------------------------------------------------
        # Evidence
        # --------------------------------------------------------------------

        evidence = (
            data.get(
                "evidence"
            )
            or []
        )

        # --------------------------------------------------------------------
        # Explanation
        # --------------------------------------------------------------------

        explanation = (
            data.get(
                "explanation"
            )
            or {}
        )

        # --------------------------------------------------------------------
        # Additional fields
        # --------------------------------------------------------------------

        execution_time_ms = data.get(
            "execution_time_ms"
        )

        error = data.get(
            "error"
        )

        error_type = data.get(
            "error_type"
        )

        feature_key = data.get(
            "feature_key"
        )

        metadata = dict(
            data.get(
                "metadata"
            )
            or {}
        )

        # --------------------------------------------------------------------
        # Preserve unknown fields
        # --------------------------------------------------------------------

        known_fields = {

            "schema_version",

            "agent",
            "agent_name",

            "status",
            "analysis_status",
            "signal_available",

            "prediction",
            "final_verdict",
            "verdict",

            "probability",
            "prediction_probability",
            "phishing_probability",

            "confidence",

            "risk_score",
            "final_risk_score",
            "risk_level",

            "prediction_source",
            "source",
            "model_status",

            "risk_factors",
            "evidence",
            "explanation",

            "execution_time_ms",

            "feature_key",

            "error",
            "error_type",

            "metadata",
        }

        for key, value in data.items():

            if key not in known_fields:

                metadata.setdefault(
                    key,
                    value,
                )

        return cls(

            agent=agent,

            status=status,

            prediction=prediction,

            probability=probability,

            confidence=confidence,

            risk_score=risk_score,

            risk_level=(
                risk_level
                if risk_level is not None
                else RISK_UNKNOWN
            ),

            prediction_source=prediction_source,

            model_status=model_status,

            risk_factors=risk_factors,

            evidence=evidence,

            explanation=explanation,

            execution_time_ms=execution_time_ms,

            error=error,

            error_type=error_type,

            feature_key=feature_key,

            metadata=metadata,
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def make_success_result(
    agent: str,
    prediction: str,
    probability: Optional[float] = None,
    confidence: float = 0.0,
    risk_score: Optional[float] = None,
    prediction_source: str = (
        PREDICTION_SOURCE_TRAINED_ML
    ),
    model_status: str = (
        MODEL_STATUS_AVAILABLE
    ),
    risk_factors: Optional[List[Any]] = None,
    evidence: Optional[List[Any]] = None,
    explanation: Optional[Dict[str, Any]] = None,
    execution_time_ms: Optional[float] = None,
    feature_key: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    return AgentResult.success(

        agent=agent,

        prediction=prediction,

        probability=probability,

        confidence=confidence,

        risk_score=risk_score,

        prediction_source=prediction_source,

        model_status=model_status,

        risk_factors=risk_factors,

        evidence=evidence,

        explanation=explanation,

        execution_time_ms=execution_time_ms,

        feature_key=feature_key,

        metadata=metadata,

    ).to_dict()


# ============================================================================
# UNAVAILABLE RESULT
# ============================================================================

def make_unavailable_result(
    agent: str,
    reason: Optional[str] = None,
    feature_key: Optional[str] = None,
    execution_time_ms: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    return AgentResult.unavailable(

        agent=agent,

        reason=reason,

        feature_key=feature_key,

        execution_time_ms=execution_time_ms,

        metadata=metadata,

    ).to_dict()


# ============================================================================
# ERROR RESULT
# ============================================================================

def make_error_result(
    agent: str,
    error: Optional[str] = None,
    error_type: Optional[str] = None,
    feature_key: Optional[str] = None,
    execution_time_ms: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    return AgentResult.error_result(

        agent=agent,

        error=error,

        error_type=error_type,

        feature_key=feature_key,

        execution_time_ms=execution_time_ms,

        metadata=metadata,

    ).to_dict()


# ============================================================================
# VALIDATION
# ============================================================================

def validate_agent_result(
    result: Any,
) -> bool:
    """
    Validate the standard AgentResult dictionary.
    """

    if isinstance(
        result,
        AgentResult,
    ):

        return True

    if not isinstance(
        result,
        dict,
    ):

        return False

    required_fields = {

        "agent",
        "status",
        "prediction",
        "probability",
        "confidence",
        "risk_score",
        "risk_level",
        "prediction_source",
        "model_status",
        "risk_factors",
        "evidence",
        "explanation",
    }

    if not required_fields.issubset(
        result.keys()
    ):

        return False

    if result.get(
        "status"
    ) not in VALID_STATUSES:

        return False

    if result.get(
        "prediction"
    ) not in VALID_PREDICTIONS:

        return False

    if result.get(
        "risk_level"
    ) not in VALID_RISK_LEVELS:

        return False

    if result.get(
        "prediction_source"
    ) not in VALID_PREDICTION_SOURCES:

        return False

    if result.get(
        "model_status"
    ) not in VALID_MODEL_STATUSES:

        return False

    return True


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [

    # Main class
    "AgentResult",

    # Functions
    "get_risk_level",
    "normalize_risk_score",
    "normalize_probability",
    "normalize_confidence",

    "make_success_result",
    "make_unavailable_result",
    "make_error_result",

    "validate_agent_result",

    # Status
    "STATUS_SUCCESS",
    "STATUS_PARTIAL",
    "STATUS_UNAVAILABLE",
    "STATUS_ERROR",

    # Predictions
    "PREDICTION_LEGITIMATE",
    "PREDICTION_PHISHING",
    "PREDICTION_SUSPICIOUS",
    "PREDICTION_MALICIOUS",
    "PREDICTION_UNKNOWN",

    # Prediction sources
    "PREDICTION_SOURCE_TRAINED_ML",
    "PREDICTION_SOURCE_HEURISTIC",
    "PREDICTION_SOURCE_FALLBACK_HEURISTIC",
    "PREDICTION_SOURCE_UNAVAILABLE",
    "PREDICTION_SOURCE_ERROR",

    # Model statuses
    "MODEL_STATUS_AVAILABLE",
    "MODEL_STATUS_FALLBACK",
    "MODEL_STATUS_UNAVAILABLE",
    "MODEL_STATUS_ERROR",

    # Risk levels
    "RISK_LOW",
    "RISK_MEDIUM",
    "RISK_HIGH",
    "RISK_CRITICAL",
    "RISK_UNKNOWN",

    # Schema
    "AGENT_RESULT_SCHEMA_VERSION",
]