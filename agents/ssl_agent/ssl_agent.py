"""
SSL/TLS Security AI Agent
=========================

Main controller for the SSL/TLS Security AI Agent.

Day 13 goals
------------

This controller provides a standardized AgentResult contract while
preserving the existing SSL/TLS prediction, risk-scoring and
explainability pipeline.

Architecture
------------

Unified Feature Vector
        |
        v
+----------------------+
| SSL Feature Payload  |
+----------------------+
        |
        v
+----------------------+
|   SSLPredictor       |
|   XGBoost / Fallback |
+----------------------+
        |
        +---- prediction
        +---- probability
        +---- confidence
        +---- model status
        +---- prediction source
        |
        v
+----------------------+
|   SSLRiskScorer      |
+----------------------+
        |
        +---- risk score
        +---- risk level
        +---- risk indicators
        +---- contextual observations
        +---- protective indicators
        |
        v
+----------------------+
|    SSLExplainer      |
|      SHAP / XAI      |
+----------------------+
        |
        v
+----------------------+
|      AgentResult     |
|  Standardized Output |
+----------------------+
        |
        v
Decision Fusion Engine


Day 13 standardized fields
---------------------------

Every result must expose:

    agent
    status
    prediction
    probability
    confidence
    risk_score
    risk_level
    prediction_source
    model_status
    risk_factors
    evidence
    explanation

Compatibility fields are retained by AgentResult.to_dict(), including
agent_name, analysis_status and signal_available.


Security principle
------------------

Unavailable SSL evidence is NOT legitimate evidence.

Therefore:

    unavailable
        ->
    prediction = unknown
        ->
    probability = None
        ->
    confidence = 0.0
        ->
    risk_score = None
        ->
    risk_level = unknown
        ->
    signal_available = False

Missing evidence must NEVER be converted into a legitimate/low-risk
signal.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import logging

from datetime import datetime, timezone

from typing import (
    Dict,
    Any,
    Optional,
    List
)

# ============================================================================
# LOCAL IMPORTS
# ============================================================================

from .predictor import SSLPredictor
from .risk_score import SSLRiskScorer
from .explain import SSLExplainer

from agents.agent_result import AgentResult


# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# SSL AI AGENT
# ============================================================================

class SSLAIAgent:
    """
    Main controller for the SSL/TLS Security AI Agent.

    The controller coordinates:

        SSLPredictor
        SSLRiskScorer
        SSLExplainer

    and converts their output into the project-wide AgentResult format.
    """

    # ========================================================================
    # AGENT METADATA
    # ========================================================================

    AGENT_NAME = "SSL_AI_Agent"

    AGENT_VERSION = "2.2.0"

    AGENT_TYPE = "SSL/TLS Security Analysis"

    MODEL_TYPE = "XGBoost"

    FEATURE_KEY = "ssl_features"

    EXPECTED_FEATURE_COUNT = 17

    # ========================================================================
    # CLASS MAPPING
    # ========================================================================

    CLASS_MAPPING = {
        0: "legitimate",
        1: "phishing"
    }

    # ========================================================================
    # STANDARD STATUS VALUES
    # ========================================================================

    STATUS_SUCCESS = "success"

    STATUS_PARTIAL = "partial"

    STATUS_UNAVAILABLE = "unavailable"

    STATUS_ERROR = "error"

    # ========================================================================
    # STANDARD PREDICTION SOURCE VALUES
    # ========================================================================

    SOURCE_TRAINED_ML = "trained_ml"

    SOURCE_HEURISTIC = "heuristic"

    SOURCE_FALLBACK_HEURISTIC = "fallback_heuristic"

    SOURCE_UNAVAILABLE = "unavailable"

    SOURCE_ERROR = "error"

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(
        self,
        model_path: Optional[str] = None
    ) -> None:
        """
        Initialize the SSL AI Agent.

        Args:
            model_path:
                Optional path to the trained SSL model.
        """

        logger.info(
            "Initializing %s version %s...",
            self.AGENT_NAME,
            self.AGENT_VERSION
        )

        self.is_initialized = False

        self.initialization_error: Optional[str] = None

        self.predictor: Optional[SSLPredictor] = None

        self.risk_scorer: Optional[SSLRiskScorer] = None

        self.explainer: Optional[SSLExplainer] = None

        try:

            # =================================================================
            # 1. PREDICTOR
            # =================================================================

            self.predictor = self._initialize_predictor(
                model_path
            )

            logger.info(
                "SSL predictor initialized successfully."
            )

            # =================================================================
            # 2. RISK SCORER
            # =================================================================

            self.risk_scorer = SSLRiskScorer()

            logger.info(
                "SSL risk scorer initialized successfully."
            )

            # =================================================================
            # 3. MODEL FOR EXPLAINER
            # =================================================================

            model = self._get_predictor_model()

            # =================================================================
            # 4. EXPLAINER
            # =================================================================

            self.explainer = SSLExplainer(
                model=model
            )

            logger.info(
                "SSL explainer initialized successfully."
            )

            # =================================================================
            # 5. COMPLETE
            # =================================================================

            self.is_initialized = True

            logger.info(
                "%s initialized successfully.",
                self.AGENT_NAME
            )

        except Exception as exc:

            self.is_initialized = False

            self.initialization_error = str(
                exc
            )

            logger.error(
                "Critical SSL Agent initialization failure: %s",
                str(exc),
                exc_info=True
            )

            raise

    # ========================================================================
    # PREDICTOR INITIALIZATION
    # ========================================================================

    @staticmethod
    def _initialize_predictor(
        model_path: Optional[str]
    ) -> SSLPredictor:
        """
        Initialize SSLPredictor.

        Supports the current predictor API and older constructors.
        """

        try:

            return SSLPredictor(
                model_path=model_path
            )

        except TypeError:

            logger.debug(
                "SSLPredictor does not accept model_path. "
                "Initializing without it."
            )

            return SSLPredictor()

    # ========================================================================
    # MODEL ACCESS
    # ========================================================================

    def _get_predictor_model(
        self
    ) -> Any:
        """
        Safely obtain the underlying trained model.
        """

        if self.predictor is None:

            return None

        try:

            method = getattr(
                self.predictor,
                "get_model",
                None
            )

            if callable(method):

                return method()

        except Exception as exc:

            logger.warning(
                "Unable to obtain SSL model: %s",
                str(exc)
            )

        return None

    # ========================================================================
    # HEALTH CHECK
    # ========================================================================

    def health_check(
        self
    ) -> Dict[str, Any]:
        """
        Return detailed SSL Agent health information.
        """

        predictor_available = (
            self.predictor is not None
        )

        risk_scorer_available = (
            self.risk_scorer is not None
        )

        explainer_available = (
            self.explainer is not None
        )

        model_available = False

        if predictor_available:

            try:

                model_available = bool(
                    getattr(
                        self.predictor,
                        "is_model_loaded",
                        False
                    )
                )

                if not model_available:

                    model = self._get_predictor_model()

                    model_available = (
                        model is not None
                    )

            except Exception:

                model_available = False

        shap_available = False

        if explainer_available:

            try:

                shap_available = bool(
                    getattr(
                        self.explainer,
                        "explainer",
                        None
                    )
                )

            except Exception:

                shap_available = False

        healthy = (
            self.is_initialized
            and predictor_available
            and risk_scorer_available
            and explainer_available
        )

        return {

            "agent_name":
                self.AGENT_NAME,

            "agent_version":
                self.AGENT_VERSION,

            "initialized":
                self.is_initialized,

            "initialization_error":
                self.initialization_error,

            "predictor_available":
                predictor_available,

            "risk_scorer_available":
                risk_scorer_available,

            "explainer_available":
                explainer_available,

            "model_available":
                model_available,

            "shap_explainer_ready":
                shap_available,

            "status":
                "healthy"
                if healthy
                else "degraded"
        }

    # ========================================================================
    # STATUS
    # ========================================================================

    def get_status(
        self
    ) -> Dict[str, Any]:
        """
        Return integration-friendly status information.
        """

        health = self.health_check()

        predictor_status = {}

        if self.predictor is not None:

            try:

                predictor_status = (
                    self.predictor.get_status()
                )

            except Exception as exc:

                predictor_status = {

                    "status":
                        "error",

                    "error":
                        str(exc)
                }

        feature_count = (
            predictor_status.get(
                "feature_count",
                17
            )
        )

        model_loaded = bool(
            predictor_status.get(
                "model_loaded",
                getattr(
                    self.predictor,
                    "is_model_loaded",
                    False
                )
            )
        )

        model_path = (
            predictor_status.get(
                "model_path"
            )
        )

        if model_path is None and self.predictor is not None:

            model_path = getattr(
                self.predictor,
                "model_path",
                None
            )

        model_exists = bool(
            predictor_status.get(
                "model_exists",
                model_loaded
            )
        )

        return {

            "agent_name":
                self.AGENT_NAME,

            "agent_version":
                self.AGENT_VERSION,

            "agent_type":
                self.AGENT_TYPE,

            "model_type":
                self.MODEL_TYPE,

            "initialized":
                self.is_initialized,

            "model_available":
                bool(
                    health.get(
                        "model_available",
                        False
                    )
                ),

            "model_loaded":
                model_loaded,

            "model_ready":
                model_loaded,

            "model_path":
                model_path,

            "model_exists":
                model_exists,

            "feature_count":
                feature_count,

            "predictor_available":
                health.get(
                    "predictor_available",
                    False
                ),

            "risk_scorer_available":
                health.get(
                    "risk_scorer_available",
                    False
                ),

            "explainer_available":
                health.get(
                    "explainer_available",
                    False
                ),

            "shap_explainer_ready":
                health.get(
                    "shap_explainer_ready",
                    False
                ),

            "status":
                health.get(
                    "status",
                    "unknown"
                ),

            "initialization_error":
                self.initialization_error
        }

    # ========================================================================
    # AGENT INFORMATION
    # ========================================================================

    def get_agent_info(
        self
    ) -> Dict[str, Any]:
        """
        Return SSL Agent metadata and component information.
        """

        return {

            "agent_name":
                self.AGENT_NAME,

            "agent_version":
                self.AGENT_VERSION,

            "agent_type":
                self.AGENT_TYPE,

            "model_type":
                self.MODEL_TYPE,

            "feature_key":
                self.FEATURE_KEY,

            "class_mapping":
                self.CLASS_MAPPING.copy(),

            "components":
                {

                    "predictor":
                        (
                            type(
                                self.predictor
                            ).__name__
                            if self.predictor is not None
                            else None
                        ),

                    "risk_scorer":
                        (
                            type(
                                self.risk_scorer
                            ).__name__
                            if self.risk_scorer is not None
                            else None
                        ),

                    "explainer":
                        (
                            type(
                                self.explainer
                            ).__name__
                            if self.explainer is not None
                            else None
                        )
                },

            "health":
                self.health_check()
        }

    # ========================================================================
    # MAIN ANALYSIS
    # ========================================================================

    def analyze(
        self,
        unified_vector: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the complete SSL/TLS analysis pipeline.

        Expected input:

            {
                "ssl_features": {...},
                "metadata": {
                    "target_url": "https://example.com"
                }
            }

        Standard output:

            AgentResult.to_dict()

        Security guarantee:

            Missing SSL evidence
                ↓
            unavailable
                ↓
            unknown
                ↓
            risk_score=None
                ↓
            signal_available=False
        """

        logger.info(
            "%s analysis triggered.",
            self.AGENT_NAME
        )

        # ====================================================================
        # PHASE 1 — INITIALIZATION
        # ====================================================================

        if not self.is_initialized:

            return self._get_unavailable_response(
                reason=(
                    "SSL AI Agent is not initialized."
                ),
                target_url="unknown_url"
            )

        # ====================================================================
        # PHASE 2 — INPUT VALIDATION
        # ====================================================================

        if not isinstance(
            unified_vector,
            dict
        ):

            return self._get_unavailable_response(
                reason=(
                    "Invalid unified vector. "
                    "Expected dictionary."
                ),
                target_url="unknown_url"
            )

        # ====================================================================
        # PHASE 3 — TARGET URL
        # ====================================================================

        target_url = self._extract_target_url(
            unified_vector
        )

        # ====================================================================
        # PHASE 4 — SSL FEATURES
        # ====================================================================

        ssl_features = self._extract_ssl_features(
            unified_vector
        )

        # ====================================================================
        # PHASE 5 — MISSING SSL FEATURES
        # ====================================================================

        if not isinstance(
            ssl_features,
            dict
        ) or not ssl_features:

            reason = (
                "Required SSL feature block "
                "'ssl_features' is missing or invalid."
            )

            logger.warning(
                "SSL feature block missing for %s.",
                target_url
            )

            return self._get_unavailable_response(
                reason=reason,
                target_url=target_url
            )

        # ====================================================================
        # PHASE 6 — COLLECTION FAILURE
        # ====================================================================

        collection_failure = (
            self._detect_feature_collection_failure(
                ssl_features
            )
        )

        if collection_failure is not None:

            logger.warning(
                "SSL collection failure for %s: %s",
                target_url,
                collection_failure
            )

            return self._get_unavailable_response(
                reason=collection_failure,
                target_url=target_url
            )

        # ====================================================================
        # PHASE 7 — PREDICTOR
        # ====================================================================

        if self.predictor is None:

            return self._get_unavailable_response(
                reason=(
                    "SSL Predictor is unavailable."
                ),
                target_url=target_url
            )

        try:

            prediction_result = (
                self.predictor.predict(
                    ssl_features
                )
            )

        except Exception as exc:

            logger.error(
                "SSL Predictor failed: %s",
                str(exc),
                exc_info=True
            )

            return self._get_error_response(
                reason=(
                    f"SSL Predictor failed: {exc}"
                ),
                target_url=target_url
            )

        if not isinstance(
            prediction_result,
            dict
        ):

            return self._get_error_response(
                reason=(
                    "SSL Predictor returned "
                    "an invalid response."
                ),
                target_url=target_url
            )

        # ====================================================================
        # PHASE 8 — PREDICTOR SIGNAL AVAILABILITY
        # ====================================================================

        predictor_signal = (
            prediction_result.get(
                "signal_available",
                True
            )
        )

        prediction = (
            self._normalize_prediction(
                prediction_result.get(
                    "prediction",
                    prediction_result.get(
                        "class_label",
                        "unknown"
                    )
                )
            )
        )

        # Predictor can explicitly report unavailable/error.
        predictor_status = str(
            prediction_result.get(
                "model_status",
                ""
            )
            or ""
        ).strip().lower()

        predictor_source = str(
            prediction_result.get(
                "prediction_source",
                ""
            )
            or ""
        ).strip().lower()

        if predictor_signal is False:

            reason = str(
                prediction_result.get(
                    "error",
                    prediction_result.get(
                        "reason",
                        "SSL Predictor signal unavailable."
                    )
                )
            )

            return self._get_unavailable_response(
                reason=reason,
                target_url=target_url,
                prediction_result=prediction_result
            )

        if prediction in {
            "unknown",
            "unavailable"
        }:

            return self._get_unavailable_response(
                reason=(
                    "SSL Predictor did not produce "
                    "a usable prediction."
                ),
                target_url=target_url,
                prediction_result=prediction_result
            )

        # ====================================================================
        # PHASE 9 — PROBABILITY
        # ====================================================================

        phishing_probability = (
            self._extract_probability(
                prediction_result
            )
        )

        if phishing_probability is None:

            return self._get_unavailable_response(
                reason=(
                    "SSL Predictor did not return "
                    "a usable phishing probability."
                ),
                target_url=target_url,
                prediction_result=prediction_result
            )

        # ====================================================================
        # PHASE 10 — CONFIDENCE
        # ====================================================================

        confidence = (
            self._extract_confidence(
                prediction_result,
                phishing_probability
            )
        )

        # ====================================================================
        # PHASE 11 — MODEL SOURCE
        # ====================================================================

        prediction_source = (
            self._normalize_prediction_source(
                prediction_result
            )
        )

        model_status = (
            self._normalize_model_status(
                prediction_result.get(
                    "model_status"
                ),
                prediction_source
            )
        )

        # ====================================================================
        # PHASE 12 — RISK SCORING
        # ====================================================================

        if self.risk_scorer is None:

            return self._get_unavailable_response(
                reason=(
                    "SSL Risk Scorer is unavailable."
                ),
                target_url=target_url,
                prediction_result=prediction_result
            )

        try:

            risk_assessment = (
                self.risk_scorer.calculate_risk(
                    probability=phishing_probability,
                    features=ssl_features
                )
            )

        except Exception as exc:

            logger.error(
                "SSL Risk Scorer failed: %s",
                str(exc),
                exc_info=True
            )

            return self._get_error_response(
                reason=(
                    f"SSL Risk Scorer failed: {exc}"
                ),
                target_url=target_url
            )

        if not isinstance(
            risk_assessment,
            dict
        ):

            return self._get_error_response(
                reason=(
                    "SSL Risk Scorer returned "
                    "an invalid response."
                ),
                target_url=target_url
            )

        # ====================================================================
        # PHASE 13 — RISK SIGNAL AVAILABILITY
        # ====================================================================

        signal_available = (
            risk_assessment.get(
                "signal_available",
                False
            )
        )

        if signal_available is not True:

            reason = str(
                risk_assessment.get(
                    "reason",
                    risk_assessment.get(
                        "scoring_explanation",
                        "SSL risk signal unavailable."
                    )
                )
            )

            return self._get_unavailable_response(
                reason=reason,
                target_url=target_url,
                prediction_result=prediction_result
            )

        # ====================================================================
        # PHASE 14 — RISK SCORE
        # ====================================================================

        risk_score = self._safe_int(
            risk_assessment.get(
                "risk_score"
            ),
            default=-1
        )

        if not 0 <= risk_score <= 100:

            return self._get_error_response(
                reason=(
                    "SSL Risk Scorer returned "
                    "an invalid risk score."
                ),
                target_url=target_url
            )

        # ====================================================================
        # PHASE 15 — STANDARD RISK LEVEL
        # ====================================================================

        risk_level = (
            self._get_risk_level(
                risk_score
            )
        )

        # ====================================================================
        # PHASE 16 — INDICATORS
        # ====================================================================

        risk_indicators = (
            self._ensure_list(
                risk_assessment.get(
                    "risk_indicators",
                    []
                )
            )
        )

        contextual_observations = (
            self._ensure_list(
                risk_assessment.get(
                    "contextual_observations",
                    []
                )
            )
        )

        protective_indicators = (
            self._ensure_list(
                risk_assessment.get(
                    "protective_indicators",
                    []
                )
            )
        )

        triggered_indicators = (
            self._ensure_list(
                risk_assessment.get(
                    "triggered_indicators",
                    risk_indicators
                )
            )
        )

        indicator_details = (
            self._ensure_list(
                risk_assessment.get(
                    "indicator_details",
                    []
                )
            )
        )

        # ====================================================================
        # PHASE 17 — EXPLANATION
        # ====================================================================

        explanation = (
            self._generate_explanation(
                ssl_features=ssl_features,
                prediction=prediction
            )
        )

        # ====================================================================
        # PHASE 18 — EVIDENCE
        # ====================================================================

        evidence = (
            self._build_evidence(
                target_url=target_url,
                prediction_result=prediction_result,
                risk_assessment=risk_assessment,
                prediction_source=prediction_source,
                model_status=model_status
            )
        )

        # ====================================================================
        # PHASE 19 — RISK FACTORS
        # ====================================================================

        risk_factors = (
            self._build_risk_factors(
                risk_indicators=risk_indicators,
                contextual_observations=contextual_observations,
                triggered_indicators=triggered_indicators,
                explanation=explanation
            )
        )

        # ====================================================================
        # PHASE 20 — RECOMMENDATIONS
        # ====================================================================

        recommendations = (
            self._generate_recommendations(
                ssl_features,
                risk_assessment
            )
        )

        # ====================================================================
        # PHASE 21 — METADATA
        # ====================================================================

        metadata = {

            "target_url":
                target_url,

            "agent_version":
                self.AGENT_VERSION,

            "agent_type":
                self.AGENT_TYPE,

            "feature_key":
                self.FEATURE_KEY,

            "feature_count":
                self._get_feature_count(
                    ssl_features
                ),

            "expected_feature_count":
                self._get_expected_feature_count(),

            "class_mapping":
                {
                    "0":
                        "legitimate",

                    "1":
                        "phishing"
                },

            "model_type":
                self.MODEL_TYPE,

            "fallback_used":
                bool(
                    prediction_source
                    == self.SOURCE_FALLBACK_HEURISTIC
                ),

            "risk_assessment":
                risk_assessment,

            "recommendations":
                recommendations,

            "pipeline_status":
                {

                    "feature_collection":
                        "completed",

                    "prediction":
                        "completed",

                    "risk_scoring":
                        "completed",

                    "explainability":
                        (
                            "completed"
                            if explanation
                            else "limited"
                        )
                }
        }

        # ====================================================================
        # PHASE 22 — STANDARD AgentResult
        # ====================================================================

        result = AgentResult.success(

            agent=self.AGENT_NAME,

            prediction=prediction,

            probability=phishing_probability,

            confidence=confidence,

            risk_score=risk_score,

            prediction_source=prediction_source,

            model_status=model_status,

            risk_factors=risk_factors,

            evidence=evidence,

            explanation=explanation,

            metadata=metadata
        )

        result_dict = result.to_dict()

        # ====================================================================
        # ADD SSL-SPECIFIC COMPATIBILITY INFORMATION
        # ====================================================================

        result_dict.update({

            "agent_version":
                self.AGENT_VERSION,

            "agent_type":
                self.AGENT_TYPE,

            "target_url":
                target_url,

            "timestamp":
                self._get_timestamp(),

            "verdict":
                prediction,

            "class_index":
                prediction_result.get(
                    "class_index"
                ),

            "confidence_score":
                confidence,

            "phishing_probability":
                phishing_probability,

            "legitimate_probability":
                self._extract_legitimate_probability(
                    prediction_result,
                    phishing_probability
                ),

            "class_probabilities":
                self._sanitize_class_probabilities(
                    prediction_result.get(
                        "class_probabilities",
                        {}
                    ),
                    phishing_probability
                ),

            "risk_assessment":
                risk_assessment,

            "recommendations":
                recommendations,

            "pipeline_status":
                metadata.get(
                    "pipeline_status",
                    {}
                )
        })

        logger.info(
            "SSL analysis completed | "
            "URL=%s | Prediction=%s | "
            "Risk=%s | Score=%s | Source=%s",
            target_url,
            prediction,
            risk_level,
            risk_score,
            prediction_source
        )

        return result_dict

    # ========================================================================
    # TARGET URL
    # ========================================================================

    @staticmethod
    def _extract_target_url(
        unified_vector: Dict[str, Any]
    ) -> str:
        """
        Safely resolve the target URL.
        """

        metadata = unified_vector.get(
            "metadata",
            {}
        )

        if not isinstance(
            metadata,
            dict
        ):

            metadata = {}

        candidates = [

            metadata.get(
                "target_url"
            ),

            unified_vector.get(
                "target_url"
            ),

            unified_vector.get(
                "url"
            ),

            metadata.get(
                "url"
            )
        ]

        for candidate in candidates:

            if candidate is None:

                continue

            value = str(
                candidate
            ).strip()

            if value:

                return value

        return "unknown_url"

    # ========================================================================
    # SSL FEATURES
    # ========================================================================

    @staticmethod
    def _extract_ssl_features(
        unified_vector: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract SSL feature block.

        Supported:

            ssl_features

        Backward compatibility:

            ssl
        """

        if "ssl_features" in unified_vector:

            return unified_vector.get(
                "ssl_features"
            )

        if "ssl" in unified_vector:

            return unified_vector.get(
                "ssl"
            )

        return None

    # ========================================================================
    # COLLECTION FAILURE DETECTION
    # ========================================================================

    @staticmethod
    def _detect_feature_collection_failure(
        features: Dict[str, Any]
    ) -> Optional[str]:
        """
        Detect explicit SSL collection/extraction failures.

        Missing features are not automatically errors because the
        SSL preprocessor may legitimately handle optional fields.

        Explicit collection failures are errors/unavailability.
        """

        if not isinstance(
            features,
            dict
        ):

            return (
                "SSL feature payload is not a dictionary."
            )

        error_keys = {

            "error",
            "errors",
            "exception",
            "collection_error",
            "extraction_error",
            "error_message"
        }

        for key in error_keys:

            if key not in features:

                continue

            value = features.get(
                key
            )

            if value:

                return (
                    f"SSL extraction error: {value}"
                )

        if features.get(
            "success"
        ) is False:

            return (
                "SSL feature extraction "
                "reported success=False."
            )

        if features.get(
            "signal_available"
        ) is False:

            return (
                "SSL signal was explicitly "
                "marked unavailable."
            )

        status = str(
            features.get(
                "status",
                ""
            )
            or ""
        ).strip().lower()

        if status in {

            "error",
            "failed",
            "failure",
            "timeout",
            "unavailable"
        }:

            return (
                f"SSL extraction status is '{status}'."
            )

        return None

    # ========================================================================
    # EXPLANATION
    # ========================================================================

    def _generate_explanation(
        self,
        ssl_features: Dict[str, Any],
        prediction: str
    ) -> Dict[str, Any]:
        """
        Generate SSL explanation.

        Explanation failure does not invalidate an otherwise valid
        SSL prediction.
        """

        if self.explainer is None:

            return {

                "summary":
                    (
                        "SSL analysis completed "
                        "without an explanation engine."
                    ),

                "method":
                    "none",

                "top_factors":
                    [],

                "risk_factors":
                    []
            }

        try:

            explanation = (
                self.explainer.generate_explanation(
                    features=ssl_features,
                    prediction=prediction
                )
            )

            if not isinstance(
                explanation,
                dict
            ):

                return {

                    "summary":
                        "No explanation available.",

                    "method":
                        "none",

                    "top_factors":
                        [],

                    "risk_factors":
                        []
                }

            normalized = dict(
                explanation
            )

            if not isinstance(
                normalized.get(
                    "top_factors"
                ),
                list
            ):

                normalized[
                    "top_factors"
                ] = []

            if "summary" not in normalized:

                normalized[
                    "summary"
                ] = (
                    "SSL analysis completed."
                )

            if "method" not in normalized:

                normalized[
                    "method"
                ] = (
                    "SHAP"
                    if getattr(
                        self.explainer,
                        "explainer",
                        None
                    ) is not None
                    else "heuristic"
                )

            return normalized

        except Exception as exc:

            logger.warning(
                "SSL explanation generation failed: %s",
                str(exc)
            )

            return {

                "summary":
                    (
                        "SSL analysis completed, "
                        "but explanation generation failed."
                    ),

                "method":
                    "none",

                "top_factors":
                    [],

                "risk_factors":
                    [],

                "error":
                    str(exc)
            }

    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================

    def _generate_recommendations(
        self,
        features: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """
        Generate SSL recommendations when supported by the explainer.
        """

        recommendations: List[str] = []

        if self.explainer is not None:

            try:

                method = getattr(
                    self.explainer,
                    "generate_recommendations",
                    None
                )

                if callable(method):

                    generated = method(
                        features
                    )

                    if isinstance(
                        generated,
                        list
                    ):

                        recommendations.extend(
                            str(item)
                            for item in generated
                        )

            except Exception as exc:

                logger.debug(
                    "SSL recommendation generation failed: %s",
                    str(exc)
                )

        # --------------------------------------------------------------------
        # Risk-based fallback recommendations
        # --------------------------------------------------------------------

        if not recommendations:

            risk_indicators = (
                self._ensure_list(
                    risk_assessment.get(
                        "risk_indicators",
                        []
                    )
                )
            )

            if risk_indicators:

                recommendations.append(
                    (
                        "Review the SSL/TLS risk indicators "
                        "before trusting the target."
                    )
                )

        return recommendations

    # ========================================================================
    # RISK FACTORS
    # ========================================================================

    @staticmethod
    def _build_risk_factors(
        risk_indicators: List[Any],
        contextual_observations: List[Any],
        triggered_indicators: List[Any],
        explanation: Dict[str, Any]
    ) -> List[Any]:
        """
        Combine SSL risk factors without duplicates.
        """

        factors: List[Any] = []

        sources = [

            risk_indicators,

            contextual_observations,

            triggered_indicators,

            explanation.get(
                "risk_factors",
                []
            )
            if isinstance(
                explanation,
                dict
            )
            else []
        ]

        for source in sources:

            if not isinstance(
                source,
                list
            ):

                continue

            for factor in source:

                if factor not in factors:

                    factors.append(
                        factor
                    )

        return factors

    # ========================================================================
    # EVIDENCE
    # ========================================================================

    def _build_evidence(
        self,
        target_url: str,
        prediction_result: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        prediction_source: str,
        model_status: str
    ) -> List[Any]:
        """
        Build compact evidence for Fusion/UI.
        """

        feature_count = (
            self._get_feature_count(
                prediction_result.get(
                    "features",
                    {}
                )
            )
        )

        evidence: List[Any] = []

        evidence.append({

            "type":
                "analysis_target",

            "target":
                target_url
        })

        evidence.append({

            "type":
                "model_signal",

            "model_type":
                self.MODEL_TYPE,

            "model_status":
                model_status,

            "prediction_source":
                prediction_source
        })

        evidence.append({

            "type":
                "risk_assessment",

            "risk_score":
                risk_assessment.get(
                    "risk_score"
                ),

            "risk_level":
                self._get_risk_level(
                    self._safe_int(
                        risk_assessment.get(
                            "risk_score"
                        ),
                        default=0
                    )
                )
                if risk_assessment.get(
                    "risk_score"
                ) is not None
                else "unknown",

            "signal_available":
                bool(
                    risk_assessment.get(
                        "signal_available",
                        False
                    )
                )
        })

        if feature_count > 0:

            evidence.append({

                "type":
                    "feature_schema",

                "feature_count":
                    feature_count,

                "expected_feature_count":
                    self._get_expected_feature_count(),

                "schema_complete":
                    (
                        feature_count
                        == self._get_expected_feature_count()
                    )
            })

        return evidence

    # ========================================================================
    # UNAVAILABLE RESPONSE
    # ========================================================================

        # ========================================================================
    # UNAVAILABLE RESPONSE
    # ========================================================================

    def _get_unavailable_response(
        self,
        reason: str,
        target_url: str,
        prediction_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build a standardized unavailable AgentResult.

        SECURITY RULE
        -------------

        unavailable != legitimate

        Therefore an unavailable SSL signal MUST produce:

            status = "unavailable"
            prediction = "unknown"
            probability = None
            confidence = 0.0
            risk_score = None
            risk_level = "unknown"
            signal_available = False

        Missing SSL evidence must NEVER become:

            prediction = "legitimate"
            risk_score = 0
            signal_available = True

        The canonical AgentResult object is the source of truth.
        Legacy SSL fields are retained only for compatibility with the
        current Orchestrator/Fusion code.
        """

        prediction_result = (
            prediction_result
            if isinstance(
                prediction_result,
                dict
            )
            else {}
        )

        # ====================================================================
        # STANDARD AgentResult
        # ====================================================================

        result = AgentResult.unavailable(

            agent=self.AGENT_NAME,

            reason=str(
                reason
            ),

            execution_time_ms=None,

            feature_key=getattr(
                self,
                "FEATURE_KEY",
                "ssl_features"
            ),

            metadata={

                "target_url":
                    target_url,

                "agent_version":
                    self.AGENT_VERSION,

                "agent_type":
                    self.AGENT_TYPE,

                "model_type":
                    self.MODEL_TYPE,

                "expected_feature_count":
                    getattr(
                        self,
                        "EXPECTED_FEATURE_COUNT",
                        17
                    ),

                "reason":
                    str(reason),

                "prediction_result":
                    prediction_result,
            }
        )

        standard_result = result.to_dict()

        # ====================================================================
        # LEGACY COMPATIBILITY
        # ====================================================================
        #
        # The current Orchestrator/Fusion still reads fields such as:
        #
        #   agent_name
        #   analysis_status
        #   signal_available
        #
        # Therefore we retain them until the Fusion migration is completed.
        #
        # These fields MUST agree with AgentResult.
        # ====================================================================

        standard_result.update({

            # ----------------------------------------------------------------
            # Identity
            # ----------------------------------------------------------------

            "agent":
                self.AGENT_NAME,

            "agent_name":
                self.AGENT_NAME,

            "agent_version":
                self.AGENT_VERSION,

            "agent_type":
                self.AGENT_TYPE,

            "target_url":
                target_url,

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            # ----------------------------------------------------------------
            # Execution state
            # ----------------------------------------------------------------

            "status":
                "unavailable",

            "analysis_status":
                "unavailable",

            "signal_available":
                False,

            # ----------------------------------------------------------------
            # Prediction
            # ----------------------------------------------------------------

            "prediction":
                "unknown",

            "verdict":
                "unknown",

            "class_index":
                None,

            "probability":
                None,

            "confidence":
                0.0,

            "confidence_score":
                0.0,

            "phishing_probability":
                None,

            "legitimate_probability":
                None,

            "class_probabilities":
                {
                    "legitimate":
                        None,

                    "phishing":
                        None,
                },

            # ----------------------------------------------------------------
            # Risk
            # ----------------------------------------------------------------

            "risk_score":
                None,

            "risk_level":
                "unknown",

            "risk_assessment":
                {
                    "signal_available":
                        False,

                    "risk_score":
                        None,

                    "risk_level":
                        "unknown",

                    "base_probability":
                        None,

                    "contextual_adjustment":
                        None,

                    "final_probability":
                        None,

                    "risk_indicators":
                        [],

                    "contextual_observations":
                        [],

                    "protective_indicators":
                        [],

                    "triggered_indicators":
                        [],

                    "indicator_details":
                        [],

                    "severity_summary":
                        {
                            "critical":
                                0,

                            "high":
                                0,

                            "medium":
                                0,

                            "low":
                                0,

                            "risk_indicator_count":
                                0,

                            "contextual_observation_count":
                                0,

                            "total_indicators":
                                0,
                        },

                    "scoring_method":
                        "unavailable",

                    "scoring_explanation":
                        (
                            "SSL risk scoring was not performed "
                            "because the SSL signal was unavailable."
                        ),
                },

            # ----------------------------------------------------------------
            # Model information
            # ----------------------------------------------------------------

            "prediction_source":
                "unavailable",

            "model_status":
                "unavailable",

            "prediction_method":
                "unavailable",

            "model_based":
                False,

            "model":
                {
                    "status":
                        "unavailable",

                    "prediction_source":
                        "unavailable",

                    "model_based":
                        False,

                    "fallback_used":
                        False,

                    "model_loaded":
                        False,
                },

            # ----------------------------------------------------------------
            # Features
            # ----------------------------------------------------------------

            "feature_count":
                0,

            "features_evaluated_count":
                0,

            "feature_validation":
                {
                    "schema_complete":
                        False,

                    "supplied_feature_count":
                        0,

                    "expected_feature_count":
                        getattr(
                            self,
                            "EXPECTED_FEATURE_COUNT",
                            17
                        ),
                },

            # ----------------------------------------------------------------
            # Explainability
            # ----------------------------------------------------------------

            "risk_factors":
                [],

            "evidence":
                [],

            "risk_indicators":
                [],

            "contextual_observations":
                [],

            "protective_indicators":
                [],

            "explanation":
                {
                    "summary":
                        (
                            "SSL analysis could not provide "
                            "a usable security signal."
                        ),

                    "method":
                        "none",

                    "top_shap_factors":
                        [],

                    "top_factors":
                        [],

                    "risk_factors":
                        [],

                    "protective_factors":
                        [],

                    "shap_status":
                        "unavailable",

                    "explanation_source":
                        "none",
                },

            # ----------------------------------------------------------------
            # Error / reason
            # ----------------------------------------------------------------

            "reason":
                str(reason),

            "error":
                None,

            "error_type":
                None,

            "error_details":
                str(reason),

            # ----------------------------------------------------------------
            # Execution metadata
            # ----------------------------------------------------------------

            "execution_time_ms":
                standard_result.get(
                    "execution_time_ms"
                ),

            "feature_key":
                getattr(
                    self,
                    "FEATURE_KEY",
                    "ssl_features"
                ),
        })

        logger.warning(
            (
                "SSL Agent unavailable | "
                "URL=%s | Reason=%s"
            ),
            target_url,
            reason,
        )

        return standard_result

    # ========================================================================
    # ERROR RESPONSE
    # ========================================================================

    def _get_error_response(
        self,
        reason: str,
        target_url: str
    ) -> Dict[str, Any]:
        """
        Build a standardized error AgentResult.

        Errors are never converted into legitimate predictions.
        """

        explanation = {

            "summary":
                (
                    "SSL analysis encountered "
                    "an execution error."
                ),

            "method":
                "none",

            "top_factors":
                [],

            "risk_factors":
                []
        }

        evidence = [

            {

                "type":
                    "analysis_target",

                "target":
                    target_url
            },

            {

                "type":
                    "execution_error",

                "message":
                    str(reason)
            }
        ]

        metadata = {

            "target_url":
                target_url,

            "agent_version":
                self.AGENT_VERSION,

            "agent_type":
                self.AGENT_TYPE,

            "feature_key":
                self.FEATURE_KEY
        }

        result = AgentResult.error_result(

            agent=self.AGENT_NAME,

            error=str(reason),

            error_type="SSLAnalysisError",

            risk_factors=[],

            evidence=evidence,

            explanation=explanation,

            metadata=metadata
        )

        result_dict = result.to_dict()

        result_dict.update({

            "agent_version":
                self.AGENT_VERSION,

            "agent_type":
                self.AGENT_TYPE,

            "target_url":
                target_url,

            "timestamp":
                self._get_timestamp(),

            "verdict":
                "unknown",

            "class_index":
                None,

            "confidence_score":
                0.0,

            "phishing_probability":
                None,

            "legitimate_probability":
                None,

            "class_probabilities":
                {

                    "legitimate":
                        None,

                    "phishing":
                        None
                },

            "risk_assessment":
                {

                    "signal_available":
                        False,

                    "risk_score":
                        None,

                    "risk_level":
                        "unknown"
                },

            "pipeline_status":
                {

                    "feature_collection":
                        "error",

                    "prediction":
                        "error",

                    "risk_scoring":
                        "not_completed",

                    "explainability":
                        "not_completed"
                },

            "reason":
                str(reason),

            "error_details":
                str(reason)
        })

        return result_dict

    # ========================================================================
    # PREDICTION NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_prediction(
        value: Any
    ) -> str:
        """
        Normalize SSL prediction labels.
        """

        normalized = str(
            value or "unknown"
        ).strip().lower()

        if normalized in {

            "legitimate",
            "safe",
            "benign",
            "normal"
        }:

            return "legitimate"

        if normalized in {

            "phishing",
            "suspicious",
            "malicious",
            "unsafe",
            "fraudulent"
        }:

            return "phishing"

        return "unknown"

    # ========================================================================
    # PREDICTION SOURCE NORMALIZATION
    # ========================================================================

    @classmethod
    def _normalize_prediction_source(
        cls,
        prediction_result: Dict[str, Any]
    ) -> str:
        """
        Normalize predictor source into the project-wide Day 13 values.
        """

        fallback_used = bool(
            prediction_result.get(
                "fallback_used",
                False
            )
        )

        if fallback_used:

            return cls.SOURCE_FALLBACK_HEURISTIC

        source = str(
            prediction_result.get(
                "prediction_source",
                ""
            )
            or ""
        ).strip().lower()

        model_based = bool(
            prediction_result.get(
                "model_based",
                False
            )
        )

        if source in {

            "xgboost",
            "xgb",
            "trained",
            "trained_ml",
            "ml",
            "machine_learning",
            "loaded_ml_model"
        }:

            return cls.SOURCE_TRAINED_ML

        if source in {

            "heuristic",
            "rule_based",
            "rules"
        }:

            return cls.SOURCE_HEURISTIC

        if source in {

            "fallback",
            "heuristic_fallback",
            "fallback_heuristic",
            "fallback_rule_based"
        }:

            return cls.SOURCE_FALLBACK_HEURISTIC

        if source in {

            "error",
            "failed"
        }:

            return cls.SOURCE_ERROR

        if source in {

            "unavailable",
            "none",
            "unknown",
            ""
        }:

            if model_based:

                return cls.SOURCE_TRAINED_ML

            return cls.SOURCE_UNAVAILABLE

        if model_based:

            return cls.SOURCE_TRAINED_ML

        logger.warning(
            "Unknown SSL prediction source '%s'. "
            "Marking as unavailable.",
            source
        )

        return cls.SOURCE_UNAVAILABLE

    # ========================================================================
    # MODEL STATUS NORMALIZATION
    # ========================================================================

    @classmethod
    def _normalize_model_status(
        cls,
        value: Any,
        prediction_source: str
    ) -> str:
        """
        Normalize model status into:

            available
            fallback
            unavailable
            error
        """

        status = str(
            value or ""
        ).strip().lower()

        if prediction_source == cls.SOURCE_TRAINED_ML:

            return "available"

        if prediction_source == cls.SOURCE_FALLBACK_HEURISTIC:

            return "fallback"

        if prediction_source == cls.SOURCE_HEURISTIC:

            return "fallback"

        if prediction_source == cls.SOURCE_ERROR:

            return "error"

        if status in {

            "loaded",
            "ready",
            "available",
            "loaded_model",
            "loaded_ml_model",
            "xgboost",
            "model_loaded"
        }:

            return "available"

        if status in {

            "fallback",
            "heuristic",
            "heuristic_fallback"
        }:

            return "fallback"

        if status in {

            "error",
            "failed",
            "error_fallback"
        }:

            return "error"

        if status in {

            "unavailable",
            "none",
            "unknown",
            "model_unavailable",
            ""
        }:

            return "unavailable"

        return status

    # ========================================================================
    # PROBABILITY
    # ========================================================================

    @staticmethod
    def _extract_probability(
        prediction_result: Dict[str, Any]
    ) -> Optional[float]:
        """
        Extract phishing probability safely.
        """

        candidates = [

            prediction_result.get(
                "phishing_probability"
            ),

            prediction_result.get(
                "probability"
            ),

            prediction_result.get(
                "phishing_prob"
            )
        ]

        for value in candidates:

            if value is None:

                continue

            try:

                probability = float(
                    value
                )

            except (
                TypeError,
                ValueError,
                OverflowError
            ):

                continue

            if probability != probability:

                continue

            if probability < 0.0:

                probability = 0.0

            if probability > 1.0:

                probability = 1.0

            return probability

        return None

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    @staticmethod
    def _extract_confidence(
        prediction_result: Dict[str, Any],
        phishing_probability: float
    ) -> float:
        """
        Extract confidence safely.

        If the predictor provides confidence, use it.
        Otherwise derive confidence from the class probability.
        """

        value = prediction_result.get(
            "confidence"
        )

        if value is None:

            value = prediction_result.get(
                "confidence_score"
            )

        if value is not None:

            try:

                confidence = float(
                    value
                )

                if confidence == confidence:

                    return max(
                        0.0,
                        min(
                            1.0,
                            confidence
                        )
                    )

            except (
                TypeError,
                ValueError,
                OverflowError
            ):

                pass

        # Confidence = probability of predicted class.
        prediction = SSLAIAgent._normalize_prediction(
            prediction_result.get(
                "prediction"
            )
        )

        if prediction == "phishing":

            return round(
                phishing_probability,
                4
            )

        if prediction == "legitimate":

            return round(
                1.0 - phishing_probability,
                4
            )

        return 0.0

    # ========================================================================
    # LEGITIMATE PROBABILITY
    # ========================================================================

    @staticmethod
    def _extract_legitimate_probability(
        prediction_result: Dict[str, Any],
        phishing_probability: Optional[float]
    ) -> Optional[float]:
        """
        Extract legitimate probability.
        """

        value = prediction_result.get(
            "legitimate_probability"
        )

        if value is not None:

            try:

                probability = float(
                    value
                )

                return max(
                    0.0,
                    min(
                        1.0,
                        probability
                    )
                )

            except (
                TypeError,
                ValueError,
                OverflowError
            ):

                pass

        if phishing_probability is None:

            return None

        return round(
            1.0 - phishing_probability,
            4
        )

    # ========================================================================
    # CLASS PROBABILITIES
    # ========================================================================

    @staticmethod
    def _sanitize_class_probabilities(
        probabilities: Any,
        phishing_probability: Optional[float]
    ) -> Dict[str, Optional[float]]:
        """
        Normalize class probability output.
        """

        if not isinstance(
            probabilities,
            dict
        ):

            probabilities = {}

        phishing = (
            probabilities.get(
                "phishing"
            )
        )

        if phishing is None:

            phishing = phishing_probability

        legitimate = (
            probabilities.get(
                "legitimate"
            )
        )

        if legitimate is None:

            if phishing is not None:

                legitimate = (
                    1.0 - phishing
                )

        def safe_probability(
            value: Any
        ) -> Optional[float]:

            if value is None:

                return None

            try:

                number = float(
                    value
                )

                if number != number:

                    return None

                return round(
                    max(
                        0.0,
                        min(
                            1.0,
                            number
                        )
                    ),
                    4
                )

            except (
                TypeError,
                ValueError,
                OverflowError
            ):

                return None

        return {

            "legitimate":
                safe_probability(
                    legitimate
                ),

            "phishing":
                safe_probability(
                    phishing
                )
        }

    # ========================================================================
    # RISK LEVEL
    # ========================================================================

    @staticmethod
    def _get_risk_level(
        risk_score: int
    ) -> str:
        """
        Project-wide risk mapping.

            0-29   low
            30-59  medium
            60-79  high
            80-100 critical
        """

        try:

            score = int(
                risk_score
            )

        except (
            TypeError,
            ValueError,
            OverflowError
        ):

            return "unknown"

        if score < 0 or score > 100:

            return "unknown"

        if score < 30:

            return "low"

        if score < 60:

            return "medium"

        if score < 80:

            return "high"

        return "critical"

    # ========================================================================
    # FEATURE COUNT
    # ========================================================================

    @staticmethod
    def _get_feature_count(
        features: Any
    ) -> int:
        """
        Count supplied SSL features.

        Uses the canonical SSLFeatureSchema when possible.
        """

        if not isinstance(
            features,
            dict
        ):

            return 0

        # The predictor's feature vector is canonical.
        try:

            from .feature_schema import SSLFeatureSchema

            expected = (
                SSLFeatureSchema.get_feature_names()
            )

            return int(
                sum(
                    1
                    for feature in expected
                    if feature in features
                )
            )

        except Exception:

            return len(
                features
            )

    # ========================================================================
    # EXPECTED FEATURE COUNT
    # ========================================================================

    @staticmethod
    def _get_expected_feature_count() -> int:
        """
        Return canonical SSL feature count.
        """

        try:

            from .feature_schema import SSLFeatureSchema

            return int(
                SSLFeatureSchema.get_feature_count()
            )

        except Exception:

            return 17

    # ========================================================================
    # SAFE INTEGER
    # ========================================================================

    @staticmethod
    def _safe_int(
        value: Any,
        default: int = 0
    ) -> int:
        """
        Safely convert to integer.
        """

        try:

            return int(
                float(
                    value
                )
            )

        except (
            TypeError,
            ValueError,
            OverflowError
        ):

            return default

    # ========================================================================
    # LIST NORMALIZATION
    # ========================================================================

    @staticmethod
    def _ensure_list(
        value: Any
    ) -> List[Any]:
        """
        Ensure a value is represented as a list.
        """

        if isinstance(
            value,
            list
        ):

            return value

        if value is None:

            return []

        return [value]

    # ========================================================================
    # TIMESTAMP
    # ========================================================================

    @staticmethod
    def _get_timestamp() -> str:
        """
        Return UTC ISO-8601 timestamp.
        """

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ========================================================================
    # DIRECT FEATURE ANALYSIS
    # ========================================================================

    def analyze_features(
        self,
        ssl_features: Dict[str, Any],
        target_url: str = "unknown_url"
    ) -> Dict[str, Any]:
        """
        Convenience method for directly analyzing SSL features.
        """

        return self.analyze({

            "ssl_features":
                ssl_features,

            "metadata":
                {

                    "target_url":
                        target_url
                }
        })


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "SSLAIAgent"
]