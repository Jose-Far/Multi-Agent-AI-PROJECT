"""
HTML AI Agent
=============

Day 13 - Standardized AgentResult Output

Master controller for HTML-based phishing analysis.

Pipeline
--------

Unified Feature Vector
        |
        v
HTML Feature Collection
        |
        v
HTML Feature Validation
        |
        v
HTML Predictor
        |
        v
HTML Risk Scorer
        |
        v
HTML Explainer
        |
        v
AgentResult
        |
        v
Decision Fusion Engine


IMPORTANT SECURITY RULE
-----------------------

HTML collection failure is NOT legitimate evidence.

Therefore:

    unavailable
    failed
    timeout
    empty HTML
    invalid HTML
    missing HTML features
    predictor failure

must result in:

    status = "unavailable" OR "error"
    prediction = "unknown"
    probability = None
    risk_score = None
    risk_level = "unknown"
    signal_available = False

It must NEVER result in:

    prediction = "legitimate"
    risk_score = 0


A predictor fallback is allowed only when usable HTML
evidence was successfully collected.

Day 13 standard result:

    AgentResult
        |
        +-- agent
        +-- status
        +-- prediction
        +-- probability
        +-- confidence
        +-- risk_score
        +-- risk_level
        +-- prediction_source
        +-- model_status
        +-- risk_factors
        +-- evidence
        +-- explanation
"""

from __future__ import annotations

import logging

from datetime import datetime, timezone

from typing import (
    Dict,
    Any,
    Optional,
    List,
)


# ============================================================================
# LOCAL COMPONENTS
# ============================================================================

from .predictor import HTMLPredictor
from .risk_score import HTMLRiskScorer
from .explain import HTMLExplainer
from .feature_schema import HTMLFeatureSchema


# ============================================================================
# STANDARD RESULT
# ============================================================================

from agents.agent_result import AgentResult


# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# HTML AI AGENT
# ============================================================================

class HTMLAIAgent:
    """
    Master controller for the HTML AI Agent.

    Responsibilities
    -----------------

    1. Validate the unified feature vector.
    2. Detect HTML collection failures.
    3. Validate HTML feature availability.
    4. Execute HTMLPredictor.
    5. Execute HTMLRiskScorer.
    6. Generate XAI explanations.
    7. Produce standardized AgentResult output.

    This controller does NOT:

        - perform browser automation
        - collect HTML directly
        - train the ML model
        - calculate ML probabilities itself
        - calculate SHAP values itself
        - make the final system-wide decision
    """

    # ========================================================================
    # AGENT METADATA
    # ========================================================================

    AGENT_NAME = "HTML_AI_Agent"

    AGENT_VERSION = "4.0.0"

    AGENT_TYPE = "HTML Security Analysis"

    FEATURE_KEY = "html_features"

    EXPECTED_FEATURE_COUNT = 50

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(
        self,
        model_path: Optional[str] = None
    ) -> None:
        """
        Initialize HTML predictor, risk scorer and explainer.
        """

        logger.info(
            "Initializing %s version %s...",
            self.AGENT_NAME,
            self.AGENT_VERSION,
        )

        self.is_initialized = False

        self.initialization_error: Optional[str] = None

        self.predictor: Optional[
            HTMLPredictor
        ] = None

        self.risk_scorer: Optional[
            HTMLRiskScorer
        ] = None

        self.explainer: Optional[
            HTMLExplainer
        ] = None

        try:

            # ================================================================
            # PREDICTOR
            # ================================================================

            self.predictor = (
                self._initialize_predictor(
                    model_path
                )
            )

            logger.info(
                "HTML Predictor initialized."
            )

            # ================================================================
            # RISK SCORER
            # ================================================================

            self.risk_scorer = (
                HTMLRiskScorer()
            )

            logger.info(
                "HTML Risk Scorer initialized."
            )

            # ================================================================
            # MODEL FOR EXPLAINER
            # ================================================================

            model = (
                self._get_predictor_model()
            )

            # ================================================================
            # EXPLAINER
            # ================================================================

            self.explainer = (
                HTMLExplainer(
                    model=model
                )
            )

            logger.info(
                "HTML Explainer initialized."
            )

            # ================================================================
            # FINAL INITIALIZATION
            # ================================================================

            self.is_initialized = True

            logger.info(
                "%s initialized successfully.",
                self.AGENT_NAME,
            )

        except Exception as exc:

            self.is_initialized = False

            self.initialization_error = str(
                exc
            )

            logger.error(
                "Critical %s initialization failure: %s",
                self.AGENT_NAME,
                str(exc),
                exc_info=True,
            )

            raise


    # ========================================================================
    # PREDICTOR INITIALIZATION
    # ========================================================================

    @staticmethod
    def _initialize_predictor(
        model_path: Optional[str]
    ) -> HTMLPredictor:
        """
        Initialize HTMLPredictor.

        Supports the current predictor API and older versions
        that do not accept model_path.
        """

        try:

            return HTMLPredictor(
                model_path=model_path
            )

        except TypeError:

            logger.debug(
                (
                    "HTMLPredictor does not accept "
                    "model_path. Initializing without it."
                )
            )

            return HTMLPredictor()


    # ========================================================================
    # MODEL ACCESS
    # ========================================================================

    def _get_predictor_model(
        self
    ) -> Any:
        """
        Safely obtain the underlying model used by the explainer.
        """

        if self.predictor is None:

            return None

        try:

            method = getattr(
                self.predictor,
                "get_model",
                None,
            )

            if callable(method):

                return method()

        except Exception as exc:

            logger.warning(
                "Unable to obtain HTML predictor model: %s",
                str(exc),
            )

        return None


    # ========================================================================
    # MAIN ANALYSIS
    # ========================================================================

    def analyze(
        self,
        unified_vector: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute complete HTML analysis.

        Expected input:

            {
                "html_features": {
                    ... 50 canonical features ...
                },

                "metadata": {
                    "target_url": "https://example.com"
                }
            }

        Returns:

            AgentResult.to_dict()

        Important:

            Missing HTML evidence is returned as unavailable.

            It is NEVER converted to legitimate.
        """

        started_at = (
            datetime.now(
                timezone.utc
            )
        )

        # ====================================================================
        # PHASE 1 - INPUT VALIDATION
        # ====================================================================

        if not isinstance(
            unified_vector,
            dict
        ):

            return self._get_error_response(
                reason=(
                    "Unified feature vector must be a dictionary."
                ),
                target_url="unknown_target",
                started_at=started_at,
            )


        # ====================================================================
        # TARGET URL
        # ====================================================================

        metadata = (
            unified_vector.get(
                "metadata",
                {}
            )
        )

        if not isinstance(
            metadata,
            dict
        ):

            metadata = {}


        target_url = (

            metadata.get(
                "target_url"
            )

            or unified_vector.get(
                "url"
            )

            or unified_vector.get(
                "target_url"
            )

            or "unknown_target"
        )


        if not isinstance(
            target_url,
            str
        ):

            target_url = str(
                target_url
            )


        target_url = target_url.strip()


        # ====================================================================
        # PHASE 2 - COLLECTION FAILURE
        # ====================================================================

        collection_failure = (
            self._detect_collection_failure(
                unified_vector
            )
        )

        if collection_failure:

            return self._get_unavailable_response(
                reason=collection_failure,
                target_url=target_url,
                started_at=started_at,
            )


        # ====================================================================
        # PHASE 3 - HTML FEATURES
        # ====================================================================

        html_features = (
            unified_vector.get(
                self.FEATURE_KEY
            )
        )


        if not isinstance(
            html_features,
            dict
        ):

            return self._get_unavailable_response(
                reason=(
                    "Required HTML feature block "
                    "'html_features' is missing or invalid."
                ),
                target_url=target_url,
                started_at=started_at,
            )


        if not html_features:

            return self._get_unavailable_response(
                reason=(
                    "HTML feature block is empty."
                ),
                target_url=target_url,
                started_at=started_at,
            )


        # ====================================================================
        # PHASE 4 - FEATURE AVAILABILITY
        # ====================================================================

        feature_failure = (
            self._detect_feature_failure(
                html_features
            )
        )

        if feature_failure:

            return self._get_unavailable_response(
                reason=feature_failure,
                target_url=target_url,
                started_at=started_at,
            )


        # ====================================================================
        # PHASE 5 - FEATURE SCHEMA
        # ====================================================================

        expected_features = list(
            HTMLFeatureSchema.get_schema()
        )


        expected_count = len(
            expected_features
        )


        if expected_count != (
            self.EXPECTED_FEATURE_COUNT
        ):

            return self._get_error_response(
                reason=(
                    "HTML feature schema configuration error. "
                    f"Expected {self.EXPECTED_FEATURE_COUNT} "
                    f"features but schema contains "
                    f"{expected_count}."
                ),
                target_url=target_url,
                started_at=started_at,
            )


        supplied_feature_count = (
            self._get_feature_count(
                html_features
            )
        )


        if supplied_feature_count == 0:

            return self._get_unavailable_response(
                reason=(
                    "HTML feature block contains "
                    "no canonical model features."
                ),
                target_url=target_url,
                started_at=started_at,
            )


        # ====================================================================
        # PHASE 6 - PREDICTOR
        # ====================================================================

        if self.predictor is None:

            return self._get_unavailable_response(
                reason=(
                    "HTML Predictor is unavailable."
                ),
                target_url=target_url,
                started_at=started_at,
            )


        try:

            prediction_result = (
                self.predictor.predict(
                    html_features
                )
            )

        except Exception as exc:

            logger.error(
                (
                    "HTML Predictor execution failed | "
                    "URL=%s | Error=%s"
                ),
                target_url,
                str(exc),
                exc_info=True,
            )

            return self._get_error_response(
                reason=(
                    f"HTML Predictor execution failed: {exc}"
                ),
                target_url=target_url,
                started_at=started_at,
            )


        if not isinstance(
            prediction_result,
            dict
        ):

            return self._get_error_response(
                reason=(
                    "HTML Predictor returned "
                    "an invalid response."
                ),
                target_url=target_url,
                started_at=started_at,
            )


        # ====================================================================
        # PHASE 7 - PREDICTOR SIGNAL AVAILABILITY
        # ====================================================================

        signal_available = (
            prediction_result.get(
                "signal_available",
                True
            )
        )


        if signal_available is False:

            return self._get_unavailable_response(
                reason=str(
                    prediction_result.get(
                        "error",
                        "HTML prediction signal unavailable."
                    )
                ),
                target_url=target_url,
                started_at=started_at,
            )


        # ====================================================================
        # PHASE 8 - PREDICTION
        # ====================================================================

        prediction = (
            self._normalize_prediction(
                prediction_result.get(
                    "class_label",
                    prediction_result.get(
                        "prediction",
                        "unknown"
                    )
                )
            )
        )


        # --------------------------------------------------------------------
        # Unknown is NEVER a legitimate prediction.
        # --------------------------------------------------------------------

        if prediction == "unknown":

            return self._get_unavailable_response(
                reason=(
                    "HTML Predictor returned "
                    "an unknown prediction."
                ),
                target_url=target_url,
                started_at=started_at,
            )


        # ====================================================================
        # PHASE 9 - CONFIDENCE
        # ====================================================================

        confidence = (
            self._clamp_probability(
                prediction_result.get(
                    "confidence",
                    prediction_result.get(
                        "confidence_score",
                        0.0
                    )
                )
            )
        )


        # ====================================================================
        # PHASE 10 - PHISHING PROBABILITY
        # ====================================================================

        phishing_probability = (
            prediction_result.get(
                "phishing_probability"
            )
        )


        if phishing_probability is None:

            class_probabilities = (
                prediction_result.get(
                    "class_probabilities",
                    {}
                )
            )

            if isinstance(
                class_probabilities,
                dict
            ):

                phishing_probability = (
                    class_probabilities.get(
                        "phishing"
                    )
                )


        if phishing_probability is None:

            return self._get_unavailable_response(
                reason=(
                    "HTML Predictor did not provide "
                    "a phishing probability."
                ),
                target_url=target_url,
                started_at=started_at,
            )


        phishing_probability = (
            self._clamp_probability(
                phishing_probability
            )
        )


        # ====================================================================
        # PHASE 11 - CLASS PROBABILITIES
        # ====================================================================

        class_probabilities = (
            prediction_result.get(
                "class_probabilities",
                {}
            )
        )


        if not isinstance(
            class_probabilities,
            dict
        ):

            class_probabilities = {}


        legitimate_probability = (
            self._clamp_probability(
                class_probabilities.get(
                    "legitimate",
                    prediction_result.get(
                        "legitimate_probability",
                        1.0 - phishing_probability
                    )
                )
            )
        )


        normalized_class_probabilities = (
            self._normalize_class_probabilities(
                class_probabilities,
                legitimate_probability,
                phishing_probability,
            )
        )


        # ====================================================================
        # PHASE 12 - MODEL STATUS
        # ====================================================================

        raw_model_status = (
            prediction_result.get(
                "model_status",
                "unknown"
            )
        )


        model_status = (
            self._normalize_model_status(
                raw_model_status
            )
        )


        # ====================================================================
        # PHASE 13 - PREDICTION SOURCE
        # ====================================================================

        prediction_source = (
            self._normalize_prediction_source(
                prediction_result
            )
        )


        fallback_used = bool(
            prediction_result.get(
                "fallback_used",
                False
            )
        )


        # --------------------------------------------------------------------
        # Safety:
        #
        # fallback_used=True means heuristic prediction,
        # but does NOT mean unavailable when HTML evidence exists.
        # --------------------------------------------------------------------

        if fallback_used:

            prediction_source = (
                "fallback_heuristic"
            )

            model_status = (
                "fallback"
            )


        # ====================================================================
        # PHASE 14 - RISK SCORING
        # ====================================================================

        if self.risk_scorer is None:

            return self._get_unavailable_response(
                reason=(
                    "HTML Risk Scorer is unavailable."
                ),
                target_url=target_url,
                started_at=started_at,
            )


        try:

            risk_assessment = (
                self.risk_scorer.calculate_risk(
                    probability=phishing_probability,
                    features=html_features,
                )
            )

        except Exception as exc:

            logger.error(
                (
                    "HTML Risk Scorer failed | "
                    "URL=%s | Error=%s"
                ),
                target_url,
                str(exc),
                exc_info=True,
            )

            return self._get_error_response(
                reason=(
                    f"HTML Risk Scorer failed: {exc}"
                ),
                target_url=target_url,
                started_at=started_at,
            )


        if not isinstance(
            risk_assessment,
            dict
        ):

            return self._get_unavailable_response(
                reason=(
                    "HTML Risk Scorer returned "
                    "an invalid response."
                ),
                target_url=target_url,
                started_at=started_at,
            )


        # ====================================================================
        # PHASE 15 - RISK SIGNAL AVAILABILITY
        # ====================================================================

        risk_signal_available = (
            risk_assessment.get(
                "signal_available",
                True
            )
        )


        if risk_signal_available is False:

            return self._get_unavailable_response(
                reason=str(
                    risk_assessment.get(
                        "reason",
                        "HTML risk signal unavailable."
                    )
                ),
                target_url=target_url,
                started_at=started_at,
            )


        # ====================================================================
        # PHASE 16 - RISK SCORE
        # ====================================================================

        raw_risk_score = (
            risk_assessment.get(
                "risk_score"
            )
        )


        if raw_risk_score is None:

            return self._get_unavailable_response(
                reason=(
                    "HTML Risk Scorer returned "
                    "no risk score."
                ),
                target_url=target_url,
                started_at=started_at,
            )


        risk_score = (
            self._safe_int(
                raw_risk_score,
                default=-1
            )
        )


        if not (
            0 <= risk_score <= 100
        ):

            return self._get_unavailable_response(
                reason=(
                    "HTML Risk Scorer returned "
                    "an invalid risk score."
                ),
                target_url=target_url,
                started_at=started_at,
            )


        # ====================================================================
        # PHASE 17 - RISK LEVEL
        # ====================================================================

        risk_level = (
            self._get_risk_level(
                risk_score
            )
        )


        # ====================================================================
        # PHASE 18 - RISK INDICATORS
        # ====================================================================

        triggered_indicators = (
            risk_assessment.get(
                "triggered_indicators",
                risk_assessment.get(
                    "risk_indicators",
                    []
                )
            )
        )


        if not isinstance(
            triggered_indicators,
            list
        ):

            triggered_indicators = []


        risk_indicators = (
            risk_assessment.get(
                "risk_indicators",
                triggered_indicators
            )
        )


        if not isinstance(
            risk_indicators,
            list
        ):

            risk_indicators = (
                triggered_indicators
            )


        contextual_observations = (
            risk_assessment.get(
                "contextual_observations",
                []
            )
        )


        if not isinstance(
            contextual_observations,
            list
        ):

            contextual_observations = []


        protective_indicators = (
            risk_assessment.get(
                "protective_indicators",
                []
            )
        )


        if not isinstance(
            protective_indicators,
            list
        ):

            protective_indicators = []


        # ====================================================================
        # PHASE 19 - EXPLANATION
        # ====================================================================

        explanation = (
            self._generate_explanation(
                features=html_features,
                prediction=prediction,
                predictor_explanation=(
                    prediction_result.get(
                        "explanation"
                    )
                ),
            )
        )


        # ====================================================================
        # PHASE 20 - BUILD STANDARD AgentResult
        # ====================================================================

        execution_time_ms = (
            self._execution_time_ms(
                started_at
            )
        )


        evidence = (
            self._build_evidence(
                target_url=target_url,
                risk_assessment=risk_assessment,
                prediction_result=prediction_result,
                feature_count=supplied_feature_count,
            )
        )


        risk_factors = (
            self._build_risk_factors(
                risk_assessment=risk_assessment,
                explanation=explanation,
            )
        )


        metadata_output = {

            "target_url":
                target_url,

            "agent_version":
                self.AGENT_VERSION,

            "agent_type":
                self.AGENT_TYPE,

            "feature_key":
                self.FEATURE_KEY,

            "feature_count":
                supplied_feature_count,

            "expected_feature_count":
                self.EXPECTED_FEATURE_COUNT,

            "schema_version":
                self._get_schema_version(),

            "class_probabilities":
                normalized_class_probabilities,

            "decision_threshold":
                prediction_result.get(
                    "decision_threshold",
                    prediction_result.get(
                        "threshold"
                    )
                ),

            "fallback_used":
                fallback_used,

            "model_path":
                prediction_result.get(
                    "model_path"
                ),

            "training_evaluation":
                prediction_result.get(
                    "training_evaluation",
                    {}
                ),

            "risk_assessment":
                risk_assessment,

            "contextual_observations":
                contextual_observations,

            "protective_indicators":
                protective_indicators,

            "feature_validation":
                {

                    "schema_complete":
                        (
                            supplied_feature_count
                            == self.EXPECTED_FEATURE_COUNT
                        ),

                    "supplied_feature_count":
                        supplied_feature_count,

                    "expected_feature_count":
                        self.EXPECTED_FEATURE_COUNT,
                },
        }


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

            execution_time_ms=execution_time_ms,

            feature_key=self.FEATURE_KEY,

            metadata=metadata_output,
        )


        standard_result = (
            result.to_dict()
        )


        # ====================================================================
        # TEMPORARY LEGACY COMPATIBILITY
        # ====================================================================
        #
        # The current Orchestrator/Fusion still expects some older fields.
        #
        # These will be removed after the Orchestrator is migrated completely
        # to AgentResult.
        #
        # IMPORTANT:
        #
        # The NEW canonical fields above are the fields Fusion/UI should use.
        # ====================================================================

        standard_result.update({

            "agent_name":
                self.AGENT_NAME,

            "agent_version":
                self.AGENT_VERSION,

            "agent_type":
                self.AGENT_TYPE,

            "target_url":
                target_url,

            "timestamp":
                self._get_timestamp(),

            "analysis_status":
                "success",

            "signal_available":
                True,

            "verdict":
                prediction,

            "confidence_score":
                confidence,

            "phishing_probability":
                phishing_probability,

            "legitimate_probability":
                normalized_class_probabilities.get(
                    "legitimate"
                ),

            "class_probabilities":
                normalized_class_probabilities,

            "feature_count":
                supplied_feature_count,

            "features_evaluated_count":
                supplied_feature_count,

            "prediction_method":
                (
                    "xgboost_ml"
                    if prediction_source == "trained_ml"
                    else prediction_source
                ),

            "model_based":
                (
                    prediction_source
                    == "trained_ml"
                    and not fallback_used
                ),

            "model":
                {

                    "status":
                        model_status,

                    "prediction_source":
                        prediction_source,

                    "model_based":
                        (
                            prediction_source
                            == "trained_ml"
                            and not fallback_used
                        ),

                    "fallback_used":
                        fallback_used,

                    "model_loaded":
                        model_status
                        == "available",
                },

            "feature_validation":
                metadata_output[
                    "feature_validation"
                ],

            "risk_indicators":
                risk_indicators,

            "contextual_observations":
                contextual_observations,

            "protective_indicators":
                protective_indicators,

            "model_metadata":
                {

                    "model_status":
                        model_status,

                    "prediction_source":
                        prediction_source,

                    "fallback_used":
                        fallback_used,

                    "model_path":
                        prediction_result.get(
                            "model_path"
                        ),

                    "decision_threshold":
                        prediction_result.get(
                            "decision_threshold",
                            prediction_result.get(
                                "threshold"
                            )
                        ),
                },
        })


        logger.info(
            (
                "HTML Agent completed | "
                "URL=%s | Prediction=%s | "
                "Probability=%.4f | Risk=%d | "
                "Risk Level=%s | Source=%s | "
                "Model Status=%s | Fallback=%s"
            ),
            target_url,
            prediction,
            phishing_probability,
            risk_score,
            risk_level,
            prediction_source,
            model_status,
            fallback_used,
        )


        return standard_result


    # ========================================================================
    # COLLECTION FAILURE DETECTION
    # ========================================================================

    @staticmethod
    def _detect_collection_failure(
        unified_vector: Dict[str, Any]
    ) -> Optional[str]:
        """
        Detect HTML collection failures.

        Supported blocks:

            html_analysis
            html_collection
            html_data
            html_features

        This function checks evidence availability only.

        It does NOT classify the website.
        """

        blocks = [

            unified_vector.get(
                "html_analysis"
            ),

            unified_vector.get(
                "html_collection"
            ),

            unified_vector.get(
                "html_data"
            ),

            unified_vector.get(
                "html_features"
            ),
        ]


        for block in blocks:

            if not isinstance(
                block,
                dict
            ):

                continue


            # ----------------------------------------------------------------
            # Explicit success=False
            # ----------------------------------------------------------------

            if block.get(
                "success"
            ) is False:

                error = (

                    block.get(
                        "error"
                    )

                    or block.get(
                        "error_message"
                    )

                    or block.get(
                        "reason"
                    )

                    or "HTML collection reported failure."
                )

                return (
                    f"HTML collection failed: {error}"
                )


            # ----------------------------------------------------------------
            # Explicit signal unavailable
            # ----------------------------------------------------------------

            if block.get(
                "signal_available"
            ) is False:

                reason = (

                    block.get(
                        "error"
                    )

                    or block.get(
                        "reason"
                    )

                    or "HTML signal marked unavailable."
                )

                return (
                    f"HTML collection unavailable: {reason}"
                )


            # ----------------------------------------------------------------
            # Status field
            # ----------------------------------------------------------------

            status = str(
                block.get(
                    "status",
                    ""
                )
            ).strip().lower()


            if status in {

                "error",
                "failed",
                "failure",
                "timeout",
                "unavailable",

            }:

                error = (

                    block.get(
                        "error"
                    )

                    or block.get(
                        "error_message"
                    )

                    or f"status={status}"
                )

                return (
                    f"HTML collection failed: {error}"
                )


            # ----------------------------------------------------------------
            # Timeout
            # ----------------------------------------------------------------

            if block.get(
                "timeout"
            ) is True:

                return (
                    "HTML collection failed: "
                    "browser/page timeout."
                )


            # ----------------------------------------------------------------
            # Explicit error fields
            # ----------------------------------------------------------------

            for key in (

                "exception",
                "collection_error",
                "extraction_error",

            ):

                value = block.get(
                    key
                )

                if value:

                    return (
                        f"HTML collection failed: {value}"
                    )


        return None


    # ========================================================================
    # FEATURE FAILURE DETECTION
    # ========================================================================

    @staticmethod
    def _detect_feature_failure(
        html_features: Dict[str, Any]
    ) -> Optional[str]:
        """
        Detect invalid or unavailable HTML feature payloads.

        IMPORTANT:

        Missing evidence is represented as unavailable.

        It is NOT interpreted as legitimate.
        """

        if not isinstance(
            html_features,
            dict
        ):

            return (
                "HTML feature payload is not a dictionary."
            )


        # --------------------------------------------------------------------
        # Explicit success=False
        # --------------------------------------------------------------------

        if html_features.get(
            "success"
        ) is False:

            return (

                html_features.get(
                    "error"
                )

                or html_features.get(
                    "error_message"
                )

                or "HTML feature extraction failed."
            )


        # --------------------------------------------------------------------
        # Explicit signal unavailable
        # --------------------------------------------------------------------

        if html_features.get(
            "signal_available"
        ) is False:

            return (
                html_features.get(
                    "error",
                    "HTML feature signal is unavailable."
                )
            )


        # --------------------------------------------------------------------
        # Status
        # --------------------------------------------------------------------

        status = str(
            html_features.get(
                "status",
                ""
            )
        ).strip().lower()


        if status in {

            "error",
            "failed",
            "failure",
            "timeout",
            "unavailable",

        }:

            return (
                f"HTML feature extraction status is '{status}'."
            )


        # --------------------------------------------------------------------
        # Timeout
        # --------------------------------------------------------------------

        if html_features.get(
            "timeout"
        ) is True:

            return (
                "HTML feature extraction timed out."
            )


        # --------------------------------------------------------------------
        # Explicit error fields
        # --------------------------------------------------------------------

        for key in (

            "exception",
            "collection_error",
            "extraction_error",

        ):

            value = html_features.get(
                key
            )

            if value:

                return (
                    f"HTML feature extraction failed: {value}"
                )


        # --------------------------------------------------------------------
        # Canonical feature availability
        # --------------------------------------------------------------------

        schema = list(
            HTMLFeatureSchema.get_schema()
        )


        canonical_features = [

            feature

            for feature
            in schema

            if feature in html_features
        ]


        if not canonical_features:

            return (
                "HTML extraction produced "
                "no canonical model features."
            )


        return None


    # ========================================================================
    # PREDICTION NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_prediction(
        value: Any
    ) -> str:
        """
        Normalize HTML prediction labels.
        """

        if value is None:

            return "unknown"


        normalized = str(
            value
        ).strip().lower()


        if normalized in {

            "legitimate",
            "benign",
            "safe",
            "0",

        }:

            return "legitimate"


        if normalized in {

            "phishing",
            "malicious",
            "unsafe",
            "1",

        }:

            return "phishing"


        if normalized == "suspicious":

            return "suspicious"


        return "unknown"


    # ========================================================================
    # PREDICTION SOURCE
    # ========================================================================

    @staticmethod
    def _normalize_prediction_source(
        prediction_result: Dict[str, Any]
    ) -> str:
        """
        Normalize predictor source information.

        Standard values:

            trained_ml
            heuristic
            fallback_heuristic
            unavailable
            error
        """

        fallback_used = bool(
            prediction_result.get(
                "fallback_used",
                False
            )
        )


        if fallback_used:

            return "fallback_heuristic"


        source = str(
            prediction_result.get(
                "prediction_source",
                ""
            )
        ).strip().lower()


        if source in {

            "xgboost",
            "trained_ml",
            "ml",
            "machine_learning",
            "trained",
            "xgb",

        }:

            return "trained_ml"


        if source in {

            "heuristic",
            "rule_based",
            "rules",

        }:

            return "heuristic"


        if source in {

            "fallback",
            "heuristic_fallback",
            "fallback_heuristic",
            "fallback_rule_based",

        }:

            return "fallback_heuristic"


        if source in {

            "error",
            "failed",

        }:

            return "error"


        if source in {

            "unavailable",
            "none",
            "unknown",
            "",

        }:

            return "unavailable"


        logger.warning(
            (
                "Unknown HTML prediction source '%s'. "
                "Marking source as unavailable."
            ),
            source,
        )


        return "unavailable"


    # ========================================================================
    # MODEL STATUS NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_model_status(
        value: Any
    ) -> str:
        """
        Normalize model status.

        Standard values used by Day 13:

            available
            fallback
            unavailable
            error
        """

        status = str(
            value or ""
        ).strip().lower()


        if status in {

            "loaded",
            "ready",
            "loaded_ml_model",
            "available",
            "model_loaded",

        }:

            return "available"


        if status in {

            "fallback",
            "heuristic",
            "heuristic_fallback",

        }:

            return "fallback"


        if status in {

            "error",
            "failed",

        }:

            return "error"


        if status in {

            "unavailable",
            "none",
            "unknown",
            "",

        }:

            return "unavailable"


        return status


    # ========================================================================
    # RISK LEVEL
    # ========================================================================

    @staticmethod
    def _get_risk_level(
        risk_score: int
    ) -> str:
        """
        Project-wide risk level mapping.

            0-29   = low
            30-59  = medium
            60-79  = high
            80-100 = critical
        """

        if risk_score < 30:

            return "low"


        if risk_score < 60:

            return "medium"


        if risk_score < 80:

            return "high"


        return "critical"


    # ========================================================================
    # SAFE INTEGER
    # ========================================================================

    @staticmethod
    def _safe_int(
        value: Any,
        default: int = 0
    ) -> int:
        """
        Safely convert a value to integer.
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
            OverflowError,
        ):

            return default


    # ========================================================================
    # PROBABILITY CLAMP
    # ========================================================================

    @staticmethod
    def _clamp_probability(
        value: Any,
        default: float = 0.0
    ) -> float:
        """
        Clamp probability to [0, 1].
        """

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):

            return default


        if number != number:

            return default


        if number == float(
            "inf"
        ):

            return 1.0


        if number == float(
            "-inf"
        ):

            return 0.0


        return max(
            0.0,
            min(
                1.0,
                number
            )
        )


    # ========================================================================
    # CLASS PROBABILITY NORMALIZATION
    # ========================================================================

    def _normalize_class_probabilities(
        self,
        probabilities: Dict[str, Any],
        legitimate: float,
        phishing: float
    ) -> Dict[str, float]:
        """
        Normalize legitimate/phishing probabilities.
        """

        if not isinstance(
            probabilities,
            dict
        ):

            probabilities = {}


        legitimate_value = (
            self._clamp_probability(
                probabilities.get(
                    "legitimate",
                    legitimate
                )
            )
        )


        phishing_value = (
            self._clamp_probability(
                probabilities.get(
                    "phishing",
                    phishing
                )
            )
        )


        total = (
            legitimate_value
            + phishing_value
        )


        if total > 0:

            legitimate_value = (
                legitimate_value
                / total
            )

            phishing_value = (
                phishing_value
                / total
            )


        return {

            "legitimate":
                round(
                    legitimate_value,
                    4
                ),

            "phishing":
                round(
                    phishing_value,
                    4
                ),
        }


    # ========================================================================
    # EXPLANATION
    # ========================================================================

    def _generate_explanation(
        self,
        features: Dict[str, Any],
        prediction: str,
        predictor_explanation: Optional[
            Dict[str, Any]
        ] = None
    ) -> Dict[str, Any]:
        """
        Build standardized explanation.

        The predictor's explanation is preferred because the current
        HTMLPredictor already generates the per-sample explanation.

        Explanation failure does NOT invalidate an otherwise valid
        prediction.
        """

        # ====================================================================
        # PREDICTOR EXPLANATION
        # ====================================================================

        if isinstance(
            predictor_explanation,
            dict
        ) and predictor_explanation:

            explanation = dict(
                predictor_explanation
            )

        else:

            explanation = {}


        # ====================================================================
        # SECONDARY EXPLAINER
        # ====================================================================

        if not explanation:

            if self.explainer is not None:

                try:

                    generated = (
                        self.explainer.generate_explanation(
                            features=features,
                            prediction=prediction,
                        )
                    )

                    if isinstance(
                        generated,
                        dict
                    ):

                        explanation = generated

                except Exception as exc:

                    logger.warning(
                        (
                            "HTML Explainer failed: %s"
                        ),
                        str(exc),
                    )


        # ====================================================================
        # NORMALIZE EXPLANATION
        # ====================================================================

        if not isinstance(
            explanation,
            dict
        ):

            explanation = {}


        summary = (
            explanation.get(
                "summary",
                "HTML analysis completed."
            )
        )


        if not summary:

            summary = (
                "HTML analysis completed."
            )


        top_shap_factors = (
            explanation.get(
                "top_shap_factors",
                []
            )
        )


        if not isinstance(
            top_shap_factors,
            list
        ):

            top_shap_factors = []


        top_factors = (
            explanation.get(
                "top_factors",
                top_shap_factors
            )
        )


        if not isinstance(
            top_factors,
            list
        ):

            top_factors = []


        risk_factors = (
            explanation.get(
                "risk_factors",
                []
            )
        )


        if not isinstance(
            risk_factors,
            list
        ):

            risk_factors = []


        protective_factors = (
            explanation.get(
                "protective_factors",
                []
            )
        )


        if not isinstance(
            protective_factors,
            list
        ):

            protective_factors = []


        method = str(
            explanation.get(
                "method",
                explanation.get(
                    "explanation_method",
                    "unknown"
                )
            )
        )


        shap_status = str(
            explanation.get(
                "shap_status",
                "unknown"
            )
        )


        return {

            "summary":
                str(summary),

            "method":
                method,

            "top_shap_factors":
                top_shap_factors,

            "top_factors":
                top_factors,

            "risk_factors":
                risk_factors,

            "protective_factors":
                protective_factors,

            "shap_status":
                shap_status,

            "shap_error":
                explanation.get(
                    "shap_error"
                ),

            "phishing_probability":
                explanation.get(
                    "phishing_probability"
                ),

            "explanation_source":
                (
                    "predictor"
                    if predictor_explanation
                    else "explainer"
                ),
        }


    # ========================================================================
    # RISK FACTORS
    # ========================================================================

    @staticmethod
    def _build_risk_factors(
        risk_assessment: Dict[str, Any],
        explanation: Dict[str, Any]
    ) -> List[Any]:
        """
        Build unified risk factor list.
        """

        factors: List[Any] = []


        for key in (

            "triggered_indicators",
            "risk_indicators",
            "risk_factors",

        ):

            value = risk_assessment.get(
                key
            )


            if isinstance(
                value,
                list
            ):

                factors.extend(
                    value
                )


        explanation_factors = (
            explanation.get(
                "risk_factors",
                []
            )
        )


        if isinstance(
            explanation_factors,
            list
        ):

            factors.extend(
                explanation_factors
            )


        # --------------------------------------------------------------------
        # Remove duplicates while preserving order.
        # --------------------------------------------------------------------

        unique_factors = []


        for factor in factors:

            if factor not in unique_factors:

                unique_factors.append(
                    factor
                )


        return unique_factors


    # ========================================================================
    # EVIDENCE
    # ========================================================================

    @staticmethod
    def _build_evidence(
        target_url: str,
        risk_assessment: Dict[str, Any],
        prediction_result: Dict[str, Any],
        feature_count: int
    ) -> List[Any]:
        """
        Build compact evidence for Fusion/UI.
        """

        evidence: List[Any] = []


        # --------------------------------------------------------------------
        # Target
        # --------------------------------------------------------------------

        evidence.append({

            "type":
                "analysis_target",

            "target":
                target_url,
        })


        # --------------------------------------------------------------------
        # Feature evidence
        # --------------------------------------------------------------------

        evidence.append({

            "type":
                "feature_schema",

            "feature_count":
                feature_count,

            "expected_feature_count":
                50,

            "schema_complete":
                feature_count == 50,
        })


        # --------------------------------------------------------------------
        # Risk indicators
        # --------------------------------------------------------------------

        indicators = (
            risk_assessment.get(
                "triggered_indicators",
                risk_assessment.get(
                    "risk_indicators",
                    []
                )
            )
        )


        if isinstance(
            indicators,
            list
        ):

            for indicator in indicators:

                evidence.append({

                    "type":
                        "risk_indicator",

                    "value":
                        indicator,
                })


        # --------------------------------------------------------------------
        # Predictor evidence
        # --------------------------------------------------------------------

        predictor_evidence = (
            prediction_result.get(
                "evidence"
            )
        )


        if isinstance(
            predictor_evidence,
            list
        ):

            evidence.extend(
                predictor_evidence
            )

        elif predictor_evidence is not None:

            evidence.append(
                predictor_evidence
            )


        return evidence


    # ========================================================================
    # FEATURE COUNT
    # ========================================================================

    @staticmethod
    def _get_feature_count(
        html_features: Any
    ) -> int:
        """
        Count canonical HTML model features supplied.
        """

        if not isinstance(
            html_features,
            dict
        ):

            return 0


        expected = (
            HTMLFeatureSchema.get_schema()
        )


        return int(
            sum(
                1
                for feature
                in expected
                if feature in html_features
            )
        )


    # ========================================================================
    # SCHEMA VERSION
    # ========================================================================

    @staticmethod
    def _get_schema_version() -> Optional[str]:
        """
        Safely retrieve HTML schema version.
        """

        try:

            return getattr(
                HTMLFeatureSchema,
                "SCHEMA_VERSION",
                None
            )

        except Exception:

            return None


    # ========================================================================
    # EXECUTION TIME
    # ========================================================================

    @staticmethod
    def _execution_time_ms(
        started_at: datetime
    ) -> float:
        """
        Calculate elapsed execution time in milliseconds.
        """

        elapsed = (
            datetime.now(
                timezone.utc
            )
            - started_at
        )


        return round(
            elapsed.total_seconds()
            * 1000.0,
            3
        )


    # ========================================================================
    # TIMESTAMP
    # ========================================================================

    @staticmethod
    def _get_timestamp() -> str:
        """
        Return UTC ISO-8601 timestamp.
        """

        return (
            datetime.now(
                timezone.utc
            )
            .isoformat()
            .replace(
                "+00:00",
                "Z"
            )
        )


    # ========================================================================
    # UNAVAILABLE RESPONSE
    # ========================================================================

    def _get_unavailable_response(
        self,
        reason: str,
        target_url: str,
        started_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Build standardized unavailable AgentResult.

        IMPORTANT:

            unavailable != legitimate

        Therefore:

            prediction = unknown
            probability = None
            confidence = 0.0
            risk_score = None
            risk_level = unknown
            signal_available = False
        """

        if started_at is None:

            started_at = (
                datetime.now(
                    timezone.utc
                )
            )


        execution_time_ms = (
            self._execution_time_ms(
                started_at
            )
        )


        result = AgentResult.unavailable(

            agent=self.AGENT_NAME,

            reason=reason,

            execution_time_ms=(
                execution_time_ms
            ),

            feature_key=self.FEATURE_KEY,

            metadata={

                "target_url":
                    target_url,

                "agent_version":
                    self.AGENT_VERSION,

                "agent_type":
                    self.AGENT_TYPE,

                "expected_feature_count":
                    self.EXPECTED_FEATURE_COUNT,

                "reason":
                    reason,
            },
        )


        standard_result = (
            result.to_dict()
        )


        # ====================================================================
        # TEMPORARY LEGACY COMPATIBILITY
        # ====================================================================

        standard_result.update({

            "agent_name":
                self.AGENT_NAME,

            "agent_version":
                self.AGENT_VERSION,

            "agent_type":
                self.AGENT_TYPE,

            "target_url":
                target_url,

            "timestamp":
                self._get_timestamp(),

            "analysis_status":
                "unavailable",

            "signal_available":
                False,

            "verdict":
                "unknown",

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

            "feature_count":
                0,

            "features_evaluated_count":
                0,

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

            "feature_validation":
                {

                    "schema_complete":
                        False,

                    "supplied_feature_count":
                        0,

                    "expected_feature_count":
                        self.EXPECTED_FEATURE_COUNT,
                },

            "risk_indicators":
                [],

            "contextual_observations":
                [],

            "protective_indicators":
                [],

            "reason":
                reason,

            "explanation":
                {

                    "summary":
                        (
                            "HTML analysis was unavailable. "
                            "No HTML risk signal was generated."
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
                },
        })


        logger.warning(
            (
                "HTML Agent unavailable | "
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
        target_url: str,
        started_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Build standardized error AgentResult.

        An error NEVER becomes legitimate.
        """

        if started_at is None:

            started_at = (
                datetime.now(
                    timezone.utc
                )
            )


        execution_time_ms = (
            self._execution_time_ms(
                started_at
            )
        )


        result = AgentResult.error_result(

            agent=self.AGENT_NAME,

            error=reason,

            error_type="HTMLAgentError",

            execution_time_ms=(
                execution_time_ms
            ),

            feature_key=self.FEATURE_KEY,

            metadata={

                "target_url":
                    target_url,

                "agent_version":
                    self.AGENT_VERSION,

                "agent_type":
                    self.AGENT_TYPE,

                "reason":
                    reason,
            },
        )


        standard_result = (
            result.to_dict()
        )


        # ====================================================================
        # TEMPORARY LEGACY COMPATIBILITY
        # ====================================================================

        standard_result.update({

            "agent_name":
                self.AGENT_NAME,

            "agent_version":
                self.AGENT_VERSION,

            "agent_type":
                self.AGENT_TYPE,

            "target_url":
                target_url,

            "timestamp":
                self._get_timestamp(),

            "analysis_status":
                "error",

            "signal_available":
                False,

            "verdict":
                "unknown",

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

            "feature_count":
                0,

            "features_evaluated_count":
                0,

            "prediction_method":
                "error",

            "model_based":
                False,

            "model":
                {

                    "status":
                        "error",

                    "prediction_source":
                        "error",

                    "model_based":
                        False,

                    "fallback_used":
                        False,

                    "model_loaded":
                        False,
                },

            "feature_validation":
                {

                    "schema_complete":
                        False,

                    "supplied_feature_count":
                        0,

                    "expected_feature_count":
                        self.EXPECTED_FEATURE_COUNT,
                },

            "risk_indicators":
                [],

            "contextual_observations":
                [],

            "protective_indicators":
                [],

            "reason":
                reason,

            "error_details":
                reason,

            "explanation":
                {

                    "summary":
                        "HTML Agent encountered an error.",

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
                },
        })


        logger.error(
            (
                "HTML Agent error | "
                "URL=%s | Error=%s"
            ),
            target_url,
            reason,
        )


        return standard_result


    # ========================================================================
    # HEALTH CHECK
    # ========================================================================

    def health_check(
        self
    ) -> Dict[str, Any]:
        """
        Return HTML Agent health information.
        """

        model_available = False


        if self.predictor is not None:

            model = (
                self._get_predictor_model()
            )

            model_available = (
                model is not None
            )


        explainer_available = (
            self.explainer is not None
        )


        return {

            "agent_name":
                self.AGENT_NAME,

            "agent_version":
                self.AGENT_VERSION,

            "initialized":
                self.is_initialized,

            "model_available":
                model_available,

            "predictor_available":
                self.predictor is not None,

            "risk_scorer_available":
                self.risk_scorer is not None,

            "explainer_available":
                explainer_available,

            "feature_count":
                len(
                    HTMLFeatureSchema.get_schema()
                ),

            "expected_feature_count":
                self.EXPECTED_FEATURE_COUNT,

            "status":
                (
                    "healthy"
                    if (
                        self.is_initialized
                        and self.predictor is not None
                        and self.risk_scorer is not None
                        and self.explainer is not None
                    )
                    else "degraded"
                ),
        }


    # ========================================================================
    # AGENT INFORMATION
    # ========================================================================

    def get_agent_info(
        self
    ) -> Dict[str, Any]:
        """
        Return metadata about the HTML Agent.
        """

        return {

            "agent_name":
                self.AGENT_NAME,

            "agent_version":
                self.AGENT_VERSION,

            "agent_type":
                self.AGENT_TYPE,

            "feature_key":
                self.FEATURE_KEY,

            "feature_count":
                len(
                    HTMLFeatureSchema.get_schema()
                ),

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
                        ),
                },

            "health":
                self.health_check(),
        }


    # ========================================================================
    # DIRECT FEATURE ANALYSIS
    # ========================================================================

    def analyze_features(
        self,
        html_features: Dict[str, Any],
        target_url: str = "unknown_target"
    ) -> Dict[str, Any]:
        """
        Convenience method for directly analyzing HTML features.
        """

        return self.analyze({

            "html_features":
                html_features,

            "metadata":
                {

                    "target_url":
                        target_url
                }
        })


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "HTMLAIAgent",
]