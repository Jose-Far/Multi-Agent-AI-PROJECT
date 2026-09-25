"""
URL AI Agent
============

Day 13 - Standardized Agent Output

The URL AI Agent is responsible for:

    URL Features
         ↓
    URLPredictor
         ↓
    Prediction
         ↓
    URLRiskScorer
         ↓
    URLExplainer
         ↓
    AgentResult
         ↓
    Orchestrator / Fusion

Important Day 13 principles:

    - Successful analysis returns AgentResult.
    - Partial analysis returns AgentResult.
    - Unavailable analysis returns AgentResult with prediction=unknown.
    - Errors return AgentResult with prediction=unknown.
    - Unavailable/error is NEVER converted into legitimate.
    - Trained ML and heuristic fallback are explicitly distinguished.
    - URL-specific legacy fields are retained temporarily for compatibility
      with the existing Orchestrator/Fusion layer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from .predictor import URLPredictor
from .risk_score import URLRiskScorer
from .explain import URLExplainer
from feature_extraction.url.extractor import URLExtractor

from agents.agent_result import (
    AgentResult,
    STATUS_SUCCESS,
    STATUS_PARTIAL,
    STATUS_UNAVAILABLE,
    STATUS_ERROR,
    PREDICTION_LEGITIMATE,
    PREDICTION_PHISHING,
    PREDICTION_SUSPICIOUS,
    PREDICTION_MALICIOUS,
    PREDICTION_UNKNOWN,
    PREDICTION_SOURCE_TRAINED_ML,
    PREDICTION_SOURCE_HEURISTIC,
    PREDICTION_SOURCE_FALLBACK_HEURISTIC,
    PREDICTION_SOURCE_UNAVAILABLE,
    PREDICTION_SOURCE_ERROR,
    MODEL_STATUS_AVAILABLE,
    MODEL_STATUS_FALLBACK,
    MODEL_STATUS_UNAVAILABLE,
    MODEL_STATUS_ERROR,
)


logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

URL_AGENT_NAME = "URL_AI_Agent"


# ============================================================================
# URL AI AGENT
# ============================================================================

class URLAIAgent:
    """
    Main Controller for the URL AI Agent.

    Responsibilities
    ----------------
    1. Receive URL features from the unified feature vector.
    2. Pass raw URL features to URLPredictor.
    3. Ensure the predictor uses the authoritative URLFeatureSchema.
    4. Obtain the ML/heuristic prediction.
    5. Calculate the URL-specific risk score.
    6. Generate the XAI explanation.
    7. Return a standardized AgentResult dictionary.

    Pipeline
    --------

        Unified Feature Vector
                ↓
          url_features
                ↓
          URLPredictor
                ↓
       URLFeatureSchema
                ↓
        URLPreprocessor
                ↓
        Feature Validation
                ↓
       XGBoost / Fallback
                ↓
        Phishing Probability
                ↓
         URLRiskScorer
                ↓
          URLExplainer
                ↓
          AgentResult
                ↓
       Orchestrator / Fusion
    """


    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(
        self,
        model_path: Optional[str] = None
    ):
        """
        Initialize the URL AI Agent.

        Args:
            model_path:
                Optional custom URL XGBoost model path.
        """

        logger.info(
            "Initializing URL AI Agent..."
        )

        try:

            # ----------------------------------------------------------------
            # Prediction engine
            # ----------------------------------------------------------------

            self.predictor = URLPredictor(
                model_path=model_path
            )

            # ----------------------------------------------------------------
            # Risk scoring engine
            # ----------------------------------------------------------------

            self.risk_scorer = URLRiskScorer()

            # ----------------------------------------------------------------
            # Explainability engine
            # ----------------------------------------------------------------

            self.explainer = URLExplainer(
                model=self.predictor.get_model()
            )

            logger.info(
                "URL AI Agent initialized successfully."
            )

            logger.info(
                "URL model status: %s",
                self.predictor.get_status()
            )

        except Exception as exc:

            logger.critical(
                (
                    "Critical failure during URL AI Agent "
                    "initialization: %s"
                ),
                str(exc),
                exc_info=True
            )

            raise


    # ========================================================================
    # STATUS
    # ========================================================================

    def get_status(
        self
    ) -> Dict[str, Any]:
        """
        Return URL Agent health/status information.

        This method intentionally remains a health/status dictionary rather
        than AgentResult because it describes the agent itself rather than
        an analysis result.
        """

        predictor_status = {}

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

        return {

            "agent_name":
                URL_AGENT_NAME,

            "status":
                (
                    "ready"
                    if self.predictor.is_model_loaded
                    else "degraded"
                ),

            "model_loaded":
                bool(
                    self.predictor.is_model_loaded
                ),

            "feature_count":
                getattr(
                    self.predictor,
                    "feature_count",
                    None
                ),

            "feature_schema":
                (
                    self.predictor.get_feature_schema()
                    if hasattr(
                        self.predictor,
                        "get_feature_schema"
                    )
                    else []
                ),

            "predictor":
                predictor_status,
        }


    # ========================================================================
    # MAIN ANALYSIS
    # ========================================================================

    def analyze(
        self,
        unified_vector: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the complete URL AI analysis pipeline.

        Accepted input forms:

            1. URL only:

                {
                    "url": "https://example.com"
                }


            2. URL + pre-extracted features:

                {
                    "url": "https://example.com",

                    "url_features": {
                        ...
                    },

                    "metadata": {
                        "target_url": "https://example.com"
                    }
                }

        If url_features are missing, URLExtractor is used automatically.

        The final result is always returned as a dictionary generated from
        AgentResult.
        """

        logger.info(
            "URL AI Agent analysis triggered."
        )

        # ====================================================================
        # STEP 1 - Validate input
        # ====================================================================

        if not isinstance(
            unified_vector,
            dict
        ):

            return self._get_error_response(
                error_message=(
                    "Unified feature vector must be a dictionary."
                ),
                target_url="unknown"
            )


        # ====================================================================
        # STEP 2 - Resolve target URL
        # ====================================================================

        metadata = unified_vector.get(
            "metadata",
            {}
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

            or ""
        )


        if not isinstance(
            target_url,
            str
        ):

            target_url = str(
                target_url
            )


        target_url = target_url.strip()


        if not target_url:

            return self._get_error_response(
                error_message=(
                    "No target URL was provided."
                ),
                target_url="unknown"
            )


        # ====================================================================
        # STEP 3 - Obtain URL features
        # ====================================================================

        url_features = unified_vector.get(
            "url_features"
        )


        # --------------------------------------------------------------------
        # Use orchestrator-provided features when available.
        # --------------------------------------------------------------------

        if url_features is not None:

            if not isinstance(
                url_features,
                dict
            ):

                return self._get_error_response(
                    error_message=(
                        "url_features must be a dictionary."
                    ),
                    target_url=target_url
                )


            if not url_features:

                url_features = None


        # --------------------------------------------------------------------
        # Extract URL features automatically when not supplied.
        # --------------------------------------------------------------------

        if url_features is None:

            try:

                logger.info(
                    (
                        "URL features missing. "
                        "Extracting features for: %s"
                    ),
                    target_url
                )

                url_features = (
                    URLExtractor(
                        target_url
                    ).extract()
                )

            except Exception as exc:

                logger.error(
                    (
                        "URL feature extraction failed "
                        "for '%s': %s"
                    ),
                    target_url,
                    str(exc),
                    exc_info=True
                )

                return self._get_unavailable_response(
                    target_url=target_url,
                    error_message=(
                        f"URL feature extraction failed: {str(exc)}"
                    )
                )


        if not url_features:

            return self._get_unavailable_response(
                target_url=target_url,
                error_message=(
                    "URL feature extraction returned no features."
                )
            )


        # ====================================================================
        # STEP 4 - Prediction
        # ====================================================================

        try:

            prediction_result = (
                self.predictor.predict(
                    url_features
                )
            )


            if not isinstance(
                prediction_result,
                dict
            ):

                raise RuntimeError(
                    "URLPredictor returned an invalid result."
                )


            # =================================================================
            # STEP 5 - Signal availability
            # =================================================================

            signal_available = bool(
                prediction_result.get(
                    "signal_available",
                    True
                )
            )


            if not signal_available:

                return self._get_unavailable_response(
                    target_url=target_url,
                    error_message=(
                        prediction_result.get(
                            "error",
                            "URL prediction unavailable."
                        )
                    )
                )


            # =================================================================
            # STEP 6 - Prediction
            # =================================================================

            raw_prediction = (

                prediction_result.get(
                    "class_label"
                )

                or prediction_result.get(
                    "verdict"
                )

                or prediction_result.get(
                    "prediction"
                )

                or PREDICTION_UNKNOWN
            )


            prediction = (
                self._normalize_prediction(
                    raw_prediction
                )
            )


            # =================================================================
            # STEP 7 - Confidence
            # =================================================================

            confidence = self._safe_float(
                prediction_result.get(
                    "confidence",
                    prediction_result.get(
                        "confidence_score",
                        0.0
                    )
                ),
                default=0.0
            )


            # =================================================================
            # STEP 8 - Phishing probability
            # =================================================================

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


            phishing_probability = (
                self._safe_probability(
                    phishing_probability
                )
            )


            # =================================================================
            # STEP 9 - Class probabilities
            # =================================================================

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


            class_probabilities = (
                self._normalize_probability_dict(
                    class_probabilities
                )
            )


            # =================================================================
            # STEP 10 - Get normalized model features
            # =================================================================

            features_dataframe = (
                prediction_result.get(
                    "features_dataframe"
                )
            )


            if features_dataframe is not None:

                if getattr(
                    features_dataframe,
                    "empty",
                    False
                ):

                    raise RuntimeError(
                        (
                            "URLPredictor returned "
                            "an empty feature dataframe."
                        )
                    )


                normalized_features = (
                    features_dataframe
                    .iloc[0]
                    .to_dict()
                )


            else:

                normalized_features = (
                    prediction_result.get(
                        "normalized_features"
                    )
                )


                if not isinstance(
                    normalized_features,
                    dict
                ):

                    from .feature_schema import (
                        URLFeatureSchema
                    )


                    aligned = (
                        URLFeatureSchema.align_and_validate(
                            url_features
                        )
                    )


                    if aligned.empty:

                        raise RuntimeError(
                            (
                                "URLFeatureSchema returned "
                                "an empty dataframe."
                            )
                        )


                    normalized_features = (
                        aligned
                        .iloc[0]
                        .to_dict()
                    )


            # =================================================================
            # STEP 11 - Risk scoring
            # =================================================================

            risk_assessment = (
                self.risk_scorer.calculate_risk(
                    probability=(
                        phishing_probability
                        if phishing_probability is not None
                        else 0.0
                    ),
                    features=normalized_features
                )
            )


            if not isinstance(
                risk_assessment,
                dict
            ):

                raise RuntimeError(
                    (
                        "URLRiskScorer returned "
                        "an invalid result."
                    )
                )


            # =================================================================
            # STEP 12 - Risk score
            # =================================================================

            risk_score = self._safe_risk_score(
                risk_assessment.get(
                    "risk_score"
                )
            )


            # =================================================================
            # STEP 13 - Explainability
            # =================================================================

            try:

                explanation = (
                    self.explainer.generate_explanation(
                        features=normalized_features,
                        prediction=prediction
                    )
                )

            except Exception as exc:

                logger.warning(
                    (
                        "URL explanation failed "
                        "for '%s': %s"
                    ),
                    target_url,
                    str(exc)
                )

                explanation = {

                    "summary":
                        (
                            "Prediction generated; "
                            "explanation unavailable."
                        ),

                    "top_factors":
                        []
                }


            if not isinstance(
                explanation,
                dict
            ):

                explanation = {

                    "summary":
                        "No explanation available.",

                    "top_factors":
                        []
                }


            # =================================================================
            # STEP 14 - Determine model source
            # =================================================================

            prediction_source = (
                self._get_prediction_source(
                    prediction_result
                )
            )


            model_status = (
                self._get_model_status(
                    prediction_result,
                    prediction_source
                )
            )


            # =================================================================
            # STEP 15 - Risk factors
            # =================================================================

            risk_factors = (
                self._extract_risk_factors(
                    risk_assessment
                )
            )


            # =================================================================
            # STEP 16 - Evidence
            # =================================================================

            evidence = (
                self._extract_evidence(
                    risk_assessment,
                    prediction_result
                )
            )


            # =================================================================
            # STEP 17 - Feature metadata
            # =================================================================

            feature_names = (
                prediction_result.get(
                    "feature_names",
                    []
                )
            )


            if not isinstance(
                feature_names,
                list
            ):

                feature_names = []


            feature_count = (
                prediction_result.get(
                    "feature_count",
                    len(normalized_features)
                )
            )


            # =================================================================
            # STEP 18 - Training metadata
            # =================================================================

            training_evaluation = (
                prediction_result.get(
                    "training_evaluation",
                    {}
                )
            )


            if not isinstance(
                training_evaluation,
                dict
            ):

                training_evaluation = {}


            # =================================================================
            # STEP 19 - Build AgentResult
            # =================================================================

            result = AgentResult.success(

                agent=URL_AGENT_NAME,

                prediction=prediction,

                probability=phishing_probability,

                confidence=confidence,

                risk_score=risk_score,

                prediction_source=prediction_source,

                model_status=model_status,

                risk_factors=risk_factors,

                evidence=evidence,

                explanation={

                    "summary":
                        explanation.get(
                            "summary",
                            "No explanation available."
                        ),

                    "top_factors":
                        explanation.get(
                            "top_factors",
                            []
                        ),

                    "key_indicators_triggered":
                        risk_assessment.get(
                            "triggered_indicators",
                            []
                        )
                },

                execution_time_ms=None,

                feature_key="url_features",

                metadata={

                    "target_url":
                        target_url,

                    "class_probabilities":
                        class_probabilities,

                    "features_evaluated_count":
                        len(
                            normalized_features
                        ),

                    "feature_count":
                        feature_count,

                    "feature_names":
                        feature_names,

                    "schema_version":
                        prediction_result.get(
                            "schema_version"
                        ),

                    "decision_threshold":
                        prediction_result.get(
                            "decision_threshold"
                        ),

                    "fallback_used":
                        bool(
                            prediction_result.get(
                                "fallback_used",
                                prediction_source
                                == PREDICTION_SOURCE_FALLBACK_HEURISTIC
                            )
                        ),

                    "training_evaluation":
                        training_evaluation,

                    "url_risk_assessment":
                        risk_assessment,
                }
            )


            # =================================================================
            # STEP 20 - Convert to dictionary
            # =================================================================

            standard_result = (
                result.to_dict()
            )


            # =================================================================
            # STEP 21 - Temporary legacy compatibility
            # =================================================================
            #
            # The current Orchestrator/Fusion layer still understands fields
            # such as:
            #
            #     agent_name
            #     analysis_status
            #     signal_available
            #     verdict
            #     confidence_score
            #     phishing_probability
            #
            # We keep them temporarily so Day 13 does not break the current
            # system. They can be removed after the Orchestrator is migrated
            # completely to AgentResult.
            # =================================================================

            standard_result.update({

                "target_url":
                    target_url,

                "timestamp":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "feature_key":
                    "url_features",

                "verdict":
                    prediction,

                "confidence_score":
                    confidence,

                "phishing_probability":
                    phishing_probability,

                "class_probabilities":
                    class_probabilities,

                "features_evaluated_count":
                    len(
                        normalized_features
                    ),

                "model_metadata": {

                    "model_type":
                        "XGBoost",

                    "model_status":
                        model_status,

                    "prediction_source":
                        prediction_source,

                    "fallback_used":
                        bool(
                            prediction_result.get(
                                "fallback_used",
                                prediction_source
                                == PREDICTION_SOURCE_FALLBACK_HEURISTIC
                            )
                        ),

                    "schema_version":
                        prediction_result.get(
                            "schema_version"
                        ),

                    "feature_count":
                        feature_count,

                    "feature_names":
                        feature_names,

                    "class_mapping": {

                        "0":
                            "legitimate",

                        "1":
                            "phishing"
                    },

                    "training_metrics":
                        training_evaluation
                }
            })


            logger.info(
                (
                    "URL AI Agent analysis completed: "
                    "prediction=%s, risk_score=%s, "
                    "risk_level=%s, source=%s"
                ),
                prediction,
                result.risk_score,
                result.risk_level,
                result.prediction_source
            )


            return standard_result


        except Exception as exc:

            logger.error(
                (
                    "URL Agent pipeline failed "
                    "for '%s': %s"
                ),
                target_url,
                str(exc),
                exc_info=True
            )


            return self._get_error_response(
                error_message=(
                    f"Agent pipeline processing failure: {str(exc)}"
                ),
                target_url=target_url
            )


    # ========================================================================
    # PREDICTION NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_prediction(
        prediction: Any
    ) -> str:
        """
        Normalize URL predictor labels into the project-wide prediction
        vocabulary.
        """

        if prediction is None:

            return PREDICTION_UNKNOWN


        value = str(
            prediction
        ).strip().lower()


        aliases = {

            "legit":
                PREDICTION_LEGITIMATE,

            "benign":
                PREDICTION_LEGITIMATE,

            "safe":
                PREDICTION_LEGITIMATE,

            "phish":
                PREDICTION_PHISHING,

            "malware":
                PREDICTION_MALICIOUS,

            "malicious":
                PREDICTION_MALICIOUS,

            "suspicious":
                PREDICTION_SUSPICIOUS,

            "unknown":
                PREDICTION_UNKNOWN,

            "unavailable":
                PREDICTION_UNKNOWN,

            "none":
                PREDICTION_UNKNOWN,
        }


        return aliases.get(
            value,
            (
                value
                if value in {
                    PREDICTION_LEGITIMATE,
                    PREDICTION_PHISHING,
                    PREDICTION_SUSPICIOUS,
                    PREDICTION_MALICIOUS,
                    PREDICTION_UNKNOWN,
                }
                else PREDICTION_UNKNOWN
            )
        )


    # ========================================================================
    # PREDICTION SOURCE
    # ========================================================================

    @staticmethod
    def _get_prediction_source(
        prediction_result: Dict[str, Any]
    ) -> str:
        """
        Convert predictor-specific source names into the Day 13 standard.

        Predictor currently uses values such as:

            xgboost
            heuristic_fallback

        AgentResult uses:

            trained_ml
            heuristic
            fallback_heuristic
            unavailable
            error
        """

        raw_source = (
            prediction_result.get(
                "prediction_source"
            )
        )


        fallback_used = bool(
            prediction_result.get(
                "fallback_used",
                False
            )
        )


        if fallback_used:

            return (
                PREDICTION_SOURCE_FALLBACK_HEURISTIC
            )


        if raw_source is None:

            if bool(
                prediction_result.get(
                    "model_loaded",
                    False
                )
            ):

                return (
                    PREDICTION_SOURCE_TRAINED_ML
                )

            return (
                PREDICTION_SOURCE_UNAVAILABLE
            )


        value = str(
            raw_source
        ).strip().lower()


        if value in {
            "xgboost",
            "trained_ml",
            "ml",
            "machine_learning",
            "trained",
        }:

            return (
                PREDICTION_SOURCE_TRAINED_ML
            )


        if value in {
            "heuristic",
            "rule_based",
            "rules",
        }:

            return (
                PREDICTION_SOURCE_HEURISTIC
            )


        if value in {
            "heuristic_fallback",
            "fallback",
            "fallback_heuristic",
            "fallback_rule_based",
        }:

            return (
                PREDICTION_SOURCE_FALLBACK_HEURISTIC
            )


        if value in {
            "error",
            "failed",
        }:

            return (
                PREDICTION_SOURCE_ERROR
            )


        if value in {
            "unavailable",
            "none",
            "unknown",
        }:

            return (
                PREDICTION_SOURCE_UNAVAILABLE
            )


        logger.warning(
            (
                "Unknown URL prediction source '%s'. "
                "Marking as unavailable."
            ),
            raw_source
        )


        return (
            PREDICTION_SOURCE_UNAVAILABLE
        )


    # ========================================================================
    # MODEL STATUS
    # ========================================================================

    @staticmethod
    def _get_model_status(
        prediction_result: Dict[str, Any],
        prediction_source: str
    ) -> str:
        """
        Convert predictor model state into the standard model_status field.
        """

        if prediction_source == (
            PREDICTION_SOURCE_FALLBACK_HEURISTIC
        ):

            return MODEL_STATUS_FALLBACK


        if prediction_source == (
            PREDICTION_SOURCE_TRAINED_ML
        ):

            return MODEL_STATUS_AVAILABLE


        if prediction_source == (
            PREDICTION_SOURCE_HEURISTIC
        ):

            return MODEL_STATUS_FALLBACK


        if prediction_source == (
            PREDICTION_SOURCE_ERROR
        ):

            return MODEL_STATUS_ERROR


        return MODEL_STATUS_UNAVAILABLE


    # ========================================================================
    # RISK SCORE
    # ========================================================================

    @staticmethod
    def _safe_risk_score(
        value: Any
    ) -> Optional[float]:
        """
        Safely normalize risk score to 0-100.

        None remains None.
        """

        if value is None:

            return None


        try:

            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return None


        numeric_value = max(
            0.0,
            min(
                100.0,
                numeric_value
            )
        )


        if numeric_value.is_integer():

            return int(
                numeric_value
            )


        return round(
            numeric_value,
            4
        )


    # ========================================================================
    # PROBABILITY
    # ========================================================================

    @staticmethod
    def _safe_probability(
        value: Any
    ) -> Optional[float]:
        """
        Safely normalize probability to 0.0-1.0.
        """

        if value is None:

            return None


        try:

            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return None


        numeric_value = max(
            0.0,
            min(
                1.0,
                numeric_value
            )
        )


        return round(
            numeric_value,
            6
        )


    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0
    ) -> float:
        """
        Safely convert numeric value.
        """

        if value is None:

            return default


        try:

            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return default


        return max(
            0.0,
            min(
                1.0,
                numeric_value
            )
        )


    # ========================================================================
    # PROBABILITY DICTIONARY
    # ========================================================================

    @classmethod
    def _normalize_probability_dict(
        cls,
        probabilities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Normalize class probability values.
        """

        normalized = {}


        for key, value in probabilities.items():

            normalized[
                str(key)
            ] = cls._safe_probability(
                value
            )


        return normalized


    # ========================================================================
    # RISK FACTORS
    # ========================================================================

    @staticmethod
    def _extract_risk_factors(
        risk_assessment: Dict[str, Any]
    ) -> list:
        """
        Extract explainability/risk indicators from URLRiskScorer.
        """

        factors = []


        for key in (
            "triggered_indicators",
            "risk_factors",
            "indicators",
            "factors",
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


            elif value is not None:

                factors.append(
                    value
                )


        return factors


    # ========================================================================
    # EVIDENCE
    # ========================================================================

    @staticmethod
    def _extract_evidence(
        risk_assessment: Dict[str, Any],
        prediction_result: Dict[str, Any]
    ) -> list:
        """
        Collect evidence useful for Fusion/UI explainability.
        """

        evidence = []


        # --------------------------------------------------------------------
        # Risk scorer evidence
        # --------------------------------------------------------------------

        for key in (
            "triggered_indicators",
            "risk_factors",
            "indicators",
        ):

            value = risk_assessment.get(
                key
            )


            if isinstance(
                value,
                list
            ):

                evidence.extend(
                    value
                )


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
    # UNAVAILABLE RESPONSE
    # ========================================================================

    def _get_unavailable_response(
        self,
        target_url: str,
        error_message: str
    ) -> Dict[str, Any]:
        """
        Return an unavailable AgentResult.

        IMPORTANT:

            unavailable != legitimate

        No mathematical risk signal is produced.
        """

        result = AgentResult.unavailable(

            agent=URL_AGENT_NAME,

            reason=error_message,

            feature_key="url_features",

            metadata={

                "target_url":
                    target_url,

                "error_details":
                    error_message
            }
        )


        standard_result = (
            result.to_dict()
        )


        # --------------------------------------------------------------------
        # Temporary compatibility fields
        # --------------------------------------------------------------------

        standard_result.update({

            "target_url":
                target_url,

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "feature_key":
                "url_features",

            "verdict":
                "unknown",

            "confidence_score":
                0.0,

            "phishing_probability":
                None,

            "class_probabilities": {

                "legitimate":
                    None,

                "phishing":
                    None
            },

            "features_evaluated_count":
                0,

            "error_details":
                error_message,

            "model_metadata": {

                "model_type":
                    "XGBoost",

                "model_status":
                    MODEL_STATUS_UNAVAILABLE,

                "prediction_source":
                    PREDICTION_SOURCE_UNAVAILABLE,

                "fallback_used":
                    False
            },

            "explanation": {

                "summary":
                    (
                        "URL analysis was unavailable. "
                        "No URL risk signal was generated."
                    ),

                "top_shap_factors":
                    [],

                "key_indicators_triggered":
                    [
                        "url_analysis_unavailable"
                    ]
            }
        })


        logger.warning(
            (
                "URL AI Agent unavailable for '%s': %s"
            ),
            target_url,
            error_message
        )


        return standard_result


    # ========================================================================
    # ERROR RESPONSE
    # ========================================================================

    def _get_error_response(
        self,
        error_message: str,
        target_url: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Return a safe error AgentResult.

        IMPORTANT:

        A pipeline error must NEVER become:

            prediction = legitimate
            risk_score = 0

        Instead:

            prediction = unknown
            risk_score = None
            signal_available = False
        """

        result = AgentResult.error_result(

            agent=URL_AGENT_NAME,

            error=error_message,

            error_type="URLAgentError",

            feature_key="url_features",

            metadata={

                "target_url":
                    target_url
            }
        )


        standard_result = (
            result.to_dict()
        )


        # --------------------------------------------------------------------
        # Temporary compatibility fields
        # --------------------------------------------------------------------

        standard_result.update({

            "target_url":
                target_url,

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "feature_key":
                "url_features",

            "verdict":
                "unknown",

            "confidence_score":
                0.0,

            "phishing_probability":
                None,

            "class_probabilities": {

                "legitimate":
                    None,

                "phishing":
                    None
            },

            "features_evaluated_count":
                0,

            "error_details":
                error_message,

            "model_metadata": {

                "model_type":
                    "XGBoost",

                "model_status":
                    MODEL_STATUS_ERROR,

                "prediction_source":
                    PREDICTION_SOURCE_ERROR,

                "fallback_used":
                    False,

                "class_mapping": {

                    "0":
                        "legitimate",

                    "1":
                        "phishing"
                }
            },

            "explanation": {

                "summary":
                    "URL Agent failed during analysis.",

                "top_shap_factors":
                    [],

                "key_indicators_triggered":
                    [
                        "agent_pipeline_error"
                    ]
            }
        })


        return standard_result


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "URLAIAgent",
]