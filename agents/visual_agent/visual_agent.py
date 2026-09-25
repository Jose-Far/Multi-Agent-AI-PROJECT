"""
agents/visual_agent/visual_agent.py

Visual AI Agent
===============

Day 13 standardized-output version.

Architecture
------------

    Visual Feature Collection
                |
                v
        Visual Feature Schema
                |
                v
          Preprocessing
                |
                v
            Predictor
                |
        +-------+-------+
        |               |
        v               v
     ML Model       Heuristic
        |               |
        +-------+-------+
                |
                v
          Risk Scoring
                |
                v
          Explainability
                |
                v
          AgentResult
                |
                v
       Decision Fusion


Important security contract
---------------------------

Operational failure is NOT legitimate evidence.

Therefore:

    status = "error" / "unavailable"
    prediction = "unknown"
    probability = None
    confidence = 0.0
    risk_score = None
    risk_level = "unknown"
    signal_available = False

A fallback heuristic is allowed only when usable visual features
were successfully obtained.

Day 13 common AgentResult fields
--------------------------------

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

Backward-compatible Visual fields are also retained so the existing
orchestrator, reports, Fusion Engine, and UI can continue consuming
the richer Visual result.

No standalone execution hook is included.
"""

from __future__ import annotations

# ============================================================================
# STANDARD LIBRARY
# ============================================================================

import logging
import math

from datetime import datetime, timezone

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)

# ============================================================================
# VISUAL COMPONENTS
# ============================================================================

from .feature_schema import VisualFeatureSchema
from .preprocessing import VisualPreprocessor
from .predictor import VisualPredictor
from .risk_score import VisualRiskScorer
from .explain import VisualExplainer

try:
    from feature_extraction.visual.extractor import VisualFeatureExtractor
except ImportError:
    VisualFeatureExtractor = None


# ============================================================================
# COMMON AGENT RESULT
# ============================================================================

from agents.agent_result import (
    AgentResult,
    get_risk_level,
    normalize_confidence,
    normalize_probability,
    normalize_risk_score,
)


# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# VISUAL AI AGENT
# ============================================================================

class VisualAIAgent:
    """
    Master orchestration controller for the Visual AI Agent.

    Responsibilities
    -----------------

    1. Validate incoming unified feature vector.
    2. Extract visual feature block.
    3. Optionally extract visual features from screenshot input.
    4. Validate visual feature schema.
    5. Normalize feature aliases.
    6. Execute VisualPredictor.
    7. Execute VisualRiskScorer.
    8. Generate VisualExplainer output.
    9. Convert the result to common AgentResult format.
    10. Preserve backward-compatible Visual fields.
    """

    # =========================================================================
    # AGENT IDENTITY
    # =========================================================================

    AGENT_NAME = "Visual_AI_Agent"

    AGENT_VERSION = "3.0.0"

    AGENT_TYPE = "Visual Security Analysis"

    FEATURE_KEY = "visual_features"

    EXPECTED_FEATURE_COUNT = 12

    # =========================================================================
    # CLASSIFICATION CONTRACT
    # =========================================================================

    VALID_VERDICTS = {
        "legitimate",
        "phishing",
        "suspicious",
    }

    CLASS_PROBABILITY_KEYS = (
        "legitimate",
        "suspicious",
        "phishing",
    )

    # =========================================================================
    # FALLBACK CONFIGURATION
    # =========================================================================

    FALLBACK_CONFIDENCE = 0.0

    # =========================================================================
    # MODEL STATUS
    # =========================================================================

    MODEL_STATUS_ML = "available"

    MODEL_STATUS_HEURISTIC = "fallback"

    MODEL_STATUS_UNAVAILABLE = "unavailable"

    MODEL_STATUS_ERROR = "error"

    # =========================================================================
    # PREDICTION SOURCE
    # =========================================================================

    PREDICTION_SOURCE_ML = "trained_ml"

    PREDICTION_SOURCE_HEURISTIC = "fallback_heuristic"

    PREDICTION_SOURCE_UNAVAILABLE = "unavailable"

    PREDICTION_SOURCE_ERROR = "error"

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        model_path: Optional[str] = None,
    ) -> None:
        """
        Initialize Visual predictor, preprocessor, risk scorer,
        schema, and explainer.
        """

        logger.info(
            "Initializing %s version %s...",
            self.AGENT_NAME,
            self.AGENT_VERSION,
        )

        self.is_initialized = False

        self.initialization_error: Optional[str] = None

        self.predictor: Optional[
            VisualPredictor
        ] = None

        self.preprocessor: Optional[
            VisualPreprocessor
        ] = None

        self.risk_scorer: Optional[
            VisualRiskScorer
        ] = None

        self.explainer: Optional[
            VisualExplainer
        ] = None

        self.feature_schema = []

        try:

            # -----------------------------------------------------------------
            # Feature schema
            # -----------------------------------------------------------------

            self.feature_schema = (
                self._load_feature_schema()
            )

            self._validate_feature_schema()

            # -----------------------------------------------------------------
            # Preprocessor
            # -----------------------------------------------------------------

            self.preprocessor = (
                VisualPreprocessor()
            )

            # -----------------------------------------------------------------
            # Predictor
            # -----------------------------------------------------------------

            self.predictor = (
                self._initialize_predictor(
                    model_path
                )
            )

            # -----------------------------------------------------------------
            # Risk scorer
            # -----------------------------------------------------------------

            self.risk_scorer = (
                VisualRiskScorer()
            )

            # -----------------------------------------------------------------
            # Explainer
            # -----------------------------------------------------------------

            model = (
                self._get_predictor_model()
            )

            self.explainer = (
                VisualExplainer(
                    model=model
                )
            )

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
                "%s initialization failed: %s",
                self.AGENT_NAME,
                exc,
                exc_info=True,
            )

            raise

    # =========================================================================
    # SCHEMA
    # =========================================================================

    @staticmethod
    def _load_feature_schema():
        """
        Load the canonical Visual ML feature schema.
        """

        # Prefer the public schema API.
        get_schema = getattr(
            VisualFeatureSchema,
            "get_schema",
            None,
        )

        if callable(get_schema):

            schema = get_schema()

            if isinstance(
                schema,
                (list, tuple),
            ):

                return list(schema)

        # Fall back to known canonical schema.
        return [
            "screenshot_width",
            "screenshot_height",
            "image_entropy",
            "dominant_color_count",
            "text_density",
            "image_density",
            "form_area_ratio",
            "button_count",
            "logo_detected",
            "login_form_detected",
            "layout_complexity",
            "blank_area_ratio",
        ]

    # =========================================================================
    # FEATURE SCHEMA VALIDATION
    # =========================================================================

    def _validate_feature_schema(
        self,
    ) -> None:
        """
        Validate the central Visual feature contract.

        The Visual model expects exactly 12 canonical features.
        """

        if not self.feature_schema:

            raise RuntimeError(
                "Visual feature schema is empty."
            )

        if len(
            self.feature_schema
        ) != self.EXPECTED_FEATURE_COUNT:

            raise RuntimeError(
                (
                    "Visual Agent requires exactly "
                    f"{self.EXPECTED_FEATURE_COUNT} "
                    "features, but schema contains "
                    f"{len(self.feature_schema)}."
                )
            )

        if len(
            set(
                self.feature_schema
            )
        ) != len(
            self.feature_schema
        ):

            raise RuntimeError(
                "Visual feature schema contains duplicate feature names."
            )

        verify_method = getattr(
            VisualFeatureSchema,
            "verify_schema_integrity",
            None,
        )

        if callable(
            verify_method
        ):

            verify_method()

        logger.debug(
            "Visual feature schema validated: %s",
            self.feature_schema,
        )

    # =========================================================================
    # PREDICTOR INITIALIZATION
    # =========================================================================

    @staticmethod
    def _initialize_predictor(
        model_path: Optional[str],
    ) -> VisualPredictor:
        """
        Initialize VisualPredictor.

        Supports predictors that accept model_path and older predictors
        that do not.
        """

        if model_path is not None:

            try:

                return VisualPredictor(
                    model_path=model_path
                )

            except TypeError:

                logger.debug(
                    (
                        "VisualPredictor does not accept "
                        "model_path; using default initialization."
                    )
                )

        return VisualPredictor()

    # =========================================================================
    # PREDICTOR MODEL
    # =========================================================================

    def _get_predictor_model(
        self,
    ) -> Any:
        """
        Safely obtain the underlying predictor model for XAI.
        """

        if self.predictor is None:

            return None

        for attribute in (
            "model",
            "xgb_model",
            "classifier",
            "estimator",
        ):

            model = getattr(
                self.predictor,
                attribute,
                None,
            )

            if model is not None:

                return model

        return None

    # =========================================================================
    # PUBLIC ANALYSIS API
    # =========================================================================

    def analyze(
        self,
        unified_vector: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute complete Visual AI analysis.

        Input
        -----

        {
            "visual_features": {
                ...
            },
            "metadata": {
                "target_url": "https://example.com"
            }
        }

        Returns
        -------

        JSON-safe dictionary containing:

            Day-13 AgentResult fields

        plus:

            legacy Visual analysis fields.
        """

        logger.info(
            "%s analysis triggered.",
            self.AGENT_NAME,
        )

        # =====================================================================
        # TOP LEVEL INPUT VALIDATION
        # =====================================================================

        if not isinstance(
            unified_vector,
            Mapping,
        ):

            return self._build_error_result(
                error_message=(
                    "Unified feature vector was empty or invalid."
                ),
                target_url="unknown",
                error_type="VisualInputError",
            )

        if not unified_vector:

            return self._build_error_result(
                error_message=(
                    "Unified feature vector was empty."
                ),
                target_url="unknown",
                error_type="VisualInputError",
            )

        # =====================================================================
        # METADATA
        # =====================================================================

        metadata = unified_vector.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            Mapping,
        ):

            metadata = {}

        target_url = self._extract_target_url(
            unified_vector,
            metadata,
        )

        # =====================================================================
        # VISUAL FEATURES
        # =====================================================================

        visual_features = unified_vector.get(
            self.FEATURE_KEY,
            {},
        )

        if not isinstance(
            visual_features,
            Mapping,
        ):

            return self._build_error_result(
                error_message=(
                    "The 'visual_features' block must be a dictionary."
                ),
                target_url=target_url,
                error_type="VisualFeatureError",
            )

        # =====================================================================
        # SCREENSHOT-BASED FEATURE EXTRACTION
        # =====================================================================

        visual_features = self._ensure_visual_features(
            unified_vector=unified_vector,
            visual_features=visual_features,
        )

        if not visual_features:

            return self._build_unavailable_result(
                reason=(
                    "Missing visual feature extraction block in payload."
                ),
                target_url=target_url,
            )

        # =====================================================================
        # FEATURE VALIDATION
        # =====================================================================

        validation_result = (
            self._validate_visual_features(
                visual_features
            )
        )

        if validation_result.get(
            "missing_features"
        ):

            logger.warning(
                "Visual Agent has missing features for %s: %s",
                target_url,
                validation_result[
                    "missing_features"
                ],
            )

        # =====================================================================
        # FEATURE NORMALIZATION
        # =====================================================================

        normalized_input_features = (
            self._normalize_input_features(
                visual_features
            )
        )

        # =====================================================================
        # PREDICTOR AVAILABILITY
        # =====================================================================

        if self.predictor is None:

            return self._build_unavailable_result(
                reason=(
                    "Visual predictor is unavailable."
                ),
                target_url=target_url,
                feature_validation=validation_result,
            )

        # =====================================================================
        # ANALYSIS
        # =====================================================================

        try:

            # -----------------------------------------------------------------
            # Prediction
            # -----------------------------------------------------------------

            prediction_result = (
                self.predictor.predict(
                    dict(
                        normalized_input_features
                    )
                )
            )

            if not isinstance(
                prediction_result,
                Mapping,
            ):

                raise RuntimeError(
                    "VisualPredictor returned an invalid result."
                )

            # -----------------------------------------------------------------
            # Prediction
            # -----------------------------------------------------------------

            prediction = self._normalize_verdict(
                prediction_result.get(
                    "class_label",
                    prediction_result.get(
                        "prediction",
                        "unknown",
                    ),
                )
            )

            # -----------------------------------------------------------------
            # Confidence
            # -----------------------------------------------------------------

            confidence = normalize_confidence(
                prediction_result.get(
                    "confidence",
                    prediction_result.get(
                        "confidence_score",
                        0.0,
                    ),
                )
            )

            # -----------------------------------------------------------------
            # Phishing probability
            # -----------------------------------------------------------------

            phishing_probability = normalize_probability(
                prediction_result.get(
                    "phishing_probability"
                )
            )

            # -----------------------------------------------------------------
            # Class probabilities
            # -----------------------------------------------------------------

            class_probabilities = (
                self._normalize_class_probabilities(
                    prediction_result.get(
                        "class_probabilities",
                        {},
                    ),
                    legitimate_probability=(
                        prediction_result.get(
                            "legitimate_probability"
                        )
                    ),
                    phishing_probability=(
                        phishing_probability
                    ),
                )
            )

            # -----------------------------------------------------------------
            # Class index
            # -----------------------------------------------------------------

            class_index = (
                self._safe_integer(
                    prediction_result.get(
                        "class_index"
                    ),
                    minimum=0,
                    maximum=2,
                )
            )

            # -----------------------------------------------------------------
            # Model state
            # -----------------------------------------------------------------

            raw_model_status = str(
                prediction_result.get(
                    "model_status",
                    "unknown",
                )
            ).strip().lower()

            (
                model_status,
                prediction_source,
                model_based,
                fallback_used,
            ) = self._normalize_model_information(
                raw_model_status,
                prediction_result,
            )

            # -----------------------------------------------------------------
            # Processed features
            # -----------------------------------------------------------------

            features = (
                self._extract_processed_features(
                    prediction_result=prediction_result,
                    raw_features=normalized_input_features,
                )
            )

            # -----------------------------------------------------------------
            # Risk scoring
            # -----------------------------------------------------------------

            risk_assessment = (
                self._calculate_risk(
                    prediction_result=prediction_result,
                    prediction=prediction,
                    phishing_probability=phishing_probability,
                    features=features,
                )
            )

            risk_score = normalize_risk_score(
                risk_assessment.get(
                    "risk_score",
                    prediction_result.get(
                        "risk_score"
                    ),
                )
            )

            risk_level = get_risk_level(
                risk_score
            )

            # -----------------------------------------------------------------
            # Explainability
            # -----------------------------------------------------------------

            explanation_output = (
                self._generate_explanation(
                    target_url=target_url,
                    features=features,
                    prediction_result=prediction_result,
                    risk_assessment=risk_assessment,
                    prediction=prediction,
                    phishing_probability=phishing_probability,
                )
            )

            # -----------------------------------------------------------------
            # Risk factors
            # -----------------------------------------------------------------

            risk_factors = self._build_risk_factors(
                risk_assessment=risk_assessment,
                explanation=explanation_output,
            )

            # -----------------------------------------------------------------
            # Evidence
            # -----------------------------------------------------------------

            evidence = self._build_evidence(
                target_url=target_url,
                risk_assessment=risk_assessment,
                prediction_result=prediction_result,
                feature_count=len(
                    self.feature_schema
                ),
            )

            # -----------------------------------------------------------------
            # Analysis status
            # -----------------------------------------------------------------

            analysis_status = "success"

            if (
                model_status
                == self.MODEL_STATUS_HEURISTIC
            ):

                analysis_status = "partial"

            # -----------------------------------------------------------------
            # Standard AgentResult
            # -----------------------------------------------------------------

            standard_result = AgentResult(
                agent=self.AGENT_NAME,
                status=analysis_status,
                prediction=prediction,
                probability=phishing_probability,
                confidence=confidence,
                risk_score=risk_score,
                risk_level=risk_level,
                prediction_source=prediction_source,
                model_status=model_status,
                risk_factors=risk_factors,
                evidence=evidence,
                explanation=explanation_output,
                feature_key=self.FEATURE_KEY,
                metadata={
                    "agent_version": self.AGENT_VERSION,
                    "agent_type": self.AGENT_TYPE,
                    "target_url": target_url,
                    "class_index": class_index,
                    "class_probabilities": class_probabilities,
                    "model_based": model_based,
                    "fallback_used": fallback_used,
                    "feature_validation": validation_result,
                    "prediction_method": prediction_source,
                },
            )

            result = standard_result.to_dict()

            # =================================================================
            # BACKWARD-COMPATIBLE VISUAL FIELDS
            # =================================================================

            result.update(
                {
                    "agent_name": self.AGENT_NAME,

                    "agent_version": self.AGENT_VERSION,

                    "target_url": target_url,

                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),

                    "analysis_status":
                        analysis_status,

                    "signal_available":
                        analysis_status
                        in {
                            "success",
                            "partial",
                        },

                    "verdict":
                        prediction,

                    "class_index":
                        class_index,

                    "confidence_score":
                        confidence,

                    "phishing_probability":
                        phishing_probability,

                    "class_probabilities":
                        class_probabilities,

                    "legitimate_probability":
                        class_probabilities.get(
                            "legitimate"
                        ),

                    "risk_score":
                        risk_score,

                    "risk_level":
                        risk_level,

                    "features_evaluated_count":
                        len(
                            self.feature_schema
                        ),

                    "feature_schema":
                        list(
                            self.feature_schema
                        ),

                    "features":
                        dict(
                            features
                        ),

                    "model_status":
                        model_status,

                    "prediction_method":
                        prediction_source,

                    "model":
                        {
                            "status":
                                model_status,

                            "prediction_source":
                                prediction_source,

                            "model_based":
                                model_based,

                            "fallback_used":
                                fallback_used,

                            "model_loaded":
                                bool(
                                    getattr(
                                        self.predictor,
                                        "is_model_loaded",
                                        False,
                                    )
                                ),
                        },

                    "model_metadata":
                        {
                            "model_type":
                                "XGBoost",

                            "model_status":
                                model_status,

                            "prediction_source":
                                prediction_source,

                            "fallback_used":
                                fallback_used,

                            "model_based":
                                model_based,

                            "class_mapping":
                                {
                                    "0":
                                        "legitimate",

                                    "1":
                                        "phishing",
                                },
                        },

                    "feature_validation":
                        self._json_safe_value(
                            validation_result
                        ),

                    "fusion_metadata":
                        {
                            "agent_type":
                                "visual",

                            "usable_for_fusion":
                                analysis_status
                                in {
                                    "success",
                                    "partial",
                                },

                            "model_based":
                                model_based,

                            "heuristic_fallback":
                                fallback_used,

                            "signal_available":
                                analysis_status
                                in {
                                    "success",
                                    "partial",
                                },

                            "requires_ml_caution":
                                prediction_source
                                != self.PREDICTION_SOURCE_ML,
                        },

                    "training_evaluation":
                        self._json_safe_value(
                            prediction_result.get(
                                "training_evaluation",
                                {},
                            )
                        ),
                }
            )

            # Ensure common standard fields remain authoritative.
            result["agent"] = self.AGENT_NAME
            result["status"] = analysis_status
            result["prediction"] = prediction
            result["probability"] = phishing_probability
            result["confidence"] = confidence
            result["risk_score"] = risk_score
            result["risk_level"] = risk_level
            result["prediction_source"] = prediction_source
            result["model_status"] = model_status
            result["risk_factors"] = risk_factors
            result["evidence"] = evidence
            result["explanation"] = explanation_output

            return self._json_safe_value(
                result
            )

        except Exception as exc:

            logger.error(
                "%s analysis failed for %s: %s",
                self.AGENT_NAME,
                target_url,
                exc,
                exc_info=True,
            )

            return self._build_error_result(
                error_message=str(exc),
                target_url=target_url,
                error_type=type(exc).__name__,
                feature_validation=validation_result,
            )

    # =========================================================================
    # SCREENSHOT / FEATURE ENSURE
    # =========================================================================

    def _ensure_visual_features(
        self,
        unified_vector: Mapping[str, Any],
        visual_features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Ensure visual features exist.

        If the caller supplied screenshot information but no extracted
        features, attempt extraction using the optional VisualFeatureExtractor.
        """

        if visual_features:

            return dict(
                visual_features
            )

        screenshot_path = (
            unified_vector.get(
                "screenshot_path"
            )
        )

        visual_block = (
            unified_vector.get(
                "visual",
                {}
            )
        )

        if isinstance(
            visual_block,
            Mapping,
        ):

            screenshot_path = (
                screenshot_path
                or visual_block.get(
                    "screenshot_path"
                )
            )

        if (
            not screenshot_path
            or VisualFeatureExtractor is None
        ):

            return {}

        try:

            extractor = (
                VisualFeatureExtractor()
            )

            extract_method = getattr(
                extractor,
                "extract",
                None,
            )

            if not callable(
                extract_method
            ):

                return {}

            extracted = extract_method(
                screenshot_path
            )

            if isinstance(
                extracted,
                Mapping,
            ):

                return dict(
                    extracted
                )

        except Exception as exc:

            logger.warning(
                "Visual feature extraction failed: %s",
                exc,
            )

        return {}

    # =========================================================================
    # TARGET URL
    # =========================================================================

    @staticmethod
    def _extract_target_url(
        unified_vector: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> str:
        """
        Extract target URL from common payload locations.
        """

        candidates = (
            unified_vector.get(
                "target_url"
            ),
            unified_vector.get(
                "url"
            ),
            metadata.get(
                "target_url"
            ),
            metadata.get(
                "url"
            ),
        )

        for value in candidates:

            if value:

                return str(
                    value
                )

        return "unknown"

    # =========================================================================
    # FEATURE VALIDATION
    # =========================================================================

    def _validate_visual_features(
        self,
        visual_features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate visual feature names against canonical schema.
        """

        received = set(
            visual_features.keys()
        )

        canonical = set(
            self.feature_schema
        )

        missing_features = sorted(
            canonical
            - received
        )

        extra_features = sorted(
            received
            - canonical
        )

        alias_resolutions = {}

        aliases = {
            "layout_complexity_score":
                "layout_complexity",

            "has_login_form_visual":
                "login_form_detected",

            "logo_presence":
                "logo_detected",

            "text_area_ratio":
                "text_density",

            "image_area_ratio":
                "image_density",
        }

        for alias, canonical_name in aliases.items():

            if (
                alias in received
                and canonical_name not in received
            ):

                alias_resolutions[
                    alias
                ] = canonical_name

        return {
            "schema_complete":
                len(
                    missing_features
                ) == 0,

            "supplied_feature_count":
                len(
                    received
                ),

            "expected_feature_count":
                len(
                    self.feature_schema
                ),

            "missing_features":
                missing_features,

            "extra_features":
                extra_features,

            "unknown_features":
                extra_features,

            "alias_resolutions":
                alias_resolutions,

            "warnings":
                [],
        }

    # =========================================================================
    # FEATURE NORMALIZATION
    # =========================================================================

    def _normalize_input_features(
        self,
        visual_features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalize known Visual feature aliases.

        Unknown non-model fields are retained for compatibility but the
        predictor receives only the canonical model feature set when possible.
        """

        features = dict(
            visual_features
        )

        aliases = {
            "layout_complexity_score":
                "layout_complexity",

            "has_login_form_visual":
                "login_form_detected",

            "logo_presence":
                "logo_detected",

            "text_area_ratio":
                "text_density",

            "image_area_ratio":
                "image_density",
        }

        for alias, canonical_name in aliases.items():

            if (
                alias in features
                and canonical_name not in features
            ):

                features[
                    canonical_name
                ] = features[
                    alias
                ]

        # ---------------------------------------------------------------------
        # Keep only canonical model features.
        # ---------------------------------------------------------------------

        canonical_features = {}

        for feature_name in self.feature_schema:

            if feature_name in features:

                canonical_features[
                    feature_name
                ] = features[
                    feature_name
                ]

        # ---------------------------------------------------------------------
        # If no canonical subset could be constructed, preserve original.
        # The preprocessor may still perform schema alignment.
        # ---------------------------------------------------------------------

        if canonical_features:

            return canonical_features

        return features

    # =========================================================================
    # RISK SCORING
    # =========================================================================

    def _calculate_risk(
        self,
        prediction_result: Mapping[str, Any],
        prediction: str,
        phishing_probability: Optional[float],
        features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute VisualRiskScorer using its supported API.

        The scorer's native output is preserved whenever available.
        """

        if self.risk_scorer is not None:

            # ---------------------------------------------------------------
            # Try common scorer signatures.
            # ---------------------------------------------------------------

            methods = (
                "calculate",
                "score",
                "assess",
            )

            for method_name in methods:

                method = getattr(
                    self.risk_scorer,
                    method_name,
                    None,
                )

                if not callable(
                    method
                ):

                    continue

                attempts = [
                    {
                        "prediction_result":
                            dict(
                                prediction_result
                            ),

                        "features":
                            dict(
                                features
                            ),
                    },

                    {
                        "prediction":
                            prediction,

                        "phishing_probability":
                            phishing_probability,

                        "features":
                            dict(
                                features
                            ),
                    },

                    {
                        "prediction":
                            prediction,

                        "probability":
                            phishing_probability,
                    },
                ]

                for kwargs in attempts:

                    try:

                        assessment = method(
                            **kwargs
                        )

                        if isinstance(
                            assessment,
                            Mapping,
                        ):

                            return dict(
                                assessment
                            )

                    except TypeError:

                        continue

                    except Exception as exc:

                        logger.debug(
                            (
                                "Visual risk scorer method "
                                "%s failed: %s"
                            ),
                            method_name,
                            exc,
                        )

        # ---------------------------------------------------------------------
        # Safe deterministic fallback.
        # ---------------------------------------------------------------------

        if phishing_probability is not None:

            risk_score = (
                phishing_probability
                * 100.0
            )

        elif prediction == "phishing":

            risk_score = 90.0

        elif prediction == "suspicious":

            risk_score = 50.0

        else:

            risk_score = 10.0

        return {
            "risk_score":
                round(
                    risk_score,
                    4,
                ),

            "risk_level":
                get_risk_level(
                    risk_score
                ),

            "risk_indicators":
                [],

            "triggered_indicators":
                [],

            "signal_available":
                True,
        }

    # =========================================================================
    # EXPLANATION
    # =========================================================================

    def _generate_explanation(
        self,
        target_url: str,
        features: Mapping[str, Any],
        prediction_result: Mapping[str, Any],
        risk_assessment: Mapping[str, Any],
        prediction: str,
        phishing_probability: Optional[float],
    ) -> Dict[str, Any]:
        """
        Generate a normalized explanation dictionary.
        """

        explanation: Dict[str, Any] = {}

        if self.explainer is not None:

            methods = (
                "explain",
                "generate_explanation",
            )

            for method_name in methods:

                method = getattr(
                    self.explainer,
                    method_name,
                    None,
                )

                if not callable(
                    method
                ):

                    continue

                attempts = [
                    {
                        "features":
                            dict(
                                features
                            ),

                        "prediction_result":
                            dict(
                                prediction_result
                            ),
                    },

                    {
                        "features":
                            dict(
                                features
                            ),

                        "prediction":
                            prediction,
                    },

                    {
                        "features":
                            dict(
                                features
                            ),
                    },
                ]

                for kwargs in attempts:

                    try:

                        generated = method(
                            **kwargs
                        )

                        if isinstance(
                            generated,
                            Mapping,
                        ):

                            explanation = dict(
                                generated
                            )

                            break

                    except TypeError:

                        continue

                    except Exception as exc:

                        logger.debug(
                            (
                                "Visual explainer method "
                                "%s failed: %s"
                            ),
                            method_name,
                            exc,
                        )

                if explanation:

                    break

        # ---------------------------------------------------------------------
        # Normalize explanation.
        # ---------------------------------------------------------------------

        summary = explanation.get(
            "summary"
        )

        if not summary:

            summary = (
                f"Visual analysis classified the target as "
                f"{prediction}."
            )

        return {
            "summary":
                str(
                    summary
                ),

            "method":
                explanation.get(
                    "method",
                    explanation.get(
                        "explanation_method",
                        "visual_analysis",
                    ),
                ),

            "top_shap_factors":
                self._safe_list(
                    explanation.get(
                        "top_shap_factors",
                        []
                    )
                ),

            "top_factors":
                self._safe_list(
                    explanation.get(
                        "top_factors",
                        []
                    )
                ),

            "risk_factors":
                self._safe_list(
                    explanation.get(
                        "risk_factors",
                        []
                    )
                ),

            "protective_factors":
                self._safe_list(
                    explanation.get(
                        "protective_factors",
                        []
                    )
                ),

            "key_indicators_triggered":
                self._safe_list(
                    explanation.get(
                        "key_indicators_triggered",
                        risk_assessment.get(
                            "triggered_indicators",
                            [],
                        ),
                    )
                ),

            "phishing_probability":
                phishing_probability,

            "explanation_reliability":
                explanation.get(
                    "explanation_reliability",
                    "available",
                ),
        }

    # =========================================================================
    # RISK FACTORS
    # =========================================================================

    @staticmethod
    def _build_risk_factors(
        risk_assessment: Mapping[str, Any],
        explanation: Mapping[str, Any],
    ):
        """
        Build unified Visual risk-factor list.
        """

        factors = []

        for key in (
            "triggered_indicators",
            "risk_indicators",
            "risk_factors",
        ):

            value = risk_assessment.get(
                key,
                [],
            )

            if isinstance(
                value,
                list,
            ):

                factors.extend(
                    value
                )

        explanation_factors = explanation.get(
            "risk_factors",
            [],
        )

        if isinstance(
            explanation_factors,
            list,
        ):

            factors.extend(
                explanation_factors
            )

        unique = []

        for factor in factors:

            if factor not in unique:

                unique.append(
                    factor
                )

        return unique

    # =========================================================================
    # EVIDENCE
    # =========================================================================

    def _build_evidence(
        self,
        target_url: str,
        risk_assessment: Mapping[str, Any],
        prediction_result: Mapping[str, Any],
        feature_count: int,
    ):
        """
        Build compact Fusion/UI evidence.
        """

        evidence = []

        evidence.append(
            {
                "type":
                    "analysis_target",

                "target":
                    target_url,
            }
        )

        evidence.append(
            {
                "type":
                    "feature_schema",

                "feature_count":
                    feature_count,

                "expected_feature_count":
                    self.EXPECTED_FEATURE_COUNT,

                "schema_complete":
                    feature_count
                    == self.EXPECTED_FEATURE_COUNT,
            }
        )

        indicators = risk_assessment.get(
            "triggered_indicators",
            [],
        )

        if isinstance(
            indicators,
            list,
        ) and indicators:

            evidence.append(
                {
                    "type":
                        "risk_indicators",

                    "indicators":
                        indicators,
                }
            )

        return evidence

    # =========================================================================
    # MODEL INFORMATION
    # =========================================================================

    def _normalize_model_information(
        self,
        raw_model_status: str,
        prediction_result: Mapping[str, Any],
    ):
        """
        Convert predictor-specific model status values into Day-13
        standardized model_status and prediction_source values.
        """

        model_loaded = bool(
            getattr(
                self.predictor,
                "is_model_loaded",
                False,
            )
        )

        raw_method = str(
            prediction_result.get(
                "prediction_method",
                ""
            )
        ).strip().lower()

        raw_source = str(
            prediction_result.get(
                "prediction_source",
                ""
            )
        ).strip().lower()

        # ---------------------------------------------------------------------
        # Explicit heuristic indicators
        # ---------------------------------------------------------------------

        heuristic_tokens = {
            "heuristic",
            "fallback",
            "fallback_heuristic",
            "rule_based",
            "rule-based",
        }

        if (
            raw_method in heuristic_tokens
            or raw_source in heuristic_tokens
            or "heuristic" in raw_model_status
            or "fallback" in raw_model_status
        ):

            return (
                self.MODEL_STATUS_HEURISTIC,
                self.PREDICTION_SOURCE_HEURISTIC,
                False,
                True,
            )

        # ---------------------------------------------------------------------
        # Loaded ML model
        # ---------------------------------------------------------------------

        if (
            model_loaded
            or raw_model_status in {
                "loaded_ml_model",
                "available",
                "loaded",
                "ml",
                "xgboost",
                "trained_ml",
            }
            or raw_source == "trained_ml"
        ):

            return (
                self.MODEL_STATUS_ML,
                self.PREDICTION_SOURCE_ML,
                True,
                False,
            )

        # ---------------------------------------------------------------------
        # Explicit unavailable
        # ---------------------------------------------------------------------

        if raw_model_status in {
            "unavailable",
            "disabled",
            "not_loaded",
            "missing",
        }:

            return (
                self.MODEL_STATUS_UNAVAILABLE,
                self.PREDICTION_SOURCE_UNAVAILABLE,
                False,
                False,
            )

        # ---------------------------------------------------------------------
        # Explicit error
        # ---------------------------------------------------------------------

        if raw_model_status in {
            "error",
            "error_fallback",
            "failed",
        }:

            return (
                self.MODEL_STATUS_ERROR,
                self.PREDICTION_SOURCE_ERROR,
                False,
                False,
            )

        # ---------------------------------------------------------------------
        # Conservative default.
        #
        # Never claim ML when the predictor did not explicitly establish it.
        # ---------------------------------------------------------------------

        return (
            self.MODEL_STATUS_HEURISTIC,
            self.PREDICTION_SOURCE_HEURISTIC,
            False,
            True,
        )

    # =========================================================================
    # SUCCESS / PARTIAL STANDARD RESULT
    # =========================================================================

    def _build_standard_success_result(
        self,
        prediction: str,
        probability: Optional[float],
        confidence: float,
        risk_score: Optional[float],
        prediction_source: str,
        model_status: str,
        risk_factors,
        evidence,
        explanation,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Helper for producing a Day-13 compliant AgentResult.
        """

        result = AgentResult(
            agent=self.AGENT_NAME,
            status=status,
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
                if isinstance(
                    risk_factors,
                    list,
                )
                else []
            ),
            evidence=(
                evidence
                if isinstance(
                    evidence,
                    list,
                )
                else []
            ),
            explanation=(
                explanation
                if isinstance(
                    explanation,
                    Mapping,
                )
                else {}
            ),
            feature_key=self.FEATURE_KEY,
            metadata=(
                metadata
                if metadata is not None
                else {}
            ),
        )

        return result.to_dict()

    # =========================================================================
    # UNAVAILABLE RESULT
    # =========================================================================

    def _build_unavailable_result(
        self,
        reason: Optional[str] = None,
        target_url: str = "unknown",
        feature_validation: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Build a secure unavailable result.

        Missing data MUST NOT become legitimate evidence.
        """

        result = AgentResult.unavailable(
            agent=self.AGENT_NAME,
            reason=reason,
            feature_key=self.FEATURE_KEY,
            metadata={
                "agent_version":
                    self.AGENT_VERSION,

                "agent_type":
                    self.AGENT_TYPE,

                "target_url":
                    target_url,

                "feature_validation":
                    dict(
                        feature_validation
                        or {}
                    ),
            },
        ).to_dict()

        result.update(
            {
                "agent_name":
                    self.AGENT_NAME,

                "agent_version":
                    self.AGENT_VERSION,

                "target_url":
                    target_url,

                "timestamp":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

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

                        "suspicious":
                            None,

                        "phishing":
                            None,
                    },

                "risk_score":
                    None,

                "risk_level":
                    "unknown",

                "features_evaluated_count":
                    0,

                "feature_schema":
                    list(
                        self.feature_schema
                    ),

                "features":
                    {},

                "model_status":
                    self.MODEL_STATUS_UNAVAILABLE,

                "prediction_method":
                    self.PREDICTION_SOURCE_UNAVAILABLE,

                "model":
                    {
                        "status":
                            self.MODEL_STATUS_UNAVAILABLE,

                        "prediction_source":
                            self.PREDICTION_SOURCE_UNAVAILABLE,

                        "model_based":
                            False,

                        "fallback_used":
                            False,

                        "model_loaded":
                            False,
                    },

                "feature_validation":
                    dict(
                        feature_validation
                        or {
                            "schema_complete":
                                False,

                            "missing_features":
                                list(
                                    self.feature_schema
                                ),

                            "extra_features":
                                [],

                            "unknown_features":
                                [],

                            "alias_resolutions":
                                {},

                            "warnings":
                                [],
                        }
                    ),

                "fusion_metadata":
                    {
                        "agent_type":
                            "visual",

                        "usable_for_fusion":
                            False,

                        "model_based":
                            False,

                        "heuristic_fallback":
                            False,

                        "signal_available":
                            False,

                        "requires_ml_caution":
                            True,
                    },
            }
        )

        return self._json_safe_value(
            result
        )

    # =========================================================================
    # ERROR RESULT
    # =========================================================================

    def _build_error_result(
        self,
        error_message: str,
        target_url: str = "unknown",
        error_type: Optional[str] = None,
        feature_validation: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Build a secure error result.

        IMPORTANT:

            error != suspicious
            error != legitimate

        An operational fault produces no mathematical security signal.
        """

        result = AgentResult.error_result(
            agent=self.AGENT_NAME,
            error=error_message,
            error_type=(
                error_type
                or "VisualAgentError"
            ),
            feature_key=self.FEATURE_KEY,
            metadata={
                "agent_version":
                    self.AGENT_VERSION,

                "agent_type":
                    self.AGENT_TYPE,

                "target_url":
                    target_url,

                "feature_validation":
                    dict(
                        feature_validation
                        or {}
                    ),
            },
        ).to_dict()

        result.update(
            {
                "agent_name":
                    self.AGENT_NAME,

                "agent_version":
                    self.AGENT_VERSION,

                "target_url":
                    target_url,

                "timestamp":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "analysis_status":
                    "error",

                "signal_available":
                    False,

                "error_details":
                    str(
                        error_message
                    ),

                "verdict":
                    "unknown",

                "class_index":
                    None,

                "confidence_score":
                    0.0,

                "class_probabilities":
                    {
                        "legitimate":
                            None,

                        "suspicious":
                            None,

                        "phishing":
                            None,
                    },

                "phishing_probability":
                    None,

                "legitimate_probability":
                    None,

                "risk_score":
                    None,

                "risk_level":
                    "unknown",

                "features_evaluated_count":
                    0,

                "feature_schema":
                    list(
                        self.feature_schema
                    ),

                "features":
                    {},

                "model_status":
                    self.MODEL_STATUS_ERROR,

                "prediction_method":
                    self.PREDICTION_SOURCE_ERROR,

                "model":
                    {
                        "status":
                            self.MODEL_STATUS_ERROR,

                        "prediction_source":
                            self.PREDICTION_SOURCE_ERROR,

                        "model_based":
                            False,

                        "fallback_used":
                            False,

                        "model_loaded":
                            False,
                    },

                "explanation":
                    {
                        "mode":
                            "error",

                        "summary":
                            (
                                "Visual Agent encountered an "
                                "operational error. No security "
                                "signal was produced."
                            ),

                        "top_shap_factors":
                            [],

                        "top_factors":
                            [],

                        "risk_factors":
                            [],

                        "protective_factors":
                            [],

                        "key_indicators_triggered":
                            [],

                        "explanation_reliability":
                            "unavailable",
                    },

                "feature_validation":
                    dict(
                        feature_validation
                        or {
                            "schema_complete":
                                False,

                            "missing_features":
                                [],

                            "extra_features":
                                [],

                            "unknown_features":
                                [],

                            "alias_resolutions":
                                {},

                            "warnings":
                                [],
                        }
                    ),

                "fusion_metadata":
                    {
                        "agent_type":
                            "visual",

                        "usable_for_fusion":
                            False,

                        "model_based":
                            False,

                        "heuristic_fallback":
                            False,

                        "signal_available":
                            False,

                        "requires_ml_caution":
                            True,
                    },
            }
        )

        return self._json_safe_value(
            result
        )

    # =========================================================================
    # VERDICT NORMALIZATION
    # =========================================================================

    @classmethod
    def _normalize_verdict(
        cls,
        value: Any,
    ) -> str:
        """
        Normalize predictor-specific class labels.
        """

        if value is None:

            return "unknown"

        normalized = str(
            value
        ).strip().lower()

        aliases = {
            "benign":
                "legitimate",

            "safe":
                "legitimate",

            "legit":
                "legitimate",

            "malicious":
                "phishing",

            "phish":
                "phishing",

            "unknown":
                "unknown",
        }

        normalized = aliases.get(
            normalized,
            normalized,
        )

        if normalized in cls.VALID_VERDICTS:

            return normalized

        return "unknown"

    # =========================================================================
    # CLASS PROBABILITIES
    # =========================================================================

    @staticmethod
    def _normalize_class_probabilities(
        value: Any,
        legitimate_probability: Any = None,
        phishing_probability: Any = None,
    ) -> Dict[str, Optional[float]]:
        """
        Normalize Visual class probability output.
        """

        probabilities = {
            "legitimate":
                None,

            "suspicious":
                None,

            "phishing":
                None,
        }

        if isinstance(
            value,
            Mapping,
        ):

            for key in probabilities:

                if key in value:

                    probabilities[
                        key
                    ] = normalize_probability(
                        value.get(
                            key
                        )
                    )

        if (
            probabilities["legitimate"] is None
            and legitimate_probability is not None
        ):

            probabilities[
                "legitimate"
            ] = normalize_probability(
                legitimate_probability
            )

        if (
            probabilities["phishing"] is None
            and phishing_probability is not None
        ):

            probabilities[
                "phishing"
            ] = normalize_probability(
                phishing_probability
            )

        return probabilities

    # =========================================================================
    # PROCESSED FEATURES
    # =========================================================================

    @staticmethod
    def _extract_processed_features(
        prediction_result: Mapping[str, Any],
        raw_features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract JSON-safe processed features.
        """

        for key in (
            "features",
            "processed_features",
            "feature_values",
        ):

            value = prediction_result.get(
                key
            )

            if isinstance(
                value,
                Mapping,
            ):

                return dict(
                    value
                )

        return dict(
            raw_features
        )

    # =========================================================================
    # INTEGER SAFETY
    # =========================================================================

    @staticmethod
    def _safe_integer(
        value: Any,
        minimum: int,
        maximum: int,
    ) -> Optional[int]:
        """
        Safely normalize integer-like values.
        """

        if value is None:

            return None

        try:

            result = int(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

        return max(
            minimum,
            min(
                maximum,
                result,
            ),
        )

    # =========================================================================
    # LIST SAFETY
    # =========================================================================

    @staticmethod
    def _safe_list(
        value: Any,
    ):
        """
        Return a safe list.
        """

        if isinstance(
            value,
            list,
        ):

            return list(
                value
            )

        if isinstance(
            value,
            tuple,
        ):

            return list(
                value
            )

        return []

    # =========================================================================
    # JSON SAFETY
    # =========================================================================

    @classmethod
    def _json_safe_value(
        cls,
        value: Any,
    ) -> Any:
        """
        Recursively convert values into JSON-safe Python structures.
        """

        if value is None:

            return None

        if isinstance(
            value,
            bool,
        ):

            return value

        if isinstance(
            value,
            (str, int),
        ):

            return value

        if isinstance(
            value,
            float,
        ):

            if math.isfinite(
                value
            ):

                return value

            return None

        if isinstance(
            value,
            Mapping,
        ):

            return {
                str(
                    key
                ):
                    cls._json_safe_value(
                        item
                    )
                for key, item
                in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):

            return [
                cls._json_safe_value(
                    item
                )
                for item
                in value
            ]

        # ---------------------------------------------------------------------
        # NumPy-like scalar support without importing NumPy here.
        # ---------------------------------------------------------------------

        item_method = getattr(
            value,
            "item",
            None,
        )

        if callable(
            item_method
        ):

            try:

                return cls._json_safe_value(
                    item_method()
                )

            except Exception:

                pass

        # ---------------------------------------------------------------------
        # DataFrame / Series-like structures.
        # ---------------------------------------------------------------------

        to_dict = getattr(
            value,
            "to_dict",
            None,
        )

        if callable(
            to_dict
        ):

            try:

                return cls._json_safe_value(
                    to_dict()
                )

            except Exception:

                pass

        return str(
            value
        )

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:
        """
        Return operational health of the complete Visual Agent.
        """

        try:

            predictor_model_loaded = bool(
                getattr(
                    self.predictor,
                    "is_model_loaded",
                    False,
                )
            )

            # -----------------------------------------------------------------
            # Predictor health
            # -----------------------------------------------------------------

            predictor_health = {}

            predictor_health_method = getattr(
                self.predictor,
                "health_check",
                None,
            )

            if callable(
                predictor_health_method
            ):

                try:

                    predictor_health = (
                        predictor_health_method()
                    )

                except Exception as exc:

                    predictor_health = {
                        "status":
                            "error",

                        "error":
                            str(
                                exc
                            ),
                    }

            # -----------------------------------------------------------------
            # XAI health
            # -----------------------------------------------------------------

            xai_health = {}

            xai_health_method = getattr(
                self.explainer,
                "health_check",
                None,
            )

            if callable(
                xai_health_method
            ):

                try:

                    xai_health = (
                        xai_health_method()
                    )

                except Exception as exc:

                    xai_health = {
                        "status":
                            "error",

                        "error":
                            str(
                                exc
                            ),
                    }

            return {
                "agent_name":
                    self.AGENT_NAME,

                "agent_version":
                    self.AGENT_VERSION,

                "status":
                    "ready"
                    if self.is_initialized
                    else "error",

                "feature_count":
                    len(
                        self.feature_schema
                    ),

                "feature_schema_valid":
                    len(
                        self.feature_schema
                    )
                    == self.EXPECTED_FEATURE_COUNT,

                "model_loaded":
                    predictor_model_loaded,

                "model_status":
                    (
                        self.MODEL_STATUS_ML
                        if predictor_model_loaded
                        else self.MODEL_STATUS_HEURISTIC
                    ),

                "predictor":
                    predictor_health,

                "xai":
                    xai_health,

                "components":
                    {
                        "preprocessor":
                            self.preprocessor
                            is not None,

                        "predictor":
                            self.predictor
                            is not None,

                        "risk_scorer":
                            self.risk_scorer
                            is not None,

                        "explainer":
                            self.explainer
                            is not None,
                    },
            }

        except Exception as exc:

            return {
                "agent_name":
                    self.AGENT_NAME,

                "agent_version":
                    self.AGENT_VERSION,

                "status":
                    "error",

                "error":
                    str(
                        exc
                    ),
            }


# ============================================================================
# PUBLIC EXPORT
# ============================================================================

__all__ = [
    "VisualAIAgent",
]