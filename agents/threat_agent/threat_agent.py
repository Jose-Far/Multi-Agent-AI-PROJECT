"""
Threat Intelligence AI Agent
============================

File:
    agents/threat_agent/threat_agent.py

Purpose
-------
Master orchestration layer for the Threat Intelligence AI Agent.

Pipeline
--------

Unified Feature Vector
        |
        v
Threat Feature Extraction
        |
        v
ThreatFeatureSchema
        |
        v
ThreatPreprocessor
        |
        v
ThreatPredictor
        |
        +----------------------+
        |                      |
        v                      v
 Threat Probability       Normalized Features
        |                      |
        v                      v
 ThreatRiskScorer       ThreatExplainer
        |                      |
        +----------+-----------+
                   |
                   v
          Standard Agent Result
                   |
                   v
          Decision Fusion Engine


Classification
--------------

    0 -> clean_reputation
    1 -> malicious_threat

Canonical feature schema
------------------------

ThreatFeatureSchema remains the SINGLE SOURCE OF TRUTH.

The canonical schema contains 20 features.

The trained XGBoost model may legitimately consume a subset of the
canonical schema. The controller therefore distinguishes:

    canonical_feature_count
        20

from:

    model_feature_count
        13

This is intentional and is NOT treated as a schema failure as long as
every model feature exists inside the canonical schema.

No standalone execution hook is included.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from agents.agent_result import (
    AgentResult,
    MODEL_STATUS_AVAILABLE,
    MODEL_STATUS_FALLBACK,
    MODEL_STATUS_UNAVAILABLE,
    PREDICTION_LEGITIMATE,
    PREDICTION_PHISHING,
    PREDICTION_SOURCE_FALLBACK_HEURISTIC,
    PREDICTION_SOURCE_TRAINED_ML,
    PREDICTION_SOURCE_UNAVAILABLE,
    PREDICTION_UNKNOWN,
)

from .feature_schema import ThreatFeatureSchema
from .preprocessing import ThreatPreprocessor
from .predictor import ThreatPredictor
from .risk_score import ThreatRiskScorer
from .explain import ThreatExplainer


logger = logging.getLogger(__name__)


class ThreatIntelligenceAIAgent:
    """
    Master controller for the Threat Intelligence AI Agent.

    Responsibilities
    ----------------
    1. Validate the unified project input.
    2. Extract threat intelligence features.
    3. Validate the canonical feature contract.
    4. Run ThreatPredictor.
    5. Calculate policy-based threat risk.
    6. Generate explainability information.
    7. Produce a stable Decision Fusion payload.
    8. Fail safely without crashing the complete pipeline.
    """

    # =========================================================================
    # AGENT IDENTITY
    # =========================================================================

    AGENT_NAME = "Threat_Intel_Agent"

    AGENT_DISPLAY_NAME = (
        "Threat Intelligence AI Agent"
    )

    # =========================================================================
    # CLASSIFICATION
    # =========================================================================

    CLEAN_VERDICT = "clean_reputation"

    THREAT_VERDICT = "malicious_threat"

    CLASS_LABELS = {
        0: CLEAN_VERDICT,
        1: THREAT_VERDICT,
    }

    FIELD_SEMANTICS = {
        "prediction": (
            "Classifier label for this agent. Fusion vocabulary is "
            "legitimate | phishing | unknown. Independent of risk_score."
        ),
        "confidence": (
            "Model certainty in the predicted class, in [0, 1]. "
            "This is not phishing probability and not a risk score. "
            "0 means confidence was unavailable, not that the model "
            "is certain the site is benign."
        ),
        "risk_score": (
            "Operational threat-intelligence severity on 0-100, "
            "derived from malicious probability plus evidence. "
            "It is not a substitute for prediction or confidence."
        ),
        "risk_level": (
            "Bucket of risk_score: low (0-29), medium (30-59), "
            "high (60-79), critical (80-100)."
        ),
    }

    # =========================================================================
    # CANONICAL SCHEMA
    # =========================================================================

    EXPECTED_FEATURE_COUNT = 20

    # =========================================================================
    # CONSTRUCTOR
    # =========================================================================

    def __init__(
        self,
        model_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the Threat Intelligence AI Agent.

        Parameters
        ----------
        model_path:
            Optional path to the trained XGBoost model.
        """

        logger.info(
            "Initializing %s...",
            self.AGENT_DISPLAY_NAME,
        )

        # ---------------------------------------------------------------------
        # Canonical schema
        # ---------------------------------------------------------------------

        self.feature_schema = ThreatFeatureSchema

        self.expected_feature_columns = list(
            self.feature_schema.get_schema_columns()
        )

        self.expected_feature_count = int(
            self.feature_schema.get_feature_count()
        )

        self._validate_schema_contract()

        # ---------------------------------------------------------------------
        # Components
        # ---------------------------------------------------------------------

        self.preprocessor = ThreatPreprocessor()

        self.predictor = ThreatPredictor(
            model_path=model_path
        )

        self.risk_scorer = ThreatRiskScorer()

        self.explainer = ThreatExplainer(
            model=self.predictor.get_model()
        )

        # ---------------------------------------------------------------------
        # Model status
        # ---------------------------------------------------------------------

        self.model_loaded = (
            self.predictor.get_model_status()
            == "loaded_ml_model"
        )

        logger.info(
            "%s initialized successfully.",
            self.AGENT_DISPLAY_NAME,
        )

        logger.info(
            "Canonical Threat features: %d",
            self.expected_feature_count,
        )

        logger.info(
            "Threat model features: %d",
            self.get_model_feature_count(),
        )

        logger.info(
            "Threat model status: %s",
            self.get_model_status(),
        )

    # =========================================================================
    # SCHEMA VALIDATION
    # =========================================================================

    def _validate_schema_contract(self) -> None:
        """
        Validate the canonical ThreatFeatureSchema.

        This validates the 20-feature canonical schema only.

        The trained model is allowed to use a subset of the canonical
        schema, provided no model feature exists outside the canonical
        schema.
        """

        if self.expected_feature_count != (
            self.EXPECTED_FEATURE_COUNT
        ):
            raise ValueError(
                "Threat canonical feature count mismatch: "
                f"received={self.expected_feature_count}, "
                f"expected={self.EXPECTED_FEATURE_COUNT}"
            )

        if len(self.expected_feature_columns) != (
            self.EXPECTED_FEATURE_COUNT
        ):
            raise ValueError(
                "Threat canonical feature schema length is invalid."
            )

        if len(
            set(self.expected_feature_columns)
        ) != self.EXPECTED_FEATURE_COUNT:
            raise ValueError(
                "Threat canonical feature schema contains "
                "duplicate feature names."
            )

    # =========================================================================
    # MODEL STATUS
    # =========================================================================

    def get_model_status(self) -> str:
        """
        Return the current Threat model operational status.
        """

        try:
            return str(
                self.predictor.get_model_status()
            )
        except Exception:
            return "unknown"

    # =========================================================================
    # MODEL ACCESS
    # =========================================================================

    def get_model(self) -> Any:
        """
        Return the loaded underlying model when available.
        """

        try:
            return self.predictor.get_model()
        except Exception:
            return None

    # =========================================================================
    # MODEL FEATURE COUNT
    # =========================================================================

    def get_model_feature_count(self) -> int:
        """
        Return the number of features actually consumed by the
        trained model.

        This may be smaller than the canonical 20-feature schema.
        """

        try:
            return int(
                self.predictor.get_model_feature_count()
            )
        except Exception:
            return 0

    # =========================================================================
    # MODEL FEATURE SCHEMA
    # =========================================================================

    def get_model_feature_schema(self) -> list[str]:
        """
        Return the exact feature order consumed by the trained model.
        """

        try:
            return list(
                self.predictor.get_model_feature_schema()
            )
        except Exception:
            return []

    # =========================================================================
    # MODEL DIAGNOSTICS
    # =========================================================================

    def get_model_diagnostics(self) -> Dict[str, Any]:
        """
        Return model-level diagnostics.

        The model may use fewer than 20 canonical features.
        """

        try:
            diagnostics = (
                self.predictor.get_model_diagnostics()
            )

            if isinstance(
                diagnostics,
                dict,
            ):
                return diagnostics

        except Exception as exc:
            logger.warning(
                "Unable to obtain Threat model diagnostics: %s",
                exc,
            )

        model_features = (
            self.get_model_feature_schema()
        )

        canonical_features = (
            list(self.expected_feature_columns)
        )

        outside_schema = [
            feature
            for feature in model_features
            if feature not in canonical_features
        ]

        unused_canonical = [
            feature
            for feature in canonical_features
            if feature not in model_features
        ]

        return {
            "model_status":
                self.get_model_status(),

            "canonical_feature_count":
                len(canonical_features),

            "canonical_features":
                canonical_features,

            "model_feature_count":
                len(model_features),

            "model_features":
                model_features,

            "model_features_outside_canonical_schema":
                outside_schema,

            "canonical_features_not_used_by_model":
                unused_canonical,
        }

    # =========================================================================
    # FEATURE COUNT
    # =========================================================================

    def get_feature_count(self) -> int:
        """
        Return the canonical Threat feature count.
        """

        return self.expected_feature_count

    # =========================================================================
    # FEATURE SCHEMA
    # =========================================================================

    def get_feature_schema(self) -> list[str]:
        """
        Return a copy of the canonical Threat feature schema.
        """

        return list(
            self.expected_feature_columns
        )

    # =========================================================================
    # COMPLETE SCHEMA VALIDATION
    # =========================================================================

    def validate_feature_schema(
        self,
    ) -> Dict[str, Any]:
        """
        Validate schema consistency across all Threat Agent layers.

        Important
        ---------
        The XGBoost model may consume a subset of the canonical 20
        features.

        Therefore:

            model_features ⊆ canonical_features

        is considered valid.

        A model feature outside the canonical schema is an error.
        """

        errors = []

        canonical_schema = list(
            self.feature_schema.get_schema_columns()
        )

        canonical_count = int(
            self.feature_schema.get_feature_count()
        )

        # ---------------------------------------------------------------------
        # Preprocessor
        # ---------------------------------------------------------------------

        try:
            preprocessor_count = int(
                self.preprocessor.get_feature_count()
            )
        except Exception:
            preprocessor_count = 0

        try:
            preprocessor_order = list(
                self.preprocessor.get_feature_order()
            )
        except Exception:
            preprocessor_order = []

        # ---------------------------------------------------------------------
        # Predictor
        # ---------------------------------------------------------------------

        try:
            predictor_canonical_count = int(
                self.predictor.get_feature_count()
            )
        except Exception:
            predictor_canonical_count = 0

        try:
            predictor_schema = list(
                self.predictor.get_feature_schema()
            )
        except Exception:
            predictor_schema = []

        model_features = (
            self.get_model_feature_schema()
        )

        model_feature_count = len(
            model_features
        )

        # ---------------------------------------------------------------------
        # Risk scorer
        # ---------------------------------------------------------------------

        try:
            risk_count = int(
                self.risk_scorer.get_feature_count()
            )
        except Exception:
            risk_count = 0

        try:
            risk_schema = list(
                self.risk_scorer.get_feature_schema()
            )
        except Exception:
            risk_schema = []

        # ---------------------------------------------------------------------
        # Explainer
        # ---------------------------------------------------------------------

        try:
            explainer_count = int(
                self.explainer.get_feature_count()
            )
        except Exception:
            explainer_count = 0

        try:
            explainer_schema = list(
                self.explainer.get_feature_schema()
            )
        except Exception:
            explainer_schema = []

        # ---------------------------------------------------------------------
        # Layer schema checks
        # ---------------------------------------------------------------------

        preprocessor_matches = (
            canonical_schema
            == preprocessor_order
        )

        predictor_matches = (
            canonical_schema
            == predictor_schema
        )

        risk_matches = (
            canonical_schema
            == risk_schema
        )

        explainer_matches = (
            canonical_schema
            == explainer_schema
        )

        # ---------------------------------------------------------------------
        # Model subset check
        # ---------------------------------------------------------------------

        model_features_outside_schema = [
            feature
            for feature in model_features
            if feature not in canonical_schema
        ]

        model_is_subset = (
            len(model_features_outside_schema)
            == 0
        )

        # ---------------------------------------------------------------------
        # Count checks
        # ---------------------------------------------------------------------

        layer_counts_match = (
            canonical_count
            == preprocessor_count
            == predictor_canonical_count
            == risk_count
            == explainer_count
        )

        canonical_schema_valid = (
            canonical_count
            == self.EXPECTED_FEATURE_COUNT
            and len(canonical_schema)
            == self.EXPECTED_FEATURE_COUNT
            and len(set(canonical_schema))
            == self.EXPECTED_FEATURE_COUNT
        )

        # ---------------------------------------------------------------------
        # Collect errors
        # ---------------------------------------------------------------------

        if not canonical_schema_valid:
            errors.append(
                "Canonical Threat schema is invalid."
            )

        if not preprocessor_matches:
            errors.append(
                "ThreatPreprocessor schema does not match "
                "the canonical schema."
            )

        if not predictor_matches:
            errors.append(
                "ThreatPredictor canonical schema does not "
                "match the canonical schema."
            )

        if not risk_matches:
            errors.append(
                "ThreatRiskScorer schema does not match "
                "the canonical schema."
            )

        if not explainer_matches:
            errors.append(
                "ThreatExplainer schema does not match "
                "the canonical schema."
            )

        if not layer_counts_match:
            errors.append(
                "Threat component feature counts are inconsistent."
            )

        if not model_is_subset:
            errors.append(
                "The trained Threat model contains features "
                "outside the canonical schema: "
                + ", ".join(
                    model_features_outside_schema
                )
            )

        valid = (
            len(errors) == 0
        )

        return {
            "valid":
                valid,

            "expected_feature_count":
                self.EXPECTED_FEATURE_COUNT,

            "canonical_feature_count":
                canonical_count,

            "preprocessor_feature_count":
                preprocessor_count,

            "predictor_canonical_feature_count":
                predictor_canonical_count,

            "risk_scorer_feature_count":
                risk_count,

            "explainer_feature_count":
                explainer_count,

            "model_feature_count":
                model_feature_count,

            "model_features":
                model_features,

            "model_features_outside_schema":
                model_features_outside_schema,

            "model_is_subset_of_canonical_schema":
                model_is_subset,

            "schema_matches_every_layer":
                (
                    preprocessor_matches
                    and predictor_matches
                    and risk_matches
                    and explainer_matches
                ),

            "layer_counts_match":
                layer_counts_match,

            "feature_order":
                canonical_schema,

            "errors":
                errors,
        }

    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================

    def analyze(
        self,
        unified_vector: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute the complete Threat Intelligence pipeline.
        """

        logger.info(
            "%s analysis triggered.",
            self.AGENT_DISPLAY_NAME,
        )

        # =====================================================================
        # STEP 1 - INPUT VALIDATION
        # =====================================================================

        if not isinstance(
            unified_vector,
            dict,
        ):
            return self._get_fallback_response(
                error_message=(
                    "Unified feature vector must be a dictionary."
                )
            )

        if not unified_vector:
            return self._get_fallback_response(
                error_message=(
                    "Unified feature vector was empty."
                )
            )

        # =====================================================================
        # STEP 2 - METADATA
        # =====================================================================

        metadata = unified_vector.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        target_url = self._resolve_target_url(
            unified_vector,
            metadata,
        )

        logger.info(
            "Threat Intelligence target: %s",
            target_url,
        )

        # =====================================================================
        # STEP 3 - THREAT FEATURES
        # =====================================================================

        threat_features = unified_vector.get(
            "threat_features",
            {},
        )

        if threat_features is None:
            threat_features = {}

        if not isinstance(
            threat_features,
            dict,
        ):
            return self._get_fallback_response(
                error_message=(
                    "Threat feature block must be a dictionary."
                ),
                target_url=target_url,
            )

        if not threat_features:
            return self._get_fallback_response(
                error_message=(
                    "Missing threat intelligence feature block."
                ),
                target_url=target_url,
            )

        # =====================================================================
        # STEP 4 - INPUT DIAGNOSTICS
        # =====================================================================

        input_diagnostics = (
            self._inspect_input_features(
                threat_features
            )
        )

        logger.info(
            "Threat input diagnostics | "
            "recognized=%d | missing=%d | extra=%d",
            input_diagnostics[
                "recognized_feature_count"
            ],
            input_diagnostics[
                "missing_feature_count"
            ],
            input_diagnostics[
                "extra_feature_count"
            ],
        )

        # =====================================================================
        # STEP 5 - PREDICTION
        # =====================================================================

        try:

            logger.info(
                "Running Threat Predictor..."
            )

            prediction_result = (
                self.predictor.predict(
                    threat_features
                )
            )

            if not isinstance(
                prediction_result,
                dict,
            ):
                raise TypeError(
                    "ThreatPredictor returned "
                    "an invalid result."
                )

            # -----------------------------------------------------------------
            # Prediction
            # -----------------------------------------------------------------

            prediction = self._normalize_verdict(
                prediction_result.get(
                    "class_label",
                    prediction_result.get(
                        "prediction",
                        self.CLEAN_VERDICT,
                    ),
                )
            )

            # -----------------------------------------------------------------
            # Probability
            # -----------------------------------------------------------------

            threat_probability = (
                self._safe_probability(
                    prediction_result.get(
                        "threat_probability",
                        0.0,
                    )
                )
            )

            class_probabilities = (
                self._normalize_class_probabilities(
                    prediction_result.get(
                        "class_probabilities",
                        {},
                    ),
                    threat_probability,
                )
            )

            confidence = self._derive_model_confidence(
                prediction_result.get(
                    "confidence",
                    prediction_result.get(
                        "confidence_score",
                    ),
                ),
                threat_probability,
                class_probabilities,
                prediction,
            )

            # -----------------------------------------------------------------
            # Model status
            # -----------------------------------------------------------------

            model_status = str(
                prediction_result.get(
                    "model_status",
                    self.get_model_status(),
                )
            )

            # -----------------------------------------------------------------
            # Training evaluation
            # -----------------------------------------------------------------

            training_evaluation = (
                prediction_result.get(
                    "training_evaluation",
                    {},
                )
            )

            if not isinstance(
                training_evaluation,
                dict,
            ):
                training_evaluation = {}

            # =================================================================
            # STEP 6 - NORMALIZED FEATURES
            # =================================================================

            normalized_features = (
                self._extract_normalized_features(
                    prediction_result,
                    threat_features,
                )
            )

            # Risk and explanation operate on the complete canonical
            # feature vector, not merely the 13 model features.

            if len(
                normalized_features
            ) != self.expected_feature_count:
                raise ValueError(
                    "Normalized Threat feature vector contains "
                    f"{len(normalized_features)} features; "
                    f"expected {self.expected_feature_count}."
                )

            # =================================================================
            # STEP 7 - RISK SCORING
            # =================================================================

            logger.info(
                "Running Threat Risk Scorer..."
            )

            risk_assessment = (
                self.risk_scorer.calculate_risk(
                    probability=threat_probability,
                    features=normalized_features,
                )
            )

            if not isinstance(
                risk_assessment,
                dict,
            ):
                raise TypeError(
                    "ThreatRiskScorer returned "
                    "an invalid result."
                )

            # =================================================================
            # STEP 8 - EXPLANATION
            # =================================================================

            logger.info(
                "Running Threat Explainer..."
            )

            explanation = (
                self.explainer.generate_explanation(
                    features=normalized_features,
                    prediction=prediction,
                )
            )

            if not isinstance(
                explanation,
                dict,
            ):
                explanation = {
                    "summary":
                        "No explanation available.",
                    "top_factors":
                        [],
                    "explanation_method":
                        "fallback",
                }

            # =================================================================
            # STEP 9 - FINAL RESULT
            # =================================================================

            return self._build_standard_output(
                target_url=target_url,
                prediction=prediction,
                confidence=confidence,
                threat_probability=threat_probability,
                class_probabilities=class_probabilities,
                risk_assessment=risk_assessment,
                explanation=explanation,
                normalized_features=normalized_features,
                model_status=model_status,
                training_evaluation=training_evaluation,
                input_diagnostics=input_diagnostics,
            )

        except Exception as exc:

            logger.error(
                "Threat Intelligence analysis failed for '%s': %s",
                target_url,
                exc,
                exc_info=True,
            )

            return self._get_fallback_response(
                error_message=(
                    "Threat Intelligence Agent processing failure: "
                    f"{str(exc)}"
                ),
                target_url=target_url,
            )

    # =========================================================================
    # TARGET URL
    # =========================================================================

    @staticmethod
    def _resolve_target_url(
        unified_vector: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> str:
        """
        Resolve target URL.

        Priority:

            metadata.target_url
            unified_vector.url
            unified_vector.target_url
        """

        candidates = (
            metadata.get("target_url"),
            unified_vector.get("url"),
            unified_vector.get("target_url"),
        )

        for candidate in candidates:

            if candidate is None:
                continue

            value = str(candidate).strip()

            if value:
                return value

        return "unknown_target"

    # =========================================================================
    # INPUT DIAGNOSTICS
    # =========================================================================

    def _inspect_input_features(
        self,
        threat_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare incoming features with the canonical schema.
        """

        expected = set(
            self.expected_feature_columns
        )

        incoming = set(
            threat_features.keys()
        )

        recognized = expected.intersection(
            incoming
        )

        missing = [
            feature
            for feature in self.expected_feature_columns
            if feature not in incoming
        ]

        extra = sorted(
            incoming.difference(
                expected
            )
        )

        model_features = (
            self.get_model_feature_schema()
        )

        model_missing = [
            feature
            for feature in model_features
            if feature not in incoming
        ]

        return {
            "expected_feature_count":
                self.expected_feature_count,

            "received_feature_count":
                len(incoming),

            "recognized_feature_count":
                len(recognized),

            "missing_feature_count":
                len(missing),

            "missing_features":
                missing,

            "extra_feature_count":
                len(extra),

            "extra_features":
                extra,

            "model_feature_count":
                len(model_features),

            "model_missing_feature_count":
                len(model_missing),

            "model_missing_features":
                model_missing,

            "model_features":
                model_features,
        }

    # =========================================================================
    # NORMALIZED FEATURES
    # =========================================================================

    def _extract_normalized_features(
        self,
        prediction_result: Dict[str, Any],
        original_features: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Extract the complete canonical normalized feature vector.

        Preference:

            1. Predictor normalized_features
            2. Predictor processed_features
            3. Original feature values
            4. Safe zero default

        The output always follows the canonical 20-feature order.
        """

        candidate = prediction_result.get(
            "normalized_features"
        )

        if not isinstance(
            candidate,
            dict,
        ):
            candidate = prediction_result.get(
                "processed_features",
                {},
            )

        if not isinstance(
            candidate,
            dict,
        ):
            candidate = {}

        normalized: Dict[str, float] = {}

        for feature in self.expected_feature_columns:

            value = candidate.get(
                feature,
                original_features.get(
                    feature,
                    0.0,
                ),
            )

            normalized[feature] = (
                self._safe_numeric_value(
                    value
                )
            )

        return normalized

    # =========================================================================
    # VERDICT NORMALIZATION
    # =========================================================================

    @classmethod
    def _normalize_verdict(
        cls,
        value: Any,
    ) -> str:
        """
        Normalize arbitrary prediction values into the project verdict
        vocabulary.
        """

        if isinstance(
            value,
            bool,
        ):
            return (
                cls.THREAT_VERDICT
                if value
                else cls.CLEAN_VERDICT
            )

        if isinstance(
            value,
            (int, float),
        ):
            try:
                return (
                    cls.THREAT_VERDICT
                    if int(value) == 1
                    else cls.CLEAN_VERDICT
                )
            except (TypeError, ValueError):
                pass

        text = str(
            value
        ).strip().lower()

        if text in {
            "1",
            "true",
            "malicious",
            "malicious_threat",
            "threat",
            "phishing",
            "phishing_threat",
            "dangerous",
        }:
            return cls.THREAT_VERDICT

        return cls.CLEAN_VERDICT

    @classmethod
    def _to_fusion_prediction(
        cls,
        value: Any,
    ) -> str:
        """
        Convert internal threat labels to Fusion vocabulary.
        """

        normalized = cls._normalize_verdict(
            value
        )

        if normalized == cls.THREAT_VERDICT:
            return PREDICTION_PHISHING

        return PREDICTION_LEGITIMATE

    @classmethod
    def _derive_model_confidence(
        cls,
        confidence: Any,
        threat_probability: Any,
        class_probabilities: Any,
        prediction: Any = None,
    ) -> float:
        """
        Resolve certainty in the predicted class, in [0, 1].

        0 is reserved for "confidence unavailable".
        """

        fusion_prediction = cls._to_fusion_prediction(
            prediction
        ) if prediction is not None else None

        explicit = cls._safe_probability(
            confidence
        )

        predicted_class_probability = None

        if isinstance(class_probabilities, dict):

            if fusion_prediction == PREDICTION_PHISHING:

                predicted_class_probability = cls._safe_probability(
                    class_probabilities.get(
                        cls.THREAT_VERDICT,
                        class_probabilities.get(
                            "phishing",
                            class_probabilities.get(
                                "malicious",
                                threat_probability,
                            ),
                        ),
                    )
                )

            elif fusion_prediction == PREDICTION_LEGITIMATE:

                predicted_class_probability = cls._safe_probability(
                    class_probabilities.get(
                        cls.CLEAN_VERDICT,
                        class_probabilities.get(
                            "legitimate",
                            class_probabilities.get(
                                "clean",
                                1.0 - cls._safe_probability(
                                    threat_probability
                                ),
                            ),
                        ),
                    )
                )

        if predicted_class_probability is None:

            threat_probability = cls._safe_probability(
                threat_probability
            )

            if fusion_prediction == PREDICTION_PHISHING:
                predicted_class_probability = threat_probability
            elif fusion_prediction == PREDICTION_LEGITIMATE:
                predicted_class_probability = (
                    1.0 - threat_probability
                )

        if explicit > 0.0:
            return round(explicit, 4)

        if (
            predicted_class_probability is not None
            and predicted_class_probability > 0.0
        ):
            return round(
                predicted_class_probability,
                4,
            )

        return 0.0

    @staticmethod
    def _compose_reason(
        explanation: Dict[str, Any],
        prediction: str,
        confidence: float,
        risk_score: Any,
    ) -> str:
        """
        Always produce a non-empty human-readable reason on success.
        """

        summary = explanation.get(
            "summary"
        )

        if isinstance(summary, str) and summary.strip():
            return summary.strip()

        top_factors = explanation.get(
            "top_factors"
        )

        if isinstance(top_factors, list) and top_factors:

            first = top_factors[0]

            if isinstance(first, dict):

                label = (
                    first.get("feature")
                    or first.get("name")
                    or first.get("factor")
                )

                if label:
                    return (
                        f"Threat Intelligence prediction '{prediction}' "
                        f"is supported by factor '{label}'."
                    )

            return (
                f"Threat Intelligence prediction '{prediction}' "
                "is supported by extracted threat factors."
            )

        return (
            f"Threat Intelligence classified the target as "
            f"{prediction} with confidence {confidence:.2%} "
            f"and operational risk score {risk_score}."
        )

    @staticmethod
    def _map_prediction_source(
        model_status: str,
        inference_method: str,
    ) -> str:

        if inference_method in {
            "xgboost",
            "trained_ml",
            "ml",
        } or model_status == "loaded_ml_model":

            return PREDICTION_SOURCE_TRAINED_ML

        if inference_method in {
            "heuristic_fallback",
            "heuristic",
        } or model_status == "fallback_heuristic":

            return PREDICTION_SOURCE_FALLBACK_HEURISTIC

        return PREDICTION_SOURCE_UNAVAILABLE

    @staticmethod
    def _map_model_status(
        model_status: str,
    ) -> str:

        if model_status in {
            "loaded_ml_model",
            "available",
            "loaded",
        }:
            return MODEL_STATUS_AVAILABLE

        if model_status in {
            "fallback_heuristic",
            "fallback",
        }:
            return MODEL_STATUS_FALLBACK

        return MODEL_STATUS_UNAVAILABLE

    # =========================================================================
    # SAFE PROBABILITY
    # =========================================================================

    @staticmethod
    def _safe_probability(
        value: Any,
    ) -> float:
        """
        Normalize a probability into [0, 1].
        """

        try:
            numeric = float(value)

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if not math.isfinite(
            numeric
        ):
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                numeric,
            ),
        )

    # =========================================================================
    # SAFE NUMERIC VALUE
    # =========================================================================

    @staticmethod
    def _safe_numeric_value(
        value: Any,
    ) -> float:
        """
        Convert a feature value into a finite float.
        """

        if isinstance(
            value,
            bool,
        ):
            return (
                1.0
                if value
                else 0.0
            )

        try:
            numeric = float(value)

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if not math.isfinite(
            numeric
        ):
            return 0.0

        return numeric

    # =========================================================================
    # CLASS PROBABILITIES
    # =========================================================================

    def _normalize_class_probabilities(
        self,
        supplied: Any,
        threat_probability: float,
    ) -> Dict[str, float]:
        """
        Normalize class probabilities.

        Output:

            {
                "clean_reputation": x,
                "malicious_threat": y
            }
        """

        threat_probability = (
            self._safe_probability(
                threat_probability
            )
        )

        clean_probability = (
            1.0
            - threat_probability
        )

        if isinstance(
            supplied,
            dict,
        ):

            supplied_clean = supplied.get(
                self.CLEAN_VERDICT,
                supplied.get(
                    "clean",
                    supplied.get(
                        "0",
                        0.0,
                    ),
                ),
            )

            supplied_threat = supplied.get(
                self.THREAT_VERDICT,
                supplied.get(
                    "malicious",
                    supplied.get(
                        "1",
                        0.0,
                    ),
                ),
            )

            supplied_clean = (
                self._safe_probability(
                    supplied_clean
                )
            )

            supplied_threat = (
                self._safe_probability(
                    supplied_threat
                )
            )

            total = (
                supplied_clean
                + supplied_threat
            )

            if total > 0.0:

                return {
                    self.CLEAN_VERDICT:
                        round(
                            supplied_clean
                            / total,
                            4,
                        ),

                    self.THREAT_VERDICT:
                        round(
                            supplied_threat
                            / total,
                            4,
                        ),
                }

        return {
            self.CLEAN_VERDICT:
                round(
                    clean_probability,
                    4,
                ),

            self.THREAT_VERDICT:
                round(
                    threat_probability,
                    4,
                ),
        }

    # =========================================================================
    # RISK SCORE
    # =========================================================================

    @staticmethod
    def _safe_risk_score(
        value: Any,
    ) -> int:
        """
        Normalize risk score into [0, 100].
        """

        try:
            score = int(
                round(
                    float(value)
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0

        return max(
            0,
            min(
                100,
                score,
            ),
        )

    # =========================================================================
    # STANDARD OUTPUT
    # =========================================================================

    def _build_standard_output(
        self,
        target_url: str,
        prediction: str,
        confidence: float,
        threat_probability: float,
        class_probabilities: Dict[str, float],
        risk_assessment: Dict[str, Any],
        explanation: Dict[str, Any],
        normalized_features: Dict[str, float],
        model_status: str,
        training_evaluation: Dict[str, Any],
        input_diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the standardized Threat Agent result.
        """

        fusion_prediction = self._to_fusion_prediction(
            prediction
        )

        threat_probability = self._safe_probability(
            threat_probability
        )

        confidence = self._derive_model_confidence(
            confidence,
            threat_probability,
            class_probabilities,
            fusion_prediction,
        )

        if confidence <= 0.0:

            return self._get_fallback_response(
                error_message=(
                    "Threat Intelligence confidence was unavailable. "
                    "A phishing/legitimate label was not published "
                    "because certainty could not be derived from "
                    "class probabilities."
                ),
                target_url=target_url,
            )

        risk_score = (
            self._safe_risk_score(
                risk_assessment.get(
                    "risk_score",
                    0,
                )
            )
        )

        risk_level = str(
            risk_assessment.get(
                "risk_level",
                "Unknown",
            )
        )

        indicators = (
            risk_assessment.get(
                "triggered_indicators",
                [],
            )
        )

        if not isinstance(
            indicators,
            list,
        ):
            indicators = []

        top_factors = (
            explanation.get(
                "top_factors",
                [],
            )
        )

        if not isinstance(
            top_factors,
            list,
        ):
            top_factors = []

        explanation_summary = self._compose_reason(
            explanation,
            fusion_prediction,
            confidence,
            risk_score,
        )

        explanation_method = str(
            explanation.get(
                "explanation_method",
                "unknown",
            )
        )

        inference_method = self._resolve_inference_method(
            model_status
        )

        prediction_source = self._map_prediction_source(
            model_status,
            inference_method,
        )

        standard_model_status = self._map_model_status(
            model_status
        )

        explanation_payload = {
            "summary": explanation_summary,
            "top_factors": top_factors,
            "explanation_method": explanation_method,
            "triggered_indicators": indicators,
        }

        result = AgentResult.success(
            agent=self.AGENT_NAME,
            prediction=fusion_prediction,
            probability=threat_probability,
            confidence=confidence,
            risk_score=risk_score,
            prediction_source=prediction_source,
            model_status=standard_model_status,
            risk_factors=indicators,
            evidence=indicators,
            explanation=explanation_payload,
            feature_key="threat_features",
            metadata={
                "target_url": target_url,
                "internal_verdict": self._normalize_verdict(
                    prediction
                ),
                "class_probabilities": class_probabilities,
                "threat_probability": threat_probability,
                "field_semantics": dict(self.FIELD_SEMANTICS),
                "training_evaluation": training_evaluation,
                "input_diagnostics": input_diagnostics,
            },
        )

        output = result.to_dict()

        output.update({
            "agent_name": self.AGENT_NAME,
            "agent_display_name": self.AGENT_DISPLAY_NAME,
            "target_url": target_url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_status": "success",
            "status": "success",
            "signal_available": True,
            "verdict": fusion_prediction,
            "prediction": fusion_prediction,
            "internal_verdict": self._normalize_verdict(
                prediction
            ),
            "confidence": round(confidence, 4),
            "confidence_score": round(confidence, 4),
            "probability": round(threat_probability, 4),
            "phishing_probability": round(threat_probability, 4),
            "threat_probability": round(threat_probability, 4),
            "class_probabilities": class_probabilities,
            "risk_score": risk_score,
            "risk_level": str(output.get("risk_level") or risk_level),
            "risk_score_meaning": "operational_severity",
            "confidence_meaning": "class_certainty",
            "triggered_indicators": indicators,
            "reason": explanation_summary,
            "explanation": explanation_payload,
            "explanation_method": explanation_method,
            "top_factors": top_factors,
            "features_evaluated_count": len(normalized_features),
            "expected_feature_count": self.expected_feature_count,
            "model_feature_count": self.get_model_feature_count(),
            "feature_schema": list(self.expected_feature_columns),
            "model_feature_schema": self.get_model_feature_schema(),
            "normalized_features": normalized_features,
            "input_diagnostics": input_diagnostics,
            "training_evaluation": training_evaluation,
            "inference_method": inference_method,
            "model_status": model_status,
            "field_semantics": dict(self.FIELD_SEMANTICS),
            "agent_metadata": {
                "agent_name": self.AGENT_NAME,
                "classification": {
                    "clean": 0,
                    "malicious": 1,
                },
                "canonical_feature_count": self.expected_feature_count,
                "model_feature_count": self.get_model_feature_count(),
                "model_uses_canonical_subset": (
                    self.get_model_feature_count()
                    <= self.expected_feature_count
                ),
                "field_semantics": dict(self.FIELD_SEMANTICS),
            },
        })

        return output

    # =========================================================================
    # INFERENCE METHOD
    # =========================================================================

    @staticmethod
    def _resolve_inference_method(
        model_status: str,
    ) -> str:
        """
        Convert model status into an inference method label.
        """

        if model_status == "loaded_ml_model":
            return "xgboost"

        if model_status == "fallback_heuristic":
            return "heuristic_fallback"

        return "unavailable"

    # =========================================================================
    # FALLBACK RESPONSE
    # =========================================================================

    def _get_fallback_response(
        self,
        error_message: str,
        target_url: str = "unknown_target",
    ) -> Dict[str, Any]:
        """
        Build a safe failure response.

        The response is deliberately structured so that the Decision
        Fusion Engine can continue processing other agents.
        """

        model_status = self.get_model_status()
        reason = (
            error_message
            or "Threat Intelligence analysis could not be completed."
        )

        return {
            "agent_name":
                self.AGENT_NAME,

            "agent_display_name":
                self.AGENT_DISPLAY_NAME,

            "target_url":
                target_url,

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "analysis_status":
                "unavailable",

            "status":
                "unavailable",

            "signal_available":
                False,

            "verdict":
                PREDICTION_UNKNOWN,

            "prediction":
                PREDICTION_UNKNOWN,

            "internal_verdict":
                "unknown",

            "confidence":
                0.0,

            "confidence_score":
                0.0,

            "probability":
                None,

            "phishing_probability":
                None,

            "threat_probability":
                None,

            "class_probabilities": {
                self.CLEAN_VERDICT:
                    0.0,

                self.THREAT_VERDICT:
                    0.0,
            },

            "risk_score":
                None,

            "risk_level":
                "unknown",

            "triggered_indicators":
                [],

            "features_evaluated_count":
                0,

            "expected_feature_count":
                self.expected_feature_count,

            "model_feature_count":
                self.get_model_feature_count(),

            "feature_schema":
                list(
                    self.expected_feature_columns
                ),

            "model_feature_schema":
                self.get_model_feature_schema(),

            "model_status":
                model_status,

            "inference_method":
                self._resolve_inference_method(
                    model_status
                ),

            "reason":
                reason,

            "explanation": {
                "summary": reason,
            },

            "explanation_method":
                "fallback",

            "top_factors":
                [],

            "training_evaluation":
                {},

            "error":
                error_message,

            "field_semantics":
                dict(self.FIELD_SEMANTICS),

            "agent_metadata": {
                "agent_name":
                    self.AGENT_NAME,

                "canonical_feature_count":
                    self.expected_feature_count,

                "model_feature_count":
                    self.get_model_feature_count(),

                "analysis_failed":
                    True,
            },
        }


# ============================================================================
# OPTIONAL FUNCTIONAL ENTRY POINT
# ============================================================================

def analyze_threat(
    unified_vector: Dict[str, Any],
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Functional convenience wrapper around ThreatIntelligenceAIAgent.

    Parameters
    ----------
    unified_vector:
        Unified project feature vector.

    model_path:
        Optional Threat XGBoost model path.

    Returns
    -------
    Dict[str, Any]
        Standardized Threat Agent result.
    """

    agent = ThreatIntelligenceAIAgent(
        model_path=model_path
    )

    return agent.analyze(
        unified_vector
    )