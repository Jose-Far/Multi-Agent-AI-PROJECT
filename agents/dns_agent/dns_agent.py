"""
DNS Security AI Agent
=====================

Production master controller for the DNS Security AI Agent.

Pipeline
--------

Unified Feature Vector
        |
        v
    dns_features
        |
        v
    DNSPredictor
        |
        +--> XGBoost / heuristic prediction
        |
        v
    DNSRiskScorer
        |
        v
    DNSExplainer
        |
        v
 Standardized AgentResult
        |
        v
 Decision Fusion Engine


Security principles
-------------------

1. DNS is an independent evidence source.
2. DNS alone does not determine the final system verdict.
3. Missing DNS telemetry is NOT legitimate evidence.
4. Unavailable DNS analysis must return:
       prediction = unknown
       probability = None
       confidence = 0.0
       risk_score = None
       risk_level = unknown
       signal_available = False
5. A usable DNS feature block may use either:
       trained ML
       or
       controlled heuristic fallback
6. The canonical DNS schema contains 35 features.
7. The standardized AgentResult contract is exposed directly.
8. Legacy fields are preserved for compatibility with the existing
   orchestrator and fusion layer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from .preprocessing import DNSPreprocessor
from .predictor import DNSPredictor
from .risk_score import DNSRiskScorer
from .explain import DNSExplainer
from .feature_schema import DNSFeatureSchema

from agents.agent_result import AgentResult


# =============================================================================
# LOGGER
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

DNS_AGENT_VERSION = "2.0.0"

AGENT_NAME = "DNS_AI_Agent"
AGENT_TYPE = "DNS Security Analysis"
FEATURE_KEY = "dns_features"


# =============================================================================
# DNS AI AGENT
# =============================================================================

class DNSAIAgent:
    """
    Master controller for DNS security analysis.

    Responsibilities
    -----------------

    1. Validate the unified vector.
    2. Extract DNS features.
    3. Validate the DNS feature block.
    4. Execute DNS prediction.
    5. Calculate DNS risk.
    6. Generate SHAP / heuristic explanation.
    7. Produce standardized AgentResult output.
    8. Preserve compatibility fields for the existing orchestrator.
    """

    AGENT_NAME = AGENT_NAME
    AGENT_VERSION = DNS_AGENT_VERSION
    AGENT_TYPE = AGENT_TYPE
    FEATURE_KEY = FEATURE_KEY

    EXPECTED_FEATURE_COUNT = 35

    VALID_VERDICTS = {
        "legitimate",
        "phishing",
    }

    STATUS_SUCCESS = "success"
    STATUS_UNAVAILABLE = "unavailable"
    STATUS_ERROR = "error"

    SOURCE_TRAINED_ML = "trained_ml"
    SOURCE_HEURISTIC = "heuristic"
    SOURCE_UNAVAILABLE = "unavailable"
    SOURCE_ERROR = "error"

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        model_path: Optional[str] = None,
    ) -> None:

        logger.info(
            "Initializing %s version %s...",
            self.AGENT_NAME,
            self.AGENT_VERSION,
        )

        self.initialized = False
        self.initialization_error: Optional[str] = None

        try:

            self.preprocessor = DNSPreprocessor()

            self.predictor = DNSPredictor(
                model_path=model_path
            )

            self.risk_scorer = DNSRiskScorer()

            self.explainer = DNSExplainer(
                model=self.predictor.get_model()
            )

            self.model_path = model_path

            self.initialized = True

            logger.info(
                "%s initialized successfully.",
                self.AGENT_NAME,
            )

        except Exception as exc:

            self.initialized = False
            self.initialization_error = str(exc)

            logger.critical(
                "Critical failure while initializing %s: %s",
                self.AGENT_NAME,
                exc,
                exc_info=True,
            )

            raise

    # =========================================================================
    # TIMESTAMP
    # =========================================================================

    @staticmethod
    def _utc_timestamp() -> str:
        return (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

    # =========================================================================
    # TARGET URL
    # =========================================================================

    @staticmethod
    def _extract_target_url(
        unified_vector: Mapping[str, Any],
    ) -> str:

        metadata = unified_vector.get(
            "metadata",
            {}
        )

        if not isinstance(
            metadata,
            Mapping,
        ):
            metadata = {}

        target_url = (
            metadata.get("target_url")
            or unified_vector.get("url")
        )

        if target_url is None:
            return "unknown_target"

        target_url = str(
            target_url
        ).strip()

        return (
            target_url
            if target_url
            else "unknown_target"
        )

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_input(
        unified_vector: Any,
    ) -> Optional[str]:

        if not isinstance(
            unified_vector,
            Mapping,
        ):

            return (
                "Unified feature vector must be "
                "a dictionary or mapping."
            )

        return None

    # =========================================================================
    # DNS FEATURE EXTRACTION
    # =========================================================================

    @staticmethod
    def _extract_dns_features(
        unified_vector: Mapping[str, Any],
    ) -> Dict[str, Any]:

        dns_features = unified_vector.get(
            FEATURE_KEY
        )

        if dns_features is None:
            return {}

        if not isinstance(
            dns_features,
            Mapping,
        ):

            raise TypeError(
                "'dns_features' must be a dictionary or mapping."
            )

        return dict(
            dns_features
        )

    # =========================================================================
    # FEATURE COUNT
    # =========================================================================

    @classmethod
    def _count_canonical_features(
        cls,
        features: Mapping[str, Any],
    ) -> int:

        if not isinstance(
            features,
            Mapping,
        ):
            return 0

        schema = DNSFeatureSchema.get_schema()

        return sum(
            1
            for feature in schema
            if feature in features
        )

    # =========================================================================
    # FEATURE VALIDATION
    # =========================================================================

    @classmethod
    def _validate_dns_features(
        cls,
        features: Mapping[str, Any],
    ) -> Dict[str, Any]:

        supplied_count = (
            cls._count_canonical_features(
                features
            )
        )

        expected_count = (
            cls.EXPECTED_FEATURE_COUNT
        )

        missing = [
            feature
            for feature in DNSFeatureSchema.get_schema()
            if feature not in features
        ]

        extra = [
            feature
            for feature in features
            if feature not in DNSFeatureSchema.get_schema()
        ]

        return {
            "schema_complete": (
                supplied_count == expected_count
                and not missing
            ),
            "supplied_feature_count": supplied_count,
            "expected_feature_count": expected_count,
            "missing_features": missing,
            "extra_features": extra,
        }

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def _run_prediction(
        self,
        dns_features: Dict[str, Any],
    ) -> Dict[str, Any]:

        result = self.predictor.predict(
            dns_features
        )

        if not isinstance(
            result,
            Mapping,
        ):

            raise RuntimeError(
                "DNSPredictor returned an invalid prediction result."
            )

        return dict(
            result
        )

    # =========================================================================
    # NORMALIZED FEATURES
    # =========================================================================

    @staticmethod
    def _extract_normalized_features(
        prediction_result: Mapping[str, Any],
        fallback_features: Mapping[str, Any],
    ) -> Dict[str, Any]:

        dataframe = prediction_result.get(
            "features_dataframe"
        )

        if dataframe is not None:

            try:

                if (
                    hasattr(
                        dataframe,
                        "empty",
                    )
                    and not dataframe.empty
                ):

                    row = (
                        dataframe.iloc[0]
                        .to_dict()
                    )

                    if isinstance(
                        row,
                        dict,
                    ):

                        return row

            except Exception:

                logger.debug(
                    "Unable to extract normalized DNS "
                    "features from predictor DataFrame.",
                    exc_info=True,
                )

        return dict(
            fallback_features
        )

    # =========================================================================
    # PROBABILITY NORMALIZATION
    # =========================================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:
            result = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return default

        if result != result:
            return default

        if result in (
            float("inf"),
            float("-inf"),
        ):
            return default

        return result

    # =========================================================================
    # CLASS PROBABILITIES
    # =========================================================================

    @classmethod
    def _normalize_class_probabilities(
        cls,
        probabilities: Any,
    ) -> Dict[str, float]:

        if not isinstance(
            probabilities,
            Mapping,
        ):

            return {
                "legitimate": 0.0,
                "phishing": 0.0,
            }

        legitimate = max(
            0.0,
            min(
                1.0,
                cls._safe_float(
                    probabilities.get(
                        "legitimate",
                        0.0
                    )
                )
            )
        )

        phishing = max(
            0.0,
            min(
                1.0,
                cls._safe_float(
                    probabilities.get(
                        "phishing",
                        0.0
                    )
                )
            )
        )

        total = (
            legitimate
            + phishing
        )

        if total > 0:

            legitimate /= total
            phishing /= total

        return {
            "legitimate": round(
                legitimate,
                6
            ),
            "phishing": round(
                phishing,
                6
            ),
        }

    # =========================================================================
    # VERDICT NORMALIZATION
    # =========================================================================

    @classmethod
    def _normalize_verdict(
        cls,
        verdict: Any,
    ) -> str:

        normalized = str(
            verdict
        ).strip().lower()

        if normalized in {
            "legitimate",
            "safe",
            "benign",
            "normal",
        }:

            return "legitimate"

        if normalized in {
            "phishing",
            "malicious",
            "suspicious",
        }:

            return "phishing"

        return "unknown"

    # =========================================================================
    # PREDICTION SOURCE
    # =========================================================================

    @staticmethod
    def _get_prediction_source(
        prediction_result: Mapping[str, Any],
    ) -> str:

        source = str(
            prediction_result.get(
                "prediction_source",
                ""
            )
        ).strip().lower()

        if source in {
            "trained_ml",
            "ml",
            "xgboost",
            "loaded_ml_model",
            "loaded_model",
        }:

            return "trained_ml"

        if source in {
            "heuristic",
            "fallback",
            "heuristic_fallback",
        }:

            return "heuristic"

        fallback_used = bool(
            prediction_result.get(
                "fallback_used",
                False
            )
        )

        if fallback_used:
            return "heuristic"

        return source or "unknown"

    # =========================================================================
    # MODEL STATUS
    # =========================================================================

    @staticmethod
    def _get_model_status(
        prediction_result: Mapping[str, Any],
    ) -> str:

        value = (
            prediction_result.get(
                "model_status",
                "unknown"
            )
        )

        return str(
            value
        )

    # =========================================================================
    # RISK SCORING
    # =========================================================================

    def _run_risk_scoring(
        self,
        phishing_probability: float,
        normalized_features: Dict[str, Any],
    ) -> Dict[str, Any]:

        result = (
            self.risk_scorer.calculate_risk(
                probability=phishing_probability,
                features=normalized_features,
            )
        )

        if not isinstance(
            result,
            Mapping,
        ):

            raise RuntimeError(
                "DNSRiskScorer returned an invalid result."
            )

        return dict(
            result
        )

    # =========================================================================
    # EXPLAINABILITY
    # =========================================================================

    def _run_explainability(
        self,
        normalized_features: Dict[str, Any],
        prediction: str,
    ) -> Dict[str, Any]:

        result = (
            self.explainer.generate_explanation(
                features=normalized_features,
                prediction=prediction,
            )
        )

        if not isinstance(
            result,
            Mapping,
        ):

            raise RuntimeError(
                "DNSExplainer returned an invalid result."
            )

        return dict(
            result
        )

    # =========================================================================
    # RISK FACTORS
    # =========================================================================

    @staticmethod
    def _build_risk_factors(
        risk_assessment: Mapping[str, Any],
        explanation: Mapping[str, Any],
    ) -> List[Any]:

        factors: List[Any] = []

        for key in (
            "triggered_indicators",
            "risk_indicators",
            "risk_factors",
        ):

            values = risk_assessment.get(
                key,
                []
            )

            if isinstance(
                values,
                list,
            ):
                factors.extend(
                    values
                )

        explanation_factors = (
            explanation.get(
                "risk_factors",
                []
            )
        )

        if isinstance(
            explanation_factors,
            list,
        ):
            factors.extend(
                explanation_factors
            )

        unique: List[Any] = []

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
        prediction_result: Mapping[str, Any],
        risk_assessment: Mapping[str, Any],
        feature_validation: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:

        evidence: List[Dict[str, Any]] = []

        evidence.append({
            "type": "analysis_target",
            "target": target_url,
        })

        evidence.append({
            "type": "model_signal",
            "model_type": "XGBoost",
            "model_status": self._get_model_status(
                prediction_result
            ),
            "prediction_source": self._get_prediction_source(
                prediction_result
            ),
        })

        evidence.append({
            "type": "risk_assessment",
            "risk_score": risk_assessment.get(
                "risk_score"
            ),
            "risk_level": risk_assessment.get(
                "risk_level",
                "unknown"
            ),
            "signal_available": True,
        })

        evidence.append({
            "type": "feature_schema",
            "feature_count": feature_validation.get(
                "supplied_feature_count",
                0
            ),
            "expected_feature_count": (
                self.EXPECTED_FEATURE_COUNT
            ),
            "schema_complete": bool(
                feature_validation.get(
                    "schema_complete",
                    False
                )
            ),
        })

        return evidence

    # =========================================================================
    # SUCCESS RESULT
    # =========================================================================

    def _build_success_result(
        self,
        target_url: str,
        prediction_result: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        explanation: Dict[str, Any],
        normalized_features: Dict[str, Any],
        feature_validation: Dict[str, Any],
        started_at: datetime,
    ) -> Dict[str, Any]:

        prediction = self._normalize_verdict(
            prediction_result.get(
                "class_label",
                prediction_result.get(
                    "prediction",
                    "unknown"
                )
            )
        )

        if prediction == "unknown":

            raise RuntimeError(
                "DNS predictor returned an unknown classification."
            )

        phishing_probability = max(
            0.0,
            min(
                1.0,
                self._safe_float(
                    prediction_result.get(
                        "phishing_probability",
                        0.0
                    )
                )
            )
        )

        class_probabilities = (
            self._normalize_class_probabilities(
                prediction_result.get(
                    "class_probabilities",
                    {}
                )
            )
        )

        confidence = max(
            0.0,
            min(
                1.0,
                self._safe_float(
                    prediction_result.get(
                        "confidence",
                        max(
                            class_probabilities.values()
                        )
                        if class_probabilities
                        else 0.0
                    )
                )
            )
        )

        risk_score_raw = risk_assessment.get(
            "risk_score"
        )

        if risk_score_raw is None:

            raise RuntimeError(
                "DNSRiskScorer returned no risk score."
            )

        risk_score = int(
            max(
                0,
                min(
                    100,
                    self._safe_float(
                        risk_score_raw
                    )
                )
            )
        )

        risk_level = str(
            risk_assessment.get(
                "risk_level",
                "unknown"
            )
        )

        prediction_source = (
            self._get_prediction_source(
                prediction_result
            )
        )

        model_status = (
            self._get_model_status(
                prediction_result
            )
        )

        fallback_used = bool(
            prediction_result.get(
                "fallback_used",
                prediction_source == "heuristic"
            )
        )

        risk_factors = (
            self._build_risk_factors(
                risk_assessment,
                explanation,
            )
        )

        evidence = (
            self._build_evidence(
                target_url=target_url,
                prediction_result=prediction_result,
                risk_assessment=risk_assessment,
                feature_validation=feature_validation,
            )
        )

        execution_time_ms = round(
            (
                datetime.now(
                    timezone.utc
                )
                - started_at
            ).total_seconds()
            * 1000.0,
            3,
        )

        metadata = {
            "target_url": target_url,
            "agent_version": self.AGENT_VERSION,
            "agent_type": self.AGENT_TYPE,
            "feature_key": self.FEATURE_KEY,
            "feature_count": feature_validation.get(
                "supplied_feature_count",
                0
            ),
            "expected_feature_count": (
                self.EXPECTED_FEATURE_COUNT
            ),
            "schema_version": getattr(
                DNSFeatureSchema,
                "SCHEMA_VERSION",
                "1.0.0",
            ),
            "feature_validation": feature_validation,
            "class_probabilities": class_probabilities,
            "fallback_used": fallback_used,
            "model_path": prediction_result.get(
                "model_path"
            ),
            "decision_threshold": prediction_result.get(
                "decision_threshold",
                prediction_result.get(
                    "threshold"
                )
            ),
            "risk_assessment": risk_assessment,
            "normalized_features": normalized_features,
        }

        # ---------------------------------------------------------------------
        # Canonical AgentResult
        # ---------------------------------------------------------------------

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

            metadata=metadata,
        )

        standard = result.to_dict()

        # ---------------------------------------------------------------------
        # Legacy compatibility fields
        # ---------------------------------------------------------------------

        standard.update({

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
                self._utc_timestamp(),

            "status":
                "success",

            "analysis_status":
                "success",

            "signal_available":
                True,

            "prediction":
                prediction,

            "verdict":
                prediction,

            "probability":
                phishing_probability,

            "phishing_probability":
                phishing_probability,

            "confidence":
                confidence,

            "confidence_score":
                confidence,

            "legitimate_probability":
                class_probabilities.get(
                    "legitimate"
                ),

            "class_probabilities":
                class_probabilities,

            "risk_score":
                risk_score,

            "risk_level":
                risk_level,

            "feature_count":
                feature_validation.get(
                    "supplied_feature_count",
                    0
                ),

            "features_evaluated_count":
                feature_validation.get(
                    "supplied_feature_count",
                    0
                ),

            "prediction_source":
                prediction_source,

            "model_status":
                model_status,

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
                        in {
                            "loaded_ml_model",
                            "available",
                            "loaded",
                            "xgboost",
                        },
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
                        (
                            prediction_source
                            == "trained_ml"
                            and not fallback_used
                        ),

                    "class_mapping":
                        {
                            "0":
                                "legitimate",

                            "1":
                                "phishing",
                        },
                },

            "feature_validation":
                feature_validation,

            "risk_indicators":
                risk_assessment.get(
                    "triggered_indicators",
                    []
                ),

            "contextual_observations":
                risk_assessment.get(
                    "contextual_observations",
                    []
                ),

            "protective_indicators":
                risk_assessment.get(
                    "protective_indicators",
                    []
                ),

            "risk_assessment":
                risk_assessment,

            "explanation":
                explanation,

            "evidence":
                evidence,

            "fusion_metadata":
                {
                    "agent_signal":
                        "dns",

                    "binary_classification":
                        True,

                    "supported_classes":
                        [
                            "legitimate",
                            "phishing",
                        ],

                    "final_decision_made_by_agent":
                        False,

                    "ready_for_decision_fusion":
                        True,

                    "signal_available":
                        True,
                },
        })

        return standard

    # =========================================================================
    # UNAVAILABLE RESULT
    # =========================================================================

    def _get_unavailable_response(
        self,
        reason: str,
        target_url: str,
        started_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:

        if started_at is None:

            started_at = datetime.now(
                timezone.utc
            )

        execution_time_ms = round(
            (
                datetime.now(
                    timezone.utc
                )
                - started_at
            ).total_seconds()
            * 1000.0,
            3,
        )

        explanation = {
            "summary":
                "DNS analysis could not provide a usable security signal.",

            "method":
                "none",

            "top_factors":
                [],

            "top_shap_factors":
                [],

            "risk_factors":
                [],

            "protective_factors":
                [],

            "key_indicators_triggered":
                [],

            "explanation_source":
                "none",

            "shap_available":
                False,

            "shap_used":
                False,

            "shap_status":
                "unavailable",
        }

        evidence = [
            {
                "type":
                    "analysis_target",

                "target":
                    target_url,
            },

            {
                "type":
                    "agent_status",

                "status":
                    "unavailable",

                "reason":
                    reason,
            },

            {
                "type":
                    "feature_schema",

                "feature_count":
                    0,

                "expected_feature_count":
                    self.EXPECTED_FEATURE_COUNT,

                "schema_complete":
                    False,
            },
        ]

        metadata = {
            "target_url":
                target_url,

            "agent_version":
                self.AGENT_VERSION,

            "agent_type":
                self.AGENT_TYPE,

            "feature_key":
                self.FEATURE_KEY,

            "expected_feature_count":
                self.EXPECTED_FEATURE_COUNT,

            "reason":
                reason,
        }

        # ---------------------------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT pass risk_factors/evidence/explanation into AgentResult
        # unavailable() because the current AgentResult API does not accept
        # those arguments.
        # ---------------------------------------------------------------------

        try:

            result = AgentResult.unavailable(

                agent=self.AGENT_NAME,

                reason=reason,

                execution_time_ms=execution_time_ms,

                feature_key=self.FEATURE_KEY,

                metadata=metadata,
            )

            standard = result.to_dict()

        except Exception as exc:

            logger.error(
                "AgentResult.unavailable() failed: %s",
                exc,
                exc_info=True,
            )

            standard = {}

        # ---------------------------------------------------------------------
        # Standard + compatibility contract
        # ---------------------------------------------------------------------

        standard.update({

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
                self._utc_timestamp(),

            "status":
                "unavailable",

            "analysis_status":
                "unavailable",

            "prediction":
                "unknown",

            "verdict":
                "unknown",

            "probability":
                None,

            "phishing_probability":
                None,

            "legitimate_probability":
                None,

            "confidence":
                0.0,

            "confidence_score":
                0.0,

            "risk_score":
                None,

            "risk_level":
                "unknown",

            "signal_available":
                False,

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

            "model_metadata":
                {
                    "model_type":
                        "XGBoost",

                    "model_status":
                        "unavailable",

                    "prediction_source":
                        "unavailable",

                    "fallback_used":
                        False,

                    "model_based":
                        False,

                    "class_mapping":
                        {
                            "0":
                                "legitimate",

                            "1":
                                "phishing",
                        },
                },

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
                        self.EXPECTED_FEATURE_COUNT,

                    "missing_features":
                        DNSFeatureSchema.get_schema(),

                    "extra_features":
                        [],
                },

            "risk_factors":
                [],

            "risk_indicators":
                [],

            "contextual_observations":
                [],

            "protective_indicators":
                [],

            "evidence":
                evidence,

            "reason":
                reason,

            "error_details":
                reason,

            "explanation":
                explanation,

            "fusion_metadata":
                {
                    "agent_signal":
                        "dns",

                    "binary_classification":
                        True,

                    "supported_classes":
                        [
                            "legitimate",
                            "phishing",
                        ],

                    "final_decision_made_by_agent":
                        False,

                    "ready_for_decision_fusion":
                        False,

                    "signal_available":
                        False,
                },

            "execution_time_ms":
                execution_time_ms,
        })

        logger.warning(
            "DNS Agent unavailable | URL=%s | Reason=%s",
            target_url,
            reason,
        )

        return standard

    # =========================================================================
    # ERROR RESULT
    # =========================================================================

    def _get_error_response(
        self,
        reason: str,
        target_url: str,
        started_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:

        if started_at is None:

            started_at = datetime.now(
                timezone.utc
            )

        execution_time_ms = round(
            (
                datetime.now(
                    timezone.utc
                )
                - started_at
            ).total_seconds()
            * 1000.0,
            3,
        )

        explanation = {
            "summary":
                "DNS Agent encountered an analysis error.",

            "method":
                "none",

            "top_factors":
                [],

            "top_shap_factors":
                [],

            "risk_factors":
                [],

            "protective_factors":
                [],

            "explanation_source":
                "none",

            "shap_available":
                False,

            "shap_used":
                False,

            "shap_status":
                "unavailable",
        }

        evidence = [
            {
                "type":
                    "analysis_target",

                "target":
                    target_url,
            },

            {
                "type":
                    "agent_error",

                "error":
                    reason,
            },
        ]

        metadata = {
            "target_url":
                target_url,

            "agent_version":
                self.AGENT_VERSION,

            "agent_type":
                self.AGENT_TYPE,

            "feature_key":
                self.FEATURE_KEY,

            "reason":
                reason,
        }

        try:

            result = AgentResult.error_result(

                agent=self.AGENT_NAME,

                error=reason,

                error_type="DNSAgentError",

                execution_time_ms=execution_time_ms,

                feature_key=self.FEATURE_KEY,

                metadata=metadata,
            )

            standard = result.to_dict()

        except Exception as exc:

            logger.error(
                "AgentResult.error_result() failed: %s",
                exc,
                exc_info=True,
            )

            standard = {}

        standard.update({

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
                self._utc_timestamp(),

            "status":
                "error",

            "analysis_status":
                "error",

            "prediction":
                "unknown",

            "verdict":
                "unknown",

            "probability":
                None,

            "phishing_probability":
                None,

            "legitimate_probability":
                None,

            "confidence":
                0.0,

            "confidence_score":
                0.0,

            "risk_score":
                None,

            "risk_level":
                "unknown",

            "signal_available":
                False,

            "prediction_source":
                "error",

            "model_status":
                "error",

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

            "feature_count":
                0,

            "features_evaluated_count":
                0,

            "risk_factors":
                [],

            "risk_indicators":
                [],

            "contextual_observations":
                [],

            "protective_indicators":
                [],

            "evidence":
                evidence,

            "reason":
                reason,

            "error_details":
                reason,

            "explanation":
                explanation,

            "fusion_metadata":
                {
                    "agent_signal":
                        "dns",

                    "binary_classification":
                        True,

                    "supported_classes":
                        [
                            "legitimate",
                            "phishing",
                        ],

                    "final_decision_made_by_agent":
                        False,

                    "ready_for_decision_fusion":
                        False,

                    "signal_available":
                        False,
                },

            "execution_time_ms":
                execution_time_ms,
        })

        logger.error(
            "DNS Agent error | URL=%s | Reason=%s",
            target_url,
            reason,
        )

        return standard

    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================

    def analyze(
        self,
        unified_vector: Dict[str, Any],
    ) -> Dict[str, Any]:

        started_at = datetime.now(
            timezone.utc
        )

        logger.info(
            "%s analysis triggered.",
            self.AGENT_NAME,
        )

        # ---------------------------------------------------------------------
        # 1. Validate unified vector
        # ---------------------------------------------------------------------

        validation_error = (
            self._validate_input(
                unified_vector
            )
        )

        if validation_error:

            return self._get_unavailable_response(
                reason=validation_error,
                target_url="unknown_target",
                started_at=started_at,
            )

        # ---------------------------------------------------------------------
        # 2. Target URL
        # ---------------------------------------------------------------------

        target_url = (
            self._extract_target_url(
                unified_vector
            )
        )

        # ---------------------------------------------------------------------
        # 3. DNS feature block
        # ---------------------------------------------------------------------

        try:

            dns_features = (
                self._extract_dns_features(
                    unified_vector
                )
            )

        except Exception as exc:

            return self._get_error_response(
                reason=(
                    f"DNS feature extraction failed: {exc}"
                ),
                target_url=target_url,
                started_at=started_at,
            )

        # ---------------------------------------------------------------------
        # 4. Missing / empty DNS telemetry
        # ---------------------------------------------------------------------

        if not dns_features:

            return self._get_unavailable_response(
                reason=(
                    "Required DNS feature block "
                    "'dns_features' is missing or empty."
                ),
                target_url=target_url,
                started_at=started_at,
            )

        # ---------------------------------------------------------------------
        # 5. Feature validation
        # ---------------------------------------------------------------------

        feature_validation = (
            self._validate_dns_features(
                dns_features
            )
        )

        # ---------------------------------------------------------------------
        # We allow the predictor to perform its own canonical normalization.
        #
        # However, a completely malformed feature block must not be treated
        # as legitimate evidence.
        # ---------------------------------------------------------------------

        if (
            feature_validation["supplied_feature_count"]
            == 0
        ):

            return self._get_unavailable_response(
                reason=(
                    "DNS feature block contains no "
                    "recognized canonical DNS features."
                ),
                target_url=target_url,
                started_at=started_at,
            )

        # ---------------------------------------------------------------------
        # 6. Prediction
        # ---------------------------------------------------------------------

        try:

            prediction_result = (
                self._run_prediction(
                    dns_features
                )
            )

        except Exception as exc:

            return self._get_error_response(
                reason=(
                    f"DNS prediction failed: {exc}"
                ),
                target_url=target_url,
                started_at=started_at,
            )

        # ---------------------------------------------------------------------
        # 7. Prediction validation
        # ---------------------------------------------------------------------

        raw_prediction = (
            prediction_result.get(
                "class_label",
                prediction_result.get(
                    "prediction"
                )
            )
        )

        prediction = (
            self._normalize_verdict(
                raw_prediction
            )
        )

        if prediction == "unknown":

            return self._get_error_response(
                reason=(
                    "DNS predictor returned an "
                    "unknown classification."
                ),
                target_url=target_url,
                started_at=started_at,
            )

        # ---------------------------------------------------------------------
        # 8. Normalized features
        # ---------------------------------------------------------------------

        normalized_features = (
            self._extract_normalized_features(
                prediction_result,
                dns_features,
            )
        )

        # ---------------------------------------------------------------------
        # 9. Risk scoring
        # ---------------------------------------------------------------------

        phishing_probability = max(
            0.0,
            min(
                1.0,
                self._safe_float(
                    prediction_result.get(
                        "phishing_probability",
                        0.0
                    )
                )
            )
        )

        try:

            risk_assessment = (
                self._run_risk_scoring(
                    phishing_probability,
                    normalized_features,
                )
            )

        except Exception as exc:

            return self._get_error_response(
                reason=(
                    f"DNS risk scoring failed: {exc}"
                ),
                target_url=target_url,
                started_at=started_at,
            )

        # ---------------------------------------------------------------------
        # 10. Explainability
        # ---------------------------------------------------------------------

        try:

            explanation = (
                self._run_explainability(
                    normalized_features,
                    prediction,
                )
            )

        except Exception as exc:

            logger.warning(
                "DNS explainability unavailable for '%s': %s",
                target_url,
                exc,
                exc_info=True,
            )

            explanation = {
                "summary":
                    (
                        "DNS prediction completed, "
                        "but explainability was unavailable."
                    ),

                "method":
                    "none",

                "top_factors":
                    [],

                "top_shap_factors":
                    [],

                "risk_factors":
                    [],

                "protective_factors":
                    [],

                "explanation_source":
                    "none",

                "shap_available":
                    False,

                "shap_used":
                    False,

                "shap_status":
                    "unavailable",

                "shap_error":
                    str(exc),
            }

        # ---------------------------------------------------------------------
        # 11. Build standardized result
        # ---------------------------------------------------------------------

        try:

            result = (
                self._build_success_result(
                    target_url=target_url,
                    prediction_result=prediction_result,
                    risk_assessment=risk_assessment,
                    explanation=explanation,
                    normalized_features=normalized_features,
                    feature_validation=feature_validation,
                    started_at=started_at,
                )
            )

        except Exception as exc:

            return self._get_error_response(
                reason=(
                    f"DNS output construction failed: {exc}"
                ),
                target_url=target_url,
                started_at=started_at,
            )

        logger.info(
            (
                "DNS analysis completed successfully | "
                "URL=%s | Prediction=%s | "
                "Probability=%.4f | Risk=%s/%s | "
                "Source=%s"
            ),
            target_url,
            result.get("prediction"),
            result.get("probability", 0.0),
            result.get("risk_level"),
            result.get("risk_score"),
            result.get("prediction_source"),
        )

        return result

    # =========================================================================
    # DIRECT FEATURE ANALYSIS
    # =========================================================================

    def analyze_features(
        self,
        dns_features: Dict[str, Any],
        target_url: str = "unknown_target",
    ) -> Dict[str, Any]:

        unified_vector = {
            "url": target_url,
            "metadata": {
                "target_url":
                    target_url,
            },
            "dns_features":
                dns_features,
        }

        return self.analyze(
            unified_vector
        )

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:

        try:

            model_loaded = bool(
                self.predictor.is_ml_model_available()
            )

        except Exception:

            model_loaded = bool(
                getattr(
                    self.predictor,
                    "is_model_loaded",
                    False,
                )
            )

        try:

            model_status = (
                self.predictor.get_model_status()
            )

        except Exception:

            model_status = {}

        try:

            explainer_status = (
                self.explainer.get_status()
            )

        except Exception:

            explainer_status = {}

        return {

            "agent_name":
                self.AGENT_NAME,

            "agent_version":
                self.AGENT_VERSION,

            "agent_type":
                self.AGENT_TYPE,

            "initialized":
                bool(
                    self.initialized
                ),

            "model_loaded":
                model_loaded,

            "model_path":
                model_status.get(
                    "model_path",
                    self.model_path,
                ),

            "model_status":
                model_status,

            "feature_count":
                self.EXPECTED_FEATURE_COUNT,

            "feature_schema":
                DNSFeatureSchema.get_schema(),

            "schema_version":
                getattr(
                    DNSFeatureSchema,
                    "SCHEMA_VERSION",
                    "1.0.0",
                ),

            "explainer":
                explainer_status,

            "ready_for_analysis":
                bool(
                    self.initialized
                ),

            "initialization_error":
                self.initialization_error,
        }

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:

        status = self.get_status()

        return {
            "healthy":
                bool(
                    self.initialized
                    and status.get(
                        "model_loaded",
                        False,
                    )
                ),

            "agent_name":
                self.AGENT_NAME,

            "agent_version":
                self.AGENT_VERSION,

            "model_loaded":
                status.get(
                    "model_loaded",
                    False,
                ),

            "feature_count":
                self.EXPECTED_FEATURE_COUNT,

            "status":
                (
                    "healthy"
                    if (
                        self.initialized
                        and status.get(
                            "model_loaded",
                            False,
                        )
                    )
                    else "degraded"
                ),
        }

    # =========================================================================
    # MODEL RELOAD
    # =========================================================================

    def reload_model(
        self,
        model_path: Optional[str] = None,
    ) -> bool:

        load_path = (
            model_path
            or self.model_path
        )

        try:

            if hasattr(
                self.predictor,
                "reload_model",
            ):

                success = (
                    self.predictor.reload_model(
                        model_path=load_path
                    )
                )

                if not success:
                    return False

            else:

                self.predictor.model_wrapper.load_model(
                    custom_path=load_path
                )

                self.predictor.model = (
                    self.predictor.model_wrapper.model
                )

                self.predictor.is_model_loaded = (
                    self.predictor.model_wrapper.is_loaded
                )

            self.explainer.set_model(
                self.predictor.get_model()
            )

            self.model_path = load_path

            logger.info(
                "DNS model successfully reloaded from %s",
                load_path,
            )

            return True

        except Exception as exc:

            logger.error(
                "DNS model reload failed: %s",
                exc,
                exc_info=True,
            )

            return False


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "DNSAIAgent",
    "DNS_AGENT_VERSION",
]