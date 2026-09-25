"""
DNS Security AI Agent - Predictor
=================================

This module implements the inference layer of the DNS Security AI Agent.

Responsibilities
----------------
1. Accept raw DNS feature telemetry.
2. Validate the incoming feature payload.
3. Preprocess the raw DNS features.
4. Align features with DNSFeatureSchema.
5. Verify ML model availability.
6. Execute XGBoost inference when a trained model is available.
7. Provide a controlled heuristic fallback when the ML model is unavailable.
8. Calculate legitimate/phishing probabilities.
9. Determine the predicted class.
10. Calculate prediction confidence.
11. Return structured inference results.
12. Expose processed features for downstream explainability.
13. Provide model status information.
14. Protect the inference pipeline from malformed input.
15. Keep the predictor independent from risk scoring and explainability.

Architecture
------------

        Raw DNS Feature Dictionary
                    |
                    v
            DNSPredictor
                    |
                    v
             DNSPreprocessor
                    |
                    v
          DNSFeatureSchema
                    |
                    v
            37 Numeric Features
                    |
             ┌──────┴──────┐
             |             |
             v             v
        ML Model       Heuristic
             |          Fallback
             |             |
             └──────┬──────┘
                    v
           Phishing Probability
                    |
                    v
              Class Decision
                    |
                    v
             Structured Result
                    |
          ┌─────────┼─────────┐
          v         v         v
       Predictor  Features   Status
         Result      │
                     v
               Explainability
                     |
                     v
                Risk Scoring


Classification
--------------

    0 = legitimate
    1 = phishing

Probability
-----------

    0.0 = very low phishing probability
    1.0 = very high phishing probability

Important Security Principle
----------------------------

The DNS Predictor does NOT make the final cybersecurity decision.

Its output is only one evidence source for the future Multi-Agent
Decision Fusion Engine.

The final system will combine DNS evidence with:

    URL Agent
    HTML Agent
    SSL Agent
    Visual Agent
    Brand Agent
    Behavior Agent
    Reputation Agent
    Threat Intelligence Agent
    Malware Agent

A DNS prediction of "phishing" therefore means:

    "DNS characteristics resemble the malicious class."

It does NOT mean:

    "The website is definitely phishing."


Fallback Behavior
-----------------

If the trained DNS XGBoost model is unavailable, the predictor can use
a conservative deterministic heuristic.

The fallback is intentionally exposed in the result:

    model_status = "fallback_heuristic"

This allows the Decision Fusion Engine to assign lower trust to a
heuristic result than to a properly trained ML result.


Author:
    Multi-Agent AI Cybersecurity Analyst

Agent:
    DNS Security AI Agent

Version:
    1.0.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .feature_schema import (
    DNSFeatureSchema,
    SCHEMA_VERSION,
)
from .model import (
    DNSModel,
    MODEL_WRAPPER_VERSION,
)
from .preprocessing import (
    DNSPreprocessor,
    PREPROCESSOR_VERSION,
)


# =============================================================================
# LOGGER
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# PREDICTOR VERSION
# =============================================================================

PREDICTOR_VERSION = "1.0.0"


# =============================================================================
# DNS PREDICTOR
# =============================================================================

class DNSPredictor:
    """
    Production-oriented inference engine for the DNS Security AI Agent.

    The predictor sits between:

        DNS Feature Extraction
                |
                v
        DNSPredictor
                |
                v
        DNS Risk Scoring / Explainability

    It does not perform DNS resolution itself.

    The DNS resolver / feature extractor is responsible for generating
    the raw feature dictionary.

    Example raw input:

        {
            "has_a_records": True,
            "has_aaaa_records": False,
            "has_mx_records": True,
            "a_record_count": 2,
            ...
        }

    Example output:

        {
            "class_label": "phishing",
            "class_index": 1,
            "confidence": 0.91,
            "phishing_probability": 0.91,
            "legitimate_probability": 0.09,
            "class_probabilities": {
                "legitimate": 0.09,
                "phishing": 0.91
            },
            "model_status": "loaded_ml_model",
            ...
        }
    """

    # =========================================================================
    # 1. CLASS DEFINITIONS
    # =========================================================================

    LEGITIMATE_CLASS = 0

    PHISHING_CLASS = 1

    CLASS_LABELS: Dict[int, str] = {
        LEGITIMATE_CLASS: "legitimate",
        PHISHING_CLASS: "phishing",
    }

    # =========================================================================
    # 2. DEFAULT PREDICTION THRESHOLD
    # =========================================================================

    DEFAULT_THRESHOLD = 0.50

    # =========================================================================
    # 3. HEURISTIC CONFIGURATION
    # =========================================================================

    HEURISTIC_RULES: Dict[str, float] = {

        # Infrastructure failures.
        "dns_resolution_failed": 0.50,

        "has_routing_anomalies": 0.40,

        # Fast-flux indicators.
        "is_active_fast_flux": 0.40,

        "is_fast_flux_candidate": 0.20,

        # Email infrastructure.
        "uses_disposable_mail_provider": 0.30,

        "is_email_spoofable": 0.20,

        # TXT payload anomalies.
        "has_base64_payload_in_txt": 0.30,
    }

    # =========================================================================
    # 4. INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        model_path: Optional[str] = None,
        prediction_threshold: float = DEFAULT_THRESHOLD,
        allow_heuristic_fallback: bool = True,
        strict_model_loading: bool = False,
    ) -> None:
        """
        Initialize the DNS prediction engine.

        Parameters
        ----------
        model_path:
            Optional path to the trained DNS XGBoost model.

        prediction_threshold:
            Probability threshold used to convert phishing probability into
            a binary class.

            Default:

                0.50

        allow_heuristic_fallback:
            If True, use the deterministic DNS heuristic when the ML model
            is unavailable.

        strict_model_loading:
            If True, model loading errors are surfaced rather than silently
            switching to fallback mode.
        """

        # ---------------------------------------------------------------------
        # Validate threshold.
        # ---------------------------------------------------------------------

        if not 0.0 <= prediction_threshold <= 1.0:

            raise ValueError(
                "prediction_threshold must be between 0.0 and 1.0."
            )

        self.prediction_threshold = float(
            prediction_threshold
        )

        self.allow_heuristic_fallback = bool(
            allow_heuristic_fallback
        )

        self.strict_model_loading = bool(
            strict_model_loading
        )

        # ---------------------------------------------------------------------
        # Preprocessor.
        # ---------------------------------------------------------------------

        self.preprocessor = (
            DNSPreprocessor()
        )

        # ---------------------------------------------------------------------
        # Model wrapper.
        #
        # The trainer writes the production model to:
        #
        #     data/models/dns_agent/dns_xgb_model.json
        #
        # The previous predictor relied on DNSModel's legacy default:
        #
        #     agents/dns_agent/weights/dns_xgb_model.json
        #
        # That caused a trained model to exist while the predictor reported
        # model_available=False. Keep an explicit model_path override for
        # callers, but use the trainer's canonical artifact location by
        # default.
        # ---------------------------------------------------------------------

        if model_path is None:
            project_root = Path(__file__).resolve().parents[2]
            model_path = str(
                project_root
                / "data"
                / "models"
                / "dns_agent"
                / "dns_xgb_model.json"
            )

        self.model_path = str(
            Path(model_path).expanduser().resolve()
        )

        self.model_wrapper = DNSModel(
            model_path=self.model_path
        )

        # ---------------------------------------------------------------------
        # Attempt to load trained model.
        # ---------------------------------------------------------------------

        self.is_model_loaded = False

        try:

            self.is_model_loaded = (
                self.model_wrapper.load_model(
                    strict=self.strict_model_loading
                )
            )

        except Exception:

            self.is_model_loaded = False

            if self.strict_model_loading:

                raise

            logger.exception(
                "DNS model loading failed. "
                "Predictor will use fallback mode if enabled."
            )

        # ---------------------------------------------------------------------
        # Direct model reference.
        # ---------------------------------------------------------------------

        self.model = (
            self.model_wrapper.model
        )

        # ---------------------------------------------------------------------
        # Runtime statistics.
        # ---------------------------------------------------------------------

        self.total_predictions = 0

        self.ml_predictions = 0

        self.fallback_predictions = 0

        self.failed_predictions = 0

        logger.info(
            "DNSPredictor initialized. "
            "ML model loaded=%s, fallback=%s",
            self.is_model_loaded,
            self.allow_heuristic_fallback,
        )

    # =========================================================================
    # 5. MODEL ACCESS
    # =========================================================================

    def get_model(
        self,
    ) -> Optional[Any]:
        """
        Return the underlying XGBoost model.

        This is primarily useful for:

            - SHAP
            - model inspection
            - feature importance
            - explainability

        Returns
        -------
        Optional[Any]
            Loaded XGBoost model or None.
        """

        return self.model

    # =========================================================================
    # 6. MODEL AVAILABILITY
    # =========================================================================

    def is_ml_model_available(
        self,
    ) -> bool:
        """
        Determine whether the trained ML model is available.
        """

        return bool(
            self.is_model_loaded
            and self.model is not None
            and self.model_wrapper.is_available()
        )

    # =========================================================================
    # 7. INPUT VALIDATION
    # =========================================================================

    def _validate_raw_features(
        self,
        raw_dns_features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate and normalize the raw DNS feature payload.

        The predictor accepts Mapping objects, but internally converts
        them into a regular dictionary.
        """

        if raw_dns_features is None:

            logger.warning(
                "DNSPredictor received None as raw DNS features."
            )

            return {}

        if not isinstance(
            raw_dns_features,
            Mapping,
        ):

            raise TypeError(
                "raw_dns_features must be a dictionary-like Mapping."
            )

        return dict(
            raw_dns_features
        )

    # =========================================================================
    # 8. FEATURE PREPROCESSING
    # =========================================================================

    def preprocess_features(
        self,
        raw_dns_features: Mapping[str, Any],
    ) -> pd.DataFrame:
        """
        Convert raw DNS telemetry into the canonical model-ready DataFrame.

        Processing sequence:

            Raw Dictionary
                  ↓
            DNSPreprocessor
                  ↓
            DNSFeatureSchema
                  ↓
            Numeric DataFrame
        """

        validated_features = (
            self._validate_raw_features(
                raw_dns_features
            )
        )

        try:

            df_features = (
                self.preprocessor.transform_single(
                    validated_features
                )
            )

            # -------------------------------------------------------------
            # Validate final schema.
            # -------------------------------------------------------------

            DNSFeatureSchema.validate_dataframe(
                df_features
            )

            return df_features

        except Exception as exc:

            logger.exception(
                "DNS feature preprocessing failed: %s",
                exc,
            )

            raise

    # =========================================================================
    # 9. SAFE PROBABILITY
    # =========================================================================

    @staticmethod
    def _clamp_probability(
        probability: float,
    ) -> float:
        """
        Clamp a probability into the mathematically valid [0, 1] range.
        """

        try:

            value = float(
                probability
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

        if not np.isfinite(
            value
        ):

            return 0.0

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    # =========================================================================
    # 10. CLASS DECISION
    # =========================================================================

    def _classify_probability(
        self,
        phishing_probability: float,
    ) -> int:
        """
        Convert phishing probability into a binary class.
        """

        probability = (
            self._clamp_probability(
                phishing_probability
            )
        )

        if probability >= (
            self.prediction_threshold
        ):

            return self.PHISHING_CLASS

        return self.LEGITIMATE_CLASS

    # =========================================================================
    # 11. CLASS PROBABILITIES
    # =========================================================================

    def _build_class_probabilities(
        self,
        phishing_probability: float,
    ) -> Dict[str, float]:
        """
        Build the complete probability distribution.
        """

        phishing_probability = (
            self._clamp_probability(
                phishing_probability
            )
        )

        legitimate_probability = (
            1.0
            - phishing_probability
        )

        return {
            "legitimate": round(
                legitimate_probability,
                6,
            ),
            "phishing": round(
                phishing_probability,
                6,
            ),
        }

    # =========================================================================
    # 12. CONFIDENCE CALCULATION
    # =========================================================================

    def _calculate_confidence(
        self,
        class_probabilities: Dict[str, float],
        predicted_class: int,
    ) -> float:
        """
        Calculate confidence for the predicted class.

        For binary classification, confidence is the probability associated
        with the selected class.
        """

        label = self.CLASS_LABELS[
            predicted_class
        ]

        return round(
            self._clamp_probability(
                class_probabilities[
                    label
                ]
            ),
            6,
        )

    # =========================================================================
    # 13. HEURISTIC FALLBACK
    # =========================================================================

    def _heuristic_fallback_predict(
        self,
        df_features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Generate a deterministic DNS-based fallback prediction.

        This is NOT a machine-learning prediction.

        The purpose is resilience when:

            - model file does not exist
            - model loading failed
            - model is unavailable

        The result explicitly identifies itself as:

            fallback_heuristic

        This allows downstream decision fusion to treat it differently
        from an ML prediction.
        """

        if not isinstance(
            df_features,
            pd.DataFrame,
        ):

            raise TypeError(
                "df_features must be a pandas DataFrame."
            )

        if df_features.empty:

            raise ValueError(
                "Cannot perform heuristic prediction on empty features."
            )

        # ---------------------------------------------------------------------
        # Ensure exactly one row.
        # ---------------------------------------------------------------------

        if len(df_features) != 1:

            raise ValueError(
                "Heuristic prediction expects exactly one feature row."
            )

        features = (
            df_features.iloc[0]
            .to_dict()
        )

        risk_points = 0.0

        triggered_rules = []

        # ---------------------------------------------------------------------
        # Apply heuristic rules.
        # ---------------------------------------------------------------------

        for feature_name, weight in (
            self.HEURISTIC_RULES.items()
        ):

            feature_value = features.get(
                feature_name,
                0.0,
            )

            try:

                numeric_value = float(
                    feature_value
                )

            except (
                TypeError,
                ValueError,
            ):

                numeric_value = 0.0

            if numeric_value >= 1.0:

                risk_points += weight

                triggered_rules.append(
                    {
                        "feature": feature_name,
                        "weight": weight,
                        "value": numeric_value,
                    }
                )

        # ---------------------------------------------------------------------
        # Convert rule points to probability.
        #
        # Maximum raw score:
        #
        #   0.50
        # + 0.40
        # + 0.40
        # + 0.20
        # + 0.30
        # + 0.20
        # + 0.30
        #
        # = 2.30
        #
        # We intentionally cap probability at 0.99.
        # ---------------------------------------------------------------------

        phishing_probability = min(
            0.99,
            max(
                0.01,
                risk_points,
            ),
        )

        phishing_probability = (
            self._clamp_probability(
                phishing_probability
            )
        )

        class_probabilities = (
            self._build_class_probabilities(
                phishing_probability
            )
        )

        predicted_class = (
            self._classify_probability(
                phishing_probability
            )
        )

        class_label = (
            self.CLASS_LABELS[
                predicted_class
            ]
        )

        confidence = (
            self._calculate_confidence(
                class_probabilities,
                predicted_class,
            )
        )

        self.fallback_predictions += 1

        return {
            "class_label": class_label,

            "class_index": predicted_class,

            "confidence": confidence,

            "phishing_probability": round(
                phishing_probability,
                6,
            ),

            "legitimate_probability": (
                class_probabilities[
                    "legitimate"
                ]
            ),

            "class_probabilities": (
                class_probabilities
            ),

            "features_dataframe": (
                df_features
            ),

            "model_status": (
                "fallback_heuristic"
            ),

            "prediction_source": (
                "heuristic"
            ),

            "prediction_threshold": (
                self.prediction_threshold
            ),

            "heuristic_risk_points": round(
                risk_points,
                6,
            ),

            "heuristic_triggered_rules": (
                triggered_rules
            ),

            "schema_version": (
                SCHEMA_VERSION
            ),

            "predictor_version": (
                PREDICTOR_VERSION
            ),
        }

    # =========================================================================
    # 14. ML PREDICTION
    # =========================================================================

    def _ml_predict(
        self,
        df_features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Execute prediction using the trained XGBoost model.
        """

        if not self.is_ml_model_available():

            raise RuntimeError(
                "DNS ML model is not available."
            )

        # ---------------------------------------------------------------------
        # Probability.
        # ---------------------------------------------------------------------

        phishing_probability = (
            self.model_wrapper.predict_proba(
                df_features
            )
        )

        phishing_probability = (
            self._clamp_probability(
                phishing_probability
            )
        )

        # ---------------------------------------------------------------------
        # Class probabilities.
        # ---------------------------------------------------------------------

        class_probabilities = (
            self._build_class_probabilities(
                phishing_probability
            )
        )

        # ---------------------------------------------------------------------
        # Class decision.
        # ---------------------------------------------------------------------

        predicted_class = (
            self._classify_probability(
                phishing_probability
            )
        )

        class_label = (
            self.CLASS_LABELS[
                predicted_class
            ]
        )

        # ---------------------------------------------------------------------
        # Confidence.
        # ---------------------------------------------------------------------

        confidence = (
            self._calculate_confidence(
                class_probabilities,
                predicted_class,
            )
        )

        self.ml_predictions += 1

        return {
            "class_label": class_label,

            "class_index": predicted_class,

            "confidence": confidence,

            "phishing_probability": round(
                phishing_probability,
                6,
            ),

            "legitimate_probability": (
                class_probabilities[
                    "legitimate"
                ]
            ),

            "class_probabilities": (
                class_probabilities
            ),

            "features_dataframe": (
                df_features
            ),

            "model_status": (
                "loaded_ml_model"
            ),

            "prediction_source": (
                "xgboost"
            ),

            "prediction_threshold": (
                self.prediction_threshold
            ),

            "schema_version": (
                SCHEMA_VERSION
            ),

            "predictor_version": (
                PREDICTOR_VERSION
            ),

            "model_wrapper_version": (
                MODEL_WRAPPER_VERSION
            ),

            "preprocessor_version": (
                PREPROCESSOR_VERSION
            ),
        }

    # =========================================================================
    # 15. MAIN PREDICT METHOD
    # =========================================================================

    def predict(
        self,
        raw_dns_features: Optional[
            Mapping[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Execute the complete DNS inference pipeline.

        Parameters
        ----------
        raw_dns_features:
            Raw DNS feature dictionary generated by the DNS feature
            extraction layer.

        Returns
        -------
        Dict[str, Any]
            Structured DNS prediction result.

        Pipeline
        --------

            raw features
                 ↓
            validation
                 ↓
            preprocessing
                 ↓
            schema validation
                 ↓
            ML model available?
                 |
              ┌──┴──┐
             YES    NO
              |      |
              v      v
             XGB  heuristic
              |      |
              └──┬───┘
                 v
              result
        """

        self.total_predictions += 1

        # ---------------------------------------------------------------------
        # Input validation.
        # ---------------------------------------------------------------------

        try:

            validated_features = (
                self._validate_raw_features(
                    raw_dns_features
                )
            )

            # -------------------------------------------------------------
            # Empty input is allowed for resilience.
            #
            # The schema/preprocessor will produce default values.
            # -------------------------------------------------------------

            if not validated_features:

                logger.warning(
                    "DNSPredictor received empty DNS feature set. "
                    "Using schema defaults."
                )

            # -------------------------------------------------------------
            # Preprocess.
            # -------------------------------------------------------------

            df_features = (
                self.preprocess_features(
                    validated_features
                )
            )

            # -------------------------------------------------------------
            # ML model.
            # -------------------------------------------------------------

            if self.is_ml_model_available():

                return self._ml_predict(
                    df_features
                )

            # -------------------------------------------------------------
            # Fallback.
            # -------------------------------------------------------------

            if self.allow_heuristic_fallback:

                logger.warning(
                    "DNS ML model unavailable. "
                    "Using heuristic fallback."
                )

                return (
                    self._heuristic_fallback_predict(
                        df_features
                    )
                )

            # -------------------------------------------------------------
            # No model + fallback disabled.
            # -------------------------------------------------------------

            raise RuntimeError(
                "DNS ML model is unavailable and "
                "heuristic fallback is disabled."
            )

        except Exception as exc:

            self.failed_predictions += 1

            logger.exception(
                "DNS inference pipeline failed: %s",
                exc,
            )

            # -----------------------------------------------------------------
            # Controlled emergency fallback.
            #
            # This branch is only used when the normal prediction pipeline
            # itself encounters an unexpected error.
            # -----------------------------------------------------------------

            if self.allow_heuristic_fallback:

                try:

                    emergency_features = (
                        self.preprocess_features(
                            {}
                        )
                    )

                    result = (
                        self._heuristic_fallback_predict(
                            emergency_features
                        )
                    )

                    result[
                        "prediction_warning"
                    ] = (
                        "Normal DNS inference failed; "
                        "emergency heuristic fallback was used."
                    )

                    return result

                except Exception as fallback_exc:

                    logger.exception(
                        "Emergency DNS fallback also failed: %s",
                        fallback_exc,
                    )

            # -----------------------------------------------------------------
            # If fallback is disabled, surface the failure.
            # -----------------------------------------------------------------

            raise

    # =========================================================================
    # 16. PREDICT FROM DATAFRAME
    # =========================================================================

    def predict_from_dataframe(
        self,
        df_features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Execute prediction when the caller already has a processed
        one-row DataFrame.

        This method is useful for:

            - testing
            - pipeline integration
            - batch preprocessing systems
            - explainability modules
        """

        if not isinstance(
            df_features,
            pd.DataFrame,
        ):

            raise TypeError(
                "df_features must be a pandas DataFrame."
            )

        if df_features.empty:

            raise ValueError(
                "df_features cannot be empty."
            )

        if len(df_features) != 1:

            raise ValueError(
                "predict_from_dataframe expects exactly one row."
            )

        DNSFeatureSchema.validate_dataframe(
            df_features
        )

        if self.is_ml_model_available():

            return self._ml_predict(
                df_features
            )

        if self.allow_heuristic_fallback:

            return (
                self._heuristic_fallback_predict(
                    df_features
                )
            )

        raise RuntimeError(
            "DNS model unavailable and heuristic fallback disabled."
        )

    # =========================================================================
    # 17. BATCH PREDICTION
    # =========================================================================

    def predict_batch(
        self,
        df_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Perform prediction for multiple already-processed DNS feature rows.

        This method is primarily useful during:

            - dataset evaluation
            - benchmarking
            - research experiments
            - bulk analysis

        Returns
        -------
        pandas.DataFrame
            One prediction record per input row.
        """

        if not isinstance(
            df_features,
            pd.DataFrame,
        ):

            raise TypeError(
                "df_features must be a pandas DataFrame."
            )

        if df_features.empty:

            raise ValueError(
                "df_features cannot be empty."
            )

        DNSFeatureSchema.validate_dataframe(
            df_features
        )

        # ---------------------------------------------------------------------
        # ML batch prediction.
        # ---------------------------------------------------------------------

        if self.is_ml_model_available():

            probabilities = (
                self.model_wrapper.predict_proba_batch(
                    df_features
                )
            )

            results = []

            for probability in probabilities:

                probability = (
                    self._clamp_probability(
                        probability
                    )
                )

                class_probabilities = (
                    self._build_class_probabilities(
                        probability
                    )
                )

                predicted_class = (
                    self._classify_probability(
                        probability
                    )
                )

                class_label = (
                    self.CLASS_LABELS[
                        predicted_class
                    ]
                )

                confidence = (
                    self._calculate_confidence(
                        class_probabilities,
                        predicted_class,
                    )
                )

                results.append(
                    {
                        "class_label": class_label,
                        "class_index": predicted_class,
                        "confidence": confidence,
                        "phishing_probability": round(
                            probability,
                            6,
                        ),
                        "legitimate_probability": (
                            class_probabilities[
                                "legitimate"
                            ]
                        ),
                        "model_status": (
                            "loaded_ml_model"
                        ),
                        "prediction_source": (
                            "xgboost"
                        ),
                    }
                )

            self.ml_predictions += len(
                results
            )

            return pd.DataFrame(
                results
            )

        # ---------------------------------------------------------------------
        # Heuristic batch prediction.
        # ---------------------------------------------------------------------

        if self.allow_heuristic_fallback:

            results = []

            for index in range(
                len(df_features)
            ):

                single_df = (
                    df_features.iloc[
                        [index]
                    ]
                )

                result = (
                    self._heuristic_fallback_predict(
                        single_df
                    )
                )

                # DataFrames cannot be cleanly nested in the output table.
                result.pop(
                    "features_dataframe",
                    None,
                )

                result.pop(
                    "heuristic_triggered_rules",
                    None,
                )

                results.append(
                    result
                )

            return pd.DataFrame(
                results
            )

        raise RuntimeError(
            "DNS ML model unavailable and "
            "heuristic fallback disabled."
        )

    # =========================================================================
    # 18. REFRESH MODEL
    # =========================================================================

    def reload_model(
        self,
        model_path: Optional[str] = None,
        strict: bool = False,
    ) -> bool:
        """
        Reload the DNS model from disk.

        Useful when a newly trained model has replaced the old model while
        the application process is still running.
        """

        if model_path is None:
            model_path = self.model_path

        loaded = (
            self.model_wrapper.load_model(
                custom_path=model_path,
                strict=strict,
            )
        )

        self.model = (
            self.model_wrapper.model
        )

        self.is_model_loaded = (
            self.model_wrapper.is_loaded
        )

        logger.info(
            "DNS model reload completed. loaded=%s",
            self.is_model_loaded,
        )

        return loaded

    # =========================================================================
    # 19. MODEL STATUS
    # =========================================================================

    def get_model_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return detailed model availability information.
        """

        return {
            "model_available": (
                self.is_ml_model_available()
            ),

            "model_loaded": (
                self.is_model_loaded
            ),

            "model_path": (
                self.model_wrapper.get_model_path()
            ),

            "model_status": (
                "loaded_ml_model"
                if self.is_ml_model_available()
                else "fallback_available"
                if self.allow_heuristic_fallback
                else "unavailable"
            ),

            "fallback_enabled": (
                self.allow_heuristic_fallback
            ),

            "prediction_threshold": (
                self.prediction_threshold
            ),

            "schema_version": (
                SCHEMA_VERSION
            ),

            "model_wrapper_version": (
                MODEL_WRAPPER_VERSION
            ),

            "preprocessor_version": (
                PREPROCESSOR_VERSION
            ),

            "predictor_version": (
                PREDICTOR_VERSION
            ),
        }

    # =========================================================================
    # 20. RUNTIME STATISTICS
    # =========================================================================

    def get_statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Return runtime inference statistics.

        These counters are process-local and are intended for monitoring,
        debugging and dashboard integration.
        """

        return {
            "total_predictions": (
                self.total_predictions
            ),

            "ml_predictions": (
                self.ml_predictions
            ),

            "fallback_predictions": (
                self.fallback_predictions
            ),

            "failed_predictions": (
                self.failed_predictions
            ),

            "ml_prediction_ratio": (
                round(
                    self.ml_predictions
                    / self.total_predictions,
                    6,
                )
                if self.total_predictions
                else 0.0
            ),

            "fallback_prediction_ratio": (
                round(
                    self.fallback_predictions
                    / self.total_predictions,
                    6,
                )
                if self.total_predictions
                else 0.0
            ),
        }

    # =========================================================================
    # 21. PREDICTOR METADATA
    # =========================================================================

    def get_metadata(
        self,
    ) -> Dict[str, Any]:
        """
        Return predictor configuration and model metadata.
        """

        return {
            "agent": (
                "DNS Security AI Agent"
            ),

            "component": (
                "DNSPredictor"
            ),

            "predictor_version": (
                PREDICTOR_VERSION
            ),

            "model_wrapper_version": (
                MODEL_WRAPPER_VERSION
            ),

            "preprocessor_version": (
                PREPROCESSOR_VERSION
            ),

            "schema_version": (
                SCHEMA_VERSION
            ),

            "feature_count": (
                DNSFeatureSchema.feature_count()
            ),

            "feature_names": (
                DNSFeatureSchema.get_schema()
            ),

            "prediction_threshold": (
                self.prediction_threshold
            ),

            "fallback_enabled": (
                self.allow_heuristic_fallback
            ),

            "model_status": (
                self.get_model_status()
            ),

            "runtime_statistics": (
                self.get_statistics()
            ),
        }

    # =========================================================================
    # 22. UPDATE THRESHOLD
    # =========================================================================

    def set_prediction_threshold(
        self,
        threshold: float,
    ) -> None:
        """
        Change the binary classification threshold.

        Example:

            threshold = 0.50

        means:

            probability >= 0.50
                -> phishing

            probability < 0.50
                -> legitimate

        A different threshold may be selected later based on validation
        experiments and the desired precision/recall tradeoff.
        """

        if not 0.0 <= threshold <= 1.0:

            raise ValueError(
                "Prediction threshold must be between 0.0 and 1.0."
            )

        self.prediction_threshold = float(
            threshold
        )

        logger.info(
            "DNS prediction threshold updated to %.4f",
            self.prediction_threshold,
        )

    # =========================================================================
    # 23. RESET STATISTICS
    # =========================================================================

    def reset_statistics(
        self,
    ) -> None:
        """
        Reset runtime prediction counters.
        """

        self.total_predictions = 0

        self.ml_predictions = 0

        self.fallback_predictions = 0

        self.failed_predictions = 0

        logger.debug(
            "DNS predictor runtime statistics reset."
        )