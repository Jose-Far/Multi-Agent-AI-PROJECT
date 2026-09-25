"""
URL AI Agent - Predictor
========================

Production-grade inference engine for the URL AI Agent.

Classification:
    0 = legitimate
    1 = phishing

Pipeline:

    Raw URL Features
          ↓
    URLFeatureSchema
          ↓
    Strict Feature Alignment
          ↓
    URLPreprocessor
          ↓
    Model Feature Validation
          ↓
    XGBoost / Calibrated Model
          ↓
    Phishing Probability
          ↓
    Decision Threshold
          ↓
    Final Classification

IMPORTANT
---------
This module DOES NOT maintain its own independent feature schema.

The authoritative schema comes from:

    agents/url_agent/feature_schema.py

This prevents:

    trainer.py
        ↓
    model.py
        ↓
    predictor.py

from accidentally using different feature orders.
"""

import os
import json
import logging

from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from .feature_schema import URLFeatureSchema
from .model import URLXGBoostModel


logger = logging.getLogger(__name__)


class URLPredictor:
    """
    Production inference engine for the URL AI Agent.

    Responsibilities
    ----------------
    1. Load the trained URL model.
    2. Obtain the feature schema from URLFeatureSchema.
    3. Normalize incoming feature dictionaries.
    4. Guarantee deterministic feature ordering.
    5. Verify compatibility between model and live features.
    6. Generate phishing probability.
    7. Apply the configured decision threshold.
    8. Provide a safe heuristic fallback if the model is unavailable.
    9. Never convert an inference failure into a fake legitimate result.

    Classification:

        0 = legitimate
        1 = phishing
    """

    # ==================================================================
    # CLASS LABELS
    # ==================================================================

    CLASS_LABELS: Dict[int, str] = {
        0: "legitimate",
        1: "phishing"
    }

    # ==================================================================
    # DEFAULT THRESHOLD
    # ==================================================================

    DEFAULT_DECISION_THRESHOLD = 0.55

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        model_path: Optional[str] = None
    ):
        """
        Initialize URL Predictor.

        Args:
            model_path:
                Optional custom path to the trained model.
        """

        logger.info(
            "Initializing URLPredictor..."
        )

        # --------------------------------------------------------------
        # SINGLE SOURCE OF TRUTH
        # --------------------------------------------------------------

        self.feature_schema: List[str] = (
            URLFeatureSchema.get_schema()
        )

        self.schema_version: str = (
            URLFeatureSchema.SCHEMA_VERSION
        )

        self.feature_count: int = (
            len(self.feature_schema)
        )

        logger.info(
            "URL schema version: %s",
            self.schema_version
        )

        logger.info(
            "URL feature count: %d",
            self.feature_count
        )

        logger.info(
            "URL feature order: %s",
            self.feature_schema
        )

        # --------------------------------------------------------------
        # MODEL WRAPPER
        # --------------------------------------------------------------

        self.model_wrapper = (
            URLXGBoostModel(
                model_path=model_path
            )
            if model_path
            else URLXGBoostModel()
        )

        self.model: Optional[Any] = None

        self.is_model_loaded = False

        # --------------------------------------------------------------
        # DECISION THRESHOLD
        # --------------------------------------------------------------

        self.decision_threshold = (
            self.DEFAULT_DECISION_THRESHOLD
        )

        # --------------------------------------------------------------
        # LOAD MODEL
        # --------------------------------------------------------------

        self._load_active_model()

        # --------------------------------------------------------------
        # LOAD THRESHOLD
        # --------------------------------------------------------------

        self.decision_threshold = (
            self._get_decision_threshold()
        )

        logger.info(
            "URLPredictor initialization complete."
        )

    # ==================================================================
    # LOAD MODEL
    # ==================================================================

    def _load_active_model(
        self
    ) -> None:
        """
        Load the trained model and verify feature compatibility.

        The model loader now validates:

            model schema
                ==
            current URLFeatureSchema

        If the model was trained using the old feature set, it will
        NOT silently be used.
        """

        try:

            self.model = (
                self.model_wrapper.load_model()
            )

            # ----------------------------------------------------------
            # Verify schema one more time at predictor level.
            # ----------------------------------------------------------

            loaded_features = (
                self.model_wrapper.get_feature_names()
            )

            self._validate_model_feature_alignment(
                loaded_features
            )

            self.is_model_loaded = True

            logger.info(
                "URLPredictor: trained model loaded successfully."
            )

        except Exception as e:

            self.model = None

            self.is_model_loaded = False

            logger.warning(
                "URLPredictor: trained model unavailable. "
                "Heuristic fallback will be used. "
                "Reason: %s",
                str(e)
            )

    # ==================================================================
    # MODEL FEATURE VALIDATION
    # ==================================================================

    def _validate_model_feature_alignment(
        self,
        model_features: List[str]
    ) -> None:
        """
        Verify that the loaded model uses exactly the same features
        and ordering as the current live inference pipeline.

        This is one of the most important protections against silent
        feature mismatch.

        Example of an INVALID configuration:

            Model:
                url_entropy
                domain_entropy
                path_entropy
                query_entropy

            Predictor:
                url_entropy
                domain_entropy
                query_entropy
                path_entropy

        Same features, WRONG ORDER.

        That must be rejected.
        """

        expected = (
            self.feature_schema
        )

        received = list(
            model_features
        )

        if received != expected:

            logger.error(
                "CRITICAL URL MODEL FEATURE MISMATCH"
            )

            logger.error(
                "Expected: %s",
                expected
            )

            logger.error(
                "Received: %s",
                received
            )

            missing = [
                feature
                for feature in expected
                if feature not in received
            ]

            extra = [
                feature
                for feature in received
                if feature not in expected
            ]

            if missing:

                logger.error(
                    "Missing model features: %s",
                    missing
                )

            if extra:

                logger.error(
                    "Unexpected model features: %s",
                    extra
                )

            raise ValueError(
                "URL model feature schema does not match "
                "the live URLFeatureSchema. Retrain the model."
            )

        logger.info(
            "URL model feature schema verified successfully."
        )

    # ==================================================================
    # GET MODEL
    # ==================================================================

    def get_model(
        self
    ) -> Optional[Any]:
        """
        Return the loaded model.

        Used by URLExplainer / SHAP.
        """

        return self.model

    # ==================================================================
    # STATUS
    # ==================================================================

    def get_status(
        self
    ) -> Dict[str, Any]:
        """
        Return complete predictor health information.
        """

        return {

            "status":
                (
                    "ready"
                    if self.is_model_loaded
                    else "fallback"
                ),

            "model_available":
                self.is_model_loaded,

            "model_loaded":
                self.is_model_loaded,

            "model_type":
                (
                    type(
                        self.model
                    ).__name__
                    if self.model is not None
                    else None
                ),

            "schema_version":
                self.schema_version,

            "feature_count":
                self.feature_count,

            "feature_names":
                self.feature_schema.copy(),

            "decision_threshold":
                round(
                    float(
                        self.decision_threshold
                    ),
                    4
                ),

            "prediction_source":
                (
                    "xgboost"
                    if self.is_model_loaded
                    else "heuristic_fallback"
                )
        }

    # ==================================================================
    # GET FEATURE SCHEMA
    # ==================================================================

    def get_feature_schema(
        self
    ) -> List[str]:
        """
        Return the authoritative feature order.

        IMPORTANT:
        Never define another feature list inside predictor.py.
        """

        return self.feature_schema.copy()

    # ==================================================================
    # TRAINING METRICS
    # ==================================================================

    def _get_training_metrics(
        self
    ) -> Dict[str, Any]:
        """
        Load training metrics generated by trainer.py.
        """

        metrics_path = (
            "data/models/url_agent/training_metrics.json"
        )

        try:

            if not os.path.exists(
                metrics_path
            ):

                return {
                    "notice":
                        (
                            "Training metrics unavailable. "
                            "Run URL model training first."
                        )
                }

            with open(
                metrics_path,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(
                    file
                )

        except (
            OSError,
            json.JSONDecodeError
        ) as e:

            logger.warning(
                "Unable to load URL training metrics: %s",
                str(e)
            )

            return {
                "notice":
                    "Training metrics could not be read."
            }

    # ==================================================================
    # DECISION THRESHOLD
    # ==================================================================

    def _get_decision_threshold(
        self
    ) -> float:
        """
        Load the decision threshold generated by trainer.py.
        """

        threshold_path = (
            "data/models/url_agent/decision_threshold.json"
        )

        default_threshold = (
            self.DEFAULT_DECISION_THRESHOLD
        )

        try:

            if not os.path.exists(
                threshold_path
            ):

                logger.warning(
                    "Decision threshold file not found. "
                    "Using %.4f.",
                    default_threshold
                )

                return default_threshold

            with open(
                threshold_path,
                "r",
                encoding="utf-8"
            ) as file:

                threshold_data = json.load(
                    file
                )

            threshold = float(
                threshold_data.get(
                    "decision_threshold",
                    default_threshold
                )
            )

            if not (
                0.0 < threshold < 1.0
            ):

                logger.warning(
                    "Invalid decision threshold %.4f. "
                    "Using %.4f.",
                    threshold,
                    default_threshold
                )

                return default_threshold

            logger.info(
                "Loaded URL decision threshold: %.4f",
                threshold
            )

            return threshold

        except (
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError
        ) as e:

            logger.warning(
                "Could not load decision threshold: %s. "
                "Using %.4f.",
                str(e),
                default_threshold
            )

            return default_threshold

    # ==================================================================
    # NORMALIZE FEATURES
    # ==================================================================

    def _normalize_feature_payload(
        self,
        raw_features: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Convert raw URL features into the EXACT current schema.

        The output DataFrame columns are explicitly ordered using:

            self.feature_schema

        This guarantees that XGBoost receives the same feature ordering
        used during training.
        """

        if not isinstance(
            raw_features,
            dict
        ):

            raw_features = {}

        # ==============================================================
        # ALIAS MAP
        # ==============================================================

        alias_map = {

            # HTTPS
            "https":
                "is_https",

            "uses_https":
                "is_https",

            "is_secure":
                "is_https",

            # IP
            "has_ip":
                "contains_ip",

            "is_ip":
                "contains_ip",

            "ip_address":
                "contains_ip",

            # Global entropy
            "entropy":
                "url_entropy",

            "url_randomness":
                "url_entropy",

            # Domain entropy
            "domain_randomness":
                "domain_entropy",

            "domain_entropy_value":
                "domain_entropy",

            # Path entropy
            "path_randomness":
                "path_entropy",

            "path_entropy_value":
                "path_entropy",

            # Query entropy
            "query_randomness":
                "query_entropy",

            "query_entropy_value":
                "query_entropy",

            # URL length
            "length_url":
                "url_length",

            "url_len":
                "url_length",

            # Domain length
            "length_hostname":
                "domain_length",

            "domain_len":
                "domain_length",

            "dom_len":
                "domain_length",

            # Dots
            "qty_dot_url":
                "num_dots",

            "dot_count":
                "num_dots",

            # Hyphens
            "qty_hyphen_url":
                "num_hyphens",

            "hyphen_count":
                "num_hyphens",

            # Underscores
            "qty_underline_url":
                "num_underscores",

            "underscore_count":
                "num_underscores",

            # Slashes
            "qty_slash_url":
                "num_slashes",

            "slash_count":
                "num_slashes",

            # Question marks
            "qty_questionmark_url":
                "num_question_marks",

            "question_mark_count":
                "num_question_marks",

            # Equal signs
            "qty_equal_url":
                "num_equal_signs",

            "equal_sign_count":
                "num_equal_signs",

            # @
            "qty_at_url":
                "num_at_symbols",

            "has_at_symbol":
                "num_at_symbols",

            "at_symbol_count":
                "num_at_symbols",

            # Percent
            "qty_percent_url":
                "num_percent_signs",

            "percent_count":
                "num_percent_signs",

            # Digits
            "qty_numeric_url":
                "num_digits",

            "numeric_count":
                "num_digits",

            # Special characters
            "special_characters":
                "num_special_chars",

            "special_chars_count":
                "num_special_chars",

            "special_character_count":
                "num_special_chars",

            # Suspicious words
            "suspicious_keywords":
                "suspicious_word_count",

            "suspicious_words_count":
                "suspicious_word_count",

            "num_suspicious_words":
                "suspicious_word_count",

            # Punycode
            "is_homograph":
                "has_punycode",

            "punycode":
                "has_punycode",

            # Subdomains
            "subdomains":
                "subdomain_count",

            "subdomain_count_value":
                "subdomain_count",

            # TLD
            "tld_len":
                "tld_length"
        }

        # ==============================================================
        # APPLY ALIASES
        # ==============================================================

        normalized: Dict[str, Any] = {}

        for key, value in raw_features.items():

            mapped_key = (
                alias_map.get(
                    key,
                    key
                )
            )

            normalized[
                mapped_key
            ] = value

        # ==============================================================
        # BUILD STRICT FEATURE VECTOR
        # ==============================================================

        row_vector: Dict[str, float] = {}

        for feature in self.feature_schema:

            value = normalized.get(
                feature,
                0.0
            )

            # ----------------------------------------------------------
            # Boolean
            # ----------------------------------------------------------

            if isinstance(
                value,
                bool
            ):

                row_vector[
                    feature
                ] = (
                    1.0
                    if value
                    else 0.0
                )

            # ----------------------------------------------------------
            # List
            # ----------------------------------------------------------

            elif isinstance(
                value,
                list
            ):

                row_vector[
                    feature
                ] = float(
                    len(value)
                )

            # ----------------------------------------------------------
            # Numeric
            # ----------------------------------------------------------

            else:

                try:

                    numeric_value = float(
                        value
                    )

                    # Protect XGBoost from NaN / infinity.
                    if not np.isfinite(
                        numeric_value
                    ):

                        numeric_value = 0.0

                    row_vector[
                        feature
                    ] = numeric_value

                except (
                    TypeError,
                    ValueError
                ):

                    logger.warning(
                        "Invalid value for URL feature '%s'. "
                        "Using 0.0.",
                        feature
                    )

                    row_vector[
                        feature
                    ] = 0.0

        # ==============================================================
        # CRITICAL:
        #
        # Explicit column ordering.
        # ==============================================================

        dataframe = pd.DataFrame(
            [row_vector],
            columns=self.feature_schema
        )

        # ==============================================================
        # FINAL SAFETY CHECK
        # ==============================================================

        if list(
            dataframe.columns
        ) != self.feature_schema:

            raise RuntimeError(
                "Live URL feature ordering is incorrect."
            )

        if len(
            dataframe.columns
        ) != self.feature_count:

            raise RuntimeError(
                "Live URL feature count does not match schema."
            )

        return dataframe

    # ==================================================================
    # HEURISTIC FALLBACK
    # ==================================================================

    def _heuristic_fallback_predict(
        self,
        df_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Conservative heuristic fallback.

        IMPORTANT:
        Path entropy is NOT treated as a strong standalone phishing
        signal because legitimate services frequently use long/random
        generated identifiers.

        Domain entropy receives stronger consideration.
        """

        features = (
            df_features.iloc[
                0
            ].to_dict()
        )

        # --------------------------------------------------------------
        # Extract features
        # --------------------------------------------------------------

        domain_entropy = float(
            features.get(
                "domain_entropy",
                0.0
            )
        )

        path_entropy = float(
            features.get(
                "path_entropy",
                0.0
            )
        )

        query_entropy = float(
            features.get(
                "query_entropy",
                0.0
            )
        )

        suspicious_words = float(
            features.get(
                "suspicious_word_count",
                0.0
            )
        )

        contains_ip = float(
            features.get(
                "contains_ip",
                0.0
            )
        )

        is_https = float(
            features.get(
                "is_https",
                1.0
            )
        )

        special_chars = float(
            features.get(
                "num_special_chars",
                0.0
            )
        )

        url_length = float(
            features.get(
                "url_length",
                0.0
            )
        )

        hyphens = float(
            features.get(
                "num_hyphens",
                0.0
            )
        )

        # --------------------------------------------------------------
        # Risk calculation
        # --------------------------------------------------------------

        risk_points = 0.0

        # Strong signal
        if contains_ip == 1.0:

            risk_points += 0.40

        # Strong signal
        if suspicious_words > 0:

            risk_points += min(
                0.30,
                suspicious_words * 0.15
            )

        # DOMAIN entropy:
        # stronger than path entropy.
        if domain_entropy > 4.5:

            risk_points += 0.20

        elif domain_entropy > 3.8:

            risk_points += 0.08

        # PATH entropy:
        # intentionally weak contextual signal.
        if path_entropy > 5.0:

            risk_points += 0.04

        # QUERY entropy:
        # moderate contextual signal.
        if query_entropy > 4.5:

            risk_points += 0.08

        if is_https == 0.0:

            risk_points += 0.15

        if special_chars > 5:

            risk_points += 0.10

        if url_length > 100:

            risk_points += 0.08

        if hyphens > 3:

            risk_points += 0.05

        phishing_probability = min(
            0.99,
            max(
                0.01,
                risk_points
            )
        )

        legitimate_probability = (
            1.0
            -
            phishing_probability
        )

        # --------------------------------------------------------------
        # Decision
        # --------------------------------------------------------------

        decision_threshold = (
            self.decision_threshold
        )

        predicted_class = (

            1

            if phishing_probability
            >= decision_threshold

            else 0
        )

        class_label = (
            self.CLASS_LABELS[
                predicted_class
            ]
        )

        confidence = (

            phishing_probability
            if predicted_class == 1
            else legitimate_probability
        )

        return {

            "class_label":
                class_label,

            "class_index":
                predicted_class,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "phishing_probability":
                round(
                    phishing_probability,
                    4
                ),

            "class_probabilities": {

                "legitimate":
                    round(
                        legitimate_probability,
                        4
                    ),

                "phishing":
                    round(
                        phishing_probability,
                        4
                    )
            },

            "decision_threshold":
                round(
                    decision_threshold,
                    4
                ),

            "features_dataframe":
                df_features,

            "model_status":
                "fallback_heuristic",

            "prediction_source":
                "heuristic_fallback",

            "fallback_used":
                True,

            "signal_available":
                True,

            "schema_version":
                self.schema_version,

            "feature_count":
                self.feature_count,

            "feature_names":
                self.feature_schema.copy(),

            "training_evaluation":
                self._get_training_metrics()
        }

    # ==================================================================
    # MAIN PREDICTION
    # ==================================================================

    def predict(
        self,
        url_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute URL classification.

        Live inference uses EXACTLY the same feature ordering as
        training.
        """

        if not isinstance(
            url_features,
            dict
        ):

            logger.warning(
                "URLPredictor received invalid feature payload."
            )

            url_features = {}

        try:

            # ==========================================================
            # STEP 1
            # Strict normalization
            # ==========================================================

            df_features = (
                self._normalize_feature_payload(
                    url_features
                )
            )

            # ==========================================================
            # STEP 2
            # Validate columns before model inference
            # ==========================================================

            if list(
                df_features.columns
            ) != self.feature_schema:

                raise RuntimeError(
                    "Live feature columns do not exactly match "
                    "URLFeatureSchema."
                )

            # ==========================================================
            # STEP 3
            # Model availability
            # ==========================================================

            if (
                not self.is_model_loaded
                or self.model is None
            ):

                return (
                    self._heuristic_fallback_predict(
                        df_features
                    )
                )

            # ==========================================================
            # STEP 4
            # Verify model feature metadata
            # ==========================================================

            model_features = (
                self.model_wrapper.get_feature_names()
            )

            self._validate_model_feature_alignment(
                model_features
            )

            # ==========================================================
            # STEP 5
            # XGBoost prediction
            # ==========================================================

            probabilities = (
                self.model.predict_proba(
                    df_features
                )
            )

            if (
                probabilities is None
                or len(probabilities) == 0
            ):

                raise ValueError(
                    "URL model returned no probability output."
                )

            probabilities = (
                probabilities[0]
            )

            if len(
                probabilities
            ) < 2:

                raise ValueError(
                    "URL model probability vector "
                    "does not contain both classes."
                )

            # ==========================================================
            # STEP 6
            # Extract probabilities
            # ==========================================================

            legitimate_probability = float(
                probabilities[0]
            )

            phishing_probability = float(
                probabilities[1]
            )

            legitimate_probability = max(
                0.0,
                min(
                    1.0,
                    legitimate_probability
                )
            )

            phishing_probability = max(
                0.0,
                min(
                    1.0,
                    phishing_probability
                )
            )

            # ==========================================================
            # STEP 7
            # Decision threshold
            # ==========================================================

            decision_threshold = (
                self.decision_threshold
            )

            predicted_class = (

                1

                if phishing_probability
                >= decision_threshold

                else 0
            )

            class_label = (
                self.CLASS_LABELS[
                    predicted_class
                ]
            )

            # ==========================================================
            # STEP 8
            # Confidence
            # ==========================================================

            confidence = (

                phishing_probability
                if predicted_class == 1
                else legitimate_probability
            )

            # ==========================================================
            # STEP 9
            # Output
            # ==========================================================

            return {

                "class_label":
                    class_label,

                "class_index":
                    predicted_class,

                "confidence":
                    round(
                        confidence,
                        4
                    ),

                "phishing_probability":
                    round(
                        phishing_probability,
                        4
                    ),

                "class_probabilities": {

                    "legitimate":
                        round(
                            legitimate_probability,
                            4
                        ),

                    "phishing":
                        round(
                            phishing_probability,
                            4
                        )
                },

                "decision_threshold":
                    round(
                        decision_threshold,
                        4
                    ),

                "features_dataframe":
                    df_features,

                "model_status":
                    "loaded_ml_model",

                "prediction_source":
                    "xgboost",

                "fallback_used":
                    False,

                "signal_available":
                    True,

                "schema_version":
                    self.schema_version,

                "feature_count":
                    self.feature_count,

                "feature_names":
                    self.feature_schema.copy(),

                "training_evaluation":
                    self._get_training_metrics()
            }

        except Exception as e:

            logger.error(
                "URLPredictor inference failed: %s",
                str(e),
                exc_info=True
            )

            # ----------------------------------------------------------
            # IMPORTANT:
            #
            # Never turn an inference error into:
            #
            #     legitimate / 0 risk
            #
            # because that creates false security.
            # ----------------------------------------------------------

            return {

                "class_label":
                    "unknown",

                "class_index":
                    None,

                "confidence":
                    0.0,

                "phishing_probability":
                    0.0,

                "class_probabilities": {

                    "legitimate":
                        0.0,

                    "phishing":
                        0.0
                },

                "decision_threshold":
                    round(
                        self.decision_threshold,
                        4
                    ),

                "features_dataframe":
                    pd.DataFrame(),

                "model_status":
                    "error",

                "prediction_source":
                    "none",

                "fallback_used":
                    False,

                "signal_available":
                    False,

                "schema_version":
                    self.schema_version,

                "feature_count":
                    self.feature_count,

                "error":
                    str(e),

                "training_evaluation":
                    self._get_training_metrics()
            }