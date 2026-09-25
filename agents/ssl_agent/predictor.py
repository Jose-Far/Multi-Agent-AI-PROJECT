"""
SSL/TLS Security AI Agent - Predictor
=====================================

Production-grade inference engine for the SSL/TLS Security AI Agent.

Pipeline
--------

    Raw SSL/TLS Features
            |
            v
    SSLPreprocessor
            |
            v
    Canonical 17 Features
            |
            v
    SSLModel
            |
            v
    XGBoost Probability
            |
            v
    SSL Prediction
            |
            v
    phishing_probability
            |
            v
    SSLRiskScorer
            |
            v
    SSLExplainer


Prediction convention
---------------------

    class_index = 0
        legitimate

    class_index = 1
        phishing / suspicious


Important
---------

This class does NOT calculate the final SSL risk score.

It only produces the model prediction and normalized probability.

The final risk score belongs to:

    agents/ssl_agent/risk_score.py
"""

import os
import logging

from pathlib import Path

from typing import (
    Dict,
    Any,
    Optional,
    List
)

import numpy as np
import pandas as pd

from .model import SSLModel
from .preprocessing import SSLPreprocessor
from .feature_schema import SSLFeatureSchema


logger = logging.getLogger(__name__)


class SSLPredictor:
    """
    Production-grade inference engine for the SSL/TLS AI Agent.

    Responsibilities
    ----------------

        • Load trained SSL model
        • Validate model/schema compatibility
        • Preprocess raw SSL features
        • Execute XGBoost inference
        • Produce class probabilities
        • Produce phishing probability
        • Produce confidence
        • Provide heuristic fallback
        • Provide model status
        • Provide feature vector
        • Support explainability tools

    The predictor deliberately does NOT calculate the final risk score.
    """

    # ======================================================================
    # METADATA
    # ======================================================================

    PREDICTOR_NAME = (
        "SSL Predictor"
    )

    PREDICTOR_VERSION = "1.1.0"

    # ======================================================================
    # CLASS LABELS
    # ======================================================================

    CLASS_LABELS: Dict[
        int,
        str
    ] = {

        0:
            "legitimate",

        1:
            "phishing"
    }

    # ======================================================================
    # DEFAULT PROBABILITY
    # ======================================================================

    DEFAULT_PROBABILITY = 0.50

    # ======================================================================
    # PROJECT MODEL PATH
    # ======================================================================

    @staticmethod
    def _get_project_root() -> Path:
        """
        Resolve the project root.

        predictor.py is located at:

            <project_root>/agents/ssl_agent/predictor.py

        Therefore:

            parents[0] = ssl_agent
            parents[1] = agents
            parents[2] = project_root
        """

        return Path(
            __file__
        ).resolve().parents[2]

    @classmethod
    def _get_default_model_path(cls) -> str:
        """
        Return the canonical trained SSL model artifact path.
        """

        project_root = (
            cls._get_project_root()
        )

        model_path = (
            project_root
            / "data"
            / "models"
            / "ssl_agent"
            / "ssl_xgboost_model.joblib"
        )

        return str(
            model_path
        )

    # ======================================================================
    # INITIALIZATION
    # ======================================================================

    def __init__(
        self,
        model_path: Optional[str] = None,
        allow_heuristic_fallback: bool = True
    ):
        """
        Initialize the SSL predictor.

        Args:
            model_path:
                Optional model artifact path OR model directory.

                If omitted, the predictor automatically uses:

                    data/models/ssl_agent/
                    ssl_xgboost_model.joblib

            allow_heuristic_fallback:
                If True, inference can continue using the deterministic
                SSL heuristic when no trained ML model is available.
        """

        # ------------------------------------------------------------------
        # Predictor configuration
        # ------------------------------------------------------------------

        self.allow_heuristic_fallback = bool(
            allow_heuristic_fallback
        )

        # ------------------------------------------------------------------
        # Canonical default model path
        # ------------------------------------------------------------------

        self.default_model_path = (
            self._get_default_model_path()
        )

        # ------------------------------------------------------------------
        # Preprocessor
        # ------------------------------------------------------------------

        self.preprocessor = (
            SSLPreprocessor()
        )

        # ------------------------------------------------------------------
        # Model path resolution
        # ------------------------------------------------------------------

        self.model_wrapper = (
            self._initialize_model(
                model_path
            )
        )

        # ------------------------------------------------------------------
        # Underlying model
        # ------------------------------------------------------------------

        self.model = (
            self.model_wrapper.get_model()
        )

        # ------------------------------------------------------------------
        # Model status
        # ------------------------------------------------------------------

        self.is_model_loaded = (
            self.model_wrapper.is_trained
            and
            self.model is not None
        )

        # ------------------------------------------------------------------
        # Last prediction
        # ------------------------------------------------------------------

        self.last_prediction: Optional[
            Dict[str, Any]
        ] = None

        # ------------------------------------------------------------------
        # Last error
        # ------------------------------------------------------------------

        self.last_error: Optional[
            str
        ] = None

        # ------------------------------------------------------------------
        # Validate model compatibility
        # ------------------------------------------------------------------

        if self.is_model_loaded:

            compatible, errors = (
                self.model_wrapper.check_compatibility()
            )

            if not compatible:

                logger.error(
                    (
                        "Loaded SSL model is incompatible "
                        "with the current feature schema: %s"
                    ),
                    errors
                )

                self.is_model_loaded = False

                self.last_error = (
                    "Model/schema compatibility failure."
                )

        # ------------------------------------------------------------------
        # Initialization status
        # ------------------------------------------------------------------

        if self.is_model_loaded:

            logger.info(
                (
                    "SSLPredictor initialized with "
                    "trained ML model."
                )
            )

            logger.info(
                "SSL model path: %s",
                self.default_model_path
            )

        else:

            if self.allow_heuristic_fallback:

                logger.warning(
                    (
                        "No compatible trained SSL model is available. "
                        "Heuristic fallback is enabled."
                    )
                )

            else:

                logger.warning(
                    (
                        "No compatible trained SSL model is available. "
                        "Heuristic fallback is disabled."
                    )
                )

    # ======================================================================
    # MODEL INITIALIZATION
    # ======================================================================

    def _initialize_model(
        self,
        model_path: Optional[str]
    ) -> SSLModel:
        """
        Initialize SSLModel and attempt to load the model artifact.

        Handles:

            1. No custom path
            2. Model directory
            3. Direct .joblib model path
        """

        # ------------------------------------------------------------------
        # No custom path
        #
        # IMPORTANT:
        # Never rely on SSLModel's internal default path here.
        # Explicitly point to the trained project model.
        # ------------------------------------------------------------------

        if not model_path:

            resolved_path = (
                self.default_model_path
            )

            logger.info(
                (
                    "No custom SSL model path supplied. "
                    "Using project model: %s"
                ),
                resolved_path
            )

            model_wrapper = (
                SSLModel(
                    model_path=resolved_path
                )
            )

            if not os.path.isfile(
                resolved_path
            ):

                logger.warning(
                    (
                        "SSL model artifact not found at: %s"
                    ),
                    resolved_path
                )

                return model_wrapper

            model_loaded = (
                model_wrapper.load(
                    resolved_path
                )
            )

            if model_loaded:

                logger.info(
                    (
                        "Default project SSL model "
                        "loaded successfully."
                    )
                )

            else:

                logger.warning(
                    (
                        "Default project SSL model "
                        "could not be loaded."
                    )
                )

            return model_wrapper

        # ------------------------------------------------------------------
        # Normalize custom path
        # ------------------------------------------------------------------

        resolved_path = os.path.abspath(
            os.path.expanduser(
                model_path
            )
        )

        # ------------------------------------------------------------------
        # Directory path
        # ------------------------------------------------------------------

        if os.path.isdir(
            resolved_path
        ):

            model_wrapper = (
                SSLModel(
                    model_dir=resolved_path
                )
            )

            model_loaded = (
                model_wrapper.load()
            )

            if model_loaded:

                logger.info(
                    (
                        "SSL model loaded from "
                        "directory: %s"
                    ),
                    resolved_path
                )

            else:

                logger.warning(
                    (
                        "No SSL model artifact found "
                        "inside directory: %s"
                    ),
                    resolved_path
                )

            return model_wrapper

        # ------------------------------------------------------------------
        # Explicit file path
        # ------------------------------------------------------------------

        model_wrapper = (
            SSLModel(
                model_path=resolved_path
            )
        )

        model_loaded = (
            model_wrapper.load(
                resolved_path
            )
        )

        if model_loaded:

            logger.info(
                (
                    "SSL model loaded from file: %s"
                ),
                resolved_path
            )

        else:

            logger.warning(
                (
                    "Unable to load SSL model "
                    "from file: %s"
                ),
                resolved_path
            )

        return model_wrapper

    # ======================================================================
    # GET MODEL
    # ======================================================================

    def get_model(self) -> Optional[Any]:
        """
        Return the underlying XGBoost model.
        """

        return self.model

    # ======================================================================
    # GET MODEL WRAPPER
    # ======================================================================

    def get_model_wrapper(
        self
    ) -> SSLModel:
        """
        Return the SSLModel wrapper.
        """

        return self.model_wrapper

    # ======================================================================
    # REFRESH MODEL
    # ======================================================================

    def reload_model(
        self,
        model_path: Optional[str] = None
    ) -> bool:
        """
        Reload the SSL model artifact.

        If model_path is omitted, reload the canonical project model.
        """

        try:

            self.model_wrapper = (
                self._initialize_model(
                    model_path
                )
            )

            self.model = (
                self.model_wrapper.get_model()
            )

            self.is_model_loaded = (
                self.model_wrapper.is_trained
                and
                self.model is not None
            )

            if self.is_model_loaded:

                compatible, errors = (
                    self.model_wrapper.check_compatibility()
                )

                if not compatible:

                    self.is_model_loaded = False

                    self.last_error = (
                        "Reloaded model is incompatible: "
                        +
                        str(errors)
                    )

                    return False

            self.last_error = None

            return self.is_model_loaded

        except Exception as e:

            self.last_error = str(
                e
            )

            self.is_model_loaded = False

            logger.error(
                "Failed to reload SSL model: %s",
                str(e),
                exc_info=True
            )

            return False

    # ======================================================================
    # NORMALIZE PROBABILITY
    # ======================================================================

    @staticmethod
    def _normalize_probability(
        value: Any
    ) -> float:
        """
        Safely normalize probability into [0.0, 1.0].
        """

        try:

            probability = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return 0.50

        if not np.isfinite(
            probability
        ):

            return 0.50

        return float(
            np.clip(
                probability,
                0.0,
                1.0
            )
        )

    # ======================================================================
    # BUILD CLASS PROBABILITIES
    # ======================================================================

    def _build_class_probabilities(
        self,
        legitimate_probability: float,
        phishing_probability: float
    ) -> Dict[str, float]:
        """
        Build standardized class probability dictionary.
        """

        legitimate_probability = (
            self._normalize_probability(
                legitimate_probability
            )
        )

        phishing_probability = (
            self._normalize_probability(
                phishing_probability
            )
        )

        total = (
            legitimate_probability
            +
            phishing_probability
        )

        if total <= 0:

            legitimate_probability = 0.50
            phishing_probability = 0.50

        else:

            legitimate_probability = (
                legitimate_probability
                /
                total
            )

            phishing_probability = (
                phishing_probability
                /
                total
            )

        return {

            "legitimate":
                round(
                    legitimate_probability,
                    6
                ),

            "phishing":
                round(
                    phishing_probability,
                    6
                )
        }

    # ======================================================================
    # DETERMINE CLASS
    # ======================================================================

    def _determine_class(
        self,
        phishing_probability: float
    ) -> Dict[str, Any]:
        """
        Convert phishing probability into class decision.

        Threshold:

            >= 0.50 → phishing
            <  0.50 → legitimate
        """

        phishing_probability = (
            self._normalize_probability(
                phishing_probability
            )
        )

        legitimate_probability = (
            1.0
            -
            phishing_probability
        )

        predicted_class = (
            1
            if phishing_probability >= 0.50
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

        class_probabilities = (
            self._build_class_probabilities(
                legitimate_probability,
                phishing_probability
            )
        )

        return {

            "class_label":
                class_label,

            "class_index":
                predicted_class,

            "confidence":
                round(
                    float(
                        confidence
                    ),
                    6
                ),

            "phishing_probability":
                round(
                    float(
                        phishing_probability
                    ),
                    6
                ),

            "legitimate_probability":
                round(
                    float(
                        legitimate_probability
                    ),
                    6
                ),

            "class_probabilities":
                class_probabilities
        }

    # ======================================================================
    # HEURISTIC FALLBACK
    # ======================================================================

    def _heuristic_fallback_predict(
        self,
        df_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Deterministic heuristic fallback.

        This is NOT the trained ML model.
        """

        try:

            if not isinstance(
                df_features,
                pd.DataFrame
            ):

                raise TypeError(
                    "Heuristic input must be a DataFrame."
                )

            if len(
                df_features
            ) != 1:

                raise ValueError(
                    (
                        "Heuristic predictor expects exactly "
                        "one feature row."
                    )
                )

            features = (
                df_features.iloc[
                    0
                ].to_dict()
            )

            risk_points = 0.0

            reasons: List[
                str
            ] = []

            # ==============================================================
            # SSL AVAILABILITY
            # ==============================================================

            has_ssl = float(
                features.get(
                    "has_ssl",
                    0.0
                )
            )

            if has_ssl == 0.0:

                risk_points += 0.45

                reasons.append(
                    "No valid SSL/TLS certificate detected."
                )

            # ==============================================================
            # EXPIRATION
            # ==============================================================

            is_expired = float(
                features.get(
                    "is_expired",
                    0.0
                )
            )

            if is_expired == 1.0:

                risk_points += 0.40

                reasons.append(
                    "Certificate is expired."
                )

            # ==============================================================
            # SELF SIGNED
            # ==============================================================

            is_self_signed = float(
                features.get(
                    "is_self_signed",
                    0.0
                )
            )

            if is_self_signed == 1.0:

                risk_points += 0.40

                reasons.append(
                    "Certificate is self-signed."
                )

            # ==============================================================
            # SUSPICIOUS CA
            # ==============================================================

            is_suspicious_ca = float(
                features.get(
                    "is_suspicious_ca",
                    0.0
                )
            )

            if is_suspicious_ca == 1.0:

                risk_points += 0.30

                reasons.append(
                    "Certificate authority appears suspicious."
                )

            # ==============================================================
            # FREE AUTOMATED CA
            # ==============================================================

            is_free_automated_ca = float(
                features.get(
                    "is_free_automated_ca",
                    0.0
                )
            )

            if is_free_automated_ca == 1.0:

                risk_points += 0.05

                reasons.append(
                    "Certificate was issued by a free/automated CA."
                )

            # ==============================================================
            # RECENTLY ISSUED
            # ==============================================================

            is_recently_issued = float(
                features.get(
                    "is_recently_issued",
                    0.0
                )
            )

            if is_recently_issued == 1.0:

                risk_points += 0.15

                reasons.append(
                    "Certificate was recently issued."
                )

            # ==============================================================
            # SHORT LIVED
            # ==============================================================

            is_short_lived = float(
                features.get(
                    "is_short_lived",
                    0.0
                )
            )

            if is_short_lived == 1.0:

                risk_points += 0.10

                reasons.append(
                    "Certificate has a short validity period."
                )

            # ==============================================================
            # WEAK PROTOCOL
            # ==============================================================

            is_weak_protocol = float(
                features.get(
                    "is_weak_protocol",
                    0.0
                )
            )

            if is_weak_protocol == 1.0:

                risk_points += 0.30

                reasons.append(
                    "Weak or obsolete TLS protocol detected."
                )

            # ==============================================================
            # VULNERABLE CIPHER
            # ==============================================================

            is_vulnerable_cipher = float(
                features.get(
                    "is_vulnerable_cipher",
                    0.0
                )
            )

            if is_vulnerable_cipher == 1.0:

                risk_points += 0.20

                reasons.append(
                    "Potentially vulnerable cipher configuration detected."
                )

            # ==============================================================
            # CRYPTO HEALTH
            # ==============================================================

            crypto_health = float(
                features.get(
                    "crypto_health_score",
                    0.0
                )
            )

            if crypto_health < 50.0:

                risk_points += 0.15

                reasons.append(
                    "Overall cryptographic health is poor."
                )

            elif crypto_health >= 90.0:

                risk_points -= 0.10

            # ==============================================================
            # SECURE CONNECTION
            # ==============================================================

            is_secure_connection = float(
                features.get(
                    "is_secure_connection",
                    0.0
                )
            )

            if is_secure_connection == 0.0:

                risk_points += 0.20

                reasons.append(
                    "Connection is not considered secure."
                )

            # ==============================================================
            # ISSUER TRUST
            # ==============================================================

            issuer_trust_score = float(
                features.get(
                    "issuer_trust_score",
                    0.0
                )
            )

            if issuer_trust_score < 30.0:

                risk_points += 0.15

                reasons.append(
                    "Certificate issuer trust score is low."
                )

            # ==============================================================
            # TRUSTED CA
            # ==============================================================

            is_trusted_ca = float(
                features.get(
                    "is_trusted_ca",
                    0.0
                )
            )

            if (
                is_trusted_ca == 1.0
                and
                is_suspicious_ca == 0.0
            ):

                risk_points -= 0.05

            # ==============================================================
            # FINAL PROBABILITY
            # ==============================================================

            phishing_probability = np.clip(
                risk_points,
                0.01,
                0.99
            )

            prediction = (
                self._determine_class(
                    phishing_probability
                )
            )

            return {

                **prediction,

                "features_dataframe":
                    df_features,

                "features":
                    self._dataframe_to_feature_dict(
                        df_features
                    ),

                "model_status":
                    "fallback_heuristic",

                "model_type":
                    "heuristic",

                "heuristic_reasons":
                    reasons,

                "heuristic_risk_points":
                    round(
                        float(
                            risk_points
                        ),
                        6
                    )
            }

        except Exception as e:

            logger.error(
                (
                    "SSL heuristic fallback failed: %s"
                ),
                str(e),
                exc_info=True
            )

            return (
                self._safe_neutral_result(
                    df_features,
                    model_status=(
                        "heuristic_error"
                    )
                )
            )

    # ======================================================================
    # ML PREDICTION
    # ======================================================================

    def _ml_predict(
        self,
        df_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Execute XGBoost prediction on one preprocessed SSL feature vector.
        """

        if not self.is_model_loaded:

            raise RuntimeError(
                "SSL ML model is not loaded."
            )

        probabilities = (
            self.model_wrapper.predict_proba(
                df_features
            )
        )

        if probabilities.shape != (
            1,
            2
        ):

            raise RuntimeError(
                (
                    "Unexpected SSL probability shape: "
                    f"{probabilities.shape}"
                )
            )

        legitimate_probability = float(
            probabilities[
                0,
                0
            ]
        )

        phishing_probability = float(
            probabilities[
                0,
                1
            ]
        )

        prediction = (
            self._determine_class(
                phishing_probability
            )
        )

        return {

            **prediction,

            "features_dataframe":
                df_features,

            "features":
                self._dataframe_to_feature_dict(
                    df_features
                ),

            "model_status":
                "loaded_ml_model",

            "model_type":
                "xgboost",

            "model_version":
                self.model_wrapper.MODEL_VERSION,

            "model_based":
                True,

            "prediction_source":
                "xgboost"
        }

    # ======================================================================
    # MAIN PREDICT METHOD
    # ======================================================================

    def predict(
        self,
        raw_ssl_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute SSL inference.
        """

        if not isinstance(
            raw_ssl_features,
            dict
        ):

            logger.warning(
                (
                    "SSLPredictor received invalid input type: %s. "
                    "Using empty feature dictionary."
                ),
                type(
                    raw_ssl_features
                ).__name__
            )

            raw_ssl_features = {}

        try:

            # ==============================================================
            # STEP 1 — PREPROCESS
            # ==============================================================

            df_features = (
                self.preprocessor.transform_single(
                    raw_ssl_features
                )
            )

            # ==============================================================
            # STEP 2 — VERIFY FEATURE VECTOR
            # ==============================================================

            expected_features = (
                SSLFeatureSchema.get_feature_names()
            )

            if list(
                df_features.columns
            ) != expected_features:

                raise ValueError(
                    (
                        "SSL predictor received an invalid "
                        "feature vector.\n"
                        f"Expected={expected_features}\n"
                        f"Received={list(df_features.columns)}"
                    )
                )

            if len(
                df_features
            ) != 1:

                raise ValueError(
                    (
                        "SSL predictor expects exactly "
                        "one feature row."
                    )
                )

            # ==============================================================
            # STEP 3 — ML MODEL
            # ==============================================================

            if self.is_model_loaded:

                result = (
                    self._ml_predict(
                        df_features
                    )
                )

            # ==============================================================
            # STEP 4 — HEURISTIC FALLBACK
            # ==============================================================

            elif self.allow_heuristic_fallback:

                result = (
                    self._heuristic_fallback_predict(
                        df_features
                    )
                )

                result[
                    "model_based"
                ] = False

                result[
                    "prediction_source"
                ] = "heuristic_fallback"

            # ==============================================================
            # STEP 5 — NO MODEL
            # ==============================================================

            else:

                result = (
                    self._safe_neutral_result(
                        df_features,
                        model_status=(
                            "model_unavailable"
                        )
                    )
                )

                result[
                    "model_based"
                ] = False

                result[
                    "prediction_source"
                ] = "unavailable"

            self.last_prediction = (
                result
            )

            self.last_error = None

            return result

        except Exception as e:

            self.last_error = str(
                e
            )

            logger.error(
                (
                    "SSL inference pipeline execution "
                    "error: %s"
                ),
                str(e),
                exc_info=True
            )

            return (
                self._safe_neutral_result(
                    pd.DataFrame(),
                    model_status=(
                        "error_fallback"
                    ),
                    error=str(e)
                )
            )

    # ======================================================================
    # PREDICT FROM DATAFRAME
    # ======================================================================

    def predict_dataframe(
        self,
        df_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Predict from a single-row DataFrame.
        """

        if not isinstance(
            df_features,
            pd.DataFrame
        ):

            raise TypeError(
                "df_features must be a pandas DataFrame."
            )

        if len(
            df_features
        ) != 1:

            raise ValueError(
                (
                    "predict_dataframe() expects "
                    "exactly one row."
                )
            )

        try:

            raw_features = (
                df_features.iloc[
                    0
                ].to_dict()
            )

            return self.predict(
                raw_features
            )

        except Exception as e:

            logger.error(
                (
                    "DataFrame SSL prediction failed: %s"
                ),
                str(e),
                exc_info=True
            )

            return (
                self._safe_neutral_result(
                    df_features,
                    model_status=(
                        "error_fallback"
                    ),
                    error=str(e)
                )
            )

    # ======================================================================
    # FEATURE DICTIONARY
    # ======================================================================

    @staticmethod
    def _dataframe_to_feature_dict(
        df_features: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Convert one-row feature DataFrame into a JSON-friendly dictionary.
        """

        if (
            not isinstance(
                df_features,
                pd.DataFrame
            )
            or
            df_features.empty
        ):

            return {}

        row = (
            df_features.iloc[
                0
            ]
        )

        result = {}

        for feature_name in (
            df_features.columns
        ):

            try:

                value = float(
                    row[
                        feature_name
                    ]
                )

            except (
                TypeError,
                ValueError
            ):

                value = 0.0

            if not np.isfinite(
                value
            ):

                value = 0.0

            result[
                feature_name
            ] = value

        return result

    # ======================================================================
    # SAFE NEUTRAL RESULT
    # ======================================================================

    def _safe_neutral_result(
        self,
        df_features: pd.DataFrame,
        model_status: str,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Return a deterministic neutral prediction.
        """

        phishing_probability = (
            self.DEFAULT_PROBABILITY
        )

        prediction = (
            self._determine_class(
                phishing_probability
            )
        )

        return {

            **prediction,

            "features_dataframe":
                df_features,

            "features":
                self._dataframe_to_feature_dict(
                    df_features
                ),

            "model_status":
                model_status,

            "model_type":
                "unavailable",

            "model_based":
                False,

            "prediction_source":
                "unavailable",

            "error":
                error
        }

    # ======================================================================
    # GET FEATURE IMPORTANCES
    # ======================================================================

    def get_feature_importances(
        self
    ) -> Dict[str, float]:

        if not self.is_model_loaded:

            return {}

        return (
            self.model_wrapper.get_feature_importances()
        )

    # ======================================================================
    # GET FEATURE NAMES
    # ======================================================================

    def get_feature_names(
        self
    ) -> List[str]:

        return (
            SSLFeatureSchema.get_feature_names()
        )

    # ======================================================================
    # GET FEATURE COUNT
    # ======================================================================

    def get_feature_count(
        self
    ) -> int:

        return len(
            SSLFeatureSchema.get_feature_names()
        )

    # ======================================================================
    # GET MODEL STATUS
    # ======================================================================

    def get_status(
        self
    ) -> Dict[str, Any]:
        """
        Return complete predictor status.
        """

        return {

            "predictor_name":
                self.PREDICTOR_NAME,

            "predictor_version":
                self.PREDICTOR_VERSION,

            "model_loaded":
                self.is_model_loaded,

            "model_ready":
                self.is_ready(),

            "model_path":
                self.default_model_path,

            "model_exists":
                os.path.isfile(
                    self.default_model_path
                ),

            "heuristic_fallback_enabled":
                self.allow_heuristic_fallback,

            "feature_count":
                self.get_feature_count(),

            "feature_names":
                self.get_feature_names(),

            "model_status":
                self.model_wrapper.get_status(),

            "last_error":
                self.last_error
        }

    # ======================================================================
    # CHECK MODEL
    # ======================================================================

    def is_ready(
        self
    ) -> bool:
        """
        Return True when a trained compatible ML model is ready.
        """

        return bool(
            self.is_model_loaded
        )

    # ======================================================================
    # CHECK INFERENCE AVAILABILITY
    # ======================================================================

    def can_predict(
        self
    ) -> bool:
        """
        Return True if either ML inference or heuristic inference
        is available.
        """

        return bool(
            self.is_model_loaded
            or
            self.allow_heuristic_fallback
        )