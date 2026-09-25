"""
agents/visual_agent/predictor.py

Production inference engine for the Visual AI Agent.

Day 13 standardized version.

Architecture
------------

    Raw Visual Features
            |
            v
    VisualFeatureSchema
            |
            v
    VisualPreprocessor
            |
            v
    12-feature DataFrame
            |
            v
    VisualXGBoostModel
            |
       +----+----+
       |         |
       v         v
      ML      Heuristic
       |         |
       +----+----+
            |
            v
    Structured Prediction
            |
            v
       Visual AI Agent
            |
            v
       AgentResult

Classification Contract
------------------------

    0 = legitimate
    1 = phishing

There is NO "suspicious" ML class.

A suspicious/high-risk interpretation may be produced by the
risk-scoring or fusion layer, but the Visual ML classifier itself
only predicts legitimate or phishing.

Day 13 failure contract
-----------------------

Operational failure is NOT legitimate evidence.

If feature preparation, model inference, or another prediction
operation fails:

    class_label = "unknown"
    class_index = None
    confidence = 0.0
    phishing_probability = None
    legitimate_probability = None
    class_probabilities = {}
    model_status = "error"
    prediction_source = "unavailable"
    model_based = False
    fallback_used = False

If the trained model is unavailable and heuristic fallback is
explicitly enabled, the result is:

    model_status = "fallback"
    prediction_source = "fallback_heuristic"
    model_based = False
    fallback_used = True

The fallback is never represented as a trained ML prediction.

Responsibilities
----------------

This module performs:

    - feature validation
    - preprocessing orchestration
    - trained-model loading
    - ML inference
    - probability validation
    - controlled heuristic fallback
    - prediction metadata
    - model health information

This module does NOT perform:

    - screenshot extraction
    - OCR
    - object detection
    - website crawling
    - final project-wide fusion
    - final combined risk scoring
    - human-readable explanation generation

Those responsibilities belong to other components.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .feature_schema import VisualFeatureSchema
from .model import VisualXGBoostModel
from .preprocessing import VisualPreprocessor


logger = logging.getLogger(__name__)


class VisualPredictor:
    """
    Production inference engine for the Visual AI Agent.

    Classification:

        0 -> legitimate
        1 -> phishing

    The predictor always exposes a stable two-class probability
    structure when a valid prediction is available.
    """

    # =========================================================================
    # 1. CLASSIFICATION CONTRACT
    # =========================================================================

    CLASS_LABELS: Dict[int, str] = {
        0: "legitimate",
        1: "phishing",
    }

    NUM_CLASSES: int = 2

    UNKNOWN_LABEL: str = "unknown"

    # =========================================================================
    # 2. MODEL STATUS
    # =========================================================================

    MODEL_STATUS_ML: str = "available"

    MODEL_STATUS_HEURISTIC: str = "fallback"

    MODEL_STATUS_UNAVAILABLE: str = "unavailable"

    MODEL_STATUS_ERROR: str = "error"

    # =========================================================================
    # 3. PREDICTION SOURCE
    # =========================================================================

    PREDICTION_SOURCE_ML: str = "trained_ml"

    PREDICTION_SOURCE_HEURISTIC: str = (
        "fallback_heuristic"
    )

    PREDICTION_SOURCE_UNAVAILABLE: str = "unavailable"

    PREDICTION_SOURCE_ERROR: str = "unavailable"

    # =========================================================================
    # 4. DEFAULT MODEL PATH
    # =========================================================================

    DEFAULT_MODEL_PATH: str = (
        "data/models/visual_agent/"
        "xgboost_visual_model.pkl"
    )

    # =========================================================================
    # 5. DEFAULT TRAINING METRICS PATH
    # =========================================================================

    DEFAULT_METRICS_PATH: str = (
        "data/models/visual_agent/"
        "visual_training_metrics.json"
    )

    # =========================================================================
    # 6. DEFAULT TRAINING METADATA PATH
    # =========================================================================

    DEFAULT_METADATA_PATH: str = (
        "data/models/visual_agent/"
        "visual_training_metadata.json"
    )

    DEFAULT_CALIBRATION_PATH: str = (
        "data/models/visual_agent/"
        "visual_probability_calibration.json"
    )

    # =========================================================================
    # 7. INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        metrics_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        allow_heuristic_fallback: bool = True,
    ) -> None:
        """
        Initialize the Visual Predictor.

        Parameters
        ----------
        model_path:
            Optional path to the trained XGBoost model.

        metrics_path:
            Optional path to offline training metrics.

        metadata_path:
            Optional path to training metadata.

        allow_heuristic_fallback:
            Whether deterministic heuristic fallback is permitted
            when the trained ML model cannot be used.
        """

        self.model_path = str(
            model_path or self.DEFAULT_MODEL_PATH
        )

        self.metrics_path = str(
            metrics_path or self.DEFAULT_METRICS_PATH
        )

        self.metadata_path = str(
            metadata_path or self.DEFAULT_METADATA_PATH
        )

        self.calibration_path = str(
            calibration_path or self.DEFAULT_CALIBRATION_PATH
        )

        self.allow_heuristic_fallback = bool(
            allow_heuristic_fallback
        )

        # ---------------------------------------------------------------------
        # Preprocessor
        # ---------------------------------------------------------------------

        self.preprocessor = (
            VisualPreprocessor()
        )

        # ---------------------------------------------------------------------
        # Model wrapper
        # ---------------------------------------------------------------------

        self.model_wrapper = (
            VisualXGBoostModel(
                model_path=self.model_path
            )
        )

        self.model: Optional[Any] = None

        self.is_model_loaded: bool = False

        self.model_load_error: Optional[str] = None

        self.calibration: Optional[Dict[str, float]] = (
            self._load_calibration()
        )

        # ---------------------------------------------------------------------
        # Canonical feature schema
        # ---------------------------------------------------------------------

        VisualFeatureSchema.verify_schema_integrity()

        self.feature_schema: List[str] = (
            VisualFeatureSchema.get_schema()
        )

        self.feature_count: int = len(
            self.feature_schema
        )

        # ---------------------------------------------------------------------
        # Contract validation
        # ---------------------------------------------------------------------

        self._validate_class_contract()

        self._validate_feature_contract()

        # ---------------------------------------------------------------------
        # Load trained model
        # ---------------------------------------------------------------------

        self._load_active_model()

        logger.info(
            "VisualPredictor initialized. "
            "features=%d model_loaded=%s "
            "fallback=%s",
            self.feature_count,
            self.is_model_loaded,
            self.allow_heuristic_fallback,
        )

    # =========================================================================
    # 8. CLASS CONTRACT VALIDATION
    # =========================================================================

    def _load_calibration(self) -> Optional[Dict[str, float]]:
        """Load optional out-of-fold probability calibration parameters."""

        path = Path(self.calibration_path)
        if not path.is_absolute():
            project_path = Path(__file__).resolve().parents[2] / path
            if project_path.exists():
                path = project_path
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("method") != "sigmoid":
                raise ValueError("Unsupported calibration method.")
            a = float(payload["a"])
            b = float(payload["b"])
            if not math.isfinite(a) or not math.isfinite(b) or a <= 0:
                raise ValueError("Invalid calibration parameters.")
            return {"a": a, "b": b}
        except (OSError, TypeError, ValueError, KeyError) as exc:
            logger.warning("Visual calibration unavailable: %s", exc)
            return None

    def _calibrate_probability(self, probability: float) -> float:
        if self.calibration is None:
            return probability

        clipped = min(max(float(probability), 1e-6), 1.0 - 1e-6)
        logit = math.log(clipped / (1.0 - clipped))
        calibrated_logit = (
            self.calibration["a"] * logit
            + self.calibration["b"]
        )
        return 1.0 / (1.0 + math.exp(-calibrated_logit))

    def _validate_class_contract(
        self,
    ) -> None:
        """
        Validate the binary Visual classification contract.
        """

        expected_labels = {
            0: "legitimate",
            1: "phishing",
        }

        if self.CLASS_LABELS != expected_labels:

            raise ValueError(
                "Visual predictor class contract mismatch. "
                f"Expected {expected_labels}, "
                f"received {self.CLASS_LABELS}."
            )

        if self.NUM_CLASSES != 2:

            raise ValueError(
                "Visual predictor must use exactly 2 classes."
            )

    # =========================================================================
    # 9. FEATURE CONTRACT VALIDATION
    # =========================================================================

    def _validate_feature_contract(
        self,
    ) -> None:
        """
        Validate the authoritative 12-feature contract.
        """

        if self.feature_count != 12:

            raise ValueError(
                "Visual predictor requires exactly 12 features. "
                f"Received {self.feature_count}."
            )

        if len(
            set(self.feature_schema)
        ) != self.feature_count:

            raise ValueError(
                "Visual feature schema contains duplicate names."
            )

        VisualFeatureSchema.verify_schema_integrity()

    # =========================================================================
    # 10. LOAD ACTIVE MODEL
    # =========================================================================

    def _load_active_model(
        self,
    ) -> None:
        """
        Load the trained Visual XGBoost model.

        Failure is recorded explicitly.

        A missing/incompatible model does NOT become a legitimate
        prediction. It only enables fallback mode if fallback is allowed.
        """

        self.model = None

        self.is_model_loaded = False

        self.model_load_error = None

        try:

            self.model = (
                self.model_wrapper.load_model()
            )

            self.is_model_loaded = True

            logger.info(
                "Visual trained ML model loaded successfully: %s",
                self.model_path,
            )

        except FileNotFoundError as exc:

            self.model_load_error = str(
                exc
            )

            logger.warning(
                "Visual ML model file was not found: %s",
                self.model_path,
            )

        except Exception as exc:

            self.model_load_error = str(
                exc
            )

            logger.error(
                "Visual ML model could not be loaded: %s",
                exc,
                exc_info=True,
            )

    # =========================================================================
    # 11. RELOAD MODEL
    # =========================================================================

    def reload_model(
        self,
    ) -> bool:
        """
        Reload the trained model from disk.
        """

        logger.info(
            "Reloading Visual ML model."
        )

        self._load_active_model()

        return self.is_model_loaded

    # =========================================================================
    # 12. MODEL ACCESSOR
    # =========================================================================

    def get_model(
        self,
    ) -> Optional[Any]:
        """
        Return the loaded XGBoost model.
        """

        return self.model

    # =========================================================================
    # 13. MODEL WRAPPER ACCESSOR
    # =========================================================================

    def get_model_wrapper(
        self,
    ) -> VisualXGBoostModel:
        """
        Return the Visual XGBoost model wrapper.
        """

        return self.model_wrapper

    # =========================================================================
    # 14. FEATURE SCHEMA ACCESSOR
    # =========================================================================

    def get_feature_schema(
        self,
    ) -> List[str]:
        """
        Return the exact ordered feature schema.
        """

        return list(
            self.feature_schema
        )

    # =========================================================================
    # 15. MODEL STATUS
    # =========================================================================

    def get_model_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return structured model availability information.
        """

        if self.is_model_loaded:

            status = self.MODEL_STATUS_ML

        elif self.allow_heuristic_fallback:

            status = self.MODEL_STATUS_HEURISTIC

        else:

            status = self.MODEL_STATUS_UNAVAILABLE

        return {
            "status": status,
            "loaded": self.is_model_loaded,
            "model_path": self.model_path,
            "fallback_enabled": (
                self.allow_heuristic_fallback
            ),
            "load_error": self.model_load_error,
        }

    # =========================================================================
    # 16. TRAINING METRICS
    # =========================================================================

    def _get_training_metrics(
        self,
    ) -> Dict[str, Any]:
        """
        Load offline training metrics.

        These are model-evaluation metrics, NOT live prediction confidence.
        """

        path = Path(
            self.metrics_path
        )

        if not path.exists():

            return {
                "available": False,
                "notice": (
                    "Training metrics unavailable."
                ),
            }

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                metrics = json.load(
                    file
                )

            if isinstance(
                metrics,
                dict,
            ):

                return {
                    "available": True,
                    "data": metrics,
                }

        except Exception as exc:

            logger.warning(
                "Could not load Visual training metrics: %s",
                exc,
            )

        return {
            "available": False,
            "notice": (
                "Training metrics could not be loaded."
            ),
        }

    # =========================================================================
    # 17. TRAINING METADATA
    # =========================================================================

    def _get_training_metadata(
        self,
    ) -> Dict[str, Any]:
        """
        Load training metadata.
        """

        path = Path(
            self.metadata_path
        )

        if not path.exists():

            return {
                "available": False,
                "notice": (
                    "Training metadata unavailable."
                ),
            }

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                metadata = json.load(
                    file
                )

            if isinstance(
                metadata,
                dict,
            ):

                return {
                    "available": True,
                    "data": metadata,
                }

        except Exception as exc:

            logger.warning(
                "Could not load Visual training metadata: %s",
                exc,
            )

        return {
            "available": False,
            "notice": (
                "Training metadata could not be loaded."
            ),
        }

    # =========================================================================
    # 18. RAW FEATURE VALIDATION
    # =========================================================================

    def _validate_raw_features(
        self,
        raw_visual_features: Any,
    ) -> Dict[str, Any]:
        """
        Validate the raw feature payload.

        Missing canonical fields may be completed by the centralized
        VisualPreprocessor because the schema defines their controlled
        defaults.

        The payload itself must be a mapping.
        """

        if not isinstance(
            raw_visual_features,
            Mapping,
        ):

            raise TypeError(
                "Visual features must be supplied as a mapping/dictionary."
            )

        cleaned: Dict[str, Any] = dict(
            raw_visual_features
        )

        return cleaned

    # =========================================================================
    # 19. PREPROCESS FEATURES
    # =========================================================================

    def _prepare_features(
        self,
        raw_visual_features: Mapping[str, Any],
    ) -> pd.DataFrame:
        """
        Convert raw Visual features into the exact 12-feature ML matrix.
        """

        df_features = (
            self.preprocessor.transform_single(
                dict(raw_visual_features)
            )
        )

        if not isinstance(
            df_features,
            pd.DataFrame,
        ):

            raise TypeError(
                "VisualPreprocessor must return a pandas DataFrame."
            )

        # ---------------------------------------------------------------------
        # Exact feature count.
        # ---------------------------------------------------------------------

        if df_features.shape[1] != self.feature_count:

            raise ValueError(
                "Visual preprocessing produced an incorrect "
                f"number of features. Expected {self.feature_count}, "
                f"received {df_features.shape[1]}."
            )

        # ---------------------------------------------------------------------
        # Exact feature names and order.
        # ---------------------------------------------------------------------

        actual_columns = list(
            df_features.columns
        )

        if actual_columns != self.feature_schema:

            raise ValueError(
                "Visual preprocessing produced an incorrect "
                "feature schema/order.\n"
                f"Expected: {self.feature_schema}\n"
                f"Received: {actual_columns}"
            )

        # ---------------------------------------------------------------------
        # Exactly one sample.
        # ---------------------------------------------------------------------

        if len(df_features) != 1:

            raise ValueError(
                "Visual prediction requires exactly one feature row."
            )

        # ---------------------------------------------------------------------
        # Numeric conversion.
        # ---------------------------------------------------------------------

        try:

            numeric_matrix = (
                df_features.to_numpy(
                    dtype=float
                )
            )

        except Exception as exc:

            raise TypeError(
                "Visual feature matrix could not be converted "
                "to numerical values."
            ) from exc

        # ---------------------------------------------------------------------
        # Reject NaN/infinite values.
        # ---------------------------------------------------------------------

        if not np.isfinite(
            numeric_matrix
        ).all():

            raise ValueError(
                "Visual feature matrix contains NaN or infinite values."
            )

        return (
            df_features[
                self.feature_schema
            ]
            .astype("float64")
            .copy()
        )

    # =========================================================================
    # 20. FEATURE SNAPSHOT
    # =========================================================================

    def _feature_snapshot(
        self,
        df_features: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Convert one processed feature row into JSON-safe values.
        """

        if len(df_features) != 1:

            return {}

        row = df_features.iloc[0]

        return {
            feature: float(
                row[feature]
            )
            for feature in self.feature_schema
        }

    # =========================================================================
    # 21. PROBABILITY VALIDATION
    # =========================================================================

    def _validate_probabilities(
        self,
        probabilities: Any,
    ) -> np.ndarray:
        """
        Validate binary XGBoost probabilities.

        Expected:

            shape = (1, 2)

            column 0 = legitimate
            column 1 = phishing
        """

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        if probabilities.ndim != 2:

            raise ValueError(
                "Visual model probability output must be 2-dimensional."
            )

        if probabilities.shape != (
            1,
            self.NUM_CLASSES,
        ):

            raise ValueError(
                "Visual model returned an unexpected probability shape. "
                f"Expected (1, {self.NUM_CLASSES}), "
                f"received {probabilities.shape}."
            )

        if not np.isfinite(
            probabilities
        ).all():

            raise ValueError(
                "Visual model returned NaN or infinite probabilities."
            )

        if (
            probabilities < 0.0
        ).any() or (
            probabilities > 1.0
        ).any():

            raise ValueError(
                "Visual model returned probabilities outside [0, 1]."
            )

        row_sum = float(
            probabilities[0].sum()
        )

        if not math.isclose(
            row_sum,
            1.0,
            abs_tol=1e-5,
        ):

            raise ValueError(
                "Visual model probabilities do not sum to 1. "
                f"Received sum={row_sum:.8f}"
            )

        return probabilities

    # =========================================================================
    # 22. PROBABILITY DICTIONARY
    # =========================================================================

    def _probability_dictionary(
        self,
        probabilities: np.ndarray,
    ) -> Dict[str, float]:
        """
        Convert probability matrix into named class probabilities.
        """

        return {
            "legitimate": round(
                float(
                    probabilities[0, 0]
                ),
                6,
            ),
            "phishing": round(
                float(
                    probabilities[0, 1]
                ),
                6,
            ),
        }

    # =========================================================================
    # 23. NORMALIZE PROBABILITIES
    # =========================================================================

    @staticmethod
    def _normalize_probabilities(
        probabilities: Mapping[str, float],
    ) -> Dict[str, float]:
        """
        Normalize a two-class probability mapping.
        """

        legitimate = max(
            0.0,
            float(
                probabilities.get(
                    "legitimate",
                    0.0,
                )
            ),
        )

        phishing = max(
            0.0,
            float(
                probabilities.get(
                    "phishing",
                    0.0,
                )
            ),
        )

        total = (
            legitimate
            + phishing
        )

        if total <= 0.0:

            return {
                "legitimate": 0.5,
                "phishing": 0.5,
            }

        legitimate /= total

        phishing /= total

        result = {
            "legitimate": round(
                legitimate,
                6,
            ),
            "phishing": round(
                phishing,
                6,
            ),
        }

        difference = round(
            1.0
            - sum(
                result.values()
            ),
            6,
        )

        if difference != 0.0:

            largest_key = max(
                result,
                key=result.get,
            )

            result[
                largest_key
            ] = round(
                result[
                    largest_key
                ]
                + difference,
                6,
            )

        return result

    # =========================================================================
    # 24. BUILD ML RESULT
    # =========================================================================

    def _build_ml_result(
        self,
        df_features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Perform trained-ML inference.

        This method never invents a third class.
        """

        if not self.is_model_loaded:

            raise RuntimeError(
                "Visual trained ML model is not loaded."
            )

        if self.model is None:

            raise RuntimeError(
                "Visual trained ML model reference is None."
            )

        # ---------------------------------------------------------------------
        # Prediction probabilities
        # ---------------------------------------------------------------------

        probabilities = (
            self.model.predict_proba(
                df_features
            )
        )

        probabilities = (
            self._validate_probabilities(
                probabilities
            )
        )

        # ---------------------------------------------------------------------
        # Predicted class
        # ---------------------------------------------------------------------

        raw_phishing_probability = float(probabilities[0][1])
        phishing_probability = self._calibrate_probability(
            raw_phishing_probability
        )
        legitimate_probability = 1.0 - phishing_probability

        probabilities = np.array(
            [[legitimate_probability, phishing_probability]],
            dtype=float,
        )

        predicted_class = int(
            np.argmax(
                probabilities[0]
            )
        )

        if predicted_class not in self.CLASS_LABELS:

            raise ValueError(
                "Visual model returned an unsupported class index: "
                f"{predicted_class}"
            )

        class_label = (
            self.CLASS_LABELS[
                predicted_class
            ]
        )

        # ---------------------------------------------------------------------
        # Named probabilities
        # ---------------------------------------------------------------------

        class_probabilities = (
            self._probability_dictionary(
                probabilities
            )
        )

        confidence = float(
            class_probabilities[
                class_label
            ]
        )

        phishing_probability = float(class_probabilities["phishing"])
        legitimate_probability = float(class_probabilities["legitimate"])

        # ---------------------------------------------------------------------
        # Feature snapshot
        # ---------------------------------------------------------------------

        feature_values = (
            self._feature_snapshot(
                df_features
            )
        )

        return {
            # ---------------------------------------------------------------
            # Core prediction
            # ---------------------------------------------------------------
            "class_label": class_label,
            "class_index": predicted_class,

            # ---------------------------------------------------------------
            # Probability
            # ---------------------------------------------------------------
            "confidence": round(
                confidence,
                6,
            ),
            "phishing_probability": round(
                phishing_probability,
                6,
            ),
            "legitimate_probability": round(
                legitimate_probability,
                6,
            ),
            "class_probabilities": (
                class_probabilities
            ),

            # ---------------------------------------------------------------
            # Features
            # ---------------------------------------------------------------
            "features": feature_values,
            "features_dataframe": df_features,

            # ---------------------------------------------------------------
            # Day-13 model provenance
            # ---------------------------------------------------------------
            "model_status": self.MODEL_STATUS_ML,
            "prediction_source": (
                self.PREDICTION_SOURCE_ML
            ),
            "model_based": True,
            "fallback_used": False,

            # ---------------------------------------------------------------
            # Model information
            # ---------------------------------------------------------------
            "model_path": self.model_path,
            "model_name": (
                self.model_wrapper.MODEL_NAME
            ),
            "model_version": (
                self.model_wrapper.MODEL_VERSION
            ),

            # ---------------------------------------------------------------
            # Training artifacts
            # ---------------------------------------------------------------
            "training_evaluation": (
                self._get_training_metrics()
            ),
            "training_metadata": (
                self._get_training_metadata()
            ),
        }

    # =========================================================================
    # 25. HEURISTIC FALLBACK
    # =========================================================================

    def _heuristic_fallback_predict(
        self,
        df_features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Controlled deterministic heuristic fallback.

        IMPORTANT:

        This is NOT a trained ML prediction.

        The result explicitly states:

            model_status = fallback
            prediction_source = fallback_heuristic
            model_based = False
            fallback_used = True
        """

        features = (
            self._feature_snapshot(
                df_features
            )
        )

        # ---------------------------------------------------------------------
        # Extract normalized feature values.
        # ---------------------------------------------------------------------

        login_form = float(
            features.get(
                "login_form_detected",
                0.0,
            )
        )

        logo_detected = float(
            features.get(
                "logo_detected",
                0.0,
            )
        )

        form_area = float(
            features.get(
                "form_area_ratio",
                0.0,
            )
        )

        layout_complexity = float(
            features.get(
                "layout_complexity",
                0.0,
            )
        )

        entropy = float(
            features.get(
                "image_entropy",
                0.0,
            )
        )

        button_count = float(
            features.get(
                "button_count",
                0.0,
            )
        )

        image_density = float(
            features.get(
                "image_density",
                0.0,
            )
        )

        blank_area = float(
            features.get(
                "blank_area_ratio",
                0.0,
            )
        )

        text_density = float(
            features.get(
                "text_density",
                0.0,
            )
        )

        # ---------------------------------------------------------------------
        # Deterministic heuristic score.
        #
        # This score is deliberately kept separate from the project's final
        # risk score. It only estimates phishing probability for fallback
        # classification when the trained model is unavailable.
        # ---------------------------------------------------------------------

        phishing_probability = 0.0

        heuristic_reasons: List[str] = []

        if login_form >= 1.0:

            phishing_probability += 0.30

            heuristic_reasons.append(
                "A login/credential form was detected."
            )

        if login_form >= 1.0 and logo_detected >= 1.0:

            phishing_probability += 0.25

            heuristic_reasons.append(
                "A login form and logo-like element "
                "were detected together."
            )

        if form_area > 0.40:

            phishing_probability += 0.15

            heuristic_reasons.append(
                "A large portion of the page is occupied "
                "by form elements."
            )

        if layout_complexity > 75.0:

            phishing_probability += 0.10

            heuristic_reasons.append(
                "The visual layout has relatively high complexity."
            )

        if entropy > 7.5:

            phishing_probability += 0.05

            heuristic_reasons.append(
                "The screenshot contains relatively high "
                "visual entropy."
            )

        if button_count > 20:

            phishing_probability += 0.05

            heuristic_reasons.append(
                "A relatively large number of buttons were detected."
            )

        if image_density > 0.70:

            phishing_probability += 0.03

        if blank_area > 0.70:

            phishing_probability += 0.02

        if (
            image_density > 0.60
            and text_density < 0.15
        ):

            phishing_probability += 0.03

        phishing_probability = float(
            np.clip(
                phishing_probability,
                0.01,
                0.99,
            )
        )

        legitimate_probability = (
            1.0
            - phishing_probability
        )

        # ---------------------------------------------------------------------
        # Binary classification.
        # ---------------------------------------------------------------------

        if phishing_probability >= 0.50:

            predicted_class = 1

            class_label = "phishing"

        else:

            predicted_class = 0

            class_label = "legitimate"

        class_probabilities = (
            self._normalize_probabilities(
                {
                    "legitimate": legitimate_probability,
                    "phishing": phishing_probability,
                }
            )
        )

        confidence = float(
            class_probabilities[
                class_label
            ]
        )

        phishing_probability = float(
            class_probabilities[
                "phishing"
            ]
        )

        legitimate_probability = float(
            class_probabilities[
                "legitimate"
            ]
        )

        return {
            # -----------------------------------------------------------------
            # Prediction
            # -----------------------------------------------------------------
            "class_label": class_label,
            "class_index": predicted_class,

            # -----------------------------------------------------------------
            # Probability
            # -----------------------------------------------------------------
            "confidence": round(
                confidence,
                6,
            ),
            "phishing_probability": round(
                phishing_probability,
                6,
            ),
            "legitimate_probability": round(
                legitimate_probability,
                6,
            ),
            "class_probabilities": (
                class_probabilities
            ),

            # -----------------------------------------------------------------
            # Features
            # -----------------------------------------------------------------
            "features": features,
            "features_dataframe": df_features,

            # -----------------------------------------------------------------
            # Provenance
            # -----------------------------------------------------------------
            "model_status": (
                self.MODEL_STATUS_HEURISTIC
            ),
            "prediction_source": (
                self.PREDICTION_SOURCE_HEURISTIC
            ),
            "model_based": False,
            "fallback_used": True,

            # -----------------------------------------------------------------
            # Model information
            # -----------------------------------------------------------------
            "model_path": self.model_path,
            "model_name": (
                self.model_wrapper.MODEL_NAME
            ),
            "model_version": (
                self.model_wrapper.MODEL_VERSION
            ),

            # -----------------------------------------------------------------
            # Explainability support
            # -----------------------------------------------------------------
            "heuristic_reasons": (
                heuristic_reasons
            ),

            # -----------------------------------------------------------------
            # Training artifacts
            # -----------------------------------------------------------------
            "training_evaluation": (
                self._get_training_metrics()
            ),
            "training_metadata": (
                self._get_training_metadata()
            ),
        }

    # =========================================================================
    # 26. ERROR RESULT
    # =========================================================================

    def _build_error_result(
        self,
        error: Exception,
        df_features: Optional[
            pd.DataFrame
        ] = None,
    ) -> Dict[str, Any]:
        """
        Build a truthful error result.

        IMPORTANT:

        An inference error is NEVER represented as:

            suspicious
            0.50 probability
            class_index=1

        Instead it becomes:

            unknown
            no probability
            no class index
            error status
            unavailable prediction source
        """

        logger.error(
            "Visual inference failed: %s",
            error,
            exc_info=True,
        )

        if df_features is None:

            df_features = pd.DataFrame(
                columns=self.feature_schema
            )

        feature_values: Dict[str, float] = {}

        if (
            isinstance(
                df_features,
                pd.DataFrame,
            )
            and len(df_features) == 1
        ):

            try:

                feature_values = (
                    self._feature_snapshot(
                        df_features
                    )
                )

            except Exception:

                feature_values = {}

        return {
            # -----------------------------------------------------------------
            # Truthful failure state
            # -----------------------------------------------------------------
            "class_label": self.UNKNOWN_LABEL,
            "class_index": None,

            # -----------------------------------------------------------------
            # No probability is invented
            # -----------------------------------------------------------------
            "confidence": 0.0,
            "phishing_probability": None,
            "legitimate_probability": None,
            "class_probabilities": {},

            # -----------------------------------------------------------------
            # Features
            # -----------------------------------------------------------------
            "features": feature_values,
            "features_dataframe": df_features,

            # -----------------------------------------------------------------
            # Day-13 provenance
            # -----------------------------------------------------------------
            "model_status": self.MODEL_STATUS_ERROR,
            "prediction_source": (
                self.PREDICTION_SOURCE_ERROR
            ),
            "model_based": False,
            "fallback_used": False,

            # -----------------------------------------------------------------
            # Error
            # -----------------------------------------------------------------
            "error": str(error),
            "error_type": type(error).__name__,

            # -----------------------------------------------------------------
            # Model information
            # -----------------------------------------------------------------
            "model_path": self.model_path,
            "model_name": (
                self.model_wrapper.MODEL_NAME
            ),
            "model_version": (
                self.model_wrapper.MODEL_VERSION
            ),

            # -----------------------------------------------------------------
            # Training artifacts
            # -----------------------------------------------------------------
            "training_evaluation": (
                self._get_training_metrics()
            ),
            "training_metadata": (
                self._get_training_metadata()
            ),
        }

    # =========================================================================
    # 27. MAIN PREDICTION
    # =========================================================================

    def predict(
        self,
        raw_visual_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute one complete Visual prediction.

        Processing:

            raw features
                ↓
            validation
                ↓
            preprocessing
                ↓
            trained ML inference
                ↓
            controlled heuristic fallback
                ↓
            structured result

        A prediction failure returns an explicit unknown/error result.
        """

        df_features: Optional[
            pd.DataFrame
        ] = None

        try:

            # -----------------------------------------------------------------
            # Step 1 — Validate
            # -----------------------------------------------------------------

            validated_features = (
                self._validate_raw_features(
                    raw_visual_features
                )
            )

            # -----------------------------------------------------------------
            # Step 2 — Preprocess
            # -----------------------------------------------------------------

            df_features = (
                self._prepare_features(
                    validated_features
                )
            )

            # -----------------------------------------------------------------
            # Step 3 — Trained ML
            # -----------------------------------------------------------------

            if (
                self.is_model_loaded
                and self.model is not None
            ):

                return self._build_ml_result(
                    df_features
                )

            # -----------------------------------------------------------------
            # Step 4 — Controlled heuristic fallback
            # -----------------------------------------------------------------

            if self.allow_heuristic_fallback:

                return (
                    self._heuristic_fallback_predict(
                        df_features
                    )
                )

            # -----------------------------------------------------------------
            # Step 5 — No model, no fallback
            # -----------------------------------------------------------------

            raise RuntimeError(
                "Visual trained ML model is unavailable "
                "and heuristic fallback is disabled."
            )

        except Exception as exc:

            return self._build_error_result(
                exc,
                df_features,
            )

    # =========================================================================
    # 28. STRICT PREDICTION
    # =========================================================================

    def predict_strict(
        self,
        raw_visual_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute prediction using ONLY the trained ML model.

        Unlike predict(), this method does not silently fall back
        to heuristics.

        Exceptions are propagated to the caller.
        """

        if not self.is_model_loaded:

            raise RuntimeError(
                "Visual trained ML model is not loaded. "
                "Strict prediction requires the trained model."
            )

        if self.model is None:

            raise RuntimeError(
                "Visual trained ML model reference is None."
            )

        validated_features = (
            self._validate_raw_features(
                raw_visual_features
            )
        )

        df_features = (
            self._prepare_features(
                validated_features
            )
        )

        return self._build_ml_result(
            df_features
        )

    # =========================================================================
    # 29. PREDICT CLASS ONLY
    # =========================================================================

    def predict_class(
        self,
        raw_visual_features: Dict[str, Any],
    ) -> Optional[int]:
        """
        Return the predicted numeric class.

        Returns:

            0 = legitimate
            1 = phishing
            None = unavailable/error
        """

        result = self.predict(
            raw_visual_features
        )

        value = result.get(
            "class_index"
        )

        if value is None:

            return None

        return int(value)

    # =========================================================================
    # 30. PREDICT LABEL ONLY
    # =========================================================================

    def predict_label(
        self,
        raw_visual_features: Dict[str, Any],
    ) -> str:
        """
        Return the predicted human-readable label.

        Possible values:

            legitimate
            phishing
            unknown
        """

        result = self.predict(
            raw_visual_features
        )

        return str(
            result.get(
                "class_label",
                self.UNKNOWN_LABEL,
            )
        )

    # =========================================================================
    # 31. PHISHING PROBABILITY ONLY
    # =========================================================================

    def get_phishing_probability(
        self,
        raw_visual_features: Dict[str, Any],
    ) -> Optional[float]:
        """
        Return phishing probability.

        Returns None when the prediction is unavailable.
        """

        result = self.predict(
            raw_visual_features
        )

        value = result.get(
            "phishing_probability"
        )

        if value is None:

            return None

        return float(value)

    # =========================================================================
    # 32. CLASS PROBABILITIES ONLY
    # =========================================================================

    def get_class_probabilities(
        self,
        raw_visual_features: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Return the two-class probability mapping.

        Error/unavailable result returns an empty dictionary.
        """

        result = self.predict(
            raw_visual_features
        )

        probabilities = result.get(
            "class_probabilities",
            {},
        )

        if not isinstance(
            probabilities,
            Mapping,
        ):

            return {}

        return {
            str(key): float(value)
            for key, value in probabilities.items()
            if value is not None
        }

    # =========================================================================
    # 33. MODEL HEALTH CHECK
    # =========================================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:
        """
        Return the current Visual predictor health state.
        """

        schema_status = "healthy"

        try:

            VisualFeatureSchema.verify_schema_integrity()

        except Exception as exc:

            schema_status = (
                f"error: {exc}"
            )

        # ---------------------------------------------------------------------
        # Model state
        # ---------------------------------------------------------------------

        if self.is_model_loaded:

            model_status = "ready"

        elif self.allow_heuristic_fallback:

            model_status = "fallback_mode"

        else:

            model_status = "unavailable"

        overall_status = (
            "healthy"
            if (
                schema_status == "healthy"
                and model_status in {
                    "ready",
                    "fallback_mode",
                }
            )
            else "error"
        )

        return {
            "status": overall_status,

            "model_status": model_status,

            "model_loaded": (
                self.is_model_loaded
            ),

            "model_path": self.model_path,

            "fallback_enabled": (
                self.allow_heuristic_fallback
            ),

            "model_load_error": (
                self.model_load_error
            ),

            "feature_count": (
                self.feature_count
            ),

            "features": list(
                self.feature_schema
            ),

            "class_mapping": dict(
                self.CLASS_LABELS
            ),

            "num_classes": self.NUM_CLASSES,

            "schema_status": schema_status,
        }

    # =========================================================================
    # 34. PREDICTOR INFORMATION
    # =========================================================================

    def get_predictor_info(
        self,
    ) -> Dict[str, Any]:
        """
        Return complete predictor configuration information.
        """

        return {
            "component": "VisualPredictor",
            "version": "1.2.0",

            "classification": {
                "type": "binary",
                "num_classes": self.NUM_CLASSES,
                "class_mapping": dict(
                    self.CLASS_LABELS
                ),
            },

            "model": {
                "name": (
                    self.model_wrapper.MODEL_NAME
                ),
                "version": (
                    self.model_wrapper.MODEL_VERSION
                ),
                "path": self.model_path,
                "loaded": self.is_model_loaded,
                "load_error": (
                    self.model_load_error
                ),
            },

            "features": {
                "count": self.feature_count,
                "schema": list(
                    self.feature_schema
                ),
            },

            "fallback": {
                "enabled": (
                    self.allow_heuristic_fallback
                ),
                "prediction_source": (
                    self.PREDICTION_SOURCE_HEURISTIC
                ),
            },

            "training_metrics_path": (
                self.metrics_path
            ),

            "training_metadata_path": (
                self.metadata_path
            ),
        }

    # =========================================================================
    # 35. FEATURE PREVIEW
    # =========================================================================

    def preprocess_for_inspection(
        self,
        raw_visual_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Preprocess raw features without performing prediction.

        Useful for validating the extractor -> schema -> preprocessor
        pipeline before ML inference.
        """

        validated_features = (
            self._validate_raw_features(
                raw_visual_features
            )
        )

        df_features = (
            self._prepare_features(
                validated_features
            )
        )

        return {
            "feature_count": self.feature_count,

            "feature_order": list(
                df_features.columns
            ),

            "features": (
                self._feature_snapshot(
                    df_features
                )
            ),

            "features_dataframe": df_features,
        }

    # =========================================================================
    # 36. BATCH PREDICTION
    # =========================================================================

    def predict_batch(
        self,
        raw_visual_features_list: List[
            Dict[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Predict multiple Visual feature dictionaries.

        Each item is processed independently.
        """

        if not isinstance(
            raw_visual_features_list,
            list,
        ):

            raise TypeError(
                "predict_batch() expects a list of feature dictionaries."
            )

        results: List[
            Dict[str, Any]
        ] = []

        for index, features in enumerate(
            raw_visual_features_list
        ):

            logger.debug(
                "Processing Visual batch item %d.",
                index,
            )

            results.append(
                self.predict(
                    features
                )
            )

        return results