"""
agents/visual_agent/model.py

Production-grade XGBoost model wrapper for the Visual AI Agent.

Day 13 responsibilities
-----------------------

    Visual Feature Extractor
              |
              v
    VisualFeatureSchema
              |
              v
    VisualPreprocessor
              |
              v
    VisualXGBoostModel
              |
              +--> Training
              +--> Prediction
              +--> Probability prediction
              +--> Feature importance
              +--> Model persistence
              +--> Model validation
              +--> Model metadata
              |
              v
    Visual Risk / Explanation Layer


CLASSIFICATION CONTRACT
-----------------------

The currently trained Visual XGBoost model is a TWO-CLASS classifier:

    0 -> legitimate
    1 -> phishing

There is intentionally NO "suspicious" ML class.

If the wider project needs a "suspicious" risk category, that should be
derived by the risk/fusion layer from probability/risk scoring rather than
invented as a third ML class.


FEATURE CONTRACT
----------------

The model expects exactly the canonical 12 Visual features:

    1.  screenshot_width
    2.  screenshot_height
    3.  image_entropy
    4.  dominant_color_count
    5.  text_density
    6.  image_density
    7.  form_area_ratio
    8.  button_count
    9.  logo_detected
    10. login_form_detected
    11. layout_complexity
    12. blank_area_ratio


IMPORTANT
---------

This module contains the MACHINE LEARNING MODEL only.

Feature extraction belongs to the extractor.

Feature naming/order belongs to VisualFeatureSchema.

Feature preprocessing belongs to VisualPreprocessor.

Risk calculation belongs to risk_score.py.

Human-readable explanation belongs to explain.py.

This separation keeps the Visual Agent modular and Fusion-ready.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .feature_schema import VisualFeatureSchema


logger = logging.getLogger(__name__)


class VisualXGBoostModel:
    """
    Production wrapper around the Visual XGBoost classifier.

    Classification contract:

        0 = legitimate
        1 = phishing

    Feature contract:

        Exactly the 12 features supplied by VisualFeatureSchema.
    """

    # =========================================================================
    # 1. CLASSIFICATION CONTRACT
    # =========================================================================

    CLASS_LABELS: Dict[int, str] = {
        0: "legitimate",
        1: "phishing",
    }

    CLASS_NAMES: Tuple[str, ...] = (
        "legitimate",
        "phishing",
    )

    NUM_CLASSES: int = 2

    # =========================================================================
    # 2. MODEL VERSION
    # =========================================================================

    MODEL_NAME: str = "visual_xgboost_classifier"

    MODEL_VERSION: str = "1.1.0"

    # =========================================================================
    # 3. DEFAULT MODEL PATH
    # =========================================================================

    DEFAULT_MODEL_PATH: str = (
        "data/models/visual_agent/xgboost_visual_model.pkl"
    )

    # =========================================================================
    # 4. INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        hyperparameters: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> None:
        """
        Initialize the Visual XGBoost model wrapper.

        Parameters
        ----------
        model_path:
            Path to the trained XGBoost model.

        hyperparameters:
            Optional hyperparameter overrides used when building a NEW model.

        Notes
        -----
        The model is not automatically trained or loaded during initialization.

        Call:

            build_model()

        to create a new untrained model.

        Or:

            load_model()

        to load the existing trained model.
        """

        self.model_path: str = str(model_path)

        self.model: Optional[
            xgb.XGBClassifier
        ] = None

        # ---------------------------------------------------------------------
        # Canonical feature schema.
        # ---------------------------------------------------------------------

        self.feature_schema: List[str] = (
            VisualFeatureSchema.get_schema()
        )

        self.feature_count: int = len(
            self.feature_schema
        )

        # ---------------------------------------------------------------------
        # Validate the canonical schema immediately.
        # ---------------------------------------------------------------------

        VisualFeatureSchema.verify_schema_integrity()

        if self.feature_count != 12:

            raise ValueError(
                "Visual model requires exactly 12 canonical features. "
                f"Received {self.feature_count}."
            )

        # ---------------------------------------------------------------------
        # Binary XGBoost configuration.
        #
        # The trained production model is binary:
        #
        #     0 = legitimate
        #     1 = phishing
        #
        # `binary:logistic` produces a single positive-class probability.
        # The wrapper normalizes that output into a two-column probability
        # matrix:
        #
        #     [P(legitimate), P(phishing)]
        # ---------------------------------------------------------------------

        self.hyperparameters: Dict[str, Any] = {
            "n_estimators": 250,
            "max_depth": 7,
            "learning_rate": 0.05,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }

        # ---------------------------------------------------------------------
        # Apply optional overrides.
        # ---------------------------------------------------------------------

        if hyperparameters is not None:

            self.hyperparameters.update(
                dict(hyperparameters)
            )

        self._validate_hyperparameters()

        logger.info(
            "VisualXGBoostModel initialized. "
            "features=%d classes=%d model_path=%s",
            self.feature_count,
            self.NUM_CLASSES,
            self.model_path,
        )

    # =========================================================================
    # 5. BUILD MODEL
    # =========================================================================

    def build_model(
        self,
    ) -> xgb.XGBClassifier:
        """
        Build a fresh untrained binary XGBoost classifier.
        """

        logger.info(
            "Building new binary Visual XGBoost classifier."
        )

        self.model = xgb.XGBClassifier(
            **self.hyperparameters
        )

        logger.debug(
            "Visual XGBoost classifier built successfully."
        )

        return self.model

    # =========================================================================
    # 6. MODEL EXISTENCE CHECK
    # =========================================================================

    def is_built(self) -> bool:
        """
        Return True if an XGBoost model exists in memory.
        """

        return self.model is not None

    def is_fitted(self) -> bool:
        """
        Return True if the model appears to contain a trained booster.
        """

        if self.model is None:

            return False

        try:

            booster = self.model.get_booster()

            return booster is not None

        except Exception:

            return False

    # =========================================================================
    # 7. ENSURE MODEL EXISTS
    # =========================================================================

    def _ensure_model(
        self,
    ) -> xgb.XGBClassifier:
        """
        Return the current model or raise a clear error.
        """

        if self.model is None:

            raise RuntimeError(
                "Visual XGBoost model has not been initialized. "
                "Call build_model() or load_model() first."
            )

        return self.model

    # =========================================================================
    # 8. TRAINING
    # =========================================================================

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[Any],
        *,
        eval_set: Optional[
            Sequence[Tuple[pd.DataFrame, Sequence[Any]]]
        ] = None,
        verbose: bool = False,
    ) -> xgb.XGBClassifier:
        """
        Train the Visual XGBoost classifier.

        Expected labels:

            0 = legitimate
            1 = phishing

        X must already have been processed by VisualPreprocessor.
        """

        X_valid = self._validate_feature_matrix(
            X,
            allow_batch=True,
        )

        y_array = self._validate_labels(
            y,
            expected_rows=len(X_valid),
        )

        if self.model is None:

            self.build_model()

        model = self._ensure_model()

        validated_eval_set = None

        if eval_set is not None:

            validated_eval_set = []

            for X_eval, y_eval in eval_set:

                X_eval_valid = (
                    self._validate_feature_matrix(
                        X_eval,
                        allow_batch=True,
                    )
                )

                y_eval_array = (
                    self._validate_labels(
                        y_eval,
                        expected_rows=len(
                            X_eval_valid
                        ),
                    )
                )

                validated_eval_set.append(
                    (
                        X_eval_valid,
                        y_eval_array,
                    )
                )

        logger.info(
            "Training Visual XGBoost model. "
            "samples=%d features=%d classes=%d",
            len(X_valid),
            X_valid.shape[1],
            self.NUM_CLASSES,
        )

        try:

            model.fit(
                X_valid,
                y_array,
                eval_set=validated_eval_set,
                verbose=verbose,
            )

        except Exception as exc:

            logger.error(
                "Visual XGBoost training failed: %s",
                exc,
                exc_info=True,
            )

            raise

        logger.info(
            "Visual XGBoost model trained successfully."
        )

        return model

    # =========================================================================
    # 9. PREDICT CLASS
    # =========================================================================

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict the class for one or more samples.

        Returns integer class indices:

            0 = legitimate
            1 = phishing
        """

        model = self._ensure_fitted_model()

        X_valid = self._validate_feature_matrix(
            X,
            allow_batch=True,
        )

        try:

            predictions = model.predict(
                X_valid
            )

        except Exception as exc:

            logger.error(
                "Visual model prediction failed: %s",
                exc,
                exc_info=True,
            )

            raise

        predictions = np.asarray(
            predictions,
            dtype=int,
        )

        for prediction in predictions:

            self._validate_class_index(
                int(prediction)
            )

        return predictions

    # =========================================================================
    # 10. PREDICT SINGLE CLASS
    # =========================================================================

    def predict_single(
        self,
        X: pd.DataFrame,
    ) -> int:
        """
        Predict exactly one class.

        Returns:

            0 = legitimate
            1 = phishing
        """

        X_valid = self._validate_feature_matrix(
            X,
            allow_batch=False,
        )

        predictions = self.predict(
            X_valid
        )

        if len(predictions) != 1:

            raise ValueError(
                "predict_single() expected exactly one prediction."
            )

        prediction = int(
            predictions[0]
        )

        self._validate_class_index(
            prediction
        )

        return prediction

    # =========================================================================
    # 11. PREDICT PROBABILITIES
    # =========================================================================

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return two-column class probabilities.

        Output columns:

            column 0 -> legitimate
            column 1 -> phishing

        The underlying trained model may expose binary probabilities as a
        one-dimensional positive-class vector. This wrapper normalizes that
        output into the standard two-column format expected by the predictor.
        """

        model = self._ensure_fitted_model()

        X_valid = self._validate_feature_matrix(
            X,
            allow_batch=True,
        )

        try:

            probabilities = model.predict_proba(
                X_valid
            )

        except Exception as exc:

            logger.error(
                "Visual probability prediction failed: %s",
                exc,
                exc_info=True,
            )

            raise

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        # ---------------------------------------------------------------------
        # Standard binary XGBoost output:
        #
        #     shape = (samples, 2)
        #
        # Expected:
        #
        #     [P(legitimate), P(phishing)]
        #
        # Some binary model configurations can expose only the positive-class
        # probability. Normalize that representation if encountered.
        # ---------------------------------------------------------------------

        if probabilities.ndim == 1:

            phishing_probability = probabilities

            legitimate_probability = (
                1.0 - phishing_probability
            )

            probabilities = np.column_stack(
                (
                    legitimate_probability,
                    phishing_probability,
                )
            )

        elif (
            probabilities.ndim == 2
            and probabilities.shape[1] == 1
        ):

            phishing_probability = (
                probabilities[:, 0]
            )

            legitimate_probability = (
                1.0 - phishing_probability
            )

            probabilities = np.column_stack(
                (
                    legitimate_probability,
                    phishing_probability,
                )
            )

        self._validate_probability_matrix(
            probabilities
        )

        return probabilities

    # =========================================================================
    # 12. SINGLE SAMPLE PROBABILITIES
    # =========================================================================

    def predict_single_proba(
        self,
        X: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Return named probabilities for exactly one sample.
        """

        X_valid = self._validate_feature_matrix(
            X,
            allow_batch=False,
        )

        probabilities = self.predict_proba(
            X_valid
        )

        if probabilities.shape[0] != 1:

            raise ValueError(
                "predict_single_proba() expected exactly one row."
            )

        result: Dict[str, float] = {}

        for index, class_name in enumerate(
            self.CLASS_NAMES
        ):

            result[class_name] = float(
                probabilities[
                    0,
                    index,
                ]
            )

        return result

    # =========================================================================
    # 13. COMPLETE SINGLE PREDICTION RESULT
    # =========================================================================

    def predict_single_result(
        self,
        X: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Return raw ML information for one Visual prediction.

        Risk scoring is intentionally not performed here.
        """

        prediction = self.predict_single(
            X
        )

        probabilities = self.predict_single_proba(
            X
        )

        return {
            "predicted_class": prediction,
            "predicted_label": self.get_class_label(
                prediction
            ),
            "probabilities": probabilities,
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
            "num_classes": self.NUM_CLASSES,
            "feature_count": self.feature_count,
        }

    # =========================================================================
    # 14. CLASS LABEL
    # =========================================================================

    @classmethod
    def get_class_label(
        cls,
        class_index: int,
    ) -> str:
        """
        Convert a numeric class index into a human-readable label.
        """

        cls._validate_class_index(
            class_index
        )

        return cls.CLASS_LABELS[
            class_index
        ]

    @classmethod
    def get_class_index(
        cls,
        label: str,
    ) -> int:
        """
        Convert a class label into a numeric class index.

        Accepted labels:

            legitimate
            phishing
        """

        if not isinstance(
            label,
            str,
        ):

            raise TypeError(
                "Class label must be a string."
            )

        normalized = (
            label.strip()
            .lower()
        )

        for index, class_name in (
            cls.CLASS_LABELS.items()
        ):

            if normalized == class_name:

                return index

        raise ValueError(
            f"Unknown Visual class label: {label!r}. "
            f"Expected one of "
            f"{list(cls.CLASS_LABELS.values())}."
        )

    # =========================================================================
    # 15. FEATURE IMPORTANCE
    # =========================================================================

    def get_feature_importance(
        self,
        importance_type: str = "gain",
    ) -> Dict[str, float]:
        """
        Return XGBoost feature importance mapped to schema feature names.
        """

        model = self._ensure_fitted_model()

        supported_types = {
            "weight",
            "gain",
            "cover",
            "total_gain",
            "total_cover",
        }

        if importance_type not in supported_types:

            raise ValueError(
                "Unsupported importance_type. "
                "Expected one of: "
                "weight, gain, cover, total_gain, total_cover."
            )

        try:

            importance = (
                model.get_booster().get_score(
                    importance_type=importance_type
                )
            )

        except Exception as exc:

            logger.error(
                "Failed to retrieve Visual feature importance: %s",
                exc,
                exc_info=True,
            )

            raise

        result: Dict[str, float] = {
            feature: 0.0
            for feature in self.feature_schema
        }

        for feature_name, score in (
            importance.items()
        ):

            if feature_name in result:

                result[
                    feature_name
                ] = float(score)

        return result

    # =========================================================================
    # 16. SORTED FEATURE IMPORTANCE
    # =========================================================================

    def get_sorted_feature_importance(
        self,
        importance_type: str = "gain",
    ) -> List[Dict[str, Any]]:
        """
        Return feature importance sorted highest to lowest.
        """

        importance = self.get_feature_importance(
            importance_type=importance_type
        )

        sorted_items = sorted(
            importance.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            {
                "feature": feature,
                "importance": float(score),
            }
            for feature, score in sorted_items
        ]

    # =========================================================================
    # 17. MODEL SAVE
    # =========================================================================

    def save_model(
        self,
        custom_path: Optional[str] = None,
    ) -> bool:
        """
        Save the trained XGBoost model to disk using joblib.
        """

        model = self._ensure_fitted_model()

        save_path = Path(
            custom_path or self.model_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Saving Visual XGBoost model to: %s",
            save_path,
        )

        try:

            joblib.dump(
                model,
                save_path,
            )

        except Exception as exc:

            logger.error(
                "Failed to save Visual XGBoost model "
                "to %s: %s",
                save_path,
                exc,
                exc_info=True,
            )

            raise

        if not save_path.exists():

            raise IOError(
                "Model save operation completed without "
                f"creating the expected file: {save_path}"
            )

        logger.info(
            "Visual XGBoost model saved successfully. "
            "size=%d bytes",
            save_path.stat().st_size,
        )

        return True

    # =========================================================================
    # 18. MODEL LOAD
    # =========================================================================

    def load_model(
        self,
        custom_path: Optional[str] = None,
    ) -> xgb.XGBClassifier:
        """
        Load and strictly validate the trained Visual XGBoost model.
        """

        load_path = Path(
            custom_path or self.model_path
        )

        if not load_path.exists():

            raise FileNotFoundError(
                "Cannot load Visual XGBoost model. "
                f"File does not exist: {load_path}"
            )

        if not load_path.is_file():

            raise IOError(
                f"Visual model path is not a file: {load_path}"
            )

        logger.info(
            "Loading Visual XGBoost model from: %s",
            load_path,
        )

        try:

            loaded_model = joblib.load(
                load_path
            )

        except Exception as exc:

            logger.error(
                "Failed to deserialize Visual model "
                "from %s: %s",
                load_path,
                exc,
                exc_info=True,
            )

            raise

        if not isinstance(
            loaded_model,
            xgb.XGBClassifier,
        ):

            raise TypeError(
                "Loaded file is not a valid "
                "xgboost.XGBClassifier instance. "
                f"Received: {type(loaded_model).__name__}"
            )

        self._validate_loaded_model(
            loaded_model
        )

        self.model = loaded_model

        logger.info(
            "Visual XGBoost model loaded successfully. "
            "classes=%d features=%d",
            self.NUM_CLASSES,
            self.feature_count,
        )

        return self.model

    # =========================================================================
    # 19. MODEL METADATA
    # =========================================================================

    def get_model_metadata(
        self,
    ) -> Dict[str, Any]:
        """
        Return metadata describing the model and its feature contract.
        """

        objective = None

        if self.model is not None:

            try:

                objective = (
                    self.model.get_params().get(
                        "objective"
                    )
                )

            except Exception:

                objective = None

        if objective is None:

            objective = self.hyperparameters.get(
                "objective"
            )

        return {
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
            "model_type": "XGBClassifier",
            "objective": objective,
            "num_classes": self.NUM_CLASSES,
            "classes": dict(
                self.CLASS_LABELS
            ),
            "feature_count": self.feature_count,
            "features": list(
                self.feature_schema
            ),
            "model_path": self.model_path,
            "is_built": self.is_built(),
            "is_fitted": self.is_fitted(),
            "hyperparameters": dict(
                self.hyperparameters
            ),
        }

    # =========================================================================
    # 20. HYPERPARAMETERS
    # =========================================================================

    def get_hyperparameters(
        self,
    ) -> Dict[str, Any]:
        """
        Return a defensive copy of the configured hyperparameters.
        """

        return dict(
            self.hyperparameters
        )

    # =========================================================================
    # 21. FEATURE SCHEMA
    # =========================================================================

    def get_feature_schema(
        self,
    ) -> List[str]:
        """
        Return the exact feature schema expected by this model.
        """

        return list(
            self.feature_schema
        )

    # =========================================================================
    # 22. MODEL SUMMARY
    # =========================================================================

    def get_model_summary(
        self,
    ) -> Dict[str, Any]:
        """
        Return a concise model summary.
        """

        summary: Dict[str, Any] = {
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
            "model_type": "XGBoost Classifier",
            "classification_type": "binary",
            "class_mapping": dict(
                self.CLASS_LABELS
            ),
            "feature_count": self.feature_count,
            "features": list(
                self.feature_schema
            ),
            "built": self.is_built(),
            "fitted": self.is_fitted(),
        }

        if self.model is not None:

            try:

                params = self.model.get_params()

                summary[
                    "objective"
                ] = params.get(
                    "objective"
                )

                summary[
                    "n_estimators"
                ] = int(
                    params.get(
                        "n_estimators",
                        0,
                    )
                )

                summary[
                    "max_depth"
                ] = params.get(
                    "max_depth"
                )

                summary[
                    "learning_rate"
                ] = params.get(
                    "learning_rate"
                )

            except Exception:

                logger.debug(
                    "Unable to retrieve detailed model parameters.",
                    exc_info=True,
                )

        return summary

    # =========================================================================
    # 23. HYPERPARAMETER VALIDATION
    # =========================================================================

    def _validate_hyperparameters(
        self,
    ) -> None:
        """
        Validate critical XGBoost hyperparameters for binary classification.
        """

        objective = self.hyperparameters.get(
            "objective"
        )

        if objective not in {
            "binary:logistic",
            "binary:logitraw",
        }:

            raise ValueError(
                "Visual XGBoost model requires a binary objective. "
                "Expected 'binary:logistic' or 'binary:logitraw'. "
                f"Received: {objective!r}"
            )

        learning_rate = self.hyperparameters.get(
            "learning_rate"
        )

        if learning_rate is None or not (
            0.0 < float(learning_rate)
        ):

            raise ValueError(
                "learning_rate must be greater than 0."
            )

        n_estimators = self.hyperparameters.get(
            "n_estimators"
        )

        if n_estimators is None or int(
            n_estimators
        ) <= 0:

            raise ValueError(
                "n_estimators must be greater than 0."
            )

        max_depth = self.hyperparameters.get(
            "max_depth"
        )

        if max_depth is not None and int(
            max_depth
        ) <= 0:

            raise ValueError(
                "max_depth must be greater than 0."
            )

    # =========================================================================
    # 24. FEATURE MATRIX VALIDATION
    # =========================================================================

    def _validate_feature_matrix(
        self,
        X: pd.DataFrame,
        *,
        allow_batch: bool,
    ) -> pd.DataFrame:
        """
        Validate the feature matrix before sending it to XGBoost.

        Requirements:

            - pandas DataFrame
            - exactly 12 columns
            - exact canonical feature names
            - exact canonical feature order
            - numeric values
            - finite values
            - one row for single-sample operations
        """

        if not isinstance(
            X,
            pd.DataFrame,
        ):

            raise TypeError(
                "Visual XGBoost input X must be a pandas DataFrame. "
                f"Received: {type(X).__name__}"
            )

        if X.empty:

            raise ValueError(
                "Visual XGBoost input cannot be empty."
            )

        expected_columns = list(
            self.feature_schema
        )

        actual_columns = list(
            X.columns
        )

        # ---------------------------------------------------------------------
        # Feature count.
        # ---------------------------------------------------------------------

        if len(actual_columns) != self.feature_count:

            raise ValueError(
                "Visual model feature count mismatch. "
                f"Expected {self.feature_count}, "
                f"received {len(actual_columns)}."
            )

        # ---------------------------------------------------------------------
        # Duplicate columns.
        # ---------------------------------------------------------------------

        if len(
            actual_columns
        ) != len(
            set(actual_columns)
        ):

            duplicates = [
                column
                for column in dict.fromkeys(
                    actual_columns
                )
                if actual_columns.count(column) > 1
            ]

            raise ValueError(
                "Visual model input contains duplicate feature columns: "
                f"{duplicates}"
            )

        # ---------------------------------------------------------------------
        # Exact feature-name validation.
        # ---------------------------------------------------------------------

        missing = [
            feature
            for feature in expected_columns
            if feature not in actual_columns
        ]

        unexpected = [
            feature
            for feature in actual_columns
            if feature not in expected_columns
        ]

        if missing:

            raise ValueError(
                "Visual model input is missing required features: "
                f"{missing}"
            )

        if unexpected:

            raise ValueError(
                "Visual model input contains unexpected features: "
                f"{unexpected}"
            )

        # ---------------------------------------------------------------------
        # Exact feature-order validation.
        #
        # We intentionally do NOT silently reorder here.
        # The caller must provide the canonical order.
        # ---------------------------------------------------------------------

        if actual_columns != expected_columns:

            raise ValueError(
                "Visual model feature order mismatch. "
                f"Expected order: {expected_columns}; "
                f"Received order: {actual_columns}"
            )

        # ---------------------------------------------------------------------
        # Batch/single-row validation.
        # ---------------------------------------------------------------------

        if not allow_batch and len(X) != 1:

            raise ValueError(
                "Visual single-sample operation requires exactly one row. "
                f"Received {len(X)} rows."
            )

        # ---------------------------------------------------------------------
        # Convert every feature to numeric.
        # ---------------------------------------------------------------------

        validated = X[
            expected_columns
        ].copy()

        for feature in expected_columns:

            try:

                validated[
                    feature
                ] = pd.to_numeric(
                    validated[
                        feature
                    ],
                    errors="raise",
                )

            except Exception as exc:

                raise TypeError(
                    "Visual feature contains a non-numeric value. "
                    f"Feature={feature!r}"
                ) from exc

        # ---------------------------------------------------------------------
        # Reject NaN/infinite values.
        #
        # Missing values should be handled explicitly by the preprocessing
        # layer, not silently by this model wrapper.
        # ---------------------------------------------------------------------

        numeric_values = (
            validated.to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            numeric_values
        ).all():

            raise ValueError(
                "Visual model input contains NaN or infinite values. "
                "Preprocess the feature matrix before model inference."
            )

        # ---------------------------------------------------------------------
        # Return a defensive copy in exact schema order.
        # ---------------------------------------------------------------------

        return validated[
            expected_columns
        ].astype(
            "float64"
        ).copy()

    # =========================================================================
    # 25. LABEL VALIDATION
    # =========================================================================

    def _validate_labels(
        self,
        labels: Sequence[Any],
        *,
        expected_rows: int,
    ) -> np.ndarray:
        """
        Validate binary training/validation labels.

        Accepted labels:

            0 = legitimate
            1 = phishing
        """

        if labels is None:

            raise ValueError(
                "Training labels cannot be None."
            )

        y = np.asarray(
            labels
        )

        if y.ndim != 1:

            y = y.reshape(-1)

        if len(y) != expected_rows:

            raise ValueError(
                "Feature/label row count mismatch. "
                f"X has {expected_rows} rows, "
                f"y has {len(y)} labels."
            )

        if len(y) == 0:

            raise ValueError(
                "Training labels cannot be empty."
            )

        try:

            original_numeric = np.asarray(
                y,
                dtype=float,
            )

        except (
            ValueError,
            TypeError,
        ) as exc:

            raise TypeError(
                "Visual class labels must be numeric "
                "values 0 or 1."
            ) from exc

        if not np.isfinite(
            original_numeric
        ).all():

            raise ValueError(
                "Visual labels contain NaN or infinity."
            )

        y_numeric = original_numeric.astype(
            "int64"
        )

        if not np.allclose(
            original_numeric,
            y_numeric,
        ):

            raise ValueError(
                "Visual class labels must contain "
                "integer values 0 or 1."
            )

        invalid_labels = sorted(
            set(
                int(value)
                for value in y_numeric
                if int(value)
                not in self.CLASS_LABELS
            )
        )

        if invalid_labels:

            raise ValueError(
                "Invalid Visual class labels detected: "
                f"{invalid_labels}. "
                "Expected only 0 or 1."
            )

        return y_numeric

    # =========================================================================
    # 26. FITTED MODEL VALIDATION
    # =========================================================================

    def _ensure_fitted_model(
        self,
    ) -> xgb.XGBClassifier:
        """
        Ensure a trained model is available.
        """

        model = self._ensure_model()

        if not self.is_fitted():

            raise RuntimeError(
                "Visual XGBoost model exists but does not appear "
                "to be fitted. Train it with fit() or load a "
                "trained model using load_model()."
            )

        return model

    # =========================================================================
    # 27. CLASS INDEX VALIDATION
    # =========================================================================

    @classmethod
    def _validate_class_index(
        cls,
        class_index: int,
    ) -> None:
        """
        Validate a binary class index.
        """

        if class_index not in cls.CLASS_LABELS:

            raise ValueError(
                f"Invalid Visual class index: {class_index}. "
                f"Expected one of "
                f"{list(cls.CLASS_LABELS.keys())}."
            )

    # =========================================================================
    # 28. PROBABILITY MATRIX VALIDATION
    # =========================================================================

    def _validate_probability_matrix(
        self,
        probabilities: np.ndarray,
    ) -> None:
        """
        Validate normalized binary probability output.

        Expected shape:

            (number_of_samples, 2)

        Column mapping:

            0 -> legitimate
            1 -> phishing
        """

        if probabilities.ndim != 2:

            raise ValueError(
                "Visual model probability output must be 2-dimensional."
            )

        if probabilities.shape[1] != self.NUM_CLASSES:

            raise ValueError(
                "Visual model probability class count mismatch. "
                f"Expected {self.NUM_CLASSES}, "
                f"received {probabilities.shape[1]}."
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
                "Visual model returned probabilities outside "
                "the [0, 1] range."
            )

        row_sums = probabilities.sum(
            axis=1
        )

        if not np.allclose(
            row_sums,
            1.0,
            atol=1e-5,
        ):

            raise ValueError(
                "Visual model probability rows do not sum to 1."
            )

    # =========================================================================
    # 29. LOADED MODEL VALIDATION
    # =========================================================================

    def _validate_loaded_model(
        self,
        loaded_model: xgb.XGBClassifier,
    ) -> None:
        """
        Strictly validate a deserialized model against the actual Visual
        production contract.

        Validation includes:

            - XGBoost classifier type
            - fitted booster
            - binary class count
            - expected class labels
            - feature count
            - feature names
            - feature order
        """

        # ---------------------------------------------------------------------
        # Booster validation.
        # ---------------------------------------------------------------------

        try:

            booster = loaded_model.get_booster()

        except Exception as exc:

            raise ValueError(
                "Loaded Visual model does not contain a valid XGBoost booster."
            ) from exc

        if booster is None:

            raise ValueError(
                "Loaded Visual model does not contain "
                "a valid XGBoost booster."
            )

        # ---------------------------------------------------------------------
        # Class-count validation.
        # ---------------------------------------------------------------------

        try:

            model_classes = getattr(
                loaded_model,
                "classes_",
                None,
            )

            if model_classes is not None:

                model_classes_array = np.asarray(
                    model_classes
                ).reshape(-1)

                expected_classes = np.asarray(
                    [0, 1]
                )

                if not np.array_equal(
                    model_classes_array,
                    expected_classes,
                ):

                    raise ValueError(
                        "Loaded Visual model has incompatible classes_. "
                        f"Expected {expected_classes.tolist()}, "
                        f"received {model_classes_array.tolist()}."
                    )

        except ValueError:

            raise

        except Exception as exc:

            logger.warning(
                "Could not inspect loaded Visual model classes_: %s",
                exc,
            )

        # ---------------------------------------------------------------------
        # Number of features.
        # ---------------------------------------------------------------------

        try:

            n_features_in = getattr(
                loaded_model,
                "n_features_in_",
                None,
            )

            if n_features_in is not None:

                if int(
                    n_features_in
                ) != self.feature_count:

                    raise ValueError(
                        "Loaded Visual model has incompatible "
                        f"feature count. Expected {self.feature_count}, "
                        f"received {n_features_in}."
                    )

        except ValueError:

            raise

        except Exception as exc:

            logger.warning(
                "Could not inspect loaded Visual model feature count: %s",
                exc,
            )

        # ---------------------------------------------------------------------
        # Feature-name validation.
        #
        # XGBoost stores feature names on the booster when training was
        # performed with a named pandas DataFrame.
        # ---------------------------------------------------------------------

        try:

            model_feature_names = (
                booster.feature_names
            )

            if model_feature_names:

                model_feature_names = list(
                    model_feature_names
                )

                expected_feature_names = list(
                    self.feature_schema
                )

                if model_feature_names != (
                    expected_feature_names
                ):

                    raise ValueError(
                        "Loaded Visual model feature schema mismatch. "
                        f"Expected {expected_feature_names}, "
                        f"received {model_feature_names}."
                    )

        except ValueError:

            raise

        except Exception as exc:

            logger.warning(
                "Could not inspect loaded Visual model feature names: %s",
                exc,
            )

        # ---------------------------------------------------------------------
        # Objective validation.
        # ---------------------------------------------------------------------

        try:

            objective = loaded_model.get_params().get(
                "objective"
            )

            if objective not in {
                "binary:logistic",
                "binary:logitraw",
                None,
            }:

                raise ValueError(
                    "Loaded Visual model has incompatible objective. "
                    f"Received {objective!r}."
                )

        except ValueError:

            raise

        except Exception as exc:

            logger.warning(
                "Could not inspect loaded Visual model objective: %s",
                exc,
            )

        logger.info(
            "Loaded Visual model passed compatibility validation."
        )

    # =========================================================================
    # 30. MODEL FILE INFORMATION
    # =========================================================================

    def get_model_file_info(
        self,
        custom_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return information about a model file on disk.
        """

        path = Path(
            custom_path or self.model_path
        )

        exists = path.exists()

        result: Dict[str, Any] = {
            "path": str(path),
            "exists": exists,
            "is_file": (
                path.is_file()
                if exists
                else False
            ),
        }

        if exists and path.is_file():

            stat = path.stat()

            result[
                "size_bytes"
            ] = int(
                stat.st_size
            )

            result[
                "modified_timestamp"
            ] = float(
                stat.st_mtime
            )

        return result

    # =========================================================================
    # 31. MODEL HEALTH CHECK
    # =========================================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:
        """
        Return a structured health report for the Visual ML model.
        """

        status = "healthy"

        warnings: List[str] = []

        # ---------------------------------------------------------------------
        # Model state.
        # ---------------------------------------------------------------------

        if self.model is None:

            status = "not_ready"

            warnings.append(
                "No model is currently loaded or built."
            )

        elif not self.is_fitted():

            status = "not_ready"

            warnings.append(
                "Model exists but is not fitted."
            )

        # ---------------------------------------------------------------------
        # Feature schema.
        # ---------------------------------------------------------------------

        try:

            VisualFeatureSchema.verify_schema_integrity()

        except Exception as exc:

            status = "error"

            warnings.append(
                f"Feature schema validation failed: {exc}"
            )

        # ---------------------------------------------------------------------
        # Loaded-model compatibility.
        # ---------------------------------------------------------------------

        if self.model is not None:

            try:

                self._validate_loaded_model(
                    self.model
                )

            except Exception as exc:

                status = "error"

                warnings.append(
                    f"Loaded model validation failed: {exc}"
                )

        return {
            "status": status,
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
            "model_type": "XGBClassifier",
            "classification_type": "binary",
            "feature_count": self.feature_count,
            "num_classes": self.NUM_CLASSES,
            "classes": dict(
                self.CLASS_LABELS
            ),
            "features": list(
                self.feature_schema
            ),
            "is_built": self.is_built(),
            "is_fitted": self.is_fitted(),
            "model_path": self.model_path,
            "warnings": warnings,
        }

    # =========================================================================
    # 32. MODEL RELOAD
    # =========================================================================

    def reload_model(
        self,
        custom_path: Optional[str] = None,
    ) -> xgb.XGBClassifier:
        """
        Explicitly replace the current model with the serialized model.
        """

        logger.info(
            "Reloading Visual XGBoost model."
        )

        return self.load_model(
            custom_path=custom_path
        )
