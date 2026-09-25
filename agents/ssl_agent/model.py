"""
SSL/TLS Security AI Agent - Model
=================================

Production-grade XGBoost model wrapper for the SSL/TLS Security AI Agent.

Responsibilities
----------------
    • XGBoost model construction
    • Model training
    • Validation
    • Probability prediction
    • Class prediction
    • Model persistence
    • Model loading
    • Feature-order validation
    • Feature importance extraction
    • Training metadata
    • Model health/status information

Classification
--------------
    0 = legitimate / lower SSL security concern
    1 = suspicious / malicious / higher SSL security concern

IMPORTANT
---------
This module does NOT perform raw SSL/TLS feature extraction.

The expected pipeline is:

    SSL/TLS Extractor
            ↓
    SSLFeatureSchema
            ↓
    SSLPreprocessor
            ↓
    SSLModel
            ↓
    SSLPredictor
            ↓
    SSLRiskScorer
            ↓
    SSLExplainer
            ↓
    SSL AI Agent
            ↓
    Decision Fusion Engine


Feature Contract
----------------
The model expects the exact 17-feature schema defined in:

    agents/ssl_agent/feature_schema.py

Training and inference MUST use the same:

    • feature names
    • feature count
    • feature order
    • numerical representation
"""

import os

import logging

from datetime import (
    datetime,
    timezone
)

from typing import (
    Dict,
    Any,
    Optional,
    List,
    Tuple
)

import joblib

import numpy as np

import pandas as pd

import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from .feature_schema import SSLFeatureSchema


logger = logging.getLogger(__name__)


class SSLModel:
    """
    Production-grade XGBoost model wrapper for SSL/TLS analysis.

    The class provides a stable interface between the preprocessing,
    training, prediction, and explainability layers.

    Model architecture:

        Input:
            17 SSL/TLS features

        Algorithm:
            XGBoost Binary Classifier

        Output:
            P(class 0)
            P(class 1)

        Class mapping:

            0 → legitimate
            1 → suspicious / malicious
    """

    # ======================================================================
    # MODEL METADATA
    # ======================================================================

    MODEL_NAME = (
        "SSL_XGBoost_Model"
    )

    MODEL_VERSION = "1.0.0"

    MODEL_ALGORITHM = (
        "XGBoost"
    )

    MODEL_TYPE = (
        "binary_classification"
    )

    # ======================================================================
    # CLASS MAPPING
    # ======================================================================

    CLASS_LABELS: Dict[int, str] = {

        0:
            "legitimate",

        1:
            "suspicious"
    }

    # ======================================================================
    # DEFAULT ARTIFACT
    # ======================================================================

    DEFAULT_MODEL_FILENAME = (
        "ssl_xgboost_model.joblib"
    )

    # ======================================================================
    # DEFAULT XGBOOST PARAMETERS
    # ======================================================================

    DEFAULT_PARAMS: Dict[str, Any] = {

        "n_estimators":
            200,

        "max_depth":
            5,

        "learning_rate":
            0.05,

        "subsample":
            0.8,

        "colsample_bytree":
            0.8,

        "objective":
            "binary:logistic",

        "eval_metric":
            "logloss",

        "random_state":
            42,

        "n_jobs":
            -1
    }

    # ======================================================================
    # INITIALIZATION
    # ======================================================================

    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize SSL model wrapper.

        Args:
            model_dir:
                Directory used for model artifacts.

            model_path:
                Optional explicit model artifact path.

        Priority:

            model_path
                ↓
            model_dir
                ↓
            agents/ssl_agent/artifacts/
        """

        # ------------------------------------------------------------------
        # Resolve default model directory
        # ------------------------------------------------------------------

        default_model_dir = os.path.join(

            os.path.dirname(
                os.path.abspath(__file__)
            ),

            "artifacts"
        )

        self.model_dir = (
            model_dir
            or
            default_model_dir
        )

        # ------------------------------------------------------------------
        # Create artifact directory
        # ------------------------------------------------------------------

        os.makedirs(
            self.model_dir,
            exist_ok=True
        )

        # ------------------------------------------------------------------
        # Resolve model path
        # ------------------------------------------------------------------

        if model_path:

            self.model_path = (
                model_path
            )

            # Ensure parent directory exists
            model_parent = os.path.dirname(
                os.path.abspath(
                    self.model_path
                )
            )

            os.makedirs(
                model_parent,
                exist_ok=True
            )

        else:

            self.model_path = os.path.join(

                self.model_dir,

                self.DEFAULT_MODEL_FILENAME
            )

        # ------------------------------------------------------------------
        # Model object
        # ------------------------------------------------------------------

        self.model: Optional[
            xgb.XGBClassifier
        ] = None

        # ------------------------------------------------------------------
        # Feature schema
        # ------------------------------------------------------------------

        self.feature_names: List[str] = (
            SSLFeatureSchema.get_feature_names()
        )

        # ------------------------------------------------------------------
        # Training state
        # ------------------------------------------------------------------

        self.is_trained: bool = False

        # ------------------------------------------------------------------
        # Training metadata
        # ------------------------------------------------------------------

        self.training_metrics: Dict[
            str,
            Any
        ] = {}

        self.model_params: Dict[
            str,
            Any
        ] = {}

        self.created_at: Optional[str] = None

        self.loaded_from: Optional[str] = None

        # ------------------------------------------------------------------
        # Last error
        # ------------------------------------------------------------------

        self.last_error: Optional[str] = None

        logger.info(
            "%s initialized. Features=%d ModelPath=%s",
            self.MODEL_NAME,
            len(self.feature_names),
            self.model_path
        )

    # ======================================================================
    # BUILD MODEL
    # ======================================================================

    def build_model(
        self,
        custom_params: Optional[
            Dict[str, Any]
        ] = None
    ) -> xgb.XGBClassifier:
        """
        Build a new XGBoost classifier.

        Args:
            custom_params:
                Optional XGBoost parameter overrides.

        Returns:
            xgb.XGBClassifier:
                Initialized XGBoost classifier.
        """

        # ------------------------------------------------------------------
        # Copy defaults so the class-level dictionary is never mutated.
        # ------------------------------------------------------------------

        params = (
            self.DEFAULT_PARAMS.copy()
        )

        # ------------------------------------------------------------------
        # Apply custom parameters
        # ------------------------------------------------------------------

        if custom_params:

            if not isinstance(
                custom_params,
                dict
            ):

                raise TypeError(
                    "custom_params must be a dictionary."
                )

            params.update(
                custom_params
            )

        # ------------------------------------------------------------------
        # Force binary objective
        # ------------------------------------------------------------------

        params[
            "objective"
        ] = "binary:logistic"

        # ------------------------------------------------------------------
        # Create model
        # ------------------------------------------------------------------

        self.model = (
            xgb.XGBClassifier(
                **params
            )
        )

        self.model_params = (
            params.copy()
        )

        self.is_trained = False

        logger.info(
            "SSL XGBoost model built successfully."
        )

        return self.model

    # ======================================================================
    # FEATURE MATRIX VALIDATION
    # ======================================================================

    def _validate_feature_matrix(
        self,
        X: pd.DataFrame,
        require_exact_columns: bool = True
    ) -> pd.DataFrame:
        """
        Validate and normalize an input feature matrix.

        Args:
            X:
                Feature DataFrame.

            require_exact_columns:
                If True, feature names and order must exactly match
                SSLFeatureSchema.

        Returns:
            Validated DataFrame.

        Raises:
            TypeError:
                If X is not a DataFrame.

            ValueError:
                If feature structure is invalid.
        """

        if not isinstance(
            X,
            pd.DataFrame
        ):

            raise TypeError(
                (
                    "SSLModel expects X to be a "
                    "pandas DataFrame."
                )
            )

        if X.empty:

            raise ValueError(
                "SSLModel received an empty feature DataFrame."
            )

        expected_features = (
            self.feature_names
        )

        actual_features = list(
            X.columns
        )

        # ------------------------------------------------------------------
        # Exact schema check
        # ------------------------------------------------------------------

        if require_exact_columns:

            if actual_features != expected_features:

                missing = [

                    feature

                    for feature
                    in expected_features

                    if feature not in actual_features
                ]

                extra = [

                    feature

                    for feature
                    in actual_features

                    if feature not in expected_features
                ]

                raise ValueError(
                    (
                        "SSL feature schema mismatch.\n"
                        f"Expected order: {expected_features}\n"
                        f"Received order: {actual_features}\n"
                        f"Missing: {missing}\n"
                        f"Extra: {extra}"
                    )
                )

        else:

            # ----------------------------------------------------------------
            # Flexible mode: reorder columns if all required columns exist.
            # ----------------------------------------------------------------

            missing = [

                feature

                for feature
                in expected_features

                if feature not in actual_features
            ]

            if missing:

                raise ValueError(
                    (
                        "Missing required SSL features: "
                        f"{missing}"
                    )
                )

            X = X[
                expected_features
            ]

        # ------------------------------------------------------------------
        # Numerical validation
        # ------------------------------------------------------------------

        for feature_name in (
            expected_features
        ):

            if not pd.api.types.is_numeric_dtype(
                X[
                    feature_name
                ]
            ):

                raise ValueError(
                    (
                        f"SSL feature '{feature_name}' "
                        "must be numerical."
                    )
                )

        # ------------------------------------------------------------------
        # NaN / infinity validation
        # ------------------------------------------------------------------

        numeric_array = X[
            expected_features
        ].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(
            numeric_array
        ).all():

            raise ValueError(
                (
                    "SSL feature matrix contains "
                    "NaN or infinite values."
                )
            )

        # ------------------------------------------------------------------
        # Return deterministic order
        # ------------------------------------------------------------------

        return X[
            expected_features
        ].copy()

    # ======================================================================
    # TARGET VALIDATION
    # ======================================================================

    @staticmethod
    def _validate_target(
        y: pd.Series,
        expected_length: int
    ) -> pd.Series:
        """
        Validate binary target labels.

        Allowed labels:

            0 = legitimate
            1 = suspicious
        """

        if not isinstance(
            y,
            (
                pd.Series,
                pd.DataFrame
            )
        ):

            raise TypeError(
                "Target y must be a pandas Series."
            )

        if isinstance(
            y,
            pd.DataFrame
        ):

            if y.shape[1] != 1:

                raise ValueError(
                    "Target DataFrame must contain exactly one column."
                )

            y = y.iloc[
                :,
                0
            ]

        y = y.copy()

        # ------------------------------------------------------------------
        # Length
        # ------------------------------------------------------------------

        if len(y) != expected_length:

            raise ValueError(
                (
                    "Feature/target length mismatch. "
                    f"X={expected_length}, "
                    f"y={len(y)}"
                )
            )

        # ------------------------------------------------------------------
        # Numeric conversion
        # ------------------------------------------------------------------

        y = pd.to_numeric(
            y,
            errors="coerce"
        )

        if y.isna().any():

            raise ValueError(
                "Target contains missing or non-numeric labels."
            )

        # ------------------------------------------------------------------
        # Integer conversion
        # ------------------------------------------------------------------

        y = y.astype(
            int
        )

        # ------------------------------------------------------------------
        # Binary validation
        # ------------------------------------------------------------------

        unique_values = set(
            y.unique().tolist()
        )

        invalid_labels = (
            unique_values
            -
            {0, 1}
        )

        if invalid_labels:

            raise ValueError(
                (
                    "SSL target contains invalid labels: "
                    f"{sorted(invalid_labels)}. "
                    "Expected only 0 and 1."
                )
            )

        return y

    # ======================================================================
    # CLASS WEIGHT
    # ======================================================================

    @staticmethod
    def _calculate_scale_pos_weight(
        y: pd.Series
    ) -> float:
        """
        Calculate scale_pos_weight for imbalanced binary classes.

        Formula:

            negative_samples / positive_samples

        Returns:
            float
        """

        positive_count = int(
            (y == 1).sum()
        )

        negative_count = int(
            (y == 0).sum()
        )

        if positive_count == 0:

            return 1.0

        return max(
            1.0,
            negative_count
            /
            positive_count
        )

    # ======================================================================
    # TRAIN
    # ======================================================================

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[
            pd.DataFrame
        ] = None,
        y_val: Optional[
            pd.Series
        ] = None,
        custom_params: Optional[
            Dict[str, Any]
        ] = None,
        use_class_weight: bool = False
    ) -> Dict[str, Any]:
        """
        Train the SSL XGBoost model.

        Args:
            X_train:
                Preprocessed training features.

            y_train:
                Training labels.

            X_val:
                Optional validation features.

            y_val:
                Optional validation labels.

            custom_params:
                Optional XGBoost parameter overrides.

            use_class_weight:
                If True, automatically calculates scale_pos_weight
                from training labels.

        Returns:
            Dictionary containing training and validation metrics.
        """

        try:

            # ==============================================================
            # VALIDATE TRAINING DATA
            # ==============================================================

            X_train = (
                self._validate_feature_matrix(
                    X_train
                )
            )

            y_train = (
                self._validate_target(
                    y_train,
                    len(X_train)
                )
            )

            # ==============================================================
            # VALIDATE VALIDATION DATA
            # ==============================================================

            if (
                X_val is not None
                and
                y_val is None
            ):

                raise ValueError(
                    "X_val was provided but y_val is missing."
                )

            if (
                X_val is None
                and
                y_val is not None
            ):

                raise ValueError(
                    "y_val was provided but X_val is missing."
                )

            if X_val is not None:

                X_val = (
                    self._validate_feature_matrix(
                        X_val
                    )
                )

                y_val = (
                    self._validate_target(
                        y_val,
                        len(X_val)
                    )
                )

            # ==============================================================
            # CHECK BOTH CLASSES
            # ==============================================================

            train_classes = set(
                y_train.unique().tolist()
            )

            if len(
                train_classes
            ) < 2:

                raise ValueError(
                    (
                        "Training data must contain both "
                        "classes 0 and 1."
                    )
                )

            # ==============================================================
            # BUILD MODEL
            # ==============================================================

            self.build_model(
                custom_params
            )

            # ==============================================================
            # OPTIONAL CLASS BALANCING
            # ==============================================================

            if use_class_weight:

                calculated_weight = (
                    self._calculate_scale_pos_weight(
                        y_train
                    )
                )

                self.model.set_params(
                    scale_pos_weight=(
                        calculated_weight
                    )
                )

                self.model_params[
                    "scale_pos_weight"
                ] = calculated_weight

                logger.info(
                    "Using scale_pos_weight=%.4f",
                    calculated_weight
                )

            # ==============================================================
            # EVALUATION SET
            # ==============================================================

            eval_set = [
                (
                    X_train,
                    y_train
                )
            ]

            if (
                X_val is not None
                and
                y_val is not None
            ):

                eval_set.append(
                    (
                        X_val,
                        y_val
                    )
                )

            # ==============================================================
            # TRAIN
            # ==============================================================

            logger.info(
                (
                    "Starting SSL XGBoost training. "
                    "Samples=%d Features=%d"
                ),
                len(X_train),
                len(self.feature_names)
            )

            self.model.fit(

                X_train,

                y_train,

                eval_set=eval_set,

                verbose=False
            )

            self.is_trained = True

            self.feature_names = (
                list(
                    X_train.columns
                )
            )

            self.created_at = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            # ==============================================================
            # TRAINING METRICS
            # ==============================================================

            metrics = (
                self._calculate_metrics(
                    X_train,
                    y_train,
                    prefix="train"
                )
            )

            # ==============================================================
            # VALIDATION METRICS
            # ==============================================================

            if (
                X_val is not None
                and
                y_val is not None
            ):

                validation_metrics = (
                    self._calculate_metrics(
                        X_val,
                        y_val,
                        prefix="val"
                    )
                )

                metrics.update(
                    validation_metrics
                )

            # ==============================================================
            # CLASS DISTRIBUTION
            # ==============================================================

            metrics[
                "train_samples"
            ] = int(
                len(X_train)
            )

            metrics[
                "feature_count"
            ] = int(
                len(
                    self.feature_names
                )
            )

            metrics[
                "positive_samples"
            ] = int(
                (y_train == 1).sum()
            )

            metrics[
                "negative_samples"
            ] = int(
                (y_train == 0).sum()
            )

            metrics[
                "model_algorithm"
            ] = self.MODEL_ALGORITHM

            metrics[
                "model_version"
            ] = self.MODEL_VERSION

            metrics[
                "schema_version"
            ] = (
                SSLFeatureSchema.SCHEMA_VERSION
            )

            self.training_metrics = (
                metrics.copy()
            )

            logger.info(
                "SSL model training completed successfully."
            )

            if (
                "val_auc"
                in metrics
            ):

                logger.info(
                    (
                        "SSL Validation: "
                        "AUC=%.4f Accuracy=%.4f F1=%.4f"
                    ),
                    metrics.get(
                        "val_auc",
                        0.0
                    ),
                    metrics.get(
                        "val_accuracy",
                        0.0
                    ),
                    metrics.get(
                        "val_f1",
                        0.0
                    )
                )

            return metrics

        except Exception as e:

            self.last_error = str(
                e
            )

            self.is_trained = False

            logger.error(
                "SSL model training failed: %s",
                str(e),
                exc_info=True
            )

            raise

    # ======================================================================
    # METRICS
    # ======================================================================

    def _calculate_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prefix: str
    ) -> Dict[str, Any]:
        """
        Calculate classification metrics.

        Metrics:

            accuracy
            precision
            recall
            F1
            ROC-AUC
            confusion matrix
            classification report
        """

        if not self.is_trained or self.model is None:

            raise RuntimeError(
                "Cannot calculate metrics before model training."
            )

        X = self._validate_feature_matrix(
            X
        )

        y = self._validate_target(
            y,
            len(X)
        )

        predictions = (
            self.model.predict(
                X
            )
        )

        probabilities = (
            self.model.predict_proba(
                X
            )[:, 1]
        )

        metrics: Dict[
            str,
            Any
        ] = {}

        # ------------------------------------------------------------------
        # Accuracy
        # ------------------------------------------------------------------

        metrics[
            f"{prefix}_accuracy"
        ] = float(
            accuracy_score(
                y,
                predictions
            )
        )

        # ------------------------------------------------------------------
        # Precision
        # ------------------------------------------------------------------

        metrics[
            f"{prefix}_precision"
        ] = float(
            precision_score(
                y,
                predictions,
                zero_division=0
            )
        )

        # ------------------------------------------------------------------
        # Recall
        # ------------------------------------------------------------------

        metrics[
            f"{prefix}_recall"
        ] = float(
            recall_score(
                y,
                predictions,
                zero_division=0
            )
        )

        # ------------------------------------------------------------------
        # F1
        # ------------------------------------------------------------------

        metrics[
            f"{prefix}_f1"
        ] = float(
            f1_score(
                y,
                predictions,
                zero_division=0
            )
        )

        # ------------------------------------------------------------------
        # ROC-AUC
        # ------------------------------------------------------------------

        if len(
            np.unique(
                y
            )
        ) > 1:

            metrics[
                f"{prefix}_auc"
            ] = float(
                roc_auc_score(
                    y,
                    probabilities
                )
            )

        else:

            metrics[
                f"{prefix}_auc"
            ] = None

        # ------------------------------------------------------------------
        # Confusion matrix
        # ------------------------------------------------------------------

        cm = confusion_matrix(
            y,
            predictions,
            labels=[
                0,
                1
            ]
        )

        metrics[
            f"{prefix}_confusion_matrix"
        ] = cm.tolist()

        # ------------------------------------------------------------------
        # Classification report
        # ------------------------------------------------------------------

        report = classification_report(
            y,
            predictions,
            labels=[
                0,
                1
            ],
            target_names=[
                self.CLASS_LABELS[
                    0
                ],

                self.CLASS_LABELS[
                    1
                ]
            ],
            output_dict=True,
            zero_division=0
        )

        metrics[
            f"{prefix}_classification_report"
        ] = report

        return metrics

    # ======================================================================
    # PREDICT PROBABILITY
    # ======================================================================

    def predict_proba(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Return class probabilities.

        Output:

            [
                [P(legitimate), P(suspicious)],
                ...
            ]

        Args:
            X:
                Preprocessed SSL feature DataFrame.

        Returns:
            NumPy array with shape:

                (n_samples, 2)
        """

        if not self.is_trained:

            raise RuntimeError(
                (
                    "Cannot predict: SSL model "
                    "has not been trained or loaded."
                )
            )

        if self.model is None:

            raise RuntimeError(
                "Cannot predict: model object is unavailable."
            )

        X = self._validate_feature_matrix(
            X
        )

        probabilities = (
            self.model.predict_proba(
                X
            )
        )

        if probabilities.ndim != 2:

            raise RuntimeError(
                (
                    "Unexpected probability output shape: "
                    f"{probabilities.shape}"
                )
            )

        if probabilities.shape[1] != 2:

            raise RuntimeError(
                (
                    "SSL binary model must return two class "
                    f"probabilities. Received shape={probabilities.shape}"
                )
            )

        return probabilities

    # ======================================================================
    # PREDICT CLASS
    # ======================================================================

    def predict(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict class labels.

        Returns:

            0 = legitimate
            1 = suspicious
        """

        if not self.is_trained:

            raise RuntimeError(
                "Cannot predict: SSL model is not trained."
            )

        if self.model is None:

            raise RuntimeError(
                "Cannot predict: model object is unavailable."
            )

        X = self._validate_feature_matrix(
            X
        )

        predictions = (
            self.model.predict(
                X
            )
        )

        return predictions.astype(
            int
        )

    # ======================================================================
    # PREDICT SINGLE
    # ======================================================================

    def predict_single(
        self,
        X: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Predict one SSL feature vector and return a standardized result.

        Args:
            X:
                DataFrame containing exactly one row.

        Returns:
            Dictionary containing:

                class_index
                class_label
                confidence
                legitimate_probability
                suspicious_probability
        """

        if len(X) != 1:

            raise ValueError(
                (
                    "predict_single() expects exactly "
                    "one feature row."
                )
            )

        probabilities = (
            self.predict_proba(
                X
            )
        )

        predicted_class = int(
            np.argmax(
                probabilities[
                    0
                ]
            )
        )

        legitimate_probability = float(
            probabilities[
                0,
                0
            ]
        )

        suspicious_probability = float(
            probabilities[
                0,
                1
            ]
        )

        confidence = float(
            max(
                legitimate_probability,
                suspicious_probability
            )
        )

        return {

            "class_index":
                predicted_class,

            "class_label":
                self.CLASS_LABELS[
                    predicted_class
                ],

            "confidence":
                confidence,

            "legitimate_probability":
                legitimate_probability,

            "suspicious_probability":
                suspicious_probability
        }

    # ======================================================================
    # FEATURE IMPORTANCE
    # ======================================================================

    def get_feature_importances(
        self
    ) -> Dict[str, float]:
        """
        Return XGBoost feature importance values.

        Returns:
            Dictionary sorted from highest to lowest importance.
        """

        if not self.is_trained:

            return {}

        if self.model is None:

            return {}

        importances = (
            self.model.feature_importances_
        )

        importance_map = {

            feature_name:
                float(
                    importance
                )

            for feature_name,
            importance

            in zip(
                self.feature_names,
                importances
            )
        }

        return dict(
            sorted(
                importance_map.items(),
                key=lambda item:
                    item[1],
                reverse=True
            )
        )

    # ======================================================================
    # GET MODEL
    # ======================================================================

    def get_model(
        self
    ) -> Optional[
        xgb.XGBClassifier
    ]:
        """
        Return the underlying XGBoost model.

        Used by:

            explain.py
            SHAP
            diagnostics
        """

        return self.model

    # ======================================================================
    # GET MODEL PARAMETERS
    # ======================================================================

    def get_model_params(
        self
    ) -> Dict[str, Any]:
        """
        Return model parameters.
        """

        if self.model is None:

            return self.model_params.copy()

        try:

            return self.model.get_params()

        except Exception:

            return self.model_params.copy()

    # ======================================================================
    # SAVE
    # ======================================================================

    def save(
        self,
        filepath: Optional[str] = None
    ) -> str:
        """
        Persist trained model and metadata to disk.

        The artifact contains:

            model
            feature_names
            schema version
            model version
            model parameters
            training metrics
            creation timestamp
            class labels
        """

        if not self.is_trained:

            raise RuntimeError(
                "Cannot save an untrained SSL model."
            )

        if self.model is None:

            raise RuntimeError(
                "Cannot save because model object is unavailable."
            )

        save_target = (
            filepath
            or
            self.model_path
        )

        save_parent = os.path.dirname(
            os.path.abspath(
                save_target
            )
        )

        os.makedirs(
            save_parent,
            exist_ok=True
        )

        # ------------------------------------------------------------------
        # Validate model schema before saving
        # ------------------------------------------------------------------

        expected_features = (
            SSLFeatureSchema.get_feature_names()
        )

        if self.feature_names != expected_features:

            raise ValueError(
                (
                    "Cannot save SSL model because its feature schema "
                    "does not match SSLFeatureSchema."
                )
            )

        # ------------------------------------------------------------------
        # Artifact payload
        # ------------------------------------------------------------------

        payload = {

            "model":
                self.model,

            "feature_names":
                self.feature_names.copy(),

            "is_trained":
                self.is_trained,

            "model_name":
                self.MODEL_NAME,

            "model_version":
                self.MODEL_VERSION,

            "model_algorithm":
                self.MODEL_ALGORITHM,

            "model_type":
                self.MODEL_TYPE,

            "schema_name":
                SSLFeatureSchema.SCHEMA_NAME,

            "schema_version":
                SSLFeatureSchema.SCHEMA_VERSION,

            "class_labels":
                self.CLASS_LABELS.copy(),

            "model_params":
                self.get_model_params(),

            "training_metrics":
                self.training_metrics.copy(),

            "created_at":
                self.created_at
                or
                datetime.now(
                    timezone.utc
                ).isoformat()
        }

        try:

            joblib.dump(
                payload,
                save_target
            )

            logger.info(
                "SSL model successfully saved to %s",
                save_target
            )

            self.model_path = (
                save_target
            )

            return save_target

        except Exception as e:

            self.last_error = str(
                e
            )

            logger.error(
                "Failed to save SSL model: %s",
                str(e),
                exc_info=True
            )

            raise

    # ======================================================================
    # LOAD
    # ======================================================================

    def load(
        self,
        filepath: Optional[str] = None
    ) -> bool:
        """
        Load a previously trained SSL model.

        Returns:
            True:
                Successfully loaded.

            False:
                File unavailable or invalid.
        """

        load_target = (
            filepath
            or
            self.model_path
        )

        if not os.path.exists(
            load_target
        ):

            logger.warning(
                "No SSL model artifact found at %s",
                load_target
            )

            return False

        try:

            payload = joblib.load(
                load_target
            )

            # --------------------------------------------------------------
            # Validate payload
            # --------------------------------------------------------------

            if not isinstance(
                payload,
                dict
            ):

                raise ValueError(
                    "SSL model artifact is not a valid dictionary."
                )

            if "model" not in payload:

                raise ValueError(
                    "SSL model artifact does not contain 'model'."
                )

            loaded_model = (
                payload[
                    "model"
                ]
            )

            if not isinstance(
                loaded_model,
                xgb.XGBClassifier
            ):

                raise TypeError(
                    (
                        "Loaded SSL artifact does not contain "
                        "an XGBClassifier."
                    )
                )

            # --------------------------------------------------------------
            # Feature schema
            # --------------------------------------------------------------

            loaded_features = (
                payload.get(
                    "feature_names",
                    []
                )
            )

            expected_features = (
                SSLFeatureSchema.get_feature_names()
            )

            if loaded_features:

                if list(
                    loaded_features
                ) != expected_features:

                    raise ValueError(
                        (
                            "Loaded SSL model feature schema does not "
                            "match current SSLFeatureSchema."
                        )
                    )

            else:

                logger.warning(
                    (
                        "Loaded SSL model artifact does not contain "
                        "feature_names metadata. Using current schema."
                    )
                )

                loaded_features = (
                    expected_features
                )

            # --------------------------------------------------------------
            # Assign model
            # --------------------------------------------------------------

            self.model = (
                loaded_model
            )

            self.feature_names = (
                list(
                    loaded_features
                )
            )

            self.is_trained = bool(
                payload.get(
                    "is_trained",
                    True
                )
            )

            self.model_params = (
                payload.get(
                    "model_params",
                    {}
                )
            )

            self.training_metrics = (
                payload.get(
                    "training_metrics",
                    {}
                )
            )

            self.created_at = (
                payload.get(
                    "created_at"
                )
            )

            self.loaded_from = (
                load_target
            )

            self.model_path = (
                load_target
            )

            self.last_error = None

            logger.info(
                "SSL model successfully loaded from %s",
                load_target
            )

            return True

        except Exception as e:

            self.model = None

            self.is_trained = False

            self.last_error = str(
                e
            )

            logger.error(
                "Failed to load SSL model artifact: %s",
                str(e),
                exc_info=True
            )

            return False

    # ======================================================================
    # CHECK MODEL COMPATIBILITY
    # ======================================================================

    def check_compatibility(
        self
    ) -> Tuple[bool, List[str]]:
        """
        Check whether the currently loaded model is compatible with
        the current SSL feature schema.

        Returns:
            (
                compatible,
                errors
            )
        """

        errors = []

        # ------------------------------------------------------------------
        # Model existence
        # ------------------------------------------------------------------

        if self.model is None:

            errors.append(
                "Model object is not loaded."
            )

            return (
                False,
                errors
            )

        # ------------------------------------------------------------------
        # Feature names
        # ------------------------------------------------------------------

        expected_features = (
            SSLFeatureSchema.get_feature_names()
        )

        if self.feature_names != expected_features:

            errors.append(
                (
                    "Model feature names do not match "
                    "SSLFeatureSchema."
                )
            )

        # ------------------------------------------------------------------
        # Model feature count
        # ------------------------------------------------------------------

        try:

            model_feature_count = (
                self.model.n_features_in_
            )

            if (
                model_feature_count
                !=
                len(
                    expected_features
                )
            ):

                errors.append(
                    (
                        "Model feature count mismatch. "
                        f"Model={model_feature_count}, "
                        f"Schema={len(expected_features)}"
                    )
                )

        except AttributeError:

            logger.debug(
                "Model does not expose n_features_in_."
            )

        return (
            len(errors) == 0,
            errors
        )

    # ======================================================================
    # STATUS
    # ======================================================================

    def get_status(
        self
    ) -> Dict[str, Any]:
        """
        Return complete model health/status information.

        Useful for:

            ssl_agent.py
            debugging
            API health endpoints
        """

        compatible = False

        compatibility_errors: List[
            str
        ] = []

        if self.model is not None:

            compatible, compatibility_errors = (
                self.check_compatibility()
            )

        return {

            "model_name":
                self.MODEL_NAME,

            "model_version":
                self.MODEL_VERSION,

            "algorithm":
                self.MODEL_ALGORITHM,

            "model_type":
                self.MODEL_TYPE,

            "trained":
                self.is_trained,

            "model_available":
                self.model is not None,

            "schema_name":
                SSLFeatureSchema.SCHEMA_NAME,

            "schema_version":
                SSLFeatureSchema.SCHEMA_VERSION,

            "feature_count":
                len(
                    self.feature_names
                ),

            "feature_names":
                self.feature_names.copy(),

            "model_path":
                self.model_path,

            "loaded_from":
                self.loaded_from,

            "schema_compatible":
                compatible,

            "compatibility_errors":
                compatibility_errors,

            "last_error":
                self.last_error
        }

    # ======================================================================
    # GET TRAINING METRICS
    # ======================================================================

    def get_training_metrics(
        self
    ) -> Dict[str, Any]:
        """
        Return a copy of the latest training metrics.
        """

        return self.training_metrics.copy()

    # ======================================================================
    # GET CLASS LABELS
    # ======================================================================

    def get_class_labels(
        self
    ) -> Dict[int, str]:
        """
        Return model class mapping.
        """

        return self.CLASS_LABELS.copy()