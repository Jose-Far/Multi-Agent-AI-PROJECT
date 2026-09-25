"""
DNS Security AI Agent - XGBoost Model
======================================

This module manages the machine-learning model used by the DNS Security
AI Agent.

Responsibilities
----------------
1. Configure the XGBoost classifier.
2. Build a fresh model for training.
3. Train the model.
4. Validate training data.
5. Validate inference data.
6. Save trained model weights.
7. Load trained model weights.
8. Predict phishing/malicious probability.
9. Predict binary class.
10. Perform batch inference.
11. Validate compatibility with DNSFeatureSchema.
12. Expose model metadata.
13. Provide model health/status information.

Architecture
------------

    DNS Feature Extractor
             |
             v
    DNSFeatureSchema
             |
             v
    DNSPreprocessor
             |
             v
       DNSModel
             |
       +-----+------+
       |            |
       v            v
    Probability    Class
       |            |
       +-----+------+
             |
             v
       Risk Scoring
             |
             v
       Explainability
             |
             v
      Decision Fusion


Model
-----

XGBoost binary classifier.

Target:

    0 = Legitimate / Non-Phishing
    1 = Phishing / Suspicious

Output:

    predict_proba() -> probability between 0.0 and 1.0

Important Security Principle
----------------------------

The DNS model is NOT the final phishing decision engine.

A high DNS probability means that DNS-related characteristics resemble
patterns present in the malicious class used during training.

It does NOT prove that the website is phishing.

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

through the Multi-Agent Decision Fusion Engine.

Author:
    Multi-Agent AI Cybersecurity Analyst

Agent:
    DNS Security AI Agent

Version:
    1.0.0
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from .feature_schema import (
    DNSFeatureSchema,
    SCHEMA_VERSION,
)


# =============================================================================
# LOGGER
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL VERSION
# =============================================================================

MODEL_WRAPPER_VERSION = "1.0.0"


# =============================================================================
# DNS MODEL
# =============================================================================

class DNSModel:
    """
    Production-oriented XGBoost model wrapper for the DNS Security AI Agent.

    This class deliberately keeps ML lifecycle responsibilities separate
    from feature extraction, preprocessing, risk scoring and explainability.

    Lifecycle
    ---------

        model = DNSModel()

        model.build_model()

        model.train(
            X_train,
            y_train
        )

        model.save_model()

        # Later:
        model.load_model()

        probability = model.predict_proba(
            processed_features
        )

    The class expects already-preprocessed numeric DNS features.

    Feature schema:
        DNSFeatureSchema.EXPECTED_FEATURE_ORDER

    Number of features:
        37
    """

    # =========================================================================
    # 1. DEFAULT MODEL HYPERPARAMETERS
    # =========================================================================

    DEFAULT_HYPERPARAMETERS: Dict[str, Any] = {

        # Number of boosting trees.
        "n_estimators": 200,

        # Maximum tree depth.
        #
        # DNS feature interactions can be nonlinear, but excessive depth
        # can overfit the training data.
        "max_depth": 6,

        # Learning rate controls the contribution of each tree.
        "learning_rate": 0.05,

        # Binary classification.
        "objective": "binary:logistic",

        # Log loss is appropriate for probability-based binary classification.
        "eval_metric": "logloss",

        # Use 80% of training samples per boosting round.
        "subsample": 0.8,

        # Use 80% of features for each tree.
        "colsample_bytree": 0.8,

        # Reproducibility.
        "random_state": 42,

        # Use available CPU cores.
        "n_jobs": -1,

        # Prevent excessive output during normal training.
        "verbosity": 0,
    }

    # =========================================================================
    # 2. TARGET LABELS
    # =========================================================================

    LEGITIMATE_CLASS = 0
    PHISHING_CLASS = 1

    CLASS_NAMES: Dict[int, str] = {
        0: "legitimate",
        1: "phishing",
    }

    # =========================================================================
    # 3. INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        model_path: Optional[str] = None,
        hyperparameters: Optional[Mapping[str, Any]] = None,
        auto_load: bool = False,
    ) -> None:
        """
        Initialize the DNS XGBoost model wrapper.

        Parameters
        ----------
        model_path:
            Optional path to the XGBoost model file.

        hyperparameters:
            Optional custom XGBoost configuration.

        auto_load:
            If True, attempt to load the model immediately if the model file
            exists.

        Notes
        -----
        The default model path is:

            agents/dns_agent/weights/dns_xgb_model.json
        """

        # ---------------------------------------------------------------------
        # Resolve default model location.
        # ---------------------------------------------------------------------

        default_path = (
            Path(__file__).resolve().parent
            / "weights"
            / "dns_xgb_model.json"
        )

        self.model_path: Path = Path(
            model_path
        ).expanduser().resolve() if model_path else default_path

        # ---------------------------------------------------------------------
        # Model object.
        # ---------------------------------------------------------------------

        self.model: Optional[xgb.XGBClassifier] = None

        # ---------------------------------------------------------------------
        # State.
        # ---------------------------------------------------------------------

        self.is_loaded: bool = False
        self.is_trained: bool = False

        # ---------------------------------------------------------------------
        # Metadata.
        # ---------------------------------------------------------------------

        self.model_version: str = MODEL_WRAPPER_VERSION
        self.schema_version: str = SCHEMA_VERSION

        # ---------------------------------------------------------------------
        # Hyperparameters.
        # ---------------------------------------------------------------------

        self.hyperparameters: Dict[str, Any] = dict(
            self.DEFAULT_HYPERPARAMETERS
        )

        if hyperparameters:

            self.hyperparameters.update(
                dict(hyperparameters)
            )

        # ---------------------------------------------------------------------
        # Validate configuration.
        # ---------------------------------------------------------------------

        self._validate_hyperparameters()

        logger.info(
            "DNSModel initialized. Model path=%s",
            self.model_path,
        )

        # ---------------------------------------------------------------------
        # Optional automatic loading.
        # ---------------------------------------------------------------------

        if auto_load:

            self.load_model()

    # =========================================================================
    # 4. HYPERPARAMETER VALIDATION
    # =========================================================================

    def _validate_hyperparameters(self) -> None:
        """
        Validate important XGBoost configuration values before model creation.
        """

        n_estimators = self.hyperparameters.get(
            "n_estimators"
        )

        if not isinstance(
            n_estimators,
            int,
        ) or n_estimators <= 0:

            raise ValueError(
                "n_estimators must be a positive integer."
            )

        max_depth = self.hyperparameters.get(
            "max_depth"
        )

        if not isinstance(
            max_depth,
            int,
        ) or max_depth <= 0:

            raise ValueError(
                "max_depth must be a positive integer."
            )

        learning_rate = float(
            self.hyperparameters.get(
                "learning_rate",
                0.05,
            )
        )

        if not 0.0 < learning_rate <= 1.0:

            raise ValueError(
                "learning_rate must be between 0 and 1."
            )

        subsample = float(
            self.hyperparameters.get(
                "subsample",
                1.0,
            )
        )

        if not 0.0 < subsample <= 1.0:

            raise ValueError(
                "subsample must be between 0 and 1."
            )

        colsample = float(
            self.hyperparameters.get(
                "colsample_bytree",
                1.0,
            )
        )

        if not 0.0 < colsample <= 1.0:

            raise ValueError(
                "colsample_bytree must be between 0 and 1."
            )

        objective = self.hyperparameters.get(
            "objective"
        )

        if objective != "binary:logistic":

            raise ValueError(
                "DNSModel currently requires "
                "objective='binary:logistic'."
            )

    # =========================================================================
    # 5. BUILD MODEL
    # =========================================================================

    def build_model(
        self,
        force_rebuild: bool = False,
    ) -> xgb.XGBClassifier:
        """
        Build a fresh untrained XGBoost classifier.

        Parameters
        ----------
        force_rebuild:
            If True, replace an existing model object.

        Returns
        -------
        xgb.XGBClassifier
            Fresh XGBoost classifier.
        """

        if (
            self.model is not None
            and not force_rebuild
        ):

            logger.debug(
                "Existing DNS XGBoost model already exists."
            )

            return self.model

        logger.info(
            "Building DNS XGBoost classifier."
        )

        self.model = xgb.XGBClassifier(
            **self.hyperparameters
        )

        self.is_loaded = False
        self.is_trained = False

        return self.model

    # =========================================================================
    # 6. INPUT VALIDATION
    # =========================================================================

    def _validate_feature_dataframe(
        self,
        X: pd.DataFrame,
        allow_multiple_rows: bool = True,
    ) -> None:
        """
        Validate a model feature DataFrame against DNSFeatureSchema.
        """

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "DNS model input must be a pandas DataFrame."
            )

        if X.empty:

            raise ValueError(
                "DNS model input DataFrame is empty."
            )

        if (
            not allow_multiple_rows
            and len(X) != 1
        ):

            raise ValueError(
                "This operation expects exactly one row."
            )

        # Authoritative schema validation.
        DNSFeatureSchema.validate_dataframe(
            X
        )

        # Explicit numeric safety.
        values = X.to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():

            raise ValueError(
                "DNS model input contains NaN or infinite values."
            )

    # =========================================================================
    # 7. TARGET VALIDATION
    # =========================================================================

    def _validate_target(
        self,
        y: Iterable[Any],
    ) -> np.ndarray:
        """
        Validate and normalize binary training labels.
        """

        y_array = np.asarray(
            list(y)
        )

        if y_array.ndim != 1:

            raise ValueError(
                "DNS target labels must be one-dimensional."
            )

        if len(y_array) == 0:

            raise ValueError(
                "DNS target labels are empty."
            )

        try:

            y_numeric = y_array.astype(
                int
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "DNS target labels must contain "
                "binary numeric values 0 and 1."
            ) from exc

        unique_labels = set(
            np.unique(
                y_numeric
            ).tolist()
        )

        if not unique_labels.issubset(
            {
                self.LEGITIMATE_CLASS,
                self.PHISHING_CLASS,
            }
        ):

            raise ValueError(
                "DNS target labels must contain only 0 and 1. "
                f"Received labels: {sorted(unique_labels)}"
            )

        if len(unique_labels) < 2:

            raise ValueError(
                "DNS training data must contain both classes: "
                "0=legitimate and 1=phishing."
            )

        return y_numeric

    # =========================================================================
    # 8. TRAIN MODEL
    # =========================================================================

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: Iterable[Any],
        X_eval: Optional[pd.DataFrame] = None,
        y_eval: Optional[Iterable[Any]] = None,
        verbose: bool = False,
    ) -> xgb.XGBClassifier:
        """
        Train the DNS XGBoost classifier.

        Parameters
        ----------
        X_train:
            Preprocessed DNS training features.

        y_train:
            Binary labels.

        X_eval:
            Optional validation features.

        y_eval:
            Optional validation labels.

        verbose:
            Whether XGBoost should output training progress.

        Returns
        -------
        xgb.XGBClassifier
            Trained model.

        Notes
        -----
        Preprocessing must already have been performed.

        Expected feature count:

            37
        """

        logger.info(
            "Starting DNS model training."
        )

        # ---------------------------------------------------------------------
        # Validate training features.
        # ---------------------------------------------------------------------

        self._validate_feature_dataframe(
            X_train,
            allow_multiple_rows=True,
        )

        # ---------------------------------------------------------------------
        # Validate labels.
        # ---------------------------------------------------------------------

        y_train_array = self._validate_target(
            y_train
        )

        if len(X_train) != len(
            y_train_array
        ):

            raise ValueError(
                "Training feature and label counts do not match. "
                f"X={len(X_train)}, y={len(y_train_array)}"
            )

        # ---------------------------------------------------------------------
        # Validation data.
        # ---------------------------------------------------------------------

        evaluation_set = None

        if X_eval is not None:

            if y_eval is None:

                raise ValueError(
                    "y_eval must be supplied when X_eval is supplied."
                )

            self._validate_feature_dataframe(
                X_eval,
                allow_multiple_rows=True,
            )

            y_eval_array = self._validate_target(
                y_eval
            )

            if len(X_eval) != len(
                y_eval_array
            ):

                raise ValueError(
                    "Evaluation feature and label counts do not match. "
                    f"X_eval={len(X_eval)}, "
                    f"y_eval={len(y_eval_array)}"
                )

            evaluation_set = [
                (
                    X_eval,
                    y_eval_array,
                )
            ]

        # ---------------------------------------------------------------------
        # Build model.
        # ---------------------------------------------------------------------

        self.build_model(
            force_rebuild=True
        )

        assert self.model is not None

        # ---------------------------------------------------------------------
        # Train.
        # ---------------------------------------------------------------------

        logger.info(
            "Training DNS XGBoost model with %d samples and %d features.",
            len(X_train),
            len(X_train.columns),
        )

        fit_kwargs: Dict[str, Any] = {
            "verbose": verbose,
        }

        if evaluation_set is not None:

            fit_kwargs["eval_set"] = evaluation_set

        self.model.fit(
            X_train,
            y_train_array,
            **fit_kwargs,
        )

        # ---------------------------------------------------------------------
        # Update state.
        # ---------------------------------------------------------------------

        self.is_trained = True
        self.is_loaded = True

        logger.info(
            "DNS XGBoost model training completed successfully."
        )

        return self.model

    # =========================================================================
    # 9. SAVE MODEL
    # =========================================================================

    def save_model(
        self,
        custom_path: Optional[str] = None,
    ) -> bool:
        """
        Save the trained XGBoost model to disk.

        XGBoost's native JSON format is used.

        Example:

            agents/dns_agent/weights/dns_xgb_model.json
        """

        if self.model is None:

            raise ValueError(
                "No DNS model instance exists. "
                "Call build_model() or train() first."
            )

        if not self.is_trained:

            raise ValueError(
                "DNS model has not been trained. "
                "A model must be trained before saving."
            )

        save_path = (
            Path(custom_path).expanduser().resolve()
            if custom_path
            else self.model_path
        )

        try:

            # -------------------------------------------------------------
            # Create parent directory.
            # -------------------------------------------------------------

            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # -------------------------------------------------------------
            # Save native XGBoost model.
            # -------------------------------------------------------------

            self.model.save_model(
                str(save_path)
            )

            # -------------------------------------------------------------
            # Save companion metadata.
            # -------------------------------------------------------------

            self._save_metadata(
                save_path
            )

            # Update current path.
            self.model_path = save_path

            logger.info(
                "DNS XGBoost model saved successfully: %s",
                save_path,
            )

            return True

        except Exception as exc:

            logger.exception(
                "Failed to save DNS model to %s: %s",
                save_path,
                exc,
            )

            raise

    # =========================================================================
    # 10. SAVE MODEL METADATA
    # =========================================================================

    def _save_metadata(
        self,
        model_path: Path,
    ) -> None:
        """
        Save model metadata next to the XGBoost model.

        Example:

            dns_xgb_model.json
            dns_xgb_model.metadata.json
        """

        metadata_path = model_path.with_suffix(
            ".metadata.json"
        )

        metadata = self.get_model_metadata()

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as metadata_file:

            json.dump(
                metadata,
                metadata_file,
                indent=4,
                sort_keys=True,
            )

        logger.debug(
            "DNS model metadata saved to %s",
            metadata_path,
        )

    # =========================================================================
    # 11. LOAD MODEL
    # =========================================================================

    def load_model(
        self,
        custom_path: Optional[str] = None,
        strict: bool = True,
    ) -> bool:
        """
        Load a trained XGBoost model from disk.

        Parameters
        ----------
        custom_path:
            Optional model file path.

        strict:
            If True, incompatible/corrupt models raise exceptions.
            If False, errors are logged and False is returned.

        Returns
        -------
        bool
            True when loading succeeds.
        """

        load_path = (
            Path(custom_path).expanduser().resolve()
            if custom_path
            else self.model_path
        )

        if not load_path.exists():

            message = (
                f"DNS model file not found: {load_path}"
            )

            if strict:

                logger.warning(
                    message
                )

                return False

            logger.warning(
                message
            )

            return False

        try:

            logger.info(
                "Loading DNS XGBoost model from %s",
                load_path,
            )

            loaded_model = xgb.XGBClassifier()

            loaded_model.load_model(
                str(load_path)
            )

            self.model = loaded_model

            self.model_path = load_path

            self.is_loaded = True
            self.is_trained = True

            # -------------------------------------------------------------
            # Validate loaded model schema if possible.
            # -------------------------------------------------------------

            self._validate_loaded_model_schema()

            logger.info(
                "DNS XGBoost model loaded successfully."
            )

            return True

        except Exception as exc:

            self.is_loaded = False
            self.is_trained = False
            self.model = None

            logger.exception(
                "Failed to load DNS XGBoost model: %s",
                exc,
            )

            if strict:

                raise

            return False

    # =========================================================================
    # 12. LOADED MODEL SCHEMA VALIDATION
    # =========================================================================

    def _validate_loaded_model_schema(self) -> None:
        """
        Validate the loaded XGBoost model's feature count where possible.

        XGBoost models may expose `n_features_in_` after loading.
        """

        if self.model is None:
            raise RuntimeError(
                "Cannot validate schema because model is None."
            )

        expected_count = (
            DNSFeatureSchema.feature_count()
        )

        model_feature_count = getattr(
            self.model,
            "n_features_in_",
            None,
        )

        if model_feature_count is not None:

            if int(model_feature_count) != expected_count:

                raise ValueError(
                    "Loaded DNS model feature count mismatch. "
                    f"Expected={expected_count}, "
                    f"Model={model_feature_count}"
                )

    # =========================================================================
    # 13. MODEL AVAILABILITY
    # =========================================================================

    def is_available(self) -> bool:
        """
        Return True if a trained model is currently available for inference.
        """

        return bool(
            self.model is not None
            and self.is_trained
        )

    # =========================================================================
    # 14. SINGLE PROBABILITY PREDICTION
    # =========================================================================

    def predict_proba(
        self,
        df_processed: pd.DataFrame,
    ) -> float:
        """
        Predict the probability that a single domain belongs to the
        phishing/malicious class.

        Parameters
        ----------
        df_processed:
            Exactly one row of preprocessed DNS features.

        Returns
        -------
        float
            Probability between 0.0 and 1.0.

        Example
        -------

            probability = model.predict_proba(
                processed_features
            )

            print(probability)
        """

        if not self.is_available():

            raise RuntimeError(
                "DNS model is not loaded/trained. "
                "Call load_model() or train() first."
            )

        self._validate_feature_dataframe(
            df_processed,
            allow_multiple_rows=False,
        )

        assert self.model is not None

        try:

            probabilities = (
                self.model.predict_proba(
                    df_processed
                )
            )

            if probabilities.ndim != 2:

                raise ValueError(
                    "Unexpected XGBoost probability output shape: "
                    f"{probabilities.shape}"
                )

            if probabilities.shape[0] != 1:

                raise ValueError(
                    "predict_proba() expected exactly one row."
                )

            if probabilities.shape[1] < 2:

                raise ValueError(
                    "Binary probability output must contain "
                    "at least two class probabilities."
                )

            malicious_probability = float(
                probabilities[0][
                    self.PHISHING_CLASS
                ]
            )

            # Defensive bounds.
            malicious_probability = max(
                0.0,
                min(
                    1.0,
                    malicious_probability,
                ),
            )

            logger.debug(
                "DNS phishing probability: %.6f",
                malicious_probability,
            )

            return malicious_probability

        except Exception as exc:

            logger.exception(
                "DNS model probability prediction failed: %s",
                exc,
            )

            raise

    # =========================================================================
    # 15. SINGLE CLASS PREDICTION
    # =========================================================================

    def predict(
        self,
        df_processed: pd.DataFrame,
    ) -> int:
        """
        Predict the binary class for a single domain.

        Returns:

            0 = legitimate
            1 = phishing
        """

        if not self.is_available():

            raise RuntimeError(
                "DNS model is not loaded/trained."
            )

        self._validate_feature_dataframe(
            df_processed,
            allow_multiple_rows=False,
        )

        assert self.model is not None

        try:

            prediction = self.model.predict(
                df_processed
            )

            predicted_class = int(
                prediction[0]
            )

            if predicted_class not in {
                self.LEGITIMATE_CLASS,
                self.PHISHING_CLASS,
            }:

                raise ValueError(
                    f"Unexpected DNS model class: {predicted_class}"
                )

            return predicted_class

        except Exception as exc:

            logger.exception(
                "DNS class prediction failed: %s",
                exc,
            )

            raise

    # =========================================================================
    # 16. SINGLE PREDICTION RESULT
    # =========================================================================

    def predict_result(
        self,
        df_processed: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Return a structured prediction result.

        Example result:

            {
                "probability": 0.82,
                "prediction": 1,
                "class_name": "phishing"
            }

        This result can later be consumed by risk_score.py.
        """

        probability = self.predict_proba(
            df_processed
        )

        predicted_class = self.predict(
            df_processed
        )

        return {
            "probability": probability,
            "prediction": predicted_class,
            "class_name": self.CLASS_NAMES.get(
                predicted_class,
                "unknown",
            ),
        }

    # =========================================================================
    # 17. BATCH PROBABILITY PREDICTION
    # =========================================================================

    def predict_proba_batch(
        self,
        df_processed: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict phishing probabilities for multiple DNS records.

        Returns
        -------
        numpy.ndarray
            One probability per row.
        """

        if not self.is_available():

            raise RuntimeError(
                "DNS model is not loaded/trained."
            )

        self._validate_feature_dataframe(
            df_processed,
            allow_multiple_rows=True,
        )

        assert self.model is not None

        try:

            probabilities = (
                self.model.predict_proba(
                    df_processed
                )
            )

            if probabilities.ndim != 2:

                raise ValueError(
                    "Unexpected probability matrix shape."
                )

            if probabilities.shape[1] < 2:

                raise ValueError(
                    "Binary model did not return two class probabilities."
                )

            phishing_probabilities = (
                probabilities[:, self.PHISHING_CLASS]
            )

            phishing_probabilities = np.clip(
                phishing_probabilities,
                0.0,
                1.0,
            )

            return phishing_probabilities.astype(
                float
            )

        except Exception as exc:

            logger.exception(
                "DNS batch probability prediction failed: %s",
                exc,
            )

            raise

    # =========================================================================
    # 18. BATCH CLASS PREDICTION
    # =========================================================================

    def predict_batch(
        self,
        df_processed: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict binary classes for a batch.
        """

        if not self.is_available():

            raise RuntimeError(
                "DNS model is not loaded/trained."
            )

        self._validate_feature_dataframe(
            df_processed,
            allow_multiple_rows=True,
        )

        assert self.model is not None

        predictions = self.model.predict(
            df_processed
        )

        return predictions.astype(
            int
        )

    # =========================================================================
    # 19. MODEL METADATA
    # =========================================================================

    def get_model_metadata(
        self,
    ) -> Dict[str, Any]:
        """
        Return serializable metadata describing the DNS model.

        This metadata is useful for:

            - debugging
            - experiment tracking
            - audit logs
            - reports
            - model management
        """

        return {
            "agent": "DNS Security AI Agent",
            "model_type": "XGBoost",
            "model_wrapper_version": MODEL_WRAPPER_VERSION,
            "schema_version": SCHEMA_VERSION,
            "feature_count": DNSFeatureSchema.feature_count(),
            "features": DNSFeatureSchema.get_schema(),
            "target": {
                "0": "legitimate",
                "1": "phishing",
            },
            "hyperparameters": dict(
                self.hyperparameters
            ),
            "model_path": str(
                self.model_path
            ),
            "is_loaded": self.is_loaded,
            "is_trained": self.is_trained,
        }

    # =========================================================================
    # 20. MODEL STATUS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Return the current model health/status.
        """

        model_exists = (
            self.model_path.exists()
        )

        return {
            "available": self.is_available(),
            "is_loaded": self.is_loaded,
            "is_trained": self.is_trained,
            "model_exists_on_disk": model_exists,
            "model_path": str(
                self.model_path
            ),
            "schema_version": SCHEMA_VERSION,
            "model_wrapper_version": MODEL_WRAPPER_VERSION,
            "feature_count": DNSFeatureSchema.feature_count(),
        }

    # =========================================================================
    # 21. MODEL FEATURE NAMES
    # =========================================================================

    def get_feature_names(self) -> List[str]:
        """
        Return the exact feature order expected by the model.

        The authoritative source remains DNSFeatureSchema.
        """

        return DNSFeatureSchema.get_schema()

    # =========================================================================
    # 22. MODEL FEATURE COUNT
    # =========================================================================

    def get_feature_count(self) -> int:
        """
        Return the expected DNS feature count.
        """

        return DNSFeatureSchema.feature_count()

    # =========================================================================
    # 23. MODEL PATH
    # =========================================================================

    def get_model_path(self) -> str:
        """
        Return the current model path as a string.
        """

        return str(
            self.model_path
        )

    # =========================================================================
    # 24. DELETE MODEL
    # =========================================================================

    def delete_model(
        self,
        delete_metadata: bool = True,
    ) -> bool:
        """
        Delete the model file from disk.

        This is primarily a development/maintenance utility.

        It does not silently delete a currently loaded in-memory model.
        """

        model_path = self.model_path

        if not model_path.exists():

            logger.info(
                "DNS model file does not exist: %s",
                model_path,
            )

            return False

        model_path.unlink()

        if delete_metadata:

            metadata_path = model_path.with_suffix(
                ".metadata.json"
            )

            if metadata_path.exists():

                metadata_path.unlink()

        logger.info(
            "DNS model file deleted: %s",
            model_path,
        )

        return True

    # =========================================================================
    # 25. CLEAR IN-MEMORY MODEL
    # =========================================================================

    def clear(self) -> None:
        """
        Clear the currently loaded/trained model from memory.
        """

        self.model = None
        self.is_loaded = False
        self.is_trained = False

        logger.debug(
            "DNS model cleared from memory."
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_dns_model(
    model_path: Optional[str] = None,
) -> DNSModel:
    """
    Create a DNS model wrapper.
    """

    return DNSModel(
        model_path=model_path
    )


def load_dns_model(
    model_path: Optional[str] = None,
) -> DNSModel:
    """
    Create and load a DNS model.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """

    model = DNSModel(
        model_path=model_path
    )

    loaded = model.load_model(
        strict=False
    )

    if not loaded:

        raise FileNotFoundError(
            f"DNS model could not be loaded from: "
            f"{model.get_model_path()}"
        )

    return model


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    logger.info(
        "Starting DNSModel self-test."
    )

    # =========================================================================
    # 1. Initialize
    # =========================================================================

    model_wrapper = DNSModel()

    assert (
        model_wrapper.get_feature_count()
        == 37
    )

    assert (
        len(model_wrapper.get_feature_names())
        == 37
    )

    logger.info(
        "Feature schema check: PASS"
    )

    # =========================================================================
    # 2. Build
    # =========================================================================

    model = model_wrapper.build_model()

    assert isinstance(
        model,
        xgb.XGBClassifier,
    )

    logger.info(
        "Model construction check: PASS"
    )

    # =========================================================================
    # 3. Create synthetic DNS training data
    # =========================================================================

    rng = np.random.default_rng(
        42
    )

    feature_names = (
        DNSFeatureSchema.get_schema()
    )

    number_of_samples = 100

    X = pd.DataFrame(
        rng.random(
            (
                number_of_samples,
                len(feature_names),
            )
        ),
        columns=feature_names,
    )

    # Create balanced labels.
    y = np.array(
        [
            0 if index < 50 else 1
            for index in range(
                number_of_samples
            )
        ]
    )

    # =========================================================================
    # 4. Train
    # =========================================================================

    model_wrapper.train(
        X_train=X,
        y_train=y,
        verbose=False,
    )

    assert (
        model_wrapper.is_trained
    )

    logger.info(
        "Model training check: PASS"
    )

    # =========================================================================
    # 5. Single-row prediction
    # =========================================================================

    single_row = X.iloc[
        [0]
    ]

    probability = (
        model_wrapper.predict_proba(
            single_row
        )
    )

    assert 0.0 <= probability <= 1.0

    logger.info(
        "Probability prediction check: PASS "
        "(%.6f)",
        probability,
    )

    # =========================================================================
    # 6. Class prediction
    # =========================================================================

    predicted_class = (
        model_wrapper.predict(
            single_row
        )
    )

    assert predicted_class in {
        0,
        1,
    }

    logger.info(
        "Class prediction check: PASS "
        "(%d)",
        predicted_class,
    )

    # =========================================================================
    # 7. Structured result
    # =========================================================================

    result = (
        model_wrapper.predict_result(
            single_row
        )
    )

    assert (
        "probability"
        in result
    )

    assert (
        "prediction"
        in result
    )

    assert (
        "class_name"
        in result
    )

    logger.info(
        "Structured prediction result check: PASS"
    )

    # =========================================================================
    # 8. Batch prediction
    # =========================================================================

    batch_probabilities = (
        model_wrapper.predict_proba_batch(
            X.iloc[:10]
        )
    )

    assert (
        len(batch_probabilities)
        == 10
    )

    assert np.all(
        (
            batch_probabilities >= 0.0
        )
        & (
            batch_probabilities <= 1.0
        )
    )

    logger.info(
        "Batch probability prediction check: PASS"
    )

    batch_predictions = (
        model_wrapper.predict_batch(
            X.iloc[:10]
        )
    )

    assert (
        len(batch_predictions)
        == 10
    )

    logger.info(
        "Batch class prediction check: PASS"
    )

    # =========================================================================
    # 9. Save model
    # =========================================================================

    test_directory = (
        Path(__file__).resolve().parent
        / "weights"
    )

    test_model_path = (
        test_directory
        / "dns_xgb_model_test.json"
    )

    model_wrapper.save_model(
        custom_path=str(
            test_model_path
        )
    )

    assert test_model_path.exists()

    logger.info(
        "Model save check: PASS"
    )

    # =========================================================================
    # 10. Load model
    # =========================================================================

    loaded_wrapper = DNSModel(
        model_path=str(
            test_model_path
        )
    )

    loaded = (
        loaded_wrapper.load_model(
            strict=True
        )
    )

    assert loaded
    assert loaded_wrapper.is_loaded
    assert loaded_wrapper.is_trained
    assert loaded_wrapper.model is not None

    logger.info(
        "Model load check: PASS"
    )

    # =========================================================================
    # 11. Loaded model prediction
    # =========================================================================

    loaded_probability = (
        loaded_wrapper.predict_proba(
            single_row
        )
    )

    assert (
        0.0
        <= loaded_probability
        <= 1.0
    )

    logger.info(
        "Loaded-model prediction check: PASS"
    )

    # =========================================================================
    # 12. Compare original and loaded predictions
    # =========================================================================

    difference = abs(
        probability
        - loaded_probability
    )

    assert difference < 1e-6

    logger.info(
        "Save/load prediction consistency check: PASS"
    )

    # =========================================================================
    # 13. Metadata
    # =========================================================================

    metadata = (
        loaded_wrapper.get_model_metadata()
    )

    assert (
        metadata[
            "feature_count"
        ]
        == 37
    )

    assert (
        metadata[
            "schema_version"
        ]
        == SCHEMA_VERSION
    )

    logger.info(
        "Metadata check: PASS"
    )

    # =========================================================================
    # 14. Status
    # =========================================================================

    status = (
        loaded_wrapper.get_status()
    )

    assert status[
        "available"
    ]

    logger.info(
        "Model status check: PASS"
    )

    # =========================================================================
    # 15. Clean test model files
    # =========================================================================

    try:

        loaded_wrapper.delete_model(
            delete_metadata=True
        )

        logger.info(
            "Temporary test model cleanup: PASS"
        )

    except Exception as exc:

        logger.warning(
            "Temporary model cleanup failed: %s",
            exc,
        )