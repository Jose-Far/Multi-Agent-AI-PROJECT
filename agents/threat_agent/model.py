"""
Threat Intelligence AI Agent - XGBoost Model Wrapper
=====================================================

This module manages the XGBoost model lifecycle for the Threat Intelligence
AI Agent.

Responsibilities
----------------
1. Define the XGBoost binary-classification configuration.
2. Build fresh XGBoost classifiers.
3. Save trained model artifacts safely.
4. Load trained model artifacts safely.
5. Validate model/schema compatibility.
6. Preserve deterministic feature ordering.
7. Expose model configuration to the trainer.
8. Provide model/artifact metadata.
9. Prevent incompatible model artifacts from being used.

Classification contract
-----------------------

    Class 0 -> clean / benign
    Class 1 -> malicious / threat-associated

Feature contract
----------------

The model MUST consume exactly the features defined by:

    ThreatFeatureSchema.get_schema_columns()

ThreatFeatureSchema is the single source of truth.

This module does NOT maintain a second independent feature-order list.

Artifact format
---------------

The serialized artifact is a joblib file containing:

    {
        "artifact_type": "ThreatXGBoostModel",
        "artifact_version": "...",
        "model": XGBClassifier,
        "feature_order": [...],
        "feature_count": ...,
        "feature_order_hash": "...",
        "model_hyperparameters": {...},
        "classification": {...},
        "schema_metadata": {...},
        "runtime_metadata": {...}
    }

Backward compatibility
----------------------

Legacy artifacts containing a direct XGBClassifier are supported.

No standalone execution/test hook is included.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .feature_schema import ThreatFeatureSchema


logger = logging.getLogger(__name__)


class ThreatXGBoostModel:
    """
    Schema-driven XGBoost model wrapper for the Threat Intelligence Agent.

    Classification:

        0 = clean
        1 = malicious

    Feature contract:

        ThreatFeatureSchema.get_schema_columns()

    The class is responsible only for model lifecycle and compatibility.
    """

    # =========================================================================
    # ARTIFACT CONTRACT
    # =========================================================================

    ARTIFACT_TYPE = "ThreatXGBoostModel"
    ARTIFACT_VERSION = "2.0.0"

    MODEL_FRAMEWORK = "xgboost"

    # =========================================================================
    # CLASSIFICATION CONTRACT
    # =========================================================================

    CLASS_LABELS: Dict[int, str] = {
        0: "clean",
        1: "malicious",
    }

    POSITIVE_CLASS_INDEX = 1
    POSITIVE_CLASS_LABEL = "malicious"

    # =========================================================================
    # DEFAULT MODEL PATH
    # =========================================================================

    DEFAULT_MODEL_PATH = (
        "data/models/threat_agent/xgboost_threat_model.pkl"
    )

    # =========================================================================
    # CONSTRUCTOR
    # =========================================================================

    def __init__(
        self,
        model_path: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the Threat XGBoost model wrapper.

        Parameters
        ----------
        model_path:
            Optional serialized model artifact path.

        hyperparameters:
            Optional XGBoost parameter overrides.
        """

        self.model_path = (
            str(model_path)
            if model_path
            else self.DEFAULT_MODEL_PATH
        )

        self.model: Optional[
            xgb.XGBClassifier
        ] = None

        # ---------------------------------------------------------------------
        # SINGLE SOURCE OF TRUTH
        # ---------------------------------------------------------------------

        self.feature_schema: List[str] = (
            ThreatFeatureSchema.get_schema_columns()
        )

        self.feature_count: int = (
            ThreatFeatureSchema.get_feature_count()
        )

        # ---------------------------------------------------------------------
        # Validate schema immediately.
        # ---------------------------------------------------------------------

        if len(self.feature_schema) != self.feature_count:
            raise RuntimeError(
                "ThreatFeatureSchema returned an inconsistent "
                "feature count."
            )

        if self.feature_count <= 0:
            raise RuntimeError(
                "ThreatFeatureSchema contains no ML features."
            )

        # ---------------------------------------------------------------------
        # Schema signature.
        # ---------------------------------------------------------------------

        self.feature_order_hash = (
            self._calculate_feature_order_hash(
                self.feature_schema
            )
        )

        # ---------------------------------------------------------------------
        # Model configuration.
        # ---------------------------------------------------------------------

        self.hyperparameters: Dict[str, Any] = (
            self._build_default_hyperparameters()
        )

        if hyperparameters:
            validated = (
                self._validate_hyperparameter_overrides(
                    hyperparameters
                )
            )

            self.hyperparameters.update(
                validated
            )

        logger.info(
            "ThreatXGBoostModel initialized | "
            "artifact_version=%s | features=%d | "
            "schema_hash=%s | model_path=%s",
            self.ARTIFACT_VERSION,
            self.feature_count,
            self.feature_order_hash[:12],
            self.model_path,
        )

    # =========================================================================
    # DEFAULT HYPERPARAMETERS
    # =========================================================================

    @staticmethod
    def _build_default_hyperparameters() -> Dict[str, Any]:
        """
        Return the baseline XGBoost configuration.

        These parameters are intentionally kept independent from the
        ThreatFeatureSchema because model hyperparameters are not feature
        definitions.
        """

        return {
            "n_estimators": 250,
            "max_depth": 6,
            "learning_rate": 0.05,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "verbosity": 0,
        }

    # =========================================================================
    # HYPERPARAMETER VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_hyperparameter_overrides(
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate obvious invalid XGBoost hyperparameter values.

        XGBoost remains responsible for the complete parameter contract.
        """

        if not isinstance(parameters, dict):
            raise TypeError(
                "XGBoost hyperparameter overrides must be a dictionary."
            )

        validated = dict(parameters)

        # ---------------------------------------------------------------------
        # Integer parameters.
        # ---------------------------------------------------------------------

        integer_parameters = {
            "n_estimators",
            "max_depth",
            "n_jobs",
            "min_child_weight",
            "random_state",
        }

        for name in integer_parameters:

            if name not in validated:
                continue

            try:
                value = int(
                    validated[name]
                )
            except (
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    f"XGBoost parameter '{name}' "
                    "must be an integer."
                ) from exc

            if name in {
                "n_estimators",
                "max_depth",
                "min_child_weight",
            } and value < 1:

                raise ValueError(
                    f"XGBoost parameter '{name}' must be >= 1."
                )

            validated[name] = value

        # ---------------------------------------------------------------------
        # Learning rate.
        # ---------------------------------------------------------------------

        if "learning_rate" in validated:

            try:
                value = float(
                    validated["learning_rate"]
                )
            except (
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    "learning_rate must be numeric."
                ) from exc

            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "learning_rate must be finite and > 0."
                )

            validated["learning_rate"] = value

        # ---------------------------------------------------------------------
        # Subsampling parameters.
        # ---------------------------------------------------------------------

        for name in {
            "subsample",
            "colsample_bytree",
        }:

            if name not in validated:
                continue

            try:
                value = float(
                    validated[name]
                )
            except (
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    f"{name} must be numeric."
                ) from exc

            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite."
                )

            if not 0.0 < value <= 1.0:
                raise ValueError(
                    f"{name} must be in the interval (0, 1]."
                )

            validated[name] = value

        return validated

    # =========================================================================
    # MODEL BUILDING
    # =========================================================================

    def build_model(self) -> xgb.XGBClassifier:
        """
        Build a fresh untrained XGBoost classifier.

        This method does not train the model.
        """

        logger.info(
            "Building Threat XGBoost classifier | "
            "features=%d | estimators=%s | depth=%s",
            self.feature_count,
            self.hyperparameters.get(
                "n_estimators"
            ),
            self.hyperparameters.get(
                "max_depth"
            ),
        )

        self.model = xgb.XGBClassifier(
            **self.hyperparameters
        )

        return self.model

    # =========================================================================
    # MODEL STATE
    # =========================================================================

    def has_model(self) -> bool:
        """
        Return True when an XGBoost model object is registered.
        """

        return self.model is not None

    def is_fitted(self) -> bool:
        """
        Return True when the registered XGBoost model has a fitted booster.
        """

        if self.model is None:
            return False

        try:
            self.model.get_booster()
            return True
        except Exception:
            return False

    # =========================================================================
    # FEATURE CONTRACT
    # =========================================================================

    def get_feature_schema(self) -> List[str]:
        """
        Return the exact canonical Threat feature order.
        """

        return list(
            ThreatFeatureSchema.get_schema_columns()
        )

    def get_feature_count(self) -> int:
        """
        Return the exact number of canonical Threat ML features.
        """

        return (
            ThreatFeatureSchema.get_feature_count()
        )

    def get_feature_order_hash(self) -> str:
        """
        Return the deterministic hash of the current feature ordering.
        """

        return self.feature_order_hash

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Return a copy of the model hyperparameters.
        """

        return dict(
            self.hyperparameters
        )

    def get_model_type(self) -> str:
        """
        Return the ML framework identifier.
        """

        return self.MODEL_FRAMEWORK

    def get_class_labels(self) -> Dict[int, str]:
        """
        Return the classification mapping.
        """

        return dict(
            self.CLASS_LABELS
        )

    # =========================================================================
    # FEATURE ORDER HASH
    # =========================================================================

    @staticmethod
    def _calculate_feature_order_hash(
        feature_order: List[str],
    ) -> str:
        """
        Calculate SHA-256 over the canonical feature ordering.
        """

        serialized = "|".join(
            feature_order
        ).encode("utf-8")

        return hashlib.sha256(
            serialized
        ).hexdigest()

    # =========================================================================
    # MODEL VALIDATION
    # =========================================================================

    def validate_model(
        self,
        model: Optional[xgb.XGBClassifier] = None,
    ) -> Dict[str, Any]:
        """
        Validate an XGBoost model against the Threat ML contract.

        Checks:

        - XGBClassifier type
        - fitted state
        - feature count
        - probability prediction support
        - binary objective
        """

        candidate = (
            model
            if model is not None
            else self.model
        )

        result: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        if candidate is None:

            result["valid"] = False

            result["errors"].append(
                "No XGBoost model is loaded."
            )

            return result

        if not isinstance(
            candidate,
            xgb.XGBClassifier,
        ):

            result["valid"] = False

            result["errors"].append(
                "Model is not an XGBClassifier."
            )

            return result

        # ---------------------------------------------------------------------
        # Fitted state.
        # ---------------------------------------------------------------------

        try:
            booster = (
                candidate.get_booster()
            )
        except Exception:

            result["valid"] = False

            result["errors"].append(
                "XGBoost model is not fitted."
            )

            return result

        # ---------------------------------------------------------------------
        # Feature count.
        # ---------------------------------------------------------------------

        try:

            booster_feature_count = int(
                booster.num_features()
            )

            if (
                booster_feature_count
                != self.feature_count
            ):

                result["valid"] = False

                result["errors"].append(
                    "Model feature count mismatch: "
                    f"model={booster_feature_count}, "
                    f"schema={self.feature_count}."
                )

        except Exception as exc:

            result["warnings"].append(
                "Unable to inspect booster feature count: "
                f"{exc}"
            )

        # ---------------------------------------------------------------------
        # Probability prediction.
        # ---------------------------------------------------------------------

        if not callable(
            getattr(
                candidate,
                "predict_proba",
                None,
            )
        ):

            result["valid"] = False

            result["errors"].append(
                "Model does not expose predict_proba()."
            )

        # ---------------------------------------------------------------------
        # Objective.
        # ---------------------------------------------------------------------

        try:

            objective = (
                candidate
                .get_xgb_params()
                .get("objective")
            )

            if objective not in {
                "binary:logistic",
                "binary:logitraw",
            }:

                result["warnings"].append(
                    "Unexpected model objective: "
                    f"{objective!r}"
                )

        except Exception as exc:

            result["warnings"].append(
                "Unable to inspect XGBoost objective: "
                f"{exc}"
            )

        return result

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    def validate_prediction_input(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Strictly validate model input.

        The DataFrame must contain exactly the 20 canonical Threat features
        in exactly the schema-defined order.
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            raise TypeError(
                "Threat model input must be a pandas DataFrame."
            )

        if dataframe.empty:

            raise ValueError(
                "Threat model input DataFrame is empty."
            )

        expected_columns = (
            ThreatFeatureSchema.get_schema_columns()
        )

        actual_columns = list(
            dataframe.columns
        )

        # ---------------------------------------------------------------------
        # Exact feature order.
        # ---------------------------------------------------------------------

        if actual_columns != expected_columns:

            raise ValueError(
                "Threat model feature-order mismatch.\n"
                f"Expected: {expected_columns}\n"
                f"Received: {actual_columns}"
            )

        # ---------------------------------------------------------------------
        # Exact feature count.
        # ---------------------------------------------------------------------

        expected_count = (
            ThreatFeatureSchema.get_feature_count()
        )

        if len(dataframe.columns) != expected_count:

            raise ValueError(
                "Threat model feature-count mismatch: "
                f"received={len(dataframe.columns)}, "
                f"expected={expected_count}."
            )

        # ---------------------------------------------------------------------
        # No duplicate columns.
        # ---------------------------------------------------------------------

        if dataframe.columns.duplicated().any():

            raise ValueError(
                "Threat model input contains duplicate feature columns."
            )

        # ---------------------------------------------------------------------
        # No NaN.
        # ---------------------------------------------------------------------

        if dataframe.isna().any().any():

            raise ValueError(
                "Threat model input contains NaN values."
            )

        # ---------------------------------------------------------------------
        # All numeric.
        # ---------------------------------------------------------------------

        non_numeric = [
            column
            for column in dataframe.columns
            if not pd.api.types.is_numeric_dtype(
                dataframe[column]
            )
        ]

        if non_numeric:

            raise TypeError(
                "Threat model input contains non-numeric "
                f"features: {non_numeric}"
            )

        # ---------------------------------------------------------------------
        # Finite values.
        # ---------------------------------------------------------------------

        values = dataframe.to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(
            values
        ).all():

            raise ValueError(
                "Threat model input contains "
                "non-finite values."
            )

    # =========================================================================
    # SAVE MODEL
    # =========================================================================

    def save_model(
        self,
        custom_path: Optional[str] = None,
    ) -> bool:
        """
        Save a fitted Threat XGBoost model as a structured joblib artifact.

        Saving is atomic.
        """

        if self.model is None:

            raise ValueError(
                "No XGBoost model exists."
            )

        validation = (
            self.validate_model()
        )

        if not validation["valid"]:

            raise ValueError(
                "Refusing to save invalid Threat model: "
                + "; ".join(
                    validation["errors"]
                )
            )

        save_path = (
            str(custom_path)
            if custom_path
            else self.model_path
        )

        destination = (
            Path(save_path)
            .expanduser()
            .resolve()
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        artifact = (
            self._build_artifact()
        )

        temporary_path: Optional[
            Path
        ] = None

        try:

            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.stem}_",
                suffix=".tmp",
                dir=str(
                    destination.parent
                ),
                delete=False,
            ) as temporary_file:

                temporary_path = Path(
                    temporary_file.name
                )

                joblib.dump(
                    artifact,
                    temporary_file,
                    compress=3,
                )

                temporary_file.flush()

                os.fsync(
                    temporary_file.fileno()
                )

            os.replace(
                temporary_path,
                destination,
            )

            temporary_path = None

            logger.info(
                "Threat XGBoost artifact saved: %s",
                destination,
            )

            return True

        except Exception as exc:

            logger.error(
                "Failed to save Threat model: %s",
                exc,
                exc_info=True,
            )

            raise

        finally:

            if (
                temporary_path is not None
                and temporary_path.exists()
            ):

                try:
                    temporary_path.unlink()
                except OSError:
                    logger.warning(
                        "Unable to remove temporary "
                        "Threat model artifact: %s",
                        temporary_path,
                    )

    # =========================================================================
    # ARTIFACT CREATION
    # =========================================================================

    def _build_artifact(
        self,
    ) -> Dict[str, Any]:
        """
        Build the structured serialized artifact.
        """

        if self.model is None:

            raise ValueError(
                "Cannot build an artifact without a model."
            )

        return {
            "artifact_type": (
                self.ARTIFACT_TYPE
            ),

            "artifact_version": (
                self.ARTIFACT_VERSION
            ),

            "created_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),

            "model_framework": (
                self.MODEL_FRAMEWORK
            ),

            "model": self.model,

            "feature_order": list(
                self.feature_schema
            ),

            "feature_count": (
                self.feature_count
            ),

            "feature_order_hash": (
                self.feature_order_hash
            ),

            "model_hyperparameters": (
                dict(
                    self.hyperparameters
                )
            ),

            "classification": {
                "type": "binary",
                "classes": dict(
                    self.CLASS_LABELS
                ),
                "positive_class_index": (
                    self.POSITIVE_CLASS_INDEX
                ),
                "positive_class_label": (
                    self.POSITIVE_CLASS_LABEL
                ),
            },

            "schema_metadata": (
                self._get_schema_metadata()
            ),

            "runtime_metadata": (
                self._get_runtime_metadata()
            ),
        }

    # =========================================================================
    # SCHEMA METADATA
    # =========================================================================

    def _get_schema_metadata(
        self,
    ) -> Dict[str, Any]:
        """
        Return schema metadata for model compatibility checks.
        """

        metadata: Dict[str, Any] = {
            "feature_count": (
                self.feature_count
            ),

            "feature_order": list(
                self.feature_schema
            ),

            "feature_order_hash": (
                self.feature_order_hash
            ),
        }

        schema_version = getattr(
            ThreatFeatureSchema,
            "SCHEMA_VERSION",
            None,
        )

        if schema_version is not None:

            metadata["schema_version"] = (
                schema_version
            )

        # Include schema signature when available.
        get_signature = getattr(
            ThreatFeatureSchema,
            "get_schema_signature",
            None,
        )

        if callable(
            get_signature
        ):

            try:
                metadata[
                    "schema_signature"
                ] = get_signature()

            except Exception as exc:

                logger.warning(
                    "Unable to obtain Threat schema signature: %s",
                    exc,
                )

        return metadata

    # =========================================================================
    # RUNTIME METADATA
    # =========================================================================

    def _get_runtime_metadata(
        self,
    ) -> Dict[str, Any]:
        """
        Return reproducibility/runtime metadata.
        """

        return {
            "python_version": (
                platform.python_version()
            ),

            "platform": (
                platform.platform()
            ),

            "xgboost_version": getattr(
                xgb,
                "__version__",
                "unknown",
            ),

            "numpy_version": (
                np.__version__
            ),

            "pandas_version": (
                pd.__version__
            ),
        }

    # =========================================================================
    # LOAD MODEL
    # =========================================================================

    def load_model(
        self,
        custom_path: Optional[str] = None,
    ) -> xgb.XGBClassifier:
        """
        Load and validate a Threat XGBoost model artifact.

        Raises
        ------
        FileNotFoundError
            Artifact does not exist.

        TypeError
            Unsupported artifact.

        ValueError
            Incompatible artifact/model.
        """

        load_path = (
            str(custom_path)
            if custom_path
            else self.model_path
        )

        path = (
            Path(load_path)
            .expanduser()
            .resolve()
        )

        if not path.exists():

            logger.error(
                "Threat model artifact not found: %s",
                path,
            )

            raise FileNotFoundError(
                f"Threat model artifact does not exist: {path}"
            )

        if not path.is_file():

            raise ValueError(
                f"Threat model path is not a file: {path}"
            )

        try:

            logger.info(
                "Loading Threat XGBoost model: %s",
                path,
            )

            artifact = joblib.load(
                path
            )

            model, metadata = (
                self._extract_model_from_artifact(
                    artifact
                )
            )

            # -------------------------------------------------------------
            # Validate serialized schema metadata.
            # -------------------------------------------------------------

            self._validate_artifact_metadata(
                metadata
            )

            # -------------------------------------------------------------
            # Validate actual XGBoost model.
            # -------------------------------------------------------------

            validation = (
                self.validate_model(
                    model
                )
            )

            if not validation["valid"]:

                raise ValueError(
                    "Loaded Threat model failed validation: "
                    + "; ".join(
                        validation["errors"]
                    )
                )

            # -------------------------------------------------------------
            # Register only after validation succeeds.
            # -------------------------------------------------------------

            self.model = model

            if validation["warnings"]:

                logger.warning(
                    "Threat model load warnings: %s",
                    validation["warnings"],
                )

            logger.info(
                "Threat XGBoost model loaded successfully | "
                "features=%d | schema_hash=%s",
                self.feature_count,
                self.feature_order_hash[:12],
            )

            return self.model

        except FileNotFoundError:
            raise

        except Exception as exc:

            logger.error(
                "Failed to load Threat model from %s: %s",
                path,
                exc,
                exc_info=True,
            )

            raise

    # =========================================================================
    # ARTIFACT EXTRACTION
    # =========================================================================

    def _extract_model_from_artifact(
        self,
        artifact: Any,
    ) -> Tuple[
        xgb.XGBClassifier,
        Dict[str, Any],
    ]:
        """
        Extract the XGBClassifier and metadata from a serialized artifact.

        Supports:

        1. Structured ThreatXGBoostModel artifact.
        2. Legacy direct XGBClassifier artifact.
        """

        # ---------------------------------------------------------------------
        # Legacy direct model.
        # ---------------------------------------------------------------------

        if isinstance(
            artifact,
            xgb.XGBClassifier,
        ):

            logger.warning(
                "Loaded legacy direct-XGBClassifier Threat artifact. "
                "Schema metadata is unavailable."
            )

            return artifact, {
                "legacy_artifact": True,
                "feature_order": None,
                "feature_count": None,
                "feature_order_hash": None,
            }

        # ---------------------------------------------------------------------
        # Structured artifact.
        # ---------------------------------------------------------------------

        if not isinstance(
            artifact,
            dict,
        ):

            raise TypeError(
                "Unsupported Threat model artifact type: "
                f"{type(artifact).__name__}"
            )

        artifact_type = artifact.get(
            "artifact_type"
        )

        if artifact_type != (
            self.ARTIFACT_TYPE
        ):

            raise TypeError(
                "Invalid Threat artifact type: "
                f"{artifact_type!r}"
            )

        model = artifact.get(
            "model"
        )

        if not isinstance(
            model,
            xgb.XGBClassifier,
        ):

            raise TypeError(
                "Threat artifact does not contain "
                "a valid XGBClassifier."
            )

        return model, {
            "legacy_artifact": False,

            "artifact_type": artifact_type,

            "artifact_version": (
                artifact.get(
                    "artifact_version"
                )
            ),

            "feature_order": (
                artifact.get(
                    "feature_order"
                )
            ),

            "feature_count": (
                artifact.get(
                    "feature_count"
                )
            ),

            "feature_order_hash": (
                artifact.get(
                    "feature_order_hash"
                )
            ),

            "schema_metadata": (
                artifact.get(
                    "schema_metadata",
                    {},
                )
            ),

            "classification": (
                artifact.get(
                    "classification",
                    {},
                )
            ),

            "model_hyperparameters": (
                artifact.get(
                    "model_hyperparameters",
                    {},
                )
            ),
        }

    # =========================================================================
    # ARTIFACT METADATA VALIDATION
    # =========================================================================

    def _validate_artifact_metadata(
        self,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Validate serialized model metadata against the current schema.

        Schema mismatches are hard failures.

        A model trained against a different feature order must never be
        silently accepted.
        """

        if metadata.get(
            "legacy_artifact",
            False,
        ):
            return

        # ---------------------------------------------------------------------
        # Feature count.
        # ---------------------------------------------------------------------

        stored_count = metadata.get(
            "feature_count"
        )

        if stored_count is not None:

            try:
                stored_count = int(
                    stored_count
                )
            except (
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    "Threat model artifact contains "
                    "an invalid feature_count."
                ) from exc

            if stored_count != (
                self.feature_count
            ):

                raise ValueError(
                    "Threat model feature-count mismatch: "
                    f"artifact={stored_count}, "
                    f"current_schema={self.feature_count}."
                )

        # ---------------------------------------------------------------------
        # Feature order.
        # ---------------------------------------------------------------------

        stored_order = metadata.get(
            "feature_order"
        )

        if stored_order is not None:

            if list(stored_order) != (
                self.feature_schema
            ):

                raise ValueError(
                    "Threat model feature-order mismatch. "
                    "The model was trained against a different "
                    "ThreatFeatureSchema."
                )

        # ---------------------------------------------------------------------
        # Feature-order hash.
        # ---------------------------------------------------------------------

        stored_hash = metadata.get(
            "feature_order_hash"
        )

        if stored_hash is not None:

            if stored_hash != (
                self.feature_order_hash
            ):

                raise ValueError(
                    "Threat model feature-order hash mismatch."
                )

        # ---------------------------------------------------------------------
        # Nested schema metadata.
        # ---------------------------------------------------------------------

        schema_metadata = metadata.get(
            "schema_metadata",
            {},
        )

        if isinstance(
            schema_metadata,
            dict,
        ):

            nested_hash = (
                schema_metadata.get(
                    "feature_order_hash"
                )
            )

            if (
                nested_hash is not None
                and nested_hash
                != self.feature_order_hash
            ):

                raise ValueError(
                    "Threat model nested schema hash mismatch."
                )

        # ---------------------------------------------------------------------
        # Classification contract.
        # ---------------------------------------------------------------------

        classification = (
            metadata.get(
                "classification",
                {},
            )
        )

        if isinstance(
            classification,
            dict,
        ):

            classification_type = (
                classification.get(
                    "type"
                )
            )

            if classification_type not in {
                None,
                "binary",
            }:

                raise ValueError(
                    "Threat model artifact is not "
                    "a binary classifier."
                )

            stored_positive_index = (
                classification.get(
                    "positive_class_index"
                )
            )

            if (
                stored_positive_index is not None
                and int(
                    stored_positive_index
                )
                != self.POSITIVE_CLASS_INDEX
            ):

                raise ValueError(
                    "Threat model positive-class index mismatch."
                )

    # =========================================================================
    # MODEL INFORMATION
    # =========================================================================

    def get_model_info(
        self,
    ) -> Dict[str, Any]:
        """
        Return a diagnostic snapshot of the model and schema contract.
        """

        info: Dict[str, Any] = {
            "artifact_type": (
                self.ARTIFACT_TYPE
            ),

            "artifact_version": (
                self.ARTIFACT_VERSION
            ),

            "framework": (
                self.MODEL_FRAMEWORK
            ),

            "model_path": (
                self.model_path
            ),

            "feature_count": (
                self.feature_count
            ),

            "feature_order": list(
                self.feature_schema
            ),

            "feature_order_hash": (
                self.feature_order_hash
            ),

            "hyperparameters": dict(
                self.hyperparameters
            ),

            "classification": {
                "type": "binary",
                "classes": dict(
                    self.CLASS_LABELS
                ),
                "positive_class_index": (
                    self.POSITIVE_CLASS_INDEX
                ),
                "positive_class_label": (
                    self.POSITIVE_CLASS_LABEL
                ),
            },

            "model_loaded": (
                self.model is not None
            ),

            "model_fitted": (
                self.is_fitted()
            ),

            "schema_metadata": (
                self._get_schema_metadata()
            ),
        }

        if self.model is not None:

            try:

                info[
                    "xgboost_parameters"
                ] = (
                    self.model
                    .get_xgb_params()
                )

            except Exception:

                info[
                    "xgboost_parameters"
                ] = {}

            try:

                booster = (
                    self.model
                    .get_booster()
                )

                info[
                    "booster_feature_count"
                ] = int(
                    booster.num_features()
                )

            except Exception:

                info[
                    "booster_feature_count"
                ] = None

        return info

    # =========================================================================
    # FEATURE NAME VALIDATION
    # =========================================================================

    def validate_feature_names(
        self,
        feature_names: List[str],
    ) -> bool:
        """
        Validate an external feature list against the canonical schema.
        """

        if not isinstance(
            feature_names,
            list,
        ):

            return False

        return feature_names == (
            ThreatFeatureSchema.get_schema_columns()
        )

    # =========================================================================
    # ARTIFACT EXISTENCE
    # =========================================================================

    def model_artifact_exists(
        self,
        custom_path: Optional[str] = None,
    ) -> bool:
        """
        Return True if a model artifact exists.
        """

        path = (
            Path(
                custom_path
                if custom_path
                else self.model_path
            )
            .expanduser()
        )

        return (
            path.exists()
            and path.is_file()
        )

    # =========================================================================
    # ARTIFACT SIZE
    # =========================================================================

    def get_model_artifact_size(
        self,
        custom_path: Optional[str] = None,
    ) -> Optional[int]:
        """
        Return the serialized artifact size in bytes.
        """

        path = (
            Path(
                custom_path
                if custom_path
                else self.model_path
            )
            .expanduser()
        )

        if not path.exists():
            return None

        try:
            return int(
                path.stat().st_size
            )
        except OSError:
            return None

    # =========================================================================
    # ARTIFACT HASH
    # =========================================================================

    def get_model_artifact_hash(
        self,
        custom_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Calculate SHA-256 of the serialized model artifact.
        """

        path = (
            Path(
                custom_path
                if custom_path
                else self.model_path
            )
            .expanduser()
        )

        if not path.exists():
            return None

        sha256 = hashlib.sha256()

        try:

            with path.open(
                "rb"
            ) as model_file:

                for chunk in iter(
                    lambda: model_file.read(
                        1024 * 1024
                    ),
                    b"",
                ):

                    sha256.update(
                        chunk
                    )

            return sha256.hexdigest()

        except OSError as exc:

            logger.warning(
                "Unable to calculate Threat model hash: %s",
                exc,
            )

            return None

    # =========================================================================
    # CLEAR MODEL
    # =========================================================================

    def clear_model(self) -> None:
        """
        Clear the in-memory model.

        The serialized artifact on disk is not deleted.
        """

        self.model = None

        logger.debug(
            "Threat XGBoost in-memory model cleared."
        )
