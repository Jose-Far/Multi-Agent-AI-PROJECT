"""
agents/url_agent/model.py

Production-grade XGBoost model manager for the URL AI Agent.

Classification:
    0 = legitimate
    1 = phishing

Responsibilities:
    - Maintain the authoritative URL feature schema
    - Build XGBoost models
    - Validate training/inference feature compatibility
    - Save trained models with metadata
    - Load new and legacy model artifacts safely
    - Expose feature metadata to Predictor and Explainer
    - Prevent silent schema mismatches
"""

import logging
import os
from typing import Any, Dict, List, Optional

import joblib
import xgboost as xgb

from .feature_schema import URLFeatureSchema


logger = logging.getLogger(__name__)


class URLXGBoostModel:
    """
    Central model-management layer for the URL AI Agent.

    Binary classification:

        0 -> legitimate
        1 -> phishing

    The current URL schema contains exactly 24 features.

    The model manager guarantees that the same feature ordering is used
    during:

        Dataset creation
            ↓
        Training
            ↓
        Model serialization
            ↓
        Model loading
            ↓
        Live inference
    """

    # ==================================================================
    # VERSIONING
    # ==================================================================

    MODEL_VERSION = "2.0.0"

    DEFAULT_MODEL_PATH = os.path.join(
        "data",
        "models",
        "url_agent",
        "xgboost_url_model.pkl",
    )

    # ==================================================================
    # CLASSIFICATION
    # ==================================================================

    CLASS_MAPPING: Dict[int, str] = {
        0: "legitimate",
        1: "phishing",
    }

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
    ):
        """
        Initialize the URL XGBoost model manager.

        Args:
            model_path:
                Path to the serialized XGBoost model.
        """

        self.model_path = model_path

        # Loaded/trained XGBoost model.
        self.model: Optional[xgb.XGBClassifier] = None

        # --------------------------------------------------------------
        # Authoritative feature schema.
        # --------------------------------------------------------------

        self.feature_names: List[str] = list(
            URLFeatureSchema.get_schema()
        )

        self.feature_count: int = len(
            self.feature_names
        )

        self.schema_version: str = (
            URLFeatureSchema.SCHEMA_VERSION
        )

        self.is_trained: bool = False

        self.model_metadata: Dict[str, Any] = {}

        # --------------------------------------------------------------
        # XGBoost training configuration.
        # --------------------------------------------------------------

        self.hyperparameters: Dict[str, Any] = {
            "n_estimators": 500,
            "max_depth": 7,
            "learning_rate": 0.05,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "min_child_weight": 3,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        logger.info(
            "URLXGBoostModel initialized."
        )

        logger.info(
            "Model version: %s",
            self.MODEL_VERSION,
        )

        logger.info(
            "Schema version: %s",
            self.schema_version,
        )

        logger.info(
            "Feature count: %d",
            self.feature_count,
        )

        logger.info(
            "Feature order: %s",
            self.feature_names,
        )

    # ==================================================================
    # FEATURE API
    # ==================================================================

    def get_feature_names(self) -> List[str]:
        """
        Return a defensive copy of the active feature ordering.
        """

        return list(self.feature_names)

    # ==================================================================
    # FEATURE SCHEMA VALIDATION
    # ==================================================================

    def validate_feature_schema(
        self,
        feature_names: List[str],
    ) -> bool:
        """
        Validate an exact URL feature ordering.

        Both feature names and their order must match.

        Args:
            feature_names:
                Feature names used by a model/training dataset.

        Returns:
            True when compatible.

        Raises:
            TypeError:
                If feature_names is not a list.

            ValueError:
                If feature names/order differ.
        """

        if not isinstance(
            feature_names,
            list,
        ):
            raise TypeError(
                "feature_names must be a list."
            )

        expected = list(
            URLFeatureSchema.get_schema()
        )

        received = list(
            feature_names
        )

        if received == expected:
            return True

        logger.error(
            "URL feature schema mismatch."
        )

        logger.error(
            "Expected (%d): %s",
            len(expected),
            expected,
        )

        logger.error(
            "Received (%d): %s",
            len(received),
            received,
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
                "Missing features: %s",
                missing,
            )

        if extra:
            logger.error(
                "Unexpected features: %s",
                extra,
            )

        raise ValueError(
            "URL feature schema/order mismatch. "
            "Training and inference must use exactly "
            "the same 24-feature ordering."
        )

    # ==================================================================
    # MODEL CONSTRUCTION
    # ==================================================================

    def build_model(
        self,
    ) -> xgb.XGBClassifier:
        """
        Build a new untrained XGBoost classifier.
        """

        logger.info(
            "Building URL XGBoost classifier."
        )

        self.model = xgb.XGBClassifier(
            **self.hyperparameters
        )

        self.is_trained = False

        return self.model

    # ==================================================================
    # CLASS IMBALANCE
    # ==================================================================

    def set_scale_pos_weight(
        self,
        weight: float,
    ) -> None:
        """
        Configure positive-class weighting.
        """

        try:
            numeric_weight = float(
                weight
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "scale_pos_weight must be numeric."
            ) from exc

        if numeric_weight <= 0:
            raise ValueError(
                "scale_pos_weight must be greater than 0."
            )

        self.hyperparameters[
            "scale_pos_weight"
        ] = numeric_weight

        if self.model is not None:
            try:
                self.model.set_params(
                    scale_pos_weight=numeric_weight
                )
            except Exception as exc:
                logger.warning(
                    "Could not update existing model "
                    "scale_pos_weight: %s",
                    exc,
                )

        logger.info(
            "scale_pos_weight=%.6f",
            numeric_weight,
        )

    # ==================================================================
    # TRAINING FEATURE REGISTRATION
    # ==================================================================

    def register_training_features(
        self,
        feature_names: List[str],
    ) -> None:
        """
        Register the feature ordering used during training.

        The training feature list must exactly match the current schema.
        """

        self.validate_feature_schema(
            feature_names
        )

        self.feature_names = list(
            feature_names
        )

        self.feature_count = len(
            self.feature_names
        )

        self.schema_version = (
            URLFeatureSchema.SCHEMA_VERSION
        )

        logger.info(
            "Training feature schema registered."
        )

        logger.info(
            "Training feature count: %d",
            self.feature_count,
        )

    # ==================================================================
    # TRAINED STATE
    # ==================================================================

    def mark_trained(
        self,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Mark the active model as trained.
        """

        if self.model is None:
            raise RuntimeError(
                "Cannot mark model as trained because "
                "no model exists."
            )

        if feature_names is not None:
            self.register_training_features(
                feature_names
            )
        else:
            self.validate_feature_schema(
                self.feature_names
            )

        self.is_trained = True

        self.model_metadata = (
            self._build_metadata()
        )

        logger.info(
            "URL XGBoost model marked as trained."
        )

    # ==================================================================
    # METADATA
    # ==================================================================

    def _build_metadata(
        self,
    ) -> Dict[str, Any]:
        """
        Build persistent model metadata.
        """

        return {
            "model_version": self.MODEL_VERSION,
            "schema_version": self.schema_version,
            "feature_count": self.feature_count,
            "feature_names": list(
                self.feature_names
            ),
            "classification": dict(
                self.CLASS_MAPPING
            ),
            "hyperparameters": dict(
                self.hyperparameters
            ),
            "entropy_features": [
                "url_entropy",
                "domain_entropy",
                "path_entropy",
                "query_entropy",
            ],
            "feature_groups": {
                "global_url": [
                    "url_length",
                    "url_entropy",
                ],
                "domain": [
                    "domain_length",
                    "domain_entropy",
                ],
                "path": [
                    "path_length",
                    "path_entropy",
                ],
                "query": [
                    "query_length",
                    "query_entropy",
                ],
            },
        }

    # ==================================================================
    # SAVE
    # ==================================================================

    def save_model(
        self,
        model_instance: Any = None,
        custom_path: Optional[str] = None,
    ) -> bool:
        """
        Save the trained model using the metadata-rich artifact format.

        Saved structure:

            {
                "model": XGBClassifier,
                "metadata": {...},
                "feature_names": [...],
                "schema_version": "2.0.0",
                "model_version": "2.0.0",
                "is_trained": True
            }
        """

        save_path = (
            custom_path
            or self.model_path
        )

        model_to_save = (
            model_instance
            if model_instance is not None
            else self.model
        )

        if model_to_save is None:
            raise ValueError(
                "No model instance available for saving."
            )

        # --------------------------------------------------------------
        # Validate current schema before serialization.
        # --------------------------------------------------------------

        self.validate_feature_schema(
            list(self.feature_names)
        )

        # --------------------------------------------------------------
        # Validate that this is actually an XGBoost classifier when
        # possible.
        # --------------------------------------------------------------

        if not hasattr(
            model_to_save,
            "predict_proba",
        ):
            raise TypeError(
                "The supplied model does not provide "
                "predict_proba()."
            )

        self.model_metadata = (
            self._build_metadata()
        )

        self.model_metadata[
            "is_trained"
        ] = True

        payload = {
            "model": model_to_save,
            "metadata": dict(
                self.model_metadata
            ),
            "feature_names": list(
                self.feature_names
            ),
            "schema_version": (
                self.schema_version
            ),
            "model_version": (
                self.MODEL_VERSION
            ),
            "is_trained": True,
        }

        parent_directory = os.path.dirname(
            os.path.abspath(
                save_path
            )
        )

        os.makedirs(
            parent_directory,
            exist_ok=True,
        )

        try:
            joblib.dump(
                payload,
                save_path,
            )

            self.model = (
                model_to_save
            )

            self.is_trained = True

            logger.info(
                "URL model saved successfully."
            )

            logger.info(
                "Model path: %s",
                save_path,
            )

            logger.info(
                "Model size: %.2f MB",
                os.path.getsize(
                    save_path
                ) / (1024 * 1024),
            )

            logger.info(
                "Saved schema version: %s",
                self.schema_version,
            )

            logger.info(
                "Saved feature count: %d",
                self.feature_count,
            )

            return True

        except Exception as exc:
            logger.error(
                "Failed to save URL model: %s",
                exc,
                exc_info=True,
            )
            raise

    # ==================================================================
    # MODEL EXTRACTION HELPERS
    # ==================================================================

    @staticmethod
    def _extract_model_feature_names(
        model: Any,
    ) -> Optional[List[str]]:
        """
        Attempt to extract feature names directly from a loaded
        XGBoost model.

        This is useful for legacy artifacts that were saved as a raw
        XGBClassifier rather than inside our metadata-rich payload.
        """

        # --------------------------------------------------------------
        # sklearn XGBClassifier often exposes:
        #
        #   model.feature_names_in_
        # --------------------------------------------------------------

        feature_names_in = getattr(
            model,
            "feature_names_in_",
            None,
        )

        if feature_names_in is not None:
            try:
                names = [
                    str(name)
                    for name in feature_names_in
                ]

                if names:
                    return names
            except Exception:
                pass

        # --------------------------------------------------------------
        # Underlying Booster may expose feature_names.
        # --------------------------------------------------------------

        try:
            booster = model.get_booster()

            booster_names = getattr(
                booster,
                "feature_names",
                None,
            )

            if booster_names:
                return [
                    str(name)
                    for name in booster_names
                ]

        except Exception:
            pass

        return None

    @staticmethod
    def _extract_model_feature_count(
        model: Any,
    ) -> Optional[int]:
        """
        Determine the number of features expected by a loaded model.
        """

        # --------------------------------------------------------------
        # XGBoost sklearn wrapper.
        # --------------------------------------------------------------

        n_features_in = getattr(
            model,
            "n_features_in_",
            None,
        )

        if n_features_in is not None:
            try:
                return int(
                    n_features_in
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

        # --------------------------------------------------------------
        # XGBoost Booster.
        # --------------------------------------------------------------

        try:
            booster = model.get_booster()

            num_features = getattr(
                booster,
                "num_features",
                None,
            )

            if callable(
                num_features
            ):
                return int(
                    num_features()
                )

        except Exception:
            pass

        # --------------------------------------------------------------
        # XGBoost sklearn internal fallback.
        # --------------------------------------------------------------

        try:
            return int(
                model.get_booster().num_features()
            )
        except Exception:
            return None

    # ==================================================================
    # LEGACY MODEL HANDLING
    # ==================================================================

    def _build_legacy_metadata(
        self,
        loaded_model: Any,
        inferred_features: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        Build metadata for a compatible legacy raw XGBoost model.

        IMPORTANT:

        We only accept a legacy model automatically when we can prove
        that it uses the current 24-feature schema.

        We never blindly assume that an arbitrary old model is compatible.
        """

        if inferred_features is not None:

            self.validate_feature_schema(
                list(inferred_features)
            )

            feature_names = list(
                inferred_features
            )

        else:

            feature_count = (
                self._extract_model_feature_count(
                    loaded_model
                )
            )

            if feature_count is None:
                raise ValueError(
                    "Legacy URL model does not expose "
                    "feature names or feature count. "
                    "Retrain the URL model using the "
                    "current training pipeline."
                )

            if feature_count != self.feature_count:
                raise ValueError(
                    "Legacy URL model feature count is "
                    f"{feature_count}, but the current URL "
                    f"schema requires {self.feature_count} features. "
                    "Retrain the URL model."
                )

            # ----------------------------------------------------------
            # This is safe ONLY because the current training pipeline
            # defines the exact authoritative order.
            # ----------------------------------------------------------

            feature_names = list(
                self.feature_names
            )

            logger.warning(
                "Legacy URL model does not contain explicit "
                "feature names, but its feature count matches "
                "the current %d-feature schema.",
                self.feature_count,
            )

        metadata = (
            self._build_metadata()
        )

        metadata.update(
            {
                "legacy_artifact": True,
                "feature_names": list(
                    feature_names
                ),
                "feature_count": len(
                    feature_names
                ),
                "schema_version": (
                    self.schema_version
                ),
            }
        )

        return metadata

    # ==================================================================
    # LOAD
    # ==================================================================

    def load_model(
        self,
        custom_path: Optional[str] = None,
    ) -> Any:
        """
        Load a URL XGBoost model.

        Supports:

        1. New metadata-rich artifact.
        2. Legacy raw XGBClassifier artifact.

        A model is accepted only when it is compatible with the current
        24-feature schema.
        """

        load_path = (
            custom_path
            or self.model_path
        )

        if not os.path.isfile(
            load_path
        ):
            raise FileNotFoundError(
                f"URL model not found: {load_path}"
            )

        logger.info(
            "Loading URL model from: %s",
            load_path,
        )

        try:
            payload = joblib.load(
                load_path
            )

            # ==========================================================
            # DETECT ARTIFACT FORMAT
            # ==========================================================

            is_metadata_artifact = (
                isinstance(
                    payload,
                    dict,
                )
                and
                "model" in payload
            )

            if is_metadata_artifact:

                loaded_model = (
                    payload["model"]
                )

                metadata = payload.get(
                    "metadata",
                    {},
                )

                saved_features = (
                    metadata.get(
                        "feature_names"
                    )
                    or
                    payload.get(
                        "feature_names"
                    )
                )

                saved_schema_version = (
                    metadata.get(
                        "schema_version"
                    )
                    or
                    payload.get(
                        "schema_version"
                    )
                )

                artifact_type = (
                    "metadata_rich"
                )

            else:

                # ------------------------------------------------------
                # Legacy raw XGBoost artifact.
                # ------------------------------------------------------

                loaded_model = (
                    payload
                )

                metadata = {}

                saved_features = None

                saved_schema_version = None

                artifact_type = (
                    "legacy_raw_model"
                )

                logger.warning(
                    "Loaded legacy URL model artifact "
                    "without wrapper metadata."
                )

            # ==========================================================
            # MODEL TYPE VALIDATION
            # ==========================================================

            if not hasattr(
                loaded_model,
                "predict",
            ):
                raise TypeError(
                    "Loaded URL artifact does not provide "
                    "predict()."
                )

            if not hasattr(
                loaded_model,
                "predict_proba",
            ):
                raise TypeError(
                    "Loaded URL artifact does not provide "
                    "predict_proba()."
                )

            # ==========================================================
            # METADATA-RICH ARTIFACT
            # ==========================================================

            if saved_features is not None:

                saved_features = [
                    str(feature)
                    for feature in saved_features
                ]

                self.validate_feature_schema(
                    saved_features
                )

                current_schema_version = (
                    URLFeatureSchema.SCHEMA_VERSION
                )

                if (
                    saved_schema_version
                    and
                    str(
                        saved_schema_version
                    )
                    != str(
                        current_schema_version
                    )
                ):
                    raise ValueError(
                        "URL model schema version mismatch. "
                        f"Model={saved_schema_version}, "
                        f"Current={current_schema_version}. "
                        "Retrain the URL model."
                    )

                active_features = (
                    saved_features
                )

                active_schema_version = (
                    saved_schema_version
                    or
                    current_schema_version
                )

                self.model_metadata = (
                    dict(metadata)
                    if metadata
                    else self._build_metadata()
                )

                self.model_metadata[
                    "artifact_type"
                ] = artifact_type

            # ==========================================================
            # LEGACY RAW MODEL
            # ==========================================================

            else:

                inferred_features = (
                    self._extract_model_feature_names(
                        loaded_model
                    )
                )

                self.model_metadata = (
                    self._build_legacy_metadata(
                        loaded_model,
                        inferred_features,
                    )
                )

                active_features = list(
                    self.model_metadata[
                        "feature_names"
                    ]
                )

                active_schema_version = (
                    self.schema_version
                )

                self.model_metadata[
                    "artifact_type"
                ] = artifact_type

            # ==========================================================
            # FINAL MODEL FEATURE COUNT CHECK
            # ==========================================================

            model_feature_count = (
                self._extract_model_feature_count(
                    loaded_model
                )
            )

            if (
                model_feature_count is not None
                and
                model_feature_count
                != len(active_features)
            ):
                raise ValueError(
                    "Loaded URL model feature count mismatch. "
                    f"Model expects {model_feature_count}, "
                    f"but metadata/schema contains "
                    f"{len(active_features)} features."
                )

            # ==========================================================
            # REGISTER LOADED MODEL
            # ==========================================================

            self.model = (
                loaded_model
            )

            self.feature_names = list(
                active_features
            )

            self.feature_count = len(
                self.feature_names
            )

            self.schema_version = (
                active_schema_version
            )

            self.is_trained = True

            # ==========================================================
            # FINAL SCHEMA VALIDATION
            # ==========================================================

            self.validate_feature_schema(
                list(self.feature_names)
            )

            logger.info(
                "URL model loaded successfully."
            )

            logger.info(
                "Artifact type: %s",
                artifact_type,
            )

            logger.info(
                "Loaded schema version: %s",
                self.schema_version,
            )

            logger.info(
                "Loaded feature count: %d",
                self.feature_count,
            )

            logger.info(
                "Loaded feature order: %s",
                self.feature_names,
            )

            logger.info(
                "URL model is ready for ML inference."
            )

            return self.model

        except (
            FileNotFoundError,
            TypeError,
            ValueError,
        ):
            raise

        except Exception as exc:

            logger.error(
                "Critical URL model loading error: %s",
                exc,
                exc_info=True,
            )

            raise RuntimeError(
                "Failed to load URL XGBoost model."
            ) from exc

    # ==================================================================
    # METADATA
    # ==================================================================

    def get_model_metadata(
        self,
    ) -> Dict[str, Any]:
        """
        Return a copy of model metadata.
        """

        if not self.model_metadata:

            self.model_metadata = (
                self._build_metadata()
            )

        return dict(
            self.model_metadata
        )

    # ==================================================================
    # HYPERPARAMETERS
    # ==================================================================

    def get_hyperparameters(
        self,
    ) -> Dict[str, Any]:
        """
        Return active XGBoost hyperparameters.
        """

        return dict(
            self.hyperparameters
        )