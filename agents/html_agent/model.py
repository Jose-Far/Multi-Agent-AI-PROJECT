"""
HTML AI Agent - Model Wrapper
=============================

Responsible for loading and executing the trained XGBoost model.

Pipeline:

    HTML Feature Extraction
            ↓
    HTML Feature Schema
            ↓
    HTML Preprocessing
            ↓
    HTMLModel
            ↓
    Phishing Probability
            ↓
    HTML Predictor
            ↓
    HTML Risk Scorer
            ↓
    Decision Fusion Engine


Classification:

    0 = Legitimate
    1 = Phishing
"""

import os
import logging

from typing import Optional, Dict, Any, List

import xgboost as xgb
import pandas as pd
import numpy as np

from .feature_schema import HTMLFeatureSchema


logger = logging.getLogger(__name__)


class HTMLModel:
    """
    Core Machine Learning Model Wrapper for the HTML AI Agent.

    Responsibilities
    ----------------
    1. Locate the trained XGBoost model.
    2. Load model weights from disk.
    3. Validate model availability.
    4. Validate model compatibility with the feature schema.
    5. Validate incoming feature matrices.
    6. Execute probability inference.
    7. Return phishing probability.
    8. Support batch inference.
    9. Provide model status and metadata.
    10. Handle inference failures safely.

    This class does NOT perform:

        - HTML extraction
        - Feature extraction
        - Preprocessing
        - Risk scoring
        - Final phishing decision
        - Multi-agent fusion

    Those responsibilities belong to other modules.
    """

    # =====================================================================
    # MODEL CONFIGURATION
    # =====================================================================

    MODEL_FILENAME = "html_xgb_model.json"

    # Binary classification
    LEGITIMATE_CLASS_INDEX = 0
    PHISHING_CLASS_INDEX = 1

    EXPECTED_CLASS_COUNT = 2

    # =====================================================================
    # INITIALIZATION
    # =====================================================================

    def __init__(
        self,
        model_path: Optional[str] = None
    ):
        """
        Initialize the HTML XGBoost model wrapper.

        Args:
            model_path:
                Optional custom path to the trained XGBoost model.

                If omitted, the default path is:

                    agents/html_agent/weights/html_xgb_model.json
        """

        # ---------------------------------------------------------------
        # Resolve model path
        # ---------------------------------------------------------------

        default_path = os.path.join(
            os.path.dirname(__file__),
            "weights",
            self.MODEL_FILENAME
        )

        self.model_path = (
            os.path.abspath(
                os.path.expanduser(
                    model_path
                )
            )
            if model_path
            else os.path.abspath(
                default_path
            )
        )

        # ---------------------------------------------------------------
        # Initialize XGBoost classifier
        # ---------------------------------------------------------------

        self.model = (
            xgb.XGBClassifier()
        )

        # ---------------------------------------------------------------
        # Model state
        # ---------------------------------------------------------------

        self.is_loaded = False

        self.load_error: Optional[str] = None

        self.model_feature_count: Optional[int] = None

        self.model_classes: Optional[List[Any]] = None

        # ---------------------------------------------------------------
        # Load model
        # ---------------------------------------------------------------

        self._load_model()

    # =====================================================================
    # MODEL LOADING
    # =====================================================================

    def _load_model(
        self
    ) -> None:
        """
        Load the trained XGBoost model from disk.

        A missing or invalid model does not crash the complete application.

        Instead:

            is_loaded = False

        The Predictor can then decide whether a fallback mechanism
        should be used.
        """

        # ---------------------------------------------------------------
        # Reset state
        # ---------------------------------------------------------------

        self.is_loaded = False

        self.load_error = None

        self.model_feature_count = None

        self.model_classes = None

        # ---------------------------------------------------------------
        # Validate model path
        # ---------------------------------------------------------------

        if not self.model_path:

            self.load_error = (
                "No HTML model path was provided."
            )

            logger.error(
                self.load_error
            )

            return

        # ---------------------------------------------------------------
        # Validate model extension
        # ---------------------------------------------------------------

        if not self.model_path.lower().endswith(
            ".json"
        ):

            logger.warning(
                "HTML model file does not use "
                "the expected .json extension: %s",
                self.model_path
            )

        # ---------------------------------------------------------------
        # Check model file
        # ---------------------------------------------------------------

        if not os.path.isfile(
            self.model_path
        ):

            self.load_error = (
                "HTML model file not found: "
                f"{self.model_path}"
            )

            logger.warning(
                "%s",
                self.load_error
            )

            return

        # ---------------------------------------------------------------
        # Check file size
        # ---------------------------------------------------------------

        try:

            file_size = os.path.getsize(
                self.model_path
            )

            if file_size <= 0:

                self.load_error = (
                    "HTML model file is empty: "
                    f"{self.model_path}"
                )

                logger.error(
                    self.load_error
                )

                return

        except OSError as e:

            self.load_error = (
                "Unable to inspect HTML model file: "
                f"{str(e)}"
            )

            logger.error(
                self.load_error
            )

            return

        # ---------------------------------------------------------------
        # Load XGBoost model
        # ---------------------------------------------------------------

        try:

            logger.info(
                "Loading HTML XGBoost model from: %s",
                self.model_path
            )

            self.model.load_model(
                self.model_path
            )

        except Exception as e:

            self.is_loaded = False

            self.load_error = (
                f"Failed to load HTML XGBoost model: {str(e)}"
            )

            logger.error(
                "%s",
                self.load_error,
                exc_info=True
            )

            return

        # ---------------------------------------------------------------
        # Validate loaded model
        # ---------------------------------------------------------------

        if not self._validate_loaded_model():

            self.is_loaded = False

            if not self.load_error:

                self.load_error = (
                    "Loaded HTML XGBoost model "
                    "failed compatibility validation."
                )

            logger.error(
                "%s",
                self.load_error
            )

            return

        # ---------------------------------------------------------------
        # Mark model as loaded
        # ---------------------------------------------------------------

        self.is_loaded = True

        self.load_error = None

        logger.info(
            "HTML XGBoost model loaded successfully."
        )

        logger.info(
            "HTML model path: %s",
            self.model_path
        )

        logger.info(
            "HTML model features: %s",
            self.model_feature_count
        )

    # =====================================================================
    # LOADED MODEL VALIDATION
    # =====================================================================

    def _validate_loaded_model(
        self
    ) -> bool:
        """
        Validate the loaded XGBoost model.

        Checks:

            1. Model object exists.
            2. Feature count matches schema.
            3. Binary classification is configured.
            4. Model can expose probability predictions.
        """

        # ---------------------------------------------------------------
        # Model object
        # ---------------------------------------------------------------

        if self.model is None:

            self.load_error = (
                "HTML model validation failed: "
                "model object is None."
            )

            logger.error(
                self.load_error
            )

            return False

        try:

            # =============================================================
            # Expected feature count
            # =============================================================

            expected_feature_count = len(
                HTMLFeatureSchema.get_schema()
            )

            # =============================================================
            # Detect model feature count
            # =============================================================

            model_feature_count = getattr(
                self.model,
                "n_features_in_",
                None
            )

            # -------------------------------------------------------------
            # Some XGBoost versions may not expose n_features_in_.
            # Try booster metadata when necessary.
            # -------------------------------------------------------------

            if model_feature_count is None:

                try:

                    booster = (
                        self.model.get_booster()
                    )

                    feature_names = (
                        booster.feature_names
                    )

                    if feature_names:

                        model_feature_count = len(
                            feature_names
                        )

                except Exception as e:

                    logger.debug(
                        "Unable to obtain feature count "
                        "from XGBoost booster: %s",
                        str(e)
                    )

            # =============================================================
            # Store feature count
            # =============================================================

            if model_feature_count is not None:

                try:

                    model_feature_count = int(
                        model_feature_count
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    model_feature_count = None

            self.model_feature_count = (
                model_feature_count
            )

            # =============================================================
            # Feature count validation
            # =============================================================

            if (
                model_feature_count is not None
                and
                model_feature_count
                != expected_feature_count
            ):

                self.load_error = (
                    "HTML model feature mismatch. "
                    f"Schema expects "
                    f"{expected_feature_count} features, "
                    f"but model reports "
                    f"{model_feature_count}."
                )

                logger.error(
                    self.load_error
                )

                return False

            # =============================================================
            # Validate model classes
            # =============================================================

            model_classes = getattr(
                self.model,
                "classes_",
                None
            )

            if model_classes is not None:

                try:

                    model_classes = list(
                        model_classes
                    )

                except Exception:

                    model_classes = None

            self.model_classes = (
                model_classes
            )

            # -------------------------------------------------------------
            # If classes are available, require binary classes.
            # -------------------------------------------------------------

            if model_classes is not None:

                if len(
                    model_classes
                ) != self.EXPECTED_CLASS_COUNT:

                    self.load_error = (
                        "HTML model classification mismatch. "
                        "Expected exactly two classes "
                        "(0=legitimate, 1=phishing), "
                        f"but model contains: "
                        f"{model_classes}"
                    )

                    logger.error(
                        self.load_error
                    )

                    return False

                if set(
                    model_classes
                ) != {
                    self.LEGITIMATE_CLASS_INDEX,
                    self.PHISHING_CLASS_INDEX
                }:

                    self.load_error = (
                        "HTML model class labels are incompatible. "
                        "Expected classes [0, 1], "
                        f"received {model_classes}."
                    )

                    logger.error(
                        self.load_error
                    )

                    return False

            # =============================================================
            # Validate predict_proba availability
            # =============================================================

            if not callable(
                getattr(
                    self.model,
                    "predict_proba",
                    None
                )
            ):

                self.load_error = (
                    "HTML XGBoost model does not provide "
                    "predict_proba()."
                )

                logger.error(
                    self.load_error
                )

                return False

            # =============================================================
            # Validation successful
            # =============================================================

            logger.info(
                "HTML model validation passed. "
                "Expected features: %d | "
                "Model features: %s",
                expected_feature_count,
                model_feature_count
            )

            return True

        except Exception as e:

            self.load_error = (
                "Unexpected HTML model validation error: "
                f"{str(e)}"
            )

            logger.error(
                self.load_error,
                exc_info=True
            )

            return False

    # =====================================================================
    # INPUT DATAFRAME VALIDATION
    # =====================================================================

    def _validate_input_dataframe(
        self,
        df_processed: pd.DataFrame
    ) -> bool:
        """
        Validate the DataFrame immediately before inference.

        Validation includes:

            - DataFrame type
            - Non-empty input
            - Expected columns
            - Exact column order
            - Numeric values
            - No NaN
            - No infinity
        """

        # ---------------------------------------------------------------
        # DataFrame type
        # ---------------------------------------------------------------

        if not isinstance(
            df_processed,
            pd.DataFrame
        ):

            logger.error(
                "HTML model received invalid input type: %s",
                type(
                    df_processed
                ).__name__
            )

            return False

        # ---------------------------------------------------------------
        # Empty DataFrame
        # ---------------------------------------------------------------

        if df_processed.empty:

            logger.error(
                "HTML model received an empty DataFrame."
            )

            return False

        # ---------------------------------------------------------------
        # Expected schema
        # ---------------------------------------------------------------

        expected_columns = (
            HTMLFeatureSchema.get_schema()
        )

        actual_columns = list(
            df_processed.columns
        )

        if actual_columns != expected_columns:

            logger.error(
                "HTML model feature schema mismatch.\n"
                "Expected: %s\n"
                "Received: %s",
                expected_columns,
                actual_columns
            )

            return False

        # ---------------------------------------------------------------
        # Model feature count
        # ---------------------------------------------------------------

        if (
            self.model_feature_count is not None
            and
            len(actual_columns)
            != self.model_feature_count
        ):

            logger.error(
                "HTML input feature count %d "
                "does not match model feature count %d.",
                len(actual_columns),
                self.model_feature_count
            )

            return False

        # ---------------------------------------------------------------
        # Numeric validation
        # ---------------------------------------------------------------

        for column in expected_columns:

            if not pd.api.types.is_numeric_dtype(
                df_processed[column]
            ):

                logger.error(
                    "HTML model received non-numeric "
                    "feature '%s'.",
                    column
                )

                return False

        # ---------------------------------------------------------------
        # NaN validation
        # ---------------------------------------------------------------

        if df_processed.isna().any().any():

            logger.error(
                "HTML model received NaN values."
            )

            return False

        # ---------------------------------------------------------------
        # Infinity validation
        # ---------------------------------------------------------------

        try:

            values = (
                df_processed.to_numpy(
                    dtype=float
                )
            )

            if np.isinf(
                values
            ).any():

                logger.error(
                    "HTML model received infinite values."
                )

                return False

        except Exception as e:

            logger.error(
                "Could not validate HTML model "
                "numeric values: %s",
                str(e)
            )

            return False

        return True

    # =====================================================================
    # SINGLE SAMPLE PROBABILITY INFERENCE
    # =====================================================================

    def predict_proba(
        self,
        df_processed: pd.DataFrame
    ) -> float:
        """
        Execute XGBoost probability inference.

        Args:
            df_processed:
                Preprocessed HTML feature DataFrame.

        Returns:
            float:
                Phishing probability in [0.0, 1.0].

        Important:
            Returning 0.0 when inference fails does NOT mean that the
            website is legitimate. The Predictor/Agent must inspect
            model status when deciding whether the signal is usable.
        """

        # ================================================================
        # 1. MODEL AVAILABILITY
        # ================================================================

        if not self.is_loaded:

            logger.warning(
                "HTML model inference requested, "
                "but the model is not loaded."
            )

            return 0.0

        # ================================================================
        # 2. INPUT VALIDATION
        # ================================================================

        if not self._validate_input_dataframe(
            df_processed
        ):

            logger.error(
                "HTML model input validation failed."
            )

            return 0.0

        try:

            # ============================================================
            # 3. EXECUTE XGBOOST PREDICTION
            # ============================================================

            probabilities = (
                self.model.predict_proba(
                    df_processed
                )
            )

            # ============================================================
            # 4. VALIDATE PROBABILITY MATRIX
            # ============================================================

            if probabilities is None:

                raise ValueError(
                    "XGBoost returned None "
                    "for predict_proba()."
                )

            probabilities = np.asarray(
                probabilities,
                dtype=float
            )

            if probabilities.ndim != 2:

                raise ValueError(
                    "Unexpected probability matrix "
                    f"shape: {probabilities.shape}"
                )

            if probabilities.shape[0] < 1:

                raise ValueError(
                    "XGBoost returned an empty "
                    "probability matrix."
                )

            if probabilities.shape[1] < 2:

                raise ValueError(
                    "HTML XGBoost model does not appear "
                    "to be configured for binary classification."
                )

            # ============================================================
            # 5. EXTRACT PHISHING PROBABILITY
            # ============================================================

            phishing_probability = float(
                probabilities[
                    0,
                    self.PHISHING_CLASS_INDEX
                ]
            )

            # ============================================================
            # 6. PROBABILITY SAFETY
            # ============================================================

            if not np.isfinite(
                phishing_probability
            ):

                raise ValueError(
                    "XGBoost returned a non-finite "
                    "phishing probability."
                )

            phishing_probability = max(
                0.0,
                min(
                    1.0,
                    phishing_probability
                )
            )

            logger.debug(
                "HTML phishing probability: %.4f",
                phishing_probability
            )

            return phishing_probability

        except ValueError as e:

            logger.error(
                "HTML model inference validation error: %s",
                str(e)
            )

            return 0.0

        except Exception as e:

            logger.error(
                "Unexpected HTML model inference error: %s",
                str(e),
                exc_info=True
            )

            return 0.0

    # =====================================================================
    # BATCH PROBABILITY INFERENCE
    # =====================================================================

    def predict_proba_batch(
        self,
        df_processed: pd.DataFrame
    ) -> List[float]:
        """
        Generate phishing probabilities for multiple HTML samples.

        Returns:
            List[float]:
                One phishing probability per input row.
        """

        # ---------------------------------------------------------------
        # Model availability
        # ---------------------------------------------------------------

        if not self.is_loaded:

            logger.warning(
                "Batch inference requested, "
                "but HTML model is not loaded."
            )

            return []

        # ---------------------------------------------------------------
        # Input validation
        # ---------------------------------------------------------------

        if not self._validate_input_dataframe(
            df_processed
        ):

            logger.error(
                "HTML batch model input validation failed."
            )

            return []

        try:

            probabilities = (
                self.model.predict_proba(
                    df_processed
                )
            )

            probabilities = np.asarray(
                probabilities,
                dtype=float
            )

            # -----------------------------------------------------------
            # Validate shape
            # -----------------------------------------------------------

            if probabilities.ndim != 2:

                raise ValueError(
                    "Unexpected batch probability matrix "
                    f"shape: {probabilities.shape}"
                )

            if probabilities.shape[0] != len(
                df_processed
            ):

                raise ValueError(
                    "XGBoost returned an unexpected number "
                    "of probability rows."
                )

            if probabilities.shape[1] < 2:

                raise ValueError(
                    "HTML model does not contain "
                    "two probability classes."
                )

            # -----------------------------------------------------------
            # Extract phishing probabilities
            # -----------------------------------------------------------

            phishing_probabilities = (
                probabilities[
                    :,
                    self.PHISHING_CLASS_INDEX
                ]
            )

            # -----------------------------------------------------------
            # Validate probabilities
            # -----------------------------------------------------------

            if not np.isfinite(
                phishing_probabilities
            ).all():

                raise ValueError(
                    "HTML model returned non-finite "
                    "batch probabilities."
                )

            phishing_probabilities = np.clip(
                phishing_probabilities,
                0.0,
                1.0
            )

            return [
                float(
                    probability
                )
                for probability
                in phishing_probabilities
            ]

        except Exception as e:

            logger.error(
                "HTML batch inference failed: %s",
                str(e),
                exc_info=True
            )

            return []

    # =====================================================================
    # DIRECT CLASS PREDICTION
    # =====================================================================

    def predict_class(
        self,
        df_processed: pd.DataFrame,
        threshold: float = 0.55
    ) -> int:
        """
        Convert phishing probability into a binary class.

        Policy:

            probability >= threshold
                → 1 phishing

            probability < threshold
                → 0 legitimate

        The default threshold is 0.55 and must remain consistent
        with predictor.py.
        """

        # ---------------------------------------------------------------
        # Validate threshold
        # ---------------------------------------------------------------

        try:

            threshold = float(
                threshold
            )

        except (
            TypeError,
            ValueError
        ):

            logger.warning(
                "Invalid classification threshold '%s'. "
                "Using 0.55.",
                threshold
            )

            threshold = 0.55

        threshold = max(
            0.01,
            min(
                0.99,
                threshold
            )
        )

        # ---------------------------------------------------------------
        # Probability
        # ---------------------------------------------------------------

        probability = (
            self.predict_proba(
                df_processed
            )
        )

        # ---------------------------------------------------------------
        # Classification
        # ---------------------------------------------------------------

        return (
            self.PHISHING_CLASS_INDEX
            if probability >= threshold
            else self.LEGITIMATE_CLASS_INDEX
        )

    # =====================================================================
    # MODEL STATUS
    # =====================================================================

    def get_status(
        self
    ) -> Dict[str, Any]:
        """
        Return detailed model status.

        Useful for:

            - Health checks
            - Dashboard
            - Debugging
            - API responses
            - Multi-agent orchestration
        """

        expected_feature_count = len(
            HTMLFeatureSchema.get_schema()
        )

        return {

            "model_loaded":
                self.is_loaded,

            "model_path":
                self.model_path,

            "model_exists":
                os.path.isfile(
                    self.model_path
                ),

            "model_filename":
                os.path.basename(
                    self.model_path
                ),

            "model_feature_count":
                self.model_feature_count,

            "expected_feature_count":
                expected_feature_count,

            "feature_schema_match":
                (
                    self.model_feature_count
                    is None
                    or
                    self.model_feature_count
                    == expected_feature_count
                ),

            "model_classes":
                self.model_classes,

            "classification":
                {
                    "legitimate":
                        self.LEGITIMATE_CLASS_INDEX,

                    "phishing":
                        self.PHISHING_CLASS_INDEX
                },

            "load_error":
                self.load_error
        }

    # =====================================================================
    # MODEL METADATA
    # =====================================================================

    def get_metadata(
        self
    ) -> Dict[str, Any]:
        """
        Return model metadata for debugging and research documentation.
        """

        booster_information = {}

        if self.is_loaded:

            try:

                booster = (
                    self.model.get_booster()
                )

                booster_information = {

                    "booster_type":
                        str(
                            booster.booster
                            if hasattr(
                                booster,
                                "booster"
                            )
                            else "unknown"
                        ),

                    "feature_names":
                        booster.feature_names
                }

            except Exception as e:

                logger.debug(
                    "Unable to retrieve booster metadata: %s",
                    str(e)
                )

        return {

            "model_type":
                "XGBoost",

            "model_filename":
                self.MODEL_FILENAME,

            "model_path":
                self.model_path,

            "loaded":
                self.is_loaded,

            "feature_count":
                self.model_feature_count,

            "expected_feature_count":
                len(
                    HTMLFeatureSchema.get_schema()
                ),

            "classes":
                self.model_classes,

            "classification_type":
                "binary",

            "class_mapping":
                {
                    "0":
                        "legitimate",

                    "1":
                        "phishing"
                },

            "booster":
                booster_information
        }

    # =====================================================================
    # MODEL AVAILABILITY
    # =====================================================================

    def is_available(
        self
    ) -> bool:
        """
        Return True when a compatible model is loaded.
        """

        return bool(
            self.is_loaded
        )

    # =====================================================================
    # MODEL RELOAD
    # =====================================================================

    def reload(
        self
    ) -> bool:
        """
        Reload the model from the configured model path.

        Useful after retraining without restarting the application.
        """

        logger.info(
            "Reloading HTML XGBoost model..."
        )

        self._load_model()

        return bool(
            self.is_loaded
        )