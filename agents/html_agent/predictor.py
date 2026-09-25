"""
HTML AI Agent - Prediction Engine
=================================

Responsible for converting raw HTML features into a validated
HTML phishing prediction.

Pipeline:

    Raw HTML Features
            ↓
    HTMLPreprocessor
            ↓
    HTMLModel
            ↓
    Phishing Probability
            ↓
    Classification
            ↓
    HTML Risk Scoring
            ↓
    Explainable AI
            ↓
    HTML Agent
            ↓
    Decision Fusion Engine

Classification:

    0 = Legitimate
    1 = Phishing

IMPORTANT
---------
This module produces an HTML-agent-local prediction.

It does NOT produce the final system-wide phishing verdict.
"""

# ============================================================================
# IMPORTS
# ============================================================================

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

try:
    import shap
except ImportError:
    shap = None

from .model import HTMLModel
from .preprocessing import HTMLPreprocessor
from .feature_schema import HTMLFeatureSchema


# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# HTML PREDICTOR
# ============================================================================

class HTMLPredictor:
    """
    Prediction engine for the HTML AI Agent.

    Responsibilities
    ----------------
    1. Validate raw HTML features.
    2. Preprocess HTML features.
    3. Validate the processed feature matrix.
    4. Execute the trained 50-feature XGBoost model.
    5. Convert probability into a class.
    6. Provide a controlled fallback when the model is unavailable.
    7. Return standardized prediction results.
    8. Expose the underlying HTMLModel to HTMLAIAgent/explain.py.

    Final production model:

        data/models/html_agent/html_xgb_tuned_model.json
    """

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    AGENT_NAME = "HTML_AI_Agent"

    AGENT_VERSION = "2.0"

    EXPECTED_FEATURE_COUNT = 50

    # ------------------------------------------------------------------------
    # Production classification threshold
    #
    # This remains 0.55 to preserve the deployment behavior already used
    # by the existing HTML Agent.
    # ------------------------------------------------------------------------

    DEFAULT_THRESHOLD = 0.55

    FALLBACK_THRESHOLD = 0.55

    MIN_PROBABILITY = 0.0

    MAX_PROBABILITY = 1.0

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD
    ):
        """
        Initialize the HTML Predictor.

        Args:
            model_path:
                Optional path to the trained HTML XGBoost model.

                If omitted, the finalized tuned 50-feature model is used.

            threshold:
                Probability threshold for phishing classification.
        """

        logger.info(
            "Initializing %s Predictor...",
            self.AGENT_NAME
        )

        # --------------------------------------------------------------------
        # Preprocessor
        # --------------------------------------------------------------------

        self.preprocessor = HTMLPreprocessor()

        # --------------------------------------------------------------------
        # Resolve production model path
        # --------------------------------------------------------------------

        resolved_model_path = (
            self._resolve_model_path(
                model_path
            )
        )

        self.model_path = (
            resolved_model_path
        )

        logger.info(
            "HTML model path: %s",
            self.model_path
        )

        # --------------------------------------------------------------------
        # HTML Model
        # --------------------------------------------------------------------

        self.model = HTMLModel(
            model_path=self.model_path
        )

        # --------------------------------------------------------------------
        # SHAP explainability
        # --------------------------------------------------------------------
        # The HTMLModel class is a project wrapper around XGBoost. SHAP's
        # TreeExplainer cannot explain the wrapper object directly, so we
        # resolve the native XGBoost estimator/booster lazily and keep the
        # predictor fully operational if SHAP is unavailable.
        # --------------------------------------------------------------------

        self.shap_explainer = None
        self.shap_status = "unavailable"
        self.shap_error = None

        self._initialize_shap_explainer()

        # --------------------------------------------------------------------
        # Prediction threshold
        # --------------------------------------------------------------------

        self.threshold = (
            self._validate_threshold(
                threshold
            )
        )

        # --------------------------------------------------------------------
        # Central feature schema
        # --------------------------------------------------------------------

        self.feature_schema = (
            HTMLFeatureSchema.get_schema()
        )

        # --------------------------------------------------------------------
        # Feature count validation
        # --------------------------------------------------------------------

        if len(
            self.feature_schema
        ) != self.EXPECTED_FEATURE_COUNT:

            raise RuntimeError(
                "HTML feature schema mismatch. "
                f"Expected {self.EXPECTED_FEATURE_COUNT} features, "
                f"found {len(self.feature_schema)}."
            )

        logger.info(
            "%s Predictor initialized successfully.",
            self.AGENT_NAME
        )

        logger.info(
            "Prediction threshold: %.2f",
            self.threshold
        )

        logger.info(
            "Feature count: %d",
            len(self.feature_schema)
        )

    # ========================================================================
    # MODEL PATH RESOLUTION
    # ========================================================================

    @staticmethod
    def _resolve_model_path(
        model_path: Optional[str]
    ) -> str:
        """
        Resolve the HTML XGBoost model path.

        Priority:

            1. Explicit model_path supplied by caller.
            2. Finalized tuned model.

        The tuned model is intentionally preferred because the
        50-feature tuning stage has already been completed.
        """

        # --------------------------------------------------------------------
        # Explicit path
        # --------------------------------------------------------------------

        if model_path:

            explicit_path = Path(
                model_path
            )

            return str(
                explicit_path
            )

        # --------------------------------------------------------------------
        # Project root
        #
        # predictor.py:
        #
        # project/
        #   agents/
        #       html_agent/
        #           predictor.py
        #
        # Therefore:
        #
        # parent.parent.parent
        # --------------------------------------------------------------------

        project_root = (
            Path(__file__)
            .resolve()
            .parent
            .parent
            .parent
        )

        tuned_model = (
            project_root
            / "data"
            / "models"
            / "html_agent"
            / "html_xgb_tuned_model.json"
        )

        baseline_model = (
            project_root
            / "data"
            / "models"
            / "html_agent"
            / "html_xgb_model.json"
        )

        # --------------------------------------------------------------------
        # Prefer tuned model
        # --------------------------------------------------------------------

        if tuned_model.exists():

            logger.info(
                "Using finalized tuned HTML XGBoost model: %s",
                tuned_model
            )

            return str(
                tuned_model
            )

        # --------------------------------------------------------------------
        # Baseline fallback
        #
        # This is only a file-level compatibility fallback.
        # It is NOT the heuristic prediction fallback.
        # --------------------------------------------------------------------

        if baseline_model.exists():

            logger.warning(
                "Tuned HTML model was not found. "
                "Using baseline 50-feature HTML XGBoost model: %s",
                baseline_model
            )

            return str(
                baseline_model
            )

        # --------------------------------------------------------------------
        # Return tuned path even if missing.
        #
        # HTMLModel will report model unavailability and the predictor
        # will use its controlled fallback.
        # --------------------------------------------------------------------

        logger.warning(
            "No HTML XGBoost model file was found. "
            "Expected tuned model at: %s",
            tuned_model
        )

        return str(
            tuned_model
        )

    # ========================================================================
    # GET MODEL
    # ========================================================================

    def get_model(
        self
    ) -> HTMLModel:
        """
        Return the underlying HTMLModel wrapper.

        Required by html_agent.py.
        """

        return self.model

    # ========================================================================
    # MAIN PREDICTION
    # ========================================================================

    def predict(
        self,
        raw_html_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict whether HTML structure is associated with phishing.
        """

        # ====================================================================
        # 1. INPUT VALIDATION
        # ====================================================================

        if not isinstance(
            raw_html_features,
            dict
        ):

            logger.warning(
                "HTML Predictor received invalid feature payload."
            )

            return self._get_error_result(
                "Invalid HTML feature payload. "
                "Expected dictionary."
            )

        # ====================================================================
        # 2. PREPROCESSING
        # ====================================================================

        try:

            df_processed = (
                self.preprocessor.transform_single(
                    raw_html_features
                )
            )

        except Exception as exc:

            logger.error(
                "HTML preprocessing failed: %s",
                str(exc),
                exc_info=True
            )

            return self._get_error_result(
                f"HTML preprocessing failed: {str(exc)}"
            )

        # ====================================================================
        # 3. VALIDATE PROCESSED FEATURES
        # ====================================================================

        if not self._validate_processed_features(
            df_processed
        ):

            return self._get_error_result(
                "Processed HTML features failed "
                "schema or numerical validation."
            )

        # ====================================================================
        # 4. MODEL AVAILABILITY
        # ====================================================================

        try:

            model_available = (
                self.model.is_available()
            )

        except Exception as exc:

            logger.error(
                "Could not determine HTML model availability: %s",
                str(exc),
                exc_info=True
            )

            model_available = False

        # ====================================================================
        # 5. XGBOOST PREDICTION
        # ====================================================================

        if model_available:

            return self._predict_with_model(
                df_processed
            )

        # ====================================================================
        # 6. CONTROLLED FALLBACK
        # ====================================================================

        logger.warning(
            "HTML XGBoost model unavailable. "
            "Using heuristic fallback."
        )

        return self._predict_with_fallback(
            df_processed
        )

    # ========================================================================
    # XGBOOST PREDICTION
    # ========================================================================

    def _predict_with_model(
        self,
        df_processed: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Execute prediction using the finalized XGBoost model.
        """

        try:

            # ----------------------------------------------------------------
            # Get phishing probability
            # ----------------------------------------------------------------

            phishing_probability = (
                self.model.predict_proba(
                    df_processed
                )
            )

            phishing_probability = (
                self._normalize_probability(
                    phishing_probability
                )
            )

            # ----------------------------------------------------------------
            # Legitimate probability
            # ----------------------------------------------------------------

            legitimate_probability = (
                1.0
                - phishing_probability
            )

            legitimate_probability = (
                self._normalize_probability(
                    legitimate_probability
                )
            )

            # ----------------------------------------------------------------
            # Classification
            # ----------------------------------------------------------------

            class_index = (
                1
                if phishing_probability >= self.threshold
                else 0
            )

            class_label = (
                "phishing"
                if class_index == 1
                else "legitimate"
            )

            # ----------------------------------------------------------------
            # Confidence
            # ----------------------------------------------------------------

            confidence = (
                phishing_probability
                if class_index == 1
                else legitimate_probability
            )

            confidence = (
                self._normalize_probability(
                    confidence
                )
            )

            # ----------------------------------------------------------------
            # Result
            # ----------------------------------------------------------------

            result = {

                "agent":
                    "html",

                "agent_name":
                    self.AGENT_NAME,

                "agent_version":
                    self.AGENT_VERSION,

                "analysis_status":
                    "success",

                "signal_available":
                    True,

                "model_status":
                    "loaded",

                "model_path":
                    self.model_path,

                "prediction_source":
                    "xgboost",

                "model_available":
                    True,

                "class_index":
                    class_index,

                "class_label":
                    class_label,

                "prediction":
                    class_label,

                "confidence":
                    round(
                        confidence,
                        4
                    ),

                "confidence_score":
                    round(
                        confidence,
                        4
                    ),

                "phishing_probability":
                    round(
                        phishing_probability,
                        4
                    ),

                "legitimate_probability":
                    round(
                        legitimate_probability,
                        4
                    ),

                "class_probabilities":
                    {
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

                "threshold":
                    round(
                        self.threshold,
                        4
                    ),

                "feature_count":
                    len(
                        self.feature_schema
                    ),

                "fallback_used":
                    False
            }

            logger.info(
                "HTML XGBoost prediction | "
                "label=%s | probability=%.4f | "
                "threshold=%.4f",
                class_label,
                phishing_probability,
                self.threshold
            )

            # ----------------------------------------------------------------
            # Model-based explanation
            # ----------------------------------------------------------------
            # Explanation failure must NEVER invalidate an otherwise valid
            # XGBoost prediction. The prediction remains authoritative.
            # ----------------------------------------------------------------

            try:
                result["explanation"] = self.explain_prediction(
                    df_processed,
                    phishing_probability=phishing_probability
                )
            except Exception as explanation_error:
                logger.warning(
                    "HTML XGBoost explanation generation failed: %s",
                    explanation_error,
                    exc_info=True
                )
                result["explanation"] = self._get_explanation_fallback(
                    df_processed,
                    phishing_probability,
                    str(explanation_error)
                )

            return result

        except Exception as exc:

            logger.error(
                "HTML XGBoost prediction failed: %s",
                str(exc),
                exc_info=True
            )

            # ----------------------------------------------------------------
            # Model inference failure
            #
            # HTML evidence itself is available, therefore a controlled
            # heuristic fallback is permitted.
            # ----------------------------------------------------------------

            return self._predict_with_fallback(
                df_processed,
                model_error=str(exc)
            )

    # ========================================================================
    # HEURISTIC FALLBACK
    # ========================================================================

    def _predict_with_fallback(
        self,
        df_processed: pd.DataFrame,
        model_error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Controlled heuristic fallback.

        IMPORTANT
        ---------
        This is NOT equivalent to the trained XGBoost model.

        It is used only when usable HTML features were successfully
        extracted but the ML model cannot be used.
        """

        try:

            # ----------------------------------------------------------------
            # Extract features safely
            # ----------------------------------------------------------------

            has_password = (
                self._safe_numeric_feature(
                    df_processed,
                    "has_password_field"
                )
            )

            # IMPORTANT:
            # 50-feature schema contains email_field_count,
            # not has_email_field.
            email_field_count = (
                self._safe_numeric_feature(
                    df_processed,
                    "email_field_count"
                )
            )

            hidden_inputs = (
                self._safe_numeric_feature(
                    df_processed,
                    "hidden_input_count"
                )
            )

            credential_hidden_inputs = (
                self._safe_numeric_feature(
                    df_processed,
                    "credential_hidden_input_count"
                )
            )

            suspicious_hidden_inputs = (
                self._safe_numeric_feature(
                    df_processed,
                    "suspicious_hidden_input_count"
                )
            )

            invisible_iframes = (
                self._safe_numeric_feature(
                    df_processed,
                    "invisible_iframes"
                )
            )

            suspicious_scripts = (
                self._safe_numeric_feature(
                    df_processed,
                    "suspicious_inline_scripts"
                )
            )

            cross_domain_forms = (
                self._safe_numeric_feature(
                    df_processed,
                    "cross_domain_form_actions"
                )
            )

            empty_form_actions = (
                self._safe_numeric_feature(
                    df_processed,
                    "empty_form_action_count"
                )
            )

            external_scripts = (
                self._safe_numeric_feature(
                    df_processed,
                    "external_scripts_count"
                )
            )

            hidden_elements = (
                self._safe_numeric_feature(
                    df_processed,
                    "hidden_elements_count"
                )
            )

            right_click_disabled = (
                self._safe_numeric_feature(
                    df_processed,
                    "right_click_disabled"
                )
            )

            has_meta_refresh = (
                self._safe_numeric_feature(
                    df_processed,
                    "has_meta_refresh"
                )
            )

            empty_links = (
                self._safe_numeric_feature(
                    df_processed,
                    "empty_links_count"
                )
            )

            # ----------------------------------------------------------------
            # Risk calculation
            # ----------------------------------------------------------------

            risk_points = 0.0

            triggered_indicators: List[str] = []

            # ----------------------------------------------------------------
            # Password field
            # ----------------------------------------------------------------

            if has_password > 0:

                risk_points += 0.03

            # ----------------------------------------------------------------
            # Email field
            # ----------------------------------------------------------------

            if email_field_count > 0:

                risk_points += 0.02

            # ----------------------------------------------------------------
            # Hidden inputs
            # ----------------------------------------------------------------

            if hidden_inputs > 0:

                risk_points += min(
                    0.10,
                    hidden_inputs * 0.01
                )

                triggered_indicators.append(
                    "hidden_form_inputs_detected"
                )

            # ----------------------------------------------------------------
            # Credential hidden inputs
            # ----------------------------------------------------------------

            if credential_hidden_inputs > 0:

                risk_points += min(
                    0.20,
                    credential_hidden_inputs * 0.05
                )

                triggered_indicators.append(
                    "credential_hidden_inputs_detected"
                )

            # ----------------------------------------------------------------
            # Suspicious hidden inputs
            # ----------------------------------------------------------------

            if suspicious_hidden_inputs > 0:

                risk_points += min(
                    0.20,
                    suspicious_hidden_inputs * 0.05
                )

                triggered_indicators.append(
                    "suspicious_hidden_inputs_detected"
                )

            # ----------------------------------------------------------------
            # Invisible iframes
            # ----------------------------------------------------------------

            if invisible_iframes > 0:

                risk_points += min(
                    0.25,
                    invisible_iframes * 0.10
                )

                triggered_indicators.append(
                    "invisible_iframe_detected"
                )

            # ----------------------------------------------------------------
            # Suspicious JavaScript
            # ----------------------------------------------------------------

            if suspicious_scripts > 0:

                risk_points += min(
                    0.25,
                    suspicious_scripts * 0.10
                )

                triggered_indicators.append(
                    "suspicious_javascript_detected"
                )

            # ----------------------------------------------------------------
            # Cross-domain form action
            # ----------------------------------------------------------------

            if cross_domain_forms > 0:

                risk_points += min(
                    0.25,
                    cross_domain_forms * 0.10
                )

                triggered_indicators.append(
                    "cross_domain_form_action_detected"
                )

            # ----------------------------------------------------------------
            # Empty form action
            # ----------------------------------------------------------------

            if empty_form_actions > 0:

                risk_points += min(
                    0.10,
                    empty_form_actions * 0.02
                )

                triggered_indicators.append(
                    "empty_form_action_detected"
                )

            # ----------------------------------------------------------------
            # External scripts
            # ----------------------------------------------------------------

            if external_scripts > 10:

                risk_points += 0.05

                triggered_indicators.append(
                    "high_external_script_count"
                )

            # ----------------------------------------------------------------
            # Hidden elements
            # ----------------------------------------------------------------

            if hidden_elements > 5:

                risk_points += 0.05

                triggered_indicators.append(
                    "hidden_html_elements_detected"
                )

            # ----------------------------------------------------------------
            # Right-click disabled
            # ----------------------------------------------------------------

            if right_click_disabled > 0:

                risk_points += 0.03

                triggered_indicators.append(
                    "right_click_disabled"
                )

            # ----------------------------------------------------------------
            # Meta refresh
            # ----------------------------------------------------------------

            if has_meta_refresh > 0:

                risk_points += 0.10

                triggered_indicators.append(
                    "meta_refresh_detected"
                )

            # ----------------------------------------------------------------
            # Empty links
            # ----------------------------------------------------------------

            if empty_links > 10:

                risk_points += 0.05

                triggered_indicators.append(
                    "excessive_empty_links"
                )

            # =================================================================
            # COMBINATION RULES
            # =================================================================

            # ----------------------------------------------------------------
            # Credential capture pattern
            # ----------------------------------------------------------------

            if (
                has_password > 0
                and credential_hidden_inputs > 0
            ):

                risk_points += 0.15

                triggered_indicators.append(
                    "credential_capture_pattern"
                )

            # ----------------------------------------------------------------
            # Cross-domain credential submission
            # ----------------------------------------------------------------

            if (
                has_password > 0
                and cross_domain_forms > 0
            ):

                risk_points += 0.15

                triggered_indicators.append(
                    "cross_domain_credential_submission"
                )

            # ----------------------------------------------------------------
            # Invisible iframe + suspicious JavaScript
            # ----------------------------------------------------------------

            if (
                invisible_iframes > 0
                and suspicious_scripts > 0
            ):

                risk_points += 0.15

                triggered_indicators.append(
                    "iframe_javascript_suspicious_combination"
                )

            # =================================================================
            # NORMALIZE
            # =================================================================

            phishing_probability = (
                self._normalize_probability(
                    risk_points
                )
            )

            legitimate_probability = (
                1.0
                - phishing_probability
            )

            # =================================================================
            # CLASSIFICATION
            # =================================================================

            class_index = (
                1
                if phishing_probability >= self.FALLBACK_THRESHOLD
                else 0
            )

            class_label = (
                "phishing"
                if class_index == 1
                else "legitimate"
            )

            confidence = (
                phishing_probability
                if class_index == 1
                else legitimate_probability
            )

            # =================================================================
            # FALLBACK REASON
            # =================================================================

            if model_error:

                fallback_reason = (
                    "Trained XGBoost inference failed. "
                    "Heuristic fallback was used."
                )

            else:

                fallback_reason = (
                    "Trained XGBoost model is unavailable. "
                    "Heuristic fallback was used."
                )

            logger.warning(
                "HTML heuristic prediction | "
                "label=%s | probability=%.4f",
                class_label,
                phishing_probability
            )

            # =================================================================
            # RETURN
            # =================================================================

            return {

                "agent":
                    "html",

                "agent_name":
                    self.AGENT_NAME,

                "agent_version":
                    self.AGENT_VERSION,

                "analysis_status":
                    "degraded",

                "signal_available":
                    True,

                "model_status":
                    "unavailable",

                "model_path":
                    self.model_path,

                "prediction_source":
                    "heuristic_fallback",

                "class_index":
                    class_index,

                "class_label":
                    class_label,

                "prediction":
                    class_label,

                "confidence":
                    round(
                        confidence,
                        4
                    ),

                "confidence_score":
                    round(
                        confidence,
                        4
                    ),

                "phishing_probability":
                    round(
                        phishing_probability,
                        4
                    ),

                "legitimate_probability":
                    round(
                        legitimate_probability,
                        4
                    ),

                "class_probabilities":
                    {
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

                "threshold":
                    round(
                        self.FALLBACK_THRESHOLD,
                        4
                    ),

                "fallback_used":
                    True,

                "fallback_reason":
                    fallback_reason,

                "model_error":
                    model_error,

                "triggered_indicators":
                    triggered_indicators,

                "indicator_count":
                    len(
                        triggered_indicators
                    ),

                "feature_count":
                    len(
                        self.feature_schema
                    ),

                "model_available":
                    False
            }

        except Exception as exc:

            logger.error(
                "HTML heuristic fallback failed: %s",
                str(exc),
                exc_info=True
            )

            return self._get_error_result(
                "Both XGBoost prediction and heuristic "
                f"fallback failed: {str(exc)}"
            )

    # ========================================================================
    # PROCESSED FEATURE VALIDATION
    # ========================================================================

    def _validate_processed_features(
        self,
        df_processed: pd.DataFrame
    ) -> bool:
        """
        Validate the processed feature matrix.
        """

        if not isinstance(
            df_processed,
            pd.DataFrame
        ):

            logger.error(
                "Processed HTML features are not a DataFrame."
            )

            return False

        if df_processed.empty:

            logger.error(
                "Processed HTML feature DataFrame is empty."
            )

            return False

        # --------------------------------------------------------------------
        # Exact feature count
        # --------------------------------------------------------------------

        if len(
            df_processed.columns
        ) != self.EXPECTED_FEATURE_COUNT:

            logger.error(
                "HTML feature count mismatch. "
                "Expected %d, found %d.",
                self.EXPECTED_FEATURE_COUNT,
                len(df_processed.columns)
            )

            return False

        # --------------------------------------------------------------------
        # Schema validation
        # --------------------------------------------------------------------

        if not HTMLFeatureSchema.validate_dataframe(
            df_processed
        ):

            logger.error(
                "HTML feature schema validation failed."
            )

            return False

        # --------------------------------------------------------------------
        # Exact column order
        # --------------------------------------------------------------------

        expected_columns = (
            HTMLFeatureSchema.get_schema()
        )

        if list(
            df_processed.columns
        ) != expected_columns:

            logger.error(
                "HTML feature order mismatch."
            )

            logger.error(
                "Expected: %s",
                expected_columns
            )

            logger.error(
                "Received: %s",
                list(
                    df_processed.columns
                )
            )

            return False

        # --------------------------------------------------------------------
        # NaN
        # --------------------------------------------------------------------

        if df_processed.isna().any().any():

            logger.error(
                "Processed HTML features contain NaN."
            )

            return False

        # --------------------------------------------------------------------
        # Infinity
        # --------------------------------------------------------------------

        try:

            values = (
                df_processed.to_numpy(
                    dtype=float
                )
            )

            if not np.isfinite(
                values
            ).all():

                logger.error(
                    "Processed HTML features contain "
                    "non-finite values."
                )

                return False

        except Exception as exc:

            logger.error(
                "Could not validate HTML feature values: %s",
                str(exc)
            )

            return False

        # --------------------------------------------------------------------
        # Numeric dtype
        # --------------------------------------------------------------------

        for column in expected_columns:

            if not pd.api.types.is_numeric_dtype(
                df_processed[column]
            ):

                logger.error(
                    "HTML feature '%s' is not numeric.",
                    column
                )

                return False

        return True

    # ========================================================================
    # SAFE FEATURE ACCESS
    # ========================================================================

    @staticmethod
    def _safe_numeric_feature(
        df: pd.DataFrame,
        feature_name: str
    ) -> float:
        """
        Safely retrieve one numerical feature.
        """

        try:

            if feature_name not in df.columns:

                return 0.0

            value = float(
                df.iloc[0][feature_name]
            )

            if not np.isfinite(
                value
            ):

                return 0.0

            return value

        except (
            TypeError,
            ValueError,
            IndexError,
            KeyError
        ):

            return 0.0

    # ========================================================================
    # BATCH PREDICTION
    # ========================================================================

    def predict_batch(
        self,
        raw_features_list: List[
            Dict[str, Any]
        ]
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Predict multiple raw HTML feature dictionaries.
        """

        if not isinstance(
            raw_features_list,
            list
        ):

            raise TypeError(
                "predict_batch expects a list of "
                "HTML feature dictionaries."
            )

        results = []

        for index, raw_features in enumerate(
            raw_features_list
        ):

            try:

                result = self.predict(
                    raw_features
                )

                results.append(
                    result
                )

            except Exception as exc:

                logger.error(
                    "HTML batch prediction failed "
                    "for sample %d: %s",
                    index,
                    str(exc),
                    exc_info=True
                )

                results.append(
                    self._get_error_result(
                        f"Batch prediction failed "
                        f"for sample {index}: {str(exc)}"
                    )
                )

        return results

    # ========================================================================
    # SHAP EXPLAINABILITY
    # ========================================================================

    def _resolve_native_xgb_model(self):
        """
        Resolve the native XGBoost estimator/booster hidden inside HTMLModel.

        HTMLModel is intentionally kept as the prediction abstraction. SHAP,
        however, needs the underlying XGBoost object rather than the wrapper.

        Several attribute names are supported so this remains compatible with
        small changes to HTMLModel without changing prediction behavior.
        """

        candidates = [
            "model",
            "xgb_model",
            "booster",
            "estimator",
            "classifier",
            "_model",
            "_xgb_model",
            "_booster",
        ]

        queue = [self.model]
        visited = set()

        while queue:
            current = queue.pop(0)

            if current is None:
                continue

            object_id = id(current)

            if object_id in visited:
                continue

            visited.add(object_id)

            module_name = getattr(
                current.__class__,
                "__module__",
                ""
            )

            class_name = getattr(
                current.__class__,
                "__name__",
                ""
            )

            # Native XGBoost sklearn estimators and Booster objects.
            if (
                "xgboost" in module_name.lower()
                and (
                    "XGB" in class_name
                    or class_name == "Booster"
                )
            ):
                return current

            # A wrapper may expose get_model()/get_booster().
            for method_name in (
                "get_model",
                "get_booster",
            ):
                method = getattr(
                    current,
                    method_name,
                    None
                )

                if callable(method):
                    try:
                        nested = method()

                        if nested is not None:
                            queue.append(nested)

                    except Exception:
                        pass

            for attr_name in candidates:
                try:
                    nested = getattr(
                        current,
                        attr_name,
                        None
                    )
                except Exception:
                    nested = None

                if nested is not None and nested is not current:
                    queue.append(nested)

        return None

    def _initialize_shap_explainer(self):
        """
        Initialize a TreeExplainer against the native XGBoost model.

        Failure is non-fatal. Prediction continues normally.
        """

        if shap is None:
            self.shap_status = "unavailable"
            self.shap_error = "SHAP package is not installed."
            logger.warning(
                "SHAP package is unavailable. "
                "HTML model explanations will use feature importance."
            )
            return

        if not self.model.is_available():
            self.shap_status = "model_unavailable"
            self.shap_error = "HTML XGBoost model is unavailable."
            return

        try:
            native_model = self._resolve_native_xgb_model()

            if native_model is None:
                raise RuntimeError(
                    "Could not resolve native XGBoost estimator "
                    "from HTMLModel wrapper."
                )

            self.shap_explainer = shap.TreeExplainer(
                native_model
            )

            self.shap_status = "ready"
            self.shap_error = None

            logger.info(
                "HTML SHAP TreeExplainer initialized successfully."
            )

        except Exception as exc:
            self.shap_explainer = None
            self.shap_status = "unavailable"
            self.shap_error = str(exc)

            logger.warning(
                "HTML SHAP TreeExplainer initialization failed: %s",
                exc
            )

    def _feature_importance_explanation(
        self,
        df_processed: pd.DataFrame,
        phishing_probability: float
    ) -> Dict[str, Any]:
        """
        Provide a model-derived global feature-importance explanation when
        per-sample SHAP is unavailable.

        This is explicitly labeled feature_importance rather than heuristic.
        """

        feature_names = list(
            self.feature_schema
        )

        native_model = self._resolve_native_xgb_model()

        importances = None

        if native_model is not None:

            values = getattr(
                native_model,
                "feature_importances_",
                None
            )

            if values is not None:
                importances = np.asarray(
                    values,
                    dtype=float
                ).reshape(-1)

        # XGBoost Booster fallback.
        if importances is None and native_model is not None:

            try:
                score_map = native_model.get_score(
                    importance_type="gain"
                )

                importances = np.array(
                    [
                        float(
                            score_map.get(
                                f"f{index}",
                                score_map.get(
                                    feature_name,
                                    0.0
                                )
                            )
                        )
                        for index, feature_name
                        in enumerate(feature_names)
                    ],
                    dtype=float
                )

            except Exception:
                importances = None

        if (
            importances is None
            or len(importances) != len(feature_names)
        ):

            return {
                "method": "unavailable",
                "summary": (
                    "HTML XGBoost prediction succeeded, but a "
                    "model-based explanation could not be generated."
                ),
                "top_shap_factors": [],
                "top_factors": [],
                "shap_status": self.shap_status,
                "shap_error": self.shap_error,
            }

        order = np.argsort(
            np.abs(importances)
        )[::-1]

        factors = []

        for index in order[:10]:

            importance = float(
                importances[index]
            )

            if importance <= 0:
                continue

            feature_name = feature_names[index]

            try:
                value = float(
                    df_processed.iloc[0][feature_name]
                )
            except Exception:
                value = None

            factors.append(
                {
                    "feature": feature_name,
                    "value": value,
                    "importance": round(
                        importance,
                        6
                    ),
                    "direction": (
                        "model-important"
                    )
                }
            )

        return {
            "method": "feature_importance",
            "summary": (
                "HTML XGBoost prediction succeeded. "
                "Per-sample SHAP was unavailable, so the explanation "
                "uses the trained model's feature importance ranking."
            ),
            "top_shap_factors": [],
            "top_factors": factors,
            "shap_status": self.shap_status,
            "shap_error": self.shap_error,
            "phishing_probability": round(
                phishing_probability,
                4
            )
        }

    def explain_prediction(
        self,
        df_processed: pd.DataFrame,
        phishing_probability: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a detailed model-based explanation for one prediction.

        Preferred method:
            SHAP TreeExplainer

        Secondary method:
            XGBoost feature importance

        There is intentionally no claim of sample-specific SHAP impact when
        SHAP is unavailable.
        """

        if not isinstance(
            df_processed,
            pd.DataFrame
        ):

            raise TypeError(
                "df_processed must be a pandas DataFrame."
            )

        if phishing_probability is None:

            phishing_probability = float(
                self.model.predict_proba(
                    df_processed
                )
            )

        phishing_probability = (
            self._normalize_probability(
                phishing_probability
            )
        )

        feature_names = list(
            self.feature_schema
        )

        # --------------------------------------------------------------------
        # Preferred: SHAP
        # --------------------------------------------------------------------

        if self.shap_explainer is not None:

            try:

                shap_values = (
                    self.shap_explainer.shap_values(
                        df_processed
                    )
                )

                # Different SHAP/XGBoost versions return:
                #   array
                #   [array_class0, array_class1]
                #   Explanation object
                if hasattr(
                    shap_values,
                    "values"
                ):
                    shap_values = (
                        shap_values.values
                    )

                if isinstance(
                    shap_values,
                    list
                ):

                    if len(shap_values) >= 2:
                        shap_values = shap_values[1]
                    elif len(shap_values) == 1:
                        shap_values = shap_values[0]

                shap_array = np.asarray(
                    shap_values,
                    dtype=float
                )

                if shap_array.ndim == 3:
                    # Usually [samples, features, classes].
                    shap_array = shap_array[
                        0,
                        :,
                        -1
                    ]

                elif shap_array.ndim == 2:
                    shap_array = shap_array[
                        0
                    ]

                elif shap_array.ndim != 1:
                    raise RuntimeError(
                        "Unexpected SHAP output shape: "
                        f"{shap_array.shape}"
                    )

                if len(shap_array) != len(
                    feature_names
                ):
                    raise RuntimeError(
                        "SHAP feature count mismatch: "
                        f"{len(shap_array)} vs "
                        f"{len(feature_names)}"
                    )

                order = np.argsort(
                    np.abs(shap_array)
                )[::-1]

                factors = []

                for index in order[:10]:

                    impact = float(
                        shap_array[index]
                    )

                    feature_name = (
                        feature_names[index]
                    )

                    try:
                        value = float(
                            df_processed.iloc[0][
                                feature_name
                            ]
                        )
                    except Exception:
                        value = None

                    factors.append(
                        {
                            "feature":
                                feature_name,

                            "value":
                                value,

                            "impact":
                                round(
                                    impact,
                                    6
                                ),

                            "direction":
                                (
                                    "increases risk"
                                    if impact > 0
                                    else
                                    "decreases risk"
                                    if impact < 0
                                    else
                                    "neutral"
                                )
                        }
                    )

                positive = [
                    item
                    for item in factors
                    if item["impact"] > 0
                ]

                negative = [
                    item
                    for item in factors
                    if item["impact"] < 0
                ]

                return {
                    "method":
                        "shap",

                    "summary":
                        (
                            "HTML XGBoost model explanation "
                            "using per-sample SHAP feature impacts."
                        ),

                    "top_shap_factors":
                        factors,

                    "top_factors":
                        factors,

                    "risk_factors":
                        positive,

                    "protective_factors":
                        negative,

                    "shap_status":
                        "ready",

                    "shap_error":
                        None,

                    "phishing_probability":
                        round(
                            phishing_probability,
                            4
                        )
                }

            except Exception as exc:

                logger.warning(
                    "Per-sample HTML SHAP explanation failed: %s",
                    exc
                )

                self.shap_error = str(
                    exc
                )

        # --------------------------------------------------------------------
        # Secondary: trained-model feature importance
        # --------------------------------------------------------------------

        return self._feature_importance_explanation(
            df_processed,
            phishing_probability
        )

    def _get_explanation_fallback(
        self,
        df_processed: pd.DataFrame,
        phishing_probability: float,
        error_message: str
    ) -> Dict[str, Any]:
        """
        Last-resort explanation response.

        This is deliberately not labeled as a heuristic explanation because
        the prediction itself was still generated by XGBoost.
        """

        explanation = (
            self._feature_importance_explanation(
                df_processed,
                phishing_probability
            )
        )

        explanation[
            "prediction_remains_xgboost"
        ] = True

        explanation[
            "explanation_error"
        ] = error_message

        return explanation

    # ========================================================================
    # THRESHOLD VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_threshold(
        threshold: Any
    ) -> float:
        """
        Validate the prediction threshold.

        Valid range:

            0.0 <= threshold <= 1.0
        """

        try:

            threshold = float(
                threshold
            )

        except (
            TypeError,
            ValueError
        ):

            logger.warning(
                "Invalid threshold '%s'. "
                "Using 0.55.",
                threshold
            )

            return 0.55

        if not np.isfinite(
            threshold
        ):

            logger.warning(
                "Non-finite threshold. "
                "Using 0.55."
            )

            return 0.55

        if not (
            0.0 <= threshold <= 1.0
        ):

            logger.warning(
                "Threshold %.4f is outside "
                "the valid range. Using 0.55.",
                threshold
            )

            return 0.55

        return threshold

    # ========================================================================
    # PROBABILITY NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_probability(
        probability: Any
    ) -> float:
        """
        Normalize probability to [0.0, 1.0].
        """

        try:

            probability = float(
                probability
            )

        except (
            TypeError,
            ValueError,
            OverflowError
        ):

            return 0.0

        if not np.isfinite(
            probability
        ):

            return 0.0

        return max(
            0.0,
            min(
                1.0,
                probability
            )
        )

    # ========================================================================
    # ERROR RESULT
    # ========================================================================

    def _get_error_result(
        self,
        error_message: str
    ) -> Dict[str, Any]:
        """
        Return an explicit prediction error.

        IMPORTANT
        ---------
        This does NOT mean legitimate.

        signal_available=False tells HTMLAIAgent/Fusion that
        the predictor did not produce a usable prediction.
        """

        logger.error(
            "HTML Predictor error: %s",
            error_message
        )

        return {

            "agent":
                "html",

            "agent_name":
                self.AGENT_NAME,

            "agent_version":
                self.AGENT_VERSION,

            "analysis_status":
                "error",

            "signal_available":
                False,

            "model_status":
                "unavailable",

            "prediction_source":
                "none",

            "model_path":
                self.model_path,

            "class_index":
                None,

            "class_label":
                "unknown",

            "prediction":
                "unknown",

            "confidence":
                0.0,

            "confidence_score":
                0.0,

            "phishing_probability":
                None,

            "legitimate_probability":
                None,

            "class_probabilities":
                {
                    "legitimate":
                        None,

                    "phishing":
                        None
                },

            "threshold":
                round(
                    self.threshold,
                    4
                ),

            "fallback_used":
                False,

            "model_available":
                False,

            "feature_count":
                len(
                    self.feature_schema
                ),

            "error":
                error_message
        }

    # ========================================================================
    # MODEL STATUS
    # ========================================================================

    def get_model_status(
        self
    ) -> Dict[str, Any]:
        """
        Return detailed HTML model status.
        """

        model_available = False

        try:

            model_available = (
                self.model.is_available()
            )

        except Exception:

            model_available = False

        return {

            "agent":
                self.AGENT_NAME,

            "agent_version":
                self.AGENT_VERSION,

            "prediction_threshold":
                self.threshold,

            "feature_count":
                len(
                    self.feature_schema
                ),

            "expected_feature_count":
                self.EXPECTED_FEATURE_COUNT,

            "model_path":
                self.model_path,

            "model_available":
                model_available,

            "model":
                self.model.get_status()
        }

    # ========================================================================
    # PREDICTOR STATUS
    # ========================================================================

    def get_status(
        self
    ) -> Dict[str, Any]:
        """
        Return complete predictor status.
        """

        model_available = False

        try:

            model_available = (
                self.model.is_available()
            )

        except Exception:

            model_available = False

        return {

            "agent":
                self.AGENT_NAME,

            "agent_version":
                self.AGENT_VERSION,

            "status":
                (
                    "ready"
                    if model_available
                    else "degraded"
                ),

            "model_available":
                model_available,

            "model_path":
                self.model_path,

            "prediction_threshold":
                self.threshold,

            "fallback_threshold":
                self.FALLBACK_THRESHOLD,

            "feature_count":
                len(
                    self.feature_schema
                ),

            "expected_feature_count":
                self.EXPECTED_FEATURE_COUNT,

            "fallback_available":
                True,

            "explainability": {
                "method": self.shap_status,
                "shap_available": self.shap_explainer is not None,
                "shap_error": self.shap_error,
            },

            "model_status":
                self.model.get_status()
        }