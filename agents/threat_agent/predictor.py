"""
Threat Intelligence AI Agent - Prediction Engine
=================================================

File:
    agents/threat_agent/predictor.py

Purpose
-------
Production inference layer for the Threat Intelligence AI Agent.

Architecture
------------

    Unified Threat Features
            |
            v
    ThreatFeatureSchema
            |
            v
    ThreatPreprocessor
            |
            v
    Model-Compatible Feature Selection
            |
            v
    XGBoost Threat Model
            |
       +----+----+
       |         |
       v         v
      ML      Heuristic
       |         |
       +----+----+
            |
            v
       Standardized
        Prediction

Classification
--------------

    0 -> clean_reputation
    1 -> malicious_threat

Important Feature Design
------------------------

ThreatFeatureSchema remains the canonical INPUT schema.

The currently trained XGBoost artifact contains 13 features because
the dataset preparation stage removed 7 constant features.

Therefore:

    Canonical input schema = 20 features
    Current trained model = 13 features

The predictor automatically derives the model feature order from:

1. model.feature_names, when available
2. training metadata, when available
3. the known trained 13-feature artifact contract as a compatibility fallback

The predictor NEVER pads the XGBoost model with arbitrary zero-valued
features and NEVER changes the order of the trained model features.

Responsibilities
----------------

1. Load the trained Threat XGBoost model.
2. Detect the actual model feature count.
3. Resolve the model feature order.
4. Validate model compatibility.
5. Accept canonical threat feature dictionaries.
6. Select only features required by the trained model.
7. Preserve exact model feature order.
8. Run XGBoost inference.
9. Provide heuristic fallback if ML is unavailable.
10. Load training metrics.
11. Return a stable result contract.
12. Expose diagnostics for threat_agent.py and explain.py.

No standalone execution hook is included.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from .feature_schema import ThreatFeatureSchema
from .model import ThreatXGBoostModel
from .preprocessing import ThreatPreprocessor


logger = logging.getLogger(__name__)


class ThreatPredictor:
    """
    Production prediction engine for the Threat Intelligence Agent.

    The canonical project input remains the 20-feature ThreatFeatureSchema.

    The active trained model may use a reduced feature subset. The predictor
    resolves that subset dynamically and sends exactly the required features
    to XGBoost.
    """

    # =====================================================================
    # CLASSIFICATION CONTRACT
    # =====================================================================

    CLASS_LABELS: Dict[int, str] = {
        0: "clean_reputation",
        1: "malicious_threat",
    }

    POSITIVE_CLASS_INDEX: int = 1

    POSITIVE_CLASS_LABEL: str = (
        "malicious_threat"
    )

    DEFAULT_CLASS_INDEX: int = 0

    # =====================================================================
    # DEFAULT PATHS
    # =====================================================================

    DEFAULT_MODEL_PATH = (
        "data/models/threat_agent/"
        "xgboost_threat_model.pkl"
    )

    DEFAULT_METRICS_PATH = (
        "data/models/threat_agent/"
        "threat_training_metrics.json"
    )

    # =====================================================================
    # TRAINED 13-FEATURE COMPATIBILITY CONTRACT
    # =====================================================================
    #
    # This is NOT the canonical input schema.
    #
    # It represents the feature subset produced by the current training
    # pipeline after constant-feature removal.
    #
    # The model itself remains the primary source whenever it exposes
    # feature names.
    #

    TRAINED_MODEL_FEATURES_FALLBACK: List[str] = [
        "is_blacklisted",
        "blacklist_vendors_count",
        "malicious_engines_count",
        "suspicious_engines_count",
        "harmless_engines_count",
        "malicious_ratio",
        "suspicion_ratio",
        "is_malware_associated",
        "is_exploit_distributor",
        "malware_threat_score",
        "reputation_score",
        "weighted_threat_score",
        "overall_threat_score",
    ]

    # =====================================================================
    # HEURISTIC CONFIGURATION
    # =====================================================================

    HEURISTIC_WEIGHTS: Dict[str, float] = {
        "is_blacklisted": 0.85,
        "high_authority_vendor_flagged": 0.40,
        "is_c2_node": 0.90,
        "is_malware_associated": 0.50,
        "is_exploit_distributor": 0.60,
        "has_high_threat_consensus": 0.45,
        "is_phishing_associated": 0.65,
        "malicious_ratio": 0.40,
        "suspicion_ratio": 0.15,
        "overall_threat_score": 0.35,
        "weighted_threat_score": 0.25,
        "malware_threat_score": 0.30,
        "blacklist_vendors_count": 0.20,
    }

    # =====================================================================
    # CONSTRUCTOR
    # =====================================================================

    def __init__(
        self,
        model_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
        allow_heuristic_fallback: bool = True,
    ) -> None:
        """
        Initialize the Threat Predictor.

        Parameters
        ----------
        model_path:
            Optional path to the trained XGBoost model.

        metrics_path:
            Optional path to training metrics.

        allow_heuristic_fallback:
            If True, use deterministic heuristic inference when ML loading
            or ML prediction fails.
        """

        # -----------------------------------------------------------------
        # Paths
        # -----------------------------------------------------------------

        self.model_path = (
            str(model_path)
            if model_path
            else self.DEFAULT_MODEL_PATH
        )

        self.metrics_path = (
            str(metrics_path)
            if metrics_path
            else self.DEFAULT_METRICS_PATH
        )

        self.allow_heuristic_fallback = bool(
            allow_heuristic_fallback
        )

        # -----------------------------------------------------------------
        # Canonical input schema
        # -----------------------------------------------------------------

        self.feature_schema: List[str] = (
            ThreatFeatureSchema.get_schema_columns()
        )

        self.feature_count: int = (
            ThreatFeatureSchema.get_feature_count()
        )

        if len(self.feature_schema) != self.feature_count:
            raise RuntimeError(
                "ThreatFeatureSchema feature count is inconsistent."
            )

        # -----------------------------------------------------------------
        # Preprocessor
        # -----------------------------------------------------------------

        self.preprocessor = ThreatPreprocessor()

        # -----------------------------------------------------------------
        # Model wrapper
        # -----------------------------------------------------------------

        self.model_wrapper = ThreatXGBoostModel(
            model_path=self.model_path
        )

        # -----------------------------------------------------------------
        # Runtime model state
        # -----------------------------------------------------------------

        self.model: Optional[Any] = None

        self.is_model_loaded: bool = False

        self.model_load_error: Optional[str] = None

        # -----------------------------------------------------------------
        # Actual model feature contract
        # -----------------------------------------------------------------

        self.model_feature_schema: List[str] = []

        self.model_feature_count: int = 0

        self.model_feature_source: str = (
            "unresolved"
        )

        # -----------------------------------------------------------------
        # Load model
        # -----------------------------------------------------------------

        self._load_active_model()

        logger.info(
            "ThreatPredictor initialized | "
            "canonical_features=%d | "
            "model_features=%d | "
            "model_loaded=%s | "
            "fallback=%s",
            self.feature_count,
            self.model_feature_count,
            self.is_model_loaded,
            self.allow_heuristic_fallback,
        )

    # =====================================================================
    # MODEL LOADING
    # =====================================================================

    def _load_active_model(self) -> None:
        """
        Load the active XGBoost artifact.

        The model wrapper in the current project expects the canonical
        20-feature schema during validation. The trained artifact, however,
        contains the reduced 13-feature matrix.

        To support the currently trained model safely, this method performs
        direct artifact inspection first and resolves the actual model
        feature names/count before deciding whether the model is usable.
        """

        try:
            path = Path(self.model_path).expanduser()

            if not path.exists():
                raise FileNotFoundError(
                    f"Threat model artifact does not exist: {path}"
                )

            if not path.is_file():
                raise ValueError(
                    f"Threat model path is not a file: {path}"
                )

            # -------------------------------------------------------------
            # Load artifact directly.
            #
            # This is necessary because the current model wrapper validates
            # the artifact against the old 20-feature schema.
            # -------------------------------------------------------------

            artifact = joblib.load(path)

            model = self._extract_xgb_model(
                artifact
            )

            if model is None:
                raise TypeError(
                    "Threat model artifact does not contain "
                    "a supported XGBClassifier."
                )

            # -------------------------------------------------------------
            # Resolve actual model feature order.
            # -------------------------------------------------------------

            model_features = (
                self._resolve_model_feature_order(
                    model=model,
                    artifact=artifact,
                )
            )

            model_count = self._get_actual_model_feature_count(
                model=model
            )

            if model_count <= 0:
                raise ValueError(
                    "Unable to determine the Threat XGBoost "
                    "model feature count."
                )

            if len(model_features) != model_count:
                raise ValueError(
                    "Resolved Threat model feature count mismatch: "
                    f"resolved={len(model_features)}, "
                    f"model={model_count}"
                )

            # -------------------------------------------------------------
            # Validate every model feature belongs to canonical schema.
            # -------------------------------------------------------------

            canonical_set = set(
                self.feature_schema
            )

            invalid_features = [
                feature
                for feature in model_features
                if feature not in canonical_set
            ]

            if invalid_features:
                raise ValueError(
                    "Threat model contains features outside the "
                    "canonical ThreatFeatureSchema: "
                    + ", ".join(invalid_features)
                )

            # -------------------------------------------------------------
            # Validate uniqueness.
            # -------------------------------------------------------------

            if len(set(model_features)) != len(model_features):
                raise ValueError(
                    "Threat model feature order contains duplicates."
                )

            # -------------------------------------------------------------
            # Register model.
            # -------------------------------------------------------------

            self.model = model

            self.model_feature_schema = list(
                model_features
            )

            self.model_feature_count = int(
                model_count
            )

            self.is_model_loaded = True

            self.model_load_error = None

            logger.info(
                "Threat XGBoost model loaded successfully | "
                "canonical_features=%d | model_features=%d | "
                "feature_source=%s",
                self.feature_count,
                self.model_feature_count,
                self.model_feature_source,
            )

            logger.info(
                "Threat model features: %s",
                self.model_feature_schema,
            )

        except Exception as exc:

            self.model = None

            self.is_model_loaded = False

            self.model_load_error = str(
                exc
            )

            self.model_feature_schema = []

            self.model_feature_count = 0

            if self.allow_heuristic_fallback:

                logger.warning(
                    "ThreatPredictor: ML model unavailable. "
                    "Heuristic fallback enabled. Reason: %s",
                    exc,
                )

            else:

                logger.error(
                    "ThreatPredictor: ML model unavailable "
                    "and heuristic fallback disabled. "
                    "Reason: %s",
                    exc,
                )

    # =====================================================================
    # XGBOOST MODEL EXTRACTION
    # =====================================================================

    @staticmethod
    def _extract_xgb_model(
        artifact: Any,
    ) -> Optional[Any]:
        """
        Extract XGBClassifier from supported artifact formats.
        """

        # -------------------------------------------------------------
        # Direct XGBClassifier
        # -------------------------------------------------------------

        try:
            import xgboost as xgb

            if isinstance(
                artifact,
                xgb.XGBClassifier,
            ):
                return artifact

        except Exception:
            pass

        # -------------------------------------------------------------
        # Structured artifact
        # -------------------------------------------------------------

        if isinstance(
            artifact,
            dict,
        ):

            model = artifact.get(
                "model"
            )

            try:
                import xgboost as xgb

                if isinstance(
                    model,
                    xgb.XGBClassifier,
                ):
                    return model

            except Exception:
                return None

        return None

    # =====================================================================
    # MODEL FEATURE ORDER RESOLUTION
    # =====================================================================

    def _resolve_model_feature_order(
        self,
        model: Any,
        artifact: Any,
    ) -> List[str]:
        """
        Resolve the exact feature order used by the trained model.

        Priority:

        1. XGBoost model.feature_names
        2. Structured artifact feature_order
        3. Structured artifact schema metadata
        4. Current 13-feature training contract
        """

        # -------------------------------------------------------------
        # Priority 1: XGBoost feature names
        # -------------------------------------------------------------

        feature_names = getattr(
            model,
            "feature_names_in_",
            None,
        )

        if feature_names is not None:

            try:
                names = [
                    str(x)
                    for x in feature_names
                ]

                if names:
                    self.model_feature_source = (
                        "xgboost_feature_names"
                    )

                    return names

            except Exception:
                pass

        # -------------------------------------------------------------
        # Some XGBoost versions expose names through booster.
        # -------------------------------------------------------------

        try:

            booster = model.get_booster()

            booster_names = getattr(
                booster,
                "feature_names",
                None,
            )

            if booster_names:

                names = [
                    str(x)
                    for x in booster_names
                ]

                self.model_feature_source = (
                    "xgboost_booster_feature_names"
                )

                return names

        except Exception:
            pass

        # -------------------------------------------------------------
        # Priority 2: structured artifact feature_order
        # -------------------------------------------------------------

        if isinstance(
            artifact,
            dict,
        ):

            stored_order = artifact.get(
                "feature_order"
            )

            if isinstance(
                stored_order,
                (list, tuple),
            ):

                names = [
                    str(x)
                    for x in stored_order
                ]

                if names:

                    self.model_feature_source = (
                        "artifact_feature_order"
                    )

                    return names

        # -------------------------------------------------------------
        # Priority 3: schema metadata
        # -------------------------------------------------------------

        if isinstance(
            artifact,
            dict,
        ):

            schema_metadata = artifact.get(
                "schema_metadata",
                {},
            )

            if isinstance(
                schema_metadata,
                dict,
            ):

                for key in (
                    "feature_order",
                    "columns",
                    "feature_columns",
                ):

                    stored_order = (
                        schema_metadata.get(key)
                    )

                    if isinstance(
                        stored_order,
                        (list, tuple),
                    ):

                        names = [
                            str(x)
                            for x in stored_order
                        ]

                        if names:

                            self.model_feature_source = (
                                "artifact_schema_metadata"
                            )

                            return names

        # -------------------------------------------------------------
        # Priority 4: current trained 13-feature contract
        # -------------------------------------------------------------

        model_count = self._get_actual_model_feature_count(
            model
        )

        if model_count == len(
            self.TRAINED_MODEL_FEATURES_FALLBACK
        ):

            self.model_feature_source = (
                "training_pipeline_13_feature_fallback"
            )

            return list(
                self.TRAINED_MODEL_FEATURES_FALLBACK
            )

        raise ValueError(
            "Unable to resolve model feature names. "
            f"Model reports {model_count} features."
        )

    # =====================================================================
    # MODEL FEATURE COUNT
    # =====================================================================

    @staticmethod
    def _get_actual_model_feature_count(
        model: Any,
    ) -> int:
        """
        Determine the actual number of features expected by XGBoost.
        """

        # -------------------------------------------------------------
        # n_features_in_
        # -------------------------------------------------------------

        value = getattr(
            model,
            "n_features_in_",
            None,
        )

        if value is not None:

            try:
                return int(value)
            except (
                TypeError,
                ValueError,
            ):
                pass

        # -------------------------------------------------------------
        # Booster num_features()
        # -------------------------------------------------------------

        try:

            booster = model.get_booster()

            value = booster.num_features()

            return int(value)

        except Exception:
            pass

        # -------------------------------------------------------------
        # feature_names_in_
        # -------------------------------------------------------------

        names = getattr(
            model,
            "feature_names_in_",
            None,
        )

        if names is not None:

            try:
                return len(names)
            except Exception:
                pass

        return 0

    # =====================================================================
    # MODEL ACCESS
    # =====================================================================

    def get_model(
        self,
    ) -> Optional[Any]:
        """
        Return the loaded XGBoost model.
        """

        return self.model

    # =====================================================================
    # MODEL STATUS
    # =====================================================================

    def get_model_status(
        self,
    ) -> str:
        """
        Return current model operational status.
        """

        if (
            self.is_model_loaded
            and self.model is not None
        ):
            return "loaded_ml_model"

        if self.allow_heuristic_fallback:
            return "fallback_heuristic"

        return "model_unavailable"

    # =====================================================================
    # FEATURE SCHEMA
    # =====================================================================

    def get_feature_schema(
        self,
    ) -> List[str]:
        """
        Return the canonical 20-feature input schema.

        This is the schema expected from upstream agents.
        """

        return list(
            self.feature_schema
        )

    # =====================================================================
    # MODEL FEATURE SCHEMA
    # =====================================================================

    def get_model_feature_schema(
        self,
    ) -> List[str]:
        """
        Return the exact feature order required by the loaded model.
        """

        return list(
            self.model_feature_schema
        )

    # =====================================================================
    # FEATURE COUNT
    # =====================================================================

    def get_feature_count(
        self,
    ) -> int:
        """
        Return the canonical input feature count.

        The canonical Threat schema currently contains 20 features.
        """

        return int(
            self.feature_count
        )

    # =====================================================================
    # MODEL FEATURE COUNT
    # =====================================================================

    def get_model_feature_count(
        self,
    ) -> int:
        """
        Return the actual number of features used by the trained model.
        """

        return int(
            self.model_feature_count
        )

    # =====================================================================
    # TRAINING METRICS
    # =====================================================================

    def _get_training_metrics(
        self,
    ) -> Dict[str, Any]:
        """
        Load offline training metrics.

        Metrics are informational and never block inference.
        """

        path = (
            Path(
                self.metrics_path
            )
            .expanduser()
        )

        if not path.exists():

            return {
                "metrics_status": "unavailable",
                "notice": (
                    "Training metrics unavailable. "
                    "Run the Threat Agent trainer first."
                ),
            }

        if not path.is_file():

            return {
                "metrics_status": "invalid_path",
                "notice": (
                    "Threat training metrics path is not a file."
                ),
            }

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as handle:

                metrics = json.load(
                    handle
                )

            if not isinstance(
                metrics,
                dict,
            ):

                return {
                    "metrics_status": "invalid_format",
                    "notice": (
                        "Threat training metrics must "
                        "contain a JSON object."
                    ),
                }

            return {
                "metrics_status": "available",
                "metrics": metrics,
            }

        except Exception as exc:

            logger.warning(
                "Unable to load Threat training metrics: %s",
                exc,
            )

            return {
                "metrics_status": "load_error",
                "notice": str(exc),
            }

    # =====================================================================
    # INPUT NORMALIZATION
    # =====================================================================

    def _normalize_input_features(
        self,
        features: Mapping[str, Any],
    ) -> Dict[str, float]:
        """
        Normalize an input feature dictionary.

        Missing canonical features are represented as 0.0.

        Extra fields are ignored by the model input layer.
        """

        if not isinstance(
            features,
            Mapping,
        ):
            raise TypeError(
                "Threat features must be a mapping/dictionary."
            )

        normalized: Dict[str, float] = {}

        for feature in self.feature_schema:

            value = features.get(
                feature,
                0.0,
            )

            try:

                numeric = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                numeric = 0.0

            if not math.isfinite(
                numeric
            ):

                numeric = 0.0

            normalized[
                feature
            ] = numeric

        return normalized

    # =====================================================================
    # MODEL INPUT PREPARATION
    # =====================================================================

    def _prepare_model_input(
        self,
        features: Mapping[str, Any],
    ) -> pd.DataFrame:
        """
        Build the exact DataFrame expected by the trained XGBoost model.
        """

        if not self.model_feature_schema:

            raise RuntimeError(
                "Model feature schema is unavailable."
            )

        normalized = (
            self._normalize_input_features(
                features
            )
        )

        missing_model_features = [
            feature
            for feature in self.model_feature_schema
            if feature not in normalized
        ]

        if missing_model_features:

            raise ValueError(
                "Input is missing model-required features: "
                + ", ".join(
                    missing_model_features
                )
            )

        # -------------------------------------------------------------
        # Exact model order.
        # -------------------------------------------------------------

        values = [
            normalized[feature]
            for feature in self.model_feature_schema
        ]

        frame = pd.DataFrame(
            [values],
            columns=self.model_feature_schema,
            dtype=float,
        )

        # -------------------------------------------------------------
        # Final safety checks.
        # -------------------------------------------------------------

        if frame.shape[1] != self.model_feature_count:

            raise ValueError(
                "Prepared model input feature count mismatch: "
                f"input={frame.shape[1]}, "
                f"model={self.model_feature_count}"
            )

        if list(frame.columns) != (
            self.model_feature_schema
        ):

            raise ValueError(
                "Prepared model input feature order mismatch."
            )

        return frame

    # =====================================================================
    # FEATURE DIAGNOSTICS
    # =====================================================================

    def _build_feature_diagnostics(
        self,
        features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Build diagnostics describing canonical and model-level feature
        alignment.
        """

        supplied = set(
            features.keys()
        )

        canonical = set(
            self.feature_schema
        )

        model_features = set(
            self.model_feature_schema
        )

        missing_canonical = [
            feature
            for feature in self.feature_schema
            if feature not in supplied
        ]

        extra_features = sorted(
            supplied - canonical
        )

        missing_model_features = [
            feature
            for feature in self.model_feature_schema
            if feature not in supplied
        ]

        unused_canonical_features = [
            feature
            for feature in self.feature_schema
            if feature not in model_features
        ]

        return {
            "supplied_feature_count": len(
                supplied
            ),
            "canonical_feature_count": self.feature_count,
            "model_feature_count": self.model_feature_count,
            "recognized_feature_count": len(
                supplied & canonical
            ),
            "missing_canonical_features": (
                missing_canonical
            ),
            "missing_canonical_feature_count": len(
                missing_canonical
            ),
            "missing_model_features": (
                missing_model_features
            ),
            "missing_model_feature_count": len(
                missing_model_features
            ),
            "extra_features": extra_features,
            "extra_feature_count": len(
                extra_features
            ),
            "model_unused_canonical_features": (
                unused_canonical_features
            ),
            "model_unused_feature_count": len(
                unused_canonical_features
            ),
            "model_feature_order": list(
                self.model_feature_schema
            ),
            "model_feature_source": (
                self.model_feature_source
            ),
        }

    # =====================================================================
    # ML PREDICTION
    # =====================================================================

    def _predict_with_ml(
        self,
        features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Run XGBoost inference.
        """

        if not self.is_model_loaded:

            raise RuntimeError(
                "Threat ML model is not loaded."
            )

        if self.model is None:

            raise RuntimeError(
                "Threat ML model object is unavailable."
            )

        frame = (
            self._prepare_model_input(
                features
            )
        )

        # -------------------------------------------------------------
        # Probability prediction
        # -------------------------------------------------------------

        probabilities = (
            self.model.predict_proba(
                frame
            )
        )

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        if probabilities.ndim != 2:

            raise ValueError(
                "Threat model returned invalid probability shape: "
                f"{probabilities.shape}"
            )

        if probabilities.shape[0] != 1:

            raise ValueError(
                "Threat predictor expected one prediction row."
            )

        if probabilities.shape[1] < 2:

            raise ValueError(
                "Threat model must provide binary class probabilities."
            )

        resolved = self._resolve_binary_outcome(
            probabilities[0][0],
            probabilities[0][1],
        )

        if resolved is None:

            logger.warning(
                "Threat model returned degenerate class "
                "probabilities. Using heuristic fallback."
            )

            fallback_result = (
                self._predict_with_heuristic(
                    features
                )
            )

            fallback_result[
                "fallback_reason"
            ] = "degenerate_class_probabilities"

            return fallback_result

        (
            prediction_index,
            clean_probability,
            threat_probability,
            confidence,
        ) = resolved

        prediction_label = (
            self.CLASS_LABELS[
                prediction_index
            ]
        )

        return {
            "prediction_index": prediction_index,
            "prediction": prediction_label,
            "class_label": prediction_label,
            "clean_probability": round(
                clean_probability,
                6,
            ),
            "threat_probability": round(
                threat_probability,
                6,
            ),
            "confidence": round(
                confidence,
                6,
            ),
            "inference_method": "xgboost",
            "model_status": "loaded_ml_model",
            "feature_count": self.model_feature_count,
            "canonical_feature_count": self.feature_count,
            "model_features": list(
                self.model_feature_schema
            ),
            "feature_diagnostics": (
                self._build_feature_diagnostics(
                    features
                )
            ),
            "training_evaluation": (
                self._get_training_metrics()
            ),
        }

    # =====================================================================
    # HEURISTIC FALLBACK
    # =====================================================================

    def _predict_with_heuristic(
        self,
        features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Deterministic fallback prediction.

        This method is used only when the ML model is unavailable or
        inference fails.
        """

        normalized = (
            self._normalize_input_features(
                features
            )
        )

        weighted_score = 0.0

        total_weight = 0.0

        contributions: Dict[str, float] = {}

        for feature, weight in (
            self.HEURISTIC_WEIGHTS.items()
        ):

            if feature not in normalized:
                continue

            value = float(
                normalized.get(
                    feature,
                    0.0,
                )
            )

            # ---------------------------------------------------------
            # Normalize common integer count fields.
            # ---------------------------------------------------------

            if feature == (
                "blacklist_vendors_count"
            ):

                value = min(
                    max(
                        value / 10.0,
                        0.0,
                    ),
                    1.0,
                )

            else:

                value = min(
                    max(
                        value,
                        0.0,
                    ),
                    1.0,
                )

            contribution = (
                value * weight
            )

            contributions[
                feature
            ] = contribution

            weighted_score += (
                contribution
            )

            total_weight += (
                weight
            )

        if total_weight <= 0:

            threat_probability = 0.0

        else:

            threat_probability = (
                weighted_score
                / total_weight
            )

        threat_probability = (
            self._clamp_probability(
                threat_probability
            )
        )

        clean_probability = (
            1.0
            - threat_probability
        )

        resolved = self._resolve_binary_outcome(
            clean_probability,
            threat_probability,
        )

        if resolved is None:

            prediction_index = 0
            clean_probability = 1.0
            threat_probability = 0.0
            confidence = 1.0

        else:

            (
                prediction_index,
                clean_probability,
                threat_probability,
                confidence,
            ) = resolved

        prediction_label = (
            self.CLASS_LABELS[
                prediction_index
            ]
        )

        top_factors = sorted(
            contributions.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]

        return {
            "prediction_index": prediction_index,
            "prediction": prediction_label,
            "class_label": prediction_label,
            "clean_probability": round(
                clean_probability,
                6,
            ),
            "threat_probability": round(
                threat_probability,
                6,
            ),
            "confidence": round(
                confidence,
                6,
            ),
            "inference_method": (
                "heuristic_fallback"
            ),
            "model_status": (
                "fallback_heuristic"
            ),
            "feature_count": (
                self.model_feature_count
                if self.model_feature_count
                else self.feature_count
            ),
            "canonical_feature_count": (
                self.feature_count
            ),
            "model_features": list(
                self.model_feature_schema
            ),
            "feature_diagnostics": (
                self._build_feature_diagnostics(
                    features
                )
            ),
            "heuristic_score": round(
                threat_probability,
                6,
            ),
            "heuristic_top_factors": [
                {
                    "feature": feature,
                    "contribution": round(
                        contribution,
                        6,
                    ),
                }
                for feature, contribution
                in top_factors
            ],
            "training_evaluation": (
                self._get_training_metrics()
            ),
        }

    # =====================================================================
    # PUBLIC PREDICT METHOD
    # =====================================================================

    def predict(
        self,
        features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Predict whether the supplied threat intelligence evidence is
        associated with a malicious threat.

        Parameters
        ----------
        features:
            Dictionary containing canonical ThreatFeatureSchema features.

        Returns
        -------
        Dict[str, Any]
            Stable Threat prediction result.
        """

        if not isinstance(
            features,
            Mapping,
        ):

            raise TypeError(
                "ThreatPredictor.predict() expects "
                "a feature dictionary."
            )

        # -------------------------------------------------------------
        # ML path
        # -------------------------------------------------------------

        if (
            self.is_model_loaded
            and self.model is not None
        ):

            try:

                return self._predict_with_ml(
                    features
                )

            except Exception as exc:

                logger.error(
                    "Threat ML prediction failed: %s",
                    exc,
                    exc_info=True,
                )

                if not self.allow_heuristic_fallback:

                    raise

                logger.warning(
                    "Switching to Threat heuristic fallback."
                )

                fallback_result = (
                    self._predict_with_heuristic(
                        features
                    )
                )

                fallback_result[
                    "fallback_reason"
                ] = str(exc)

                return fallback_result

        # -------------------------------------------------------------
        # Fallback path
        # -------------------------------------------------------------

        if self.allow_heuristic_fallback:

            return (
                self._predict_with_heuristic(
                    features
                )
            )

        raise RuntimeError(
            "Threat ML model is unavailable and "
            "heuristic fallback is disabled."
        )

    # =====================================================================
    # STATUS / DIAGNOSTICS
    # =====================================================================

    def get_model_diagnostics(
        self,
    ) -> Dict[str, Any]:
        """
        Return detailed model/schema diagnostics.
        """

        canonical_set = set(
            self.feature_schema
        )

        model_set = set(
            self.model_feature_schema
        )

        return {
            "model_path": str(
                self.model_path
            ),
            "model_exists": Path(
                self.model_path
            ).expanduser().is_file(),
            "model_loaded": bool(
                self.is_model_loaded
            ),
            "model_status": (
                self.get_model_status()
            ),
            "model_load_error": (
                self.model_load_error
            ),
            "canonical_feature_count": (
                self.feature_count
            ),
            "canonical_features": list(
                self.feature_schema
            ),
            "model_feature_count": (
                self.model_feature_count
            ),
            "model_features": list(
                self.model_feature_schema
            ),
            "model_feature_source": (
                self.model_feature_source
            ),
            "model_features_outside_canonical_schema": sorted(
                model_set - canonical_set
            ),
            "canonical_features_not_used_by_model": [
                feature
                for feature in self.feature_schema
                if feature not in model_set
            ],
        }

    # =====================================================================
    # UTILITY
    # =====================================================================

    def _resolve_binary_outcome(
        self,
        clean_probability: Any,
        threat_probability: Any,
    ) -> Optional[tuple]:
        """
        Normalize a clean/malicious probability pair.

        Returns None when both class probabilities are degenerate
        (zero/invalid). Callers must not treat that state as a
        successful phishing prediction with confidence 0.
        """

        clean_probability = self._clamp_probability(
            clean_probability
        )

        threat_probability = self._clamp_probability(
            threat_probability
        )

        total = (
            clean_probability
            + threat_probability
        )

        if total <= 0.0:
            return None

        clean_probability /= total
        threat_probability /= total

        prediction_index = int(
            1
            if threat_probability >= clean_probability
            else 0
        )

        confidence = max(
            clean_probability,
            threat_probability,
        )

        return (
            prediction_index,
            clean_probability,
            threat_probability,
            confidence,
        )

    @staticmethod
    def _clamp_probability(
        value: float,
    ) -> float:
        """
        Clamp a probability to [0, 1].
        """

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if not math.isfinite(
            value
        ):
            return 0.0

        return min(
            max(
                value,
                0.0,
            ),
            1.0,
        )


# =====================================================================
# MODULE-LEVEL HELPER
# =====================================================================

def predict_threat(
    features: Mapping[str, Any],
    model_path: Optional[str] = None,
    allow_heuristic_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function for Threat prediction.
    """

    predictor = ThreatPredictor(
        model_path=model_path,
        allow_heuristic_fallback=allow_heuristic_fallback,
    )

    return predictor.predict(
        features
    )