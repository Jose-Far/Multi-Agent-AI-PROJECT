"""
Threat Intelligence AI Agent - Explainability Engine
=====================================================

Production explainability layer for the Threat Intelligence Agent.

Responsibilities
----------------
1. Use ThreatFeatureSchema as the canonical source of truth.
2. Support models trained on a subset of canonical features.
3. Align SHAP input to the exact trained model feature order.
4. Provide SHAP explanations when possible.
5. Provide deterministic heuristic explanations when SHAP is unavailable.
6. Never crash the Threat Agent because explainability fails.

Current project contract
------------------------
Canonical features : 20
Current trained model: 19

The canonical-only feature is:

    has_high_threat_consensus

The trained model does not consume that feature.

No standalone execution hook is included.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .feature_schema import ThreatFeatureSchema


# =============================================================================
# OPTIONAL SHAP
# =============================================================================

try:
    import shap

    SHAP_AVAILABLE = True

except ImportError:
    shap = None
    SHAP_AVAILABLE = False


logger = logging.getLogger(__name__)


class ThreatExplainer:
    """
    Explain Threat Intelligence predictions.

    Classification:

        0 -> clean_reputation
        1 -> malicious_threat

    Important:
        The canonical schema contains 20 features, but the trained
        XGBoost model may use a subset of those features.

        SHAP therefore uses model_feature_schema rather than blindly
        passing the complete canonical schema.
    """

    # =========================================================================
    # CLASS CONTRACT
    # =========================================================================

    COMPONENT_NAME = "ThreatExplainer"

    CLEAN_CLASS = "clean_reputation"
    MALICIOUS_CLASS = "malicious_threat"

    CLASS_INDEX_MAP: Dict[str, int] = {
        CLEAN_CLASS: 0,
        MALICIOUS_CLASS: 1,
    }

    # =========================================================================
    # EXPLANATION SETTINGS
    # =========================================================================

    DEFAULT_TOP_K = 5
    MIN_ABSOLUTE_SHAP_IMPACT = 0.0001

    # =========================================================================
    # HUMAN-READABLE DESCRIPTIONS
    # =========================================================================

    FEATURE_DESCRIPTIONS: Dict[str, str] = {
        "is_blacklisted":
            "Global Threat Feed Blacklist Status",

        "blacklist_vendors_count":
            "Count of Reporting Blacklist Feeds",

        "high_authority_vendor_flagged":
            "Tier-1 Authoritative Threat Feed Detection",

        "malicious_engines_count":
            "Aggregated Multi-Engine Malicious Detections",

        "suspicious_engines_count":
            "Aggregated Multi-Engine Suspicious Detections",

        "harmless_engines_count":
            "Aggregated Multi-Engine Harmless Verifications",

        "undetected_engines_count":
            "Threat Engines Reporting No Detection",

        "malicious_ratio":
            "Malicious Detections Relative to Scanned Engines",

        "suspicion_ratio":
            "Suspicious Detections Relative to Scanned Engines",

        "has_high_threat_consensus":
            "Strong Multi-Vendor Threat Consensus",

        "is_malware_associated":
            "Known Malware Association",

        "is_c2_node":
            "Command-and-Control Infrastructure Association",

        "is_exploit_distributor":
            "Known Exploit Distribution Association",

        "malware_threat_score":
            "Composite Malware Threat Severity Score",

        "reputation_score":
            "Normalized OSINT Community Reputation Score",

        "weighted_threat_score":
            "Weighted Multi-Engine Threat Consensus Score",

        "overall_threat_score":
            "Aggregated Threat Intelligence Score",

        "is_zero_day_candidate":
            "Potential Zero-Day or Previously Uncatalogued Threat",

        "days_since_first_submission":
            "Days Since First Threat Intelligence Submission",

        "days_since_last_analysis":
            "Days Since Most Recent Threat Intelligence Analysis",
    }

    # =========================================================================
    # HEURISTIC WEIGHTS
    # =========================================================================

    HEURISTIC_WEIGHTS: Dict[str, float] = {
        "is_blacklisted": 0.95,
        "high_authority_vendor_flagged": 0.90,
        "is_c2_node": 1.00,
        "is_malware_associated": 0.95,
        "is_exploit_distributor": 0.95,
        "has_high_threat_consensus": 0.90,

        "malicious_ratio": 0.85,
        "malicious_engines_count": 0.80,
        "blacklist_vendors_count": 0.75,
        "malware_threat_score": 0.75,
        "overall_threat_score": 0.70,
        "weighted_threat_score": 0.65,

        "suspicion_ratio": 0.55,
        "suspicious_engines_count": 0.50,

        "is_zero_day_candidate": 0.45,

        "reputation_score": 0.25,

        "harmless_engines_count": -0.45,
        "undetected_engines_count": -0.10,
    }

    # =========================================================================
    # CONSTRUCTOR
    # =========================================================================

    def __init__(
        self,
        model: Any = None,
        top_k: int = DEFAULT_TOP_K,
        min_absolute_shap_impact: float = MIN_ABSOLUTE_SHAP_IMPACT,
    ) -> None:

        # ---------------------------------------------------------------------
        # Canonical schema
        # ---------------------------------------------------------------------

        self.feature_schema: List[str] = (
            self._get_canonical_feature_schema()
        )

        self.feature_count: int = (
            len(self.feature_schema)
        )

        self._validate_schema_contract()

        # ---------------------------------------------------------------------
        # Feature metadata
        # ---------------------------------------------------------------------

        self._validate_feature_metadata_contract()

        # ---------------------------------------------------------------------
        # Model
        # ---------------------------------------------------------------------

        self.model = model

        # ---------------------------------------------------------------------
        # Model feature schema
        # ---------------------------------------------------------------------

        self.model_feature_schema: List[str] = (
            self._get_model_feature_schema(
                model
            )
        )

        self.model_feature_count: int = (
            len(self.model_feature_schema)
        )

        self.canonical_only_features: List[str] = [
            feature
            for feature in self.feature_schema
            if feature not in self.model_feature_schema
        ]

        self.model_is_subset_of_canonical_schema: bool = (
            all(
                feature in self.feature_schema
                for feature in self.model_feature_schema
            )
        )

        # ---------------------------------------------------------------------
        # Top-K
        # ---------------------------------------------------------------------

        try:
            parsed_top_k = int(top_k)
        except (TypeError, ValueError):
            parsed_top_k = self.DEFAULT_TOP_K

        self.top_k = max(
            1,
            min(
                self.feature_count,
                parsed_top_k,
            ),
        )

        # ---------------------------------------------------------------------
        # SHAP threshold
        # ---------------------------------------------------------------------

        try:
            parsed_threshold = float(
                min_absolute_shap_impact
            )
        except (TypeError, ValueError):
            parsed_threshold = (
                self.MIN_ABSOLUTE_SHAP_IMPACT
            )

        if (
            not math.isfinite(parsed_threshold)
            or parsed_threshold < 0.0
        ):
            parsed_threshold = (
                self.MIN_ABSOLUTE_SHAP_IMPACT
            )

        self.min_absolute_shap_impact = (
            parsed_threshold
        )

        # ---------------------------------------------------------------------
        # SHAP state
        # ---------------------------------------------------------------------

        self.explainer: Any = None

        self.shap_initialized: bool = False

        self.shap_initialization_error: Optional[str] = None

        self._initialize_explainer()

        logger.info(
            "ThreatExplainer initialized | "
            "canonical_features=%d | "
            "model_features=%d | "
            "top_k=%d | "
            "shap=%s",
            self.feature_count,
            self.model_feature_count,
            self.top_k,
            self.shap_initialized,
        )

    # =========================================================================
    # CANONICAL SCHEMA
    # =========================================================================

    @staticmethod
    def _get_canonical_feature_schema() -> List[str]:
        """
        Return the canonical feature schema.

        IMPORTANT:
        ThreatFeatureSchema.get_schema_columns is a class method and MUST
        be called with ().

        This fixes the previous:

            TypeError: 'method' object is not iterable
        """

        getter = getattr(
            ThreatFeatureSchema,
            "get_schema_columns",
            None,
        )

        if callable(getter):
            features = getter()
        else:
            features = getter

        if features is None:
            raise RuntimeError(
                "ThreatFeatureSchema does not expose "
                "get_schema_columns()."
            )

        if not isinstance(
            features,
            (list, tuple),
        ):
            try:
                features = list(features)
            except TypeError as exc:
                raise RuntimeError(
                    "ThreatFeatureSchema.get_schema_columns() "
                    "did not return an iterable feature schema."
                ) from exc

        return [
            str(feature)
            for feature in features
        ]

    # =========================================================================
    # MODEL SCHEMA
    # =========================================================================

    @classmethod
    def _get_model_feature_schema(
        cls,
        model: Any,
    ) -> List[str]:
        """
        Resolve the exact feature order used by the trained model.

        Priority:

        1. Native XGBoost feature_names.
        2. Wrapper/model feature schema.
        3. Canonical schema as a final compatibility fallback.

        The canonical schema is NOT blindly assumed when a model explicitly
        reports a smaller feature set.
        """

        if model is None:
            return list(
                cls._get_canonical_feature_schema()
            )

        # ---------------------------------------------------------------------
        # Native XGBoost / Booster feature names
        # ---------------------------------------------------------------------

        try:

            booster = model.get_booster()

            feature_names = getattr(
                booster,
                "feature_names",
                None,
            )

            if feature_names:

                return [
                    str(name)
                    for name in feature_names
                ]

        except Exception:
            pass

        # ---------------------------------------------------------------------
        # Wrapper-level feature schema
        # ---------------------------------------------------------------------

        for attribute in (
            "model_feature_schema",
            "model_features",
            "feature_schema",
            "feature_names",
            "feature_names_in_",
        ):

            value = getattr(
                model,
                attribute,
                None,
            )

            if value is None:
                continue

            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue

            if isinstance(
                value,
                (list, tuple, np.ndarray, pd.Index),
            ):

                names = [
                    str(item)
                    for item in value
                ]

                if names:
                    return names

        # ---------------------------------------------------------------------
        # Wrapper nested model
        # ---------------------------------------------------------------------

        for attribute in (
            "model",
            "classifier",
            "estimator",
            "xgb_model",
            "original_model",
        ):

            nested = getattr(
                model,
                attribute,
                None,
            )

            if nested is None or nested is model:
                continue

            names = cls._get_model_feature_schema(
                nested
            )

            if names:
                return names

        # ---------------------------------------------------------------------
        # Feature count fallback
        # ---------------------------------------------------------------------

        model_count = cls._get_actual_model_feature_count(
            model
        )

        canonical = cls._get_canonical_feature_schema()

        if (
            model_count > 0
            and model_count < len(canonical)
        ):
            logger.warning(
                "Threat model exposes %d features but no "
                "feature names. Falling back to the first %d "
                "canonical features.",
                model_count,
                model_count,
            )

            return canonical[
                :model_count
            ]

        return canonical

    # =========================================================================
    # MODEL FEATURE COUNT
    # =========================================================================

    @staticmethod
    def _get_actual_model_feature_count(
        model: Any,
    ) -> int:
        """
        Determine the actual feature count of the model.
        """

        if model is None:
            return 0

        # n_features_in_
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

        # Booster
        try:

            booster = model.get_booster()

            return int(
                booster.num_features()
            )

        except Exception:
            pass

        # feature_names_in_
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

        # Nested model
        for attribute in (
            "model",
            "classifier",
            "estimator",
            "xgb_model",
            "original_model",
        ):

            nested = getattr(
                model,
                attribute,
                None,
            )

            if nested is None or nested is model:
                continue

            count = (
                ThreatExplainer
                ._get_actual_model_feature_count(
                    nested
                )
            )

            if count > 0:
                return count

        return 0

    # =========================================================================
    # SCHEMA CONTRACT
    # =========================================================================

    def _validate_schema_contract(self) -> None:
        """
        Validate the canonical Threat schema.
        """

        schema = list(
            self.feature_schema
        )

        count = int(
            self.feature_count
        )

        if len(schema) != count:

            raise RuntimeError(
                "ThreatFeatureSchema is internally inconsistent: "
                f"list_length={len(schema)}, "
                f"feature_count={count}."
            )

        if count != 20:

            raise RuntimeError(
                "ThreatExplainer expects exactly 20 canonical "
                f"Threat features, received {count}."
            )

        if len(set(schema)) != len(schema):

            raise RuntimeError(
                "ThreatFeatureSchema contains duplicate feature names."
            )

    # =========================================================================
    # FEATURE METADATA CONTRACT
    # =========================================================================

    def _validate_feature_metadata_contract(self) -> None:
        """
        Ensure descriptions and heuristic weights reference only
        canonical features.
        """

        schema_set = set(
            self.feature_schema
        )

        description_keys = set(
            self.FEATURE_DESCRIPTIONS.keys()
        )

        heuristic_keys = set(
            self.HEURISTIC_WEIGHTS.keys()
        )

        missing_descriptions = (
            schema_set
            - description_keys
        )

        extra_descriptions = (
            description_keys
            - schema_set
        )

        invalid_heuristic_keys = (
            heuristic_keys
            - schema_set
        )

        if missing_descriptions:

            raise RuntimeError(
                "ThreatExplainer is missing human-readable "
                "descriptions for schema features: "
                + ", ".join(
                    sorted(
                        missing_descriptions
                    )
                )
            )

        if extra_descriptions:

            raise RuntimeError(
                "ThreatExplainer contains descriptions for "
                "features outside ThreatFeatureSchema: "
                + ", ".join(
                    sorted(
                        extra_descriptions
                    )
                )
            )

        if invalid_heuristic_keys:

            raise RuntimeError(
                "ThreatExplainer heuristic references features "
                "outside ThreatFeatureSchema: "
                + ", ".join(
                    sorted(
                        invalid_heuristic_keys
                    )
                )
            )

    # =========================================================================
    # PUBLIC SCHEMA API
    # =========================================================================

    def get_feature_schema(self) -> List[str]:
        """
        Return canonical feature order.
        """

        return list(
            self.feature_schema
        )

    def get_feature_count(self) -> int:
        """
        Return canonical feature count.
        """

        return int(
            self.feature_count
        )

    def get_model_feature_schema(self) -> List[str]:
        """
        Return exact trained-model feature order.
        """

        return list(
            self.model_feature_schema
        )

    def get_model_feature_count(self) -> int:
        """
        Return trained-model feature count.
        """

        return int(
            self.model_feature_count
        )

    def validate_feature_schema(self) -> Dict[str, Any]:
        """
        Validate canonical and model feature alignment.

        A model using a subset of the canonical schema is valid.
        """

        model_outside_schema = [
            feature
            for feature in self.model_feature_schema
            if feature not in self.feature_schema
        ]

        canonical_only = [
            feature
            for feature in self.feature_schema
            if feature not in self.model_feature_schema
        ]

        errors: List[str] = []

        if model_outside_schema:

            errors.append(
                "Model contains features outside canonical schema."
            )

        if len(
            set(self.feature_schema)
        ) != self.feature_count:

            errors.append(
                "Canonical schema contains duplicate features."
            )

        if len(
            set(self.model_feature_schema)
        ) != self.model_feature_count:

            errors.append(
                "Model schema contains duplicate features."
            )

        if self.model_feature_count <= 0:

            errors.append(
                "Unable to determine trained model feature schema."
            )

        valid = (
            len(errors) == 0
            and self.model_is_subset_of_canonical_schema
        )

        return {
            "valid": valid,
            "feature_count": self.feature_count,
            "expected_feature_count": self.feature_count,
            "model_feature_count": self.model_feature_count,
            "model_features": list(
                self.model_feature_schema
            ),
            "model_features_outside_schema":
                model_outside_schema,
            "model_is_subset_of_canonical_schema":
                self.model_is_subset_of_canonical_schema,
            "canonical_features_not_used_by_model":
                canonical_only,
            "feature_order":
                list(self.feature_schema),
            "model_feature_order":
                list(self.model_feature_schema),
            "errors":
                errors,
        }

    # =========================================================================
    # SHAP MODEL UNWRAPPING
    # =========================================================================

    @staticmethod
    def _unwrap_model(
        model: Any,
    ) -> Any:
        """
        Resolve the underlying native XGBoost estimator.
        """

        current = model

        visited = set()

        for _ in range(8):

            if current is None:
                return None

            object_id = id(
                current
            )

            if object_id in visited:
                break

            visited.add(
                object_id
            )

            module_name = getattr(
                current.__class__,
                "__module__",
                "",
            )

            class_name = getattr(
                current.__class__,
                "__name__",
                "",
            )

            if (
                "xgboost"
                in module_name.lower()
                and class_name
                in {
                    "XGBClassifier",
                    "XGBRegressor",
                    "XGBModel",
                }
            ):
                return current

            found_nested = False

            for attribute in (
                "model",
                "classifier",
                "estimator",
                "xgb_model",
                "original_model",
            ):

                nested = getattr(
                    current,
                    attribute,
                    None,
                )

                if (
                    nested is not None
                    and nested is not current
                ):

                    current = nested
                    found_nested = True
                    break

            if not found_nested:
                break

        return current

    # =========================================================================
    # SHAP INITIALIZATION
    # =========================================================================

    def _initialize_explainer(self) -> None:
        """
        Initialize SHAP.

        Failure is non-fatal.
        """

        self.explainer = None

        self.shap_initialized = False

        self.shap_initialization_error = None

        if not SHAP_AVAILABLE:

            self.shap_initialization_error = (
                "SHAP is not installed."
            )

            logger.warning(
                "SHAP is not installed. "
                "Threat heuristic explanation will be used."
            )

            return

        if self.model is None:

            self.shap_initialization_error = (
                "No model supplied."
            )

            logger.info(
                "No Threat model supplied. "
                "Heuristic explanation will be used."
            )

            return

        actual_model = self._unwrap_model(
            self.model
        )

        if actual_model is None:

            self.shap_initialization_error = (
                "Unable to resolve underlying XGBoost model."
            )

            return

        try:

            self.explainer = shap.TreeExplainer(
                actual_model
            )

            self.shap_initialized = True

            self.shap_initialization_error = None

            logger.info(
                "Threat SHAP TreeExplainer initialized."
            )

        except Exception as exc:

            self.explainer = None

            self.shap_initialized = False

            self.shap_initialization_error = str(
                exc
            )

            logger.warning(
                "Failed to initialize ThreatExplainer SHAP: %s",
                str(exc),
            )

    # =========================================================================
    # MODEL MANAGEMENT
    # =========================================================================

    def set_model(
        self,
        model: Any,
    ) -> None:
        """
        Replace the model and rebuild SHAP configuration.
        """

        self.model = model

        self.model_feature_schema = (
            self._get_model_feature_schema(
                model
            )
        )

        self.model_feature_count = (
            len(
                self.model_feature_schema
            )
        )

        self.canonical_only_features = [
            feature
            for feature in self.feature_schema
            if feature not in self.model_feature_schema
        ]

        self.model_is_subset_of_canonical_schema = (
            all(
                feature in self.feature_schema
                for feature in self.model_feature_schema
            )
        )

        self._initialize_explainer()

    # =========================================================================
    # UTILITY
    # =========================================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:

            result = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return float(
                default
            )

        if not math.isfinite(
            result
        ):

            return float(
                default
            )

        return result

    @staticmethod
    def _is_enabled(
        value: Any,
    ) -> bool:

        if isinstance(
            value,
            bool,
        ):

            return value

        try:

            return bool(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):

            return False

    @staticmethod
    def _score_0_to_100(
        value: Any,
    ) -> float:

        numeric = ThreatExplainer._safe_float(
            value
        )

        return max(
            0.0,
            min(
                100.0,
                numeric,
            ),
        )

    @staticmethod
    def _ratio(
        value: Any,
    ) -> float:

        numeric = ThreatExplainer._safe_float(
            value
        )

        return max(
            0.0,
            min(
                1.0,
                numeric,
            ),
        )

    # =========================================================================
    # FEATURE NORMALIZATION
    # =========================================================================

    def _normalize_features(
        self,
        features: Mapping[str, Any],
    ) -> Dict[str, float]:
        """
        Normalize incoming features into the canonical schema.

        Missing features receive 0.0.

        Extra features are ignored.
        """

        if not isinstance(
            features,
            Mapping,
        ):

            return {
                feature: 0.0
                for feature in self.feature_schema
            }

        normalized: Dict[str, float] = {}

        for feature in self.feature_schema:

            normalized[
                feature
            ] = self._safe_float(
                features.get(
                    feature,
                    0.0,
                )
            )

        return normalized

    # =========================================================================
    # MODEL MATRIX VALIDATION
    # =========================================================================

    def _validate_feature_matrix(
        self,
        df_features: pd.DataFrame,
    ) -> None:
        """
        Validate SHAP matrix against exact model schema.
        """

        if not isinstance(
            df_features,
            pd.DataFrame,
        ):

            raise TypeError(
                "SHAP feature matrix must be a pandas DataFrame."
            )

        expected = list(
            self.model_feature_schema
        )

        actual = list(
            df_features.columns
        )

        if actual != expected:

            raise ValueError(
                "SHAP feature order mismatch: "
                f"expected={expected}, "
                f"received={actual}"
            )

        if df_features.shape[1] != (
            self.model_feature_count
        ):

            raise ValueError(
                "SHAP feature count mismatch: "
                f"expected={self.model_feature_count}, "
                f"received={df_features.shape[1]}"
            )

        if not np.isfinite(
            df_features.to_numpy(
                dtype=float
            )
        ).all():

            raise ValueError(
                "SHAP feature matrix contains "
                "non-finite values."
            )

    # =========================================================================
    # SHAP INPUT ALIGNMENT
    # =========================================================================

    def _build_model_dataframe(
        self,
        normalized_features: Mapping[str, Any],
    ) -> pd.DataFrame:
        """
        Build a one-row DataFrame containing EXACTLY the model features.

        This is the critical fix for the 19-vs-20 SHAP failure.
        """

        row = {
            feature: self._safe_float(
                normalized_features.get(
                    feature,
                    0.0,
                )
            )
            for feature in self.model_feature_schema
        }

        df = pd.DataFrame(
            [
                row
            ],
            columns=self.model_feature_schema,
        )

        self._validate_feature_matrix(
            df
        )

        return df

    # =========================================================================
    # SHAP VECTOR EXTRACTION
    # =========================================================================

    def _extract_shap_vector(
        self,
        shap_result: Any,
    ) -> np.ndarray:
        """
        Normalize different SHAP output formats into a single vector.
        """

        values = getattr(
            shap_result,
            "values",
            shap_result,
        )

        array = np.asarray(
            values,
            dtype=float,
        )

        if array.ndim == 0:

            raise ValueError(
                "SHAP returned a scalar."
            )

        # Single sample, single output
        if array.ndim == 1:

            return array

        # Shape: (1, features)
        if array.ndim == 2:

            if array.shape[0] == 1:

                return array[0]

            return array[0]

        # Shape: (1, features, classes)
        if array.ndim == 3:

            if array.shape[0] == 1:

                array = array[0]

            else:

                array = array[0]

            if array.ndim == 2:

                # Prefer malicious class.
                if array.shape[1] > 1:

                    return array[:, 1]

                return array[:, 0]

        raise ValueError(
            "Unsupported SHAP output shape: "
            f"{array.shape}"
        )

    # =========================================================================
    # SHAP BASE VALUE
    # =========================================================================

    @staticmethod
    def _extract_base_value(
        shap_result: Any,
    ) -> Optional[float]:
        """
        Extract SHAP base value when available.
        """

        value = getattr(
            shap_result,
            "base_values",
            None,
        )

        if value is None:
            return None

        try:

            array = np.asarray(
                value,
                dtype=float,
            )

            if array.size == 0:
                return None

            return float(
                array.reshape(-1)[0]
            )

        except Exception:

            return None

    # =========================================================================
    # SHAP FACTORS
    # =========================================================================

    def _build_shap_factors(
        self,
        shap_values: Sequence[float],
        features: Mapping[str, Any],
        prediction: str,
    ) -> List[Dict[str, Any]]:
        """
        Convert SHAP values into explainable feature factors.
        """

        factors: List[Dict[str, Any]] = []

        if len(shap_values) != (
            self.model_feature_count
        ):

            raise ValueError(
                "SHAP vector length does not match "
                "model feature count."
            )

        for index, feature in enumerate(
            self.model_feature_schema
        ):

            impact = self._safe_float(
                shap_values[index]
            )

            if abs(impact) < (
                self.min_absolute_shap_impact
            ):
                continue

            value = self._safe_float(
                features.get(
                    feature,
                    0.0,
                )
            )

            if prediction == self.MALICIOUS_CLASS:

                supports_prediction = (
                    impact > 0
                )

            else:

                supports_prediction = (
                    impact < 0
                )

            direction = (
                "supports_prediction"
                if supports_prediction
                else "opposes_prediction"
            )

            supports_class = (
                prediction
                if supports_prediction
                else (
                    self.CLEAN_CLASS
                    if prediction == self.MALICIOUS_CLASS
                    else self.MALICIOUS_CLASS
                )
            )

            factors.append(
                {
                    "feature": feature,
                    "value": value,
                    "shap_value": round(
                        impact,
                        6,
                    ),
                    "absolute_impact": round(
                        abs(impact),
                        6,
                    ),
                    "direction": direction,
                    "supports_class": supports_class,
                    "description":
                        self._get_human_description(
                            feature
                        ),
                }
            )

        factors.sort(
            key=lambda item:
                item["absolute_impact"],
            reverse=True,
        )

        return factors[
            :self.top_k
        ]

    # =========================================================================
    # FACTOR SPLITTING
    # =========================================================================

    def _split_factors(
        self,
        factors: Sequence[Mapping[str, Any]],
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """
        Split factors into risk-increasing and protective factors.
        """

        risk_factors: List[Dict[str, Any]] = []

        protective_factors: List[Dict[str, Any]] = []

        for factor in factors:

            if (
                factor.get(
                    "supports_class"
                )
                == self.MALICIOUS_CLASS
            ):

                risk_factors.append(
                    dict(
                        factor
                    )
                )

            elif (
                factor.get(
                    "supports_class"
                )
                == self.CLEAN_CLASS
            ):

                protective_factors.append(
                    dict(
                        factor
                    )
                )

        return (
            risk_factors,
            protective_factors,
        )

    # =========================================================================
    # SHAP SUMMARY
    # =========================================================================

    def _build_shap_summary(
        self,
        factors: Sequence[Mapping[str, Any]],
        prediction: str,
    ) -> str:
        """
        Build human-readable SHAP summary.
        """

        if not factors:

            return (
                "SHAP analysis did not identify feature "
                "impacts above the configured threshold."
            )

        supporting = [
            factor
            for factor in factors
            if factor.get(
                "supports_class"
            ) == prediction
        ]

        if supporting:

            names = [
                str(
                    factor["feature"]
                )
                for factor in supporting[
                    :3
                ]
            ]

            return (
                f"SHAP analysis identified "
                f"{', '.join(names)} as the primary "
                f"features supporting the "
                f"'{prediction}' classification."
            )

        return (
            "SHAP analysis identified feature impacts "
            "but no dominant supporting factors."
        )

    # =========================================================================
    # HEURISTIC FACTOR
    # =========================================================================

    def _heuristic_factor(
        self,
        feature: str,
        value: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Build a deterministic heuristic factor.
        """

        weight = self.HEURISTIC_WEIGHTS.get(
            feature
        )

        if weight is None:
            return None

        numeric = self._safe_float(
            value
        )

        # Normalize feature value according to feature type.
        if feature in {
            "is_blacklisted",
            "high_authority_vendor_flagged",
            "has_high_threat_consensus",
            "is_malware_associated",
            "is_c2_node",
            "is_exploit_distributor",
            "is_zero_day_candidate",
        }:

            normalized_value = (
                1.0
                if numeric > 0
                else 0.0
            )

        elif feature in {
            "malicious_ratio",
            "suspicion_ratio",
        }:

            normalized_value = self._ratio(
                numeric
            )

        elif feature == "reputation_score":

            normalized_value = (
                1.0
                - (
                    self._score_0_to_100(
                        numeric
                    )
                    / 100.0
                )
            )

        elif feature in {
            "malware_threat_score",
            "weighted_threat_score",
            "overall_threat_score",
        }:

            normalized_value = (
                self._score_0_to_100(
                    numeric
                )
                / 100.0
            )

        else:

            # Count-like features.
            normalized_value = max(
                0.0,
                min(
                    1.0,
                    numeric / 100.0,
                ),
            )

        impact = (
            weight
            * normalized_value
        )

        if abs(impact) < 0.0001:

            return None

        supports_malicious = (
            impact > 0
        )

        return {
            "feature": feature,
            "value": numeric,
            "impact": round(
                impact,
                6,
            ),
            "absolute_impact": round(
                abs(impact),
                6,
            ),
            "direction": (
                "increases_risk"
                if supports_malicious
                else "decreases_risk"
            ),
            "supports_class": (
                self.MALICIOUS_CLASS
                if supports_malicious
                else self.CLEAN_CLASS
            ),
            "description":
                self._get_human_description(
                    feature
                ),
        }

    # =========================================================================
    # HEURISTIC EXPLANATION
    # =========================================================================

    def _heuristic_explanation(
        self,
        features: Mapping[str, Any],
        prediction: str,
    ) -> Dict[str, Any]:
        """
        Deterministic fallback explanation.
        """

        factors: List[Dict[str, Any]] = []

        for feature in self.feature_schema:

            factor = self._heuristic_factor(
                feature,
                features.get(
                    feature,
                    0.0,
                ),
            )

            if factor is not None:

                factors.append(
                    factor
                )

        factors.sort(
            key=lambda item:
                item["absolute_impact"],
            reverse=True,
        )

        factors = factors[
            :self.top_k
        ]

        risk_factors = [
            factor
            for factor in factors
            if factor.get(
                "supports_class"
            ) == self.MALICIOUS_CLASS
        ]

        protective_factors = [
            factor
            for factor in factors
            if factor.get(
                "supports_class"
            ) == self.CLEAN_CLASS
        ]

        if risk_factors:

            names = [
                factor["feature"]
                for factor in risk_factors[
                    :3
                ]
            ]

            summary = (
                "Heuristic analysis identified "
                + ", ".join(names)
                + " as the primary threat indicators."
            )

        elif protective_factors:

            names = [
                factor["feature"]
                for factor in protective_factors[
                    :3
                ]
            ]

            summary = (
                "Heuristic analysis identified "
                + ", ".join(names)
                + " as the primary protective indicators."
            )

        else:

            summary = (
                "No significant heuristic threat indicators "
                "were identified."
            )

        return {
            "method": "heuristic",
            "summary": summary,
            "top_shap_factors": [],
            "top_factors": factors,
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
            "shap_status": (
                "unavailable"
            ),
            "shap_error": (
                self.shap_initialization_error
            ),
        }

    # =========================================================================
    # SHAP EXPLANATION
    # =========================================================================

    def _generate_shap_explanation(
        self,
        features: Mapping[str, Any],
        prediction: str,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation using EXACT model feature schema.

        This prevents:

            Number of columns does not match number of features in booster

        when canonical features > model features.
        """

        if not self.shap_initialized:

            raise RuntimeError(
                "SHAP explainer is not initialized."
            )

        df_features = self._build_model_dataframe(
            features
        )

        shap_result = self.explainer(
            df_features
        )

        shap_values = self._extract_shap_vector(
            shap_result
        )

        if len(shap_values) != (
            self.model_feature_count
        ):

            raise ValueError(
                "SHAP returned "
                f"{len(shap_values)} values for "
                f"{self.model_feature_count} model features."
            )

        factors = self._build_shap_factors(
            shap_values,
            features,
            prediction,
        )

        risk_factors, protective_factors = (
            self._split_factors(
                factors
            )
        )

        return {
            "method": "shap",
            "summary":
                self._build_shap_summary(
                    factors,
                    prediction,
                ),
            "top_shap_factors":
                factors,
            "top_factors":
                factors,
            "risk_factors":
                risk_factors,
            "protective_factors":
                protective_factors,
            "shap_status":
                "ready",
            "shap_error":
                None,
            "base_value":
                self._extract_base_value(
                    shap_result
                ),
            "model_feature_count":
                self.model_feature_count,
            "model_feature_schema":
                list(
                    self.model_feature_schema
                ),
        }

    # =========================================================================
    # PREDICTION NORMALIZATION
    # =========================================================================

    def _normalize_prediction(
        self,
        prediction: Any,
    ) -> str:
        """
        Normalize prediction into the canonical class label.
        """

        if isinstance(
            prediction,
            Mapping,
        ):

            for key in (
                "prediction",
                "verdict",
                "class_label",
                "label",
            ):

                if key in prediction:

                    return self._normalize_prediction(
                        prediction[key]
                    )

            for key in (
                "class_index",
                "prediction_class",
                "class",
            ):

                if key in prediction:

                    return self._normalize_prediction(
                        prediction[key]
                    )

        if isinstance(
            prediction,
            str,
        ):

            normalized = (
                prediction
                .strip()
                .lower()
            )

            if normalized in {
                self.CLEAN_CLASS,
                "clean",
                "legitimate",
                "benign",
                "0",
            }:

                return self.CLEAN_CLASS

            if normalized in {
                self.MALICIOUS_CLASS,
                "malicious",
                "threat",
                "phishing",
                "1",
            }:

                return self.MALICIOUS_CLASS

        try:

            numeric = int(
                float(prediction)
            )

            if numeric == 0:

                return self.CLEAN_CLASS

            if numeric == 1:

                return self.MALICIOUS_CLASS

        except (
            TypeError,
            ValueError,
        ):

            pass

        raise ValueError(
            f"Unknown Threat prediction: {prediction!r}"
        )

    # =========================================================================
    # PUBLIC EXPLANATION API
    # =========================================================================

    def generate_explanation(
        self,
        features: Dict[str, Any],
        prediction: Any,
    ) -> Dict[str, Any]:
        """
        Generate explanation.

        SHAP is attempted first.

        If SHAP fails, a deterministic heuristic explanation is returned.
        """

        normalized_features = (
            self._normalize_features(
                features
            )
        )

        prediction_label = (
            self._normalize_prediction(
                prediction
            )
        )

        # ---------------------------------------------------------------------
        # SHAP
        # ---------------------------------------------------------------------

        if self.shap_initialized:

            try:

                return self._generate_shap_explanation(
                    normalized_features,
                    prediction_label,
                )

            except Exception as exc:

                logger.warning(
                    "Threat SHAP explanation failed: %s. "
                    "Using heuristic explanation.",
                    str(exc),
                    exc_info=True,
                )

                fallback = (
                    self._heuristic_explanation(
                        normalized_features,
                        prediction_label,
                    )
                )

                fallback[
                    "shap_error"
                ] = str(
                    exc
                )

                return fallback

        # ---------------------------------------------------------------------
        # Heuristic fallback
        # ---------------------------------------------------------------------

        return self._heuristic_explanation(
            normalized_features,
            prediction_label,
        )

    # =========================================================================
    # FALLBACK API
    # =========================================================================

    def _get_fallback_explanation(
        self,
        features: Mapping[str, Any],
        prediction: Any,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Publicly-compatible fallback explanation helper.
        """

        result = self._heuristic_explanation(
            features,
            self._normalize_prediction(
                prediction
            ),
        )

        if error:

            result[
                "shap_error"
            ] = str(
                error
            )

        return result

    # =========================================================================
    # STATUS
    # =========================================================================

    def is_shap_available(self) -> bool:
        """
        Return whether SHAP is operational.
        """

        return bool(
            self.shap_initialized
        )

    # =========================================================================
    # FEATURE DESCRIPTION
    # =========================================================================

    def get_feature_description(
        self,
        feature: str,
    ) -> str:
        """
        Return a human-readable feature description.
        """

        return self.FEATURE_DESCRIPTIONS.get(
            feature,
            feature,
        )

    def _get_human_description(
        self,
        feature: str,
    ) -> str:
        """
        Internal alias for feature descriptions.
        """

        return self.get_feature_description(
            feature
        )

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def get_configuration(
        self,
    ) -> Dict[str, Any]:
        """
        Return explainability and schema configuration.
        """

        return {
            "component":
                self.COMPONENT_NAME,

            "canonical_feature_count":
                self.feature_count,

            "canonical_feature_schema":
                list(
                    self.feature_schema
                ),

            "model_feature_count":
                self.model_feature_count,

            "model_feature_schema":
                list(
                    self.model_feature_schema
                ),

            "canonical_only_features":
                list(
                    self.canonical_only_features
                ),

            "model_is_subset_of_canonical_schema":
                self.model_is_subset_of_canonical_schema,

            "top_k":
                self.top_k,

            "minimum_absolute_shap_impact":
                self.min_absolute_shap_impact,

            "shap_installed":
                SHAP_AVAILABLE,

            "shap_initialized":
                self.shap_initialized,

            "shap_initialization_error":
                self.shap_initialization_error,
        }