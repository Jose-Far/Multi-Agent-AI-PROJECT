"""
Threat Intelligence AI Agent - Feature Schema
==============================================

Purpose
-------
Defines the canonical feature contract used by the Threat Intelligence AI
Agent across:

    Threat Intelligence Collector
              ↓
       Feature Extraction
              ↓
        Schema Alignment
              ↓
         Preprocessing
              ↓
          ML Model
              ↓
          Predictor
              ↓
        Risk Scoring
              ↓
         Explanation
              ↓
       Decision Fusion

Design Goals
------------
1. Maintain one canonical feature ordering.
2. Accept feature names produced by multiple collection layers.
3. Normalize aliases into canonical names.
4. Safely convert booleans, integers, floats, strings, and collections.
5. Prevent NaN / +/- infinity from reaching the ML pipeline.
6. Apply sensible numerical bounds to security telemetry.
7. Preserve missing-feature defaults.
8. Provide deterministic DataFrame output.
9. Expose metadata useful to preprocessing/training/prediction layers.
10. Remain compatible with the existing project architecture.
11. Avoid performing ML logic inside the schema layer.
12. Avoid standalone execution/test hooks.

Important
---------
This module performs FEATURE CONTRACT VALIDATION and NORMALIZATION only.

It does NOT:
    - query external threat-intelligence services
    - perform network requests
    - train models
    - perform model inference
    - calculate the final agent verdict
    - perform fusion

Those responsibilities belong to the appropriate downstream modules.
"""

from __future__ import annotations

import logging
import math
from numbers import Real
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class ThreatFeatureSchema(BaseModel):
    """
    Canonical schema for Threat Intelligence Agent features.

    Every feature is represented as a float so that the resulting feature
    vector can be consumed consistently by Pandas / NumPy / XGBoost /
    preprocessing pipelines.

    Feature groups
    --------------

    1. Blacklist & Vendor Flags
    2. Multi-Engine Consensus
    3. Malware / Infrastructure Associations
    4. Historical & Reputation Intelligence

    All fields intentionally use float values because:
        False -> 0.0
        True  -> 1.0
        integer -> float
        numeric string -> float

    This makes the schema suitable for ML inference while preserving the
    semantic meaning of binary security indicators.
    """

    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    # =========================================================================
    # 1. BLACKLIST & VENDOR FLAGS
    # =========================================================================

    is_blacklisted: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Binary indicator showing whether the domain or URL appears "
            "on one or more known threat blacklists."
        ),
    )

    blacklist_vendors_count: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Number of independent blacklist or reputation vendors "
            "currently flagging the domain or URL."
        ),
    )

    high_authority_vendor_flagged: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Binary indicator showing whether at least one high-authority "
            "threat intelligence provider has flagged the target."
        ),
    )

    # =========================================================================
    # 2. MULTI-ENGINE CONSENSUS & RATIOS
    # =========================================================================

    malicious_engines_count: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Number of threat-intelligence engines/vendors classifying "
            "the target as malicious."
        ),
    )

    suspicious_engines_count: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Number of engines/vendors classifying the target as suspicious."
        ),
    )

    harmless_engines_count: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Number of engines/vendors classifying the target as harmless."
        ),
    )

    undetected_engines_count: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Number of engines/vendors returning no detection or an "
            "undetected result."
        ),
    )

    malicious_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Malicious detection ratio calculated as malicious detections "
            "relative to the total number of engines with available results."
        ),
    )

    suspicion_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Suspicious detection ratio calculated as suspicious detections "
            "relative to the total number of engines with available results."
        ),
    )

    has_high_threat_consensus: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Binary indicator representing strong multi-engine agreement "
            "that the target presents a significant threat."
        ),
    )

    # =========================================================================
    # 3. MALWARE & THREAT ASSOCIATION INDICATORS
    # =========================================================================

    is_malware_associated: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Binary indicator showing whether the domain or URL has "
            "an observed association with malware."
        ),
    )

    is_c2_node: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Binary indicator showing whether the target has been identified "
            "as command-and-control infrastructure."
        ),
    )

    is_exploit_distributor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Binary indicator showing whether the target is associated "
            "with exploit-kit or exploit distribution activity."
        ),
    )

    malware_threat_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description=(
            "Threat score specifically representing malware-related "
            "intelligence on a 0-100 scale."
        ),
    )

    # =========================================================================
    # 4. HISTORICAL & REPUTATION METRICS
    # =========================================================================

    reputation_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description=(
            "Normalized external reputation score. Higher values represent "
            "greater threat/reputation concern according to the upstream "
            "normalization convention."
        ),
    )

    weighted_threat_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description=(
            "Threat score produced by weighting vendor detections according "
            "to their relative authority or confidence."
        ),
    )

    overall_threat_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description=(
            "Composite external threat-intelligence baseline score "
            "normalized to a 0-100 range."
        ),
    )

    is_zero_day_candidate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Binary indicator showing whether the target exhibits telemetry "
            "patterns consistent with an exceptionally new or insufficiently "
            "observed threat."
        ),
    )

    days_since_first_submission: float = Field(
        default=-1.0,
        description=(
            "Number of days since the target was first submitted to the "
            "relevant threat-intelligence platform. -1 means unavailable."
        ),
    )

    days_since_last_analysis: float = Field(
        default=-1.0,
        description=(
            "Number of days since the most recent threat-intelligence "
            "analysis. -1 means unavailable."
        ),
    )

    # =========================================================================
    # FIELD VALIDATION
    # =========================================================================

    @field_validator("*", mode="before")
    @classmethod
    def sanitize_numeric_value(cls, value: Any) -> float:
        """
        Normalize individual feature values before Pydantic validation.

        Supported inputs
        ----------------
        bool
        int
        float
        numeric strings
        lists
        tuples
        sets
        dictionaries

        Conversion rules
        ----------------
        bool:
            True  -> 1.0
            False -> 0.0

        numeric:
            Converted directly to float.

        collection:
            Converted to collection length.

        None / invalid values:
            Converted to 0.0.

        NaN / infinity:
            Converted to 0.0.

        Notes
        -----
        Collection length is intentionally supported because some upstream
        collectors may expose vendor detections as lists rather than counts.
        """

        if value is None:
            return 0.0

        if isinstance(value, bool):
            return 1.0 if value else 0.0

        if isinstance(value, (list, tuple, set, frozenset, dict)):
            return float(len(value))

        if isinstance(value, Real):
            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                return 0.0

            return numeric_value

        if isinstance(value, str):
            normalized = value.strip().lower()

            if normalized in {
                "true",
                "yes",
                "y",
                "on",
                "enabled",
                "flagged",
                "malicious",
            }:
                return 1.0

            if normalized in {
                "false",
                "no",
                "n",
                "off",
                "disabled",
                "clean",
                "harmless",
                "undetected",
            }:
                return 0.0

            try:
                numeric_value = float(normalized)

                if not math.isfinite(numeric_value):
                    return 0.0

                return numeric_value

            except (ValueError, TypeError):
                return 0.0

        try:
            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                return 0.0

            return numeric_value

        except (ValueError, TypeError):
            return 0.0

    # =========================================================================
    # CANONICAL SCHEMA INFORMATION
    # =========================================================================

    @classmethod
    def get_schema_columns(cls) -> List[str]:
        """
        Return the canonical feature names in deterministic model order.

        This ordering is extremely important.

        The model must receive the same feature ordering during:
            training
            validation
            inference

        Returns
        -------
        List[str]
            Ordered canonical feature names.
        """

        return list(cls.model_fields.keys())

    @classmethod
    def get_feature_count(cls) -> int:
        """
        Return the number of canonical features.

        This can be used by model validation to ensure that the trained
        model and the feature schema have not drifted apart.
        """

        return len(cls.get_schema_columns())

    @classmethod
    def get_default_values(cls) -> Dict[str, float]:
        """
        Return a complete dictionary containing the default value of every
        canonical feature.
        """

        instance = cls()

        return {
            key: float(value)
            for key, value in instance.model_dump().items()
        }

    # =========================================================================
    # FEATURE GROUP INFORMATION
    # =========================================================================

    @classmethod
    def get_feature_groups(cls) -> Dict[str, List[str]]:
        """
        Return the logical feature groups used by the Threat Intelligence
        Agent.

        This is useful for:
            - explanation
            - feature auditing
            - logging
            - model diagnostics
            - future feature importance reporting
        """

        return {
            "blacklist_vendor_flags": [
                "is_blacklisted",
                "blacklist_vendors_count",
                "high_authority_vendor_flagged",
            ],
            "multi_engine_consensus": [
                "malicious_engines_count",
                "suspicious_engines_count",
                "harmless_engines_count",
                "undetected_engines_count",
                "malicious_ratio",
                "suspicion_ratio",
                "has_high_threat_consensus",
            ],
            "malware_association": [
                "is_malware_associated",
                "is_c2_node",
                "is_exploit_distributor",
                "malware_threat_score",
            ],
            "historical_reputation": [
                "reputation_score",
                "weighted_threat_score",
                "overall_threat_score",
                "is_zero_day_candidate",
                "days_since_first_submission",
                "days_since_last_analysis",
            ],
        }

    # =========================================================================
    # ALIAS NORMALIZATION
    # =========================================================================

    @classmethod
    def get_alias_map(cls) -> Dict[str, str]:
        """
        Return all supported upstream aliases.

        Different collection and extraction layers may use slightly
        different terminology. The schema converts those names into one
        canonical representation.

        Canonical names remain unchanged.
        """

        return {
            # Vendor count aliases
            "malicious_vendor_count": "malicious_engines_count",
            "malicious_vendors_count": "malicious_engines_count",
            "malicious_count": "malicious_engines_count",
            "malicious_detections": "malicious_engines_count",

            "suspicious_vendor_count": "suspicious_engines_count",
            "suspicious_vendors_count": "suspicious_engines_count",
            "suspicious_count": "suspicious_engines_count",
            "suspicious_detections": "suspicious_engines_count",

            "harmless_vendor_count": "harmless_engines_count",
            "harmless_vendors_count": "harmless_engines_count",
            "harmless_count": "harmless_engines_count",
            "harmless_detections": "harmless_engines_count",

            "undetected_vendor_count": "undetected_engines_count",
            "undetected_vendors_count": "undetected_engines_count",
            "undetected_count": "undetected_engines_count",
            "undetected_detections": "undetected_engines_count",

            # Ratio aliases
            "vendor_detection_ratio": "malicious_ratio",
            "malicious_detection_ratio": "malicious_ratio",
            "malicious_vendor_ratio": "malicious_ratio",

            "vendor_suspicion_ratio": "suspicion_ratio",
            "suspicious_detection_ratio": "suspicion_ratio",
            "suspicious_vendor_ratio": "suspicion_ratio",

            # Threat-score aliases
            "threat_score": "overall_threat_score",
            "overall_score": "overall_threat_score",
            "overall_threat": "overall_threat_score",
            "threat_intelligence_score": "overall_threat_score",

            "weighted_score": "weighted_threat_score",
            "weighted_threat": "weighted_threat_score",

            "malware_score": "malware_threat_score",
            "malware_risk_score": "malware_threat_score",

            "reputation": "reputation_score",
            "reputation_risk_score": "reputation_score",

            # Consensus aliases
            "high_threat_consensus": "has_high_threat_consensus",
            "high_consensus": "has_high_threat_consensus",
            "threat_consensus": "has_high_threat_consensus",
            "strong_threat_consensus": "has_high_threat_consensus",

            # Authority aliases
            "high_authority_flagged": "high_authority_vendor_flagged",
            "authoritative_vendor_flagged": "high_authority_vendor_flagged",
            "tier1_vendor_flagged": "high_authority_vendor_flagged",

            # Association aliases
            "malware_associated": "is_malware_associated",
            "malware_detected": "is_malware_associated",

            "c2_detected": "is_c2_node",
            "command_and_control": "is_c2_node",
            "is_command_and_control": "is_c2_node",

            "exploit_distribution": "is_exploit_distributor",
            "exploit_kit": "is_exploit_distributor",

            # Temporal aliases
            "first_submission_age": "days_since_first_submission",
            "days_from_first_submission": "days_since_first_submission",

            "last_analysis_age": "days_since_last_analysis",
            "days_from_last_analysis": "days_since_last_analysis",

            # Zero-day aliases
            "zero_day_candidate": "is_zero_day_candidate",
            "potential_zero_day": "is_zero_day_candidate",
        }

    @classmethod
    def normalize_feature_names(
        cls,
        raw_features: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Normalize incoming feature names into canonical schema names.

        Parameters
        ----------
        raw_features:
            Mapping of raw feature names to values.

        Returns
        -------
        Tuple[Dict[str, Any], List[str]]
            First element:
                normalized feature dictionary.

            Second element:
                list of unknown / ignored input keys.

        Behavior
        --------
        - Canonical names are preserved.
        - Known aliases are converted.
        - Unknown keys are ignored by design.
        - Canonical values take precedence when both an alias and canonical
          value are supplied.
        """

        if not raw_features:
            return {}, []

        alias_map = cls.get_alias_map()
        canonical_fields = set(cls.get_schema_columns())

        normalized: Dict[str, Any] = {}
        unknown_keys: List[str] = []

        # First pass:
        # Preserve canonical names.
        for key, value in raw_features.items():
            if key in canonical_fields:
                normalized[key] = value

        # Second pass:
        # Apply aliases only when canonical value does not already exist.
        for key, value in raw_features.items():
            if key in canonical_fields:
                continue

            mapped_key = alias_map.get(key)

            if mapped_key is None:
                unknown_keys.append(key)
                continue

            if mapped_key not in normalized:
                normalized[mapped_key] = value

        return normalized, unknown_keys

    # =========================================================================
    # RANGE SANITIZATION
    # =========================================================================

    @classmethod
    def _sanitize_ranges(
        cls,
        features: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply deterministic security-feature bounds.

        This protects the ML pipeline from malformed upstream telemetry.

        Binary indicators:
            [0, 1]

        Counts:
            >= 0

        Scores:
            [0, 100]

        Temporal values:
            >= -1

        -1 is retained for unavailable historical telemetry.
        """

        binary_features = {
            "is_blacklisted",
            "high_authority_vendor_flagged",
            "has_high_threat_consensus",
            "is_malware_associated",
            "is_c2_node",
            "is_exploit_distributor",
            "is_zero_day_candidate",
        }

        count_features = {
            "blacklist_vendors_count",
            "malicious_engines_count",
            "suspicious_engines_count",
            "harmless_engines_count",
            "undetected_engines_count",
        }

        ratio_features = {
            "malicious_ratio",
            "suspicion_ratio",
        }

        score_features = {
            "malware_threat_score",
            "reputation_score",
            "weighted_threat_score",
            "overall_threat_score",
        }

        temporal_features = {
            "days_since_first_submission",
            "days_since_last_analysis",
        }

        for feature_name in binary_features:
            if feature_name in features:
                features[feature_name] = float(
                    max(0.0, min(1.0, features[feature_name]))
                )

        for feature_name in count_features:
            if feature_name in features:
                features[feature_name] = max(
                    0.0,
                    features[feature_name],
                )

        for feature_name in ratio_features:
            if feature_name in features:
                features[feature_name] = float(
                    max(0.0, min(1.0, features[feature_name]))
                )

        for feature_name in score_features:
            if feature_name in features:
                features[feature_name] = float(
                    max(0.0, min(100.0, features[feature_name]))
                )

        for feature_name in temporal_features:
            if feature_name in features:
                features[feature_name] = max(
                    -1.0,
                    features[feature_name],
                )

        return features

    # =========================================================================
    # RATIO DERIVATION
    # =========================================================================

    @classmethod
    def _derive_ratios(
        cls,
        features: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Derive missing vendor ratios from engine counts.

        This method does not overwrite explicitly supplied ratios.

        Total engines
        -------------
        malicious
        + suspicious
        + harmless
        + undetected

        If the total is zero, ratios remain zero.
        """

        malicious = float(
            features.get("malicious_engines_count", 0.0)
        )

        suspicious = float(
            features.get("suspicious_engines_count", 0.0)
        )

        harmless = float(
            features.get("harmless_engines_count", 0.0)
        )

        undetected = float(
            features.get("undetected_engines_count", 0.0)
        )

        total_engines = (
            malicious
            + suspicious
            + harmless
            + undetected
        )

        if total_engines <= 0.0:
            return features

        if "malicious_ratio" not in features:
            features["malicious_ratio"] = malicious / total_engines

        if "suspicion_ratio" not in features:
            features["suspicion_ratio"] = suspicious / total_engines

        return features

    # =========================================================================
    # CONSENSUS DERIVATION
    # =========================================================================

    @classmethod
    def _derive_consensus(
        cls,
        features: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Derive high-threat consensus when it has not been explicitly
        supplied.

        Conservative rule
        -----------------
        Strong consensus is indicated when either:

            malicious_ratio >= 0.50

        OR:

            malicious detections >= 3
            AND malicious_ratio >= 0.30

        Explicit upstream values always take precedence.
        """

        if "has_high_threat_consensus" in features:
            return features

        malicious_count = float(
            features.get("malicious_engines_count", 0.0)
        )

        malicious_ratio = float(
            features.get("malicious_ratio", 0.0)
        )

        consensus = (
            malicious_ratio >= 0.50
            or (
                malicious_count >= 3.0
                and malicious_ratio >= 0.30
            )
        )

        features["has_high_threat_consensus"] = (
            1.0 if consensus else 0.0
        )

        return features

    # =========================================================================
    # COMPLETE ALIGNMENT
    # =========================================================================

    @classmethod
    def align_and_validate(
        cls,
        raw_features: Optional[Mapping[str, Any]],
    ) -> pd.DataFrame:
        """
        Convert raw Threat Intelligence telemetry into a canonical
        one-row DataFrame.

        Processing sequence
        -------------------
        1. Validate input mapping.
        2. Normalize feature names.
        3. Convert values to numeric values.
        4. Apply schema defaults.
        5. Derive missing ratios.
        6. Derive missing consensus.
        7. Apply numerical bounds.
        8. Remove NaN / infinity.
        9. Enforce canonical feature ordering.
        10. Return a one-row DataFrame.

        Parameters
        ----------
        raw_features:
            Raw feature dictionary produced by the threat collector /
            feature extraction layer.

        Returns
        -------
        pandas.DataFrame
            Exactly one row containing every canonical feature in the
            deterministic schema order.

        Failure behavior
        ----------------
        If validation fails, a complete zero/default feature vector is
        returned rather than allowing malformed telemetry to break the
        downstream agent pipeline.
        """

        try:
            raw_features = raw_features or {}

            if not isinstance(raw_features, Mapping):
                logger.warning(
                    "Threat features were expected to be a mapping, "
                    "received %s. Using schema defaults.",
                    type(raw_features).__name__,
                )
                raw_features = {}

            normalized_features, unknown_keys = (
                cls.normalize_feature_names(raw_features)
            )

            if unknown_keys:
                logger.debug(
                    "Ignoring %d unknown Threat Intelligence feature keys: %s",
                    len(unknown_keys),
                    unknown_keys,
                )

            normalized_numeric: Dict[str, float] = {}

            for field_name, raw_value in normalized_features.items():
                normalized_numeric[field_name] = (
                    cls.sanitize_numeric_value(raw_value)
                )

            # Derive ratios before applying bounds.
            normalized_numeric = cls._derive_ratios(
                normalized_numeric
            )

            # Derive consensus after malicious ratio is available.
            normalized_numeric = cls._derive_consensus(
                normalized_numeric
            )

            # Apply complete schema defaults.
            defaults = cls.get_default_values()

            for field_name, default_value in defaults.items():
                normalized_numeric.setdefault(
                    field_name,
                    default_value,
                )

            # Sanitize all final values.
            for field_name, value in list(
                normalized_numeric.items()
            ):
                if not math.isfinite(float(value)):
                    normalized_numeric[field_name] = defaults.get(
                        field_name,
                        0.0,
                    )

            # Enforce security-specific bounds.
            normalized_numeric = cls._sanitize_ranges(
                normalized_numeric
            )

            # Final deterministic feature ordering.
            schema_columns = cls.get_schema_columns()

            ordered_values = {
                field_name: float(
                    normalized_numeric.get(
                        field_name,
                        defaults.get(field_name, 0.0),
                    )
                )
                for field_name in schema_columns
            }

            df_aligned = pd.DataFrame(
                [ordered_values],
                columns=schema_columns,
            )

            # Final NaN / infinity protection.
            df_aligned = df_aligned.replace(
                [float("inf"), float("-inf")],
                0.0,
            ).fillna(0.0)

            return df_aligned

        except Exception as exc:
            logger.error(
                "Threat feature schema alignment failed: %s",
                exc,
                exc_info=True,
            )

            return cls.to_default_dataframe()

    # =========================================================================
    # PYDANTIC INSTANCE CONVERSION
    # =========================================================================

    @classmethod
    def from_raw_features(
        cls,
        raw_features: Optional[Mapping[str, Any]],
    ) -> "ThreatFeatureSchema":
        """
        Create a validated ThreatFeatureSchema instance from raw telemetry.

        Unlike align_and_validate(), this method returns the Pydantic schema
        object rather than a Pandas DataFrame.
        """

        try:
            raw_features = raw_features or {}

            normalized_features, unknown_keys = (
                cls.normalize_feature_names(raw_features)
            )

            if unknown_keys:
                logger.debug(
                    "Ignoring unknown feature keys while constructing "
                    "ThreatFeatureSchema: %s",
                    unknown_keys,
                )

            return cls(**normalized_features)

        except Exception as exc:
            logger.error(
                "Unable to construct ThreatFeatureSchema: %s",
                exc,
                exc_info=True,
            )

            return cls()

    # =========================================================================
    # DATAFRAME HELPERS
    # =========================================================================

    @classmethod
    def to_default_dataframe(cls) -> pd.DataFrame:
        """
        Return a one-row DataFrame containing the complete schema defaults.

        This provides a deterministic fallback for downstream preprocessing
        and prediction components.
        """

        defaults = cls.get_default_values()
        columns = cls.get_schema_columns()

        return pd.DataFrame(
            [[defaults[column] for column in columns]],
            columns=columns,
        )

    @classmethod
    def validate_dataframe_columns(
        cls,
        dataframe: pd.DataFrame,
    ) -> bool:
        """
        Validate that a DataFrame contains the complete canonical feature set.

        Extra columns are tolerated because downstream components may carry
        metadata, but every canonical model feature must be present.
        """

        if not isinstance(dataframe, pd.DataFrame):
            return False

        required_columns = set(cls.get_schema_columns())

        return required_columns.issubset(
            set(dataframe.columns)
        )

    @classmethod
    def reorder_dataframe(
        cls,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with canonical Threat Intelligence features in
        exact schema order.

        Extra columns are discarded intentionally for ML model input.
        """

        if not isinstance(dataframe, pd.DataFrame):
            return cls.to_default_dataframe()

        columns = cls.get_schema_columns()

        missing_columns = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            logger.warning(
                "Threat DataFrame missing %d canonical columns: %s",
                len(missing_columns),
                missing_columns,
            )

            dataframe = dataframe.copy()

            defaults = cls.get_default_values()

            for column in missing_columns:
                dataframe[column] = defaults.get(
                    column,
                    0.0,
                )

        return dataframe.loc[:, columns].copy()

    # =========================================================================
    # FEATURE VECTOR HELPERS
    # =========================================================================

    @classmethod
    def to_feature_vector(
        cls,
        raw_features: Optional[Mapping[str, Any]],
    ) -> List[float]:
        """
        Convert raw Threat Intelligence telemetry into an ordered numerical
        feature vector.

        The returned list follows exactly the canonical schema ordering.
        """

        dataframe = cls.align_and_validate(
            raw_features
        )

        return [
            float(dataframe.iloc[0][column])
            for column in cls.get_schema_columns()
        ]

    @classmethod
    def get_feature_index(cls) -> Dict[str, int]:
        """
        Return feature-name -> vector-index mapping.

        Example
        -------
        {
            "is_blacklisted": 0,
            "blacklist_vendors_count": 1,
            ...
        }

        Useful for:
            - model diagnostics
            - feature importance mapping
            - explanation
            - debugging feature-order mismatches
        """

        return {
            feature_name: index
            for index, feature_name in enumerate(
                cls.get_schema_columns()
            )
        }

    # =========================================================================
    # SCHEMA VALIDATION / COMPATIBILITY
    # =========================================================================

    @classmethod
    def is_compatible_feature_set(
        cls,
        feature_names: List[str],
    ) -> bool:
        """
        Check whether a supplied feature-name list exactly matches the
        canonical model feature order.

        This is intentionally strict.

        A trained model must use exactly the same feature ordering as
        inference.
        """

        return list(feature_names) == cls.get_schema_columns()

    @classmethod
    def get_schema_signature(cls) -> str:
        """
        Return a deterministic human-readable schema signature.

        Useful for logging model/schema compatibility.

        Example
        -------
        ThreatFeatureSchema[v1-like ordered signature]
        """

        return "|".join(cls.get_schema_columns())

    # =========================================================================
    # QUALITY / AVAILABILITY METADATA
    # =========================================================================

    @classmethod
    def calculate_feature_coverage(
        cls,
        raw_features: Optional[Mapping[str, Any]],
    ) -> float:
        """
        Calculate the percentage of canonical features that were supplied
        by the upstream collector.

        Returns
        -------
        float
            Value from 0.0 to 1.0.

        Important
        ---------
        A defaulted feature is NOT considered observed merely because the
        schema provides a fallback value.
        """

        if not raw_features:
            return 0.0

        normalized_features, _ = cls.normalize_feature_names(
            raw_features
        )

        canonical_count = len(
            cls.get_schema_columns()
        )

        if canonical_count == 0:
            return 0.0

        observed_count = sum(
            1
            for feature_name in cls.get_schema_columns()
            if feature_name in normalized_features
        )

        return float(
            observed_count / canonical_count
        )

    @classmethod
    def get_missing_features(
        cls,
        raw_features: Optional[Mapping[str, Any]],
    ) -> List[str]:
        """
        Return canonical features that were not provided by the upstream
        feature extraction layer.
        """

        raw_features = raw_features or {}

        normalized_features, _ = cls.normalize_feature_names(
            raw_features
        )

        return [
            feature_name
            for feature_name in cls.get_schema_columns()
            if feature_name not in normalized_features
        ]

    # =========================================================================
    # SAFE SUMMARY FOR LOGGING / EXPLANATION
    # =========================================================================

    @classmethod
    def summarize_features(
        cls,
        raw_features: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """
        Produce a compact diagnostic summary of the normalized feature set.

        This is intended for logging, debugging, and explanation layers.
        It does not expose raw upstream payloads.
        """

        dataframe = cls.align_and_validate(
            raw_features
        )

        feature_values = dataframe.iloc[0].to_dict()

        return {
            "feature_count": cls.get_feature_count(),
            "feature_coverage": cls.calculate_feature_coverage(
                raw_features
            ),
            "missing_features": cls.get_missing_features(
                raw_features
            ),
            "is_blacklisted": float(
                feature_values["is_blacklisted"]
            ),
            "blacklist_vendors_count": float(
                feature_values["blacklist_vendors_count"]
            ),
            "malicious_engines_count": float(
                feature_values["malicious_engines_count"]
            ),
            "suspicious_engines_count": float(
                feature_values["suspicious_engines_count"]
            ),
            "malicious_ratio": float(
                feature_values["malicious_ratio"]
            ),
            "suspicion_ratio": float(
                feature_values["suspicion_ratio"]
            ),
            "has_high_threat_consensus": float(
                feature_values["has_high_threat_consensus"]
            ),
            "is_malware_associated": float(
                feature_values["is_malware_associated"]
            ),
            "is_c2_node": float(
                feature_values["is_c2_node"]
            ),
            "is_exploit_distributor": float(
                feature_values["is_exploit_distributor"]
            ),
            "malware_threat_score": float(
                feature_values["malware_threat_score"]
            ),
            "reputation_score": float(
                feature_values["reputation_score"]
            ),
            "weighted_threat_score": float(
                feature_values["weighted_threat_score"]
            ),
            "overall_threat_score": float(
                feature_values["overall_threat_score"]
            ),
            "is_zero_day_candidate": float(
                feature_values["is_zero_day_candidate"]
            ),
            "days_since_first_submission": float(
                feature_values["days_since_first_submission"]
            ),
            "days_since_last_analysis": float(
                feature_values["days_since_last_analysis"]
            ),
        }
