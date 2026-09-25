"""
Threat Intelligence AI Agent - Preprocessing
=============================================

Converts raw Threat Intelligence feature data into deterministic,
safe, model-ready numerical DataFrames.

Pipeline
--------

    Threat Intelligence Collector
              ↓
       Feature Extraction
              ↓
      ThreatFeatureSchema
              ↓
      ThreatPreprocessor
              ↓
          XGBoost
              ↓
         Predictor

Responsibilities
----------------
- Canonical schema alignment
- Feature-name normalization through ThreatFeatureSchema
- Missing-feature handling
- NaN / infinity protection
- Numeric type normalization
- Security-aware range validation
- Conservative outlier capping
- Binary normalization
- Ratio validation
- Historical-feature preservation
- Batch-training consistency
- Single-instance inference consistency
- Feature-order preservation
- Dataset diagnostics

Important
---------
ThreatFeatureSchema is the single source of truth for the ML feature
contract.

This module must NOT maintain a second independent ML feature list.

The canonical ML input is exactly:

    ThreatFeatureSchema.get_schema_columns()

Extra Threat Intelligence metadata such as:

    threat_verdict
    flagging_vendors_list
    matched_blacklist_categories
    matched_malware_families
    primary_malware_category
    blacklist_risk_severity

must NOT be passed into the XGBoost feature matrix.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .feature_schema import ThreatFeatureSchema

logger = logging.getLogger(__name__)


class ThreatPreprocessor:
    """
    Schema-driven preprocessing layer for the Threat Intelligence AI Agent.

    The class deliberately delegates feature-contract responsibilities to
    ThreatFeatureSchema.

    Therefore:

        Schema feature order
                =
        Training feature order
                =
        Inference feature order
                =
        XGBoost feature order
                =
        SHAP feature order
    """

    PREPROCESSOR_VERSION = "2.0.0"

    # =========================================================================
    # OPERATIONAL OUTLIER CAPS
    # =========================================================================

    DEFAULT_OUTLIER_CAPS: Dict[str, float] = {
        "blacklist_vendors_count": 100.0,

        "malicious_engines_count": 250.0,
        "suspicious_engines_count": 250.0,
        "harmless_engines_count": 250.0,
        "undetected_engines_count": 250.0,

        "malware_threat_score": 100.0,
        "reputation_score": 100.0,
        "weighted_threat_score": 100.0,
        "overall_threat_score": 100.0,

        # Approximately 20 years.
        "days_since_first_submission": 7300.0,

        # Approximately 10 years.
        "days_since_last_analysis": 3650.0,
    }

    # =========================================================================
    # FEATURE GROUPS
    #
    # These are preprocessing categories only.
    # The actual canonical feature list comes from ThreatFeatureSchema.
    # =========================================================================

    BINARY_FEATURES = {
        "is_blacklisted",
        "high_authority_vendor_flagged",
        "has_high_threat_consensus",
        "is_malware_associated",
        "is_c2_node",
        "is_exploit_distributor",
        "is_zero_day_candidate",
    }

    RATIO_FEATURES = {
        "malicious_ratio",
        "suspicion_ratio",
    }

    SCORE_FEATURES = {
        "malware_threat_score",
        "reputation_score",
        "weighted_threat_score",
        "overall_threat_score",
    }

    COUNT_FEATURES = {
        "blacklist_vendors_count",
        "malicious_engines_count",
        "suspicious_engines_count",
        "harmless_engines_count",
        "undetected_engines_count",
    }

    HISTORICAL_FEATURES = {
        "days_since_first_submission",
        "days_since_last_analysis",
    }

    # =========================================================================
    # TARGET COLUMN ALIASES
    # =========================================================================

    TARGET_COLUMNS = {
        "target",
        "label",
        "class_label",
        "is_malicious",
        "malicious",
        "threat_label",
        "y",
    }

    # =========================================================================
    # CONSTRUCTOR
    # =========================================================================

    def __init__(
        self,
        outlier_caps: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the Threat Intelligence preprocessor.

        Parameters
        ----------
        outlier_caps:
            Optional custom operational upper bounds.

        Notes
        -----
        The canonical feature order is always obtained from
        ThreatFeatureSchema.
        """

        self.schema_columns: List[str] = (
            ThreatFeatureSchema.get_schema_columns()
        )

        self.feature_count: int = (
            ThreatFeatureSchema.get_feature_count()
        )

        self.outlier_caps: Dict[str, float] = (
            self.DEFAULT_OUTLIER_CAPS.copy()
        )

        if outlier_caps:
            self.outlier_caps.update(
                self._sanitize_custom_caps(
                    outlier_caps
                )
            )

        logger.info(
            "ThreatPreprocessor initialized | "
            "version=%s | features=%d",
            self.PREPROCESSOR_VERSION,
            self.feature_count,
        )

    # =========================================================================
    # SINGLE INSTANCE
    # =========================================================================

    def transform_single(
        self,
        raw_features: Optional[Mapping[str, Any]],
    ) -> pd.DataFrame:
        """
        Transform one raw Threat Intelligence feature mapping into a
        model-ready one-row DataFrame.

        Returns
        -------
        pandas.DataFrame
            Exactly one row and exactly the canonical Threat feature columns.
        """

        try:
            if raw_features is None:
                raw_features = {}

            if not isinstance(
                raw_features,
                Mapping,
            ):
                raise TypeError(
                    "Threat single-instance input must be a mapping."
                )

            # -------------------------------------------------------------
            # ThreatFeatureSchema owns:
            #
            # - alias normalization
            # - value sanitization
            # - defaults
            # - ratio derivation
            # - consensus derivation
            # - semantic bounds
            # - canonical ordering
            # -------------------------------------------------------------

            df_aligned = (
                ThreatFeatureSchema.align_and_validate(
                    raw_features
                )
            )

            df_processed = (
                self._apply_transformations(
                    df_aligned
                )
            )

            df_final = (
                self._finalize_dataframe(
                    df_processed
                )
            )

            valid, problems = (
                self.validate_model_input(
                    df_final
                )
            )

            if not valid:
                logger.error(
                    "Threat single-instance preprocessing "
                    "produced invalid model input: %s",
                    problems,
                )

                return (
                    self._create_safe_fallback_dataframe()
                )

            return df_final

        except Exception as exc:
            logger.error(
                "Failed to preprocess single Threat "
                "feature vector: %s",
                exc,
                exc_info=True,
            )

            return (
                self._create_safe_fallback_dataframe()
            )

    # =========================================================================
    # BATCH TRANSFORMATION
    # =========================================================================

    def transform_batch(
        self,
        df_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform a complete Threat Intelligence training dataset.

        Important
        ---------
        Batch failures are raised rather than silently converted to a
        default feature vector.

        This prevents an entire training dataset from accidentally becoming
        a matrix of default values.
        """

        if df_raw is None:
            raise ValueError(
                "Threat training DataFrame cannot be None."
            )

        if not isinstance(
            df_raw,
            pd.DataFrame,
        ):
            raise TypeError(
                "Threat training input must be a pandas DataFrame."
            )

        if df_raw.empty:
            raise ValueError(
                "Threat training DataFrame is empty."
            )

        try:
            logger.info(
                "Preprocessing Threat batch | "
                "rows=%d | columns=%d",
                len(df_raw),
                len(df_raw.columns),
            )

            df_working = (
                df_raw.copy()
            )

            # -------------------------------------------------------------
            # 1. Remove accidental target columns.
            # -------------------------------------------------------------

            accidental_targets = [
                column
                for column in df_working.columns
                if str(column).strip().lower()
                in self.TARGET_COLUMNS
            ]

            if accidental_targets:
                logger.warning(
                    "Removing target-like columns from "
                    "Threat feature matrix: %s",
                    accidental_targets,
                )

                df_working = (
                    df_working.drop(
                        columns=accidental_targets,
                        errors="ignore",
                    )
                )

            # -------------------------------------------------------------
            # 2. Normalize aliases.
            # -------------------------------------------------------------

            df_working = (
                self._normalize_batch_columns(
                    df_working
                )
            )

            # -------------------------------------------------------------
            # 3. Align to canonical schema.
            # -------------------------------------------------------------

            df_aligned = (
                self._align_batch_columns(
                    df_working
                )
            )

            # -------------------------------------------------------------
            # 4. Numerical preprocessing.
            # -------------------------------------------------------------

            df_processed = (
                self._apply_transformations(
                    df_aligned
                )
            )

            # -------------------------------------------------------------
            # 5. Final model matrix.
            # -------------------------------------------------------------

            df_final = (
                self._finalize_dataframe(
                    df_processed
                )
            )

            # -------------------------------------------------------------
            # 6. Strict validation.
            # -------------------------------------------------------------

            valid, problems = (
                self.validate_model_input(
                    df_final
                )
            )

            if not valid:
                raise ValueError(
                    "Threat model input validation failed: "
                    + "; ".join(problems)
                )

            logger.info(
                "Threat batch preprocessing completed | "
                "rows=%d | features=%d",
                len(df_final),
                len(df_final.columns),
            )

            return df_final

        except Exception as exc:
            logger.error(
                "Failed to preprocess Threat batch dataset: %s",
                exc,
                exc_info=True,
            )

            raise

    # =========================================================================
    # BATCH COLUMN NORMALIZATION
    # =========================================================================

    def _normalize_batch_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalize batch column aliases through ThreatFeatureSchema.

        Canonical feature names always take precedence.
        """

        if dataframe.empty:
            return dataframe.copy()

        alias_map = (
            ThreatFeatureSchema.get_alias_map()
        )

        canonical_columns = set(
            self.schema_columns
        )

        normalized = dataframe.copy()

        rename_map: Dict[str, str] = {}

        for column in normalized.columns:

            column_name = str(column).strip()

            if column_name in canonical_columns:
                continue

            mapped_name = alias_map.get(
                column_name
            )

            if mapped_name is not None:
                rename_map[
                    column
                ] = mapped_name

        if rename_map:
            normalized = (
                normalized.rename(
                    columns=rename_map
                )
            )

            logger.debug(
                "Normalized Threat feature aliases: %s",
                rename_map,
            )

        normalized = (
            self._resolve_duplicate_columns(
                normalized
            )
        )

        return normalized

    # =========================================================================
    # DUPLICATE COLUMN RESOLUTION
    # =========================================================================

    def _resolve_duplicate_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Resolve duplicate columns created by alias normalization.

        When multiple columns map to the same canonical feature, the first
        non-null value is preferred row-by-row.
        """

        if not dataframe.columns.duplicated().any():
            return dataframe.copy()

        result = pd.DataFrame(
            index=dataframe.index
        )

        processed_names = set()

        for column in dataframe.columns:

            if column in processed_names:
                continue

            processed_names.add(
                column
            )

            duplicate_mask = (
                dataframe.columns == column
            )

            duplicate_block = (
                dataframe.loc[
                    :,
                    duplicate_mask,
                ]
            )

            if duplicate_block.shape[1] == 1:

                result[column] = (
                    duplicate_block.iloc[:, 0]
                )

            else:

                result[column] = (
                    duplicate_block
                    .bfill(axis=1)
                    .iloc[:, 0]
                )

        return result

    # =========================================================================
    # BATCH SCHEMA ALIGNMENT
    # =========================================================================

    def _align_batch_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align a DataFrame with the canonical ThreatFeatureSchema.

        Missing canonical features receive schema defaults.

        Extra columns are intentionally removed from the ML matrix.
        """

        dataframe = dataframe.copy()

        defaults = (
            ThreatFeatureSchema.get_default_values()
        )

        missing_columns: List[str] = []

        for feature_name in self.schema_columns:

            if feature_name not in dataframe.columns:

                dataframe[feature_name] = (
                    defaults.get(
                        feature_name,
                        0.0,
                    )
                )

                missing_columns.append(
                    feature_name
                )

        if missing_columns:
            logger.debug(
                "Added %d missing Threat features "
                "using schema defaults: %s",
                len(missing_columns),
                missing_columns,
            )

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # Only canonical ML features survive.
        # Threat evidence metadata is not part of XGBoost input.
        # -------------------------------------------------------------

        return dataframe.loc[
            :,
            self.schema_columns,
        ].copy()

    # =========================================================================
    # CORE TRANSFORMATIONS
    # =========================================================================

    def _apply_transformations(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply deterministic numerical preprocessing.

        ThreatFeatureSchema remains responsible for:
            - aliases
            - defaults
            - ratio derivation
            - consensus derivation
            - canonical schema contract

        This class is responsible for:
            - numeric conversion
            - finite-value protection
            - operational bounds
            - binary normalization
            - ratio validation
            - score validation
            - historical-value validation
            - outlier caps
        """

        if dataframe is None:
            raise ValueError(
                "Threat preprocessing DataFrame cannot be None."
            )

        df = dataframe.copy()

        # -------------------------------------------------------------
        # Ensure canonical columns.
        # -------------------------------------------------------------

        df = self._align_batch_columns(
            df
        )

        # -------------------------------------------------------------
        # Convert every canonical feature to numeric.
        # -------------------------------------------------------------

        for column in self.schema_columns:

            df[column] = (
                self._coerce_series_to_numeric(
                    df[column]
                )
            )

        # -------------------------------------------------------------
        # Replace infinity with NaN.
        # -------------------------------------------------------------

        df = df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # -------------------------------------------------------------
        # Fill missing values using schema-specific defaults.
        # -------------------------------------------------------------

        defaults = (
            ThreatFeatureSchema.get_default_values()
        )

        for column in self.schema_columns:

            default_value = defaults.get(
                column,
                0.0,
            )

            df[column] = (
                df[column].fillna(
                    default_value
                )
            )

        # -------------------------------------------------------------
        # Apply semantic bounds.
        # -------------------------------------------------------------

        df = (
            self._apply_semantic_bounds(
                df
            )
        )

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT recalculate malicious_ratio,
        # suspicion_ratio, or consensus here.
        #
        # ThreatFeatureSchema already owns those derivations.
        #
        # Recalculating them here could overwrite explicitly supplied
        # upstream values and create training/inference inconsistencies.
        # -------------------------------------------------------------

        # -------------------------------------------------------------
        # Apply operational outlier caps.
        # -------------------------------------------------------------

        df = (
            self._apply_outlier_caps(
                df
            )
        )

        # -------------------------------------------------------------
        # Normalize binary values.
        # -------------------------------------------------------------

        df = (
            self._normalize_binary_features(
                df
            )
        )

        # -------------------------------------------------------------
        # Normalize ratios.
        # -------------------------------------------------------------

        df = (
            self._normalize_ratio_features(
                df
            )
        )

        # -------------------------------------------------------------
        # Normalize scores.
        # -------------------------------------------------------------

        df = (
            self._normalize_score_features(
                df
            )
        )

        # -------------------------------------------------------------
        # Normalize historical values.
        # -------------------------------------------------------------

        df = (
            self._normalize_historical_features(
                df
            )
        )

        # -------------------------------------------------------------
        # Final finite-value protection.
        # -------------------------------------------------------------

        df = df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        for column in self.schema_columns:

            default_value = defaults.get(
                column,
                0.0,
            )

            df[column] = (
                df[column]
                .fillna(default_value)
                .astype(np.float64)
            )

        return df

    # =========================================================================
    # NUMERIC CONVERSION
    # =========================================================================

    @staticmethod
    def _coerce_series_to_numeric(
        series: pd.Series,
    ) -> pd.Series:
        """
        Convert arbitrary Pandas values into numeric values.

        Supported:
            bool
            int
            float
            numeric strings
            boolean strings
            collections
            None
            NaN
            infinity
        """

        def convert_value(
            value: Any,
        ) -> float:

            if value is None:
                return np.nan

            if isinstance(
                value,
                bool,
            ):
                return (
                    1.0
                    if value
                    else 0.0
                )

            if isinstance(
                value,
                (
                    list,
                    tuple,
                    set,
                    frozenset,
                    dict,
                ),
            ):
                return float(
                    len(value)
                )

            if isinstance(
                value,
                str,
            ):

                normalized = (
                    value.strip().lower()
                )

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
                    value = float(
                        normalized
                    )

                except (
                    ValueError,
                    TypeError,
                ):
                    return np.nan

            try:
                numeric_value = float(
                    value
                )

            except (
                ValueError,
                TypeError,
            ):
                return np.nan

            if not math.isfinite(
                numeric_value
            ):
                return np.nan

            return numeric_value

        return series.map(
            convert_value
        )

    # =========================================================================
    # SEMANTIC BOUNDS
    # =========================================================================

    def _apply_semantic_bounds(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply mathematically valid feature ranges.

        Binary:
            [0, 1]

        Ratios:
            [0, 1]

        Scores:
            [0, 100]

        Counts:
            >= 0

        Historical:
            >= -1
        """

        df = dataframe.copy()

        # -------------------------------------------------------------
        # Binary
        # -------------------------------------------------------------

        for feature in (
            self.BINARY_FEATURES
        ):

            if feature in df.columns:

                df[feature] = (
                    df[feature].clip(
                        lower=0.0,
                        upper=1.0,
                    )
                )

        # -------------------------------------------------------------
        # Ratios
        # -------------------------------------------------------------

        for feature in (
            self.RATIO_FEATURES
        ):

            if feature in df.columns:

                df[feature] = (
                    df[feature].clip(
                        lower=0.0,
                        upper=1.0,
                    )
                )

        # -------------------------------------------------------------
        # Scores
        # -------------------------------------------------------------

        for feature in (
            self.SCORE_FEATURES
        ):

            if feature in df.columns:

                df[feature] = (
                    df[feature].clip(
                        lower=0.0,
                        upper=100.0,
                    )
                )

        # -------------------------------------------------------------
        # Counts
        # -------------------------------------------------------------

        for feature in (
            self.COUNT_FEATURES
        ):

            if feature in df.columns:

                df[feature] = (
                    df[feature].clip(
                        lower=0.0
                    )
                )

        # -------------------------------------------------------------
        # Historical values
        #
        # -1 = unavailable
        # >=0 = elapsed days
        # -------------------------------------------------------------

        for feature in (
            self.HISTORICAL_FEATURES
        ):

            if feature in df.columns:

                df[feature] = (
                    df[feature].clip(
                        lower=-1.0
                    )
                )

        return df

    # =========================================================================
    # OUTLIER CAPS
    # =========================================================================

    def _apply_outlier_caps(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply operational upper bounds.

        These are data-quality protections, not threat thresholds.
        """

        df = dataframe.copy()

        for feature, maximum in (
            self.outlier_caps.items()
        ):

            if feature not in df.columns:
                continue

            try:
                maximum_value = float(
                    maximum
                )

            except (
                ValueError,
                TypeError,
            ):

                logger.warning(
                    "Ignoring invalid outlier cap "
                    "for '%s': %r",
                    feature,
                    maximum,
                )
                continue

            if not math.isfinite(
                maximum_value
            ):
                logger.warning(
                    "Ignoring non-finite outlier "
                    "cap for '%s'.",
                    feature,
                )
                continue

            df[feature] = (
                df[feature].clip(
                    upper=maximum_value
                )
            )

        return df

    # =========================================================================
    # BINARY NORMALIZATION
    # =========================================================================

    def _normalize_binary_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert binary features to exactly 0.0 or 1.0.
        """

        df = dataframe.copy()

        for feature in (
            self.BINARY_FEATURES
        ):

            if feature not in df.columns:
                continue

            numeric = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

            df[feature] = (
                numeric
                .fillna(0.0)
                .ge(0.5)
                .astype(np.float64)
            )

        return df

    # =========================================================================
    # RATIO NORMALIZATION
    # =========================================================================

    def _normalize_ratio_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Guarantee ratio features remain within [0, 1].

        No ratio is recalculated here.

        Ratio derivation belongs to ThreatFeatureSchema.
        """

        df = dataframe.copy()

        for feature in (
            self.RATIO_FEATURES
        ):

            if feature not in df.columns:
                continue

            df[feature] = (
                pd.to_numeric(
                    df[feature],
                    errors="coerce",
                )
                .clip(
                    lower=0.0,
                    upper=1.0,
                )
                .astype(np.float64)
            )

        return df

    # =========================================================================
    # SCORE NORMALIZATION
    # =========================================================================

    def _normalize_score_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Guarantee threat/reputation scores remain within [0, 100].
        """

        df = dataframe.copy()

        for feature in (
            self.SCORE_FEATURES
        ):

            if feature not in df.columns:
                continue

            df[feature] = (
                pd.to_numeric(
                    df[feature],
                    errors="coerce",
                )
                .clip(
                    lower=0.0,
                    upper=100.0,
                )
                .astype(np.float64)
            )

        return df

    # =========================================================================
    # HISTORICAL NORMALIZATION
    # =========================================================================

    def _normalize_historical_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalize historical age fields.

        Semantics:

            -1 = unavailable
             0 = today
            >0 = elapsed days
        """

        df = dataframe.copy()

        for feature in (
            self.HISTORICAL_FEATURES
        ):

            if feature not in df.columns:
                continue

            maximum = self.outlier_caps.get(
                feature,
                100000.0,
            )

            df[feature] = (
                pd.to_numeric(
                    df[feature],
                    errors="coerce",
                )
                .clip(
                    lower=-1.0,
                    upper=float(maximum),
                )
                .astype(np.float64)
            )

        return df

    # =========================================================================
    # FINAL DATAFRAME
    # =========================================================================

    def _finalize_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Produce the final XGBoost-compatible DataFrame.

        Guarantees:

            - DataFrame
            - canonical 20-feature schema
            - deterministic ordering
            - numeric dtype
            - no NaN
            - no infinity
        """

        if dataframe is None:
            raise ValueError(
                "Threat model DataFrame cannot be None."
            )

        df = dataframe.copy()

        defaults = (
            ThreatFeatureSchema.get_default_values()
        )

        # -------------------------------------------------------------
        # Add any missing canonical features.
        # -------------------------------------------------------------

        for feature in self.schema_columns:

            if feature not in df.columns:

                df[feature] = defaults.get(
                    feature,
                    0.0,
                )

        # -------------------------------------------------------------
        # Keep ONLY canonical model features.
        # -------------------------------------------------------------

        df = df.loc[
            :,
            self.schema_columns,
        ].copy()

        # -------------------------------------------------------------
        # Numeric conversion.
        # -------------------------------------------------------------

        for feature in self.schema_columns:

            df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

        # -------------------------------------------------------------
        # Remove infinity.
        # -------------------------------------------------------------

        df = df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # -------------------------------------------------------------
        # Feature-specific defaults.
        # -------------------------------------------------------------

        for feature in self.schema_columns:

            default_value = defaults.get(
                feature,
                0.0,
            )

            df[feature] = (
                df[feature]
                .fillna(default_value)
            )

        # -------------------------------------------------------------
        # Final dtype.
        # -------------------------------------------------------------

        df = df.astype(
            np.float64
        )

        # -------------------------------------------------------------
        # Final canonical order.
        # -------------------------------------------------------------

        df = df.loc[
            :,
            self.schema_columns,
        ]

        return df

    # =========================================================================
    # SAFE SINGLE-INSTANCE FALLBACK
    # =========================================================================

    def _create_safe_fallback_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Return the complete Threat schema default vector.

        This is intended only for single-instance inference failure.

        Training/batch failures are never silently replaced by this vector.
        """

        try:

            dataframe = (
                ThreatFeatureSchema
                .to_default_dataframe()
            )

            dataframe = dataframe.loc[
                :,
                self.schema_columns,
            ]

            return dataframe.astype(
                np.float64
            )

        except Exception as exc:

            logger.critical(
                "Unable to construct Threat preprocessing "
                "fallback: %s",
                exc,
                exc_info=True,
            )

            return pd.DataFrame(
                [
                    {
                        feature: 0.0
                        for feature in self.schema_columns
                    }
                ],
                columns=self.schema_columns,
                dtype=np.float64,
            )

    # =========================================================================
    # CUSTOM CAP VALIDATION
    # =========================================================================

    @staticmethod
    def _sanitize_custom_caps(
        caps: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Validate custom operational outlier caps.
        """

        sanitized: Dict[str, float] = {}

        for feature, value in caps.items():

            try:

                numeric_value = float(
                    value
                )

            except (
                ValueError,
                TypeError,
            ):

                logger.warning(
                    "Ignoring invalid custom cap "
                    "for '%s': %r",
                    feature,
                    value,
                )

                continue

            if not math.isfinite(
                numeric_value
            ):

                logger.warning(
                    "Ignoring non-finite custom cap "
                    "for '%s'.",
                    feature,
                )

                continue

            if numeric_value < 0.0:

                logger.warning(
                    "Ignoring negative custom cap "
                    "for '%s'.",
                    feature,
                )

                continue

            # Only canonical model features are accepted.
            if feature not in (
                ThreatFeatureSchema.get_schema_columns()
            ):

                logger.warning(
                    "Ignoring custom cap for unknown "
                    "Threat feature '%s'.",
                    feature,
                )

                continue

            sanitized[
                feature
            ] = numeric_value

        return sanitized

    # =========================================================================
    # DATASET DIAGNOSTICS
    # =========================================================================

    def get_dataset_diagnostics(
        self,
        dataframe: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Generate diagnostics for a Threat dataset.

        Extra columns are reported but are not considered model features.
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            return {
                "valid": False,
                "error": (
                    "Input is not a pandas DataFrame."
                ),
            }

        missing_columns = [
            feature
            for feature in self.schema_columns
            if feature not in dataframe.columns
        ]

        extra_columns = [
            column
            for column in dataframe.columns
            if column not in self.schema_columns
        ]

        numeric_dataframe = (
            dataframe.select_dtypes(
                include=[np.number]
            )
        )

        has_infinite_values = False

        if not numeric_dataframe.empty:

            has_infinite_values = bool(
                np.isinf(
                    numeric_dataframe.to_numpy()
                ).any()
            )

        return {
            "valid": True,
            "preprocessor_version": (
                self.PREPROCESSOR_VERSION
            ),
            "row_count": int(
                len(dataframe)
            ),
            "input_column_count": int(
                len(dataframe.columns)
            ),
            "expected_feature_count": int(
                self.feature_count
            ),
            "schema_feature_count": int(
                ThreatFeatureSchema.get_feature_count()
            ),
            "missing_schema_columns": (
                missing_columns
            ),
            "missing_schema_column_count": int(
                len(missing_columns)
            ),
            "extra_column_count": int(
                len(extra_columns)
            ),
            "extra_columns": (
                extra_columns
            ),
            "has_duplicate_columns": bool(
                dataframe.columns
                .duplicated()
                .any()
            ),
            "has_nan_values": bool(
                dataframe.isna()
                .any()
                .any()
            ),
            "has_infinite_values": (
                has_infinite_values
            ),
            "feature_order": list(
                self.schema_columns
            ),
        }

    # =========================================================================
    # BATCH FEATURE COVERAGE
    # =========================================================================

    def calculate_batch_feature_coverage(
        self,
        dataframe: pd.DataFrame,
    ) -> float:
        """
        Calculate the percentage of canonical ML features supplied by a
        dataset before defaults are added.
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            return 0.0

        if self.feature_count == 0:
            return 0.0

        available = sum(
            1
            for feature in self.schema_columns
            if feature in dataframe.columns
        )

        return float(
            available
            / self.feature_count
        )

    # =========================================================================
    # MODEL INPUT VALIDATION
    # =========================================================================

    def validate_model_input(
        self,
        dataframe: pd.DataFrame,
    ) -> Tuple[bool, List[str]]:
        """
        Strictly validate a DataFrame before XGBoost inference/training.
        """

        problems: List[str] = []

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            problems.append(
                "Input is not a pandas DataFrame."
            )

            return False, problems

        if dataframe.empty:

            problems.append(
                "DataFrame contains zero rows."
            )

        expected_columns = (
            self.schema_columns
        )

        actual_columns = list(
            dataframe.columns
        )

        # -------------------------------------------------------------
        # Exact feature set AND exact order.
        # -------------------------------------------------------------

        if actual_columns != expected_columns:

            problems.append(
                "Feature columns do not exactly match "
                "ThreatFeatureSchema ordering."
            )

        # -------------------------------------------------------------
        # Row count.
        # -------------------------------------------------------------

        if len(dataframe) == 0:

            problems.append(
                "DataFrame contains no observations."
            )

        # -------------------------------------------------------------
        # NaN.
        # -------------------------------------------------------------

        if dataframe.isna().any().any():

            problems.append(
                "DataFrame contains NaN values."
            )

        # -------------------------------------------------------------
        # Numeric dtype.
        # -------------------------------------------------------------

        non_numeric_columns = [
            column
            for column in dataframe.columns
            if not pd.api.types.is_numeric_dtype(
                dataframe[column]
            )
        ]

        if non_numeric_columns:

            problems.append(
                "Non-numeric model features: "
                + ", ".join(
                    map(
                        str,
                        non_numeric_columns,
                    )
                )
            )

        # -------------------------------------------------------------
        # Infinity / finite values.
        # -------------------------------------------------------------

        if not dataframe.empty:

            try:

                numeric_values = (
                    dataframe.to_numpy(
                        dtype=np.float64
                    )
                )

                if not np.isfinite(
                    numeric_values
                ).all():

                    problems.append(
                        "DataFrame contains "
                        "infinite or non-finite values."
                    )

            except Exception as exc:

                problems.append(
                    "Unable to verify numerical "
                    f"finite values: {exc}"
                )

        # -------------------------------------------------------------
        # Feature count.
        # -------------------------------------------------------------

        if len(dataframe.columns) != (
            self.feature_count
        ):

            problems.append(
                "Unexpected feature count: "
                f"{len(dataframe.columns)}; "
                f"expected {self.feature_count}."
            )

        return (
            len(problems) == 0,
            problems,
        )

    # =========================================================================
    # FEATURE ORDER
    # =========================================================================

    def get_feature_order(self) -> List[str]:
        """
        Return the exact canonical feature order.
        """

        return list(
            self.schema_columns
        )

    # =========================================================================
    # FEATURE COUNT
    # =========================================================================

    def get_feature_count(self) -> int:
        """
        Return the canonical feature count.
        """

        return self.feature_count

    # =========================================================================
    # PREPROCESSOR CONFIGURATION
    # =========================================================================

    def get_configuration(
        self,
    ) -> Dict[str, Any]:
        """
        Return a serializable preprocessing configuration.
        """

        return {
            "preprocessor_version": (
                self.PREPROCESSOR_VERSION
            ),
            "schema_feature_count": (
                ThreatFeatureSchema.get_feature_count()
            ),
            "feature_count": (
                self.feature_count
            ),
            "feature_order": list(
                self.schema_columns
            ),
            "schema_signature": (
                ThreatFeatureSchema
                .get_schema_signature()
            ),
            "outlier_caps": dict(
                self.outlier_caps
            ),
            "binary_features": sorted(
                self.BINARY_FEATURES
            ),
            "ratio_features": sorted(
                self.RATIO_FEATURES
            ),
            "score_features": sorted(
                self.SCORE_FEATURES
            ),
            "count_features": sorted(
                self.COUNT_FEATURES
            ),
            "historical_features": sorted(
                self.HISTORICAL_FEATURES
            ),
        }