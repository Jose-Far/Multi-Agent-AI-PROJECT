"""
DNS AI Agent - Preprocessing Module
====================================

This module provides the preprocessing layer for the DNS Security AI Agent.

Architecture
------------

    Raw DNS Telemetry
            |
            v
    DNSFeatureSchema
            |
            |  Schema normalization
            |  Alias resolution
            |  Boolean conversion
            |  SPF / DMARC conversion
            |  Missing feature handling
            v
    DNSPreprocessor
            |
            |  Numeric sanitization
            |  Range validation
            |  Outlier capping
            |  Optional transformations
            |  Deterministic column ordering
            v
    Model-Ready DataFrame
            |
            v
       ML Predictor


Important Design Principles
----------------------------

1. The feature schema is the single source of truth.
2. Training and inference must use exactly the same preprocessing logic.
3. Feature order must never change.
4. NaN and infinity values must never reach the ML model.
5. DNS-specific categorical policies are normalized by DNSFeatureSchema.
6. Tree-based models such as XGBoost generally do not require standardization.
7. Preprocessing must not independently decide whether a domain is malicious.
8. DNS signals are evidence and should later be combined by the Decision Fusion
   Engine with signals from other cybersecurity agents.

Author:
    Multi-Agent AI Cybersecurity Analyst

Agent:
    DNS Security AI Agent

Version:
    1.0.0
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .feature_schema import (
    DNSFeatureSchema,
    SCHEMA_VERSION,
)


# =============================================================================
# LOGGER
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# PREPROCESSOR VERSION
# =============================================================================

PREPROCESSOR_VERSION = "1.0.0"


# =============================================================================
# DNS PREPROCESSOR
# =============================================================================

class DNSPreprocessor:
    """
    Production-grade preprocessing engine for the DNS Security AI Agent.

    The preprocessor is responsible for transforming validated DNS telemetry
    into deterministic numeric data suitable for machine-learning inference.

    It supports:

        - Single-record inference
        - Batch training preprocessing
        - Schema alignment
        - Feature alias normalization
        - Missing-value handling
        - Infinity handling
        - Numeric coercion
        - Outlier capping
        - Feature-range enforcement
        - Deterministic feature ordering
        - Training/inference consistency
        - DataFrame validation
        - Preprocessing statistics
        - Feature-vector generation
        - Debugging and audit information

    It deliberately does NOT perform:

        - DNS resolution
        - DNS querying
        - ML prediction
        - Risk scoring
        - Explainability
        - phishing classification

    Those responsibilities belong to other modules.
    """

    # =========================================================================
    # 1. OUTLIER CAPS
    # =========================================================================

    """
    Reasonable operational caps are used to protect the model from extreme
    telemetry values.

    These are not intended to say that values above these limits are malicious.

    Example:

        500 A records

    should not automatically mean phishing.

    Instead, the value is capped to a range that is useful and stable for
    the initial model.

    IMPORTANT:
        The caps must remain identical during training and inference.
    """

    DEFAULT_OUTLIER_CAPS: Dict[str, float] = {

        # ---------------------------------------------------------------------
        # Routing / record counts
        # ---------------------------------------------------------------------

        "a_record_count": 50.0,
        "aaaa_record_count": 50.0,
        "ns_count": 20.0,
        "cname_count": 20.0,
        "total_resolved_ips": 100.0,

        # ---------------------------------------------------------------------
        # TTL values
        # ---------------------------------------------------------------------

        "min_ttl_value": 604800.0,
        "max_ttl_value": 604800.0,
        "avg_ttl_value": 604800.0,

        # ---------------------------------------------------------------------
        # MX
        # ---------------------------------------------------------------------

        "mx_record_count": 20.0,
        "lowest_mx_preference": 65535.0,

        # ---------------------------------------------------------------------
        # TXT
        # ---------------------------------------------------------------------

        "txt_record_count": 100.0,
        "avg_txt_length": 10000.0,

        # ---------------------------------------------------------------------
        # SPF
        # ---------------------------------------------------------------------

        "spf_record_count": 10.0,
        "spf_includes_count": 100.0,

        # ---------------------------------------------------------------------
        # Ordinal values
        # ---------------------------------------------------------------------

        "spf_strictness_score": 4.0,
        "dmarc_policy_score": 3.0,
        "dmarc_subdomain_policy_score": 3.0,
    }

    # =========================================================================
    # 2. LOWER BOUNDS
    # =========================================================================

    """
    DNS counts and TTL values cannot logically be negative.

    Boolean values are already constrained to 0/1 by DNSFeatureSchema.
    """

    DEFAULT_LOWER_BOUNDS: Dict[str, float] = {

        "a_record_count": 0.0,
        "aaaa_record_count": 0.0,
        "ns_count": 0.0,
        "cname_count": 0.0,
        "total_resolved_ips": 0.0,

        "min_ttl_value": 0.0,
        "max_ttl_value": 0.0,
        "avg_ttl_value": 0.0,

        "mx_record_count": 0.0,
        "lowest_mx_preference": 0.0,

        "txt_record_count": 0.0,
        "avg_txt_length": 0.0,

        "spf_record_count": 0.0,
        "spf_includes_count": 0.0,

        "spf_strictness_score": 0.0,
        "dmarc_policy_score": 0.0,
        "dmarc_subdomain_policy_score": 0.0,
    }

    # =========================================================================
    # 3. BOOLEAN FEATURES
    # =========================================================================

    BOOLEAN_FEATURES = {
        "has_a_records",
        "has_aaaa_records",
        "has_cname_record",
        "has_routing_anomalies",

        "is_fast_flux_candidate",
        "is_active_fast_flux",

        "has_mx_records",
        "uses_free_mail_provider",
        "uses_disposable_mail_provider",
        "has_suspicious_mx_exchange",

        "has_txt_records",
        "has_domain_verification",
        "has_suspicious_long_txt",
        "has_base64_payload_in_txt",

        "has_spf_record",
        "has_multiple_spf_records",

        "has_dmarc_record",

        "is_email_spoofable",
    }

    # =========================================================================
    # 4. COUNT / NUMERIC FEATURES
    # =========================================================================

    NUMERIC_FEATURES = {
        "a_record_count",
        "aaaa_record_count",
        "ns_count",
        "cname_count",
        "total_resolved_ips",

        "min_ttl_value",
        "max_ttl_value",
        "avg_ttl_value",

        "mx_record_count",
        "lowest_mx_preference",

        "txt_record_count",
        "avg_txt_length",

        "spf_record_count",
        "spf_includes_count",
    }

    # =========================================================================
    # 5. ORDINAL FEATURES
    # =========================================================================

    ORDINAL_FEATURES = {
        "spf_strictness_score",
        "dmarc_policy_score",
        "dmarc_subdomain_policy_score",
    }

    # =========================================================================
    # 6. INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        outlier_caps: Optional[Mapping[str, float]] = None,
        lower_bounds: Optional[Mapping[str, float]] = None,
        enable_outlier_capping: bool = True,
        enable_range_validation: bool = True,
    ) -> None:
        """
        Initialize the DNS preprocessor.

        Parameters
        ----------
        outlier_caps:
            Optional custom upper bounds.

        lower_bounds:
            Optional custom lower bounds.

        enable_outlier_capping:
            Whether numerical outlier capping should be applied.

        enable_range_validation:
            Whether schema-level range validation should be enforced.

        Notes
        -----
        For production training and inference, use the same configuration.
        """

        self.enable_outlier_capping = bool(
            enable_outlier_capping
        )

        self.enable_range_validation = bool(
            enable_range_validation
        )

        self.outlier_caps: Dict[str, float] = dict(
            self.DEFAULT_OUTLIER_CAPS
        )

        if outlier_caps:
            self.outlier_caps.update(
                {
                    str(key): float(value)
                    for key, value in outlier_caps.items()
                }
            )

        self.lower_bounds: Dict[str, float] = dict(
            self.DEFAULT_LOWER_BOUNDS
        )

        if lower_bounds:
            self.lower_bounds.update(
                {
                    str(key): float(value)
                    for key, value in lower_bounds.items()
                }
            )

        self._validate_configuration()

        logger.debug(
            "DNSPreprocessor initialized. "
            "Schema version=%s, preprocessor version=%s",
            SCHEMA_VERSION,
            PREPROCESSOR_VERSION,
        )

    # =========================================================================
    # 7. CONFIGURATION VALIDATION
    # =========================================================================

    def _validate_configuration(self) -> None:
        """
        Validate preprocessing configuration.

        This catches configuration errors early rather than during model
        inference.
        """

        schema_features = set(
            DNSFeatureSchema.get_schema()
        )

        for feature, cap in self.outlier_caps.items():

            if feature not in schema_features:
                raise ValueError(
                    f"Outlier cap references unknown DNS feature: {feature}"
                )

            if not math.isfinite(float(cap)):
                raise ValueError(
                    f"Outlier cap for '{feature}' must be finite."
                )

            if float(cap) < 0:
                raise ValueError(
                    f"Outlier cap for '{feature}' cannot be negative."
                )

        for feature, lower in self.lower_bounds.items():

            if feature not in schema_features:
                raise ValueError(
                    f"Lower bound references unknown DNS feature: {feature}"
                )

            if not math.isfinite(float(lower)):
                raise ValueError(
                    f"Lower bound for '{feature}' must be finite."
                )

        for feature in schema_features:

            if (
                feature in self.outlier_caps
                and feature in self.lower_bounds
            ):

                lower = self.lower_bounds[feature]
                upper = self.outlier_caps[feature]

                if lower > upper:
                    raise ValueError(
                        f"Invalid range for '{feature}': "
                        f"lower={lower}, upper={upper}"
                    )

    # =========================================================================
    # 8. SINGLE RECORD TRANSFORMATION
    # =========================================================================

    def transform_single(
        self,
        raw_features: Mapping[str, Any],
    ) -> pd.DataFrame:
        """
        Transform one raw DNS feature dictionary into a model-ready DataFrame.

        Parameters
        ----------
        raw_features:
            Raw DNS telemetry generated by the DNS feature extraction layer.

        Returns
        -------
        pandas.DataFrame
            One-row DataFrame containing exactly the canonical DNS features.

        Raises
        ------
        TypeError
            If raw_features is not a mapping.

        ValueError
            If preprocessing produces an invalid schema.

        Example
        -------
        raw_features = {
            "has_a_records": True,
            "a_record_count": 2,
            "spf_strictness": "strict",
            "dmarc_policy": "reject"
        }

        df = preprocessor.transform_single(raw_features)
        """

        if not isinstance(raw_features, Mapping):
            raise TypeError(
                "raw_features must be a dictionary or mapping."
            )

        try:

            logger.debug(
                "Starting DNS single-record preprocessing."
            )

            # -------------------------------------------------------------
            # Step 1:
            # Normalize extractor representations BEFORE schema validation.
            #
            # DNS TXT/DMARC extraction produces SPF strictness on 0-100,
            # while the canonical ML schema stores it on 0-4.
            # This must happen before align_and_validate(), otherwise the
            # schema validator can clamp 100 -> 4 and information is lost.
            # -------------------------------------------------------------

            normalized_features = self._normalize_raw_feature_values(
                raw_features
            )

            # -------------------------------------------------------------
            # Step 2:
            # Canonical schema normalization
            # -------------------------------------------------------------

            df_aligned = DNSFeatureSchema.align_and_validate(
                normalized_features
            )

            # -------------------------------------------------------------
            # Step 2:
            # Apply ML preprocessing
            # -------------------------------------------------------------

            df_processed = self._apply_transformations(
                df_aligned
            )

            # -------------------------------------------------------------
            # Step 3:
            # Final validation
            # -------------------------------------------------------------

            self._validate_processed_dataframe(
                df_processed,
                expected_rows=1,
            )

            logger.debug(
                "DNS single-record preprocessing completed successfully."
            )

            return df_processed

        except Exception as exc:

            logger.exception(
                "DNS single-record preprocessing failed: %s",
                exc,
            )

            raise

    # =========================================================================
    # 9. BATCH TRANSFORMATION
    # =========================================================================

    def transform_batch(
        self,
        df_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform a batch DataFrame into model-ready DNS features.

        This function is primarily intended for training.

        The function guarantees:

            - canonical feature names
            - canonical feature order
            - numeric data
            - no NaN
            - no infinity
            - consistent outlier handling

        Parameters
        ----------
        df_raw:
            Raw training DataFrame.

        Returns
        -------
        pandas.DataFrame
            Preprocessed training DataFrame.
        """

        if not isinstance(
            df_raw,
            pd.DataFrame,
        ):
            raise TypeError(
                "df_raw must be a pandas DataFrame."
            )

        if df_raw.empty:
            raise ValueError(
                "DNS training DataFrame is empty."
            )

        try:

            logger.debug(
                "Starting DNS batch preprocessing. "
                "Rows=%d, Columns=%d",
                len(df_raw),
                len(df_raw.columns),
            )

            # Normalize extractor-scale SPF scores BEFORE schema/range
            # validation. The training CSV contains extractor values such as
            # 100.0, while the model schema expects 0-4.
            df_normalized = self._normalize_raw_batch_values(
                df_raw
            )

            df_aligned = self._align_batch_schema(
                df_normalized
            )

            df_processed = self._apply_transformations(
                df_aligned
            )

            self._validate_processed_dataframe(
                df_processed,
                expected_rows=len(df_raw),
            )

            logger.debug(
                "DNS batch preprocessing completed successfully."
            )

            return df_processed

        except Exception as exc:

            logger.exception(
                "DNS batch preprocessing failed: %s",
                exc,
            )

            raise

    # =========================================================================
    # 10. BATCH SCHEMA ALIGNMENT
    # =========================================================================

    def _align_batch_schema(
        self,
        df_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align a batch DataFrame to the canonical DNS schema.

        Missing features are created with their schema defaults.

        Extra columns are removed.

        Column order is deterministic.
        """

        df = df_raw.copy()

        schema_columns = DNSFeatureSchema.get_schema()

        # ---------------------------------------------------------------------
        # Rename legacy aliases where possible.
        # ---------------------------------------------------------------------

        rename_map: Dict[str, str] = {}

        for alias, canonical in DNSFeatureSchema.FEATURE_ALIASES.items():

            if alias in df.columns and canonical not in df.columns:

                rename_map[alias] = canonical

        if rename_map:

            logger.debug(
                "Normalizing DNS batch feature aliases: %s",
                rename_map,
            )

            df = df.rename(
                columns=rename_map
            )

        # ---------------------------------------------------------------------
        # Add missing canonical features.
        # ---------------------------------------------------------------------

        for feature_name in schema_columns:

            if feature_name not in df.columns:

                logger.debug(
                    "Adding missing DNS training feature '%s' with default 0.0.",
                    feature_name,
                )

                df[feature_name] = 0.0

        # ---------------------------------------------------------------------
        # Remove non-schema columns.
        #
        # This prevents accidental metadata leakage such as:
        #
        # domain
        # url
        # label
        # timestamp
        # source
        #
        # from entering the model feature matrix.
        # ---------------------------------------------------------------------

        extra_columns = [
            column
            for column in df.columns
            if column not in schema_columns
        ]

        if extra_columns:

            logger.debug(
                "Removing non-schema DNS columns: %s",
                extra_columns,
            )

            df = df.drop(
                columns=extra_columns
            )

        # ---------------------------------------------------------------------
        # Deterministic order.
        # ---------------------------------------------------------------------

        df = df[
            schema_columns
        ]

        return df

    # =========================================================================
    # 11. MAIN TRANSFORMATION PIPELINE
    # =========================================================================

    def _apply_transformations(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply all deterministic preprocessing operations.

        Pipeline:

            1. Copy input
            2. Ensure schema
            3. Convert values to numeric
            4. Replace infinity
            5. Impute missing values
            6. Enforce lower bounds
            7. Apply upper caps
            8. Enforce ordinal ranges
            9. Cast to float
            10. Reorder columns
            11. Final sanitation
        """

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "_apply_transformations() requires a DataFrame."
            )

        processed = df.copy()

        # ---------------------------------------------------------------------
        # Step 1:
        # Ensure canonical columns.
        # ---------------------------------------------------------------------

        schema_columns = DNSFeatureSchema.get_schema()

        missing_columns = [
            feature
            for feature in schema_columns
            if feature not in processed.columns
        ]

        for feature in missing_columns:

            processed[feature] = 0.0

        # Remove anything outside the canonical feature contract.
        processed = processed[
            schema_columns
        ]

        # ---------------------------------------------------------------------
        # Step 2:
        # Numeric coercion.
        # ---------------------------------------------------------------------

        for feature in schema_columns:

            processed[feature] = pd.to_numeric(
                processed[feature],
                errors="coerce",
            )

        # ---------------------------------------------------------------------
        # Step 3:
        # Infinity → NaN.
        # ---------------------------------------------------------------------

        processed = processed.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # ---------------------------------------------------------------------
        # Step 4:
        # Missing-value imputation.
        # ---------------------------------------------------------------------

        processed = processed.fillna(
            0.0
        )

        # ---------------------------------------------------------------------
        # Step 5:
        # Lower-bound enforcement.
        # ---------------------------------------------------------------------

        processed = self._apply_lower_bounds(
            processed
        )

        # ---------------------------------------------------------------------
        # Step 6:
        # Outlier capping.
        # ---------------------------------------------------------------------

        if self.enable_outlier_capping:

            processed = self._apply_outlier_caps(
                processed
            )

        # ---------------------------------------------------------------------
        # Step 7:
        # Normalize SPF strictness.
        #
        # DNS TXT extraction currently produces SPF strictness on a 0-100
        # scale (for example: 100=strict, 50=neutral, 0=none).
        #
        # DNSFeatureSchema stores the ML representation on a 0-4 scale.
        # Convert the extractor representation before schema range validation.
        #
        # This keeps training and inference deterministic and prevents a
        # legitimate SPF score of 100 from being incorrectly treated as an
        # outlier and merely clamped to 4.
        # ---------------------------------------------------------------------

        processed = self._normalize_spf_strictness_score(
            processed
        )

        # ---------------------------------------------------------------------
        # Step 8:
        # Schema-specific range enforcement.
        # ---------------------------------------------------------------------

        if self.enable_range_validation:

            processed = self._apply_schema_ranges(
                processed
            )

        # ---------------------------------------------------------------------
        # Step 8:
        # Strict float conversion.
        # ---------------------------------------------------------------------

        for feature in schema_columns:

            processed[feature] = processed[
                feature
            ].astype(float)

        # ---------------------------------------------------------------------
        # Step 9:
        # Final NaN / infinity protection.
        # ---------------------------------------------------------------------

        processed = processed.replace(
            [np.inf, -np.inf],
            0.0,
        )

        processed = processed.fillna(
            0.0
        )

        # ---------------------------------------------------------------------
        # Step 10:
        # Final deterministic order.
        # ---------------------------------------------------------------------

        processed = processed[
            schema_columns
        ]

        return processed

    # =========================================================================
    # 12. LOWER-BOUND TRANSFORMATION
    # =========================================================================

    def _apply_lower_bounds(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply lower bounds to DNS numerical features.

        Negative record counts and negative TTL values are invalid and
        therefore normalized to zero.
        """

        processed = df.copy()

        for feature, lower_bound in self.lower_bounds.items():

            if feature not in processed.columns:
                continue

            processed[feature] = processed[
                feature
            ].clip(
                lower=lower_bound
            )

        return processed

    # =========================================================================
    # 13. OUTLIER CAPPING
    # =========================================================================

    def _apply_outlier_caps(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply deterministic upper bounds to numerical DNS features.

        This is intentionally implemented using clipping rather than
        statistical fitting so that training and inference remain identical.

        No dataset-dependent statistics are learned here.
        """

        processed = df.copy()

        for feature, upper_bound in self.outlier_caps.items():

            if feature not in processed.columns:
                continue

            before_max = processed[
                feature
            ].max()

            processed[feature] = processed[
                feature
            ].clip(
                upper=upper_bound
            )

            after_max = processed[
                feature
            ].max()

            if (
                pd.notna(before_max)
                and pd.notna(after_max)
                and before_max != after_max
            ):

                logger.debug(
                    "DNS feature '%s' capped from max=%s to max=%s.",
                    feature,
                    before_max,
                    after_max,
                )

        return processed

    # =========================================================================
    # 14. RAW SPF STRICTNESS NORMALIZATION
    # =========================================================================

    @staticmethod
    def _spf_score_to_model_scale(value: Any) -> Any:
        """
        Convert extractor SPF strictness from 0-100 to the model's 0-4 scale.

        Canonical mapping:

            0   -> 0.0
            25  -> 1.0
            50  -> 2.0
            75  -> 3.0
            100 -> 4.0

        Already-normalized values in the 0-4 range are preserved. This makes
        the method safe for inference payloads that already use the canonical
        model representation.
        """
        if value is None:
            return value

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return value

        if not math.isfinite(numeric):
            return value

        # Canonical model-scale values are already normalized.
        if 0.0 <= numeric <= 4.0:
            return numeric

        # Extractor-scale SPF scores use 0-100.
        return max(0.0, min(100.0, numeric)) / 25.0

    def _normalize_raw_feature_values(
        self,
        raw_features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Normalize extractor-scale values before schema validation."""
        normalized = dict(raw_features)

        if "spf_strictness_score" in normalized:
            normalized["spf_strictness_score"] = (
                self._spf_score_to_model_scale(
                    normalized["spf_strictness_score"]
                )
            )

        return normalized

    def _normalize_raw_batch_values(
        self,
        df_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalize extractor-scale values before batch schema validation."""
        normalized = df_raw.copy()

        if "spf_strictness_score" in normalized.columns:
            normalized["spf_strictness_score"] = normalized[
                "spf_strictness_score"
            ].apply(self._spf_score_to_model_scale)

        return normalized

    @staticmethod
    def _normalize_spf_strictness_score(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert SPF strictness from the extractor's 0-100 scale to the
        canonical DNS model's 0-4 scale before range enforcement.

        Values already in the canonical 0-4 range are preserved.
        """
        processed = df.copy()

        feature = "spf_strictness_score"
        if feature not in processed.columns:
            return processed

        values = pd.to_numeric(
            processed[feature],
            errors="coerce",
        )

        # Preserve already-normalized model values.
        mask_model_scale = values.between(0.0, 4.0, inclusive="both")

        # Convert extractor-scale values (0-100) to model scale (0-4).
        converted = values.clip(
            lower=0.0,
            upper=100.0,
        ) / 25.0

        processed[feature] = converted.where(
            ~mask_model_scale,
            values,
        )

        return processed

    # =========================================================================
    # 15. SCHEMA RANGE ENFORCEMENT
    # =========================================================================

    def _apply_schema_ranges(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Enforce min/max constraints defined in DNSFeatureSchema metadata.

        This is separate from operational outlier capping.

        Example:

            SPF strictness score
                0 <= score <= 4 (after 0-100 extractor normalization)

            DMARC policy score
                0 <= score <= 3
        """

        processed = df.copy()

        for feature_name in DNSFeatureSchema.get_schema():

            metadata = DNSFeatureSchema.get_feature_metadata(
                feature_name
            )

            if metadata is None:
                continue

            if feature_name not in processed.columns:
                continue

            if metadata.min_value is not None:

                processed[feature_name] = processed[
                    feature_name
                ].clip(
                    lower=metadata.min_value
                )

            if metadata.max_value is not None:

                processed[feature_name] = processed[
                    feature_name
                ].clip(
                    upper=metadata.max_value
                )

        return processed

    # =========================================================================
    # 16. FINAL DATAFRAME VALIDATION
    # =========================================================================

    def _validate_processed_dataframe(
        self,
        df: pd.DataFrame,
        expected_rows: Optional[int] = None,
    ) -> None:
        """
        Perform strict validation of the final model-ready DataFrame.
        """

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "Processed DNS data must be a pandas DataFrame."
            )

        if expected_rows is not None:

            if len(df) != expected_rows:

                raise ValueError(
                    "Unexpected number of rows after DNS preprocessing. "
                    f"Expected={expected_rows}, received={len(df)}."
                )

        # Use the authoritative schema validator.
        DNSFeatureSchema.validate_dataframe(
            df
        )

        # Additional numerical safety.
        values = df.to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():

            raise ValueError(
                "DNS preprocessing produced non-finite values."
            )

    # =========================================================================
    # 17. FEATURE VECTOR
    # =========================================================================

    def transform_to_vector(
        self,
        raw_features: Mapping[str, Any],
    ) -> List[float]:
        """
        Transform raw DNS telemetry directly into an ordered ML vector.

        Useful when a predictor expects:

            model.predict([vector])

        instead of a pandas DataFrame.
        """

        dataframe = self.transform_single(
            raw_features
        )

        return [
            float(value)
            for value in dataframe.iloc[0].tolist()
        ]

    # =========================================================================
    # 18. NUMPY ARRAY OUTPUT
    # =========================================================================

    def transform_to_numpy(
        self,
        raw_features: Mapping[str, Any],
    ) -> np.ndarray:
        """
        Transform raw DNS telemetry into a NumPy array.

        Shape:

            (1, 37)
        """

        dataframe = self.transform_single(
            raw_features
        )

        return dataframe.to_numpy(
            dtype=np.float64
        )

    # =========================================================================
    # 19. BATCH NUMPY OUTPUT
    # =========================================================================

    def transform_batch_to_numpy(
        self,
        df_raw: pd.DataFrame,
    ) -> np.ndarray:
        """
        Transform a batch DataFrame into a numeric NumPy matrix.
        """

        dataframe = self.transform_batch(
            df_raw
        )

        return dataframe.to_numpy(
            dtype=np.float64
        )

    # =========================================================================
    # 20. FEATURE DICTIONARY OUTPUT
    # =========================================================================

    def transform_to_dict(
        self,
        raw_features: Mapping[str, Any],
    ) -> Dict[str, float]:
        """
        Transform raw DNS telemetry into an ordered canonical dictionary.
        """

        dataframe = self.transform_single(
            raw_features
        )

        return DNSFeatureSchema.dataframe_to_dict(
            dataframe
        )

    # =========================================================================
    # 21. PREPROCESSING INFORMATION
    # =========================================================================

    def get_configuration(self) -> Dict[str, Any]:
        """
        Return the preprocessing configuration.

        Useful for logging, debugging and experiment reproducibility.
        """

        return {
            "schema_version": SCHEMA_VERSION,
            "preprocessor_version": PREPROCESSOR_VERSION,
            "feature_count": DNSFeatureSchema.feature_count(),
            "enable_outlier_capping": (
                self.enable_outlier_capping
            ),
            "enable_range_validation": (
                self.enable_range_validation
            ),
            "outlier_caps": dict(
                self.outlier_caps
            ),
            "lower_bounds": dict(
                self.lower_bounds
            ),
        }

    # =========================================================================
    # 22. FEATURE STATISTICS
    # =========================================================================

    @staticmethod
    def get_dataframe_statistics(
        df: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate simple numerical statistics for a processed DataFrame.

        This function is useful during training/debugging.

        It does not alter the DataFrame.
        """

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        statistics: Dict[str, Dict[str, float]] = {}

        for column in df.columns:

            series = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            statistics[column] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "missing": float(series.isna().sum()),
                "unique": float(series.nunique()),
            }

        return statistics

    # =========================================================================
    # 23. PREPROCESSING AUDIT
    # =========================================================================

    def audit_single(
        self,
        raw_features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce an audit report showing how one DNS feature vector was
        interpreted by the preprocessing pipeline.

        This is useful during development and debugging.
        """

        schema_report = (
            DNSFeatureSchema.validation_report(
                raw_features
            )
        )

        processed = self.transform_single(
            raw_features
        )

        return {
            "schema_version": SCHEMA_VERSION,
            "preprocessor_version": PREPROCESSOR_VERSION,
            "schema_validation": schema_report,
            "feature_count": len(
                processed.columns
            ),
            "row_count": len(
                processed
            ),
            "feature_order": list(
                processed.columns
            ),
            "processed_features": (
                DNSFeatureSchema.dataframe_to_dict(
                    processed
                )
            ),
        }

    # =========================================================================
    # 24. MODEL FEATURE VALIDATION
    # =========================================================================

    def validate_model_feature_order(
        self,
        model_features: Iterable[str],
    ) -> bool:
        """
        Ensure the model's expected feature order matches the DNS schema.
        """

        model_features_list = list(
            model_features
        )

        return DNSFeatureSchema.validate_model_features(
            model_features_list
        )

    # =========================================================================
    # 25. RESET / STATE CHECK
    # =========================================================================

    def is_fitted(self) -> bool:
        """
        Return whether this preprocessor requires a fitted state.

        This implementation intentionally returns True because the
        preprocessing pipeline is deterministic and does not learn
        dataset-specific parameters.

        This makes it suitable for stateless production inference.
        """

        return True


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

def preprocess_dns_features(
    raw_features: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Convenience function for single DNS feature preprocessing.

    Example
    -------

        df = preprocess_dns_features(features)
    """

    preprocessor = DNSPreprocessor()

    return preprocessor.transform_single(
        raw_features
    )


def preprocess_dns_batch(
    df_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convenience function for DNS training-data preprocessing.
    """

    preprocessor = DNSPreprocessor()

    return preprocessor.transform_batch(
        df_raw
    )


def dns_features_to_numpy(
    raw_features: Mapping[str, Any],
) -> np.ndarray:
    """
    Convenience function returning a one-row NumPy matrix.
    """

    preprocessor = DNSPreprocessor()

    return preprocessor.transform_to_numpy(
        raw_features
    )


def dns_features_to_vector(
    raw_features: Mapping[str, Any],
) -> List[float]:
    """
    Convenience function returning an ordered Python list.
    """

    preprocessor = DNSPreprocessor()

    return preprocessor.transform_to_vector(
        raw_features
    )
