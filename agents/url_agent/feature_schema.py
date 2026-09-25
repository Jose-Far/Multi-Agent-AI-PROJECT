import logging
import math
from typing import Dict, Any, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class URLFeatureSchema:
    """
    Single source of truth for the URL AI Agent feature vector.

    The same ordered feature list MUST be used by:

        URL Feature Extraction
                ↓
        URLFeatureSchema
                ↓
        Preprocessing
                ↓
        Training
                ↓
        Prediction
                ↓
        Explainability

    IMPORTANT ML DESIGN DECISION
    ----------------------------

    URL entropy is separated into:

        url_entropy
        domain_entropy
        path_entropy
        query_entropy

    This prevents the model from treating all randomness in a URL
    as equally suspicious.

    Example:

        docs.google.com/forms/d/e/1FAIp...

    may have:

        low/moderate domain entropy
        high path entropy

    because Google Forms contains generated identifiers.

    Therefore:

        domain_entropy
            = stronger security signal

        path_entropy
            = contextual signal

        query_entropy
            = contextual/moderate signal

        url_entropy
            = global/contextual signal

    The schema itself does NOT assign ML weights. XGBoost learns
    those relationships during training.
    """

    # ======================================================================
    # SCHEMA VERSION
    # ======================================================================

    SCHEMA_VERSION = "2.0.0"

    # ======================================================================
    # FEATURE ORDER
    # ======================================================================

    EXPECTED_FEATURE_ORDER: List[str] = [

        # --------------------------------------------------------------
        # General URL structure
        # --------------------------------------------------------------

        "url_length",

        "domain_length",

        "path_length",

        "query_length",

        # --------------------------------------------------------------
        # Structural counts
        # --------------------------------------------------------------

        "num_dots",

        "num_hyphens",

        "num_underscores",

        "num_slashes",

        "num_question_marks",

        "num_equal_signs",

        "num_at_symbols",

        "num_percent_signs",

        "num_digits",

        "num_special_chars",

        # --------------------------------------------------------------
        # Entropy / randomness
        # --------------------------------------------------------------

        "url_entropy",

        "domain_entropy",

        "path_entropy",

        "query_entropy",

        # --------------------------------------------------------------
        # Domain/security indicators
        # --------------------------------------------------------------

        "contains_ip",

        "is_https",

        "has_punycode",

        "subdomain_count",

        "tld_length",

        "suspicious_word_count"
    ]

    # ======================================================================
    # FEATURE METADATA
    # ======================================================================

    FEATURE_METADATA: Dict[str, str] = {

        "url_length":
            "Overall character length of the complete URL.",

        "domain_length":
            "Character length of the hostname/domain.",

        "path_length":
            "Character length of the URL path excluding the domain and query.",

        "query_length":
            "Character length of the URL query string.",

        "num_dots":
            "Number of dot characters in the complete URL.",

        "num_hyphens":
            "Number of hyphen characters in the complete URL.",

        "num_underscores":
            "Number of underscore characters in the complete URL.",

        "num_slashes":
            "Number of slash characters in the complete URL.",

        "num_question_marks":
            "Number of question mark characters in the complete URL.",

        "num_equal_signs":
            "Number of equal-sign characters in the complete URL.",

        "num_at_symbols":
            "Number of @ characters in the complete URL.",

        "num_percent_signs":
            "Number of percent-encoded characters in the complete URL.",

        "num_digits":
            "Number of numeric characters in the complete URL.",

        "num_special_chars":
            "Number of special/non-alphanumeric characters.",

        "url_entropy":
            (
                "Shannon entropy of the complete URL. "
                "Used as a contextual global randomness feature, "
                "not as a standalone phishing indicator."
            ),

        "domain_entropy":
            (
                "Shannon entropy of the domain/hostname. "
                "Higher values can indicate algorithmically generated "
                "or obfuscated domains."
            ),

        "path_entropy":
            (
                "Shannon entropy of the URL path. "
                "High values may occur naturally in generated resource "
                "identifiers and should therefore be interpreted contextually."
            ),

        "query_entropy":
            (
                "Shannon entropy of the URL query string. "
                "Can indicate encoded or randomized parameters but may "
                "also occur on legitimate applications."
            ),

        "contains_ip":
            "Whether the URL uses an IP address instead of a domain name.",

        "is_https":
            "Whether the URL uses HTTPS.",

        "has_punycode":
            "Whether an internationalized/punycode domain is detected.",

        "subdomain_count":
            "Number of subdomain labels in the hostname.",

        "tld_length":
            "Character length of the top-level domain.",

        "suspicious_word_count":
            "Number of suspicious/phishing-related keywords."
    }

    # ======================================================================
    # DEFAULT VALUES
    # ======================================================================

    DEFAULT_FEATURE_VALUES: Dict[str, float] = {

        "url_length": 0.0,

        "domain_length": 0.0,

        "path_length": 0.0,

        "query_length": 0.0,

        "num_dots": 0.0,

        "num_hyphens": 0.0,

        "num_underscores": 0.0,

        "num_slashes": 0.0,

        "num_question_marks": 0.0,

        "num_equal_signs": 0.0,

        "num_at_symbols": 0.0,

        "num_percent_signs": 0.0,

        "num_digits": 0.0,

        "num_special_chars": 0.0,

        "url_entropy": 0.0,

        "domain_entropy": 0.0,

        "path_entropy": 0.0,

        "query_entropy": 0.0,

        "contains_ip": 0.0,

        "is_https": 0.0,

        "has_punycode": 0.0,

        "subdomain_count": 0.0,

        "tld_length": 0.0,

        "suspicious_word_count": 0.0
    }

    # ======================================================================
    # ALIASES
    # ======================================================================

    FEATURE_ALIASES: Dict[str, str] = {

        # --------------------------------------------------------------
        # HTTPS
        # --------------------------------------------------------------

        "https":
            "is_https",

        "uses_https":
            "is_https",

        "secure":
            "is_https",

        # --------------------------------------------------------------
        # IP
        # --------------------------------------------------------------

        "ip_address":
            "contains_ip",

        "has_ip":
            "contains_ip",

        "is_ip":
            "contains_ip",

        "uses_ip":
            "contains_ip",

        # --------------------------------------------------------------
        # URL entropy
        # --------------------------------------------------------------

        "entropy":
            "url_entropy",

        "global_entropy":
            "url_entropy",

        "url_randomness":
            "url_entropy",

        # --------------------------------------------------------------
        # Domain entropy
        # --------------------------------------------------------------

        "domain_randomness":
            "domain_entropy",

        "hostname_entropy":
            "domain_entropy",

        "host_entropy":
            "domain_entropy",

        # --------------------------------------------------------------
        # Path entropy
        # --------------------------------------------------------------

        "path_randomness":
            "path_entropy",

        "pathname_entropy":
            "path_entropy",

        # --------------------------------------------------------------
        # Query entropy
        # --------------------------------------------------------------

        "query_randomness":
            "query_entropy",

        "parameter_entropy":
            "query_entropy",

        "querystring_entropy":
            "query_entropy",

        # --------------------------------------------------------------
        # Special characters
        # --------------------------------------------------------------

        "special_characters":
            "num_special_chars",

        "special_chars_count":
            "num_special_chars",

        "special_character_count":
            "num_special_chars",

        # --------------------------------------------------------------
        # Suspicious words
        # --------------------------------------------------------------

        "suspicious_keywords":
            "suspicious_word_count",

        "suspicious_words_count":
            "suspicious_word_count",

        "num_suspicious_words":
            "suspicious_word_count",

        # --------------------------------------------------------------
        # @ symbol
        # --------------------------------------------------------------

        "has_at_symbol":
            "num_at_symbols",

        "@_symbol":
            "num_at_symbols",

        # --------------------------------------------------------------
        # URL length
        # --------------------------------------------------------------

        "length_url":
            "url_length",

        "url_len":
            "url_length",

        # --------------------------------------------------------------
        # Domain length
        # --------------------------------------------------------------

        "length_hostname":
            "domain_length",

        "hostname_length":
            "domain_length",

        "domain_len":
            "domain_length",

        # --------------------------------------------------------------
        # Path length
        # --------------------------------------------------------------

        "length_path":
            "path_length",

        "path_len":
            "path_length",

        # --------------------------------------------------------------
        # Query length
        # --------------------------------------------------------------

        "length_query":
            "query_length",

        "query_len":
            "query_length",

        # --------------------------------------------------------------
        # Dots
        # --------------------------------------------------------------

        "qty_dot_url":
            "num_dots",

        "dot_count":
            "num_dots",

        "dots":
            "num_dots",

        # --------------------------------------------------------------
        # Hyphens
        # --------------------------------------------------------------

        "qty_hyphen_url":
            "num_hyphens",

        "hyphen_count":
            "num_hyphens",

        "hyphens":
            "num_hyphens",

        # --------------------------------------------------------------
        # Underscores
        # --------------------------------------------------------------

        "qty_underline_url":
            "num_underscores",

        "underscore_count":
            "num_underscores",

        "underscores":
            "num_underscores",

        # --------------------------------------------------------------
        # Slashes
        # --------------------------------------------------------------

        "qty_slash_url":
            "num_slashes",

        "slash_count":
            "num_slashes",

        "slashes":
            "num_slashes",

        # --------------------------------------------------------------
        # Question marks
        # --------------------------------------------------------------

        "qty_questionmark_url":
            "num_question_marks",

        "question_mark_count":
            "num_question_marks",

        # --------------------------------------------------------------
        # Equal signs
        # --------------------------------------------------------------

        "qty_equal_url":
            "num_equal_signs",

        "equal_sign_count":
            "num_equal_signs",

        # --------------------------------------------------------------
        # @ symbols
        # --------------------------------------------------------------

        "qty_at_url":
            "num_at_symbols",

        "at_count":
            "num_at_symbols",

        # --------------------------------------------------------------
        # Percent
        # --------------------------------------------------------------

        "qty_percent_url":
            "num_percent_signs",

        "percent_count":
            "num_percent_signs",

        # --------------------------------------------------------------
        # Numeric
        # --------------------------------------------------------------

        "qty_numeric_url":
            "num_digits",

        "numeric_count":
            "num_digits",

        "digit_count":
            "num_digits",

        # --------------------------------------------------------------
        # Subdomains
        # --------------------------------------------------------------

        "subdomains":
            "subdomain_count",

        "subdomain_number":
            "subdomain_count",

        # --------------------------------------------------------------
        # TLD
        # --------------------------------------------------------------

        "tld_len":
            "tld_length"
    }

    # ======================================================================
    # PUBLIC METHODS
    # ======================================================================

    @classmethod
    def get_schema(
        cls
    ) -> List[str]:
        """
        Return a copy of the ordered feature schema.
        """

        return cls.EXPECTED_FEATURE_ORDER.copy()

    @classmethod
    def get_feature_names(
        cls
    ) -> List[str]:
        """
        Compatibility alias for modules that expect get_feature_names().
        """

        return cls.get_schema()

    @classmethod
    def get_descriptions(
        cls
    ) -> Dict[str, str]:
        """
        Return human-readable feature descriptions.
        """

        return cls.FEATURE_METADATA.copy()

    @classmethod
    def get_default_features(
        cls
    ) -> Dict[str, float]:
        """
        Return a complete default feature dictionary.
        """

        return cls.DEFAULT_FEATURE_VALUES.copy()

    @classmethod
    def get_feature_count(
        cls
    ) -> int:
        """
        Return number of model features.
        """

        return len(
            cls.EXPECTED_FEATURE_ORDER
        )

    @classmethod
    def has_feature(
        cls,
        feature_name: str
    ) -> bool:
        """
        Check whether a feature belongs to the schema.
        """

        return (
            feature_name
            in
            cls.EXPECTED_FEATURE_ORDER
        )

    @classmethod
    def normalize_feature_name(
        cls,
        feature_name: str
    ) -> str:
        """
        Convert an alias into its canonical feature name.
        """

        if not isinstance(
            feature_name,
            str
        ):
            return feature_name

        return cls.FEATURE_ALIASES.get(
            feature_name,
            feature_name
        )

    # ======================================================================
    # VALUE NORMALIZATION
    # ======================================================================

    @classmethod
    def _normalize_value(
        cls,
        feature_name: str,
        value: Any
    ) -> float:
        """
        Convert a raw feature value into a safe numerical value.

        Handles:

            bool
            int
            float
            numeric strings
            lists
            None
            NaN
            infinity
            invalid strings
        """

        if value is None:

            return cls.DEFAULT_FEATURE_VALUES.get(
                feature_name,
                0.0
            )

        # --------------------------------------------------------------
        # Boolean
        # --------------------------------------------------------------

        if isinstance(
            value,
            bool
        ):

            return (
                1.0
                if value
                else 0.0
            )

        # --------------------------------------------------------------
        # Lists
        # --------------------------------------------------------------

        if isinstance(
            value,
            (list, tuple, set)
        ):

            return float(
                len(value)
            )

        # --------------------------------------------------------------
        # Numeric conversion
        # --------------------------------------------------------------

        try:

            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            logger.warning(
                "Invalid value for URL feature '%s': %r. "
                "Using default value.",
                feature_name,
                value
            )

            return cls.DEFAULT_FEATURE_VALUES.get(
                feature_name,
                0.0
            )

        # --------------------------------------------------------------
        # NaN / Infinity protection
        # --------------------------------------------------------------

        if not math.isfinite(
            numeric_value
        ):

            logger.warning(
                "Non-finite value for URL feature '%s': %r. "
                "Using default value.",
                feature_name,
                value
            )

            return cls.DEFAULT_FEATURE_VALUES.get(
                feature_name,
                0.0
            )

        # --------------------------------------------------------------
        # Binary feature normalization
        # --------------------------------------------------------------

        if feature_name in {
            "contains_ip",
            "is_https",
            "has_punycode"
        }:

            return (
                1.0
                if numeric_value != 0.0
                else 0.0
            )

        # --------------------------------------------------------------
        # Counts/lengths cannot be negative
        # --------------------------------------------------------------

        non_negative_features = {

            "url_length",
            "domain_length",
            "path_length",
            "query_length",

            "num_dots",
            "num_hyphens",
            "num_underscores",
            "num_slashes",
            "num_question_marks",
            "num_equal_signs",
            "num_at_symbols",
            "num_percent_signs",
            "num_digits",
            "num_special_chars",

            "subdomain_count",
            "tld_length",
            "suspicious_word_count"
        }

        if feature_name in non_negative_features:

            numeric_value = max(
                0.0,
                numeric_value
            )

        # --------------------------------------------------------------
        # Entropy cannot be negative
        # --------------------------------------------------------------

        if feature_name in {
            "url_entropy",
            "domain_entropy",
            "path_entropy",
            "query_entropy"
        }:

            numeric_value = max(
                0.0,
                numeric_value
            )

        return numeric_value

    # ======================================================================
    # MAIN ALIGNMENT METHOD
    # ======================================================================

    @classmethod
    def align_and_validate(
        cls,
        raw_features: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Normalize, validate and strictly align URL features.

        Unknown features are intentionally ignored.

        Missing features receive safe defaults.

        Returns:

            pandas.DataFrame

        with EXACTLY:

            EXPECTED_FEATURE_ORDER

        columns and exactly one row.
        """

        if not isinstance(
            raw_features,
            dict
        ):

            logger.warning(
                "URLFeatureSchema received non-dictionary input. "
                "Using defaults."
            )

            raw_features = {}

        # ================================================================
        # NORMALIZE KEYS
        # ================================================================

        normalized: Dict[str, Any] = {}

        for key, value in raw_features.items():

            if not isinstance(
                key,
                str
            ):

                continue

            canonical_key = (
                cls.FEATURE_ALIASES.get(
                    key,
                    key
                )
            )

            # ----------------------------------------------------------
            # Do not allow an alias to overwrite an explicitly supplied
            # canonical feature.
            # ----------------------------------------------------------

            if (
                canonical_key
                in normalized
                and
                key != canonical_key
            ):

                continue

            normalized[
                canonical_key
            ] = value

        # ================================================================
        # BUILD STRICT FEATURE VECTOR
        # ================================================================

        row: Dict[str, float] = {}

        for feature_name in (
            cls.EXPECTED_FEATURE_ORDER
        ):

            raw_value = normalized.get(
                feature_name,
                cls.DEFAULT_FEATURE_VALUES.get(
                    feature_name,
                    0.0
                )
            )

            row[
                feature_name
            ] = cls._normalize_value(
                feature_name,
                raw_value
            )

        # ================================================================
        # DATAFRAME
        # ================================================================

        df = pd.DataFrame(
            [row],
            columns=cls.EXPECTED_FEATURE_ORDER
        )

        # ================================================================
        # FINAL SAFETY CHECKS
        # ================================================================

        if list(
            df.columns
        ) != cls.EXPECTED_FEATURE_ORDER:

            raise RuntimeError(
                "URL feature schema alignment failure. "
                "Column order does not match EXPECTED_FEATURE_ORDER."
            )

        if len(df) != 1:

            raise RuntimeError(
                "URL feature schema must produce exactly one row."
            )

        # Ensure all model features are numerical.

        for feature_name in (
            cls.EXPECTED_FEATURE_ORDER
        ):

            df[
                feature_name
            ] = pd.to_numeric(
                df[
                    feature_name
                ],
                errors="coerce"
            ).fillna(
                cls.DEFAULT_FEATURE_VALUES.get(
                    feature_name,
                    0.0
                )
            )

        # ================================================================
        # METADATA
        # ================================================================

        tld_value = raw_features.get(
            "tld",
            raw_features.get(
                "tld_type",
                "unknown"
            )
        )

        df.attrs[
            "tld_type"
        ] = str(
            tld_value
        )

        df.attrs[
            "schema_version"
        ] = cls.SCHEMA_VERSION

        df.attrs[
            "feature_count"
        ] = cls.get_feature_count()

        # Record whether entropy was supplied by the extractor.

        df.attrs[
            "entropy_features_present"
        ] = {
            "url_entropy":
                "url_entropy"
                in normalized,

            "domain_entropy":
                "domain_entropy"
                in normalized,

            "path_entropy":
                "path_entropy"
                in normalized,

            "query_entropy":
                "query_entropy"
                in normalized
        }

        return df

    # ======================================================================
    # BATCH ALIGNMENT
    # ======================================================================

    @classmethod
    def align_dataframe(
        cls,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align a training/inference DataFrame to the exact schema.

        This method is particularly important for trainer.py.

        It ensures:

            - missing columns are created
            - extra columns are removed
            - column order is deterministic
            - invalid values become safe numerical values
        """

        if not isinstance(
            dataframe,
            pd.DataFrame
        ):

            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        df = dataframe.copy()

        # ================================================================
        # RENAME ALIASES
        # ================================================================

        rename_map = {}

        for column in df.columns:

            canonical = (
                cls.FEATURE_ALIASES.get(
                    column,
                    column
                )
            )

            if canonical != column:

                rename_map[
                    column
                ] = canonical

        if rename_map:

            df = df.rename(
                columns=rename_map
            )

        # ================================================================
        # ADD MISSING FEATURES
        # ================================================================

        for feature_name in (
            cls.EXPECTED_FEATURE_ORDER
        ):

            if feature_name not in df.columns:

                df[
                    feature_name
                ] = cls.DEFAULT_FEATURE_VALUES.get(
                    feature_name,
                    0.0
                )

        # ================================================================
        # KEEP ONLY MODEL FEATURES
        # ================================================================

        df = df[
            cls.EXPECTED_FEATURE_ORDER
        ].copy()

        # ================================================================
        # NORMALIZE EVERY FEATURE
        # ================================================================

        for feature_name in (
            cls.EXPECTED_FEATURE_ORDER
        ):

            df[
                feature_name
            ] = df[
                feature_name
            ].apply(
                lambda value:
                    cls._normalize_value(
                        feature_name,
                        value
                    )
            )

        # ================================================================
        # FINAL NUMERICAL SAFETY
        # ================================================================

        df = df.replace(
            [float("inf"), float("-inf")],
            0.0
        )

        df = df.fillna(
            0.0
        )

        return df

    # ======================================================================
    # SCHEMA VALIDATION
    # ======================================================================

    @classmethod
    def validate_dataframe_schema(
        cls,
        dataframe: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate whether a DataFrame exactly matches the expected
        URL feature schema.

        Returns a structured validation report.
        """

        if not isinstance(
            dataframe,
            pd.DataFrame
        ):

            return {

                "valid":
                    False,

                "reason":
                    "Input is not a pandas DataFrame."
            }

        expected = (
            cls.EXPECTED_FEATURE_ORDER
        )

        actual = list(
            dataframe.columns
        )

        missing = [
            feature
            for feature
            in expected
            if feature not in actual
        ]

        extra = [
            feature
            for feature
            in actual
            if feature not in expected
        ]

        correct_order = (
            actual == expected
        )

        return {

            "valid":
                (
                    not missing
                    and
                    not extra
                    and
                    correct_order
                ),

            "schema_version":
                cls.SCHEMA_VERSION,

            "expected_feature_count":
                len(expected),

            "actual_feature_count":
                len(actual),

            "missing_features":
                missing,

            "extra_features":
                extra,

            "correct_order":
                correct_order,

            "expected_order":
                expected,

            "actual_order":
                actual
        }