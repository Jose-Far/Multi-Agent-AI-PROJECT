"""
SSL/TLS Security AI Agent - Feature Schema
==========================================

Single source of truth for SSL/TLS security features.

This module defines:

    1. Feature names
    2. Feature ordering
    3. Feature data types
    4. Default values
    5. Feature descriptions
    6. Feature ranges
    7. Feature aliases
    8. Type normalization
    9. Value sanitization
    10. Feature validation
    11. DataFrame-compatible output
    12. Schema metadata

The same schema MUST be used by:

    SSL Feature Extraction
             ↓
    SSLFeatureSchema
             ↓
    preprocessing.py
             ↓
    trainer.py
             ↓
    predictor.py
             ↓
    risk_score.py
             ↓
    explain.py
             ↓
    ssl_agent.py

IMPORTANT
---------
Do not change feature names or feature order independently in another
SSL Agent file.

If a feature is added, removed, or renamed, this file should be updated
first and all downstream components should use this schema.
"""

import logging

import math

from typing import (
    Dict,
    Any,
    List,
    Tuple,
    Optional,
    Union
)


logger = logging.getLogger(__name__)


class SSLFeatureSchema:
    """
    Central schema definition for the SSL/TLS Security AI Agent.

    The schema contains 17 numerical/boolean security features.

    Feature groups:

        Certificate Presence
        Certificate Validity
        Certificate Lifetime
        Certificate Trust
        Certificate Authority Characteristics
        TLS Protocol Security
        Cipher Security
        Cryptographic Strength
        Connection Security
    """

    # ======================================================================
    # SCHEMA METADATA
    # ======================================================================

    SCHEMA_NAME = (
        "SSL_TLS_Security_Feature_Schema"
    )

    SCHEMA_VERSION = "1.0.0"

    AGENT_NAME = (
        "SSL_AI_Agent"
    )

    # Total expected features
    FEATURE_COUNT = 17

    # ======================================================================
    # FEATURE ORDER
    # ======================================================================

    FEATURE_NAMES: List[str] = [

        # --------------------------------------------------------------
        # Certificate Presence / Validity
        # --------------------------------------------------------------

        "has_ssl",

        "is_expired",

        # --------------------------------------------------------------
        # Certificate Lifetime
        # --------------------------------------------------------------

        "cert_age_days",

        "days_until_expiry",

        "total_lifespan_days",

        # --------------------------------------------------------------
        # Certificate Type
        # --------------------------------------------------------------

        "is_self_signed",

        "is_recently_issued",

        "is_short_lived",

        # --------------------------------------------------------------
        # Certificate Authority
        # --------------------------------------------------------------

        "is_trusted_ca",

        "is_free_automated_ca",

        "is_suspicious_ca",

        "issuer_trust_score",

        # --------------------------------------------------------------
        # TLS / Cryptographic Security
        # --------------------------------------------------------------

        "is_weak_protocol",

        "is_vulnerable_cipher",

        "cipher_strength_bits",

        "crypto_health_score",

        # --------------------------------------------------------------
        # Final Connection State
        # --------------------------------------------------------------

        "is_secure_connection"
    ]

    # ======================================================================
    # DEFAULT VALUES
    # ======================================================================

    DEFAULT_FEATURES: Dict[str, Any] = {

        "has_ssl":
            False,

        "is_expired":
            True,

        "cert_age_days":
            0,

        "days_until_expiry":
            0,

        "total_lifespan_days":
            0,

        "is_self_signed":
            False,

        "is_recently_issued":
            False,

        "is_short_lived":
            False,

        "is_trusted_ca":
            False,

        "is_free_automated_ca":
            False,

        "is_suspicious_ca":
            True,

        "issuer_trust_score":
            0.0,

        "is_weak_protocol":
            True,

        "is_vulnerable_cipher":
            True,

        "cipher_strength_bits":
            0,

        "crypto_health_score":
            0.0,

        "is_secure_connection":
            False
    }

    # ======================================================================
    # FEATURE DESCRIPTIONS
    # ======================================================================

    FEATURE_DESCRIPTIONS: Dict[str, str] = {

        "has_ssl":
            (
                "Indicates whether the target website successfully "
                "establishes an SSL/TLS connection."
            ),

        "is_expired":
            (
                "Indicates whether the presented TLS certificate "
                "has passed its expiration date."
            ),

        "cert_age_days":
            (
                "Number of days elapsed since the certificate "
                "was issued."
            ),

        "days_until_expiry":
            (
                "Number of days remaining before the certificate "
                "expires. Zero or negative values indicate expiry."
            ),

        "total_lifespan_days":
            (
                "Total validity period of the certificate in days."
            ),

        "is_self_signed":
            (
                "Indicates whether the certificate is self-signed "
                "rather than issued by a recognized certificate authority."
            ),

        "is_recently_issued":
            (
                "Indicates whether the certificate was issued "
                "recently according to the configured recency threshold."
            ),

        "is_short_lived":
            (
                "Indicates whether the certificate has an unusually "
                "short total validity period."
            ),

        "is_trusted_ca":
            (
                "Indicates whether the certificate issuer is considered "
                "a trusted certificate authority."
            ),

        "is_free_automated_ca":
            (
                "Indicates whether the certificate appears to originate "
                "from a free or automated certificate authority."
            ),

        "is_suspicious_ca":
            (
                "Indicates whether the certificate issuer exhibits "
                "characteristics associated with a suspicious or "
                "untrusted certificate authority."
            ),

        "issuer_trust_score":
            (
                "Normalized trust score assigned to the certificate issuer "
                "on a 0-100 scale."
            ),

        "is_weak_protocol":
            (
                "Indicates whether the negotiated TLS protocol is weak, "
                "obsolete, or otherwise insecure."
            ),

        "is_vulnerable_cipher":
            (
                "Indicates whether the negotiated cipher suite is known "
                "to provide weak or vulnerable cryptographic protection."
            ),

        "cipher_strength_bits":
            (
                "Approximate cryptographic key strength of the negotiated "
                "cipher suite, represented in bits."
            ),

        "crypto_health_score":
            (
                "Overall normalized cryptographic security score "
                "on a 0-100 scale."
            ),

        "is_secure_connection":
            (
                "Overall indicator that the connection satisfies the "
                "minimum configured SSL/TLS security requirements."
            )
    }

    # ======================================================================
    # FEATURE TYPES
    # ======================================================================

    FEATURE_TYPES: Dict[str, type] = {

        "has_ssl":
            bool,

        "is_expired":
            bool,

        "cert_age_days":
            int,

        "days_until_expiry":
            int,

        "total_lifespan_days":
            int,

        "is_self_signed":
            bool,

        "is_recently_issued":
            bool,

        "is_short_lived":
            bool,

        "is_trusted_ca":
            bool,

        "is_free_automated_ca":
            bool,

        "is_suspicious_ca":
            bool,

        "issuer_trust_score":
            float,

        "is_weak_protocol":
            bool,

        "is_vulnerable_cipher":
            bool,

        "cipher_strength_bits":
            int,

        "crypto_health_score":
            float,

        "is_secure_connection":
            bool
    }

    # ======================================================================
    # FEATURE RANGES
    # ======================================================================

    FEATURE_RANGES: Dict[
        str,
        Tuple[Optional[float], Optional[float]]
    ] = {

        # Boolean values are represented internally as 0/1
        "has_ssl":
            (0, 1),

        "is_expired":
            (0, 1),

        # Certificate age
        "cert_age_days":
            (0, 100000),

        # Remaining validity can theoretically be negative,
        # but normalized representation uses 0 for expired values.
        "days_until_expiry":
            (0, 100000),

        "total_lifespan_days":
            (0, 100000),

        "is_self_signed":
            (0, 1),

        "is_recently_issued":
            (0, 1),

        "is_short_lived":
            (0, 1),

        "is_trusted_ca":
            (0, 1),

        "is_free_automated_ca":
            (0, 1),

        "is_suspicious_ca":
            (0, 1),

        # Trust score
        "issuer_trust_score":
            (0.0, 100.0),

        "is_weak_protocol":
            (0, 1),

        "is_vulnerable_cipher":
            (0, 1),

        # Common key strengths
        "cipher_strength_bits":
            (0, 8192),

        # Overall crypto health
        "crypto_health_score":
            (0.0, 100.0),

        "is_secure_connection":
            (0, 1)
    }

    # ======================================================================
    # FEATURE ALIASES
    # ======================================================================

    """
    Supports compatibility with different extractor naming conventions.

    Example:

        https
            ↓
        has_ssl

        certificate_expired
            ↓
        is_expired

        certificate_age_days
            ↓
        cert_age_days
    """

    FEATURE_ALIASES: Dict[str, str] = {

        # --------------------------------------------------------------
        # SSL / HTTPS
        # --------------------------------------------------------------

        "has_https":
            "has_ssl",

        "https":
            "has_ssl",

        "uses_https":
            "has_ssl",

        "ssl_enabled":
            "has_ssl",

        "tls_enabled":
            "has_ssl",

        "ssl":
            "has_ssl",

        # --------------------------------------------------------------
        # Expiry
        # --------------------------------------------------------------

        "certificate_expired":
            "is_expired",

        "cert_expired":
            "is_expired",

        "expired":
            "is_expired",

        "certificate_is_expired":
            "is_expired",

        # --------------------------------------------------------------
        # Certificate age
        # --------------------------------------------------------------

        "certificate_age_days":
            "cert_age_days",

        "certificate_age":
            "cert_age_days",

        "cert_age":
            "cert_age_days",

        # --------------------------------------------------------------
        # Remaining validity
        # --------------------------------------------------------------

        "certificate_remaining_days":
            "days_until_expiry",

        "remaining_days":
            "days_until_expiry",

        "days_remaining":
            "days_until_expiry",

        "expiry_days_remaining":
            "days_until_expiry",

        # --------------------------------------------------------------
        # Total lifespan
        # --------------------------------------------------------------

        "certificate_lifespan_days":
            "total_lifespan_days",

        "cert_lifespan_days":
            "total_lifespan_days",

        "certificate_validity_days":
            "total_lifespan_days",

        "validity_period_days":
            "total_lifespan_days",

        # --------------------------------------------------------------
        # Self signed
        # --------------------------------------------------------------

        "self_signed":
            "is_self_signed",

        "certificate_self_signed":
            "is_self_signed",

        "cert_self_signed":
            "is_self_signed",

        # --------------------------------------------------------------
        # Recently issued
        # --------------------------------------------------------------

        "recently_issued":
            "is_recently_issued",

        "certificate_recently_issued":
            "is_recently_issued",

        "recent_certificate":
            "is_recently_issued",

        # --------------------------------------------------------------
        # Short lived
        # --------------------------------------------------------------

        "short_lived":
            "is_short_lived",

        "certificate_short_lived":
            "is_short_lived",

        # --------------------------------------------------------------
        # Trusted CA
        # --------------------------------------------------------------

        "trusted_ca":
            "is_trusted_ca",

        "certificate_trusted":
            "is_trusted_ca",

        "ca_trusted":
            "is_trusted_ca",

        "issuer_trusted":
            "is_trusted_ca",

        # --------------------------------------------------------------
        # Free automated CA
        # --------------------------------------------------------------

        "free_automated_ca":
            "is_free_automated_ca",

        "automated_ca":
            "is_free_automated_ca",

        "free_ca":
            "is_free_automated_ca",

        # --------------------------------------------------------------
        # Suspicious CA
        # --------------------------------------------------------------

        "suspicious_ca":
            "is_suspicious_ca",

        "ca_suspicious":
            "is_suspicious_ca",

        "untrusted_ca":
            "is_suspicious_ca",

        # --------------------------------------------------------------
        # Issuer trust
        # --------------------------------------------------------------

        "ca_trust_score":
            "issuer_trust_score",

        "issuer_score":
            "issuer_trust_score",

        "certificate_trust_score":
            "issuer_trust_score",

        "trust_score":
            "issuer_trust_score",

        # --------------------------------------------------------------
        # Weak protocol
        # --------------------------------------------------------------

        "weak_protocol":
            "is_weak_protocol",

        "weak_tls":
            "is_weak_protocol",

        "weak_tls_version":
            "is_weak_protocol",

        "obsolete_tls":
            "is_weak_protocol",

        "insecure_protocol":
            "is_weak_protocol",

        # --------------------------------------------------------------
        # Vulnerable cipher
        # --------------------------------------------------------------

        "vulnerable_cipher":
            "is_vulnerable_cipher",

        "weak_cipher":
            "is_vulnerable_cipher",

        "insecure_cipher":
            "is_vulnerable_cipher",

        "cipher_vulnerable":
            "is_vulnerable_cipher",

        # --------------------------------------------------------------
        # Cipher strength
        # --------------------------------------------------------------

        "cipher_bits":
            "cipher_strength_bits",

        "key_strength_bits":
            "cipher_strength_bits",

        "encryption_strength_bits":
            "cipher_strength_bits",

        # --------------------------------------------------------------
        # Crypto health
        # --------------------------------------------------------------

        "cryptographic_health_score":
            "crypto_health_score",

        "crypto_score":
            "crypto_health_score",

        "cryptography_score":
            "crypto_health_score",

        # --------------------------------------------------------------
        # Secure connection
        # --------------------------------------------------------------

        "secure_connection":
            "is_secure_connection",

        "connection_secure":
            "is_secure_connection",

        "tls_secure":
            "is_secure_connection",

        "ssl_secure":
            "is_secure_connection"
    }

    # ======================================================================
    # BOOLEAN FEATURE SET
    # ======================================================================

    BOOLEAN_FEATURES = {

        "has_ssl",

        "is_expired",

        "is_self_signed",

        "is_recently_issued",

        "is_short_lived",

        "is_trusted_ca",

        "is_free_automated_ca",

        "is_suspicious_ca",

        "is_weak_protocol",

        "is_vulnerable_cipher",

        "is_secure_connection"
    }

    # ======================================================================
    # INTEGER FEATURE SET
    # ======================================================================

    INTEGER_FEATURES = {

        "cert_age_days",

        "days_until_expiry",

        "total_lifespan_days",

        "cipher_strength_bits"
    }

    # ======================================================================
    # FLOAT FEATURE SET
    # ======================================================================

    FLOAT_FEATURES = {

        "issuer_trust_score",

        "crypto_health_score"
    }

    # ======================================================================
    # PUBLIC API: GET FEATURE NAMES
    # ======================================================================

    @classmethod
    def get_feature_names(
        cls
    ) -> List[str]:
        """
        Return the ordered list of SSL feature names.

        This order MUST be used during:

            Training
            Prediction
            Explainability

        Returns:
            List[str]
                Ordered feature names.
        """

        return cls.FEATURE_NAMES.copy()

    # ======================================================================
    # PUBLIC API: FEATURE COUNT
    # ======================================================================

    @classmethod
    def get_feature_count(
        cls
    ) -> int:
        """
        Return number of features.
        """

        return len(
            cls.FEATURE_NAMES
        )

    # ======================================================================
    # PUBLIC API: DEFAULT FEATURES
    # ======================================================================

    @classmethod
    def get_default_features(
        cls
    ) -> Dict[str, Any]:
        """
        Return a fresh copy of the default SSL feature vector.

        A fresh dictionary is returned every time to prevent accidental
        mutation of the class-level defaults.
        """

        return cls.DEFAULT_FEATURES.copy()

    # ======================================================================
    # PUBLIC API: DESCRIPTIONS
    # ======================================================================

    @classmethod
    def get_feature_descriptions(
        cls
    ) -> Dict[str, str]:
        """
        Return descriptions for all features.
        """

        return cls.FEATURE_DESCRIPTIONS.copy()

    # ======================================================================
    # PUBLIC API: TYPES
    # ======================================================================

    @classmethod
    def get_feature_types(
        cls
    ) -> Dict[str, type]:
        """
        Return expected Python types for each feature.
        """

        return cls.FEATURE_TYPES.copy()

    # ======================================================================
    # PUBLIC API: RANGES
    # ======================================================================

    @classmethod
    def get_feature_ranges(
        cls
    ) -> Dict[
        str,
        Tuple[
            Optional[float],
            Optional[float]
        ]
    ]:
        """
        Return acceptable numeric ranges for features.
        """

        return cls.FEATURE_RANGES.copy()

    # ======================================================================
    # BOOLEAN NORMALIZATION
    # ======================================================================

    @classmethod
    def _normalize_boolean(
        cls,
        value: Any,
        default: bool = False
    ) -> bool:
        """
        Safely normalize arbitrary input into a boolean.

        Correctly handles:

            True
            False
            "true"
            "false"
            "yes"
            "no"
            "1"
            "0"
            1
            0
            None

        IMPORTANT:
            bool("false") is True in Python.

            Therefore we explicitly parse string values.
        """

        if value is None:

            return default

        if isinstance(
            value,
            bool
        ):

            return value

        if isinstance(
            value,
            (int, float)
        ):

            if math.isnan(
                float(value)
            ):

                return default

            return value != 0

        if isinstance(
            value,
            str
        ):

            normalized = (
                value
                .strip()
                .lower()
            )

            if normalized in {
                "true",
                "1",
                "yes",
                "y",
                "on",
                "enabled",
                "valid",
                "secure"
            }:

                return True

            if normalized in {
                "false",
                "0",
                "no",
                "n",
                "off",
                "disabled",
                "invalid",
                "insecure",
                "none",
                "null",
                ""
            }:

                return False

        logger.debug(
            "Unable to normalize boolean value '%s'. "
            "Using default=%s.",
            value,
            default
        )

        return default

    # ======================================================================
    # INTEGER NORMALIZATION
    # ======================================================================

    @classmethod
    def _normalize_integer(
        cls,
        value: Any,
        default: int = 0
    ) -> int:
        """
        Safely normalize a value to integer.
        """

        if value is None:

            return default

        try:

            numeric = float(
                value
            )

            if not math.isfinite(
                numeric
            ):

                return default

            return int(
                round(
                    numeric
                )
            )

        except (
            TypeError,
            ValueError,
            OverflowError
        ):

            logger.debug(
                "Unable to normalize integer value '%s'. "
                "Using default=%s.",
                value,
                default
            )

            return default

    # ======================================================================
    # FLOAT NORMALIZATION
    # ======================================================================

    @classmethod
    def _normalize_float(
        cls,
        value: Any,
        default: float = 0.0
    ) -> float:
        """
        Safely normalize a value to float.
        """

        if value is None:

            return default

        try:

            numeric = float(
                value
            )

            if not math.isfinite(
                numeric
            ):

                return default

            return numeric

        except (
            TypeError,
            ValueError,
            OverflowError
        ):

            logger.debug(
                "Unable to normalize float value '%s'. "
                "Using default=%s.",
                value,
                default
            )

            return default

    # ======================================================================
    # RANGE CLAMPING
    # ======================================================================

    @classmethod
    def _clamp_feature_value(
        cls,
        feature_name: str,
        value: Union[int, float]
    ) -> Union[int, float]:
        """
        Clamp numeric features to their defined safe ranges.

        This prevents extreme telemetry values from producing
        invalid model inputs.
        """

        if feature_name not in cls.FEATURE_RANGES:

            return value

        minimum, maximum = (
            cls.FEATURE_RANGES[
                feature_name
            ]
        )

        if minimum is not None:

            value = max(
                minimum,
                value
            )

        if maximum is not None:

            value = min(
                maximum,
                value
            )

        return value

    # ======================================================================
    # ALIAS NORMALIZATION
    # ======================================================================

    @classmethod
    def normalize_feature_names(
        cls,
        raw_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert alternate feature names into canonical schema names.

        Unknown fields are ignored.

        Canonical fields always have priority over aliases.
        """

        if not isinstance(
            raw_features,
            dict
        ):

            return {}

        normalized = {}

        # ------------------------------------------------------------------
        # First pass:
        # canonical names
        # ------------------------------------------------------------------

        for feature_name in cls.FEATURE_NAMES:

            if feature_name in raw_features:

                normalized[
                    feature_name
                ] = raw_features[
                    feature_name
                ]

        # ------------------------------------------------------------------
        # Second pass:
        # aliases
        #
        # Do not overwrite canonical values.
        # ------------------------------------------------------------------

        for raw_name, value in raw_features.items():

            canonical_name = (
                cls.FEATURE_ALIASES.get(
                    raw_name
                )
            )

            if canonical_name is None:

                continue

            if canonical_name in normalized:

                continue

            normalized[
                canonical_name
            ] = value

        return normalized

    # ======================================================================
    # NORMALIZE SINGLE FEATURE
    # ======================================================================

    @classmethod
    def _normalize_single_feature(
        cls,
        feature_name: str,
        value: Any
    ) -> Any:
        """
        Normalize one feature according to its expected type.
        """

        default_value = (
            cls.DEFAULT_FEATURES[
                feature_name
            ]
        )

        # ------------------------------------------------------------------
        # Boolean
        # ------------------------------------------------------------------

        if feature_name in cls.BOOLEAN_FEATURES:

            normalized_value = (
                cls._normalize_boolean(
                    value,
                    bool(
                        default_value
                    )
                )
            )

            return normalized_value

        # ------------------------------------------------------------------
        # Integer
        # ------------------------------------------------------------------

        if feature_name in cls.INTEGER_FEATURES:

            normalized_value = (
                cls._normalize_integer(
                    value,
                    int(
                        default_value
                    )
                )
            )

            normalized_value = (
                cls._clamp_feature_value(
                    feature_name,
                    normalized_value
                )
            )

            return int(
                normalized_value
            )

        # ------------------------------------------------------------------
        # Float
        # ------------------------------------------------------------------

        if feature_name in cls.FLOAT_FEATURES:

            normalized_value = (
                cls._normalize_float(
                    value,
                    float(
                        default_value
                    )
                )
            )

            normalized_value = (
                cls._clamp_feature_value(
                    feature_name,
                    normalized_value
                )
            )

            return float(
                normalized_value
            )

        # ------------------------------------------------------------------
        # Unknown type
        # ------------------------------------------------------------------

        logger.warning(
            "Feature '%s' has no registered type. "
            "Using default.",
            feature_name
        )

        return default_value

    # ======================================================================
    # COMPLETE VALIDATION / NORMALIZATION
    # ======================================================================

    @classmethod
    def validate_and_normalize(
        cls,
        raw_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and normalize a raw SSL feature dictionary.

        Behaviour:

            1. Reject invalid top-level input.
            2. Normalize aliases.
            3. Add missing features.
            4. Convert types.
            5. Clamp numerical values.
            6. Drop unknown fields.
            7. Return features in deterministic order.

        Args:
            raw_features:
                Raw feature dictionary generated by SSL extraction.

        Returns:
            Dict[str, Any]:
                Complete canonical SSL feature vector.
        """

        defaults = (
            cls.get_default_features()
        )

        # ------------------------------------------------------------------
        # Invalid input
        # ------------------------------------------------------------------

        if not isinstance(
            raw_features,
            dict
        ):

            logger.warning(
                "Invalid SSL feature payload type '%s'. "
                "Using complete default feature vector.",
                type(
                    raw_features
                ).__name__
            )

            return defaults

        # ------------------------------------------------------------------
        # Normalize aliases
        # ------------------------------------------------------------------

        normalized_input = (
            cls.normalize_feature_names(
                raw_features
            )
        )

        # ------------------------------------------------------------------
        # Build deterministic vector
        # ------------------------------------------------------------------

        normalized = {}

        for feature_name in cls.FEATURE_NAMES:

            raw_value = (
                normalized_input.get(
                    feature_name,
                    defaults[
                        feature_name
                    ]
                )
            )

            normalized[
                feature_name
            ] = cls._normalize_single_feature(
                feature_name,
                raw_value
            )

        # ------------------------------------------------------------------
        # Unknown field diagnostics
        # ------------------------------------------------------------------

        known_names = set(
            cls.FEATURE_NAMES
        )

        known_aliases = set(
            cls.FEATURE_ALIASES.keys()
        )

        ignored_fields = [

            key

            for key in raw_features.keys()

            if (
                key not in known_names
                and
                key not in known_aliases
            )
        ]

        if ignored_fields:

            logger.debug(
                "Ignoring unknown SSL feature fields: %s",
                ignored_fields
            )

        return normalized

    # ======================================================================
    # VALIDATE WITHOUT MODIFYING
    # ======================================================================

    @classmethod
    def validate(
        cls,
        features: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate whether a feature dictionary conforms to the schema.

        Unlike validate_and_normalize(), this method does not silently
        repair missing or malformed values.

        Returns:
            Tuple:

                (
                    is_valid,
                    list_of_errors
                )
        """

        errors = []

        # ------------------------------------------------------------------
        # Top-level type
        # ------------------------------------------------------------------

        if not isinstance(
            features,
            dict
        ):

            return (
                False,
                [
                    "Feature payload must be a dictionary."
                ]
            )

        # ------------------------------------------------------------------
        # Missing features
        # ------------------------------------------------------------------

        for feature_name in cls.FEATURE_NAMES:

            if feature_name not in features:

                errors.append(
                    (
                        f"Missing required feature: "
                        f"{feature_name}"
                    )
                )

        # ------------------------------------------------------------------
        # Type checks
        # ------------------------------------------------------------------

        for feature_name in cls.FEATURE_NAMES:

            if feature_name not in features:

                continue

            value = features[
                feature_name
            ]

            expected_type = cls.FEATURE_TYPES[
                feature_name
            ]

            # Boolean
            if expected_type is bool:

                if not isinstance(
                    value,
                    bool
                ):

                    errors.append(
                        (
                            f"{feature_name} must be bool; "
                            f"received {type(value).__name__}"
                        )
                    )

            # Integer
            elif expected_type is int:

                if (
                    isinstance(
                        value,
                        bool
                    )
                    or
                    not isinstance(
                        value,
                        int
                    )
                ):

                    errors.append(
                        (
                            f"{feature_name} must be int; "
                            f"received {type(value).__name__}"
                        )
                    )

            # Float
            elif expected_type is float:

                if (
                    isinstance(
                        value,
                        bool
                    )
                    or
                    not isinstance(
                        value,
                        (int, float)
                    )
                ):

                    errors.append(
                        (
                            f"{feature_name} must be numeric; "
                            f"received {type(value).__name__}"
                        )
                    )

        # ------------------------------------------------------------------
        # Range checks
        # ------------------------------------------------------------------

        for feature_name in cls.FEATURE_NAMES:

            if feature_name not in features:

                continue

            value = features[
                feature_name
            ]

            if feature_name in cls.BOOLEAN_FEATURES:

                numeric_value = (
                    1
                    if value
                    else 0
                )

            elif isinstance(
                value,
                (int, float)
            ):

                numeric_value = float(
                    value
                )

            else:

                continue

            minimum, maximum = (
                cls.FEATURE_RANGES[
                    feature_name
                ]
            )

            if (
                minimum is not None
                and
                numeric_value < minimum
            ):

                errors.append(
                    (
                        f"{feature_name} is below "
                        f"minimum allowed value {minimum}."
                    )
                )

            if (
                maximum is not None
                and
                numeric_value > maximum
            ):

                errors.append(
                    (
                        f"{feature_name} exceeds "
                        f"maximum allowed value {maximum}."
                    )
                )

        return (
            len(errors) == 0,
            errors
        )

    # ======================================================================
    # FEATURE ORDER VALIDATION
    # ======================================================================

    @classmethod
    def validate_feature_order(
        cls,
        feature_names: List[str]
    ) -> bool:
        """
        Verify that a supplied feature list exactly matches
        the canonical training/inference order.
        """

        if not isinstance(
            feature_names,
            list
        ):

            return False

        return (
            feature_names
            ==
            cls.FEATURE_NAMES
        )

    # ======================================================================
    # MODEL VECTOR
    # ======================================================================

    @classmethod
    def to_model_vector(
        cls,
        raw_features: Dict[str, Any]
    ) -> List[Union[int, float]]:
        """
        Convert raw SSL features into a deterministic ordered
        numerical vector suitable for ML inference.

        Boolean values are converted to:

            False → 0
            True  → 1
        """

        normalized = (
            cls.validate_and_normalize(
                raw_features
            )
        )

        vector = []

        for feature_name in cls.FEATURE_NAMES:

            value = normalized[
                feature_name
            ]

            if isinstance(
                value,
                bool
            ):

                vector.append(
                    1
                    if value
                    else 0
                )

            else:

                vector.append(
                    value
                )

        return vector

    # ======================================================================
    # MODEL DICTIONARY
    # ======================================================================

    @classmethod
    def to_model_dict(
        cls,
        raw_features: Dict[str, Any]
    ) -> Dict[str, Union[int, float]]:
        """
        Return a model-ready dictionary containing only canonical
        numerical feature values.
        """

        normalized = (
            cls.validate_and_normalize(
                raw_features
            )
        )

        model_dict = {}

        for feature_name in cls.FEATURE_NAMES:

            value = normalized[
                feature_name
            ]

            if isinstance(
                value,
                bool
            ):

                model_dict[
                    feature_name
                ] = (
                    1
                    if value
                    else 0
                )

            else:

                model_dict[
                    feature_name
                ] = value

        return model_dict

    # ======================================================================
    # SCHEMA METADATA
    # ======================================================================

    @classmethod
    def get_schema_metadata(
        cls
    ) -> Dict[str, Any]:
        """
        Return complete schema metadata.

        Useful for:

            Model metadata
            Reports
            Debugging
            API responses
            Dataset versioning
        """

        return {

            "schema_name":
                cls.SCHEMA_NAME,

            "schema_version":
                cls.SCHEMA_VERSION,

            "agent_name":
                cls.AGENT_NAME,

            "feature_count":
                cls.get_feature_count(),

            "feature_names":
                cls.get_feature_names(),

            "feature_types":
                {
                    key:
                        value.__name__

                    for key, value
                    in cls.FEATURE_TYPES.items()
                },

            "feature_descriptions":
                cls.get_feature_descriptions(),

            "feature_ranges":
                {
                    key:
                        list(value)

                    for key, value
                    in cls.FEATURE_RANGES.items()
                }
        }

    # ======================================================================
    # CHECK SCHEMA INTEGRITY
    # ======================================================================

    @classmethod
    def check_schema_integrity(
        cls
    ) -> Tuple[bool, List[str]]:
        """
        Check internal consistency of the schema itself.

        This catches development mistakes such as:

            - feature missing from defaults
            - feature missing from type mapping
            - feature missing from descriptions
            - feature missing from ranges
            - duplicate feature names
        """

        errors = []

        # ------------------------------------------------------------------
        # Feature count
        # ------------------------------------------------------------------

        if len(
            cls.FEATURE_NAMES
        ) != cls.FEATURE_COUNT:

            errors.append(
                (
                    "FEATURE_COUNT mismatch: "
                    f"declared={cls.FEATURE_COUNT}, "
                    f"actual={len(cls.FEATURE_NAMES)}"
                )
            )

        # ------------------------------------------------------------------
        # Duplicate names
        # ------------------------------------------------------------------

        if len(
            cls.FEATURE_NAMES
        ) != len(
            set(
                cls.FEATURE_NAMES
            )
        ):

            errors.append(
                "Duplicate feature names detected."
            )

        feature_set = set(
            cls.FEATURE_NAMES
        )

        # ------------------------------------------------------------------
        # Defaults
        # ------------------------------------------------------------------

        default_set = set(
            cls.DEFAULT_FEATURES.keys()
        )

        missing_defaults = (
            feature_set
            -
            default_set
        )

        extra_defaults = (
            default_set
            -
            feature_set
        )

        if missing_defaults:

            errors.append(
                (
                    "Features missing from DEFAULT_FEATURES: "
                    f"{sorted(missing_defaults)}"
                )
            )

        if extra_defaults:

            errors.append(
                (
                    "Unexpected DEFAULT_FEATURES entries: "
                    f"{sorted(extra_defaults)}"
                )
            )

        # ------------------------------------------------------------------
        # Types
        # ------------------------------------------------------------------

        type_set = set(
            cls.FEATURE_TYPES.keys()
        )

        missing_types = (
            feature_set
            -
            type_set
        )

        if missing_types:

            errors.append(
                (
                    "Features missing from FEATURE_TYPES: "
                    f"{sorted(missing_types)}"
                )
            )

        # ------------------------------------------------------------------
        # Descriptions
        # ------------------------------------------------------------------

        description_set = set(
            cls.FEATURE_DESCRIPTIONS.keys()
        )

        missing_descriptions = (
            feature_set
            -
            description_set
        )

        if missing_descriptions:

            errors.append(
                (
                    "Features missing from FEATURE_DESCRIPTIONS: "
                    f"{sorted(missing_descriptions)}"
                )
            )

        # ------------------------------------------------------------------
        # Ranges
        # ------------------------------------------------------------------

        range_set = set(
            cls.FEATURE_RANGES.keys()
        )

        missing_ranges = (
            feature_set
            -
            range_set
        )

        if missing_ranges:

            errors.append(
                (
                    "Features missing from FEATURE_RANGES: "
                    f"{sorted(missing_ranges)}"
                )
            )

        # ------------------------------------------------------------------
        # Boolean set
        # ------------------------------------------------------------------

        boolean_unknown = (
            cls.BOOLEAN_FEATURES
            -
            feature_set
        )

        if boolean_unknown:

            errors.append(
                (
                    "Unknown BOOLEAN_FEATURES: "
                    f"{sorted(boolean_unknown)}"
                )
            )

        # ------------------------------------------------------------------
        # Integer set
        # ------------------------------------------------------------------

        integer_unknown = (
            cls.INTEGER_FEATURES
            -
            feature_set
        )

        if integer_unknown:

            errors.append(
                (
                    "Unknown INTEGER_FEATURES: "
                    f"{sorted(integer_unknown)}"
                )
            )

        # ------------------------------------------------------------------
        # Float set
        # ------------------------------------------------------------------

        float_unknown = (
            cls.FLOAT_FEATURES
            -
            feature_set
        )

        if float_unknown:

            errors.append(
                (
                    "Unknown FLOAT_FEATURES: "
                    f"{sorted(float_unknown)}"
                )
            )

        # ------------------------------------------------------------------
        # Type grouping completeness
        # ------------------------------------------------------------------

        grouped_features = (
            cls.BOOLEAN_FEATURES
            |
            cls.INTEGER_FEATURES
            |
            cls.FLOAT_FEATURES
        )

        ungrouped_features = (
            feature_set
            -
            grouped_features
        )

        if ungrouped_features:

            errors.append(
                (
                    "Features without a type group: "
                    f"{sorted(ungrouped_features)}"
                )
            )

        # ------------------------------------------------------------------
        # Overlapping type groups
        # ------------------------------------------------------------------

        groups = [

            cls.BOOLEAN_FEATURES,

            cls.INTEGER_FEATURES,

            cls.FLOAT_FEATURES
        ]

        for index, group_a in enumerate(
            groups
        ):

            for group_b in groups[
                index + 1:
            ]:

                overlap = (
                    group_a
                    &
                    group_b
                )

                if overlap:

                    errors.append(
                        (
                            "Feature appears in multiple "
                            f"type groups: {sorted(overlap)}"
                        )
                    )

        return (
            len(errors) == 0,
            errors
        )

    # ======================================================================
    # SUMMARY
    # ======================================================================

    @classmethod
    def get_summary(
        cls
    ) -> Dict[str, Any]:
        """
        Return a concise schema summary suitable for logging.
        """

        integrity_ok, integrity_errors = (
            cls.check_schema_integrity()
        )

        return {

            "schema_name":
                cls.SCHEMA_NAME,

            "schema_version":
                cls.SCHEMA_VERSION,

            "agent":
                cls.AGENT_NAME,

            "feature_count":
                cls.FEATURE_COUNT,

            "integrity_valid":
                integrity_ok,

            "integrity_errors":
                integrity_errors
        }