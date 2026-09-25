import logging
import time
from typing import Dict, Any

from feature_extraction.ssl.certificate import analyze_certificate
from feature_extraction.ssl.issuer import analyze_issuer
from feature_extraction.ssl.cipher import analyze_cipher


logger = logging.getLogger(__name__)


class SSLExtractor:
    """
    Master SSL Feature Extractor.

    Extracts certificate, issuer, and cryptographic information from
    raw SSL/TLS telemetry and returns the exact 17-feature canonical
    vector required by the SSL XGBoost model.

    Canonical model features:

        1.  has_ssl
        2.  is_expired
        3.  cert_age_days
        4.  days_until_expiry
        5.  total_lifespan_days
        6.  is_self_signed
        7.  is_recently_issued
        8.  is_short_lived
        9.  is_trusted_ca
        10. is_free_automated_ca
        11. is_suspicious_ca
        12. issuer_trust_score
        13. is_weak_protocol
        14. is_vulnerable_cipher
        15. cipher_strength_bits
        16. crypto_health_score
        17. is_secure_connection

    Extra collection/diagnostic fields such as issuer_cn,
    issuer_org, tls_version, cipher_suite, and extraction_time_ms
    are intentionally not returned as model features.
    """

    # ------------------------------------------------------------------
    # EXACT MODEL FEATURE SCHEMA
    # ------------------------------------------------------------------

    MODEL_FEATURES = [
        "has_ssl",
        "is_expired",
        "cert_age_days",
        "days_until_expiry",
        "total_lifespan_days",
        "is_self_signed",
        "is_recently_issued",
        "is_short_lived",
        "is_trusted_ca",
        "is_free_automated_ca",
        "is_suspicious_ca",
        "issuer_trust_score",
        "is_weak_protocol",
        "is_vulnerable_cipher",
        "cipher_strength_bits",
        "crypto_health_score",
        "is_secure_connection",
    ]

    # ------------------------------------------------------------------
    # SAFE DEFAULTS
    # ------------------------------------------------------------------

    DEFAULT_FEATURES = {
        "has_ssl": False,
        "is_expired": True,
        "cert_age_days": 0,
        "days_until_expiry": 0,
        "total_lifespan_days": 0,
        "is_self_signed": False,
        "is_recently_issued": False,
        "is_short_lived": False,
        "is_trusted_ca": False,
        "is_free_automated_ca": False,
        "is_suspicious_ca": False,
        "issuer_trust_score": 0.0,
        "is_weak_protocol": False,
        "is_vulnerable_cipher": False,
        "cipher_strength_bits": 0,
        "crypto_health_score": 0.0,
        "is_secure_connection": False,
    }

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    def __init__(
        self,
        raw_ssl_analysis: Dict[str, Any],
    ):
        """
        Initialize the SSL extractor.

        Args:
            raw_ssl_analysis:
                Raw SSL/TLS telemetry from the data collection layer.

                It may either be:

                    {
                        "success": True,
                        "error": None,
                        "data": {...}
                    }

                or directly:

                    {...}
        """

        logger.info(
            "Initializing Advanced SSLExtractor engine..."
        )

        if not isinstance(
            raw_ssl_analysis,
            dict,
        ):
            raw_ssl_analysis = {}

        # --------------------------------------------------------------
        # Safely unwrap collector payload
        # --------------------------------------------------------------

        if "data" in raw_ssl_analysis:

            self.ssl_payload = (
                raw_ssl_analysis.get(
                    "data"
                )
                or {}
            )

            self.success = bool(
                raw_ssl_analysis.get(
                    "success",
                    False,
                )
            )

            self.error_msg = (
                raw_ssl_analysis.get(
                    "error"
                )
            )

        else:

            self.ssl_payload = raw_ssl_analysis

            self.success = True

            self.error_msg = None

    # ------------------------------------------------------------------
    # FEATURE SANITIZATION HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_non_negative_int(
        value: Any,
        default: int = 0,
    ) -> int:
        """
        Convert a value to a non-negative integer.
        """

        try:
            value = int(float(value))

        except (
            TypeError,
            ValueError,
        ):
            return default

        return max(
            0,
            value,
        )

    @staticmethod
    def _safe_non_negative_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Convert a value to a non-negative float.
        """

        try:
            value = float(value)

        except (
            TypeError,
            ValueError,
        ):
            return default

        if value != value:
            return default

        if value < 0:
            return default

        return value

    @staticmethod
    def _safe_bool(
        value: Any,
        default: bool = False,
    ) -> bool:
        """
        Normalize boolean-like values.
        """

        if isinstance(
            value,
            bool,
        ):
            return value

        if value is None:
            return default

        if isinstance(
            value,
            str,
        ):

            normalized = (
                value.strip()
                .lower()
            )

            if normalized in {
                "true",
                "1",
                "yes",
                "y",
            }:
                return True

            if normalized in {
                "false",
                "0",
                "no",
                "n",
            }:
                return False

        return bool(value)

    # ------------------------------------------------------------------
    # BUILD CANONICAL 17-FEATURE VECTOR
    # ------------------------------------------------------------------

    def _build_model_features(
        self,
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert the internal/extracted SSL information into the exact
        canonical 17-feature model schema.

        This is the critical boundary between:

            raw SSL telemetry

        and:

            XGBoost model input.
        """

        # --------------------------------------------------------------
        # Certificate numeric features
        # --------------------------------------------------------------

        cert_age_days = self._safe_non_negative_int(
            features.get(
                "cert_age_days",
                0,
            )
        )

        days_until_expiry = self._safe_non_negative_int(
            features.get(
                "days_until_expiry",
                0,
            )
        )

        total_lifespan_days = self._safe_non_negative_int(
            features.get(
                "total_lifespan_days",
                0,
            )
        )

        # --------------------------------------------------------------
        # Ensure certificate relationships cannot produce negatives
        # --------------------------------------------------------------

        if total_lifespan_days < cert_age_days:
            total_lifespan_days = cert_age_days

        # --------------------------------------------------------------
        # Issuer trust
        # --------------------------------------------------------------

        issuer_trust_score = self._safe_non_negative_float(
            features.get(
                "issuer_trust_score",
                0.0,
            )
        )

        issuer_trust_score = min(
            100.0,
            issuer_trust_score,
        )

        # --------------------------------------------------------------
        # Cipher strength
        # --------------------------------------------------------------

        cipher_strength_bits = self._safe_non_negative_int(
            features.get(
                "cipher_strength_bits",
                0,
            )
        )

        # --------------------------------------------------------------
        # Crypto health
        # --------------------------------------------------------------

        crypto_health_score = self._safe_non_negative_float(
            features.get(
                "crypto_health_score",
                0.0,
            )
        )

        crypto_health_score = min(
            100.0,
            crypto_health_score,
        )

        # --------------------------------------------------------------
        # FREE / AUTOMATED CA NORMALIZATION
        #
        # issuer.py may expose:
        #
        #   is_free_ca
        #   is_automated_ca
        #
        # The trained model requires:
        #
        #   is_free_automated_ca
        # --------------------------------------------------------------

        if "is_free_automated_ca" in features:

            is_free_automated_ca = self._safe_bool(
                features.get(
                    "is_free_automated_ca"
                )
            )

        else:

            is_free_ca = self._safe_bool(
                features.get(
                    "is_free_ca",
                    False,
                )
            )

            is_automated_ca = self._safe_bool(
                features.get(
                    "is_automated_ca",
                    False,
                )
            )

            is_free_automated_ca = (
                is_free_ca
                or is_automated_ca
            )

        # --------------------------------------------------------------
        # Master security flag
        # --------------------------------------------------------------

        is_secure_connection = True

        if not self._safe_bool(
            features.get(
                "has_ssl",
                False,
            )
        ):
            is_secure_connection = False

        if self._safe_bool(
            features.get(
                "is_expired",
                True,
            )
        ):
            is_secure_connection = False

        if self._safe_bool(
            features.get(
                "is_self_signed",
                False,
            )
        ):
            is_secure_connection = False

        if self._safe_bool(
            features.get(
                "is_suspicious_ca",
                False,
            )
        ):
            is_secure_connection = False

        if self._safe_bool(
            features.get(
                "is_weak_protocol",
                False,
            )
        ):
            is_secure_connection = False

        if self._safe_bool(
            features.get(
                "is_vulnerable_cipher",
                False,
            )
        ):
            is_secure_connection = False

        if crypto_health_score < 50:
            is_secure_connection = False

        if issuer_trust_score < 40:
            is_secure_connection = False

        # --------------------------------------------------------------
        # EXACT 17-FEATURE VECTOR
        # --------------------------------------------------------------

        model_features = {
            "has_ssl": self._safe_bool(
                features.get(
                    "has_ssl",
                    False,
                )
            ),

            "is_expired": self._safe_bool(
                features.get(
                    "is_expired",
                    True,
                ),
                default=True,
            ),

            "cert_age_days": cert_age_days,

            "days_until_expiry": days_until_expiry,

            "total_lifespan_days": total_lifespan_days,

            "is_self_signed": self._safe_bool(
                features.get(
                    "is_self_signed",
                    False,
                )
            ),

            "is_recently_issued": self._safe_bool(
                features.get(
                    "is_recently_issued",
                    False,
                )
            ),

            "is_short_lived": self._safe_bool(
                features.get(
                    "is_short_lived",
                    False,
                )
            ),

            "is_trusted_ca": self._safe_bool(
                features.get(
                    "is_trusted_ca",
                    False,
                )
            ),

            "is_free_automated_ca":
                is_free_automated_ca,

            "is_suspicious_ca": self._safe_bool(
                features.get(
                    "is_suspicious_ca",
                    False,
                )
            ),

            "issuer_trust_score":
                issuer_trust_score,

            "is_weak_protocol": self._safe_bool(
                features.get(
                    "is_weak_protocol",
                    False,
                )
            ),

            "is_vulnerable_cipher":
                self._safe_bool(
                    features.get(
                        "is_vulnerable_cipher",
                        False,
                    )
                ),

            "cipher_strength_bits":
                cipher_strength_bits,

            "crypto_health_score":
                crypto_health_score,

            "is_secure_connection":
                is_secure_connection,
        }

        # --------------------------------------------------------------
        # Final schema assertion
        # --------------------------------------------------------------

        if list(
            model_features.keys()
        ) != self.MODEL_FEATURES:

            raise RuntimeError(
                "SSL canonical feature schema mismatch. "
                f"Expected {len(self.MODEL_FEATURES)} features, "
                f"got {len(model_features)}."
            )

        return model_features

    # ------------------------------------------------------------------
    # MAIN EXTRACTION
    # ------------------------------------------------------------------

    def extract(
        self,
    ) -> Dict[str, Any]:
        """
        Execute all SSL sub-modules and return the exact canonical
        17-feature model vector.

        Returns:
            Dict containing exactly the 17 model features.
        """

        start_time = time.time()

        # --------------------------------------------------------------
        # Start from safe internal defaults
        # --------------------------------------------------------------

        features = dict(
            self.DEFAULT_FEATURES
        )

        # --------------------------------------------------------------
        # SSL unavailable / collection failure
        # --------------------------------------------------------------

        if (
            not self.success
            or not self.ssl_payload
            or not self.ssl_payload.get(
                "has_ssl",
                False,
            )
        ):

            logger.warning(
                "SSL telemetry missing or collection failed "
                "(%s). Defaulting to unsecured HTTP profile.",
                self.error_msg,
            )

            model_features = (
                self._build_model_features(
                    features
                )
            )

            logger.info(
                "SSL feature extraction completed with "
                "unsecured defaults."
            )

            return model_features

        # --------------------------------------------------------------
        # SSL is available
        # --------------------------------------------------------------

        features["has_ssl"] = True

        # --------------------------------------------------------------
        # Certificate analysis
        # --------------------------------------------------------------

        try:

            cert_metrics = (
                analyze_certificate(
                    self.ssl_payload
                )
            )

            if isinstance(
                cert_metrics,
                dict,
            ):
                features.update(
                    cert_metrics
                )

        except Exception as exc:

            logger.error(
                "Certificate analysis sub-module failed: %s",
                exc,
            )

        # --------------------------------------------------------------
        # Issuer analysis
        # --------------------------------------------------------------

        try:

            issuer_metrics = (
                analyze_issuer(
                    self.ssl_payload
                )
            )

            if isinstance(
                issuer_metrics,
                dict,
            ):
                features.update(
                    issuer_metrics
                )

        except Exception as exc:

            logger.error(
                "Issuer analysis sub-module failed: %s",
                exc,
            )

        # --------------------------------------------------------------
        # Cipher / TLS analysis
        # --------------------------------------------------------------

        try:

            cipher_metrics = (
                analyze_cipher(
                    self.ssl_payload
                )
            )

            if isinstance(
                cipher_metrics,
                dict,
            ):
                features.update(
                    cipher_metrics
                )

        except Exception as exc:

            logger.error(
                "Cipher analysis sub-module failed: %s",
                exc,
            )

        # --------------------------------------------------------------
        # Build canonical ML vector
        # --------------------------------------------------------------

        model_features = (
            self._build_model_features(
                features
            )
        )

        # --------------------------------------------------------------
        # Profiling
        # --------------------------------------------------------------

        extraction_time_ms = round(
            (
                time.time()
                - start_time
            )
            * 1000,
            2,
        )

        logger.info(
            "SSL feature extraction completed successfully "
            "in %s ms. Canonical model features=%s",
            extraction_time_ms,
            len(model_features),
        )

        # --------------------------------------------------------------
        # Return ONLY the canonical model vector
        # --------------------------------------------------------------

        return model_features