import logging
from typing import Any, Dict
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


class BrandLogoMatchFeatureExtractor:
    """
    Brand logo matching feature extractor.

    Converts visual/logo telemetry into a stable, deterministic feature
    block for the Brand feature-extraction pipeline.

    Output features:
        1. logo_detected
        2. logo_confidence
        3. primary_brand_name
        4. is_highly_targeted_brand
        5. is_high_confidence_impersonation
        6. possible_adversarial_logo_evasion
        7. logo_risk_tier
    """

    FEATURE_NAMES = [
        "logo_detected",
        "logo_confidence",
        "primary_brand_name",
        "is_highly_targeted_brand",
        "is_high_confidence_impersonation",
        "possible_adversarial_logo_evasion",
        "logo_risk_tier",
    ]

    FEATURE_COUNT = 7

    HIGH_CONFIDENCE_THRESHOLD = 0.85
    ADVERSARIAL_SUSPECT_LOWER_BOUND = 0.50
    ADVERSARIAL_SUSPECT_UPPER_BOUND = 0.84

    HIGHLY_TARGETED_BRANDS = {
        "microsoft",
        "paypal",
        "google",
        "apple",
        "amazon",
        "facebook",
        "chase",
        "wells fargo",
        "docusign",
        "netflix",
        "linkedin",
        "instagram",
        "yahoo",
        "adobe",
    }

    def __init__(self):
        """
        Initialize the Brand Logo Match extractor.
        """

        self.highly_targeted_brands = set(
            self.HIGHLY_TARGETED_BRANDS
        )

    # =====================================================================
    # DEFAULT FEATURES
    # =====================================================================

    @classmethod
    def get_default_features(cls) -> Dict[str, Any]:
        """
        Return the canonical default feature dictionary.
        """

        return {
            "logo_detected": False,
            "logo_confidence": 0.0,
            "primary_brand_name": "unknown",
            "is_highly_targeted_brand": False,
            "is_high_confidence_impersonation": False,
            "possible_adversarial_logo_evasion": False,
            "logo_risk_tier": "safe",
        }

    # =====================================================================
    # FEATURE VALIDATION
    # =====================================================================

    @classmethod
    def validate_features(
        cls,
        features: Dict[str, Any],
    ) -> bool:
        """
        Validate that the returned feature block exactly matches the
        canonical Logo Match feature schema.
        """

        if not isinstance(features, dict):
            return False

        if set(features.keys()) != set(cls.FEATURE_NAMES):
            return False

        if not isinstance(
            features["logo_detected"],
            bool,
        ):
            return False

        if not isinstance(
            features["logo_confidence"],
            (int, float),
        ):
            return False

        confidence = float(
            features["logo_confidence"]
        )

        if confidence < 0.0 or confidence > 1.0:
            return False

        if not isinstance(
            features["primary_brand_name"],
            str,
        ):
            return False

        if not isinstance(
            features["is_highly_targeted_brand"],
            bool,
        ):
            return False

        if not isinstance(
            features["is_high_confidence_impersonation"],
            bool,
        ):
            return False

        if not isinstance(
            features["possible_adversarial_logo_evasion"],
            bool,
        ):
            return False

        if features["logo_risk_tier"] not in {
            "safe",
            "suspicious",
            "critical",
        }:
            return False

        return True

    # =====================================================================
    # NORMALIZATION
    # =====================================================================

    @staticmethod
    def _normalize_brand_name(
        value: Any,
    ) -> str:
        """
        Normalize detected brand names.
        """

        if value is None:
            return "unknown"

        brand = str(value).strip().lower()

        if not brand:
            return "unknown"

        return brand

    @staticmethod
    def _normalize_confidence(
        value: Any,
    ) -> float:
        """
        Normalize logo confidence to [0, 1].
        """

        try:
            confidence = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        # Some telemetry systems may provide percentages.
        if confidence > 1.0 and confidence <= 100.0:
            confidence /= 100.0

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    @staticmethod
    def _normalize_boolean(
        value: Any,
    ) -> bool:
        """
        Normalize boolean-like telemetry.
        """

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):

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
                "detected",
            }:
                return True

            if normalized in {
                "false",
                "0",
                "no",
                "n",
                "not_detected",
            }:
                return False

        return False

    # =====================================================================
    # DOMAIN NORMALIZATION
    # =====================================================================

    @staticmethod
    def _extract_domain(
        target_domain: str,
    ) -> str:
        """
        Normalize a URL/domain into a hostname.
        """

        value = str(
            target_domain or ""
        ).strip().lower()

        if not value:
            return ""

        try:

            parsed = urlparse(
                value
                if "://" in value
                else f"//{value}"
            )

            hostname = (
                parsed.hostname
                or ""
            )

            return hostname.strip().lower()

        except Exception:
            return ""

    @staticmethod
    def _domain_contains_brand(
        domain: str,
        brand: str,
    ) -> bool:
        """
        Determine whether the brand is represented as a complete hostname
        token.

        Examples:

            accounts.google.com
                -> True for google

            google-security.com
                -> False for google

            amazonn-login.com
                -> False for amazon
        """

        if not domain or not brand:
            return False

        tokens = []

        for label in domain.split("."):

            for token in label.replace(
                "_",
                "-"
            ).split("-"):

                token = token.strip()

                if token:
                    tokens.append(token)

        return brand in tokens

    # =====================================================================
    # MAIN EXTRACTION
    # =====================================================================

    def extract(
        self,
        visual_features: Dict[str, Any],
        target_domain: str = "",
    ) -> Dict[str, Any]:
        """
        Extract canonical logo/brand matching features.

        Parameters
        ----------
        visual_features:
            Visual Agent telemetry.

        target_domain:
            Target URL or hostname.

        Returns
        -------
        Dict[str, Any]
            Exactly seven canonical Logo Match features.
        """

        features = (
            self.get_default_features()
        )

        if not isinstance(
            visual_features,
            dict,
        ):
            return features

        try:

            # =============================================================
            # BASE TELEMETRY
            # =============================================================

            logo_detected = (
                self._normalize_boolean(
                    visual_features.get(
                        "logo_detected",
                        False,
                    )
                )
            )

            confidence = (
                self._normalize_confidence(
                    visual_features.get(
                        "logo_confidence",
                        0.0,
                    )
                )
            )

            brand_name = (
                self._normalize_brand_name(
                    visual_features.get(
                        "primary_brand_name",
                        visual_features.get(
                            "brand_name",
                            "unknown",
                        ),
                    )
                )
            )

            features[
                "logo_detected"
            ] = logo_detected

            features[
                "logo_confidence"
            ] = round(
                confidence,
                4,
            )

            features[
                "primary_brand_name"
            ] = brand_name

            # =============================================================
            # NO LOGO / UNKNOWN BRAND
            # =============================================================

            if (
                not logo_detected
                or brand_name == "unknown"
            ):
                return features

            # =============================================================
            # TARGETED BRAND
            # =============================================================

            features[
                "is_highly_targeted_brand"
            ] = any(
                targeted == brand_name
                or targeted in brand_name
                for targeted
                in self.highly_targeted_brands
            )

            # =============================================================
            # OFFICIAL DOMAIN GUARDRAIL
            # =============================================================

            domain = self._extract_domain(
                target_domain
            )

            official_domain_match = (
                self._domain_contains_brand(
                    domain,
                    brand_name,
                )
            )

            if official_domain_match:

                features[
                    "is_high_confidence_impersonation"
                ] = False

                features[
                    "possible_adversarial_logo_evasion"
                ] = False

                features[
                    "logo_risk_tier"
                ] = "safe"

                return features

            # =============================================================
            # ADVISORY / ADVERSARIAL Evasion
            # =============================================================

            possible_evasion = (
                self.ADVERSARIAL_SUSPECT_LOWER_BOUND
                <= confidence
                <= self.ADVERSARIAL_SUSPECT_UPPER_BOUND
            )

            if possible_evasion:

                features[
                    "possible_adversarial_logo_evasion"
                ] = True

                features[
                    "logo_risk_tier"
                ] = "suspicious"

            # =============================================================
            # HIGH-CONFIDENCE IMPERSONATION
            # =============================================================

            if (
                confidence
                >= self.HIGH_CONFIDENCE_THRESHOLD
            ):

                features[
                    "is_high_confidence_impersonation"
                ] = True

                features[
                    "logo_risk_tier"
                ] = "critical"

            # =============================================================
            # TARGETED BRAND + EVASION ESCALATION
            # =============================================================

            if (
                features[
                    "is_highly_targeted_brand"
                ]
                and features[
                    "possible_adversarial_logo_evasion"
                ]
            ):

                features[
                    "logo_risk_tier"
                ] = "critical"

            # =============================================================
            # FINAL VALIDATION
            # =============================================================

            if not self.validate_features(
                features
            ):

                logger.error(
                    "Brand Logo Match produced "
                    "an invalid feature vector."
                )

                return (
                    self.get_default_features()
                )

            return features

        except Exception as exc:

            logger.error(
                "Brand Logo Match extraction "
                "failed: %s",
                exc,
                exc_info=True,
            )

            return (
                self.get_default_features()
            )