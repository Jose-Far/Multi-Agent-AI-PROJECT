import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class DNSMXFeatureExtractor:
    """
    Advanced Mail Exchange (MX) Feature Extractor.

    Supports both:

    1. Collector/list format:

        [
            {
                "exchange": "mail.example.com",
                "preference": 10
            }
        ]

    2. Dataset/dictionary format:

        {
            "mail.example.com": {
                "priority": 10,
                "related_ips": [...]
            }
        }

    Produces the canonical MX features:

        has_mx_records
        mx_record_count
        uses_free_mail_provider
        uses_disposable_mail_provider
        has_suspicious_mx_exchange
        lowest_mx_preference
    """

    def __init__(self):

        # ================================================================
        # FREE / PUBLIC MAIL PROVIDERS
        # ================================================================

        self.free_mail_providers = [
            "google.com",
            "googlemail.com",
            "outlook.com",
            "hotmail.com",
            "yahoo.com",
            "yandex.com",
            "yandex.ru",
            "zoho.com",
            "protonmail.com",
            "mail.ru",
            "aol.com",
            "icloud.com",
        ]

        # ================================================================
        # DISPOSABLE MAIL PROVIDERS
        # ================================================================

        self.disposable_mail_providers = [
            "10minutemail",
            "temp-mail",
            "guerrillamail",
            "mailinator",
            "throwawaymail",
            "yopmail",
            "getairmail",
            "sharklasers",
        ]

        # ================================================================
        # SUSPICIOUS / TYPOSQUATTED MAIL EXCHANGES
        # ================================================================

        self.suspicious_exchange_patterns = [
            re.compile(r"g[o0]{2}gle"),
            re.compile(r"yah[o0]{2}"),
            re.compile(r"0utl[o0]{2}k"),
        ]

    # ====================================================================
    # NORMALIZE MX RECORDS
    # ====================================================================

    def _normalize_mx_records(
        self,
        mx_records: Any,
    ) -> List[Dict[str, Any]]:
        """
        Convert all supported MX representations into a standard list:

            [
                {
                    "exchange": "...",
                    "preference": 10
                }
            ]
        """

        if mx_records is None:
            return []

        # ---------------------------------------------------------------
        # Already normalized list
        # ---------------------------------------------------------------

        if isinstance(
            mx_records,
            list,
        ):

            normalized = []

            for item in mx_records:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                exchange = (
                    item.get("exchange")
                    or item.get("host")
                    or item.get("name")
                    or ""
                )

                preference = (
                    item.get("preference")
                    if item.get("preference") is not None
                    else item.get("priority")
                )

                normalized.append(
                    {
                        "exchange": str(
                            exchange
                        ).strip(),
                        "preference": preference,
                    }
                )

            return normalized

        # ---------------------------------------------------------------
        # Dataset dictionary format
        # ---------------------------------------------------------------

        if isinstance(
            mx_records,
            dict,
        ):

            normalized = []

            for exchange, metadata in mx_records.items():

                if metadata is None:
                    metadata = {}

                if not isinstance(
                    metadata,
                    dict,
                ):
                    metadata = {}

                preference = (
                    metadata.get("preference")
                    if metadata.get("preference") is not None
                    else metadata.get("priority")
                )

                normalized.append(
                    {
                        "exchange": str(
                            exchange
                        ).strip(),
                        "preference": preference,
                    }
                )

            return normalized

        return []

    # ====================================================================
    # SAFE PREFERENCE
    # ====================================================================

    @staticmethod
    def _safe_preference(
        value: Any,
    ):

        if value is None:
            return None

        try:

            # Handle strings such as "10"
            if isinstance(
                value,
                str,
            ):

                value = value.strip()

                if not value:
                    return None

                if not re.fullmatch(
                    r"\d+",
                    value,
                ):
                    return None

            number = int(
                value
            )

            if number < 0:
                return None

            return number

        except (
            TypeError,
            ValueError,
        ):
            return None

    # ====================================================================
    # MAIN EXTRACTION
    # ====================================================================

    def extract(
        self,
        mx_records: Any,
        queried_domain: str = "",
    ) -> Dict[str, Any]:
        """
        Extract MX security features.
        """

        features = {
            "has_mx_records": False,
            "mx_record_count": 0,
            "uses_free_mail_provider": False,
            "uses_disposable_mail_provider": False,
            "has_suspicious_mx_exchange": False,
            "lowest_mx_preference": -1,
        }

        try:

            normalized_records = (
                self._normalize_mx_records(
                    mx_records
                )
            )

            # ------------------------------------------------------------
            # No MX records
            # ------------------------------------------------------------

            if not normalized_records:
                return features

            # ------------------------------------------------------------
            # Basic MX statistics
            # ------------------------------------------------------------

            features[
                "has_mx_records"
            ] = True

            features[
                "mx_record_count"
            ] = len(
                normalized_records
            )

            # ------------------------------------------------------------
            # Query domain
            # ------------------------------------------------------------

            queried_domain_lower = str(
                queried_domain or ""
            ).lower().strip().rstrip(".")

            preferences = []

            # ============================================================
            # ANALYZE EACH MX
            # ============================================================

            for mx in normalized_records:

                exchange = str(
                    mx.get(
                        "exchange",
                        ""
                    )
                    or ""
                ).lower().strip().rstrip(".")

                if not exchange:
                    continue

                # --------------------------------------------------------
                # Preference
                # --------------------------------------------------------

                preference = self._safe_preference(
                    mx.get(
                        "preference"
                    )
                )

                if preference is not None:
                    preferences.append(
                        preference
                    )

                # --------------------------------------------------------
                # Self-hosted check
                # --------------------------------------------------------

                is_self_hosted = False

                if queried_domain_lower:

                    # Exact domain
                    if (
                        exchange
                        == queried_domain_lower
                    ):
                        is_self_hosted = True

                    # Subdomain of queried domain
                    elif exchange.endswith(
                        "." + queried_domain_lower
                    ):
                        is_self_hosted = True

                # --------------------------------------------------------
                # Free mail provider
                # --------------------------------------------------------

                if not is_self_hosted:

                    for provider in (
                        self.free_mail_providers
                    ):

                        provider_lower = (
                            provider.lower()
                        )

                        if (
                            exchange
                            == provider_lower
                            or exchange.endswith(
                                "." + provider_lower
                            )
                        ):

                            features[
                                "uses_free_mail_provider"
                            ] = True

                            break

                # --------------------------------------------------------
                # Disposable mail provider
                # --------------------------------------------------------

                if not features[
                    "uses_disposable_mail_provider"
                ]:

                    for burner in (
                        self.disposable_mail_providers
                    ):

                        if burner.lower() in exchange:

                            features[
                                "uses_disposable_mail_provider"
                            ] = True

                            break

                # --------------------------------------------------------
                # Suspicious / typosquatted exchange
                # --------------------------------------------------------

                if not features[
                    "has_suspicious_mx_exchange"
                ]:

                    suspicious_match = any(
                        pattern.search(
                            exchange
                        )
                        for pattern in (
                            self.suspicious_exchange_patterns
                        )
                    )

                    if suspicious_match:

                        is_known_provider = any(
                            provider.lower()
                            in exchange
                            for provider
                            in self.free_mail_providers
                        )

                        if not is_known_provider:

                            features[
                                "has_suspicious_mx_exchange"
                            ] = True

            # ============================================================
            # LOWEST MX PREFERENCE
            # ============================================================

            if preferences:

                features[
                    "lowest_mx_preference"
                ] = min(
                    preferences
                )

        except Exception as exc:

            logger.error(
                "Error during DNS MX Feature Extraction: %s",
                exc,
                exc_info=True,
            )

            # Return safe neutral output.
            return {
                "has_mx_records": False,
                "mx_record_count": 0,
                "uses_free_mail_provider": False,
                "uses_disposable_mail_provider": False,
                "has_suspicious_mx_exchange": False,
                "lowest_mx_preference": -1,
            }

        return features