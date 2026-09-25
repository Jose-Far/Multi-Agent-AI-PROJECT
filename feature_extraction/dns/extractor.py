import logging
from typing import Dict, Any

from .records import DNSRecordsFeatureExtractor
from .mx import DNSMXFeatureExtractor
from .txt import DNSTXTFeatureExtractor
from .dmarc import DNSDMARCFeatureExtractor


logger = logging.getLogger(__name__)


class DNSFeatureExtractor:
    """
    Master DNS Feature Extraction Engine.

    Converts raw DNS dataset / collector telemetry into a unified
    DNS feature vector.

    Supported input formats:

    1. Dataset format:

        {
            "domain_name": "example.com",
            "dns": {
                "A": [...],
                "AAAA": [...],
                "CNAME": ...,
                "MX": {...},
                "NS": {...},
                "TXT": [...],
                "ttls": {...}
            }
        }

    2. Collector format:

        {
            "queried_domain": "example.com",
            "records": {...},
            "ttl_statistics": {...}
        }

    The canonical ML feature set contains 35 features.
    """

    # =====================================================================
    # CANONICAL MODEL FEATURES
    # =====================================================================

    MODEL_FEATURES = [
        "has_a_records",
        "has_aaaa_records",
        "has_cname_record",
        "a_record_count",
        "aaaa_record_count",
        "ns_count",
        "cname_count",
        "total_resolved_ips",
        "has_routing_anomalies",
        "min_ttl_value",
        "max_ttl_value",
        "avg_ttl_value",
        "is_fast_flux_candidate",
        "is_active_fast_flux",
        "has_mx_records",
        "mx_record_count",
        "uses_free_mail_provider",
        "uses_disposable_mail_provider",
        "has_suspicious_mx_exchange",
        "lowest_mx_preference",
        "has_txt_records",
        "txt_record_count",
        "has_domain_verification",
        "has_suspicious_long_txt",
        "has_base64_payload_in_txt",
        "avg_txt_length",
        "has_spf_record",
        "spf_record_count",
        "has_multiple_spf_records",
        "spf_includes_count",
        "spf_strictness_score",
        "has_dmarc_record",
        "dmarc_policy_score",
        "dmarc_subdomain_policy_score",
        "is_email_spoofable",
    ]

    def __init__(self):

        self.records_extractor = (
            DNSRecordsFeatureExtractor()
        )

        self.mx_extractor = (
            DNSMXFeatureExtractor()
        )

        self.txt_extractor = (
            DNSTXTFeatureExtractor()
        )

        self.dmarc_extractor = (
            DNSDMARCFeatureExtractor()
        )

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def extract_features(
        self,
        raw_dns_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract the complete DNS feature vector.

        The method automatically detects whether the input is:

            dataset format:
                {"dns": {...}}

        or:

            collector format:
                {"records": {...}}

        Returns:
            Unified feature dictionary.
        """

        if not isinstance(
            raw_dns_data,
            dict
        ):
            logger.warning(
                "DNS Feature Extractor received invalid input."
            )

            return self._get_empty_features()

        try:

            # =============================================================
            # 1. NORMALIZE INPUT
            # =============================================================

            dns_payload = self._normalize_dns_payload(
                raw_dns_data
            )

            if not dns_payload:

                logger.warning(
                    "DNS payload is empty."
                )

                return self._get_empty_features()

            records = dns_payload.get(
                "records",
                {}
            )

            ttl_statistics = dns_payload.get(
                "ttl_statistics",
                {}
            )

            queried_domain = str(
                dns_payload.get(
                    "queried_domain",
                    ""
                )
            ).strip()

            # =============================================================
            # 2. NORMALIZE RECORDS
            # =============================================================

            if not isinstance(
                records,
                dict
            ):
                records = {}

            if not isinstance(
                ttl_statistics,
                dict
            ):
                ttl_statistics = {}

            # =============================================================
            # 3. NORMALIZE INDIVIDUAL DNS RECORD TYPES
            # =============================================================

            a_records = self._normalize_list(
                records.get("A")
            )

            aaaa_records = self._normalize_list(
                records.get("AAAA")
            )

            cname_records = self._normalize_list(
                records.get("CNAME")
            )

            txt_records = self._normalize_list(
                records.get("TXT")
            )

            mx_records = records.get(
                "MX"
            )

            if mx_records is None:
                mx_records = []

            ns_records = records.get(
                "NS"
            )

            if ns_records is None:
                ns_records = []

            # =============================================================
            # 4. CREATE NORMALIZED RECORD BLOCK
            # =============================================================

            normalized_records = {
                "A": a_records,
                "AAAA": aaaa_records,
                "CNAME": cname_records,
                "MX": mx_records,
                "NS": ns_records,
                "TXT": txt_records,
            }

            # =============================================================
            # 5. RECORD FEATURES
            # =============================================================

            record_features = self._safe_extract(
                self.records_extractor.extract,
                normalized_records,
                ttl_statistics
            )

            # =============================================================
            # 6. MX FEATURES
            # =============================================================

            mx_features = self._safe_extract(
                self.mx_extractor.extract,
                mx_records,
                queried_domain=queried_domain
            )

            # =============================================================
            # 7. TXT FEATURES
            # =============================================================

            txt_features = self._safe_extract(
                self.txt_extractor.extract,
                txt_records
            )

            # =============================================================
            # 8. SPF / DMARC FEATURES
            # =============================================================

            dmarc_features = self._safe_extract(
                self.dmarc_extractor.extract,
                txt_records
            )

            # =============================================================
            # 9. MERGE FEATURES
            # =============================================================

            unified_features = {}

            unified_features.update(
                record_features
            )

            unified_features.update(
                mx_features
            )

            unified_features.update(
                txt_features
            )

            unified_features.update(
                dmarc_features
            )

            # =============================================================
            # 10. ENSURE ALL 35 CANONICAL FEATURES
            # =============================================================

            unified_features = (
                self._ensure_canonical_features(
                    unified_features
                )
            )

            # =============================================================
            # 11. COMPOSITE HEURISTICS
            # =============================================================

            unified_features = (
                self._compute_composite_heuristics(
                    unified_features
                )
            )

            # =============================================================
            # 12. FINAL CANONICAL VALIDATION
            # =============================================================

            missing = [
                feature
                for feature in self.MODEL_FEATURES
                if feature not in unified_features
            ]

            if missing:

                logger.error(
                    "DNS extractor missing canonical features: %s",
                    missing
                )

                return self._get_empty_features()

            return unified_features

        except Exception as exc:

            logger.error(
                "Critical DNS extraction failure: %s",
                exc,
                exc_info=True
            )

            return self._get_empty_features()

    # =====================================================================
    # INPUT NORMALIZATION
    # =====================================================================

    def _normalize_dns_payload(
        self,
        raw_dns_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert both supported input formats into:

            {
                "queried_domain": "...",
                "records": {...},
                "ttl_statistics": {...}
            }
        """

        # ---------------------------------------------------------------
        # Dataset format
        # ---------------------------------------------------------------

        if isinstance(
            raw_dns_data.get("dns"),
            dict
        ):

            dns = raw_dns_data.get(
                "dns",
                {}
            )

            records = {
                "A": dns.get("A"),
                "AAAA": dns.get("AAAA"),
                "CNAME": dns.get("CNAME"),
                "MX": dns.get("MX"),
                "NS": dns.get("NS"),
                "TXT": dns.get("TXT"),
            }

            ttl_statistics = dns.get(
                "ttls",
                {}
            )

            if not isinstance(
                ttl_statistics,
                dict
            ):
                ttl_statistics = {}

            queried_domain = (
                raw_dns_data.get(
                    "domain_name"
                )
                or raw_dns_data.get(
                    "queried_domain"
                )
                or dns.get(
                    "remarks",
                    {}
                ).get(
                    "zone",
                    ""
                )
            )

            return {
                "queried_domain": str(
                    queried_domain or ""
                ),
                "records": records,
                "ttl_statistics": ttl_statistics,
            }

        # ---------------------------------------------------------------
        # Collector format
        # ---------------------------------------------------------------

        if isinstance(
            raw_dns_data.get("records"),
            dict
        ):

            return {
                "queried_domain": str(
                    raw_dns_data.get(
                        "queried_domain",
                        ""
                    )
                ),
                "records": raw_dns_data.get(
                    "records",
                    {}
                ),
                "ttl_statistics": (
                    raw_dns_data.get(
                        "ttl_statistics",
                        {}
                    )
                ),
            }

        # ---------------------------------------------------------------
        # Direct DNS block
        # ---------------------------------------------------------------

        if any(
            key in raw_dns_data
            for key in (
                "A",
                "AAAA",
                "CNAME",
                "MX",
                "NS",
                "TXT",
                "ttls",
            )
        ):

            return {
                "queried_domain": str(
                    raw_dns_data.get(
                        "domain_name",
                        ""
                    )
                ),
                "records": {
                    "A": raw_dns_data.get("A"),
                    "AAAA": raw_dns_data.get("AAAA"),
                    "CNAME": raw_dns_data.get("CNAME"),
                    "MX": raw_dns_data.get("MX"),
                    "NS": raw_dns_data.get("NS"),
                    "TXT": raw_dns_data.get("TXT"),
                },
                "ttl_statistics": (
                    raw_dns_data.get(
                        "ttls",
                        {}
                    )
                ),
            }

        return {}

    # =====================================================================
    # SAFE LIST NORMALIZATION
    # =====================================================================

    @staticmethod
    def _normalize_list(
        value: Any
    ):

        if value is None:
            return []

        if isinstance(
            value,
            list
        ):
            return value

        if isinstance(
            value,
            tuple
        ):
            return list(value)

        if isinstance(
            value,
            set
        ):
            return list(value)

        if isinstance(
            value,
            str
        ):

            value = value.strip()

            if not value:
                return []

            return [value]

        return []

    # =====================================================================
    # SAFE SUB-EXTRACTOR
    # =====================================================================

    def _safe_extract(
        self,
        extractor_func,
        *args,
        **kwargs
    ) -> Dict[str, Any]:

        try:

            result = extractor_func(
                *args,
                **kwargs
            )

            if isinstance(
                result,
                dict
            ):
                return result

            logger.warning(
                "DNS sub-extractor returned non-dict: %s",
                getattr(
                    extractor_func,
                    "__qualname__",
                    str(extractor_func)
                )
            )

            return {}

        except Exception as exc:

            logger.warning(
                "DNS sub-extractor failed: %s",
                exc
            )

            return {}

    # =====================================================================
    # CANONICAL FEATURE GUARANTEE
    # =====================================================================

    def _ensure_canonical_features(
        self,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Guarantee that every one of the 35 model features exists.
        """

        defaults = {

            "has_a_records": False,
            "has_aaaa_records": False,
            "has_cname_record": False,

            "a_record_count": 0,
            "aaaa_record_count": 0,
            "ns_count": 0,
            "cname_count": 0,
            "total_resolved_ips": 0,

            "has_routing_anomalies": True,

            "min_ttl_value": 0,
            "max_ttl_value": 0,
            "avg_ttl_value": 0.0,

            "is_fast_flux_candidate": False,
            "is_active_fast_flux": False,

            "has_mx_records": False,
            "mx_record_count": 0,
            "uses_free_mail_provider": False,
            "uses_disposable_mail_provider": False,
            "has_suspicious_mx_exchange": False,
            "lowest_mx_preference": -1,

            "has_txt_records": False,
            "txt_record_count": 0,
            "has_domain_verification": False,
            "has_suspicious_long_txt": False,
            "has_base64_payload_in_txt": False,
            "avg_txt_length": 0.0,

            "has_spf_record": False,
            "spf_record_count": 0,
            "has_multiple_spf_records": False,
            "spf_includes_count": 0,
            "spf_strictness_score": 0.0,

            "has_dmarc_record": False,
            "dmarc_policy_score": 0.0,
            "dmarc_subdomain_policy_score": 0.0,

            "is_email_spoofable": True,
        }

        for feature, default in defaults.items():

            if (
                feature not in features
                or features[feature] is None
            ):
                features[feature] = default

        # ================================================================
        # Numeric normalization
        # ================================================================

        integer_features = [
            "a_record_count",
            "aaaa_record_count",
            "ns_count",
            "cname_count",
            "total_resolved_ips",
            "min_ttl_value",
            "max_ttl_value",
            "mx_record_count",
            "lowest_mx_preference",
            "txt_record_count",
            "spf_record_count",
            "spf_includes_count",
        ]

        for feature in integer_features:

            try:
                features[feature] = int(
                    features[feature]
                )
            except (
                TypeError,
                ValueError,
            ):
                features[feature] = 0

        float_features = [
            "avg_ttl_value",
            "avg_txt_length",
            "spf_strictness_score",
            "dmarc_policy_score",
            "dmarc_subdomain_policy_score",
        ]

        for feature in float_features:

            try:
                value = float(
                    features[feature]
                )

                if value != value:
                    value = 0.0

                features[feature] = value

            except (
                TypeError,
                ValueError,
            ):
                features[feature] = 0.0

        return features

    # =====================================================================
    # COMPOSITE HEURISTICS
    # =====================================================================

    def _compute_composite_heuristics(
        self,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:

        # ================================================================
        # Infrastructure health
        # ================================================================

        infrastructure_health = 100

        if features.get(
            "has_routing_anomalies",
            False
        ):
            infrastructure_health -= 50

        if features.get(
            "has_base64_payload_in_txt",
            False
        ):
            infrastructure_health -= 40

        infrastructure_health = max(
            0,
            min(
                100,
                infrastructure_health
            )
        )

        features[
            "infrastructure_health_score"
        ] = infrastructure_health

        # ================================================================
        # Fast flux health
        # ================================================================

        fast_flux_health = 100

        if features.get(
            "is_active_fast_flux",
            False
        ):

            fast_flux_health -= 80

        elif features.get(
            "is_fast_flux_candidate",
            False
        ):

            fast_flux_health -= 30

        fast_flux_health = max(
            0,
            min(
                100,
                fast_flux_health
            )
        )

        features[
            "fast_flux_health_score"
        ] = fast_flux_health

        # ================================================================
        # Email health
        # ================================================================

        email_health = 100

        if not features.get(
            "has_mx_records",
            False
        ):
            email_health -= 15

        if features.get(
            "uses_disposable_mail_provider",
            False
        ):
            email_health -= 40

        if features.get(
            "has_suspicious_mx_exchange",
            False
        ):
            email_health -= 30

        if features.get(
            "has_multiple_spf_records",
            False
        ):
            email_health -= 20

        if features.get(
            "is_email_spoofable",
            False
        ):
            email_health -= 15

        email_health = max(
            0,
            min(
                100,
                email_health
            )
        )

        features[
            "email_health_score"
        ] = email_health

        # ================================================================
        # Overall DNS health
        # ================================================================

        dns_health_score = (
            infrastructure_health * 0.50
            + fast_flux_health * 0.40
            + email_health * 0.10
        )

        features[
            "dns_health_score"
        ] = int(
            max(
                0,
                min(
                    100,
                    dns_health_score
                )
            )
        )

        # ================================================================
        # Suspicious infrastructure
        # ================================================================

        features[
            "has_suspicious_infrastructure"
        ] = (

            infrastructure_health < 60

            or

            fast_flux_health < 60

            or

            features.get(
                "has_base64_payload_in_txt",
                False
            )

            or

            features.get(
                "uses_disposable_mail_provider",
                False
            )
        )

        return features

    # =====================================================================
    # EMPTY BASELINE
    # =====================================================================

    def _get_empty_features(
        self
    ) -> Dict[str, Any]:

        features = {

            "has_a_records": False,
            "has_aaaa_records": False,
            "has_cname_record": False,

            "a_record_count": 0,
            "aaaa_record_count": 0,
            "ns_count": 0,
            "cname_count": 0,
            "total_resolved_ips": 0,

            "has_routing_anomalies": True,

            "min_ttl_value": 0,
            "max_ttl_value": 0,
            "avg_ttl_value": 0.0,

            "is_fast_flux_candidate": False,
            "is_active_fast_flux": False,

            "has_mx_records": False,
            "mx_record_count": 0,
            "uses_free_mail_provider": False,
            "uses_disposable_mail_provider": False,
            "has_suspicious_mx_exchange": False,
            "lowest_mx_preference": -1,

            "has_txt_records": False,
            "txt_record_count": 0,
            "has_domain_verification": False,
            "has_suspicious_long_txt": False,
            "has_base64_payload_in_txt": False,
            "avg_txt_length": 0.0,

            "has_spf_record": False,
            "spf_record_count": 0,
            "has_multiple_spf_records": False,
            "spf_includes_count": 0,
            "spf_strictness_score": 0.0,

            "has_dmarc_record": False,
            "dmarc_policy_score": 0.0,
            "dmarc_subdomain_policy_score": 0.0,

            "is_email_spoofable": True,
        }

        return features