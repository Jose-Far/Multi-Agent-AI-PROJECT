import logging
import statistics
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DNSRecordsFeatureExtractor:
    """
    Advanced DNS Routing and Fast-Flux Feature Extractor.

    Analyzes A, AAAA, NS, and CNAME records together with TTL telemetry
    to identify routing anomalies and possible fast-flux infrastructure.

    Output is compatible with the canonical DNS feature schema.
    """

    def __init__(self):
        # DNS TTL below this value is considered suspicious for fast-flux
        # detection, subject to CDN suppression.
        self.fast_flux_ttl_threshold = 60

        # Multiple resolved IP addresses combined with very low TTL
        # strengthens the fast-flux signal.
        self.fast_flux_ip_threshold = 3

        # Known legitimate CDN / DNS infrastructure.
        self.known_cdns = [
            "cloudflare.com",
            "awsdns",
            "googledomains",
            "akamai.net",
            "fastly.net",
            "azure-dns",
            "dnsv1.com",
            "domaincontrol.com",
        ]

    # ========================================================================
    # SAFE NORMALIZATION HELPERS
    # ========================================================================

    @staticmethod
    def _safe_collection(
        value: Any,
    ):
        """
        Normalize a DNS record container.

        The source dataset can contain:
            None
            list
            tuple
            set
            dict
            string

        For dictionaries, keys represent DNS record names.
        """

        if value is None:
            return []

        if isinstance(value, dict):
            return list(value.keys())

        if isinstance(value, (list, tuple, set)):
            return list(value)

        if isinstance(value, str):
            if not value.strip():
                return []
            return [value]

        return []

    @staticmethod
    def _safe_ttl_values(
        ttl_stats: Any,
    ):
        """
        Extract numeric TTL values from the dataset TTL structure.

        The source dataset normally provides values such as:

            {
                "A": 300,
                "AAAA": 300,
                "SOA": 1800,
                "MX": 600,
                "NS": 86400,
                "TXT": 300
            }

        Some datasets may instead use keys containing 'TTL'.
        Both formats are supported.
        """

        if not isinstance(ttl_stats, dict):
            return []

        values = []

        for key, value in ttl_stats.items():

            if not isinstance(
                value,
                (int, float),
            ):
                continue

            try:
                ttl = float(value)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if ttl < 0:
                continue

            values.append(ttl)

        return values

    @staticmethod
    def _safe_number(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Safely convert a value to a finite non-negative number.
        """

        try:
            number = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return default

        if number != number:
            return default

        if number == float("inf"):
            return default

        if number == float("-inf"):
            return default

        return max(
            0.0,
            number,
        )

    # ========================================================================
    # MAIN EXTRACTION
    # ========================================================================

    def extract(
        self,
        raw_records: Dict[str, Any],
        ttl_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract routing, TTL, and fast-flux features.

        Args:
            raw_records:
                DNS records such as A, AAAA, NS and CNAME.

            ttl_stats:
                DNS TTL statistics.

        Returns:
            Dictionary containing routing-related DNS features.
        """

        features = {
            "has_a_records": False,
            "has_aaaa_records": False,
            "has_cname_record": False,
            "a_record_count": 0,
            "aaaa_record_count": 0,
            "ns_count": 0,
            "cname_count": 0,
            "total_resolved_ips": 0,
            "min_ttl_value": 0,
            "max_ttl_value": 0,
            "avg_ttl_value": 0.0,
            "is_fast_flux_candidate": False,
            "is_active_fast_flux": False,
            "has_routing_anomalies": False,
        }

        # ====================================================================
        # NORMALIZE INPUT
        # ====================================================================

        if not isinstance(
            raw_records,
            dict,
        ):
            raw_records = {}

        if not isinstance(
            ttl_stats,
            dict,
        ):
            ttl_stats = {}

        try:

            # ================================================================
            # 1. DNS RECORD COLLECTIONS
            # ================================================================

            a_records = self._safe_collection(
                raw_records.get("A")
            )

            aaaa_records = self._safe_collection(
                raw_records.get("AAAA")
            )

            cname_records = self._safe_collection(
                raw_records.get("CNAME")
            )

            ns_records = self._safe_collection(
                raw_records.get("NS")
            )

            # ================================================================
            # 2. RECORD COUNTS
            # ================================================================

            features["a_record_count"] = len(
                a_records
            )

            features["aaaa_record_count"] = len(
                aaaa_records
            )

            features["cname_count"] = len(
                cname_records
            )

            features["ns_count"] = len(
                ns_records
            )

            features["has_a_records"] = (
                features["a_record_count"] > 0
            )

            features["has_aaaa_records"] = (
                features["aaaa_record_count"] > 0
            )

            features["has_cname_record"] = (
                features["cname_count"] > 0
            )

            # ================================================================
            # 3. RESOLVED IP COUNT
            # ================================================================

            features["total_resolved_ips"] = (
                features["a_record_count"]
                + features["aaaa_record_count"]
            )

            # ================================================================
            # 4. ROUTING ANOMALY
            # ================================================================

            # A domain normally needs either an A/AAAA record or a CNAME
            # to provide a normal resolution path.
            if not (
                features["has_a_records"]
                or features["has_aaaa_records"]
                or features["has_cname_record"]
            ):
                features["has_routing_anomalies"] = True

            # ================================================================
            # 5. CDN DETECTION
            # ================================================================

            is_cdn = False

            for nameserver in ns_records:

                nameserver_text = str(
                    nameserver
                ).lower()

                for cdn in self.known_cdns:

                    if cdn in nameserver_text:
                        is_cdn = True
                        break

                if is_cdn:
                    break

            # ================================================================
            # 6. TTL ANALYSIS
            # ================================================================

            ttl_values = self._safe_ttl_values(
                ttl_stats
            )

            if ttl_values:

                min_ttl = min(
                    ttl_values
                )

                max_ttl = max(
                    ttl_values
                )

                avg_ttl = statistics.mean(
                    ttl_values
                )

                features["min_ttl_value"] = int(
                    min_ttl
                )

                features["max_ttl_value"] = int(
                    max_ttl
                )

                features["avg_ttl_value"] = round(
                    avg_ttl,
                    2,
                )

                # ============================================================
                # 7. FAST-FLUX DETECTION
                # ============================================================

                if is_cdn:

                    # Low TTL is common with CDNs and should not by itself
                    # create a fast-flux signal.
                    features[
                        "is_fast_flux_candidate"
                    ] = False

                    features[
                        "is_active_fast_flux"
                    ] = False

                elif (
                    min_ttl
                    < self.fast_flux_ttl_threshold
                ):

                    features[
                        "is_fast_flux_candidate"
                    ] = True

                    if (
                        features[
                            "total_resolved_ips"
                        ]
                        >= self.fast_flux_ip_threshold
                    ):

                        features[
                            "is_active_fast_flux"
                        ] = True

            # ================================================================
            # 8. FINAL TYPE NORMALIZATION
            # ================================================================

            features["a_record_count"] = int(
                self._safe_number(
                    features["a_record_count"]
                )
            )

            features["aaaa_record_count"] = int(
                self._safe_number(
                    features["aaaa_record_count"]
                )
            )

            features["ns_count"] = int(
                self._safe_number(
                    features["ns_count"]
                )
            )

            features["cname_count"] = int(
                self._safe_number(
                    features["cname_count"]
                )
            )

            features["total_resolved_ips"] = int(
                self._safe_number(
                    features["total_resolved_ips"]
                )
            )

            features["min_ttl_value"] = int(
                self._safe_number(
                    features["min_ttl_value"]
                )
            )

            features["max_ttl_value"] = int(
                self._safe_number(
                    features["max_ttl_value"]
                )
            )

            features["avg_ttl_value"] = round(
                self._safe_number(
                    features["avg_ttl_value"]
                ),
                2,
            )

        except Exception as exc:

            logger.error(
                "Error during DNSRecordsFeatureExtraction: %s",
                exc,
            )

            # Return a complete neutral feature block instead of returning
            # partially corrupted data.
            return {
                "has_a_records": False,
                "has_aaaa_records": False,
                "has_cname_record": False,
                "a_record_count": 0,
                "aaaa_record_count": 0,
                "ns_count": 0,
                "cname_count": 0,
                "total_resolved_ips": 0,
                "min_ttl_value": 0,
                "max_ttl_value": 0,
                "avg_ttl_value": 0.0,
                "is_fast_flux_candidate": False,
                "is_active_fast_flux": False,
                "has_routing_anomalies": True,
            }

        return features