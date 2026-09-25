import logging
import re
import base64
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class DNSTXTFeatureExtractor:
    """
    DNS TXT Feature Extractor.

    Extracts the canonical TXT features required by the DNS ML schema:

        has_txt_records
        txt_record_count
        has_domain_verification
        has_suspicious_long_txt
        has_base64_payload_in_txt
        avg_txt_length

    SPF detection is also retained because SPF information is present
    in DNS TXT records and is consumed by the SPF/DMARC extractor.

    Important:
        Legitimate domain-verification tokens should not automatically
        be classified as suspicious Base64 payloads.
    """

    def __init__(self):

        # ================================================================
        # DOMAIN VERIFICATION SIGNATURES
        # ================================================================

        self.verification_signatures = {
            "google-site-verification": "google",
            "facebook-domain-verification": "facebook",
            "atlassian-domain-verification": "atlassian",
            "apple-domain-verification": "apple",
            "ms=ms": "microsoft",
            "docusign": "docusign",
            "stripe-verification": "stripe",
            "yandex-verification": "yandex",
            "adobe-idp-site-verification": "adobe",
        }

        # ================================================================
        # SUSPICIOUS TXT LENGTH
        # ================================================================

        self.suspicious_txt_length_threshold = 200

        # ================================================================
        # BASE64 CHARACTER PATTERN
        # ================================================================

        self.base64_pattern = re.compile(
            r"^(?:[A-Za-z0-9+/]{4})*"
            r"(?:[A-Za-z0-9+/]{2}==|"
            r"[A-Za-z0-9+/]{3}=|"
            r"[A-Za-z0-9+/]{4})$"
        )

        # ================================================================
        # KNOWN LEGITIMATE TXT TOKEN PREFIXES
        # ================================================================

        self.legitimate_token_prefixes = (
            "google-site-verification=",
            "facebook-domain-verification=",
            "atlassian-domain-verification=",
            "apple-domain-verification=",
            "ms=",
            "docusign=",
            "stripe-verification=",
            "yandex-verification=",
            "adobe-idp-site-verification=",
            "firebase=",
            "globalsign-domain-verification=",
            "comodoca.com",
        )

    # ====================================================================
    # NORMALIZATION
    # ====================================================================

    @staticmethod
    def _normalize_txt_records(
        txt_records: Any,
    ) -> List[str]:
        """
        Normalize TXT records into a list of strings.

        Supports:
            None
            list
            tuple
            set
            string
        """

        if txt_records is None:
            return []

        if isinstance(
            txt_records,
            str,
        ):

            value = txt_records.strip()

            if not value:
                return []

            return [value]

        if isinstance(
            txt_records,
            (list, tuple, set),
        ):

            normalized = []

            for record in txt_records:

                if record is None:
                    continue

                value = str(
                    record
                ).strip()

                if value:
                    normalized.append(
                        value
                    )

            return normalized

        return []

    # ====================================================================
    # DOMAIN VERIFICATION DETECTION
    # ====================================================================

    def _detect_domain_verification(
        self,
        record_lower: str,
    ) -> List[Dict[str, Any]]:
        """
        Detect known legitimate domain-verification providers.

        Only sanitized provider metadata is returned.
        """

        detected = []

        for (
            signature,
            provider,
        ) in self.verification_signatures.items():

            if signature in record_lower:

                entry = {
                    "record_type": "verification",
                    "provider": provider,
                    "present": True,
                }

                if entry not in detected:
                    detected.append(
                        entry
                    )

        return detected

    # ====================================================================
    # BASE64 DETECTION
    # ====================================================================

    def _looks_like_suspicious_base64(
        self,
        record: str,
        record_lower: str,
    ) -> bool:
        """
        Determine whether a TXT record looks like a suspicious
        standalone Base64 payload.

        A record is NOT considered suspicious merely because it is
        Base64-decodable.

        Legitimate verification records are excluded.
        SPF and common structured DNS records are excluded.
        """

        value = record.strip()

        if len(value) < 64:
            return False

        # ---------------------------------------------------------------
        # Legitimate verification / structured tokens
        # ---------------------------------------------------------------

        for prefix in self.legitimate_token_prefixes:

            if record_lower.startswith(
                prefix.lower()
            ):
                return False

        # ---------------------------------------------------------------
        # SPF is structured DNS policy, not a payload
        # ---------------------------------------------------------------

        if record_lower.startswith(
            "v=spf1"
        ):
            return False

        # ---------------------------------------------------------------
        # DMARC is structured DNS policy
        # ---------------------------------------------------------------

        if record_lower.startswith(
            "v=dmarc1"
        ):
            return False

        # ---------------------------------------------------------------
        # DKIM records are legitimate structured TXT records
        # ---------------------------------------------------------------

        if (
            "v=dkim1" in record_lower
            or "p=" in record_lower
        ):
            return False

        # ---------------------------------------------------------------
        # Domain verification tokens
        # ---------------------------------------------------------------

        for (
            signature,
            _provider,
        ) in self.verification_signatures.items():

            if signature in record_lower:
                return False

        # ---------------------------------------------------------------
        # Must actually match Base64 structure
        # ---------------------------------------------------------------

        if not self.base64_pattern.fullmatch(
            value
        ):
            return False

        # ---------------------------------------------------------------
        # Validate Base64 decoding
        # ---------------------------------------------------------------

        try:

            decoded = base64.b64decode(
                value,
                validate=True,
            )

        except (
            ValueError,
            TypeError,
        ):

            return False

        if not decoded:
            return False

        # ---------------------------------------------------------------
        # Require meaningful encoded content
        # ---------------------------------------------------------------

        # Very short decoded payloads are unlikely to represent
        # meaningful encoded content.
        if len(decoded) < 32:
            return False

        return True

    # ====================================================================
    # MAIN EXTRACTION
    # ====================================================================

    def extract(
        self,
        txt_records: Any,
    ) -> Dict[str, Any]:
        """
        Extract TXT security features.

        Returns a sanitized feature dictionary.
        """

        features = {

            "has_txt_records": False,

            "txt_record_count": 0,

            "has_domain_verification": False,

            "has_spf_record": False,

            "has_suspicious_long_txt": False,

            "has_base64_payload_in_txt": False,

            "avg_txt_length": 0.0,

            "domain_verifications": [],
        }

        normalized_records = (
            self._normalize_txt_records(
                txt_records
            )
        )

        if not normalized_records:
            return features

        try:

            # ============================================================
            # BASIC TXT FEATURES
            # ============================================================

            features[
                "has_txt_records"
            ] = True

            features[
                "txt_record_count"
            ] = len(
                normalized_records
            )

            total_length = 0

            detected_verifications = []

            # ============================================================
            # PROCESS RECORDS
            # ============================================================

            for record in normalized_records:

                record_str = (
                    record
                    .strip()
                    .strip('"')
                    .strip("'")
                )

                if not record_str:
                    continue

                record_lower = (
                    record_str.lower()
                )

                total_length += len(
                    record_str
                )

                # ========================================================
                # DOMAIN VERIFICATION
                # ========================================================

                verification_matches = (
                    self._detect_domain_verification(
                        record_lower
                    )
                )

                if verification_matches:

                    features[
                        "has_domain_verification"
                    ] = True

                    for entry in verification_matches:

                        if entry not in detected_verifications:

                            detected_verifications.append(
                                entry
                            )

                # ========================================================
                # SPF
                # ========================================================

                if record_lower.startswith(
                    "v=spf1"
                ):

                    features[
                        "has_spf_record"
                    ] = True

                # ========================================================
                # SUSPICIOUS LONG TXT
                # ========================================================

                if (
                    len(record_str)
                    > self.suspicious_txt_length_threshold
                ):

                    # Structured legitimate DNS policies should not
                    # automatically be marked suspicious.
                    if not (
                        record_lower.startswith(
                            "v=spf1"
                        )
                        or record_lower.startswith(
                            "v=dmarc1"
                        )
                        or "verification" in record_lower
                        or "domain-verification" in record_lower
                    ):

                        features[
                            "has_suspicious_long_txt"
                        ] = True

                # ========================================================
                # SUSPICIOUS BASE64
                # ========================================================

                if self._looks_like_suspicious_base64(
                    record_str,
                    record_lower,
                ):

                    features[
                        "has_base64_payload_in_txt"
                    ] = True

            # ============================================================
            # SANITIZED VERIFICATION METADATA
            # ============================================================

            features[
                "domain_verifications"
            ] = detected_verifications

            # ============================================================
            # AVERAGE TXT LENGTH
            # ============================================================

            if features[
                "txt_record_count"
            ] > 0:

                features[
                    "avg_txt_length"
                ] = round(
                    total_length
                    / features[
                        "txt_record_count"
                    ],
                    2,
                )

        except Exception as exc:

            logger.error(
                "Error during DNS TXT Feature Extraction: %s",
                exc,
                exc_info=True,
            )

        return features