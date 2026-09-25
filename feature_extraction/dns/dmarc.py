import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class DNSDMARCFeatureExtractor:
    """
    Advanced SPF and DMARC Feature Extractor.

    Produces both the canonical numeric ML features required by the
    DNS feature schema and internal categorical values used for
    explainability.

    Canonical ML features:
        has_spf_record
        spf_record_count
        has_multiple_spf_records
        spf_includes_count
        spf_strictness_score
        has_dmarc_record
        dmarc_policy_score
        dmarc_subdomain_policy_score
        is_email_spoofable
    """

    def __init__(self):
        self.spf_pattern = re.compile(
            r"^v=spf1\b",
            re.IGNORECASE,
        )

        self.dmarc_pattern = re.compile(
            r"^v=dmarc1\b",
            re.IGNORECASE,
        )

        self.dmarc_policy_regex = re.compile(
            r"\bp=(none|quarantine|reject)\b",
            re.IGNORECASE,
        )

        self.dmarc_sub_policy_regex = re.compile(
            r"\bsp=(none|quarantine|reject)\b",
            re.IGNORECASE,
        )

        # Numeric policy scores.
        #
        # Higher score = stronger anti-spoofing policy.
        self.spf_scores = {
            "none": 0.0,
            "permissive": 25.0,
            "neutral": 50.0,
            "strict": 100.0,
            "dangerously_permissive": 0.0,
            "invalid": 0.0,
        }

        self.dmarc_scores = {
            "none": 0.0,
            "quarantine": 75.0,
            "reject": 100.0,
        }

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_txt_records(
        txt_records: Any,
    ) -> List[str]:
        """
        Normalize TXT records safely.

        The source dataset can contain:
            None
            list
            tuple
            set
            string
            unexpected values

        Always returns a list.
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
            return [
                str(record)
                for record in txt_records
                if record is not None
            ]

        return []

    @staticmethod
    def _clean_record(
        record: Any,
    ) -> str:
        """
        Normalize a TXT record for matching.
        """

        return (
            str(record)
            .strip()
            .strip('"')
            .strip("'")
            .lower()
        )

    # ========================================================================
    # MAIN EXTRACTION
    # ========================================================================

    def extract(
        self,
        txt_records: List[str],
    ) -> Dict[str, Any]:
        """
        Extract SPF and DMARC features.

        Returns the canonical numeric ML features together with
        categorical diagnostic fields.
        """

        features = {
            # --------------------------------------------------------------
            # SPF
            # --------------------------------------------------------------

            "has_spf_record": False,
            "spf_record_count": 0,
            "has_multiple_spf_records": False,
            "spf_includes_count": 0,
            "spf_strictness_score": 0.0,

            # Diagnostic categorical value
            "spf_strictness": "none",

            # --------------------------------------------------------------
            # DMARC
            # --------------------------------------------------------------

            "has_dmarc_record": False,
            "dmarc_policy_score": 0.0,
            "dmarc_subdomain_policy_score": 0.0,

            # Diagnostic categorical values
            "dmarc_policy": "none",
            "dmarc_subdomain_policy": "none",

            # --------------------------------------------------------------
            # Email spoofing
            # --------------------------------------------------------------

            "is_email_spoofable": True,

            # Diagnostic posture field
            "domain_security_posture_weakness": True,
        }

        normalized_txt_records = (
            self._normalize_txt_records(
                txt_records
            )
        )

        if not normalized_txt_records:
            return features

        try:

            # =================================================================
            # SPF / DMARC DETECTION
            # =================================================================

            for record in normalized_txt_records:

                record_clean = self._clean_record(
                    record
                )

                if not record_clean:
                    continue

                # -------------------------------------------------------------
                # SPF
                # -------------------------------------------------------------

                if self.spf_pattern.match(
                    record_clean
                ):

                    features[
                        "has_spf_record"
                    ] = True

                    features[
                        "spf_record_count"
                    ] += 1

                    features[
                        "spf_includes_count"
                    ] += record_clean.count(
                        "include:"
                    )

                    # Evaluate SPF terminating mechanism.
                    #
                    # Most restrictive / strongest value is retained
                    # if multiple mechanisms appear.
                    if "+all" in record_clean:

                        features[
                            "spf_strictness"
                        ] = "dangerously_permissive"

                    elif "?all" in record_clean:

                        if features[
                            "spf_strictness"
                        ] not in {
                            "dangerously_permissive",
                        }:
                            features[
                                "spf_strictness"
                            ] = "permissive"

                    elif "~all" in record_clean:

                        if features[
                            "spf_strictness"
                        ] not in {
                            "dangerously_permissive",
                            "permissive",
                        }:
                            features[
                                "spf_strictness"
                            ] = "neutral"

                    elif "-all" in record_clean:

                        if features[
                            "spf_strictness"
                        ] == "none":
                            features[
                                "spf_strictness"
                            ] = "strict"

                # -------------------------------------------------------------
                # DMARC
                # -------------------------------------------------------------

                if self.dmarc_pattern.match(
                    record_clean
                ):

                    features[
                        "has_dmarc_record"
                    ] = True

                    policy_match = (
                        self.dmarc_policy_regex.search(
                            record_clean
                        )
                    )

                    if policy_match:

                        features[
                            "dmarc_policy"
                        ] = (
                            policy_match.group(
                                1
                            ).lower()
                        )

                    sub_policy_match = (
                        self.dmarc_sub_policy_regex.search(
                            record_clean
                        )
                    )

                    if sub_policy_match:

                        features[
                            "dmarc_subdomain_policy"
                        ] = (
                            sub_policy_match.group(
                                1
                            ).lower()
                        )

                    else:

                        # DMARC RFC behavior:
                        # when sp= is absent, the organizational
                        # policy defaults to p=.
                        features[
                            "dmarc_subdomain_policy"
                        ] = features[
                            "dmarc_policy"
                        ]

            # =================================================================
            # SPF MULTIPLE RECORD VALIDATION
            # =================================================================

            if features[
                "spf_record_count"
            ] > 1:

                features[
                    "has_multiple_spf_records"
                ] = True

                # Multiple SPF records are invalid.
                features[
                    "spf_strictness"
                ] = "invalid"

            # =================================================================
            # NUMERIC SPF SCORE
            # =================================================================

            features[
                "spf_strictness_score"
            ] = self.spf_scores.get(
                features["spf_strictness"],
                0.0,
            )

            # =================================================================
            # NUMERIC DMARC SCORES
            # =================================================================

            features[
                "dmarc_policy_score"
            ] = self.dmarc_scores.get(
                features["dmarc_policy"],
                0.0,
            )

            features[
                "dmarc_subdomain_policy_score"
            ] = self.dmarc_scores.get(
                features["dmarc_subdomain_policy"],
                0.0,
            )

            # =================================================================
            # EMAIL SPOOFABILITY
            # =================================================================

            strong_dmarc_policy = (
                features["dmarc_policy"]
                in {
                    "quarantine",
                    "reject",
                }
            )

            strong_spf_policy = (
                features["spf_strictness"]
                in {
                    "strict",
                    "neutral",
                }
            )

            valid_single_spf = (
                features["has_spf_record"]
                and not features[
                    "has_multiple_spf_records"
                ]
            )

            if (
                features["has_dmarc_record"]
                and strong_dmarc_policy
                and valid_single_spf
                and strong_spf_policy
            ):

                features[
                    "is_email_spoofable"
                ] = False

                features[
                    "domain_security_posture_weakness"
                ] = False

            # =================================================================
            # FINAL NUMERIC NORMALIZATION
            # =================================================================

            features[
                "spf_record_count"
            ] = int(
                max(
                    0,
                    features[
                        "spf_record_count"
                    ],
                )
            )

            features[
                "spf_includes_count"
            ] = int(
                max(
                    0,
                    features[
                        "spf_includes_count"
                    ],
                )
            )

            features[
                "spf_strictness_score"
            ] = float(
                max(
                    0.0,
                    min(
                        100.0,
                        features[
                            "spf_strictness_score"
                        ],
                    ),
                )
            )

            features[
                "dmarc_policy_score"
            ] = float(
                max(
                    0.0,
                    min(
                        100.0,
                        features[
                            "dmarc_policy_score"
                        ],
                    ),
                )
            )

            features[
                "dmarc_subdomain_policy_score"
            ] = float(
                max(
                    0.0,
                    min(
                        100.0,
                        features[
                            "dmarc_subdomain_policy_score"
                        ],
                    ),
                )
            )

        except Exception as exc:

            logger.error(
                "Error during DNS DMARC Feature Extraction: %s",
                exc,
            )

            # Always return the complete canonical feature block.
            return {
                "has_spf_record": False,
                "spf_record_count": 0,
                "has_multiple_spf_records": False,
                "spf_includes_count": 0,
                "spf_strictness_score": 0.0,
                "spf_strictness": "none",
                "has_dmarc_record": False,
                "dmarc_policy_score": 0.0,
                "dmarc_subdomain_policy_score": 0.0,
                "dmarc_policy": "none",
                "dmarc_subdomain_policy": "none",
                "is_email_spoofable": True,
                "domain_security_posture_weakness": True,
            }

        return features