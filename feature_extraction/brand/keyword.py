import logging
import re
from typing import Dict, Any, List, Pattern

logger = logging.getLogger(__name__)


class BrandKeywordFeatureExtractor:
    """
    Brand semantic / social-engineering keyword feature extractor.

    This component is responsible only for extracting keyword-based
    behavioral signals. It does NOT create the complete Brand ML vector.

    Output is intentionally stable so that feature_extraction.brand.extractor
    can later map these signals into the canonical BrandFeaturesSchema.
    """

    # ------------------------------------------------------------------
    # CATEGORY DEFINITIONS
    # ------------------------------------------------------------------

    CREDENTIAL_KEYWORDS = (
        "login",
        "sign-in",
        "sign in",
        "signin",
        "authenticate",
        "password",
        "passcode",
        "credentials",
        "verify identity",
        "confirm identity",
        "unlock account",
        "security question",
    )

    URGENCY_KEYWORDS = (
        "urgent",
        "suspend",
        "suspended",
        "restricted",
        "blocked",
        "unauthorized access",
        "immediate action required",
        "expires",
        "validate now",
        "alert",
        "warning",
        "compromised",
        "deactivated",
    )

    FINANCIAL_ADMIN_KEYWORDS = (
        "banking",
        "billing",
        "invoice",
        "payment",
        "transaction",
        "refund",
        "wallet",
        "crypto",
        "fund",
        "statement",
        "routing",
        "account update",
        "service desk",
    )

    DEFAULT_FEATURES = {
        "total_suspicious_keyword_count": 0,
        "has_credential_solicitation": False,
        "has_urgency_or_threat": False,
        "has_financial_admin_context": False,
        "matched_keywords": [],
        "semantic_threat_score": 0.0,
    }

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    def __init__(self):
        self.credential_keywords = list(
            self.CREDENTIAL_KEYWORDS
        )

        self.urgency_keywords = list(
            self.URGENCY_KEYWORDS
        )

        self.financial_admin_keywords = list(
            self.FINANCIAL_ADMIN_KEYWORDS
        )

        self.cred_patterns = (
            self._build_patterns(
                self.credential_keywords
            )
        )

        self.urgency_patterns = (
            self._build_patterns(
                self.urgency_keywords
            )
        )

        self.admin_patterns = (
            self._build_patterns(
                self.financial_admin_keywords
            )
        )

    # ------------------------------------------------------------------
    # PATTERN BUILDING
    # ------------------------------------------------------------------

    @staticmethod
    def _build_patterns(
        keywords: List[str],
    ) -> List[Pattern]:
        """
        Build case-insensitive regular expressions.

        Word boundaries are retained so that words such as "fund" do not
        incorrectly match inside unrelated words such as "fundamental".
        """

        patterns = []

        for keyword in keywords:

            escaped = re.escape(
                keyword
            )

            patterns.append(
                re.compile(
                    rf"\b{escaped}\b",
                    re.IGNORECASE,
                )
            )

        return patterns

    # ------------------------------------------------------------------
    # DEFAULTS
    # ------------------------------------------------------------------

    @classmethod
    def get_default_features(
        cls,
    ) -> Dict[str, Any]:
        """
        Return a fresh default keyword feature block.
        """

        defaults = dict(
            cls.DEFAULT_FEATURES
        )

        defaults[
            "matched_keywords"
        ] = []

        return defaults

    # ------------------------------------------------------------------
    # TEXT NORMALIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(
        value: Any,
    ) -> str:
        """
        Safely normalize arbitrary text input.
        """

        if value is None:
            return ""

        try:
            return str(
                value
            ).strip()
        except Exception:
            return ""

    @staticmethod
    def _combine_text(
        page_title: Any,
        ocr_text: Any,
    ) -> str:
        """
        Combine HTML title and OCR evidence.
        """

        title = BrandKeywordFeatureExtractor._normalize_text(
            page_title
        )

        ocr = BrandKeywordFeatureExtractor._normalize_text(
            ocr_text
        )

        if title and ocr:
            return f"{title} {ocr}"

        return title or ocr

    # ------------------------------------------------------------------
    # MATCHING
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_keyword(
        pattern: Pattern,
    ) -> str:
        """
        Convert a compiled pattern back into its human-readable keyword.
        """

        value = pattern.pattern

        value = value.replace(
            r"\b",
            "",
        )

        try:
            value = re.sub(
                r"\\(.)",
                r"\1",
                value,
            )
        except Exception:
            pass

        return value

    def _find_matches(
        self,
        patterns: List[Pattern],
        text: str,
    ) -> List[str]:
        """
        Return all unique keyword matches for a category.
        """

        matches = []

        for pattern in patterns:

            try:
                if pattern.search(text):

                    keyword = self._clean_keyword(
                        pattern
                    )

                    if keyword not in matches:
                        matches.append(
                            keyword
                        )

            except Exception as exc:
                logger.debug(
                    "Keyword pattern failed: %s",
                    exc,
                )

        return matches

    # ------------------------------------------------------------------
    # THREAT SCORE
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_threat_score(
        credential_hit: bool,
        urgency_hit: bool,
        financial_hit: bool,
    ) -> float:
        """
        Calculate a deterministic semantic threat score.

        Base weights:
            Credential = 40
            Urgency    = 40
            Financial  = 20

        Credential + urgency combination receives a 20-point bonus.

        Final range:
            0.0 - 100.0
        """

        score = 0.0

        if credential_hit:
            score += 40.0

        if urgency_hit:
            score += 40.0

        if financial_hit:
            score += 20.0

        if (
            credential_hit
            and urgency_hit
        ):
            score += 20.0

        return min(
            100.0,
            score,
        )

    # ------------------------------------------------------------------
    # MAIN EXTRACTION
    # ------------------------------------------------------------------

    def extract(
        self,
        page_title: str = "",
        ocr_text: str = "",
    ) -> Dict[str, Any]:
        """
        Extract keyword-based Brand features.

        Parameters:
            page_title:
                HTML document title.

            ocr_text:
                Text obtained from screenshot/OCR.

        Returns:
            Stable keyword feature dictionary.
        """

        features = (
            self.get_default_features()
        )

        combined_text = (
            self._combine_text(
                page_title,
                ocr_text,
            )
        )

        if not combined_text:
            return features

        try:

            credential_matches = (
                self._find_matches(
                    self.cred_patterns,
                    combined_text,
                )
            )

            urgency_matches = (
                self._find_matches(
                    self.urgency_patterns,
                    combined_text,
                )
            )

            financial_matches = (
                self._find_matches(
                    self.admin_patterns,
                    combined_text,
                )
            )

            # ----------------------------------------------------------
            # Combine matches deterministically
            # ----------------------------------------------------------

            all_matches = []

            for keyword in (
                credential_matches
                + urgency_matches
                + financial_matches
            ):
                if keyword not in all_matches:
                    all_matches.append(
                        keyword
                    )

            has_credential = (
                len(credential_matches) > 0
            )

            has_urgency = (
                len(urgency_matches) > 0
            )

            has_financial = (
                len(financial_matches) > 0
            )

            # ----------------------------------------------------------
            # Populate output
            # ----------------------------------------------------------

            features.update(
                {
                    "total_suspicious_keyword_count":
                        len(all_matches),

                    "has_credential_solicitation":
                        has_credential,

                    "has_urgency_or_threat":
                        has_urgency,

                    "has_financial_admin_context":
                        has_financial,

                    "matched_keywords":
                        all_matches,

                    "semantic_threat_score":
                        self._calculate_threat_score(
                            credential_hit=has_credential,
                            urgency_hit=has_urgency,
                            financial_hit=has_financial,
                        ),
                }
            )

            return features

        except Exception as exc:

            logger.exception(
                "Brand keyword extraction failed: %s",
                exc,
            )

            return (
                self.get_default_features()
            )

    # ------------------------------------------------------------------
    # BATCH EXTRACTION
    # ------------------------------------------------------------------

    def extract_from_text(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """
        Convenience method for callers that have only one text source.
        """

        return self.extract(
            page_title="",
            ocr_text=text,
        )