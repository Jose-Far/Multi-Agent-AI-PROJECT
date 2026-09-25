import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


class BrandSimilarityFeatureExtractor:
    """
    Brand similarity feature extractor.

    Produces deterministic similarity signals used by the Brand feature
    pipeline. This extractor does not perform ML prediction.

    Output features:
        1. title_brand_distance_score
        2. domain_brand_levenshtein_ratio
        3. domain_brand_jaccard_score
        4. is_brand_name_in_title
        5. is_brand_name_in_domain_tokens
        6. brand_distance_anomaly_detected
    """

    # ------------------------------------------------------------------
    # CANONICAL FEATURES
    # ------------------------------------------------------------------

    FEATURE_NAMES = [
        "title_brand_distance_score",
        "domain_brand_levenshtein_ratio",
        "domain_brand_jaccard_score",
        "is_brand_name_in_title",
        "is_brand_name_in_domain_tokens",
        "brand_distance_anomaly_detected",
    ]

    FEATURE_COUNT = 6

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    def __init__(self):
        self.token_split_regex = re.compile(
            r"[-_.:/\\@]+"
        )

    # ------------------------------------------------------------------
    # DEFAULT FEATURES
    # ------------------------------------------------------------------

    @classmethod
    def get_default_features(
        cls,
    ) -> Dict[str, Any]:
        """
        Return a fresh default feature dictionary.
        """

        return {
            "title_brand_distance_score": 0.0,
            "domain_brand_levenshtein_ratio": 0.0,
            "domain_brand_jaccard_score": 0.0,
            "is_brand_name_in_title": False,
            "is_brand_name_in_domain_tokens": False,
            "brand_distance_anomaly_detected": False,
        }

    # ------------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(
        value: Any,
    ) -> str:
        """
        Safely normalize arbitrary input to lowercase text.
        """

        if value is None:
            return ""

        try:
            return str(value).strip().lower()
        except Exception:
            return ""

    @staticmethod
    def _normalize_brand(
        matched_brand: Any,
    ) -> str:
        """
        Normalize the matched brand name.

        Examples:
            'Amazon' -> 'amazon'
            'MICROSOFT' -> 'microsoft'
            'unknown' -> ''
        """

        brand = (
            BrandSimilarityFeatureExtractor
            ._normalize_text(
                matched_brand
            )
        )

        if brand in {
            "",
            "unknown",
            "none",
            "null",
            "nan",
        }:
            return ""

        return brand

    # ------------------------------------------------------------------
    # DOMAIN NORMALIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_domain_core(
        domain: Any,
    ) -> str:
        """
        Extract the hostname/core portion from a domain or URL.

        Examples:
            https://amazon-login.com
                -> amazon-login

            amazon-login.com
                -> amazon-login

            amazon-login
                -> amazon-login
        """

        value = (
            BrandSimilarityFeatureExtractor
            ._normalize_text(
                domain
            )
        )

        if not value:
            return ""

        # Remove protocol.
        value = re.sub(
            r"^[a-z][a-z0-9+.-]*://",
            "",
            value,
        )

        # Remove credentials.
        value = value.split("@")[-1]

        # Remove path/query/fragment.
        value = re.split(
            r"[/?#]",
            value,
            maxsplit=1,
        )[0]

        # Remove port.
        value = re.sub(
            r":\d+$",
            "",
            value,
        )

        # Remove common www prefix.
        value = re.sub(
            r"^www\.",
            "",
            value,
        )

        # Remove trailing dot.
        value = value.rstrip(".")

        # Extract first meaningful hostname label.
        parts = [
            part
            for part in value.split(".")
            if part
        ]

        if not parts:
            return ""

        # For normal domains use the registrable-looking first label.
        return parts[0]

    # ------------------------------------------------------------------
    # LEVENSHTEIN / SEQUENCE SIMILARITY
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_levenshtein_similarity(
        str1: str,
        str2: str,
    ) -> float:
        """
        Calculate normalized sequence similarity.

        Returns:
            0.0 - 1.0
        """

        first = (
            BrandSimilarityFeatureExtractor
            ._normalize_text(str1)
        )

        second = (
            BrandSimilarityFeatureExtractor
            ._normalize_text(str2)
        )

        if not first or not second:
            return 0.0

        try:

            return round(
                SequenceMatcher(
                    None,
                    first,
                    second,
                ).ratio(),
                4,
            )

        except Exception as exc:

            logger.debug(
                "Similarity calculation failed: %s",
                exc,
            )

            return 0.0

    # ------------------------------------------------------------------
    # CHARACTER N-GRAMS
    # ------------------------------------------------------------------

    @staticmethod
    def _get_ngrams(
        word: str,
        n: int = 2,
    ) -> Set[str]:
        """
        Generate normalized character n-grams.
        """

        value = (
            BrandSimilarityFeatureExtractor
            ._normalize_text(word)
        )

        if not value:
            return set()

        if len(value) < n:
            return {
                value
            }

        return {
            value[index:index + n]
            for index in range(
                len(value) - n + 1
            )
        }

    # ------------------------------------------------------------------
    # JACCARD
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_jaccard_token_similarity(
        text: str,
        brand: str,
    ) -> float:
        """
        Calculate character bigram Jaccard similarity.

        Returns:
            0.0 - 1.0
        """

        first = (
            BrandSimilarityFeatureExtractor
            ._normalize_text(text)
        )

        second = (
            BrandSimilarityFeatureExtractor
            ._normalize_text(brand)
        )

        if not first or not second:
            return 0.0

        set1 = (
            BrandSimilarityFeatureExtractor
            ._get_ngrams(
                first,
                n=2,
            )
        )

        set2 = (
            BrandSimilarityFeatureExtractor
            ._get_ngrams(
                second,
                n=2,
            )
        )

        if not set1 or not set2:
            return 0.0

        union = set1 | set2

        if not union:
            return 0.0

        intersection = set1 & set2

        return round(
            len(intersection)
            / len(union),
            4,
        )

    # ------------------------------------------------------------------
    # DOMAIN TOKENIZATION
    # ------------------------------------------------------------------

    def _tokenize_domain(
        self,
        domain_core: str,
    ) -> List[str]:
        """
        Split a domain into meaningful tokens.
        """

        if not domain_core:
            return []

        try:

            tokens = (
                self.token_split_regex
                .split(domain_core)
            )

            return [
                token.strip().lower()
                for token in tokens
                if token.strip()
            ]

        except Exception as exc:

            logger.debug(
                "Domain tokenization failed: %s",
                exc,
            )

            return []

    # ------------------------------------------------------------------
    # TITLE MATCHING
    # ------------------------------------------------------------------

    @staticmethod
    def _brand_in_title(
        page_title: str,
        matched_brand: str,
    ) -> bool:
        """
        Determine whether the brand occurs in the page title.

        Uses normalized substring matching because titles commonly contain
        phrases such as:
            'Microsoft Account Login'
        """

        title = (
            BrandSimilarityFeatureExtractor
            ._normalize_text(
                page_title
            )
        )

        brand = (
            BrandSimilarityFeatureExtractor
            ._normalize_brand(
                matched_brand
            )
        )

        if not title or not brand:
            return False

        return brand in title

    # ------------------------------------------------------------------
    # DOMAIN TOKEN MATCHING
    # ------------------------------------------------------------------

    def _brand_in_domain_tokens(
        self,
        domain_core: str,
        matched_brand: str,
    ) -> bool:
        """
        Determine whether the brand appears as an independent domain token.
        """

        brand = (
            self._normalize_brand(
                matched_brand
            )
        )

        if not brand:
            return False

        tokens = (
            self._tokenize_domain(
                domain_core
            )
        )

        return brand in tokens

    # ------------------------------------------------------------------
    # ANOMALY DETECTION
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_distance_anomaly(
        levenshtein_ratio: float,
        domain_core: str,
        matched_brand: str,
    ) -> bool:
        """
        Detect likely typo/domain impersonation.

        A domain that is very similar to a brand but not exactly equal is
        treated as suspicious.

        Examples:
            amaz0n     -> suspicious
            amazonn    -> suspicious
            amazon     -> not anomalous
        """

        domain = (
            BrandSimilarityFeatureExtractor
            ._normalize_text(
                domain_core
            )
        )

        brand = (
            BrandSimilarityFeatureExtractor
            ._normalize_brand(
                matched_brand
            )
        )

        if not domain or not brand:
            return False

        if domain == brand:
            return False

        return (
            levenshtein_ratio >= 0.75
            and levenshtein_ratio < 1.0
        )

    # ------------------------------------------------------------------
    # MAIN EXTRACTION
    # ------------------------------------------------------------------

    def extract(
        self,
        domain: str,
        page_title: str,
        matched_brand: str,
    ) -> Dict[str, Any]:
        """
        Extract the canonical Brand similarity feature block.

        Parameters:
            domain:
                Target domain or URL.

            page_title:
                HTML page title.

            matched_brand:
                Brand selected by the Brand matching layer.

        Returns:
            Exactly six canonical similarity features.
        """

        features = (
            self.get_default_features()
        )

        try:

            brand = (
                self._normalize_brand(
                    matched_brand
                )
            )

            if not brand:
                return features

            domain_core = (
                self._extract_domain_core(
                    domain
                )
            )

            title = (
                self._normalize_text(
                    page_title
                )
            )

            if not domain_core and not title:
                return features

            # ----------------------------------------------------------
            # Title signal
            # ----------------------------------------------------------

            title_match = (
                self._brand_in_title(
                    title,
                    brand,
                )
            )

            # ----------------------------------------------------------
            # Domain token signal
            # ----------------------------------------------------------

            domain_token_match = (
                self._brand_in_domain_tokens(
                    domain_core,
                    brand,
                )
            )

            # ----------------------------------------------------------
            # String similarity
            # ----------------------------------------------------------

            levenshtein_ratio = (
                self.calculate_levenshtein_similarity(
                    domain_core,
                    brand,
                )
            )

            jaccard_score = (
                self.calculate_jaccard_token_similarity(
                    domain_core,
                    brand,
                )
            )

            # ----------------------------------------------------------
            # Anomaly
            # ----------------------------------------------------------

            anomaly = (
                self._detect_distance_anomaly(
                    levenshtein_ratio,
                    domain_core,
                    brand,
                )
            )

            # ----------------------------------------------------------
            # Populate canonical features
            # ----------------------------------------------------------

            features.update(
                {
                    "title_brand_distance_score":
                        levenshtein_ratio,

                    "domain_brand_levenshtein_ratio":
                        levenshtein_ratio,

                    "domain_brand_jaccard_score":
                        jaccard_score,

                    "is_brand_name_in_title":
                        title_match,

                    "is_brand_name_in_domain_tokens":
                        domain_token_match,

                    "brand_distance_anomaly_detected":
                        anomaly,
                }
            )

            return features

        except Exception as exc:

            logger.exception(
                "Brand similarity extraction failed: %s",
                exc,
            )

            return (
                self.get_default_features()
            )

    # ------------------------------------------------------------------
    # SCHEMA VALIDATION
    # ------------------------------------------------------------------

    @classmethod
    def validate_features(
        cls,
        features: Dict[str, Any],
    ) -> bool:
        """
        Verify that the similarity extractor returned exactly the expected
        feature names.
        """

        if not isinstance(
            features,
            dict,
        ):
            return False

        return (
            set(features.keys())
            == set(cls.FEATURE_NAMES)
        )

    # ------------------------------------------------------------------
    # FEATURE COUNT
    # ------------------------------------------------------------------

    @classmethod
    def get_feature_count(
        cls,
    ) -> int:
        """
        Return canonical feature count.
        """

        return cls.FEATURE_COUNT