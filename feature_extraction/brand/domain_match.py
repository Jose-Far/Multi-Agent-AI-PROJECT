from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)


class DomainMatchFeatureExtractor:
    """
    Brand-domain matching feature extractor.

    This module is intentionally independent of the 64-feature Brand
    schema. It produces domain-level evidence that the master Brand
    extractor converts into the canonical feature vector.

    Supported evidence:

        matched_brand
        is_brand_in_domain
        domain_brand_similarity
        is_typosquatted
        is_homograph_attack
        brand_embedding_type

        plus useful diagnostic signals consumed by the master extractor:

        has_character_insertion
        has_character_deletion
        has_character_substitution
        has_character_transposition
        has_repeated_character
        exact_domain_match
        brand_in_subdomain
        brand_in_registered_domain
        unicode_domain_detected
        punycode_detected
        confusable_character_count
        typosquat_edit_distance
    """

    # ======================================================================
    # CONSTANTS
    # ======================================================================

    DOMAIN_DELIMITERS = re.compile(r"[-._]+")

    SIMILARITY_THRESHOLD = 0.70

    TYPOSQUAT_THRESHOLD = 0.75

    STRONG_TYPO_THRESHOLD = 0.85

    DEFAULT_FEATURES: Dict[str, Any] = {
        "matched_brand": "unknown",
        "is_brand_in_domain": False,
        "domain_brand_similarity": 0.0,
        "is_typosquatted": False,
        "is_homograph_attack": False,
        "brand_embedding_type": "none",

        "has_character_insertion": False,
        "has_character_deletion": False,
        "has_character_substitution": False,
        "has_character_transposition": False,
        "has_repeated_character": False,

        "exact_domain_match": False,
        "brand_in_subdomain": False,
        "brand_in_registered_domain": False,

        "unicode_domain_detected": False,
        "punycode_detected": False,
        "confusable_character_count": 0,

        "typosquat_edit_distance": 0,
    }

    FEATURE_NAMES = list(
        DEFAULT_FEATURES.keys()
    )

    # The class contains only fallback brands.
    #
    # IMPORTANT:
    # The actual 181-brand targetlist should be supplied through
    # set_brands() or extract(..., brands=...).
    #
    # This prevents the old hard-coded list from limiting detection.
    FALLBACK_BRANDS = {
        "paypal",
        "chase",
        "bankofamerica",
        "wellsfargo",
        "capitalone",
        "stripe",
        "square",
        "coinbase",
        "binance",
        "kraken",
        "metamask",
        "gemini",
        "google",
        "microsoft",
        "apple",
        "facebook",
        "instagram",
        "linkedin",
        "twitter",
        "netflix",
        "amazon",
        "docusign",
        "adobe",
        "dropbox",
        "salesforce",
        "github",
        "gitlab",
    }

    # ======================================================================
    # INITIALIZATION
    # ======================================================================

    def __init__(
        self,
        brands: Optional[Iterable[str]] = None,
    ) -> None:

        self.brands: List[str] = []

        self._brand_lookup: Dict[
            str,
            str,
        ] = {}

        self.set_brands(
            brands
            if brands is not None
            else self.FALLBACK_BRANDS
        )

    # ======================================================================
    # BRAND CONFIGURATION
    # ======================================================================

    @staticmethod
    def _normalize_brand(
        value: Any,
    ) -> str:

        text = str(
            value or ""
        ).strip().lower()

        text = unicodedata.normalize(
            "NFKC",
            text,
        )

        return re.sub(
            r"[^a-z0-9]+",
            "",
            text,
        )

    def set_brands(
        self,
        brands: Iterable[str],
    ) -> None:
        """
        Configure the complete targetlist.

        Example:

            extractor.set_brands(targetlist_brands)

        The original human-readable spelling is preserved in
        _brand_lookup while matching uses normalized strings.
        """

        lookup: Dict[str, str] = {}

        for brand in brands:

            original = str(
                brand or ""
            ).strip()

            normalized = self._normalize_brand(
                original
            )

            if not normalized:
                continue

            if normalized not in lookup:

                lookup[
                    normalized
                ] = original

        self._brand_lookup = lookup

        self.brands = sorted(
            lookup.values(),
            key=str.lower,
        )

    def get_brands(self) -> List[str]:
        """
        Return currently configured brands.
        """

        return list(
            self.brands
        )

    # ======================================================================
    # DEFAULTS
    # ======================================================================

    @classmethod
    def get_default_features(
        cls,
    ) -> Dict[str, Any]:

        return dict(
            cls.DEFAULT_FEATURES
        )

    # ======================================================================
    # URL / DOMAIN NORMALIZATION
    # ======================================================================

    @staticmethod
    def _strip_scheme(
        value: str,
    ) -> str:

        return re.sub(
            r"^[a-zA-Z][a-zA-Z0-9+\-.]*://",
            "",
            str(value).strip(),
        )

    @staticmethod
    def _strip_credentials(
        value: str,
    ) -> str:

        if "@" in value:

            return value.rsplit(
                "@",
                1,
            )[-1]

        return value

    @staticmethod
    def _strip_port(
        value: str,
    ) -> str:

        # IPv6 is not a useful brand-domain input here.
        if value.count(":") > 1:
            return value

        if ":" in value:

            host, port = value.rsplit(
                ":",
                1,
            )

            if port.isdigit():
                return host

        return value

    @staticmethod
    def _normalize_unicode(
        value: str,
    ) -> str:

        try:

            return unicodedata.normalize(
                "NFKC",
                value,
            )

        except Exception:

            return value

    def _normalize_domain(
        self,
        domain: Any,
    ) -> str:

        value = str(
            domain or ""
        ).strip()

        if not value:
            return ""

        value = self._strip_scheme(
            value
        )

        value = self._strip_credentials(
            value
        )

        value = re.split(
            r"[/?#]",
            value,
            maxsplit=1,
        )[0]

        value = self._strip_port(
            value
        )

        value = value.strip(
            " ."
        )

        value = self._normalize_unicode(
            value
        )

        return value.lower()

    # ======================================================================
    # IDNA / HOMOGRAPH
    # ======================================================================

    @staticmethod
    def _contains_punycode(
        domain: str,
    ) -> bool:

        return any(
            label.lower().startswith(
                "xn--"
            )
            for label in domain.split(".")
        )

    @staticmethod
    def _contains_non_ascii(
        domain: str,
    ) -> bool:

        return any(
            ord(character) > 127
            for character in domain
        )

    @staticmethod
    def _decode_idna(
        domain: str,
    ) -> str:

        try:

            return domain.encode(
                "ascii"
            ).decode(
                "idna"
            )

        except Exception:

            return domain

    @staticmethod
    def _count_confusable_characters(
        domain: str,
    ) -> int:

        confusables = {
            "а",  # Cyrillic a
            "е",  # Cyrillic e
            "о",  # Cyrillic o
            "р",  # Cyrillic p
            "с",  # Cyrillic c
            "х",  # Cyrillic x
            "у",  # Cyrillic y
            "і",  # Cyrillic i
            "ј",  # Cyrillic j
            "ӏ",  # Cyrillic palochka
            "Α",  # Greek Alpha
            "Β",  # Greek Beta
            "Ε",  # Greek Epsilon
            "Ο",  # Greek Omicron
            "Ρ",  # Greek Rho
            "Τ",  # Greek Tau
            "Χ",  # Greek Chi
        }

        return sum(
            1
            for char in domain
            if char in confusables
        )

    # ======================================================================
    # DOMAIN STRUCTURE
    # ======================================================================

    @staticmethod
    def _domain_labels(
        domain: str,
    ) -> List[str]:

        return [
            label
            for label in domain.split(".")
            if label
        ]

    @staticmethod
    def _registrable_index(
        labels: List[str],
    ) -> int:

        if not labels:
            return -1

        if len(labels) == 1:
            return 0

        # Common two-level public suffix patterns.
        if (
            len(labels) >= 3
            and len(labels[-1]) == 2
            and labels[-2] in {
                "co",
                "com",
                "net",
                "org",
                "gov",
                "ac",
                "edu",
            }
        ):

            return len(labels) - 3

        return len(labels) - 2

    @classmethod
    def _get_registered_domain(
        cls,
        domain: str,
    ) -> str:

        labels = cls._domain_labels(
            domain
        )

        index = cls._registrable_index(
            labels
        )

        if index < 0:
            return ""

        return ".".join(
            labels[index:]
        )

    @classmethod
    def _get_core_domain(
        cls,
        domain: str,
    ) -> str:

        labels = cls._domain_labels(
            domain
        )

        index = cls._registrable_index(
            labels
        )

        if index < 0:
            return ""

        return labels[index]

    @classmethod
    def _get_subdomain_labels(
        cls,
        domain: str,
    ) -> List[str]:

        labels = cls._domain_labels(
            domain
        )

        index = cls._registrable_index(
            labels
        )

        if index <= 0:
            return []

        return labels[:index]

    def _tokenize_domain(
        self,
        domain: str,
    ) -> List[str]:

        tokens: List[str] = []

        for label in self._domain_labels(
            domain
        ):

            parts = [
                token
                for token in self.DOMAIN_DELIMITERS.split(
                    label
                )
                if token
            ]

            tokens.extend(
                parts
            )

        return tokens

    # ======================================================================
    # SIMILARITY
    # ======================================================================

    @staticmethod
    def _similarity(
        left: str,
        right: str,
    ) -> float:

        if not left or not right:
            return 0.0

        return SequenceMatcher(
            None,
            left,
            right,
        ).ratio()

    @staticmethod
    def _levenshtein_distance(
        left: str,
        right: str,
    ) -> int:

        if left == right:
            return 0

        if not left:
            return len(right)

        if not right:
            return len(left)

        previous = list(
            range(
                len(right) + 1
            )
        )

        for i, left_char in enumerate(
            left,
            start=1,
        ):

            current = [
                i
            ]

            for j, right_char in enumerate(
                right,
                start=1,
            ):

                insertion = (
                    current[j - 1]
                    + 1
                )

                deletion = (
                    previous[j]
                    + 1
                )

                substitution = (
                    previous[j - 1]
                    + (
                        0
                        if left_char
                        == right_char
                        else 1
                    )
                )

                current.append(
                    min(
                        insertion,
                        deletion,
                        substitution,
                    )
                )

            previous = current

        return previous[-1]

    # ======================================================================
    # TYPO OPERATIONS
    # ======================================================================

    @staticmethod
    def _has_insertion(
        brand: str,
        token: str,
    ) -> bool:

        if len(token) != len(brand) + 1:
            return False

        for index in range(
            len(token)
        ):

            if (
                token[:index]
                + token[index + 1:]
                == brand
            ):
                return True

        return False

    @staticmethod
    def _has_deletion(
        brand: str,
        token: str,
    ) -> bool:

        if len(brand) != len(token) + 1:
            return False

        for index in range(
            len(brand)
        ):

            if (
                brand[:index]
                + brand[index + 1:]
                == token
            ):
                return True

        return False

    @staticmethod
    def _has_substitution(
        brand: str,
        token: str,
    ) -> bool:

        if len(brand) != len(token):
            return False

        differences = sum(
            1
            for left, right
            in zip(
                brand,
                token,
            )
            if left != right
        )

        return differences == 1

    @staticmethod
    def _has_transposition(
        brand: str,
        token: str,
    ) -> bool:

        if len(brand) != len(token):
            return False

        differences = [
            index
            for index, (
                left,
                right,
            )
            in enumerate(
                zip(
                    brand,
                    token,
                )
            )
            if left != right
        ]

        if len(differences) != 2:
            return False

        first, second = differences

        return (
            brand[first]
            == token[second]
            and
            brand[second]
            == token[first]
        )

    @staticmethod
    def _has_repeated_character(
        brand: str,
        token: str,
    ) -> bool:

        if len(token) != len(brand) + 1:
            return False

        for index in range(
            len(token)
        ):

            candidate = (
                token[:index]
                + token[index + 1:]
            )

            if candidate == brand:
                return (
                    index > 0
                    and token[index]
                    == token[index - 1]
                )

        return False

    def _typo_operations(
        self,
        brand: str,
        token: str,
    ) -> Dict[str, bool]:

        return {
            "insertion":
                self._has_insertion(
                    brand,
                    token,
                ),

            "deletion":
                self._has_deletion(
                    brand,
                    token,
                ),

            "substitution":
                self._has_substitution(
                    brand,
                    token,
                ),

            "transposition":
                self._has_transposition(
                    brand,
                    token,
                ),

            "repeated":
                self._has_repeated_character(
                    brand,
                    token,
                ),
        }

    # ======================================================================
    # BRAND MATCHING
    # ======================================================================

    def _best_brand_match(
        self,
        domain: str,
    ) -> Tuple[
        str,
        float,
        str,
        bool,
        str,
    ]:
        """
        Return:

            brand
            similarity
            embedding_type
            direct_match
            matched_token
        """

        tokens = self._tokenize_domain(
            domain
        )

        if not tokens:
            return (
                "unknown",
                0.0,
                "none",
                False,
                "",
            )

        best_brand = "unknown"
        best_similarity = 0.0
        best_type = "none"
        best_direct = False
        best_token = ""

        for token in tokens:

            token_norm = self._normalize_brand(
                token
            )

            if not token_norm:
                continue

            for brand_norm, original_brand in (
                self._brand_lookup.items()
            ):

                # ----------------------------------------------------------
                # Exact
                # ----------------------------------------------------------

                if token_norm == brand_norm:

                    if (
                        1.0
                        > best_similarity
                        or (
                            best_type != "exact"
                        )
                    ):

                        best_brand = original_brand
                        best_similarity = 1.0
                        best_type = "exact"
                        best_direct = True
                        best_token = token_norm

                    continue

                # ----------------------------------------------------------
                # Brand contained in a longer token.
                # ----------------------------------------------------------

                if brand_norm in token_norm:

                    similarity = self._similarity(
                        brand_norm,
                        token_norm,
                    )

                    # Close modification = typo.
                    if (
                        similarity
                        >= self.TYPOSQUAT_THRESHOLD
                    ):

                        if (
                            similarity
                            > best_similarity
                        ):

                            best_brand = original_brand
                            best_similarity = similarity
                            best_type = "typosquat"
                            best_direct = False
                            best_token = token_norm

                        continue

                    # Legitimate brand embedding.
                    if (
                        similarity
                        >= self.SIMILARITY_THRESHOLD
                    ):

                        if (
                            similarity
                            > best_similarity
                        ):

                            best_brand = original_brand
                            best_similarity = similarity
                            best_type = "partial"
                            best_direct = True
                            best_token = token_norm

                        continue

                # ----------------------------------------------------------
                # General similarity.
                # ----------------------------------------------------------

                similarity = self._similarity(
                    brand_norm,
                    token_norm,
                )

                if (
                    similarity
                    >= self.TYPOSQUAT_THRESHOLD
                    and similarity
                    > best_similarity
                ):

                    best_brand = original_brand
                    best_similarity = similarity
                    best_type = "typosquat"
                    best_direct = False
                    best_token = token_norm

        return (
            best_brand,
            best_similarity,
            best_type,
            best_direct,
            best_token,
        )

    # ======================================================================
    # MAIN EXTRACTION
    # ======================================================================

    def extract(
        self,
        domain: str,
        brands: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:

        features = (
            self.get_default_features()
        )

        if brands is not None:

            self.set_brands(
                brands
            )

        if not isinstance(
            domain,
            str,
        ):
            return features

        if not domain.strip():
            return features

        try:

            normalized_domain = (
                self._normalize_domain(
                    domain
                )
            )

            if not normalized_domain:
                return features

            # --------------------------------------------------------------
            # Homograph / Unicode evidence
            # --------------------------------------------------------------

            punycode_detected = (
                self._contains_punycode(
                    normalized_domain
                )
            )

            unicode_detected = (
                self._contains_non_ascii(
                    normalized_domain
                )
            )

            confusable_count = (
                self._count_confusable_characters(
                    normalized_domain
                )
            )

            is_homograph_attack = (
                punycode_detected
                or unicode_detected
                or confusable_count > 0
            )

            analysis_domain = (
                self._decode_idna(
                    normalized_domain
                )
                if punycode_detected
                else normalized_domain
            )

            # --------------------------------------------------------------
            # Domain structure
            # --------------------------------------------------------------

            registered_domain = (
                self._get_registered_domain(
                    analysis_domain
                )
            )

            core_domain = (
                self._get_core_domain(
                    analysis_domain
                )
            )

            subdomain_labels = (
                self._get_subdomain_labels(
                    analysis_domain
                )
            )

            # --------------------------------------------------------------
            # Brand matching across ALL hostname tokens.
            #
            # This is the key correction over the previous implementation.
            # --------------------------------------------------------------

            (
                best_brand,
                similarity,
                embedding_type,
                direct_match,
                matched_token,
            ) = self._best_brand_match(
                analysis_domain
            )

            similarity = max(
                0.0,
                min(
                    1.0,
                    float(similarity),
                ),
            )

            accepted = (
                best_brand != "unknown"
                and similarity
                >= self.SIMILARITY_THRESHOLD
            )

            if not accepted:

                best_brand = "unknown"
                similarity = 0.0
                embedding_type = "none"
                direct_match = False
                matched_token = ""

            # --------------------------------------------------------------
            # Brand normalized value
            # --------------------------------------------------------------

            matched_brand_norm = (
                self._normalize_brand(
                    best_brand
                )
            )

            core_norm = (
                self._normalize_brand(
                    core_domain
                )
            )

            registered_norm = (
                self._normalize_brand(
                    registered_domain
                )
            )

            subdomain_norm = [
                self._normalize_brand(
                    label
                )
                for label in subdomain_labels
            ]

            # --------------------------------------------------------------
            # Exact / registered / subdomain
            # --------------------------------------------------------------

            exact_domain_match = (
                accepted
                and (
                    core_norm
                    == matched_brand_norm
                    or registered_norm
                    == matched_brand_norm
                )
            )

            brand_in_registered_domain = (
                accepted
                and (
                    matched_brand_norm
                    in registered_norm
                )
            )

            brand_in_subdomain = (
                accepted
                and any(
                    matched_brand_norm
                    in label
                    for label
                    in subdomain_norm
                )
            )

            # --------------------------------------------------------------
            # Typo operations
            # --------------------------------------------------------------

            typo_operations = (
                self._typo_operations(
                    matched_brand_norm,
                    matched_token,
                )
                if accepted
                and matched_brand_norm
                and matched_token
                else {
                    "insertion": False,
                    "deletion": False,
                    "substitution": False,
                    "transposition": False,
                    "repeated": False,
                }
            )

            typo_distance = (
                self._levenshtein_distance(
                    matched_brand_norm,
                    matched_token,
                )
                if accepted
                and matched_brand_norm
                and matched_token
                else 0
            )

            is_typo = (
                accepted
                and not exact_domain_match
                and not direct_match
                and embedding_type
                == "typosquat"
                and (
                    any(
                        typo_operations.values()
                    )
                    or similarity
                    >= self.STRONG_TYPO_THRESHOLD
                )
            )

            # --------------------------------------------------------------
            # Final result
            # --------------------------------------------------------------

            features.update(
                {
                    "matched_brand":
                        best_brand,

                    "is_brand_in_domain":
                        bool(
                            accepted
                            and direct_match
                        ),

                    "domain_brand_similarity":
                        round(
                            similarity,
                            4,
                        ),

                    "is_typosquatted":
                        bool(
                            is_typo
                        ),

                    "is_homograph_attack":
                        bool(
                            is_homograph_attack
                        ),

                    "brand_embedding_type":
                        (
                            embedding_type
                            if accepted
                            else "none"
                        ),

                    "has_character_insertion":
                        typo_operations[
                            "insertion"
                        ],

                    "has_character_deletion":
                        typo_operations[
                            "deletion"
                        ],

                    "has_character_substitution":
                        typo_operations[
                            "substitution"
                        ],

                    "has_character_transposition":
                        typo_operations[
                            "transposition"
                        ],

                    "has_repeated_character":
                        typo_operations[
                            "repeated"
                        ],

                    "exact_domain_match":
                        bool(
                            exact_domain_match
                        ),

                    "brand_in_subdomain":
                        bool(
                            brand_in_subdomain
                        ),

                    "brand_in_registered_domain":
                        bool(
                            brand_in_registered_domain
                        ),

                    "unicode_domain_detected":
                        bool(
                            unicode_detected
                        ),

                    "punycode_detected":
                        bool(
                            punycode_detected
                        ),

                    "confusable_character_count":
                        int(
                            confusable_count
                        ),

                    "typosquat_edit_distance":
                        int(
                            typo_distance
                        ),
                }
            )

            return features

        except Exception as exc:

            logger.exception(
                "Brand domain matching failed: %s",
                exc,
            )

            return (
                self.get_default_features()
            )

    # ======================================================================
    # COMPATIBILITY HELPERS
    # ======================================================================

    def extract_features(
        self,
        domain: str,
        brands: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:

        return self.extract(
            domain=domain,
            brands=brands,
        )

    @classmethod
    def validate_features(
        cls,
        features: Dict[str, Any],
    ) -> bool:

        if not isinstance(
            features,
            dict,
        ):
            return False

        return all(
            name in features
            for name in cls.FEATURE_NAMES
        )

    @classmethod
    def get_feature_count(
        cls,
    ) -> int:

        return 6