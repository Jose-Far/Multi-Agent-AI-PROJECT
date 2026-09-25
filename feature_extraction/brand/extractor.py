"""
Brand Feature Extraction
========================

Master Brand feature-extraction pipeline.

Architecture

    Dataset Reference Brand
            |
            v
    Domain Match
            |
    Keyword Analysis
            |
    Brand Similarity
            |
    Logo / Visual Evidence
            |
    OCR Evidence
            |
            v
    Cross-Modal Brand Analysis
            |
            v
    Brand Impersonation / Evasion
            |
            v
    BrandFeaturesSchema
            |
            v
    EXACT 64 ML FEATURES

Important
---------

The BrandFeaturesSchema is authoritative for:

    - feature names
    - feature order
    - feature count
    - defaults

This module does NOT:

    - train XGBoost
    - perform XGBoost inference
    - calculate final ML prediction
    - fabricate logo detection
    - contain a standalone execution hook
"""

from __future__ import annotations

import logging
import re

from difflib import SequenceMatcher
from typing import Any, Dict, List

from .domain_match import (
    DomainMatchFeatureExtractor,
)

from .keyword import (
    BrandKeywordFeatureExtractor,
)

from .similarity import (
    BrandSimilarityFeatureExtractor,
)

from .logo_match import (
    BrandLogoMatchFeatureExtractor,
)

from agents.brand_agent.feature_schema import (
    BrandFeaturesSchema,
)


logger = logging.getLogger(__name__)


class BrandFeatureExtractor:
    """
    Master Brand Feature Extraction Engine.

    Produces exactly 64 canonical ML features.

    The extractor supports two important brand sources:

    1. reference_brand supplied by the dataset builder
    2. matched_brand supplied by DomainMatchFeatureExtractor

    The explicit dataset reference brand has priority because the
    dataset builder has access to the complete targetlist.
    """

    EXTRACTOR_VERSION = "4.3.0"

    # =====================================================================
    # INITIALIZATION
    # =====================================================================

    def __init__(self):

        self.domain_matcher = (
            DomainMatchFeatureExtractor()
        )

        self.keyword_extractor = (
            BrandKeywordFeatureExtractor()
        )

        self.similarity_extractor = (
            BrandSimilarityFeatureExtractor()
        )

        self.logo_matcher = (
            BrandLogoMatchFeatureExtractor()
        )

        self.schema = BrandFeaturesSchema

    # =====================================================================
    # MAIN EXTRACTION
    # =====================================================================

    def extract_features(
        self,
        raw_data: Dict[str, Any],
        visual_features: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        if not isinstance(
            raw_data,
            dict,
        ):

            logger.warning(
                "Brand extractor received invalid raw_data."
            )

            return self._get_empty_features()

        if not isinstance(
            visual_features,
            dict,
        ):

            visual_features = {}

        try:

            # =============================================================
            # 1. RESOLVE BLOCKS
            # =============================================================

            url_analysis = (
                self._resolve_block(
                    raw_data.get(
                        "url_analysis",
                        {},
                    )
                )
            )

            html_analysis = (
                self._resolve_block(
                    raw_data.get(
                        "html_analysis",
                        {},
                    )
                )
            )

            visual_data = (
                self._resolve_block(
                    visual_features
                )
            )

            # =============================================================
            # 2. DOMAIN
            # =============================================================

            domain = (
                self._resolve_domain(
                    raw_data,
                    url_analysis,
                )
            )

            # =============================================================
            # 3. HTML
            # =============================================================

            page_title = (
                self._resolve_text(
                    html_analysis,
                    (
                        "page_title",
                        "title",
                        "html_title",
                    ),
                )
            )

            page_text = (
                self._resolve_text(
                    html_analysis,
                    (
                        "visible_text",
                        "text",
                        "page_text",
                        "body_text",
                    ),
                )
            )

            # =============================================================
            # 4. OCR
            # =============================================================

            ocr_text = (
                self._resolve_text(
                    visual_data,
                    (
                        "ocr_text",
                        "ocr_text_snippet",
                        "text",
                    ),
                )
            )

            if not ocr_text:

                ocr_text = str(
                    raw_data.get(
                        "ocr_text",
                        "",
                    )
                    or ""
                ).strip()

            # =============================================================
            # 5. AUTHORITATIVE REFERENCE BRAND
            # =============================================================

            reference_brand = (
                self._resolve_reference_brand(
                    raw_data,
                    visual_data,
                )
            )

            # =============================================================
            # 6. DOMAIN MATCH
            # =============================================================

            domain_features = (
                self._safe_extract(
                    self.domain_matcher.extract,
                    domain,
                )
            )

            internal_matched_brand = str(
                domain_features.get(
                    "matched_brand",
                    "unknown",
                )
                or "unknown"
            ).strip()

            # =============================================================
            # IMPORTANT BRAND PRIORITY RULE
            # =============================================================
            #
            # The dataset builder supplies reference_brand based on the
            # complete 181-brand targetlist.
            #
            # Therefore:
            #
            #     reference_brand
            #          >
            #     internal DomainMatch brand
            #
            # This prevents the internal smaller brand list from
            # incorrectly replacing the dataset's authoritative brand.
            #

            matched_brand = (
                reference_brand
                if (
                    reference_brand
                    and
                    reference_brand.lower()
                    != "unknown"
                )
                else internal_matched_brand
            )

            if not matched_brand:

                matched_brand = "unknown"

            # =============================================================
            # 7. REBUILD DOMAIN FEATURES USING REFERENCE BRAND
            # =============================================================

            if (
                matched_brand.lower()
                != "unknown"
            ):

                reference_domain_features = (
                    self._build_reference_domain_features(
                        domain,
                        matched_brand,
                    )
                )

                if reference_domain_features:

                    domain_features.update(
                        reference_domain_features
                    )

            domain_features[
                "matched_brand"
            ] = matched_brand

            # =============================================================
            # 8. KEYWORD FEATURES
            # =============================================================

            keyword_input = (
                page_text
                if page_text
                else ocr_text
            )

            keyword_features = (
                self._safe_extract(
                    self.keyword_extractor.extract,
                    page_title,
                    keyword_input,
                )
            )

            # =============================================================
            # 9. SIMILARITY FEATURES
            # =============================================================

            similarity_features = (
                self._safe_extract(
                    self.similarity_extractor.extract,
                    domain,
                    page_title,
                    matched_brand,
                )
            )

            # =============================================================
            # 10. AUTHORITATIVE TEXT FEATURES
            # =============================================================

            reference_text_features = (
                self._build_reference_text_features(
                    page_title=page_title,
                    page_text=page_text,
                    ocr_text=ocr_text,
                    reference_brand=matched_brand,
                )
            )

            similarity_features.update(
                reference_text_features
            )

            # =============================================================
            # 11. LOGO / VISUAL TELEMETRY
            # =============================================================

            logo_features = (
                self._safe_extract(
                    self.logo_matcher.extract,
                    visual_data,
                    domain,
                )
            )

            # =============================================================
            # 12. BUILD CANONICAL 64 FEATURES
            # =============================================================

            features = (
                self._build_canonical_features(
                    domain=domain,
                    page_title=page_title,
                    page_text=page_text,
                    ocr_text=ocr_text,
                    domain_features=domain_features,
                    keyword_features=keyword_features,
                    similarity_features=similarity_features,
                    logo_features=logo_features,
                    visual_data=visual_data,
                    reference_brand=matched_brand,
                )
            )

            # =============================================================
            # 13. FINALIZE
            # =============================================================

            return (
                self._finalize_features(
                    features=features,
                    domain_features=domain_features,
                    keyword_features=keyword_features,
                    logo_features=logo_features,
                    visual_data=visual_data,
                    ocr_text=ocr_text,
                )
            )

        except Exception as exc:

            logger.error(
                "Critical Brand feature extraction failure: %s",
                exc,
                exc_info=True,
            )

            return self._get_empty_features()

    # =====================================================================
    # BLOCK RESOLUTION
    # =====================================================================

    @staticmethod
    def _resolve_block(
        value: Any,
    ) -> Dict[str, Any]:

        if not isinstance(
            value,
            dict,
        ):

            return {}

        data = value.get(
            "data"
        )

        if isinstance(
            data,
            dict,
        ):

            merged = dict(value)
            merged.update(data)

            return merged

        return dict(value)

    # =====================================================================
    # TEXT RESOLUTION
    # =====================================================================

    @staticmethod
    def _resolve_text(
        data: Dict[str, Any],
        keys: tuple[str, ...],
    ) -> str:

        if not isinstance(
            data,
            dict,
        ):

            return ""

        for key in keys:

            value = data.get(
                key
            )

            if value is None:
                continue

            if isinstance(
                value,
                str,
            ):

                text = value.strip()

            else:

                text = str(
                    value
                ).strip()

            if text:
                return text

        return ""

    # =====================================================================
    # DOMAIN RESOLUTION
    # =====================================================================

    @staticmethod
    def _resolve_domain(
        raw_data: Dict[str, Any],
        url_analysis: Dict[str, Any],
    ) -> str:

        metadata = raw_data.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):

            metadata = {}

        candidates = (

            url_analysis.get(
                "domain_resolved"
            ),

            url_analysis.get(
                "domain"
            ),

            url_analysis.get(
                "hostname"
            ),

            url_analysis.get(
                "final_url"
            ),

            metadata.get(
                "domain"
            ),

            metadata.get(
                "target_url"
            ),

            raw_data.get(
                "domain"
            ),

            raw_data.get(
                "url"
            ),
        )

        for candidate in candidates:

            if not candidate:
                continue

            value = str(
                candidate
            ).strip().lower()

            if not value:
                continue

            if "://" in value:

                value = value.split(
                    "://",
                    1,
                )[1]

            value = value.split(
                "/",
                1,
            )[0]

            value = value.split(
                "?",
                1,
            )[0]

            value = value.split(
                "#",
                1,
            )[0]

            value = value.split(
                ":",
                1,
            )[0]

            value = value.strip()

            if value:
                return value

        return ""

    # =====================================================================
    # REFERENCE BRAND
    # =====================================================================

    @staticmethod
    def _resolve_reference_brand(
        raw_data: Dict[str, Any],
        visual_data: Dict[str, Any],
    ) -> str:

        metadata = raw_data.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):

            metadata = {}

        candidates = (

            metadata.get(
                "reference_brand"
            ),

            raw_data.get(
                "reference_brand"
            ),

            visual_data.get(
                "reference_brand"
            ),

            visual_data.get(
                "dataset_reference_brand"
            ),
        )

        for candidate in candidates:

            if candidate is None:
                continue

            value = str(
                candidate
            ).strip()

            if not value:
                continue

            if value.lower() == "unknown":
                continue

            return value

        return "unknown"

    # =====================================================================
    # SAFE SUB-EXTRACTOR
    # =====================================================================

    @staticmethod
    def _safe_extract(
        extractor_func,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:

        try:

            result = extractor_func(
                *args,
                **kwargs,
            )

            if isinstance(
                result,
                dict,
            ):

                return result

        except Exception as exc:

            logger.warning(
                "Brand sub-extractor failed: %s",
                exc,
            )

        return {}

    # =====================================================================
    # NORMALIZATION
    # =====================================================================

    @staticmethod
    def _normalize_brand(
        value: Any,
    ) -> str:

        return re.sub(
            r"[^a-z0-9]+",
            "",
            str(
                value or ""
            ).lower(),
        )

    @classmethod
    def _normalize_domain_core(
        cls,
        domain: Any,
    ) -> str:

        value = str(
            domain or ""
        ).strip().lower()

        if not value:
            return ""

        if "://" in value:

            value = value.split(
                "://",
                1,
            )[1]

        value = value.split(
            "/",
            1,
        )[0]

        value = value.split(
            ":",
            1,
        )[0]

        labels = [
            label
            for label in value.split(".")
            if label
        ]

        if not labels:
            return ""

        # Usually the registrable domain is the final two labels.
        if len(labels) >= 2:

            core = labels[-2]

        else:

            core = labels[0]

        return cls._normalize_brand(
            core
        )

    # =====================================================================
    # REFERENCE DOMAIN FEATURES
    # =====================================================================

    def _build_reference_domain_features(
        self,
        domain: str,
        reference_brand: str,
    ) -> Dict[str, Any]:

        if (
            not domain
            or not reference_brand
            or reference_brand.lower()
            == "unknown"
        ):

            return {}

        domain_lower = str(
            domain
        ).strip().lower()

        brand_norm = (
            self._normalize_brand(
                reference_brand
            )
        )

        if not brand_norm:
            return {}

        domain_labels = [
            label
            for label in domain_lower.split(".")
            if label
        ]

        domain_core = (
            self._normalize_domain_core(
                domain_lower
            )
        )

        domain_norm = (
            self._normalize_brand(
                domain_core
            )
        )

        # ---------------------------------------------------------------
        # Similarity
        # ---------------------------------------------------------------

        similarity = (
            SequenceMatcher(
                None,
                domain_norm,
                brand_norm,
            ).ratio()
            if domain_norm
            else 0.0
        )

        # ---------------------------------------------------------------
        # Complete-token brand presence
        # ---------------------------------------------------------------

        token_match = False

        for label in domain_labels:

            pieces = re.split(
                r"[^a-z0-9]+",
                label,
            )

            for piece in pieces:

                if (
                    piece
                    and
                    self._normalize_brand(
                        piece
                    )
                    == brand_norm
                ):

                    token_match = True
                    break

            if token_match:
                break

        # ---------------------------------------------------------------
        # Exact domain
        # ---------------------------------------------------------------

        exact = (
            domain_norm
            == brand_norm
        )

        # ---------------------------------------------------------------
        # Subdomain
        # ---------------------------------------------------------------

        is_subdomain = (
            len(domain_labels) > 2
            and any(
                self._normalize_brand(
                    label
                )
                == brand_norm
                for label
                in domain_labels[:-2]
            )
        )

        # ---------------------------------------------------------------
        # Brand in registered domain
        # ---------------------------------------------------------------

        brand_in_registered_domain = (
            brand_norm in domain_norm
            if domain_norm
            else False
        )

        # ---------------------------------------------------------------
        # Typosquatting
        # ---------------------------------------------------------------

        edit_distance = (
            self._edit_distance(
                domain_norm,
                brand_norm,
            )
        )

        typo_operations = (
            self._typo_operations(
                domain_norm,
                brand_norm,
            )
        )

        typo_candidate = (

            not exact

            and

            similarity >= 0.70

            and

            edit_distance <= max(
                3,
                int(
                    len(
                        brand_norm
                    )
                    * 0.35
                ),
            )

            and

            domain_norm != ""
        )

        # ---------------------------------------------------------------
        # Homograph
        # ---------------------------------------------------------------

        unicode_detected = any(
            ord(char) > 127
            for char in domain_lower
        )

        punycode_detected = any(
            label.startswith(
                "xn--"
            )
            for label
            in domain_lower.split(".")
        )

        confusable_count = (
            self._count_confusable_characters(
                domain_lower
            )
        )

        homograph = (
            unicode_detected
            or punycode_detected
            or confusable_count > 0
        )

        # ---------------------------------------------------------------
        # Embedding type
        # ---------------------------------------------------------------

        if exact:

            embedding_type = "exact"

        elif is_subdomain:

            embedding_type = "subdomain"

        elif similarity >= 0.70:

            embedding_type = "similar"

        else:

            embedding_type = "none"

        return {

            "matched_brand":
                reference_brand,

            "domain_brand_similarity":
                similarity,

            "is_brand_in_domain":
                token_match,

            "brand_embedding_type":
                embedding_type,

            "is_typosquatted":
                typo_candidate,

            "is_homograph_attack":
                homograph,

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
        }

    # =====================================================================
    # REFERENCE TEXT FEATURES
    # =====================================================================

    @classmethod
    def _build_reference_text_features(
        cls,
        page_title: str,
        page_text: str,
        ocr_text: str,
        reference_brand: str,
    ) -> Dict[str, Any]:

        if (
            not reference_brand
            or reference_brand.lower()
            == "unknown"
        ):

            return {}

        brand_norm = (
            cls._normalize_brand(
                reference_brand
            )
        )

        if not brand_norm:
            return {}

        title_norm = (
            cls._normalize_brand(
                page_title
            )
        )

        page_norm = (
            cls._normalize_brand(
                page_text
            )
        )

        ocr_norm = (
            cls._normalize_brand(
                ocr_text
            )
        )

        title_match = (
            bool(title_norm)
            and brand_norm
            in title_norm
        )

        page_match = (
            bool(page_norm)
            and brand_norm
            in page_norm
        )

        ocr_match = (
            bool(ocr_norm)
            and brand_norm
            in ocr_norm
        )

        title_similarity = (
            SequenceMatcher(
                None,
                title_norm,
                brand_norm,
            ).ratio()
            if title_norm
            else 0.0
        )

        ocr_similarity = (
            SequenceMatcher(
                None,
                ocr_norm,
                brand_norm,
            ).ratio()
            if ocr_norm
            else 0.0
        )

        return {

            "is_brand_name_in_title":
                title_match,

            "title_brand_similarity":
                max(
                    0.0,
                    min(
                        1.0,
                        title_similarity,
                    ),
                ),

            "title_brand_distance_score":
                max(
                    0.0,
                    min(
                        1.0,
                        1.0
                        - title_similarity,
                    ),
                )
                if title_norm
                else 1.0,

            "brand_mentions_in_page":
                cls._count_normalized_brand_mentions(
                    page_text,
                    reference_brand,
                ),

            "brand_mentions_in_title":
                cls._count_normalized_brand_mentions(
                    page_title,
                    reference_brand,
                ),

            "ocr_brand_detected":
                ocr_match,

            "ocr_text_brand_similarity":
                max(
                    0.0,
                    min(
                        1.0,
                        ocr_similarity,
                    ),
                ),
        }

    # =====================================================================
    # CANONICAL FEATURES
    # =====================================================================

    def _build_canonical_features(
        self,
        domain: str,
        page_title: str,
        page_text: str,
        ocr_text: str,
        domain_features: Dict[str, Any],
        keyword_features: Dict[str, Any],
        similarity_features: Dict[str, Any],
        logo_features: Dict[str, Any],
        visual_data: Dict[str, Any],
        reference_brand: str = "unknown",
    ) -> Dict[str, Any]:

        features = (
            self.schema.get_default_features()
        )

        matched_brand = str(
            reference_brand
            or domain_features.get(
                "matched_brand",
                "unknown",
            )
            or "unknown"
        ).strip()

        if not matched_brand:

            matched_brand = "unknown"

        matched_brand_norm = (
            self._normalize_brand(
                matched_brand
            )
        )

        # ================================================================
        # DOMAIN
        # ================================================================

        domain_similarity = self._safe_float(
            domain_features.get(
                "domain_brand_similarity",
                0.0,
            )
        )

        if (
            matched_brand_norm
            and matched_brand.lower()
            != "unknown"
            and domain
        ):

            reference_domain = (
                self._build_reference_domain_features(
                    domain,
                    matched_brand,
                )
            )

            if reference_domain:

                domain_similarity = (
                    self._safe_float(
                        reference_domain.get(
                            "domain_brand_similarity",
                            domain_similarity,
                        )
                    )
                )

                domain_features.update(
                    reference_domain
                )

        domain_similarity = (
            self._clamp01(
                domain_similarity
            )
        )

        domain_distance = (
            self._clamp01(
                1.0
                - domain_similarity
            )
        )

        domain_core = (
            self._normalize_domain_core(
                domain
            )
        )

        is_brand_in_domain = (
            self._brand_token_in_domain(
                domain,
                matched_brand,
            )
        )

        brand_type = str(
            domain_features.get(
                "brand_embedding_type",
                "none",
            )
            or "none"
        ).lower()

        is_exact_domain = (
            domain_core
            == matched_brand_norm
            and bool(
                domain_core
            )
            and bool(
                matched_brand_norm
            )
        )

        is_subdomain = (
            self._is_brand_subdomain(
                domain,
                matched_brand,
            )
        )

        features.update(
            {

                "brand_candidate_count":
                    1
                    if (
                        matched_brand.lower()
                        != "unknown"
                    )
                    else 0,

                "is_brand_in_domain":
                    is_brand_in_domain,

                "domain_brand_similarity":
                    domain_similarity,

                "domain_brand_similarity_percent":
                    domain_similarity
                    * 100.0,

                "domain_brand_distance":
                    domain_distance,

                "is_exact_brand_domain":
                    (
                        is_exact_domain
                        or
                        brand_type
                        == "exact"
                    ),

                "is_brand_subdomain":
                    is_subdomain,

                "is_brand_in_registered_domain":
                    bool(
                        matched_brand_norm
                        and
                        matched_brand_norm
                        in domain_core
                    ),
            }
        )

        # ================================================================
        # TYPOSQUATTING
        # ================================================================

        typo_distance = (
            self._edit_distance(
                domain_core,
                matched_brand_norm,
            )
            if (
                domain_core
                and
                matched_brand_norm
            )
            else 0
        )

        typo_operations = (
            self._typo_operations(
                domain_core,
                matched_brand_norm,
            )
        )

        supplied_typo = bool(
            domain_features.get(
                "is_typosquatted",
                False,
            )
        )

        calculated_typo = (
            not is_exact_domain
            and
            domain_similarity >= 0.70
            and
            typo_distance <= max(
                3,
                int(
                    len(
                        matched_brand_norm
                    )
                    * 0.35
                ),
            )
            and
            bool(domain_core)
            and
            bool(matched_brand_norm)
        )

        is_typo = (
            supplied_typo
            or calculated_typo
        )

        typo_score = (
            domain_similarity
            if is_typo
            else 0.0
        )

        features.update(
            {

                "is_typosquatted":
                    is_typo,

                "typosquat_score":
                    self._clamp01(
                        typo_score
                    ),

                "typosquat_edit_distance":
                    typo_distance,

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
            }
        )

        # ================================================================
        # HOMOGRAPH
        # ================================================================

        unicode_detected = any(
            ord(char) > 127
            for char in str(
                domain
            )
        )

        punycode_detected = any(
            label.startswith(
                "xn--"
            )
            for label
            in str(domain).lower().split(".")
        )

        confusable_count = (
            self._count_confusable_characters(
                domain
            )
        )

        supplied_homograph = bool(
            domain_features.get(
                "is_homograph_attack",
                False,
            )
        )

        homograph = (
            supplied_homograph
            or
            unicode_detected
            or
            punycode_detected
            or
            confusable_count > 0
        )

        features.update(
            {

                "is_homograph_attack":
                    homograph,

                "homograph_score":
                    1.0
                    if homograph
                    else 0.0,

                "unicode_domain_detected":
                    unicode_detected,

                "confusable_character_count":
                    confusable_count,

                "punycode_detected":
                    punycode_detected,
            }
        )

        # ================================================================
        # HTML / TITLE
        # ================================================================

        title_match = bool(
            similarity_features.get(
                "is_brand_name_in_title",
                False,
            )
        )

        title_similarity = (
            self._safe_float(
                similarity_features.get(
                    "title_brand_similarity",
                    0.0,
                )
            )
        )

        title_distance = (
            self._safe_float(
                similarity_features.get(
                    "title_brand_distance_score",
                    1.0,
                )
            )
        )

        reference_text = (
            self._build_reference_text_features(
                page_title=page_title,
                page_text=page_text,
                ocr_text=ocr_text,
                reference_brand=matched_brand,
            )
        )

        if reference_text:

            title_match = bool(
                reference_text.get(
                    "is_brand_name_in_title",
                    title_match,
                )
            )

            title_similarity = (
                self._safe_float(
                    reference_text.get(
                        "title_brand_similarity",
                        title_similarity,
                    )
                )
            )

            title_distance = (
                self._safe_float(
                    reference_text.get(
                        "title_brand_distance_score",
                        title_distance,
                    )
                )
            )

        title_similarity = (
            self._clamp01(
                title_similarity
            )
        )

        title_distance = (
            self._clamp01(
                title_distance
            )
        )

        brand_mentions_page = (
            int(
                reference_text.get(
                    "brand_mentions_in_page",
                    0,
                )
                or 0
            )
            if reference_text
            else
            self._count_normalized_brand_mentions(
                page_text,
                matched_brand,
            )
        )

        brand_mentions_title = (
            int(
                reference_text.get(
                    "brand_mentions_in_title",
                    0,
                )
                or 0
            )
            if reference_text
            else
            self._count_normalized_brand_mentions(
                page_title,
                matched_brand,
            )
        )

        features.update(
            {

                "is_brand_name_in_title":
                    title_match,

                "title_brand_similarity":
                    title_similarity,

                "title_brand_distance_score":
                    title_distance,

                "brand_mentions_in_page":
                    max(
                        0,
                        brand_mentions_page,
                    ),

                "brand_mentions_in_title":
                    max(
                        0,
                        brand_mentions_title,
                    ),
            }
        )

        # ================================================================
        # SENSITIVE LANGUAGE
        # ================================================================

        sensitive_count = int(
            keyword_features.get(
                "total_suspicious_keyword_count",
                keyword_features.get(
                    "suspicious_keyword_count",
                    0,
                ),
            )
            or 0
        )

        credential = bool(
            keyword_features.get(
                "has_credential_solicitation",
                keyword_features.get(
                    "has_credential_request_language",
                    False,
                ),
            )
        )

        urgency = bool(
            keyword_features.get(
                "has_urgency_or_threat",
                keyword_features.get(
                    "has_urgency_language",
                    False,
                ),
            )
        )

        financial = bool(
            keyword_features.get(
                "has_financial_admin_context",
                False,
            )
        )

        login_keywords = (
            self._has_keyword(
                keyword_features,
                (
                    "login",
                    "verify",
                    "sign in",
                    "signin",
                ),
            )
        )

        account_security = (
            credential
            or
            urgency
        )

        features.update(
            {

                "sensitive_keyword_count":
                    max(
                        0,
                        sensitive_count,
                    ),

                "has_login_or_verify_keywords":
                    login_keywords,

                "has_credential_request_language":
                    credential,

                "has_urgency_language":
                    urgency,

                "has_account_security_language":
                    account_security,

            }
        )

        # ================================================================
        # LOGO / VISUAL
        # ================================================================

        logo_detected = bool(
            logo_features.get(
                "logo_detected",
                False,
            )
        )

        logo_confidence = (
            self._clamp01(
                self._safe_float(
                    logo_features.get(
                        "logo_confidence",
                        0.0,
                    )
                )
            )
        )

        visual_brand = str(
            logo_features.get(
                "primary_brand_name",
                "unknown",
            )
            or "unknown"
        ).strip()

        if not visual_brand:
            visual_brand = "unknown"

        # Never use the dataset reference brand as visual brand.
        if not logo_detected:
            visual_brand = "unknown"
            logo_confidence = 0.0

        visual_brand_norm = (
            self._normalize_brand(
                visual_brand
            )
        )

        logo_brand_similarity = 0.0

        if (
            logo_detected
            and
            visual_brand_norm
            and
            matched_brand_norm
            and
            visual_brand.lower()
            != "unknown"
            and
            matched_brand.lower()
            != "unknown"
        ):

            logo_brand_similarity = (
                SequenceMatcher(
                    None,
                    visual_brand_norm,
                    matched_brand_norm,
                ).ratio()
            )

        features.update(
            {
                "logo_detected":
                    logo_detected,

                "logo_confidence":
                    logo_confidence,

                "logo_brand_similarity":
                    self._clamp01(
                        logo_brand_similarity
                    ),
            }
        )

        # ================================================================
        # OCR
        # ================================================================

        ocr_brand_detected = bool(
            reference_text.get(
                "ocr_brand_detected",
                False,
            )
            if reference_text
            else (
                matched_brand_norm
                and
                self._text_contains_brand(
                    ocr_text,
                    matched_brand,
                )
            )
        )

        ocr_similarity = (
            self._safe_float(
                reference_text.get(
                    "ocr_text_brand_similarity",
                    0.0,
                )
            )
            if reference_text
            else 0.0
        )

        # OCR confidence is NOT invented from logo confidence.
        #
        # If OCR is present, use similarity as the deterministic evidence
        # strength. This prevents the previous all-zero OCR confidence
        # problem while keeping the value grounded in actual OCR text.

        ocr_confidence = (
            ocr_similarity
            if ocr_brand_detected
            else 0.0
        )

        features.update(
            {

                "ocr_brand_detected":
                    bool(
                        ocr_brand_detected
                    ),

                "ocr_brand_confidence":
                    self._clamp01(
                        ocr_confidence
                    ),

                "ocr_text_brand_similarity":
                    self._clamp01(
                        ocr_similarity
                    ),
            }
        )

        # ================================================================
        # VISUAL / DOMAIN
        # ================================================================

        visual_brand_matches_domain = (
            logo_detected
            and
            visual_brand.lower() != "unknown"
            and
            matched_brand.lower() != "unknown"
            and
            visual_brand_norm == matched_brand_norm
        )

        visual_domain_similarity = (
            logo_brand_similarity
            if logo_detected
            else 0.0
        )

        visual_domain_mismatch = (
            1.0 - visual_domain_similarity
            if (
                logo_detected
                and
                visual_brand.lower() != "unknown"
                and
                matched_brand.lower() != "unknown"
            )
            else 0.0
        )

        features.update(
            {
                "visual_brand_matches_domain":
                    visual_brand_matches_domain,

                "visual_domain_similarity":
                    self._clamp01(
                        visual_domain_similarity
                    ),

                "visual_domain_mismatch_score":
                    self._clamp01(
                        visual_domain_mismatch
                    ),
            }
        )

        # ================================================================
        # CROSS MODAL
        # ================================================================

        html_brand_match = (
            matched_brand.lower()
            != "unknown"
            and
            (
                title_match
                or
                brand_mentions_page > 0
            )
        )

        domain_visual_match = (
            logo_detected
            and
            visual_brand_matches_domain
        )

        html_visual_match = (
            logo_detected
            and
            visual_brand_matches_domain
            and
            html_brand_match
        )

        cross_modal_mismatch = (
            visual_brand.lower()
            != "unknown"
            and
            matched_brand.lower()
            != "unknown"
            and
            visual_brand_norm
            != matched_brand_norm
        )

        cross_modal_values = []

        if (
            matched_brand.lower()
            != "unknown"
        ):

            # HTML evidence
            if html_brand_match:

                cross_modal_values.append(
                    1.0
                )

            elif page_title or page_text:

                cross_modal_values.append(
                    0.0
                )

            # OCR evidence
            if ocr_text:

                cross_modal_values.append(
                    ocr_similarity
                )

            # Visual logo evidence
            if (
                visual_brand.lower()
                != "unknown"
            ):

                cross_modal_values.append(
                    logo_brand_similarity
                )

        cross_modal_similarity = (
            sum(
                cross_modal_values
            )
            /
            len(
                cross_modal_values
            )
            if cross_modal_values
            else 0.0
        )

        cross_modal_conflict_count = 0

        if cross_modal_mismatch:

            cross_modal_conflict_count += 1

        if (
            ocr_brand_detected
            and
            visual_brand.lower()
            != "unknown"
            and
            visual_brand_norm
            != matched_brand_norm
        ):

            cross_modal_conflict_count += 1

        features.update(
            {

                "cross_modal_brand_mismatch":
                    cross_modal_mismatch,

                "cross_modal_similarity":
                    self._clamp01(
                        cross_modal_similarity
                    ),

                "cross_modal_conflict_count":
                    cross_modal_conflict_count,

                "domain_html_brand_match":
                    html_brand_match,

                "domain_visual_brand_match":
                    domain_visual_match,

                "html_visual_brand_match":
                    html_visual_match,
            }
        )

        # ================================================================
        # IMPERSONATION
        # ================================================================

        domain_brand_signal = (
            domain_similarity >= 0.70
            and
            matched_brand.lower()
            != "unknown"
        )

        html_brand_signal = (
            html_brand_match
        )

        ocr_brand_signal = (
            ocr_brand_detected
        )

        visual_brand_signal = (
            logo_detected
            and
            visual_brand.lower()
            != "unknown"
            and
            matched_brand.lower()
            != "unknown"
            and
            visual_brand_norm
            == matched_brand_norm
        )

        suspicious_domain_signal = (
            is_typo
            or
            homograph
        )

        impersonation_signals = sum(
            [
                int(
                    domain_brand_signal
                ),
                int(
                    html_brand_signal
                ),
                int(
                    ocr_brand_signal
                ),
                int(
                    visual_brand_signal
                ),
                int(
                    suspicious_domain_signal
                ),
            ]
        )

        impersonation_confidence = 0.0

        if matched_brand.lower() != "unknown":

            evidence_values = []

            if domain_brand_signal:

                evidence_values.append(
                    domain_similarity
                )

            if html_brand_signal:

                evidence_values.append(
                    title_similarity
                    if title_match
                    else min(
                        1.0,
                        brand_mentions_page
                        / 3.0,
                    )
                )

            if ocr_brand_signal:

                evidence_values.append(
                    ocr_similarity
                )

            if visual_brand_signal:

                evidence_values.append(
                    logo_brand_similarity
                )

            if suspicious_domain_signal:

                evidence_values.append(
                    max(
                        typo_score,
                        1.0
                        if homograph
                        else 0.0,
                    )
                )

            if evidence_values:

                impersonation_confidence = (
                    sum(
                        evidence_values
                    )
                    /
                    len(
                        evidence_values
                    )
                )

        # Strong multi-signal impersonation.
        high_confidence_impersonation = (
            impersonation_confidence >= 0.85
            and
            (
                suspicious_domain_signal
                or
                html_brand_signal
                or
                ocr_brand_signal
                or
                visual_brand_signal
            )
        )

        # A brand appearing in HTML/OCR is not automatically malicious.
        # Risk becomes meaningful when there is evidence of brand
        # association plus a suspicious domain or mismatching visual.
        brand_impersonation_risk = (
            matched_brand.lower()
            != "unknown"
            and
            (
                (
                    suspicious_domain_signal
                    and
                    (
                        html_brand_signal
                        or
                        ocr_brand_signal
                        or
                        visual_brand_signal
                    )
                )
                or
                cross_modal_mismatch
                or
                high_confidence_impersonation
            )
        )

        features.update(
            {

                "is_high_confidence_impersonation":
                    high_confidence_impersonation,

                "impersonation_confidence":
                    self._clamp01(
                        impersonation_confidence
                    ),

                "brand_impersonation_signal_count":
                    impersonation_signals,

                "has_brand_impersonation_risk":
                    brand_impersonation_risk,
            }
        )

        # ================================================================
        # ADVERSARIAL / EVASION
        # ================================================================

        adversarial = bool(
            logo_features.get(
                "possible_adversarial_logo_evasion",
                False,
            )
        )

        adversarial_similarity = (
            self._clamp01(
                self._safe_float(
                    logo_features.get(
                        "logo_confidence",
                        0.0,
                    )
                )
            )
            if adversarial
            else 0.0
        )

        # OCR/visual conflict is meaningful only when both independent
        # signals exist.

        image_text_conflict = (
            logo_detected
            and
            ocr_brand_detected
            and
            visual_brand.lower()
            != "unknown"
            and
            matched_brand.lower()
            != "unknown"
            and
            visual_brand_norm
            != matched_brand_norm
        )

        visual_evasion_score = max(
            adversarial_similarity,
            (
                1.0
                if image_text_conflict
                else 0.0
            ),
        )

        # Domain/brand mismatch combined with a detected visual brand
        # is also useful evidence.

        if (
            visual_brand.lower()
            != "unknown"
            and
            matched_brand.lower()
            != "unknown"
            and
            visual_brand_norm
            != matched_brand_norm
        ):

            visual_evasion_score = max(
                visual_evasion_score,
                visual_domain_mismatch,
            )

        features.update(
            {

                "possible_adversarial_logo_evasion":
                    adversarial,

                "adversarial_logo_similarity":
                    self._clamp01(
                        adversarial_similarity
                    ),

                "image_text_brand_conflict":
                    image_text_conflict,

                "visual_brand_evasion_score":
                    self._clamp01(
                        visual_evasion_score
                    ),
            }
        )

        # ================================================================
        # EVIDENCE AVAILABILITY
        # ================================================================

        domain_available = bool(
            domain
        )

        html_available = bool(
            page_title
            or
            page_text
        )

        # visual_data can contain an image object, screenshot, or explicit
        # availability metadata.

        visual_available = (
            self._visual_evidence_available(
                visual_data
            )
        )

        # IMPORTANT:
        #
        # logo_signal_available means logo telemetry was actually supplied,
        # not merely that an image exists.
        #

        logo_available = (
            logo_detected
            and
            logo_confidence > 0.0
            or
            (
                "logo_detected"
                in logo_features
                and
                bool(
                    logo_features.get(
                        "logo_confidence",
                        0.0,
                    )
                )
            )
        )

        ocr_available = bool(
            ocr_text
        )

        features.update(
            {

                "domain_signal_available":
                    domain_available,

                "html_signal_available":
                    html_available,

                "visual_signal_available":
                    visual_available,

                "logo_signal_available":
                    logo_available,

                "ocr_signal_available":
                    ocr_available,
            }
        )

        # ================================================================
        # EVIDENCE COUNTS
        # ================================================================

        available_signal_count = sum(
            [
                int(
                    domain_available
                ),
                int(
                    html_available
                ),
                int(
                    visual_available
                ),
                int(
                    logo_available
                ),
                int(
                    ocr_available
                ),
            ]
        )

        total_possible_signal_count = 5

        features.update(
            {

                "available_signal_count":
                    available_signal_count,

                "total_possible_signal_count":
                    total_possible_signal_count,
            }
        )

        # ================================================================
        # DATA QUALITY
        # ================================================================

        feature_completeness = (
            available_signal_count
            /
            float(
                total_possible_signal_count
            )
        )

        evidence_quality_score = (
            feature_completeness
        )

        # Known reference brand adds a modest confidence adjustment,
        # because the identity is supplied by the dataset targetlist,
        # not guessed from the ML label.

        if (
            matched_brand.lower()
            != "unknown"
        ):

            evidence_quality_score = min(
                1.0,
                evidence_quality_score
                + 0.10,
            )

        features.update(
            {

                "feature_completeness":
                    self._clamp01(
                        feature_completeness
                    ),

                "evidence_quality_score":
                    self._clamp01(
                        evidence_quality_score
                    ),
            }
        )

        return features

    # =====================================================================
    # FINALIZE
    # =====================================================================

    def _finalize_features(
        self,
        features: Dict[str, Any],
        domain_features: Dict[str, Any],
        keyword_features: Dict[str, Any],
        logo_features: Dict[str, Any],
        visual_data: Dict[str, Any],
        ocr_text: str,
    ) -> Dict[str, Any]:

        defaults = (
            self.schema.get_default_features()
        )

        canonical: Dict[str, Any] = {}

        # ================================================================
        # EXACT ML FEATURE ORDER
        # ================================================================

        for feature_name in (
            self.schema.ML_FEATURE_COLUMNS
        ):

            value = features.get(
                feature_name,
                defaults.get(
                    feature_name,
                    0.0,
                ),
            )

            canonical[
                feature_name
            ] = self._sanitize_feature_value(
                value
            )

        # ================================================================
        # NON-ML IDENTITY METADATA
        # ================================================================

        matched_brand = str(
            domain_features.get(
                "matched_brand",
                "unknown",
            )
            or "unknown"
        ).strip()

        if not matched_brand:

            matched_brand = "unknown"

        primary_brand = str(
            logo_features.get(
                "primary_brand_name",
                "unknown",
            )
            or "unknown"
        ).strip()

        if not primary_brand:

            primary_brand = "unknown"

        detected_logo_brand = str(
            logo_features.get(
                "primary_brand_name",
                "unknown",
            )
            or "unknown"
        ).strip()

        if not detected_logo_brand:

            detected_logo_brand = "unknown"

        # Do NOT call reference_brand a detected logo brand.
        #
        # If no logo was actually detected, preserve "unknown".

        if not bool(
            logo_features.get(
                "logo_detected",
                False,
            )
        ):

            detected_logo_brand = "unknown"
            primary_brand = "unknown"

        canonical[
            "matched_brand"
        ] = matched_brand

        canonical[
            "primary_brand_name"
        ] = primary_brand

        canonical[
            "detected_logo_brand"
        ] = detected_logo_brand

        # ================================================================
        # OCR BRAND NAME
        # ================================================================

        if (
            ocr_text
            and
            matched_brand.lower()
            != "unknown"
            and
            self._text_contains_brand(
                ocr_text,
                matched_brand,
            )
        ):

            canonical[
                "ocr_brand_name"
            ] = matched_brand

        else:

            canonical[
                "ocr_brand_name"
            ] = "unknown"

        # ================================================================
        # BRAND CANDIDATES
        # ================================================================

        candidates: List[str] = []

        if (
            matched_brand
            and
            matched_brand.lower()
            != "unknown"
        ):

            candidates.append(
                matched_brand
            )

        if (
            primary_brand
            and
            primary_brand.lower()
            != "unknown"
            and
            primary_brand.lower()
            not in {
                candidate.lower()
                for candidate in candidates
            }
        ):

            candidates.append(
                primary_brand
            )

        canonical[
            "detected_brand_candidates"
        ] = candidates

        # ================================================================
        # SENSITIVE KEYWORDS
        # ================================================================

        matched_keywords = (
            keyword_features.get(
                "matched_keywords",
                keyword_features.get(
                    "suspicious_keywords",
                    [],
                ),
            )
        )

        if isinstance(
            matched_keywords,
            (list, tuple, set),
        ):

            canonical[
                "matched_sensitive_keywords"
            ] = list(
                matched_keywords
            )

        elif matched_keywords:

            canonical[
                "matched_sensitive_keywords"
            ] = [
                str(
                    matched_keywords
                )
            ]

        else:

            canonical[
                "matched_sensitive_keywords"
            ] = []

        # ================================================================
        # FINAL 64-FEATURE VALIDATION
        # ================================================================

        if len(
            [
                key
                for key in canonical
                if key
                in self.schema.ML_FEATURE_COLUMNS
            ]
        ) != self.schema.get_feature_count():

            logger.error(
                (
                    "Brand extractor failed to construct "
                    "exactly %d ML features."
                ),
                self.schema.get_feature_count(),
            )

            return self._get_empty_features()

        return canonical

    # =====================================================================
    # EMPTY FEATURES
    # =====================================================================

    def _get_empty_features(
        self,
    ) -> Dict[str, Any]:

        defaults = (
            self.schema.get_default_features()
        )

        result = dict(
            defaults
        )

        # Keep semantic distance meaningful.
        if (
            "domain_brand_distance"
            in result
            and
            result[
                "domain_brand_distance"
            ]
            == 0
        ):

            result[
                "domain_brand_distance"
            ] = 1.0

        if (
            "title_brand_distance_score"
            in result
            and
            result[
                "title_brand_distance_score"
            ]
            == 0
        ):

            result[
                "title_brand_distance_score"
            ] = 1.0

        return result

    # =====================================================================
    # FEATURE COUNT
    # =====================================================================

    def get_feature_count(
        self,
    ) -> int:

        return self.schema.get_feature_count()

    # =====================================================================
    # BRAND MATCH HELPERS
    # =====================================================================

    @classmethod
    def _brand_token_in_domain(
        cls,
        domain: str,
        brand: str,
    ) -> bool:

        if (
            not domain
            or
            not brand
        ):

            return False

        brand_norm = (
            cls._normalize_brand(
                brand
            )
        )

        if not brand_norm:
            return False

        for label in str(
            domain
        ).lower().split("."):

            pieces = re.split(
                r"[^a-z0-9]+",
                label,
            )

            for piece in pieces:

                if (
                    piece
                    and
                    cls._normalize_brand(
                        piece
                    )
                    == brand_norm
                ):

                    return True

        return False

    @classmethod
    def _text_contains_brand(
        cls,
        text: str,
        brand: str,
    ) -> bool:

        if (
            not text
            or
            not brand
        ):

            return False

        brand_norm = (
            cls._normalize_brand(
                brand
            )
        )

        text_norm = (
            cls._normalize_brand(
                text
            )
        )

        if not brand_norm:

            return False

        return (
            brand_norm
            in text_norm
        )

    @classmethod
    def _count_normalized_brand_mentions(
        cls,
        text: str,
        brand: str,
    ) -> int:

        if (
            not text
            or
            not brand
        ):

            return 0

        brand_norm = (
            cls._normalize_brand(
                brand
            )
        )

        if not brand_norm:
            return 0

        text_norm = (
            cls._normalize_brand(
                text
            )
        )

        if not text_norm:
            return 0

        return int(
            text_norm.count(
                brand_norm
            )
        )

    # =====================================================================
    # SUBDOMAIN
    # =====================================================================

    @classmethod
    def _is_brand_subdomain(
        cls,
        domain: str,
        brand: str,
    ) -> bool:

        if (
            not domain
            or
            not brand
        ):

            return False

        brand_norm = (
            cls._normalize_brand(
                brand
            )
        )

        labels = [
            label
            for label
            in str(
                domain
            ).lower().split(".")
            if label
        ]

        if len(labels) <= 2:

            return False

        for label in labels[:-2]:

            normalized_label = (
                cls._normalize_brand(
                    label
                )
            )

            if (
                normalized_label
                == brand_norm
            ):

                return True

        return False

    # =====================================================================
    # EDIT DISTANCE
    # =====================================================================

    @staticmethod
    def _edit_distance(
        first: str,
        second: str,
    ) -> int:

        first = str(
            first or ""
        )

        second = str(
            second or ""
        )

        if not first:
            return len(second)

        if not second:
            return len(first)

        previous = list(
            range(
                len(second)
                + 1
            )
        )

        for i, char_a in enumerate(
            first,
            start=1,
        ):

            current = [
                i
            ]

            for j, char_b in enumerate(
                second,
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
                    +
                    (
                        0
                        if char_a
                        == char_b
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

        return int(
            previous[-1]
        )

    # =====================================================================
    # TYPO OPERATIONS
    # =====================================================================

    @classmethod
    def _typo_operations(
        cls,
        source: str,
        target: str,
    ) -> Dict[str, bool]:

        source = str(
            source or ""
        )

        target = str(
            target or ""
        )

        operations = {

            "insertion":
                False,

            "deletion":
                False,

            "substitution":
                False,

            "transposition":
                False,

            "repeated":
                False,
        }

        if (
            not source
            or
            not target
        ):

            return operations

        # ---------------------------------------------------------------
        # Repeated character
        # ---------------------------------------------------------------

        if re.search(
            r"(.)\1",
            source,
        ):

            operations[
                "repeated"
            ] = True

        # ---------------------------------------------------------------
        # One-character insertion
        # ---------------------------------------------------------------

        if (
            len(source)
            == len(target) + 1
        ):

            for index in range(
                len(source)
            ):

                candidate = (
                    source[:index]
                    +
                    source[index + 1:]
                )

                if candidate == target:

                    operations[
                        "insertion"
                    ] = True

                    break

        # ---------------------------------------------------------------
        # One-character deletion
        # ---------------------------------------------------------------

        if (
            len(target)
            == len(source) + 1
        ):

            for index in range(
                len(target)
            ):

                candidate = (
                    target[:index]
                    +
                    target[index + 1:]
                )

                if candidate == source:

                    operations[
                        "deletion"
                    ] = True

                    break

        # ---------------------------------------------------------------
        # One-character transposition
        # ---------------------------------------------------------------

        if (
            len(source)
            == len(target)
        ):

            differences = [
                index
                for index
                in range(
                    len(source)
                )
                if source[index]
                != target[index]
            ]

            if (
                len(differences)
                == 2
            ):

                first, second = (
                    differences
                )

                if (
                    source[first]
                    == target[second]
                    and
                    source[second]
                    == target[first]
                ):

                    operations[
                        "transposition"
                    ] = True

        # ---------------------------------------------------------------
        # One substitution
        # ---------------------------------------------------------------

        if (
            len(source)
            == len(target)
        ):

            difference_count = sum(
                source[index]
                != target[index]
                for index
                in range(
                    len(source)
                )
            )

            if difference_count == 1:

                operations[
                    "substitution"
                ] = True

        return operations

    # =====================================================================
    # CONFUSABLE CHARACTERS
    # =====================================================================

    @staticmethod
    def _count_confusable_characters(
        value: str,
    ) -> int:

        if not value:
            return 0

        confusable_map = {

            "0": "o",
            "1": "l",
            "3": "e",
            "4": "a",
            "5": "s",
            "7": "t",

            "@": "a",
            "$": "s",

            "а": "a",
            "е": "e",
            "і": "i",
            "о": "o",
            "р": "p",
            "с": "c",
            "х": "x",
            "у": "y",
        }

        count = 0

        for char in str(
            value
        ).lower():

            if char in confusable_map:

                count += 1

        return count

    # =====================================================================
    # KEYWORD HELPERS
    # =====================================================================

    @staticmethod
    def _has_keyword(
        keyword_features: Dict[str, Any],
        keywords: tuple[str, ...],
    ) -> bool:

        if not isinstance(
            keyword_features,
            dict,
        ):

            return False

        # Direct boolean fields.
        for keyword in keywords:

            normalized = (
                keyword
                .replace(
                    " ",
                    "_",
                )
                .lower()
            )

            candidates = (

                f"has_{normalized}_keywords",

                f"has_{normalized}",

                f"{normalized}_detected",
            )

            for key in candidates:

                if bool(
                    keyword_features.get(
                        key,
                        False,
                    )
                ):

                    return True

        # Search matched keyword collections.
        for key in (
            "matched_keywords",
            "suspicious_keywords",
            "detected_keywords",
            "keywords",
        ):

            values = keyword_features.get(
                key
            )

            if not isinstance(
                values,
                (list, tuple, set),
            ):

                continue

            normalized_values = {
                str(
                    value
                ).lower()
                for value
                in values
            }

            for keyword in keywords:

                if (
                    keyword.lower()
                    in normalized_values
                ):

                    return True

        return False

    # =====================================================================
    # VISUAL EVIDENCE
    # =====================================================================

    @staticmethod
    def _visual_evidence_available(
        visual_data: Dict[str, Any],
    ) -> bool:

        if not isinstance(
            visual_data,
            dict,
        ):

            return False

        explicit_flags = (

            "visual_data_available",

            "image_available",

            "screenshot_available",
        )

        for key in explicit_flags:

            if bool(
                visual_data.get(
                    key,
                    False,
                )
            ):

                return True

        # Actual image object/path is also evidence.
        for key in (
            "image",
            "screenshot",
            "visual_image",
            "image_path",
            "screenshot_path",
        ):

            value = visual_data.get(
                key
            )

            if value is not None:

                if isinstance(
                    value,
                    str,
                ):

                    if value.strip():

                        return True

                else:

                    return True

        return False

    # =====================================================================
    # NUMERICAL SAFETY
    # =====================================================================

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float:

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):

            return 0.0

        if (
            number != number
            or
            number == float("inf")
            or
            number == float("-inf")
        ):

            return 0.0

        return number

    @staticmethod
    def _clamp01(
        value: Any,
    ) -> float:

        number = (
            BrandFeatureExtractor._safe_float(
                value
            )
        )

        return max(
            0.0,
            min(
                1.0,
                number,
            ),
        )

    @classmethod
    def _sanitize_feature_value(
        cls,
        value: Any,
    ) -> Any:

        if isinstance(
            value,
            bool,
        ):

            return value

        if isinstance(
            value,
            (int, float),
        ):

            return cls._safe_float(
                value
            )

        try:

            return cls._safe_float(
                value
            )

        except Exception:

            return 0.0


__all__ = [
    "BrandFeatureExtractor",
]