import os
import logging
import time
import math
from typing import Dict, Any, Optional

try:
    import cv2
    import numpy as np
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Specialized visual modules
from feature_extraction.visual.colors import extract_color_palette
from feature_extraction.visual.favicon import analyze_favicon
from feature_extraction.visual.layout import analyze_layout
from feature_extraction.visual.logo import detect_logo
from feature_extraction.visual.ocr import extract_ocr_text


logger = logging.getLogger(__name__)


class VisualFeatureExtractor:
    """
    Master Visual Feature Extractor.

    Produces the exact 12 canonical Visual ML features:

        1. screenshot_width
        2. screenshot_height
        3. image_entropy
        4. dominant_color_count
        5. text_density
        6. image_density
        7. form_area_ratio
        8. button_count
        9. logo_detected
        10. login_form_detected
        11. layout_complexity
        12. blank_area_ratio

    It also preserves the existing diagnostic features used elsewhere
    in the project.
    """

    CANONICAL_FEATURES = (
        "screenshot_width",
        "screenshot_height",
        "image_entropy",
        "dominant_color_count",
        "text_density",
        "image_density",
        "form_area_ratio",
        "button_count",
        "logo_detected",
        "login_form_detected",
        "layout_complexity",
        "blank_area_ratio",
    )

    def __init__(self):
        logger.info(
            "Initialized Advanced VisualFeatureExtractor engine."
        )

    # =====================================================================
    # PUBLIC EXTRACTION API
    # =====================================================================

    def extract(
        self,
        raw_data: Dict[str, Any],
    ) -> Dict[str, Any]:

        start_time = time.time()

        if not isinstance(raw_data, dict):
            logger.warning(
                "VisualFeatureExtractor received invalid raw_data."
            )
            return self._empty_features()

        visual_telemetry = raw_data.get(
            "visual",
            {}
        )

        html_telemetry = raw_data.get(
            "html",
            {}
        )

        if not isinstance(
            visual_telemetry,
            dict,
        ):
            visual_telemetry = {}

        if not isinstance(
            html_telemetry,
            dict,
        ):
            html_telemetry = {}

        # ---------------------------------------------------------------
        # Support both:
        #
        # {
        #     "visual": {...}
        # }
        #
        # and:
        #
        # {
        #     "visual": {
        #         "data": {...}
        #     }
        # }
        # ---------------------------------------------------------------

        visual_telemetry = self._unwrap_data(
            visual_telemetry
        )

        html_telemetry = self._unwrap_data(
            html_telemetry
        )

        screenshot_path = (
            self._find_screenshot_path(
                visual_telemetry
            )
        )

        favicon_url = (
            html_telemetry.get(
                "favicon_url"
            )
            or visual_telemetry.get(
                "favicon_url"
            )
        )

        # =================================================================
        # INITIAL RESULT
        # =================================================================

        features = {

            # -------------------------------------------------------------
            # 12 CANONICAL ML FEATURES
            # -------------------------------------------------------------

            "screenshot_width": 0.0,
            "screenshot_height": 0.0,
            "image_entropy": 0.0,
            "dominant_color_count": 0.0,
            "text_density": 0.0,
            "image_density": 0.0,
            "form_area_ratio": 0.0,
            "button_count": 0.0,
            "logo_detected": False,
            "login_form_detected": False,
            "layout_complexity": 0.0,
            "blank_area_ratio": 0.0,

            # -------------------------------------------------------------
            # EXISTING DIAGNOSTIC FEATURES
            # -------------------------------------------------------------

            "logo_confidence": 0.0,
            "primary_brand_name": "unknown",

            "ocr_word_count": 0,
            "ocr_text_snippet": "",

            "contains_login_keywords": False,
            "contains_urgency_keywords": False,

            "dominant_colors": [],
            "dominant_hex": [],

            "favicon_present": False,
            "favicon_download_success": False,

            "layout_complexity_score": 0.0,

            "input_field_count": 0,
            "has_login_form_visual": False,

            "extraction_time_ms": 0.0,
        }

        # =================================================================
        # LOAD SCREENSHOT
        # =================================================================

        image = None

        if screenshot_path:
            image = self._load_image(
                screenshot_path
            )

        if image is None:

            logger.warning(
                "Screenshot unavailable: %s",
                screenshot_path,
            )

        else:

            logger.info(
                "Processing screenshot: %s",
                screenshot_path,
            )

            # =============================================================
            # 1. SCREENSHOT DIMENSIONS
            # =============================================================

            try:

                height, width = image.shape[:2]

                features[
                    "screenshot_width"
                ] = float(width)

                features[
                    "screenshot_height"
                ] = float(height)

            except Exception as exc:

                logger.error(
                    "Screenshot dimension extraction failed: %s",
                    exc,
                )

            # =============================================================
            # 2. IMAGE ENTROPY
            # =============================================================

            try:

                features[
                    "image_entropy"
                ] = self._calculate_image_entropy(
                    image
                )

            except Exception as exc:

                logger.error(
                    "Image entropy extraction failed: %s",
                    exc,
                )

            # =============================================================
            # 3. IMAGE DENSITY
            # =============================================================

            try:

                features[
                    "image_density"
                ] = self._calculate_image_density(
                    image
                )

            except Exception as exc:

                logger.error(
                    "Image density extraction failed: %s",
                    exc,
                )

            # =============================================================
            # 4. BLANK AREA RATIO
            # =============================================================

            try:

                features[
                    "blank_area_ratio"
                ] = self._calculate_blank_area_ratio(
                    image
                )

            except Exception as exc:

                logger.error(
                    "Blank area extraction failed: %s",
                    exc,
                )

        # =================================================================
        # FAVICON
        # =================================================================

        if favicon_url:

            try:

                favicon_features = (
                    analyze_favicon(
                        favicon_url,
                        screenshot_path,
                    )
                )

                for key in (
                    "favicon_present",
                    "favicon_md5",
                    "favicon_sha256",
                    "favicon_size_bytes",
                    "favicon_download_success",
                ):

                    if key in favicon_features:

                        features[key] = (
                            favicon_features[key]
                        )

            except Exception as exc:

                logger.error(
                    "Favicon analysis failed: %s",
                    exc,
                )

        # =================================================================
        # DEEP SCREENSHOT PROCESSING
        # =================================================================

        if screenshot_path and image is not None:

            # =============================================================
            # 5. LOGO DETECTION
            # =============================================================

            try:

                logo_results = detect_logo(
                    screenshot_path
                )

                features[
                    "logo_detected"
                ] = bool(
                    logo_results.get(
                        "logo_detected",
                        False,
                    )
                )

                features[
                    "logo_confidence"
                ] = float(
                    logo_results.get(
                        "confidence",
                        0.0,
                    )
                    or 0.0
                )

                features[
                    "primary_brand_name"
                ] = logo_results.get(
                    "primary_brand_name",
                    "unknown",
                )

            except Exception as exc:

                logger.error(
                    "Logo detection failed: %s",
                    exc,
                )

            # =============================================================
            # 6. OCR
            # =============================================================

            ocr_results = {}

            try:

                ocr_results = extract_ocr_text(
                    screenshot_path
                )

                features[
                    "ocr_word_count"
                ] = int(
                    ocr_results.get(
                        "ocr_word_count",
                        0,
                    )
                    or 0
                )

                features[
                    "ocr_text_snippet"
                ] = ocr_results.get(
                    "ocr_text_snippet",
                    "",
                )

                features[
                    "contains_login_keywords"
                ] = bool(
                    ocr_results.get(
                        "contains_login_keywords",
                        False,
                    )
                )

                features[
                    "contains_urgency_keywords"
                ] = bool(
                    ocr_results.get(
                        "contains_urgency_keywords",
                        False,
                    )
                )

                # OCR-derived text density
                features[
                    "text_density"
                ] = self._calculate_text_density(
                    image=image,
                    ocr_results=ocr_results,
                )

                # OCR brand fallback
                ocr_brand = ocr_results.get(
                    "ocr_detected_brand",
                    "unknown",
                )

                ocr_confidence = float(
                    ocr_results.get(
                        "ocr_brand_confidence",
                        0.0,
                    )
                    or 0.0
                )

                if (
                    features[
                        "primary_brand_name"
                    ] == "unknown"
                    and ocr_brand != "unknown"
                ):

                    features[
                        "primary_brand_name"
                    ] = ocr_brand

                    features[
                        "logo_confidence"
                    ] = ocr_confidence

                    features[
                        "logo_detected"
                    ] = True

            except Exception as exc:

                logger.error(
                    "OCR extraction failed: %s",
                    exc,
                )

            # =============================================================
            # 7. COLOR PALETTE
            # =============================================================

            try:

                color_results = (
                    extract_color_palette(
                        screenshot_path
                    )
                )

                features[
                    "dominant_colors"
                ] = color_results.get(
                    "dominant_colors",
                    [],
                )

                features[
                    "dominant_hex"
                ] = color_results.get(
                    "dominant_hex",
                    [],
                )

                # THIS WAS MISSING BEFORE.
                features[
                    "dominant_color_count"
                ] = float(
                    len(
                        features[
                            "dominant_colors"
                        ]
                    )
                )

            except Exception as exc:

                logger.error(
                    "Color extraction failed: %s",
                    exc,
                )

            # =============================================================
            # 8. LAYOUT
            # =============================================================

            try:

                layout_results = (
                    analyze_layout(
                        screenshot_path
                    )
                )

                layout_complexity = float(
                    layout_results.get(
                        "layout_complexity_score",
                        0.0,
                    )
                    or 0.0
                )

                button_count = int(
                    layout_results.get(
                        "button_count",
                        0,
                    )
                    or 0
                )

                input_count = int(
                    layout_results.get(
                        "input_field_count",
                        0,
                    )
                    or 0
                )

                features[
                    "layout_complexity_score"
                ] = layout_complexity

                features[
                    "layout_complexity"
                ] = layout_complexity

                features[
                    "button_count"
                ] = float(
                    max(
                        0,
                        button_count,
                    )
                )

                features[
                    "input_field_count"
                ] = max(
                    0,
                    input_count,
                )

                # Keep existing cross-modal login heuristic.
                has_login_keywords = (
                    features[
                        "contains_login_keywords"
                    ]
                )

                login_detected = (
                    (
                        input_count >= 2
                    )
                    or (
                        input_count >= 1
                        and has_login_keywords
                    )
                    or (
                        input_count >= 1
                        and button_count >= 1
                    )
                )

                features[
                    "has_login_form_visual"
                ] = bool(
                    login_detected
                )

                features[
                    "login_form_detected"
                ] = bool(
                    login_detected
                )

                # =========================================================
                # FORM AREA RATIO
                # =========================================================

                features[
                    "form_area_ratio"
                ] = self._calculate_form_area_ratio(
                    image=image,
                    input_count=input_count,
                    button_count=button_count,
                    login_detected=login_detected,
                )

            except Exception as exc:

                logger.error(
                    "Layout analysis failed: %s",
                    exc,
                )

        # =================================================================
        # FINAL NORMALIZATION
        # =================================================================

        features[
            "text_density"
        ] = self._clip_ratio(
            features[
                "text_density"
            ]
        )

        features[
            "image_density"
        ] = self._clip_ratio(
            features[
                "image_density"
            ]
        )

        features[
            "form_area_ratio"
        ] = self._clip_ratio(
            features[
                "form_area_ratio"
            ]
        )

        features[
            "blank_area_ratio"
        ] = self._clip_ratio(
            features[
                "blank_area_ratio"
            ]
        )

        features[
            "dominant_color_count"
        ] = float(
            max(
                0,
                features[
                    "dominant_color_count"
                ],
            )
        )

        features[
            "button_count"
        ] = float(
            max(
                0,
                features[
                    "button_count"
                ],
            )
        )

        features[
            "layout_complexity"
        ] = float(
            max(
                0.0,
                features[
                    "layout_complexity"
                ],
            )
        )

        # =================================================================
        # CANONICAL VALIDATION
        # =================================================================

        canonical_count = sum(
            1
            for feature_name in self.CANONICAL_FEATURES
            if feature_name in features
        )

        features[
            "canonical_feature_count"
        ] = canonical_count

        features[
            "expected_feature_count"
        ] = len(
            self.CANONICAL_FEATURES
        )

        features[
            "schema_complete"
        ] = (
            canonical_count
            == len(
                self.CANONICAL_FEATURES
            )
        )

        # =================================================================
        # EXTRACTION TIME
        # =================================================================

        extraction_time = (
            time.time()
            - start_time
        ) * 1000.0

        features[
            "extraction_time_ms"
        ] = round(
            extraction_time,
            2,
        )

        logger.info(
            (
                "Visual extraction completed | "
                "canonical=%d/12 | "
                "resolution=%sx%s | "
                "entropy=%.4f | "
                "colors=%d | "
                "text_density=%.4f | "
                "image_density=%.4f | "
                "form_area=%.4f | "
                "blank_area=%.4f"
            ),
            canonical_count,
            int(
                features[
                    "screenshot_width"
                ]
            ),
            int(
                features[
                    "screenshot_height"
                ]
            ),
            features[
                "image_entropy"
            ],
            int(
                features[
                    "dominant_color_count"
                ]
            ),
            features[
                "text_density"
            ],
            features[
                "image_density"
            ],
            features[
                "form_area_ratio"
            ],
            features[
                "blank_area_ratio"
            ],
        )

        return features

    # =====================================================================
    # SCREENSHOT LOADING
    # =====================================================================

    @staticmethod
    def _load_image(
        screenshot_path: str,
    ) -> Optional[np.ndarray]:

        if not screenshot_path:
            return None

        if not os.path.exists(
            screenshot_path
        ):
            return None

        if not CV_AVAILABLE:
            logger.warning(
                "OpenCV/NumPy unavailable."
            )
            return None

        try:

            image = cv2.imread(
                screenshot_path,
                cv2.IMREAD_COLOR,
            )

            if image is None:
                return None

            return image

        except Exception as exc:

            logger.error(
                "Could not load screenshot: %s",
                exc,
            )

            return None

    # =====================================================================
    # DIMENSION / PATH HELPERS
    # =====================================================================

    @staticmethod
    def _find_screenshot_path(
        visual_telemetry: Dict[str, Any],
    ) -> Optional[str]:

        possible_keys = (
            "screenshot_path",
            "screenshot_file",
            "image_path",
            "screenshot",
            "file_path",
            "path",
        )

        for key in possible_keys:

            value = visual_telemetry.get(
                key
            )

            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
                and os.path.exists(
                    value
                )
            ):
                return value

        return None

    @staticmethod
    def _unwrap_data(
        data: Dict[str, Any],
    ) -> Dict[str, Any]:

        current = data

        for _ in range(3):

            nested = current.get(
                "data"
            )

            if not isinstance(
                nested,
                dict,
            ):
                break

            current = nested

        return current

    # =====================================================================
    # IMAGE ENTROPY
    # =====================================================================

    @staticmethod
    def _calculate_image_entropy(
        image: np.ndarray,
    ) -> float:

        if image is None or image.size == 0:
            return 0.0

        if len(
            image.shape
        ) == 3:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        else:

            gray = image

        histogram = cv2.calcHist(
            [gray],
            [0],
            None,
            [256],
            [0, 256],
        )

        histogram = (
            histogram
            / max(
                float(
                    histogram.sum()
                ),
                1.0,
            )
        )

        probabilities = (
            histogram[
                histogram > 0
            ]
        )

        entropy = -float(
            np.sum(
                probabilities
                * np.log2(
                    probabilities
                )
            )
        )

        return max(
            0.0,
            entropy,
        )

    # =====================================================================
    # IMAGE DENSITY
    # =====================================================================

    @staticmethod
    def _calculate_image_density(
        image: np.ndarray,
    ) -> float:

        if image is None or image.size == 0:
            return 0.0

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        # Detect strong texture/edges.
        edges = cv2.Canny(
            gray,
            80,
            160,
        )

        edge_density = (
            float(
                cv2.countNonZero(
                    edges
                )
            )
            / float(
                max(
                    1,
                    edges.size,
                )
            )
        )

        # This is a visual texture-density estimate.
        # It is deliberately capped and not interpreted as exact DOM
        # <img> coverage.
        return float(
            np.clip(
                edge_density * 2.0,
                0.0,
                1.0,
            )
        )

    # =====================================================================
    # TEXT DENSITY
    # =====================================================================

    @staticmethod
    def _calculate_text_density(
        image: np.ndarray,
        ocr_results: Dict[str, Any],
    ) -> float:

        if image is None or image.size == 0:
            return 0.0

        word_count = int(
            ocr_results.get(
                "ocr_word_count",
                0,
            )
            or 0
        )

        if word_count <= 0:
            return 0.0

        height, width = image.shape[:2]

        if height <= 0 or width <= 0:
            return 0.0

        # Conservative normalized estimate.
        #
        # The OCR count is converted into a density rather than directly
        # treated as an area measurement.
        density = (
            min(
                word_count,
                150,
            )
            / 150.0
        ) * 0.75

        return float(
            np.clip(
                density,
                0.0,
                1.0,
            )
        )

    # =====================================================================
    # FORM AREA
    # =====================================================================

    @staticmethod
    def _calculate_form_area_ratio(
        image: np.ndarray,
        input_count: int,
        button_count: int,
        login_detected: bool,
    ) -> float:

        if image is None:
            return 0.0

        if (
            input_count <= 0
            and button_count <= 0
        ):
            return 0.0

        height, width = image.shape[:2]

        total_area = float(
            max(
                1,
                height * width,
            )
        )

        # Conservative UI element area estimate.
        #
        # Inputs are assumed to occupy approximately 4% each and buttons
        # approximately 2.5% each before the cap.
        estimated_area = (
            input_count * 0.04
            + button_count * 0.025
        )

        if login_detected:
            estimated_area *= 1.10

        return float(
            np.clip(
                estimated_area,
                0.0,
                0.50,
            )
        )

    # =====================================================================
    # BLANK AREA
    # =====================================================================

    @staticmethod
    def _calculate_blank_area_ratio(
        image: np.ndarray,
    ) -> float:

        if image is None or image.size == 0:
            return 0.0

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        # Threshold nearly-uniform background pixels.
        #
        # We calculate local variance so that large flat webpage
        # backgrounds are treated as blank while text/image regions
        # contribute less.
        blur = cv2.GaussianBlur(
            gray,
            (15, 15),
            0,
        )

        local_difference = cv2.absdiff(
            gray,
            blur,
        )

        threshold = (
            local_difference < 8
        )

        blank_ratio = (
            float(
                np.count_nonzero(
                    threshold
                )
            )
            / float(
                max(
                    1,
                    threshold.size,
                )
            )
        )

        return float(
            np.clip(
                blank_ratio,
                0.0,
                1.0,
            )
        )

    # =====================================================================
    # HELPERS
    # =====================================================================

    @staticmethod
    def _clip_ratio(
        value: Any,
    ) -> float:

        try:
            value = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if not math.isfinite(
            value
        ):
            return 0.0

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    # =====================================================================
    # EMPTY RESULT
    # =====================================================================

    def _empty_features(
        self,
    ) -> Dict[str, Any]:

        features = {

            "screenshot_width": 0.0,
            "screenshot_height": 0.0,
            "image_entropy": 0.0,
            "dominant_color_count": 0.0,
            "text_density": 0.0,
            "image_density": 0.0,
            "form_area_ratio": 0.0,
            "button_count": 0.0,
            "logo_detected": False,
            "login_form_detected": False,
            "layout_complexity": 0.0,
            "blank_area_ratio": 0.0,

            "logo_confidence": 0.0,
            "primary_brand_name": "unknown",
            "ocr_word_count": 0,
            "ocr_text_snippet": "",
            "contains_login_keywords": False,
            "contains_urgency_keywords": False,
            "dominant_colors": [],
            "dominant_hex": [],
            "favicon_present": False,
            "favicon_download_success": False,
            "layout_complexity_score": 0.0,
            "input_field_count": 0,
            "has_login_form_visual": False,
            "extraction_time_ms": 0.0,

            "canonical_feature_count": 12,
            "expected_feature_count": 12,
            "schema_complete": True,
        }

        return features