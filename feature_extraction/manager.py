import copy
import logging
from typing import Dict, Any

from .url.extractor import URLExtractor
from agents.html_agent.extractor import HTMLFeatureExtractor
from .ssl.extractor import SSLExtractor
from .visual.extractor import VisualFeatureExtractor
from .dns.extractor import DNSFeatureExtractor
from .threat.extractor import ThreatFeatureExtractor

from .storage import FeatureStorage


logger = logging.getLogger(__name__)


class FeatureManager:
    """
    Master Orchestrator for the Feature Extraction Layer.

    Produces the complete Unified Feature Vector used by:

        Multi-Agent Classifiers
                ↓
        Decision Fusion Engine
                ↓
        AI Reasoning Engine
                ↓
        Explainable AI Layer

    IMPORTANT HTML CONTRACT
    -----------------------
    The HTML Agent uses a finalized 50-feature XGBoost model.

    Therefore:

        unified_vector["html_features"]

    MUST contain exactly 50 HTML features.

    A small structured HTML summary may also be created for
    telemetry/debugging, but it MUST NOT replace the 50-feature
    ML vector.
    """

    # ------------------------------------------------------------------
    # Finalized HTML feature count
    # ------------------------------------------------------------------

    HTML_FEATURE_COUNT = 50

    def __init__(self):

        self.storage = FeatureStorage()

        # ---------------------------------------------------------------
        # Specialized extractors
        # ---------------------------------------------------------------

        self.visual_extractor = VisualFeatureExtractor()

        self.dns_extractor = DNSFeatureExtractor()

        self.threat_extractor = ThreatFeatureExtractor()

    # ===================================================================
    # BUILD UNIFIED FEATURE VECTOR
    # ===================================================================

    def build_unified_vector(
        self,
        raw_collected_data: Dict[str, Any]
    ) -> Dict[str, Any]:

        """
        Build the complete Unified Feature Vector.

        HTML output is guaranteed to use the complete 50-feature
        HTML schema.
        """

        if not isinstance(
            raw_collected_data,
            dict
        ):

            logger.error(
                "FeatureManager received invalid telemetry payload."
            )

            return {}

        # ----------------------------------------------------------------
        # Work on a copy.
        # ----------------------------------------------------------------

        working_data = copy.deepcopy(
            raw_collected_data
        )

        # ----------------------------------------------------------------
        # Resolve payload.
        # ----------------------------------------------------------------

        payload = working_data.get(
            "data",
            working_data
        )

        if not isinstance(
            payload,
            dict
        ):

            logger.error(
                "FeatureManager payload is not a valid dictionary."
            )

            return {}

        # =================================================================
        # 1. TARGET URL
        # =================================================================

        target_url = self._resolve_target_url(
            working_data,
            payload
        )

        if not target_url:

            logger.error(
                "FeatureManager Error: No target URL found "
                "in telemetry payload."
            )

            return {}

        logger.info(
            "Feature extraction started for: %s",
            target_url
        )

        # =================================================================
        # 2. MASTER UNIFIED VECTOR
        # =================================================================

        unified_vector = {

            "metadata": {
                "target_url": target_url
            },

            "url_features": {},

            "html_features": {},

            "visual_features": {},

            "ssl_features": {},

            "dns_features": {},

            "threat_features": {}
        }

        # =================================================================
        # 3. URL FEATURES
        # =================================================================

        try:

            url_extractor = URLExtractor(
                target_url
            )

            unified_vector[
                "url_features"
            ] = url_extractor.extract()

            logger.info(
                "URL features successfully extracted."
            )

        except Exception as e:

            logger.error(
                "URL Extraction failed: %s",
                str(e),
                exc_info=True
            )

                # =================================================================
        # 4. HTML FEATURES - FINALIZED 50 FEATURE PIPELINE
        # =================================================================

        try:

            html_analysis = payload.get(
                "html_analysis",
                {}
            )

            if not isinstance(
                html_analysis,
                dict
            ):
                html_analysis = {}

            html_data = html_analysis.get(
                "data",
                html_analysis
            )

            if not isinstance(
                html_data,
                dict
            ):
                html_data = {}

            # -------------------------------------------------------------
            # Get raw HTML
            # -------------------------------------------------------------

            raw_html_content = html_data.get(
                "raw_html",
                ""
            )

            if raw_html_content is None:
                raw_html_content = ""

            raw_html_content = str(
                raw_html_content
            )

            if not raw_html_content.strip():

                raise ValueError(
                    "No raw HTML content available for "
                    "50-feature HTML extraction."
                )

            # -------------------------------------------------------------
            # IMPORTANT
            #
            # Use the FINALIZED HTML AI Agent extractor.
            #
            # DO NOT use:
            #
            #     feature_extraction.html.extractor.HTMLExtractor
            #
            # because that extractor produces the older 21-feature
            # representation.
            #
            # The production HTML Agent extractor is:
            #
            #     agents.html_agent.extractor.HTMLFeatureExtractor
            #
            # Its contract is:
            #
            #     HTMLFeatureExtractor(base_url)
            #     .extract(html_content)
            #
            # and it produces the finalized 50-feature vector.
            # -------------------------------------------------------------

            html_extractor = HTMLFeatureExtractor(
                base_url=target_url
            )

            extracted_html_features = (
                html_extractor.extract(
                    raw_html_content
                )
            )

            # -------------------------------------------------------------
            # Validate extractor output
            # -------------------------------------------------------------

            if not isinstance(
                extracted_html_features,
                dict
            ):

                raise ValueError(
                    "HTMLFeatureExtractor returned "
                    "a non-dictionary result."
                )

            # -------------------------------------------------------------
            # Remove non-ML metadata if the extractor exposes any.
            #
            # The HTMLFeatureSchema is the authoritative contract.
            # -------------------------------------------------------------

            from agents.html_agent.feature_schema import (
                HTMLFeatureSchema
            )

            expected_html_features = (
                HTMLFeatureSchema.get_schema()
            )

            # -------------------------------------------------------------
            # Build the exact production vector in canonical order.
            #
            # This is important because the model was trained with
            # the authoritative 50-feature schema.
            # -------------------------------------------------------------

            normalized_html_features = {}

            for feature_name in expected_html_features:

                value = (
                    extracted_html_features.get(
                        feature_name,
                        0.0
                    )
                )

                normalized_html_features[
                    feature_name
                ] = value

            # -------------------------------------------------------------
            # Exact schema validation
            # -------------------------------------------------------------

            extracted_feature_count = len(
                normalized_html_features
            )

            if extracted_feature_count != 50:

                raise ValueError(
                    "HTML feature schema mismatch. "
                    f"Expected 50 features, found "
                    f"{extracted_feature_count}."
                )

            # -------------------------------------------------------------
            # Check that the extractor actually supplied all 50.
            #
            # Do not silently hide a broken extractor.
            # -------------------------------------------------------------

            missing_features = [
                feature
                for feature in expected_html_features
                if feature not in extracted_html_features
            ]

            if missing_features:

                raise ValueError(
                    "HTML extractor did not provide all "
                    "50 required features. Missing: "
                    f"{missing_features}"
                )

            # -------------------------------------------------------------
            # Store COMPLETE 50-feature ML vector.
            # -------------------------------------------------------------

            unified_vector[
                "html_features"
            ] = normalized_html_features

            logger.info(
                "HTML extractor returned %d features.",
                len(
                    extracted_html_features
                )
            )

            logger.info(
                "HTML 50-feature schema validation PASSED."
            )

            # -------------------------------------------------------------
            # Lightweight HTML summary.
            #
            # This is for telemetry/reporting only.
            # It MUST NOT replace html_features.
            # -------------------------------------------------------------

            html_features = (
                unified_vector[
                    "html_features"
                ]
            )

            structured_html_summary = {

                "form_count":
                    self._safe_int(
                        html_features.get(
                            "form_count",
                            0
                        )
                    ),

                "password_fields":
                    self._safe_int(
                        html_features.get(
                            "password_field_count",
                            0
                        )
                    ),

                "iframe_count":
                    self._safe_int(
                        html_features.get(
                            "iframe_count",
                            0
                        )
                    ),

                "hidden_inputs":
                    self._safe_int(
                        html_features.get(
                            "hidden_input_count",
                            0
                        )
                    ),

                "external_scripts":
                    self._safe_int(
                        html_features.get(
                            "external_scripts_count",
                            0
                        )
                    ),

                "external_links":
                    self._safe_int(
                        html_features.get(
                            "external_links_count",
                            0
                        )
                    ),

                "cross_domain_form_actions":
                    self._safe_int(
                        html_features.get(
                            "cross_domain_form_actions",
                            0
                        )
                    ),

                "page_title":
                    str(
                        html_data.get(
                            "page_title",
                            ""
                        )
                    )
            }

            # -------------------------------------------------------------
            # Preserve lightweight summary.
            # -------------------------------------------------------------

            original_html_analysis = payload.get(
                "html_analysis"
            )

            if isinstance(
                original_html_analysis,
                dict
            ):

                original_html_analysis[
                    "structured_summary"
                ] = structured_html_summary

                nested_data = (
                    original_html_analysis.get(
                        "data"
                    )
                )

                if isinstance(
                    nested_data,
                    dict
                ):

                    nested_data.pop(
                        "raw_html",
                        None
                    )

                else:

                    original_html_analysis.pop(
                        "raw_html",
                        None
                    )

            logger.info(
                "HTML features successfully extracted: "
                "50/50"
            )

        except Exception as e:

            logger.error(
                "HTML Extraction failed: %s",
                str(e),
                exc_info=True
            )

            # Never create a fake 9-feature vector.
            unified_vector[
                "html_features"
            ] = {}

        # =================================================================
        # 5. VISUAL FEATURES
        # =================================================================

        try:

            screenshot_analysis = payload.get(
                "screenshot_analysis",
                {}
            )

            if not isinstance(
                screenshot_analysis,
                dict
            ):

                screenshot_analysis = {}

            visual_data = (
                screenshot_analysis.get(
                    "data",
                    screenshot_analysis
                )
            )

            if not isinstance(
                visual_data,
                dict
            ):

                visual_data = {}

            html_visual_data = payload.get(
                "html_analysis",
                {}
            )

            if not isinstance(
                html_visual_data,
                dict
            ):

                html_visual_data = {}

            html_visual_data = (
                html_visual_data.get(
                    "data",
                    html_visual_data
                )
            )

            if not isinstance(
                html_visual_data,
                dict
            ):

                html_visual_data = {}

            visual_raw_input = {

                "visual":
                    visual_data,

                "html":
                    html_visual_data
            }

            unified_vector[
                "visual_features"
            ] = self.visual_extractor.extract(
                visual_raw_input
            )

            logger.info(
                "Visual features successfully extracted."
            )

        except Exception as e:

            logger.error(
                "Visual Extraction failed: %s",
                str(e),
                exc_info=True
            )

        # =================================================================
        # 6. SSL FEATURES
        # =================================================================

        try:

            ssl_analysis = payload.get(
                "ssl_analysis",
                {}
            )

            if not isinstance(
                ssl_analysis,
                dict
            ):

                ssl_analysis = {}

            ssl_extractor = SSLExtractor(
                ssl_analysis
            )

            unified_vector[
                "ssl_features"
            ] = ssl_extractor.extract()

            logger.info(
                "SSL features successfully extracted."
            )

        except Exception as e:

            logger.error(
                "SSL Extraction failed: %s",
                str(e),
                exc_info=True
            )

        # =================================================================
        # 7. DNS FEATURES
        # =================================================================

        try:

            dns_analysis = payload.get(
                "dns_analysis",
                {}
            )

            if not isinstance(
                dns_analysis,
                dict
            ):

                dns_analysis = {}

            dns_data = dns_analysis.get(
                "data",
                dns_analysis
            )

            if not isinstance(
                dns_data,
                dict
            ):

                dns_data = {}

            unified_vector[
                "dns_features"
            ] = self.dns_extractor.extract_features(
                dns_data
            )

            logger.info(
                "DNS features successfully extracted."
            )

        except Exception as e:

            logger.error(
                "DNS Extraction failed: %s",
                str(e),
                exc_info=True
            )

        # =================================================================
        # 8. THREAT FEATURES
        # =================================================================

        try:

            unified_vector[
                "threat_features"
            ] = self.threat_extractor.extract_features(
                payload
            )

            logger.info(
                "Threat intelligence features successfully extracted."
            )

        except Exception as e:

            logger.error(
                "Threat Intelligence Extraction failed: %s",
                str(e),
                exc_info=True
            )

        # =================================================================
        # 11. FINAL HTML VALIDATION
        # =================================================================

        html_feature_count = len(
            unified_vector.get(
                "html_features",
                {}
            )
        )

        unified_vector[
            "metadata"
        ].update({

            "feature_extraction_status":
                "completed",

            "agents_executed": [
                "url",
                "html",
                "visual",
                "ssl",
                "dns",
                "threat"
            ],

            "html_feature_count":
                html_feature_count,

            "html_feature_schema_valid":
                (
                    html_feature_count
                    == self.HTML_FEATURE_COUNT
                )
        })

        if (
            html_feature_count
            == self.HTML_FEATURE_COUNT
        ):

            logger.info(
                "HTML 50-feature schema validation PASSED."
            )

        else:

            logger.warning(
                "HTML 50-feature schema validation FAILED. "
                "Expected %d, found %d.",
                self.HTML_FEATURE_COUNT,
                html_feature_count
            )

        logger.info(
            "Unified Feature Vector successfully built."
        )

        return unified_vector

    # ===================================================================
    # PROCESS AND STORE
    # ===================================================================

    def process_and_store(
        self,
        raw_collected_data: Dict[str, Any]
    ) -> Dict[str, Any]:

        """
        Build the Unified Feature Vector and persist it.
        """

        try:

            unified_vector = (
                self.build_unified_vector(
                    raw_collected_data
                )
            )

            if not unified_vector:

                logger.warning(
                    "FeatureManager produced an empty unified vector."
                )

                return {}

            payload = raw_collected_data.get(
                "data",
                raw_collected_data
            )

            if not isinstance(
                payload,
                dict
            ):

                payload = {}

            target_url = (
                self._resolve_target_url(
                    raw_collected_data,
                    payload
                )
            )

            if not target_url:

                target_url = (
                    "unknown_target"
                )

            saved_path = (
                self.storage.save_unified_features(
                    target_url,
                    unified_vector
                )
            )

            if saved_path:

                logger.info(
                    "Unified features saved successfully to: %s",
                    saved_path
                )

                unified_vector[
                    "storage_path"
                ] = saved_path

            return unified_vector

        except Exception as e:

            logger.error(
                "Process and Store failed: %s",
                str(e),
                exc_info=True
            )

            return {}

    # ===================================================================
    # TARGET URL RESOLUTION
    # ===================================================================

    @staticmethod
    def _resolve_target_url(
        raw_data: Dict[str, Any],
        payload: Dict[str, Any]
    ) -> str:

        """
        Resolve target URL from the collection payload.
        """

        url_analysis = payload.get(
            "url_analysis",
            {}
        )

        if not isinstance(
            url_analysis,
            dict
        ):

            url_analysis = {}

        url_data = url_analysis.get(
            "data",
            url_analysis
        )

        if not isinstance(
            url_data,
            dict
        ):

            url_data = {}

        target_url = (

            raw_data.get(
                "url"
            )

            or url_data.get(
                "final_url"
            )

            or url_data.get(
                "initial_url"
            )

            or url_data.get(
                "url"
            )

            or ""
        )

        return str(
            target_url
        ).strip()

    # ===================================================================
    # SAFE INTEGER
    # ===================================================================

    @staticmethod
    def _safe_int(
        value: Any,
        default: int = 0
    ) -> int:

        """
        Safely convert a value to integer.
        """

        try:

            return int(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return default