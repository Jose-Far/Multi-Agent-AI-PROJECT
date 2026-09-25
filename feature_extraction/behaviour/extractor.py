import logging
from typing import Dict, Any, Optional

from .redirects import RedirectFeatureExtractor
from .forms import FormBehaviorFeatureExtractor
from .downloads import DownloadBehaviorFeatureExtractor
from .popups import PopupBehaviorFeatureExtractor

logger = logging.getLogger(__name__)


class BehaviourFeatureExtractor:
    """
    Master Orchestrator for the Behaviour Feature Extraction Layer.

    Synthesizes:
        - Redirect chain behaviour
        - Form submission / exfiltration behaviour
        - Automatic download / HTML smuggling behaviour
        - Popup and browser-locking behaviour

    Produces a unified ML-ready behavioural feature vector.
    """

    def __init__(self):
        self.redirect_extractor = RedirectFeatureExtractor()
        self.form_extractor = FormBehaviorFeatureExtractor()
        self.download_extractor = DownloadBehaviorFeatureExtractor()
        self.popup_extractor = PopupBehaviorFeatureExtractor()

    def extract_features(
        self,
        raw_data: Dict[str, Any],
        html_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete behavioural feature extraction pipeline.

        Args:
            raw_data:
                Aggregated scan telemetry containing URL and HTML analysis.

            html_features:
                Optional pre-extracted HTML features. These are used as the
                single source of truth for shared HTML-derived fields such as
                form_count and has_password_field.

        Returns:
            Unified behavioural feature vector.
        """

        if not isinstance(raw_data, dict) or not raw_data:
            logger.warning(
                "Behaviour Feature Extractor received empty telemetry. "
                "Returning baseline."
            )
            return self._get_empty_features()

        html_features = (
            html_features
            if isinstance(html_features, dict)
            else {}
        )

        try:
            # ================================================================
            # 1. Extract URL Analysis
            # ================================================================

            url_analysis = raw_data.get("url_analysis", {})

            if not isinstance(url_analysis, dict):
                url_analysis = {}

            # Support nested URL analysis structures
            if isinstance(url_analysis.get("data"), dict):
                url_data = url_analysis["data"]
            else:
                url_data = url_analysis

            redirect_history = url_data.get(
                "redirect_history",
                []
            )

            if not isinstance(redirect_history, list):
                redirect_history = []

            final_url = str(
                url_data.get("final_url", "")
            )

            domain = str(
                url_data.get("domain_resolved")
                or url_data.get("domain")
                or final_url
            )

            # Backward-compatible redirect count
            redirect_count = url_data.get(
                "redirect_count",
                len(redirect_history)
            )

            try:
                redirect_count = int(redirect_count)
            except (TypeError, ValueError):
                redirect_count = len(redirect_history)

            # ================================================================
            # 2. Extract HTML Analysis
            # ================================================================

            html_analysis = raw_data.get(
                "html_analysis",
                {}
            )

            if not isinstance(html_analysis, dict):
                html_analysis = {}

            # Support both:
            # {
            #     "html_analysis": {...}
            # }
            #
            # and:
            # {
            #     "html_analysis": {
            #         "data": {...}
            #     }
            # }
            html_data = html_analysis.get(
                "data",
                html_analysis
            )

            if not isinstance(html_data, dict):
                html_data = {}

            forms = html_data.get(
                "forms",
                []
            )

            if not isinstance(forms, list):
                forms = []

            try:
                script_count = int(
                    html_data.get(
                        "script_count",
                        0
                    )
                )
            except (TypeError, ValueError):
                script_count = 0

            script_sources = html_data.get(
                "external_scripts",
                []
            )

            if not isinstance(script_sources, list):
                script_sources = []

            page_html = str(
                html_data.get(
                    "html_content",
                    ""
                )
            )

            # ================================================================
            # 3. Execute Behaviour Sub-Modules
            # ================================================================

            redirect_features = self._safe_extract(
                self.redirect_extractor.extract,
                redirect_history,
                final_url
            )

            form_features = self._safe_extract(
                self.form_extractor.extract,
                forms,
                domain
            )

            download_features = self._safe_extract(
                self.download_extractor.extract,
                script_sources,
                page_html
            )

            popup_features = self._safe_extract(
                self.popup_extractor.extract,
                script_count,
                page_html
            )

            # ================================================================
            # 4. Merge Feature Vectors
            # ================================================================

            unified_features = {
                **redirect_features,
                **form_features,
                **download_features,
                **popup_features
            }

            # ================================================================
            # 5. Backward-Compatible Core Behaviour Features
            # ================================================================

            unified_features.setdefault(
                "redirect_count",
                redirect_count
            )

            unified_features.setdefault(
                "form_count",
                len(forms)
            )

            # ================================================================
            # 6. Synchronize HTML Features
            # ================================================================

            # HTML extractor remains the authoritative source for these
            # shared features.

            if "form_count" in html_features:
                unified_features["form_count"] = self._safe_int(
                    html_features.get("form_count"),
                    default=len(forms)
                )

            elif "form_count" not in unified_features:
                unified_features["form_count"] = len(forms)

            if "has_password_field" in html_features:
                unified_features["has_password_field"] = bool(
                    html_features.get("has_password_field")
                )

            else:
                unified_features.setdefault(
                    "has_password_field",
                    False
                )

            # ================================================================
            # 7. Backward-Compatible Behaviour Indicators
            # ================================================================

            unified_features.setdefault(
                "has_automatic_download_trigger",
                False
            )

            unified_features.setdefault(
                "has_browser_locking_tactics",
                False
            )

            unified_features.setdefault(
                "has_suspicious_behaviour",
                False
            )

            # Cross-domain redirect fallback
            if "has_cross_domain_redirect" not in unified_features:
                unified_features["has_cross_domain_redirect"] = (
                    redirect_count > 3
                )

            # ================================================================
            # 8. Composite Behaviour Risk
            # ================================================================

            unified_features = self._compute_behaviour_risk(
                unified_features
            )

            return unified_features

        except Exception as e:
            logger.error(
                "Critical error during Behaviour Feature Extraction "
                f"pipeline: {str(e)}",
                exc_info=True
            )

            return self._get_empty_features()

    # ========================================================================
    # SAFE EXTRACTION
    # ========================================================================

    def _safe_extract(
        self,
        extractor_func,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Isolate sub-module failures so one failed extractor does not
        terminate the complete behavioural pipeline.
        """

        try:
            result = extractor_func(
                *args,
                **kwargs
            )

            if isinstance(result, dict):
                return result

            logger.warning(
                "Extractor %s returned non-dict result.",
                extractor_func.__qualname__
            )

            return {}

        except Exception as e:
            logger.debug(
                "Sub-module execution failed in %s: %s",
                extractor_func.__qualname__,
                e,
                exc_info=True
            )

            return {}

    # ========================================================================
    # RISK CALCULATION
    # ========================================================================

    def _compute_behaviour_risk(
        self,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate the overall behavioural threat score.

        The score combines:
            - Redirect abuse
            - Form exfiltration
            - HTML smuggling
            - Browser locking
            - Popup evasion
            - Suspicious form actions
        """

        risk_score = 0.0

        # --------------------------------------------------------------------
        # Critical Behaviour
        # --------------------------------------------------------------------

        if features.get(
            "posts_to_free_cloud_service",
            False
        ):
            risk_score += 60

        if features.get(
            "is_redirect_loop_detected",
            False
        ):
            risk_score += 50

        if features.get(
            "has_html_smuggling_signatures",
            False
        ):
            risk_score += 60

        if features.get(
            "has_browser_locking_tactics",
            False
        ):
            risk_score += 50

        # --------------------------------------------------------------------
        # High-Risk Behaviour
        # --------------------------------------------------------------------

        if features.get(
            "has_cross_domain_form_action",
            False
        ):
            risk_score += 35

        if features.get(
            "has_open_redirect_pattern",
            False
        ):
            risk_score += 30

        if features.get(
            "has_mailto_form_action",
            False
        ):
            risk_score += 40

        if features.get(
            "has_interaction_blocking",
            False
        ):
            risk_score += 25

        # --------------------------------------------------------------------
        # Automatic Downloads
        # --------------------------------------------------------------------

        if features.get(
            "has_automatic_download_trigger",
            False
        ):
            risk_score += 30

        if features.get(
            "critical_payload_extension_detected",
            False
        ):
            risk_score += 40

        if features.get(
            "suspicious_download_extension_detected",
            False
        ):
            risk_score += 20

        # --------------------------------------------------------------------
        # Popup / Evasion Risk
        # --------------------------------------------------------------------

        popup_score = self._safe_float(
            features.get(
                "popup_evasion_score",
                0.0
            )
        )

        risk_score += popup_score * 20

        # --------------------------------------------------------------------
        # Form Exfiltration Risk
        # --------------------------------------------------------------------

        form_exfiltration_score = self._safe_float(
            features.get(
                "form_exfiltration_risk_score",
                0
            )
        )

        risk_score += form_exfiltration_score * 0.4

        # --------------------------------------------------------------------
        # Redirect Behaviour
        # --------------------------------------------------------------------

        redirect_count = self._safe_int(
            features.get(
                "redirect_count",
                0
            )
        )

        if redirect_count > 3:
            risk_score += 15

        if redirect_count > 7:
            risk_score += 20

        # --------------------------------------------------------------------
        # Password Collection
        # --------------------------------------------------------------------

        if features.get(
            "has_password_field",
            False
        ):
            risk_score += 10

        # --------------------------------------------------------------------
        # Normalize Score
        # --------------------------------------------------------------------

        features["behaviour_risk_score"] = min(
            100,
            int(round(risk_score))
        )

        # Master triage flag
        features["has_suspicious_behaviour"] = (
            features["behaviour_risk_score"] >= 60
        )

        return features

    # ========================================================================
    # DEFAULT FEATURE VECTOR
    # ========================================================================

    def _get_empty_features(self) -> Dict[str, Any]:
        """
        Return a standardized zero-value behavioural feature vector.

        This guarantees a stable schema even when extraction fails.
        """

        return {
            # ----------------------------------------------------------------
            # Redirects
            # ----------------------------------------------------------------
            "redirect_count": 0,
            "has_excessive_redirects": False,
            "has_cross_domain_redirect": False,
            "has_open_redirect_pattern": False,
            "is_redirect_loop_detected": False,
            "has_suspicious_status_codes": False,
            "unique_hop_domains_count": 0,
            "average_hop_latency_sec": 0.0,

            # ----------------------------------------------------------------
            # Forms
            # ----------------------------------------------------------------
            "form_count": 0,
            "has_password_field": False,
            "has_empty_or_suspicious_sfh": False,
            "has_cross_domain_form_action": False,
            "has_mailto_form_action": False,
            "has_unencrypted_form_action": False,
            "posts_to_free_cloud_service": False,
            "uses_javascript_form_action": False,
            "form_exfiltration_risk_score": 0,

            # ----------------------------------------------------------------
            # Downloads
            # ----------------------------------------------------------------
            "has_automatic_download_trigger": False,
            "has_html_smuggling_signatures": False,
            "suspicious_download_extension_detected": False,
            "critical_payload_extension_detected": False,
            "detected_dangerous_extensions": [],
            "download_execution_risk_score": 0,

            # ----------------------------------------------------------------
            # Popups / Browser Locking
            # ----------------------------------------------------------------
            "has_window_open_events": False,
            "has_forced_modal_dialogs": False,
            "has_fake_auth_prompts": False,
            "has_browser_locking_tactics": False,
            "has_interaction_blocking": False,
            "popup_evasion_score": 0.0,

            # ----------------------------------------------------------------
            # Composite Behaviour
            # ----------------------------------------------------------------
            "behaviour_risk_score": 0,
            "has_suspicious_behaviour": False
        }

    # ========================================================================
    # TYPE-SAFE HELPERS
    # ========================================================================

    @staticmethod
    def _safe_int(
        value: Any,
        default: int = 0
    ) -> int:
        """Safely convert a value to integer."""

        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0
    ) -> float:
        """Safely convert a value to float."""

        try:
            return float(value)
        except (TypeError, ValueError):
            return default