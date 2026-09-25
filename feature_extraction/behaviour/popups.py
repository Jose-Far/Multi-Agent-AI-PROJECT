import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PopupBehaviorFeatureExtractor:
    """
    Advanced Modal, Evasion, and Browser Trapping Extractor.
    
    Scans scripts and event handlers for intrusive modal popups, fake authentication 
    boxes, browser-locking mechanisms (breaking the 'Back' button), and user-interaction 
    blocking (e.g., disabling right-click or text selection to hinder analysis).
    """
    def __init__(self):
        # Window manipulation & Pop-unders (Often used to spawn hidden C2 or ad frames)
        self.window_manipulation = [
            'window.open', 'showmodaldialog', 'window.blur()', 
            'window.moveto', 'window.resizeto'
        ]

        # Browser trapping / locking (Used in Tech Support Scams & forced compliance)
        self.browser_trapping = [
            'onbeforeunload', 'onunload', 'history.pushstate', 
            'history.replacestate', 'requestfullscreen'
        ]

        # Fake native OS/Browser prompts (Used to spoof HTTP Basic Auth)
        self.fake_alerts = [
            'alert(', 'confirm(', 'prompt(', 'window.prompt'
        ]

        # Anti-Analysis / User Evasion (Prevents victim from copying text or inspecting DOM)
        self.interaction_blocking = [
            'oncontextmenu', 'oncopy', 'onpaste', 'ondragstart', 
            'onselectstart', 'preventdefault()'
        ]

    def extract(self, script_count: int, page_html: str) -> Dict[str, Any]:
        """
        Extracts popup, modal, and browser-hijacking execution metrics.
        
        Args:
            script_count (int): Total number of script tags on the page.
            page_html (str): Raw HTML document string.
            
        Returns:
            Dict[str, Any]: Popup and evasion behavior feature vector.
        """
        features = {
            "has_window_open_events": False,
            "has_forced_modal_dialogs": False,
            "has_fake_auth_prompts": False,
            "has_browser_locking_tactics": False,
            "has_interaction_blocking": False,
            "popup_evasion_score": 0.0
        }

        try:
            html_lower = (page_html or "").lower()
            risk_score = 0

            # 1. Window Manipulation & Pop-Unders
            if any(term in html_lower for term in self.window_manipulation):
                features["has_window_open_events"] = True
                risk_score += 20
                # Heavy penalty if they try to resize/move or push the window behind (blur)
                if "window.blur()" in html_lower or "window.moveto" in html_lower:
                    risk_score += 30

            # 2. Browser Trapping & Tab Hijacking
            if any(term in html_lower for term in self.browser_trapping):
                features["has_browser_locking_tactics"] = True
                features["has_forced_modal_dialogs"] = True  # Backwards compatibility
                risk_score += 40

            # 3. Fake Native Prompts / Alerts
            if any(term in html_lower for term in self.fake_alerts):
                features["has_fake_auth_prompts"] = True
                risk_score += 15

            # 4. Anti-Analysis / Interaction Blocking
            if any(term in html_lower for term in self.interaction_blocking):
                features["has_interaction_blocking"] = True
                risk_score += 25

            # 5. Contextual Heuristic: Script Volume vs. Aggressive Behaviors
            # If a site has very few scripts but exhibits highly aggressive locking/popup behavior,
            # it is almost certainly a dedicated, lightweight phishing or scam landing page.
            if script_count < 5 and risk_score >= 40:
                risk_score += 20

            # Normalize the final risk score to a 0.0 - 1.0 float scale
            features["popup_evasion_score"] = round(min(100.0, float(risk_score)) / 100.0, 4)

        except Exception as e:
            logger.error(f"Error during Popup Behavior Feature Extraction: {str(e)}")

        return features