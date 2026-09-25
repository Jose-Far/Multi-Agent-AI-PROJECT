import logging
import re
from typing import Dict, Any, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class FormBehaviorFeatureExtractor:
    """
    Advanced Submission Field Handler (SFH) and Form Behavior Extractor.
    
    Analyzes HTML form actions to detect cross-domain data exfiltration, 
    abuse of free cloud infrastructure for credential harvesting, unencrypted 
    transmissions, and dynamic JavaScript-driven evasion tactics.
    """
    def __init__(self):
        # High-risk free cloud services and form builders heavily abused by phishers 
        # to host backend credential-harvesting scripts (C2 servers).
        self.abused_cloud_services = [
            "docs.google.com/forms", "formspree.io", "netlify.app", "herokuapp.com", 
            "vercel.app", "firebaseapp.com", "000webhostapp.com", "typeform.com",
            "jotform.com", "sendgrid.net", "weebly.com"
        ]

        # Regex to strip ports from netloc for clean domain comparison
        self.port_stripper = re.compile(r':\d+$')

    def _get_base_domain(self, netloc: str) -> str:
        """Safely extracts a proxy for the base domain to evaluate cross-domain SFH."""
        clean_netloc = self.port_stripper.sub('', netloc.lower())
        parts = clean_netloc.split('.')
        if len(parts) > 2:
            if parts[-2] in ['co', 'com', 'org', 'net', 'gov', 'edu'] and len(parts[-1]) == 2:
                return '.'.join(parts[-3:])
            return '.'.join(parts[-2:])
        return clean_netloc

    def extract(self, forms: List[Dict[str, Any]], page_domain: str) -> Dict[str, Any]:
        """
        Extracts complex behavioral metrics from form SFH targets.
        
        Args:
            forms (List[Dict]): List of form objects extracted from the HTML DOM.
            page_domain (str): The primary URL/domain of the hosted page.
            
        Returns:
            Dict[str, Any]: Form behavior and evasion feature vector.
        """
        features = {
            "form_count": 0,
            "has_empty_or_suspicious_sfh": False,
            "has_cross_domain_form_action": False,
            "has_mailto_form_action": False,
            "has_unencrypted_form_action": False,
            "posts_to_free_cloud_service": False,
            "uses_javascript_form_action": False,
            "form_exfiltration_risk_score": 0  # 0 to 100 based on severity of SFH anomalies
        }

        if not forms or not isinstance(forms, list):
            return features

        try:
            features["form_count"] = len(forms)
            parsed_page_domain = self._get_base_domain(urlparse(page_domain).netloc)
            risk_score = 0

            for form in forms:
                raw_action = str(form.get("action", "")).strip()
                action_lower = raw_action.lower()

                # 1. Empty, Self-Referential, or Placeholder SFH
                # Many phishing kits leave action="#" and use obfuscated JS to steal credentials.
                if not action_lower or action_lower in ["#", "about:blank", "javascript:void(0)"]:
                    features["has_empty_or_suspicious_sfh"] = True
                    risk_score += 30
                    
                # 2. Dynamic JavaScript Action Evasion
                # e.g. action="javascript:submitForm()"
                elif action_lower.startswith("javascript:"):
                    features["uses_javascript_form_action"] = True
                    risk_score += 40

                # 3. Direct Mail Exfiltration (mailto:) - Classic low-tech phishing
                elif action_lower.startswith("mailto:"):
                    features["has_mailto_form_action"] = True
                    risk_score += 80  # Legitimate modern sites almost NEVER use mailto: for forms

                # 4. HTTP / HTTPS Exfiltration Analysis
                elif action_lower.startswith(("http://", "https://", "//")):
                    
                    # Check for Unencrypted Data Transmission (HTTP)
                    if action_lower.startswith("http://"):
                        features["has_unencrypted_form_action"] = True
                        risk_score += 50
                        
                    parsed_action = urlparse(raw_action if not raw_action.startswith("//") else f"https:{raw_action}")
                    action_base_domain = self._get_base_domain(parsed_action.netloc)
                    action_path = parsed_action.path

                    # Check for Cross-Domain Data Exfiltration
                    if action_base_domain and action_base_domain != parsed_page_domain:
                        # Before flagging, check if it's posting to an abused free cloud provider
                        full_action_url = f"{action_base_domain}{action_path}"
                        if any(cloud in full_action_url for cloud in self.abused_cloud_services):
                            features["posts_to_free_cloud_service"] = True
                            risk_score += 90  # Extremely high risk (e.g. Bank login posting to Google Forms)
                        else:
                            features["has_cross_domain_form_action"] = True
                            risk_score += 60  # Sending data to a completely different 3rd party domain

            # Cap the risk score at 100
            features["form_exfiltration_risk_score"] = min(100, risk_score)

        except Exception as e:
            logger.error(f"Error during Form Behavior Feature Extraction: {str(e)}")

        return features