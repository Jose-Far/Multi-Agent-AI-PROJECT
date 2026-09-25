from bs4 import BeautifulSoup
import logging

class FormFeatureExtractor:
    """
    Analyzes HTML form structures, input types, action attributes, 
    and checks for credential-harvesting indicators (e.g., password fields, external actions).
    """
    def __init__(self, soup: BeautifulSoup, base_domain: str):
        self.soup = soup
        self.base_domain = base_domain.lower()
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_features(self) -> dict:
        """
        Scans all forms and inputs in the DOM.
        """
        forms = self.soup.find_all('form')
        form_count = len(forms)
        
        has_password_field = False
        external_form_action = False
        hidden_input_count = 0
        input_count = 0

        for form in forms:
            # Check form action endpoint
            action = form.get('action', '').strip().lower()
            if action:
                # If action points to an external domain not matching base domain
                if action.startswith('http://') or action.startswith('https://'):
                    if self.base_domain not in action:
                        external_form_action = True

            # Analyze input fields inside forms
            inputs = form.find_all(['input', 'textarea', 'select'])
            input_count += len(inputs)

            for inp in inputs:
                input_type = inp.get('type', '').lower()
                
                if input_type == 'password':
                    has_password_field = True
                
                if input_type == 'hidden':
                    hidden_input_count += 1

        # Also check standalone password fields outside explicit <form> tags
        if not has_password_field:
            if self.soup.find('input', {'type': 'password'}):
                has_password_field = True

        return {
            "form_count": form_count,
            "input_count": input_count,
            "has_password_field": has_password_field,
            "external_form_action": external_form_action,
            "hidden_input_count": hidden_input_count
        }