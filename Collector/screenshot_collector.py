import os
import time
import logging
from typing import Dict, Any
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException, TimeoutException

logger = logging.getLogger(__name__)

class ScreenshotCollector:
    """
    Advanced visual telemetry collector. Spins up a stealth-configured headless 
    Chrome browser to render the target URL exactly as a human victim would see it.
    Crucial for bypassing text-based obfuscation and gathering data for the Visual AI Agent.
    """
    def __init__(self, output_dir: str = "storage/screenshots", timeout: int = 30):
        self.output_dir = output_dir
        self.timeout = timeout
        
        # Ensure the secure storage directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def collect(self, target_url: str) -> Dict[str, Any]:
        """
        Renders the target URL and captures a 1080p viewport screenshot.
        
        Args:
            target_url (str): The destination URL to capture.
            
        Returns:
            Dict[str, Any]: Payload containing the file path, full Base64 string 
                            (for AI vision models), and a truncated preview.
        """
        if not target_url.startswith(('http://', 'https://')):
            target_url = 'https://' + target_url

        chrome_options = self._build_stealth_options()
        driver = None
        
        try:
            # Initialize WebDriver with auto-managed binary
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Enforce strict timeouts to prevent pipeline hanging
            driver.set_page_load_timeout(self.timeout)
            driver.set_script_timeout(self.timeout)
            
            logger.info(f"ScreenshotCollector navigating to: {target_url}")
            driver.get(target_url)
            
            # Dynamic Wait: Wait until the browser reports the DOM is 'complete'
            WebDriverWait(driver, self.timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            
            # Fallback buffer for heavy asynchronous JavaScript (React/Vue phishing kits)
            # that load visual assets *after* the DOM is technically complete
            time.sleep(2.5) 
            
            # Generate a filesystem-safe filename
            parsed_url = urlparse(target_url)
            domain = parsed_url.netloc.replace(":", "_") or "unknown_domain"
            timestamp = int(time.time())
            filename = f"screenshot_{domain}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Capture the visual telemetry
            driver.save_screenshot(filepath)
            base64_img = driver.get_screenshot_as_base64()
            
            # Validate capture success
            if not base64_img:
                raise ValueError("WebDriver returned an empty Base64 screenshot payload.")
            
            return {
                "success": True,
                "error": None,
                "data": {
                    "screenshot_path": filepath,
                    "base64_image_preview": f"data:image/png;base64,{base64_img[:50]}...[TRUNCATED]",
                    "base64_image_full": base64_img,  # Direct feed for Vision LLMs
                    "resolution": "1920x1080",
                    "format": "png",
                    "timestamp": timestamp
                }
            }
            
        except TimeoutException:
            logger.warning(f"Timeout while attempting to render {target_url}")
            return self._build_error_payload("Browser timed out while waiting for the page to render.")
        except WebDriverException as e:
            logger.error(f"WebDriver execution failed for {target_url}: {str(e)}")
            return self._build_error_payload(f"WebDriver execution failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during screenshot capture for {target_url}: {str(e)}")
            return self._build_error_payload(f"Unexpected error during screenshot capture: {str(e)}")
        finally:
            # Absolutely critical to prevent zombie Chrome processes and memory leaks
            if driver is not None:
                try:
                    driver.quit()
                except Exception as e:
                    logger.debug(f"Error during WebDriver teardown: {str(e)}")

    def _build_stealth_options(self) -> Options:
        """
        Constructs the ChromeOptions payload with anti-bot evasion techniques 
        and strict resource management settings.
        """
        options = Options()
        
        # 1. Modern Headless Mode (Harder to detect than the legacy --headless flag)
        options.add_argument("--headless=new")
        
        # 2. Anti-Bot Evasion (Spoof normal user behavior)
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
        
        # 3. Security Overrides (Phishing sites often have invalid SSL or strange redirects)
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--allow-running-insecure-content")
        # options.add_argument("--disable-web-security") # Phase 10: Re-enable Same Origin Policy
        
        # 4. Resource & Layout Management
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--hide-scrollbars")
        # options.add_argument("--no-sandbox") # Phase 10: Enforce Chrome Sandbox
        options.add_argument("--site-per-process") # Phase 10: Strict site isolation
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--mute-audio")
        
        # 5. Optimization: Do not block execution waiting for heavy background sub-resources
        options.page_load_strategy = 'eager'
        
        return options

    def _build_error_payload(self, message: str) -> Dict[str, Any]:
        """Constructs standardized error response structure."""
        return {
            "success": False,
            "error": message,
            "data": {}
        }