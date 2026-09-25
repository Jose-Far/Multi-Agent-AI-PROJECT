import logging
import time
import concurrent.futures
from typing import Dict, Any

from .validator import URLValidator
from .url_collector import URLCollector
from .html_collector import HTMLCollector
from .screenshot_collector import ScreenshotCollector
from .ssl_collector import SSLCollector
from .dns_collector import DNSCollector
from .whois_collector import WHOISCollector
from .threat_collector import ThreatCollector

logger = logging.getLogger(__name__)

class CollectionManager:
    """
    Central Orchestrator for the Data Collection Layer.
    
    Architectural Upgrades:
    - Resilient Dictionary Lookups: Uses safety checks (.get("success", False)) 
      to completely prevent KeyError crashes if any collector returns malformed data.
    - Parallel Execution: Utilizes ThreadPoolExecutor to run independent telemetry 
      collectors concurrently, drastically reducing pipeline latency.
    - Fault Isolation: Implements strict try-catch boundaries around individual 
      collectors so a single module's failure doesn't crash the entire ML pipeline.
    """
    def __init__(self, max_workers: int = 6):
        # Initialize validation and collection modules
        self.validator = URLValidator()
        self.url_collector = URLCollector()
        self.html_collector = HTMLCollector()
        self.screenshot_collector = ScreenshotCollector()
        self.ssl_collector = SSLCollector()
        self.dns_collector = DNSCollector()
        self.whois_collector = WHOISCollector()
        self.threat_collector = ThreatCollector()
        
        # Max concurrent threads for executing independent collectors
        self.max_workers = max_workers

    def _safe_execute(self, collector_func, *args, **kwargs) -> Dict[str, Any]:
        """
        Wrapper to execute a collector safely. Catches any uncaught module-level 
        exceptions and normalizes them into the standard error payload schema.
        """
        try:
            return collector_func(*args, **kwargs)
        except Exception as e:
            module_name = collector_func.__self__.__class__.__name__
            logger.error(f"Critical failure in {module_name}: {str(e)}")
            return {
                "success": False, 
                "error": f"Unhandled {module_name} exception: {str(e)}", 
                "data": {}
            }

    def execute_collection(self, raw_url: str) -> Dict[str, Any]:
        """
        Executes the full multi-vector data collection pass.
        
        Args:
            raw_url (str): The initial target URL provided by the user/system.
            
        Returns:
            Dict[str, Any]: Fully aggregated telemetry payload.
        """
        start_time = time.time()
        logger.info(f"Starting Data Collection Pass for: {raw_url}")

        # ---------------------------------------------------------
        # Step 1: Input Validation & Sanitization
        # ---------------------------------------------------------
        is_valid, sanitized_url = self.validator.sanitize_and_validate(raw_url)
        if not is_valid:
            logger.warning(f"URL Validation failed for: {raw_url}")
            return {
                "status": "error", 
                "message": "Invalid URL structure or blocked by anti-SSRF policies.", 
                "execution_time_sec": round(time.time() - start_time, 3),
                "data": {}
            }

        # ---------------------------------------------------------
        # Step 2: URL / Redirect Execution (Sequential)
        # ---------------------------------------------------------
        # Must execute FIRST and sequentially to determine the actual final destination
        url_metrics = self._safe_execute(self.url_collector.collect, sanitized_url)
        
        # SAFE DICTIONARY LOOKUP FIX: Using .get() prevents KeyError 'success' crashes
        url_success = url_metrics.get("success", False) if isinstance(url_metrics, dict) else False
        if not url_success:
            logger.warning("URL collection pass failed or returned malformed structure.")
            error_msg = url_metrics.get("error", {}).get("message", "Network unreachable") if isinstance(url_metrics, dict) else "Network unreachable"
            return {
                "status": "failed", 
                "message": error_msg, 
                "execution_time_sec": round(time.time() - start_time, 3),
                "data": {"url_analysis": url_metrics}
            }

        # Extract the resolved final URL for the remaining infrastructure collectors
        final_target = url_metrics.get("final_url") or sanitized_url
        logger.info(f"Resolved final target: {final_target}. Launching parallel threads...")

        # ---------------------------------------------------------
        # Step 3 & 4: Parallel Telemetry Collection
        # ---------------------------------------------------------
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Map future tasks to their respective JSON keys
            future_to_key = {
                executor.submit(self._safe_execute, self.html_collector.collect, final_target): "html_analysis",
                executor.submit(self._safe_execute, self.screenshot_collector.collect, final_target): "screenshot_analysis",
                executor.submit(self._safe_execute, self.ssl_collector.collect, final_target): "ssl_analysis",
                executor.submit(self._safe_execute, self.dns_collector.collect, final_target): "dns_analysis",
                executor.submit(self._safe_execute, self.whois_collector.collect, final_target): "whois_analysis",
                executor.submit(self._safe_execute, self.threat_collector.collect, final_target): "threat_analysis"
            }

            # As each collector finishes (regardless of order), map it to the results dictionary
            import sys, os
            if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from security.config import SecurityConfig
            
            global_timeout = SecurityConfig.REQUEST_TIMEOUT * 2
            
            try:
                for future in concurrent.futures.as_completed(future_to_key, timeout=global_timeout):
                    key = future_to_key[future]
                    try:
                        results[key] = future.result()
                    except Exception as e:
                        logger.error(f"Thread pool exception for {key}: {e}")
                        results[key] = {"success": False, "error": f"Thread crash: {str(e)}", "data": {}}
                    
                    if time.time() - start_time > global_timeout:
                        logger.error("Hard abort: Pipeline exceeded maximum allowed time.")
                        break
            except concurrent.futures.TimeoutError:
                logger.error(f"Hard abort: Pipeline exceeded global timeout of {global_timeout}s.")
            
            for future, key in future_to_key.items():
                if key not in results:
                    results[key] = {"success": False, "error": "Global pipeline timeout exceeded", "data": {}}
                    
        # ---------------------------------------------------------
        # Step 5: Memory Optimization & Aggregation
        # ---------------------------------------------------------
        # Remove the massive Base64 string from the central JSON payload to save RAM.
        # Your Vision AI Agent / Storage Manager will read the physical file from disk instead.
        if results.get("screenshot_analysis", {}).get("success"):
            results["screenshot_analysis"].get("data", {}).pop("base64_image_full", None)

        total_execution_time = round(time.time() - start_time, 3)
        logger.info(f"Data Collection Pass completed in {total_execution_time}s")

        return {
            "status": "success",
            "message": "Data collection pass completed successfully.",
            "execution_time_sec": total_execution_time,
            "data": {
                "url_analysis": url_metrics,
                "html_analysis": results.get("html_analysis", {}),
                "screenshot_analysis": results.get("screenshot_analysis", {}),
                "ssl_analysis": results.get("ssl_analysis", {}),
                "dns_analysis": results.get("dns_analysis", {}),
                "whois_analysis": results.get("whois_analysis", {}),
                "threat_analysis": results.get("threat_analysis", {})
            }
        }