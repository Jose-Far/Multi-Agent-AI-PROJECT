import time
from collections import defaultdict
from threading import Lock
from typing import Tuple

class InMemoryRateLimiter:
    """
    Thread-safe in-memory sliding window rate limiter.
    Limits requests per client identifier (e.g. IP address or session token).
    """
    def __init__(self, max_requests: int = 100, window_sec: int = 60):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._requests = defaultdict(list)
        self._lock = Lock()

    def is_allowed(self, client_id: str) -> Tuple[bool, int]:
        """
        Check if client is permitted to make a request.
        Returns (is_allowed, retry_after_seconds).
        """
        now = time.time()
        window_start = now - self.window_sec
        with self._lock:
            timestamps = [t for t in self._requests[client_id] if t > window_start]
            if len(timestamps) >= self.max_requests:
                self._requests[client_id] = timestamps
                retry_after = int(timestamps[0] + self.window_sec - now) + 1
                return False, max(1, retry_after)
            
            timestamps.append(now)
            self._requests[client_id] = timestamps
            return True, 0

    def reset(self):
        with self._lock:
            self._requests.clear()
