import os

class SecurityConfig:
    ALLOWED_SCHEMES = {'http', 'https'}
    BLOCK_LOCALHOST = True
    BLOCK_PRIVATE_NETWORKS = True
    REQUEST_TIMEOUT = 15
    MAX_REDIRECTS = 5
    MAX_RESPONSE_SIZE = 5 * 1024 * 1024
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS = 120  # max requests
    RATE_LIMIT_WINDOW_SEC = 60 # per 60 seconds
    
    # HTTP Security Headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "X-XSS-Protection": "1; mode=block",
        "Content-Security-Policy": "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:; frame-ancestors 'none';"
    }
