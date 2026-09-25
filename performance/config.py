class PerformanceConfig:
    # Timeouts for agents (in seconds)
    AGENT_TIMEOUTS = {
        'URL_AI_Agent': 5,
        'HTML_AI_Agent': 10,
        'SSL_AI_Agent': 8,
        'DNS_AI_Agent': 5,
        'Visual_AI_Agent': 15,
        'Threat_Intel_Agent': 10
    }
    
    # Global Request limits
    MAX_CONCURRENT_ANALYSIS = 4
    REQUEST_TIMEOUT = 30
    
    # Caching
    THREAT_CACHE_TTL_SEC = 3600
    
    # Pagination
    DEFAULT_PAGE_SIZE = 25
    MAX_PAGE_SIZE = 100
    
    # Logging
    PERFORMANCE_LOGGING = True

