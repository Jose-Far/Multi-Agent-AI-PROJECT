import time
import logging
from performance.profiler import monitor

logger = logging.getLogger('phishdec.performance')

class PerformanceMetric:
    def __init__(self, operation, component):
        self.operation = operation
        self.component = component
        self.start_time = None
        self.end_time = None
        self.duration_ms = 0
        self.status = "pending"
        
    def start(self):
        self.start_time = time.time()
        
    def stop(self, status="success"):
        if not self.start_time:
            return
        self.end_time = time.time()
        self.duration_ms = int((self.end_time - self.start_time) * 1000)
        self.status = status
        
        if self.operation == "analysis":
            monitor.record_agent_time(self.component, self.duration_ms, self.status)
        elif self.operation == "total_analysis":
            monitor.record_analysis(self.duration_ms)
            
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"PERF | {self.component} | {self.operation} | {self.duration_ms}ms | {self.status}")
