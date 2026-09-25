import threading

class PerformanceMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.analysis_times = []
        self.agent_times = {
            'URL_AI_Agent': [],
            'HTML_AI_Agent': [],
            'SSL_AI_Agent': [],
            'DNS_AI_Agent': [],
            'Visual_AI_Agent': [],
            'Threat_Intel_Agent': []
        }
        self.errors = 0
        self.timeouts = 0
        self.total_requests = 0

    def record_analysis(self, duration_ms):
        with self.lock:
            self.total_requests += 1
            self.analysis_times.append(duration_ms)
            if len(self.analysis_times) > 1000:
                self.analysis_times.pop(0)

    def record_agent_time(self, agent_name, duration_ms, status):
        with self.lock:
            if agent_name in self.agent_times:
                self.agent_times[agent_name].append(duration_ms)
                if len(self.agent_times[agent_name]) > 1000:
                    self.agent_times[agent_name].pop(0)
            if status == "error":
                self.errors += 1
            elif status == "timeout":
                self.timeouts += 1

    def get_stats(self):
        with self.lock:
            avg_analysis = sum(self.analysis_times) / len(self.analysis_times) if self.analysis_times else 0
            
            agent_stats = []
            for agent, times in self.agent_times.items():
                avg = sum(times) / len(times) if times else 0
                agent_stats.append({
                    "name": agent.replace("_AI_Agent", "").replace("_Agent", ""),
                    "avg_time_ms": int(avg),
                    "status": "Healthy"
                })
            
            return {
                "avg_analysis_time_ms": int(avg_analysis),
                "total_requests": self.total_requests,
                "error_rate": round(self.errors / max(1, self.total_requests) * 100, 2),
                "agent_metrics": agent_stats
            }

monitor = PerformanceMonitor()
