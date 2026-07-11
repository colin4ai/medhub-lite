"""Structured logging and lightweight process-level service metrics."""
import json
import logging
import threading
import time
from collections import Counter


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "method", "route", "status_code", "latency_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, separators=(",", ":"))


def configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


class RuntimeMetrics:
    def __init__(self):
        self.started_at = time.time()
        self._lock = threading.Lock()
        self.requests = Counter()
        self.total_latency_ms = 0.0

    def record(self, status_code: int, latency_ms: float):
        with self._lock:
            self.requests["total"] += 1
            self.requests[f"status_{status_code // 100}xx"] += 1
            self.total_latency_ms += latency_ms

    def snapshot(self):
        with self._lock:
            total = self.requests["total"]
            return {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "requests": dict(self.requests),
                "average_latency_ms": round(self.total_latency_ms / total, 1) if total else 0.0,
            }


runtime_metrics = RuntimeMetrics()
