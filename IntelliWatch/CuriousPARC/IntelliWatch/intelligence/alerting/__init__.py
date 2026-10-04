"""
intelligence/alerting/__init__.py
Centralized safety alerting module.
"""
from intelligence.alerting.alert_engine import AlertEngine, get_alert_engine

__all__ = ["AlertEngine", "get_alert_engine"]
