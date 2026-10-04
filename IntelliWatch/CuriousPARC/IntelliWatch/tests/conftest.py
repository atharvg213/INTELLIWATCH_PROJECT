"""
tests/conftest.py
Global pytest configuration for IntelliWatch test suites.
Ensures AUTH_ENABLED defaults to False during testing so that regression suites (Steps 1-22)
can execute unhindered, while Step 23 explicit tests enable AUTH_ENABLED on isolated environments.
"""
import pytest
from configs.settings import get_settings


@pytest.fixture(autouse=True)
def configure_test_auth_default(monkeypatch):
    """
    By default in test runs, keep AUTH_ENABLED = False so legacy endpoint tests
    run with default administrative context without needing Bearer tokens.
    Tests specifically testing Step 23 auth enable AUTH_ENABLED explicitly.
    """
    settings = get_settings()
    monkeypatch.setattr(settings, "AUTH_ENABLED", False)
