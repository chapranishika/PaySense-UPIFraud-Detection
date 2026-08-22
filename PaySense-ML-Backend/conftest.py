"""
conftest.py — PaySense Guardian test configuration
Ensures src/ is on the Python path regardless of where pytest is invoked from.
Run from PaySense-ML-Backend/:  pytest tests/ -v
"""
import os
import sys

# Make src/ importable from any working directory
sys.path.insert(0, os.path.dirname(__file__))

# Force a deterministic auth mode for the test suite.
#
# main.py's get_current_user() bypasses JWT validation when
# APP_ENV == "development" and no Authorization header is sent (a dev
# convenience). The local .env ships with APP_ENV=development, which made
# tests/test_api.py's "no token" tests (expecting 401/403) fail whenever the
# suite picked up that .env — the failure was in the test environment, not
# in application logic. Tests must not depend on a developer's local .env,
# so pin APP_ENV to "production" here, before `main` (and therefore
# load_dotenv()) is imported by any test module. python-dotenv's
# load_dotenv() defaults to override=False, so this value wins over .env.
os.environ["APP_ENV"] = "production"
