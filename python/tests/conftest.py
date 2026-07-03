"""pytest configuration for the spikecorec test-suite.

Its mere presence marks ``python/tests`` as the test root so that sibling
modules (``helpers.py``) import cleanly. Tests rely on pytest's built-in
``tmp_path`` fixture for any ``.spire`` output, so nothing is written into the
repository — temp files live under pytest's per-test temporary directory and are
cleaned up automatically.
"""
