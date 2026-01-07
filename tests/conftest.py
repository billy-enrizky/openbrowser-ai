"""Pytest configuration and fixtures for openbrowser test suite.

This module provides shared configuration and fixtures used across the entire
test suite. It sets up the Python path to allow importing from openbrowser
and defines common fixtures for browser sessions, mock objects, and test data.

Configuration:
    - Adds src/ directory to Python path for test imports
    - Configures pytest-asyncio for async test support
    - Sets up logging for test visibility

Fixtures:
    The fixtures are defined in individual test files as needed since this
    project uses a modular approach to test fixtures. See individual test
    modules for their specific fixtures.

Example:
    To use shared imports in tests::

        from openbrowser import BrowserAgent, BrowserSession
        from openbrowser.browser.profile import BrowserProfile

Path Setup:
    The src directory is added to sys.path to enable imports like:
    ``from openbrowser.browser.session import BrowserSession``
"""

import sys
from pathlib import Path

# Add the src directory to the path so tests can import using openbrowser
# This allows consistent import paths across all test modules
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path.parent))
