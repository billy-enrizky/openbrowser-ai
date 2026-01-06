"""Integrations module for third-party service automation.

This module provides integration helpers for common third-party services
like Gmail, simplifying browser-based automation of these platforms.

Available Integrations:
    GmailIntegration: Helper methods for Gmail automation.

Example:
    ```python
    from openbrowser.integrations import GmailIntegration

    gmail = GmailIntegration(browser_session)
    await gmail.navigate_to_gmail()
    ```
"""

from openbrowser.integrations.gmail import GmailIntegration

__all__ = ["GmailIntegration"]

