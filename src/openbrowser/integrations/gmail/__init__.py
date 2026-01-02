"""Gmail integration module for browser-based email automation.

This module provides the GmailIntegration class with helper methods
for common Gmail operations like composing, searching, and reading emails.

Example:
    ```python
    from src.openbrowser.integrations.gmail import GmailIntegration

    gmail = GmailIntegration(browser_session)
    await gmail.navigate_to_gmail()
    await gmail.compose_email(
        recipient="user@example.com",
        subject="Hello",
        body="Message content"
    )
    ```
"""

from .service import GmailIntegration

__all__ = ["GmailIntegration"]

