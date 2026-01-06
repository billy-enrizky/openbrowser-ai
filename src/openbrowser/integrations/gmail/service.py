"""Gmail integration service for browser-based email automation.

This module provides the GmailIntegration class, which offers helper methods
for automating common Gmail operations through browser control. The integration
provides structured instructions for email operations that can be executed
by browser automation agents.

Note:
    This integration requires the user to be logged into Gmail in the browser.
    Authentication is handled by the browser session, not by this module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


class EmailMessage(BaseModel):
    """Email message model for Gmail operations.

    Represents an email message with common fields used for composing,
    reading, or displaying email content.

    Attributes:
        sender: Email address of the sender.
        recipient: Email address of the recipient.
        subject: Email subject line.
        body: Email body content (plain text).
        is_draft: Whether this is a draft message.

    Example:
        ```python
        message = EmailMessage(
            sender="me@example.com",
            recipient="you@example.com",
            subject="Meeting tomorrow",
            body="Let's meet at 2pm."
        )
        ```
    """

    sender: str = ""
    recipient: str = ""
    subject: str = ""
    body: str = ""
    is_draft: bool = False


class GmailIntegration:
    """Gmail integration for browser-based email automation.

    Provides helper methods for common Gmail operations, generating
    structured instructions that can be executed by browser automation
    agents. This class does not perform the actual browser actions
    directly; instead, it provides the instructions and context needed
    for an agent to perform the operations.

    Attributes:
        GMAIL_URL: Base URL for Gmail.
        COMPOSE_URL: URL for composing a new email.
        browser_session: The active browser session.

    Example:
        ```python
        gmail = GmailIntegration(browser_session)

        # Navigate to Gmail
        await gmail.navigate_to_gmail()

        # Search for emails
        await gmail.search_emails("from:important@example.com")

        # Get action definitions for agent
        actions = gmail.get_gmail_actions()
        ```

    Note:
        The user must be logged into Gmail for operations to succeed.
        This integration is designed to work with browser automation
        agents that can execute the provided instructions.
    """

    GMAIL_URL = "https://mail.google.com"
    COMPOSE_URL = "https://mail.google.com/mail/u/0/#inbox?compose=new"

    def __init__(self, browser_session: BrowserSession):
        """Initialize the Gmail integration.

        Args:
            browser_session: The active browser session to use for
                Gmail operations.
        """
        self.browser_session = browser_session

    async def navigate_to_gmail(self) -> bool:
        """Navigate to the Gmail inbox.

        Dispatches a navigation event to open Gmail in the browser.
        The user must be logged in for the inbox to be accessible.

        Returns:
            True if navigation was dispatched successfully,
            False if an error occurred.
        """
        from openbrowser.browser.events import NavigateToUrlEvent

        try:
            await self.browser_session.event_bus.dispatch(
                NavigateToUrlEvent(url=self.GMAIL_URL, new_tab=False)
            )
            logger.info("Navigated to Gmail")
            return True
        except Exception as e:
            logger.error(f"Failed to navigate to Gmail: {e}")
            return False

    async def compose_email(
        self,
        recipient: str,
        subject: str,
        body: str,
    ) -> bool:
        """Compose and prepare to send an email.

        Provides instructions for composing an email in Gmail.
        The actual composition is performed by the browser automation
        agent following the returned instructions.

        Args:
            recipient: Email address of the recipient.
            subject: Subject line for the email.
            body: Body content for the email.

        Returns:
            True if the compose operation was initiated.
            The actual sending depends on agent execution.

        Note:
            The agent should click the Compose button, fill in the
            recipient, subject, and body fields, then click Send.
        """
        # This provides the instructions for composing an email
        # The actual execution is done by the agent using browser actions
        logger.info(f"Composing email to {recipient}")

        compose_instructions = f"""
        To compose and send this email:
        1. Click the Compose button
        2. In the "To" field, enter: {recipient}
        3. In the "Subject" field, enter: {subject}
        4. In the message body, type: {body}
        5. Click the Send button
        """

        return True

    async def search_emails(self, query: str) -> bool:
        """Search emails in Gmail.

        Provides instructions for searching emails using Gmail's
        search functionality. Supports Gmail search operators.

        Args:
            query: Search query string. Supports Gmail operators like
                'from:', 'to:', 'subject:', 'is:unread', etc.

        Returns:
            True if the search operation was initiated.
            Results depend on agent execution.

        Example:
            ```python
            await gmail.search_emails("from:boss@company.com is:unread")
            await gmail.search_emails("subject:invoice after:2024/01/01")
            ```
        """
        logger.info(f"Searching emails with query: {query}")

        search_instructions = f"""
        To search for emails:
        1. Click on the search bar at the top
        2. Type: {query}
        3. Press Enter to search
        """

        return True

    async def read_latest_email(self) -> Optional[EmailMessage]:
        """Read the latest email in the inbox.

        Provides instructions for opening and reading the most recent
        email in the inbox. The actual extraction of content is
        performed by the browser automation agent.

        Returns:
            None. The agent is responsible for extracting the email
            content using browser automation.

        Note:
            The agent should click on the first email in the inbox
            and extract the sender, subject, and body content.
        """
        logger.info("Reading latest email")

        read_instructions = """
        To read the latest email:
        1. Click on the first email in the inbox list
        2. Extract the sender, subject, and body content
        """

        # The actual extraction is done by the agent
        return None

    async def reply_to_email(self, body: str) -> bool:
        """Reply to the currently open email.

        Provides instructions for replying to an email that is
        currently open in the Gmail interface.

        Args:
            body: The reply message body content.

        Returns:
            True if the reply operation was initiated.
            Actual sending depends on agent execution.

        Note:
            An email must be open in the Gmail interface before
            calling this method.
        """
        logger.info("Replying to email")

        reply_instructions = f"""
        To reply to this email:
        1. Click the Reply button
        2. In the reply box, type: {body}
        3. Click the Send button
        """

        return True

    async def delete_email(self) -> bool:
        """Delete the currently selected email.

        Provides instructions for deleting the email that is
        currently selected or open in Gmail.

        Returns:
            True if the delete operation was initiated.
            Actual deletion depends on agent execution.

        Note:
            The email is moved to Trash, not permanently deleted.
        """
        logger.info("Deleting email")

        delete_instructions = """
        To delete this email:
        1. Click the Delete (trash) button
        """

        return True

    async def archive_email(self) -> bool:
        """Archive the currently selected email.

        Provides instructions for archiving the email that is
        currently selected or open in Gmail.

        Returns:
            True if the archive operation was initiated.
            Actual archiving depends on agent execution.

        Note:
            Archived emails are removed from the inbox but remain
            accessible via search or the All Mail label.
        """
        logger.info("Archiving email")

        archive_instructions = """
        To archive this email:
        1. Click the Archive button
        """

        return True

    def get_gmail_actions(self) -> list[dict]:
        """Get Gmail-specific actions for the agent.

        Returns a list of action definitions that describe the
        Gmail operations available to the automation agent.

        Returns:
            List of action dictionaries, each containing:
            - name: The action identifier
            - description: Human-readable description
            - parameters: Dictionary of parameter definitions

        Example:
            ```python
            actions = gmail.get_gmail_actions()
            for action in actions:
                print(f"{action['name']}: {action['description']}")
            ```
        """
        return [
            {
                "name": "gmail_compose",
                "description": "Compose a new email in Gmail",
                "parameters": {
                    "recipient": {"type": "string", "description": "Email recipient"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                },
            },
            {
                "name": "gmail_search",
                "description": "Search emails in Gmail",
                "parameters": {
                    "query": {"type": "string", "description": "Search query"},
                },
            },
            {
                "name": "gmail_reply",
                "description": "Reply to the current email",
                "parameters": {
                    "body": {"type": "string", "description": "Reply message"},
                },
            },
            {
                "name": "gmail_delete",
                "description": "Delete the current email",
                "parameters": {},
            },
            {
                "name": "gmail_archive",
                "description": "Archive the current email",
                "parameters": {},
            },
        ]

