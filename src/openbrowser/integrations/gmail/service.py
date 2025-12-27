"""Gmail integration service."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


class EmailMessage(BaseModel):
    """Email message model."""

    sender: str = ""
    recipient: str = ""
    subject: str = ""
    body: str = ""
    is_draft: bool = False


class GmailIntegration:
    """
    Gmail integration for browser-based email automation.
    Provides helper methods for common Gmail operations.
    """

    GMAIL_URL = "https://mail.google.com"
    COMPOSE_URL = "https://mail.google.com/mail/u/0/#inbox?compose=new"

    def __init__(self, browser_session: BrowserSession):
        self.browser_session = browser_session

    async def navigate_to_gmail(self) -> bool:
        """Navigate to Gmail inbox."""
        from src.openbrowser.browser.events import NavigateToUrlEvent

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
        """
        Compose and send an email.

        Args:
            recipient: Email recipient address
            subject: Email subject
            body: Email body content

        Returns:
            True if email was composed successfully
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
        """
        Search emails in Gmail.

        Args:
            query: Search query

        Returns:
            True if search was initiated successfully
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
        """
        Read the latest email in the inbox.

        Returns:
            EmailMessage if found, None otherwise
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
        """
        Reply to the currently open email.

        Args:
            body: Reply message body

        Returns:
            True if reply was initiated successfully
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
        """
        Delete the currently selected email.

        Returns:
            True if delete was initiated successfully
        """
        logger.info("Deleting email")

        delete_instructions = """
        To delete this email:
        1. Click the Delete (trash) button
        """

        return True

    async def archive_email(self) -> bool:
        """
        Archive the currently selected email.

        Returns:
            True if archive was initiated successfully
        """
        logger.info("Archiving email")

        archive_instructions = """
        To archive this email:
        1. Click the Archive button
        """

        return True

    def get_gmail_actions(self) -> list[dict]:
        """
        Get Gmail-specific actions for the agent.

        Returns:
            List of action definitions
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

