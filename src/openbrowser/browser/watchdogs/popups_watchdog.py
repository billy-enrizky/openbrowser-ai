"""Watchdog for handling JavaScript dialogs (alert, confirm, prompt) automatically.

This module provides the PopupsWatchdog which automatically handles JavaScript
dialogs without requiring user interaction, enabling unattended automation.

Classes:
    PopupsWatchdog: Automatically accepts/dismisses JavaScript dialogs.
"""

import asyncio
import logging
from typing import Any, ClassVar

from bubus import BaseEvent
from pydantic import PrivateAttr

from src.openbrowser.browser.events import TabCreatedEvent
from src.openbrowser.browser.watchdogs.base import BaseWatchdog

logger = logging.getLogger(__name__)


class PopupsWatchdog(BaseWatchdog):
    """Handles JavaScript dialogs by automatically accepting them immediately.

    Registers CDP Page.javascriptDialogOpening handlers for each tab and
    automatically responds to dialogs:
    - alert: Accept (click OK)
    - confirm: Accept (click OK - safer for automation)
    - prompt: Dismiss (click Cancel - can't provide input)
    - beforeunload: Accept (allow navigation)

    Stores dialog messages in browser_session._closed_popup_messages
    for inclusion in browser state.

    Listens to:
        TabCreatedEvent: Registers dialog handlers for new tabs.

    Example:
        >>> # Dialogs are handled automatically
        >>> await browser.navigate_to('https://example.com')
        >>> # Any alert() calls are auto-dismissed
    """

    # Events this watchdog listens to
    LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = [
        TabCreatedEvent,
    ]

    # Events this watchdog emits
    EMITS: ClassVar[list[type[BaseEvent[Any]]]] = []

    # Track which targets have dialog handlers registered
    _dialog_listeners_registered: set[str] = PrivateAttr(default_factory=set)

    def attach_to_session(self) -> None:
        """Register event handlers.

        Subscribes to TabCreatedEvent for dialog handler registration.
        """
        self.event_bus.on(TabCreatedEvent, self.on_TabCreatedEvent)

    async def on_TabCreatedEvent(self, event: TabCreatedEvent) -> None:
        """Set up JavaScript dialog handling when a new tab is created.

        Enables Page domain and registers Page.javascriptDialogOpening
        handler for the tab. Skips if already registered for this target.

        Args:
            event: TabCreatedEvent with target_id.
        """
        target_id = event.target_id
        self.logger.debug(f'[PopupsWatchdog] Received TabCreatedEvent for target {target_id}')

        # Skip if we've already registered for this target
        if target_id in self._dialog_listeners_registered:
            self.logger.debug(f'[PopupsWatchdog] Already registered dialog handlers for target {target_id}')
            return

        try:
            # Get CDP session for this target
            cdp_session = await self.browser_session.get_or_create_cdp_session(target_id=target_id, focus=False)

            # Enable Page domain to receive dialog events
            try:
                await cdp_session.cdp_client.send.Page.enable(session_id=cdp_session.session_id)
                self.logger.debug(f'[PopupsWatchdog] Enabled Page domain for session {cdp_session.session_id[:8]}')
            except Exception as e:
                self.logger.debug(f'[PopupsWatchdog] Failed to enable Page domain: {e}')

            # Set up async handler for JavaScript dialogs - accept immediately
            async def handle_dialog(event_data, session_id: str | None = None):
                """Handle JavaScript dialog events - accept immediately."""
                try:
                    dialog_type = event_data.get('type', 'alert')
                    message = event_data.get('message', '')

                    # Store the popup message in browser session for inclusion in browser state
                    if message:
                        formatted_message = f'[{dialog_type}] {message}'
                        self.browser_session._closed_popup_messages.append(formatted_message)
                        self.logger.debug(f'[PopupsWatchdog] Stored popup message: {formatted_message[:100]}')

                    # Choose action based on dialog type:
                    # - alert: accept=true (click OK to dismiss)
                    # - confirm: accept=true (click OK to proceed - safer for automation)
                    # - prompt: accept=false (click Cancel since we can't provide input)
                    # - beforeunload: accept=true (allow navigation)
                    should_accept = dialog_type in ('alert', 'confirm', 'beforeunload')

                    action_str = 'accepting (OK)' if should_accept else 'dismissing (Cancel)'
                    self.logger.info(f"[PopupsWatchdog] JavaScript {dialog_type} dialog: '{message[:100]}' - {action_str}...")

                    dismissed = False

                    # Approach 1: Use the session that detected the dialog (most reliable)
                    if self.browser_session._cdp_client_root and session_id:
                        try:
                            await asyncio.wait_for(
                                self.browser_session._cdp_client_root.send.Page.handleJavaScriptDialog(
                                    params={'accept': should_accept},
                                    session_id=session_id,
                                ),
                                timeout=0.5,
                            )
                            dismissed = True
                            self.logger.info('[PopupsWatchdog] Dialog handled successfully via detecting session')
                        except (TimeoutError, Exception) as e:
                            self.logger.debug(f'[PopupsWatchdog] Approach 1 failed: {type(e).__name__}')

                    # Approach 2: Try with current agent focus session
                    if not dismissed and self.browser_session._cdp_client_root and self.browser_session.agent_focus:
                        try:
                            await asyncio.wait_for(
                                self.browser_session._cdp_client_root.send.Page.handleJavaScriptDialog(
                                    params={'accept': should_accept},
                                    session_id=self.browser_session.agent_focus.session_id,
                                ),
                                timeout=0.5,
                            )
                            dismissed = True
                            self.logger.info('[PopupsWatchdog] Dialog handled successfully via agent focus session')
                        except (TimeoutError, Exception) as e:
                            self.logger.debug(f'[PopupsWatchdog] Approach 2 failed: {type(e).__name__}')

                except Exception as e:
                    self.logger.error(f'[PopupsWatchdog] Critical error in dialog handler: {type(e).__name__}: {e}')

            # Register handler on the specific session
            cdp_session.cdp_client.register.Page.javascriptDialogOpening(handle_dialog)  # type: ignore[arg-type]
            self.logger.debug(
                f'[PopupsWatchdog] Registered Page.javascriptDialogOpening handler for session {cdp_session.session_id}'
            )

            # Also register on root CDP client to catch dialogs from any frame
            if hasattr(self.browser_session._cdp_client_root, 'register'):
                try:
                    self.browser_session._cdp_client_root.register.Page.javascriptDialogOpening(handle_dialog)  # type: ignore[arg-type]
                    self.logger.debug('[PopupsWatchdog] Registered dialog handler on root CDP client for all frames')
                except Exception as root_error:
                    self.logger.warning(f'[PopupsWatchdog] Failed to register on root CDP client: {root_error}')

            # Mark this target as having dialog handling set up
            self._dialog_listeners_registered.add(target_id)

            self.logger.debug(f'[PopupsWatchdog] Set up JavaScript dialog handling for tab {target_id}')

        except Exception as e:
            self.logger.warning(f'[PopupsWatchdog] Failed to set up popup handling for tab {target_id}: {e}')

