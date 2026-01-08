"""Signal handler for pause/resume/stop functionality.

Provides graceful handling of SIGINT (Ctrl+C) for agent operations,
allowing users to pause, resume, or stop the agent.

Performance optimizations:
- __slots__ for faster attribute access and reduced memory
"""

import asyncio
import logging
import signal
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class SignalHandler:
    """Handle SIGINT for pause/resume/stop functionality.
    
    On first SIGINT:
        - If running: pauses the agent
        - If paused: resumes the agent
    
    On second SIGINT (while paused):
        - Exits the agent
    
    Args:
        loop: Event loop to schedule callbacks on
        pause_callback: Called when agent should pause
        resume_callback: Called when agent should resume
        custom_exit_callback: Optional callback before exit
        exit_on_second_int: Whether to exit on second SIGINT
    """
    
    __slots__ = (
        '_loop', '_pause_callback', '_resume_callback', '_custom_exit_callback',
        '_exit_on_second_int', '_sigint_count', '_is_paused', '_original_handler', '_registered'
    )
    
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop | None = None,
        pause_callback: Callable[[], None] | None = None,
        resume_callback: Callable[[], None] | None = None,
        custom_exit_callback: Callable[[], None] | None = None,
        exit_on_second_int: bool = True,
    ):
        self._loop = loop
        self._pause_callback = pause_callback
        self._resume_callback = resume_callback
        self._custom_exit_callback = custom_exit_callback
        self._exit_on_second_int = exit_on_second_int
        self._sigint_count = 0
        self._is_paused = False
        self._original_handler: Any = None
        self._registered = False
    
    def register(self) -> None:
        """Register the signal handler."""
        if self._registered:
            return
        
        self._original_handler = signal.signal(signal.SIGINT, self._handle_sigint)
        self._registered = True
        logger.debug("Signal handler registered")
    
    def unregister(self) -> None:
        """Unregister the signal handler and restore the original."""
        if not self._registered:
            return
        
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)
        else:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        
        self._registered = False
        logger.debug("Signal handler unregistered")
    
    def _handle_sigint(self, signum: int, frame: Any) -> None:
        """Handle SIGINT signal."""
        self._sigint_count += 1
        
        if self._sigint_count == 1:
            if self._is_paused:
                # Resume
                logger.info("Resuming agent...")
                self._is_paused = False
                if self._resume_callback:
                    self._schedule_callback(self._resume_callback)
            else:
                # Pause
                logger.info("Pausing agent... Press Ctrl+C again to stop.")
                self._is_paused = True
                if self._pause_callback:
                    self._schedule_callback(self._pause_callback)
        
        elif self._sigint_count >= 2 and self._exit_on_second_int:
            logger.info("Stopping agent...")
            if self._custom_exit_callback:
                self._schedule_callback(self._custom_exit_callback)
            else:
                # Restore default handler and re-raise
                self.unregister()
                raise KeyboardInterrupt
    
    def _schedule_callback(self, callback: Callable[[], None]) -> None:
        """Schedule a callback on the event loop."""
        if self._loop:
            self._loop.call_soon_threadsafe(callback)
        else:
            callback()
    
    def reset(self) -> None:
        """Reset the interrupt count."""
        self._sigint_count = 0
        self._is_paused = False
    
    @property
    def is_paused(self) -> bool:
        """Check if the agent is currently paused."""
        return self._is_paused
    
    @property
    def interrupt_count(self) -> int:
        """Get the number of interrupts received."""
        return self._sigint_count
    
    def __enter__(self) -> 'SignalHandler':
        """Context manager entry."""
        self.register()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.unregister()


class AsyncSignalHandler:
    """Async-aware signal handler for pause/resume/stop.
    
    Provides async callbacks and integrates better with asyncio.
    """
    
    __slots__ = (
        '_pause_callback', '_resume_callback', '_stop_callback',
        '_pause_event', '_stop_event', '_is_paused', '_handler'
    )
    
    def __init__(
        self,
        pause_callback: Callable[[], Any] | None = None,
        resume_callback: Callable[[], Any] | None = None,
        stop_callback: Callable[[], Any] | None = None,
    ):
        self._pause_callback = pause_callback
        self._resume_callback = resume_callback
        self._stop_callback = stop_callback
        self._pause_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._is_paused = False
        self._handler: SignalHandler | None = None
    
    def start(self) -> None:
        """Start handling signals."""
        loop = asyncio.get_event_loop()
        self._handler = SignalHandler(
            loop=loop,
            pause_callback=self._on_pause,
            resume_callback=self._on_resume,
            custom_exit_callback=self._on_stop,
            exit_on_second_int=True,
        )
        self._handler.register()
    
    def stop(self) -> None:
        """Stop handling signals."""
        if self._handler:
            self._handler.unregister()
            self._handler = None
    
    def _on_pause(self) -> None:
        """Called when pause is requested."""
        self._is_paused = True
        self._pause_event.clear()
        if self._pause_callback:
            if asyncio.iscoroutinefunction(self._pause_callback):
                asyncio.create_task(self._pause_callback())
            else:
                self._pause_callback()
    
    def _on_resume(self) -> None:
        """Called when resume is requested."""
        self._is_paused = False
        self._pause_event.set()
        if self._resume_callback:
            if asyncio.iscoroutinefunction(self._resume_callback):
                asyncio.create_task(self._resume_callback())
            else:
                self._resume_callback()
    
    def _on_stop(self) -> None:
        """Called when stop is requested."""
        self._stop_event.set()
        if self._stop_callback:
            if asyncio.iscoroutinefunction(self._stop_callback):
                asyncio.create_task(self._stop_callback())
            else:
                self._stop_callback()
    
    async def wait_if_paused(self) -> None:
        """Wait if currently paused, return immediately otherwise."""
        if self._is_paused:
            await self._pause_event.wait()
    
    @property
    def should_stop(self) -> bool:
        """Check if stop has been requested."""
        return self._stop_event.is_set()
    
    @property
    def is_paused(self) -> bool:
        """Check if currently paused."""
        return self._is_paused
    
    def __enter__(self) -> 'AsyncSignalHandler':
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
