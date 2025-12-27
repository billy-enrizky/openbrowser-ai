"""Mouse class for mouse operations using CDP."""

import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)

MouseButton = Literal['left', 'right', 'middle']


class Mouse:
    """Mouse operations for a browser target.
    
    Provides low-level mouse interactions using Chrome DevTools Protocol.
    """

    def __init__(
        self,
        browser_session: 'BrowserSession',
        session_id: str | None = None,
        target_id: str | None = None,
    ):
        self._browser_session = browser_session
        self._session_id = session_id
        self._target_id = target_id
        self._current_x: float = 0
        self._current_y: float = 0

    @property
    def _client(self):
        """Get the CDP client from browser session."""
        return self._browser_session.cdp_client

    async def click(
        self,
        x: float,
        y: float,
        button: MouseButton = 'left',
        click_count: int = 1,
    ) -> None:
        """Click at the specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button ('left', 'right', 'middle')
            click_count: Number of clicks (1 for single, 2 for double)
        """
        # Mouse press
        await self._client.send(
            'Input.dispatchMouseEvent',
            {
                'type': 'mousePressed',
                'x': x,
                'y': y,
                'button': button,
                'clickCount': click_count,
            },
            session_id=self._session_id
        )

        # Mouse release
        await self._client.send(
            'Input.dispatchMouseEvent',
            {
                'type': 'mouseReleased',
                'x': x,
                'y': y,
                'button': button,
                'clickCount': click_count,
            },
            session_id=self._session_id
        )

        self._current_x = x
        self._current_y = y

    async def double_click(
        self,
        x: float,
        y: float,
        button: MouseButton = 'left',
    ) -> None:
        """Double-click at the specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button
        """
        await self.click(x, y, button, click_count=2)

    async def down(
        self,
        button: MouseButton = 'left',
        click_count: int = 1,
    ) -> None:
        """Press mouse button down at current position.
        
        Args:
            button: Mouse button
            click_count: Click count
        """
        await self._client.send(
            'Input.dispatchMouseEvent',
            {
                'type': 'mousePressed',
                'x': self._current_x,
                'y': self._current_y,
                'button': button,
                'clickCount': click_count,
            },
            session_id=self._session_id
        )

    async def up(
        self,
        button: MouseButton = 'left',
        click_count: int = 1,
    ) -> None:
        """Release mouse button at current position.
        
        Args:
            button: Mouse button
            click_count: Click count
        """
        await self._client.send(
            'Input.dispatchMouseEvent',
            {
                'type': 'mouseReleased',
                'x': self._current_x,
                'y': self._current_y,
                'button': button,
                'clickCount': click_count,
            },
            session_id=self._session_id
        )

    async def move(
        self,
        x: float,
        y: float,
        steps: int = 1,
    ) -> None:
        """Move mouse to the specified coordinates.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            steps: Number of intermediate steps for smooth movement
        """
        if steps > 1:
            # Smooth movement with intermediate steps
            start_x, start_y = self._current_x, self._current_y
            for i in range(1, steps + 1):
                t = i / steps
                current_x = start_x + (x - start_x) * t
                current_y = start_y + (y - start_y) * t
                await self._client.send(
                    'Input.dispatchMouseEvent',
                    {'type': 'mouseMoved', 'x': current_x, 'y': current_y},
                    session_id=self._session_id
                )
        else:
            await self._client.send(
                'Input.dispatchMouseEvent',
                {'type': 'mouseMoved', 'x': x, 'y': y},
                session_id=self._session_id
            )

        self._current_x = x
        self._current_y = y

    async def scroll(
        self,
        x: float = 0,
        y: float = 0,
        delta_x: float | None = None,
        delta_y: float | None = None,
    ) -> None:
        """Scroll the page.
        
        Args:
            x: X coordinate to scroll from (0 = center of viewport)
            y: Y coordinate to scroll from (0 = center of viewport)
            delta_x: Horizontal scroll amount (positive = right)
            delta_y: Vertical scroll amount (positive = down)
        """
        if not self._session_id:
            raise RuntimeError('Session ID is required for scroll operations')

        # Method 1: Try mouse wheel event (most reliable)
        try:
            layout_metrics = await self._client.send(
                'Page.getLayoutMetrics',
                session_id=self._session_id
            )
            viewport_width = layout_metrics['layoutViewport']['clientWidth']
            viewport_height = layout_metrics['layoutViewport']['clientHeight']

            scroll_x = x if x > 0 else viewport_width / 2
            scroll_y = y if y > 0 else viewport_height / 2
            scroll_delta_x = delta_x or 0
            scroll_delta_y = delta_y or 0

            await self._client.send(
                'Input.dispatchMouseEvent',
                {
                    'type': 'mouseWheel',
                    'x': scroll_x,
                    'y': scroll_y,
                    'deltaX': scroll_delta_x,
                    'deltaY': scroll_delta_y,
                },
                session_id=self._session_id
            )
            return
        except Exception:
            pass

        # Method 2: Fallback to synthesizeScrollGesture
        try:
            await self._client.send(
                'Input.synthesizeScrollGesture',
                {
                    'x': x,
                    'y': y,
                    'xDistance': delta_x or 0,
                    'yDistance': delta_y or 0,
                },
                session_id=self._session_id
            )
            return
        except Exception:
            pass

        # Method 3: JavaScript fallback
        scroll_js = f'window.scrollBy({delta_x or 0}, {delta_y or 0})'
        await self._client.send(
            'Runtime.evaluate',
            {'expression': scroll_js, 'returnByValue': True},
            session_id=self._session_id
        )

    async def scroll_down(self, amount: float = 100) -> None:
        """Scroll down by a specified amount.
        
        Args:
            amount: Pixels to scroll down
        """
        await self.scroll(delta_y=amount)

    async def scroll_up(self, amount: float = 100) -> None:
        """Scroll up by a specified amount.
        
        Args:
            amount: Pixels to scroll up
        """
        await self.scroll(delta_y=-amount)

    async def scroll_left(self, amount: float = 100) -> None:
        """Scroll left by a specified amount.
        
        Args:
            amount: Pixels to scroll left
        """
        await self.scroll(delta_x=-amount)

    async def scroll_right(self, amount: float = 100) -> None:
        """Scroll right by a specified amount.
        
        Args:
            amount: Pixels to scroll right
        """
        await self.scroll(delta_x=amount)

    async def drag(
        self,
        from_x: float,
        from_y: float,
        to_x: float,
        to_y: float,
        steps: int = 10,
    ) -> None:
        """Drag from one point to another.
        
        Args:
            from_x: Starting X coordinate
            from_y: Starting Y coordinate
            to_x: Ending X coordinate
            to_y: Ending Y coordinate
            steps: Number of intermediate steps
        """
        # Move to start position
        await self.move(from_x, from_y)

        # Press down
        await self.down()

        # Move to end position with steps
        await self.move(to_x, to_y, steps=steps)

        # Release
        await self.up()

    async def hover(self, x: float, y: float) -> None:
        """Hover at the specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        await self.move(x, y)

    @property
    def position(self) -> tuple[float, float]:
        """Get the current mouse position."""
        return (self._current_x, self._current_y)

