"""Element class for low-level DOM element interactions using CDP."""

import asyncio
import logging
from typing import TYPE_CHECKING, Literal, Union

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from src.openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)

# Type definitions for element operations
ModifierType = Literal['Alt', 'Control', 'Meta', 'Shift']
MouseButton = Literal['left', 'right', 'middle']


class Position(TypedDict):
    """2D position coordinates."""
    x: float
    y: float


class BoundingBox(TypedDict):
    """Element bounding box with position and dimensions."""
    x: float
    y: float
    width: float
    height: float


class ElementInfo(TypedDict):
    """Basic information about a DOM element."""
    backendNodeId: int
    nodeId: int | None
    nodeName: str
    nodeType: int
    nodeValue: str | None
    attributes: dict[str, str]
    boundingBox: BoundingBox | None
    error: str | None


class Element:
    """Element operations using BackendNodeId.
    
    Provides low-level DOM element interactions using Chrome DevTools Protocol.
    """

    def __init__(
        self,
        browser_session: 'BrowserSession',
        backend_node_id: int,
        session_id: str | None = None,
    ):
        self._browser_session = browser_session
        self._backend_node_id = backend_node_id
        self._session_id = session_id

    @property
    def _client(self):
        """Get the CDP client from browser session."""
        return self._browser_session.cdp_client

    async def _get_node_id(self) -> int:
        """Get DOM node ID from backend node ID."""
        result = await self._client.send(
            'DOM.pushNodesByBackendIdsToFrontend',
            {'backendNodeIds': [self._backend_node_id]},
            session_id=self._session_id
        )
        return result['nodeIds'][0]

    async def _get_remote_object_id(self) -> str | None:
        """Get remote object ID for this element."""
        node_id = await self._get_node_id()
        result = await self._client.send(
            'DOM.resolveNode',
            {'nodeId': node_id},
            session_id=self._session_id
        )
        return result.get('object', {}).get('objectId')

    async def click(
        self,
        button: MouseButton = 'left',
        click_count: int = 1,
        modifiers: list[ModifierType] | None = None,
    ) -> None:
        """Click the element using multiple fallback strategies.
        
        Args:
            button: Mouse button to use ('left', 'right', 'middle')
            click_count: Number of clicks
            modifiers: Modifier keys to hold during click
        """
        try:
            # Get viewport dimensions for visibility checks
            layout_metrics = await self._client.send(
                'Page.getLayoutMetrics',
                session_id=self._session_id
            )
            viewport_width = layout_metrics['layoutViewport']['clientWidth']
            viewport_height = layout_metrics['layoutViewport']['clientHeight']

            # Try multiple methods to get element geometry
            quads = await self._get_element_quads()

            # If we don't have quads, fall back to JS click
            if not quads:
                await self._js_click()
                return

            # Find the best visible quad
            center_x, center_y = self._calculate_click_point(
                quads, viewport_width, viewport_height
            )

            # Scroll element into view
            try:
                await self._client.send(
                    'DOM.scrollIntoViewIfNeeded',
                    {'backendNodeId': self._backend_node_id},
                    session_id=self._session_id
                )
                await asyncio.sleep(0.05)
            except Exception:
                pass

            # Calculate modifier bitmask
            modifier_value = self._calculate_modifiers(modifiers)

            # Perform the click using CDP
            await self._perform_click(center_x, center_y, button, click_count, modifier_value)

        except Exception as e:
            # Fall back to JavaScript click
            try:
                await self._js_click()
            except Exception as js_e:
                raise RuntimeError(f'Failed to click element: {e}, JS fallback also failed: {js_e}')

    async def _get_element_quads(self) -> list:
        """Get element geometry quads using multiple methods."""
        quads = []

        # Method 1: Try DOM.getContentQuads
        try:
            result = await self._client.send(
                'DOM.getContentQuads',
                {'backendNodeId': self._backend_node_id},
                session_id=self._session_id
            )
            if result.get('quads'):
                quads = result['quads']
        except Exception:
            pass

        # Method 2: Fall back to DOM.getBoxModel
        if not quads:
            try:
                result = await self._client.send(
                    'DOM.getBoxModel',
                    {'backendNodeId': self._backend_node_id},
                    session_id=self._session_id
                )
                if result.get('model', {}).get('content'):
                    content = result['model']['content']
                    if len(content) >= 8:
                        quads = [content]
            except Exception:
                pass

        # Method 3: Fall back to JavaScript getBoundingClientRect
        if not quads:
            try:
                result = await self._client.send(
                    'DOM.resolveNode',
                    {'backendNodeId': self._backend_node_id},
                    session_id=self._session_id
                )
                if result.get('object', {}).get('objectId'):
                    object_id = result['object']['objectId']
                    bounds_result = await self._client.send(
                        'Runtime.callFunctionOn',
                        {
                            'functionDeclaration': '''
                                function() {
                                    const rect = this.getBoundingClientRect();
                                    return {x: rect.left, y: rect.top, width: rect.width, height: rect.height};
                                }
                            ''',
                            'objectId': object_id,
                            'returnByValue': True,
                        },
                        session_id=self._session_id
                    )
                    if bounds_result.get('result', {}).get('value'):
                        rect = bounds_result['result']['value']
                        x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
                        quads = [[x, y, x + w, y, x + w, y + h, x, y + h]]
            except Exception:
                pass

        return quads

    def _calculate_click_point(
        self, quads: list, viewport_width: float, viewport_height: float
    ) -> tuple[float, float]:
        """Calculate the best click point from quads."""
        best_quad = None
        best_area = 0

        for quad in quads:
            if len(quad) < 8:
                continue

            xs = [quad[i] for i in range(0, 8, 2)]
            ys = [quad[i] for i in range(1, 8, 2)]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Check if quad intersects with viewport
            if max_x < 0 or max_y < 0 or min_x > viewport_width or min_y > viewport_height:
                continue

            visible_width = min(viewport_width, max_x) - max(0, min_x)
            visible_height = min(viewport_height, max_y) - max(0, min_y)
            visible_area = visible_width * visible_height

            if visible_area > best_area:
                best_area = visible_area
                best_quad = quad

        if not best_quad:
            best_quad = quads[0]

        center_x = sum(best_quad[i] for i in range(0, 8, 2)) / 4
        center_y = sum(best_quad[i] for i in range(1, 8, 2)) / 4

        # Ensure click point is within viewport
        center_x = max(0, min(viewport_width - 1, center_x))
        center_y = max(0, min(viewport_height - 1, center_y))

        return center_x, center_y

    def _calculate_modifiers(self, modifiers: list[ModifierType] | None) -> int:
        """Calculate modifier bitmask for CDP."""
        if not modifiers:
            return 0
        modifier_map = {'Alt': 1, 'Control': 2, 'Meta': 4, 'Shift': 8}
        return sum(modifier_map.get(mod, 0) for mod in modifiers)

    async def _perform_click(
        self,
        x: float,
        y: float,
        button: MouseButton,
        click_count: int,
        modifiers: int,
    ) -> None:
        """Perform click using CDP Input events."""
        try:
            # Move mouse to element
            await self._client.send(
                'Input.dispatchMouseEvent',
                {'type': 'mouseMoved', 'x': x, 'y': y},
                session_id=self._session_id
            )
            await asyncio.sleep(0.05)

            # Mouse down
            try:
                await asyncio.wait_for(
                    self._client.send(
                        'Input.dispatchMouseEvent',
                        {
                            'type': 'mousePressed',
                            'x': x,
                            'y': y,
                            'button': button,
                            'clickCount': click_count,
                            'modifiers': modifiers,
                        },
                        session_id=self._session_id
                    ),
                    timeout=1.0
                )
                await asyncio.sleep(0.08)
            except TimeoutError:
                pass

            # Mouse up
            try:
                await asyncio.wait_for(
                    self._client.send(
                        'Input.dispatchMouseEvent',
                        {
                            'type': 'mouseReleased',
                            'x': x,
                            'y': y,
                            'button': button,
                            'clickCount': click_count,
                            'modifiers': modifiers,
                        },
                        session_id=self._session_id
                    ),
                    timeout=3.0
                )
            except TimeoutError:
                pass

        except Exception as e:
            raise RuntimeError(f'Failed to perform click: {e}')

    async def _js_click(self) -> None:
        """Perform click using JavaScript."""
        result = await self._client.send(
            'DOM.resolveNode',
            {'backendNodeId': self._backend_node_id},
            session_id=self._session_id
        )
        if not result.get('object', {}).get('objectId'):
            raise RuntimeError('Failed to find DOM element')
        
        object_id = result['object']['objectId']
        await self._client.send(
            'Runtime.callFunctionOn',
            {
                'functionDeclaration': 'function() { this.click(); }',
                'objectId': object_id,
            },
            session_id=self._session_id
        )
        await asyncio.sleep(0.05)

    async def fill(self, value: str, clear: bool = True) -> None:
        """Fill the input element with text.
        
        Args:
            value: Text to type into the element
            clear: Whether to clear existing text first
        """
        try:
            # Scroll element into view
            try:
                await self._client.send(
                    'DOM.scrollIntoViewIfNeeded',
                    {'backendNodeId': self._backend_node_id},
                    session_id=self._session_id
                )
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.warning(f'Failed to scroll element into view: {e}')

            # Get object ID for the element
            result = await self._client.send(
                'DOM.resolveNode',
                {'backendNodeId': self._backend_node_id},
                session_id=self._session_id
            )
            if not result.get('object', {}).get('objectId'):
                raise RuntimeError('Failed to get object ID for element')
            object_id = result['object']['objectId']

            # Get element coordinates for focus
            input_coordinates = await self._get_element_coordinates(object_id)

            # Focus the element
            await self._focus_element(object_id, input_coordinates)

            # Clear existing text if requested
            if clear:
                await self._clear_text_field(object_id)

            # Type the text character by character
            await self._type_text(value)

        except Exception as e:
            raise RuntimeError(f'Failed to fill element: {str(e)}')

    async def _get_element_coordinates(self, object_id: str) -> dict | None:
        """Get element center coordinates."""
        try:
            result = await self._client.send(
                'Runtime.callFunctionOn',
                {
                    'functionDeclaration': 'function() { return this.getBoundingClientRect(); }',
                    'objectId': object_id,
                    'returnByValue': True,
                },
                session_id=self._session_id
            )
            if result.get('result', {}).get('value'):
                bounds = result['result']['value']
                return {
                    'input_x': bounds['x'] + bounds['width'] / 2,
                    'input_y': bounds['y'] + bounds['height'] / 2,
                }
        except Exception:
            pass
        return None

    async def _focus_element(self, object_id: str, coordinates: dict | None) -> bool:
        """Focus element using multiple strategies."""
        # Strategy 1: CDP focus
        try:
            await self._client.send(
                'DOM.focus',
                {'backendNodeId': self._backend_node_id},
                session_id=self._session_id
            )
            return True
        except Exception:
            pass

        # Strategy 2: JavaScript focus
        try:
            await self._client.send(
                'Runtime.callFunctionOn',
                {
                    'functionDeclaration': 'function() { this.focus(); }',
                    'objectId': object_id,
                },
                session_id=self._session_id
            )
            return True
        except Exception:
            pass

        # Strategy 3: Click to focus
        if coordinates:
            try:
                await self._client.send(
                    'Input.dispatchMouseEvent',
                    {
                        'type': 'mousePressed',
                        'x': coordinates['input_x'],
                        'y': coordinates['input_y'],
                        'button': 'left',
                        'clickCount': 1,
                    },
                    session_id=self._session_id
                )
                await self._client.send(
                    'Input.dispatchMouseEvent',
                    {
                        'type': 'mouseReleased',
                        'x': coordinates['input_x'],
                        'y': coordinates['input_y'],
                        'button': 'left',
                        'clickCount': 1,
                    },
                    session_id=self._session_id
                )
                return True
            except Exception:
                pass

        return False

    async def _clear_text_field(self, object_id: str) -> bool:
        """Clear text field using JavaScript."""
        try:
            await self._client.send(
                'Runtime.callFunctionOn',
                {
                    'functionDeclaration': '''
                        function() {
                            try { this.select(); } catch (e) {}
                            this.value = "";
                            this.dispatchEvent(new Event("input", { bubbles: true }));
                            this.dispatchEvent(new Event("change", { bubbles: true }));
                            return this.value;
                        }
                    ''',
                    'objectId': object_id,
                    'returnByValue': True,
                },
                session_id=self._session_id
            )
            return True
        except Exception:
            return False

    async def _type_text(self, text: str) -> None:
        """Type text character by character."""
        for char in text:
            if char == '\n':
                await self._send_key_event('Enter', 13)
            else:
                modifiers, vk_code, base_key = self._get_char_info(char)
                key_code = self._get_key_code(base_key)

                # Key down
                await self._client.send(
                    'Input.dispatchKeyEvent',
                    {
                        'type': 'keyDown',
                        'key': base_key,
                        'code': key_code,
                        'modifiers': modifiers,
                        'windowsVirtualKeyCode': vk_code,
                    },
                    session_id=self._session_id
                )
                await asyncio.sleep(0.001)

                # Char event
                await self._client.send(
                    'Input.dispatchKeyEvent',
                    {'type': 'char', 'text': char, 'key': char},
                    session_id=self._session_id
                )

                # Key up
                await self._client.send(
                    'Input.dispatchKeyEvent',
                    {
                        'type': 'keyUp',
                        'key': base_key,
                        'code': key_code,
                        'modifiers': modifiers,
                        'windowsVirtualKeyCode': vk_code,
                    },
                    session_id=self._session_id
                )

            await asyncio.sleep(0.018)

    async def _send_key_event(self, key: str, vk_code: int) -> None:
        """Send a key event."""
        await self._client.send(
            'Input.dispatchKeyEvent',
            {'type': 'keyDown', 'key': key, 'code': key, 'windowsVirtualKeyCode': vk_code},
            session_id=self._session_id
        )
        await asyncio.sleep(0.001)
        await self._client.send(
            'Input.dispatchKeyEvent',
            {'type': 'char', 'text': '\r', 'key': key},
            session_id=self._session_id
        )
        await self._client.send(
            'Input.dispatchKeyEvent',
            {'type': 'keyUp', 'key': key, 'code': key, 'windowsVirtualKeyCode': vk_code},
            session_id=self._session_id
        )

    def _get_char_info(self, char: str) -> tuple[int, int, str]:
        """Get modifiers, virtual key code, and base key for a character."""
        shift_chars = {
            '!': ('1', 49), '@': ('2', 50), '#': ('3', 51), '$': ('4', 52),
            '%': ('5', 53), '^': ('6', 54), '&': ('7', 55), '*': ('8', 56),
            '(': ('9', 57), ')': ('0', 48), '_': ('-', 189), '+': ('=', 187),
            '{': ('[', 219), '}': (']', 221), '|': ('\\', 220), ':': (';', 186),
            '"': ("'", 222), '<': (',', 188), '>': ('.', 190), '?': ('/', 191),
            '~': ('`', 192),
        }

        if char in shift_chars:
            base_key, vk_code = shift_chars[char]
            return (8, vk_code, base_key)  # Shift=8

        if char.isupper():
            return (8, ord(char), char.lower())

        if char.islower():
            return (0, ord(char.upper()), char)

        if char.isdigit():
            return (0, ord(char), char)

        no_shift_chars = {
            ' ': 32, '-': 189, '=': 187, '[': 219, ']': 221,
            '\\': 220, ';': 186, "'": 222, ',': 188, '.': 190,
            '/': 191, '`': 192,
        }

        if char in no_shift_chars:
            return (0, no_shift_chars[char], char)

        return (0, ord(char.upper()) if char.isalpha() else ord(char), char)

    def _get_key_code(self, char: str) -> str:
        """Get the proper key code for a character."""
        key_codes = {
            ' ': 'Space', '.': 'Period', ',': 'Comma', '-': 'Minus',
            '/': 'Slash', '=': 'Equal', '[': 'BracketLeft', ']': 'BracketRight',
            '\\': 'Backslash', ';': 'Semicolon', "'": 'Quote', '`': 'Backquote',
        }

        if char in key_codes:
            return key_codes[char]
        elif char.isalpha():
            return f'Key{char.upper()}'
        elif char.isdigit():
            return f'Digit{char}'
        else:
            return 'Unidentified'

    async def hover(self) -> None:
        """Hover over the element."""
        box = await self.get_bounding_box()
        if not box:
            raise RuntimeError('Element is not visible or has no bounding box')

        x = box['x'] + box['width'] / 2
        y = box['y'] + box['height'] / 2

        await self._client.send(
            'Input.dispatchMouseEvent',
            {'type': 'mouseMoved', 'x': x, 'y': y},
            session_id=self._session_id
        )

    async def focus(self) -> None:
        """Focus the element."""
        await self._client.send(
            'DOM.focus',
            {'backendNodeId': self._backend_node_id},
            session_id=self._session_id
        )

    async def check(self) -> None:
        """Check a checkbox or radio button."""
        await self.click()

    async def select_option(self, values: str | list[str]) -> None:
        """Select option(s) in a select element.
        
        Args:
            values: Value(s) to select
        """
        if isinstance(values, str):
            values = [values]

        try:
            await self.focus()
        except Exception:
            pass

        # Get object ID
        result = await self._client.send(
            'DOM.resolveNode',
            {'backendNodeId': self._backend_node_id},
            session_id=self._session_id
        )
        if not result.get('object', {}).get('objectId'):
            raise RuntimeError('Failed to get object ID for select element')

        object_id = result['object']['objectId']

        # Use JavaScript to select options
        values_json = str(values)
        await self._client.send(
            'Runtime.callFunctionOn',
            {
                'functionDeclaration': f'''
                    function() {{
                        const values = {values_json};
                        const options = this.options;
                        for (let i = 0; i < options.length; i++) {{
                            if (values.includes(options[i].value) || values.includes(options[i].text)) {{
                                options[i].selected = true;
                            }}
                        }}
                        this.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    }}
                ''',
                'objectId': object_id,
            },
            session_id=self._session_id
        )

    async def drag_to(
        self,
        target: Union['Element', Position],
        source_position: Position | None = None,
        target_position: Position | None = None,
    ) -> None:
        """Drag this element to another element or position."""
        # Get source coordinates
        if source_position:
            source_x = source_position['x']
            source_y = source_position['y']
        else:
            source_box = await self.get_bounding_box()
            if not source_box:
                raise RuntimeError('Source element is not visible')
            source_x = source_box['x'] + source_box['width'] / 2
            source_y = source_box['y'] + source_box['height'] / 2

        # Get target coordinates
        if isinstance(target, dict) and 'x' in target and 'y' in target:
            target_x = target['x']
            target_y = target['y']
        else:
            target_box = await target.get_bounding_box()
            if not target_box:
                raise RuntimeError('Target element is not visible')
            if target_position:
                target_x = target_box['x'] + target_position['x']
                target_y = target_box['y'] + target_position['y']
            else:
                target_x = target_box['x'] + target_box['width'] / 2
                target_y = target_box['y'] + target_box['height'] / 2

        # Perform drag operation
        await self._client.send(
            'Input.dispatchMouseEvent',
            {'type': 'mousePressed', 'x': source_x, 'y': source_y, 'button': 'left'},
            session_id=self._session_id
        )
        await self._client.send(
            'Input.dispatchMouseEvent',
            {'type': 'mouseMoved', 'x': target_x, 'y': target_y},
            session_id=self._session_id
        )
        await self._client.send(
            'Input.dispatchMouseEvent',
            {'type': 'mouseReleased', 'x': target_x, 'y': target_y, 'button': 'left'},
            session_id=self._session_id
        )

    async def get_attribute(self, name: str) -> str | None:
        """Get an attribute value."""
        node_id = await self._get_node_id()
        result = await self._client.send(
            'DOM.getAttributes',
            {'nodeId': node_id},
            session_id=self._session_id
        )

        attributes = result.get('attributes', [])
        for i in range(0, len(attributes), 2):
            if attributes[i] == name:
                return attributes[i + 1]
        return None

    async def get_bounding_box(self) -> BoundingBox | None:
        """Get the bounding box of the element."""
        try:
            node_id = await self._get_node_id()
            result = await self._client.send(
                'DOM.getBoxModel',
                {'nodeId': node_id},
                session_id=self._session_id
            )

            if 'model' not in result:
                return None

            content = result['model'].get('content', [])
            if len(content) < 8:
                return None

            x_coords = [content[i] for i in range(0, 8, 2)]
            y_coords = [content[i] for i in range(1, 8, 2)]

            x = min(x_coords)
            y = min(y_coords)
            width = max(x_coords) - x
            height = max(y_coords) - y

            return BoundingBox(x=x, y=y, width=width, height=height)

        except Exception:
            return None

    async def screenshot(self, format: str = 'jpeg', quality: int | None = None) -> str:
        """Take a screenshot of this element.
        
        Args:
            format: Image format ('jpeg', 'png', 'webp')
            quality: Quality 0-100 for JPEG format
            
        Returns:
            Base64-encoded image data
        """
        box = await self.get_bounding_box()
        if not box:
            raise RuntimeError('Element is not visible or has no bounding box')

        params = {
            'format': format,
            'clip': {
                'x': box['x'],
                'y': box['y'],
                'width': box['width'],
                'height': box['height'],
                'scale': 1.0,
            },
        }

        if quality is not None and format.lower() == 'jpeg':
            params['quality'] = quality

        result = await self._client.send(
            'Page.captureScreenshot',
            params,
            session_id=self._session_id
        )

        return result['data']

    async def evaluate(self, page_function: str, *args) -> str:
        """Execute JavaScript in the context of this element.
        
        Args:
            page_function: JavaScript arrow function
            *args: Arguments to pass to the function
            
        Returns:
            String representation of the result
        """
        import json
        import re

        object_id = await self._get_remote_object_id()
        if not object_id:
            raise RuntimeError('Element has no remote object ID')

        page_function = page_function.strip()
        if not ('=>' in page_function and (page_function.startswith('(') or page_function.startswith('async'))):
            raise ValueError('JavaScript code must be an arrow function')

        # Convert arrow function to function declaration
        is_async = page_function.startswith('async')
        async_prefix = 'async ' if is_async else ''

        func_to_parse = page_function.strip()
        if is_async:
            func_to_parse = func_to_parse[5:].strip()

        arrow_match = re.match(r'\s*\(([^)]*)\)\s*=>\s*(.+)', func_to_parse, re.DOTALL)
        if not arrow_match:
            raise ValueError(f'Could not parse arrow function: {page_function[:50]}...')

        params_str = arrow_match.group(1).strip()
        body = arrow_match.group(2).strip()

        if not body.startswith('{'):
            function_declaration = f'{async_prefix}function({params_str}) {{ return {body}; }}'
        else:
            function_declaration = f'{async_prefix}function({params_str}) {body}'

        call_arguments = [{'value': arg} for arg in args] if args else None

        params = {
            'functionDeclaration': function_declaration,
            'objectId': object_id,
            'returnByValue': True,
            'awaitPromise': True,
        }
        if call_arguments:
            params['arguments'] = call_arguments

        result = await self._client.send(
            'Runtime.callFunctionOn',
            params,
            session_id=self._session_id
        )

        if 'exceptionDetails' in result:
            raise RuntimeError(f'JavaScript evaluation failed: {result["exceptionDetails"]}')

        value = result.get('result', {}).get('value')

        if value is None:
            return ''
        elif isinstance(value, str):
            return value
        else:
            try:
                return json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            except (TypeError, ValueError):
                return str(value)

    async def get_basic_info(self) -> ElementInfo:
        """Get basic information about the element."""
        try:
            node_id = await self._get_node_id()
            result = await self._client.send(
                'DOM.describeNode',
                {'nodeId': node_id},
                session_id=self._session_id
            )

            node_info = result['node']
            bounding_box = await self.get_bounding_box()

            attributes_list = node_info.get('attributes', [])
            attributes_dict: dict[str, str] = {}
            for i in range(0, len(attributes_list), 2):
                if i + 1 < len(attributes_list):
                    attributes_dict[attributes_list[i]] = attributes_list[i + 1]

            return ElementInfo(
                backendNodeId=self._backend_node_id,
                nodeId=node_id,
                nodeName=node_info.get('nodeName', ''),
                nodeType=node_info.get('nodeType', 0),
                nodeValue=node_info.get('nodeValue'),
                attributes=attributes_dict,
                boundingBox=bounding_box,
                error=None,
            )
        except Exception as e:
            return ElementInfo(
                backendNodeId=self._backend_node_id,
                nodeId=None,
                nodeName='',
                nodeType=0,
                nodeValue=None,
                attributes={},
                boundingBox=None,
                error=str(e),
            )

