"""Page class for page-level operations using CDP."""

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from src.openbrowser.actor.utils import get_key_info

if TYPE_CHECKING:
    from src.openbrowser.actor.element import Element
    from src.openbrowser.actor.mouse import Mouse
    from src.openbrowser.browser.session import BrowserSession
    from src.openbrowser.llm.base import BaseChatModel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class Page:
    """Page operations (tab or iframe).
    
    Provides page-level interactions using Chrome DevTools Protocol.
    """

    def __init__(
        self,
        browser_session: 'BrowserSession',
        target_id: str,
        session_id: str | None = None,
        llm: 'BaseChatModel | None' = None,
    ):
        self._browser_session = browser_session
        self._target_id = target_id
        self._session_id: str | None = session_id
        self._mouse: 'Mouse | None' = None
        self._llm = llm

    @property
    def _client(self):
        """Get the CDP client from browser session."""
        return self._browser_session.cdp_client

    async def _ensure_session(self) -> str:
        """Ensure we have a session ID for this target."""
        if not self._session_id:
            result = await self._client.send(
                'Target.attachToTarget',
                {'targetId': self._target_id, 'flatten': True}
            )
            self._session_id = result['sessionId']

            # Enable necessary domains
            await asyncio.gather(
                self._client.send('Page.enable', session_id=self._session_id),
                self._client.send('DOM.enable', session_id=self._session_id),
                self._client.send('Runtime.enable', session_id=self._session_id),
                self._client.send('Network.enable', session_id=self._session_id),
            )

        return self._session_id

    @property
    async def session_id(self) -> str:
        """Get the session ID for this target."""
        return await self._ensure_session()

    @property
    async def mouse(self) -> 'Mouse':
        """Get the mouse interface for this target."""
        if not self._mouse:
            session_id = await self._ensure_session()
            from src.openbrowser.actor.mouse import Mouse
            self._mouse = Mouse(self._browser_session, session_id, self._target_id)
        return self._mouse

    async def reload(self) -> None:
        """Reload the page."""
        session_id = await self._ensure_session()
        await self._client.send('Page.reload', session_id=session_id)

    async def get_element(self, backend_node_id: int) -> 'Element':
        """Get an element by its backend node ID.
        
        Args:
            backend_node_id: CDP backend node ID
            
        Returns:
            Element instance
        """
        session_id = await self._ensure_session()
        from src.openbrowser.actor.element import Element
        return Element(self._browser_session, backend_node_id, session_id)

    async def evaluate(self, page_function: str, *args) -> str:
        """Execute JavaScript in the page context.
        
        Args:
            page_function: JavaScript arrow function (must start with (...args) =>)
            *args: Arguments to pass to the function
            
        Returns:
            String representation of the result
        """
        session_id = await self._ensure_session()

        # Clean and fix common JavaScript string parsing issues
        page_function = self._fix_javascript_string(page_function)

        # Enforce arrow function format
        if not (page_function.startswith('(') and '=>' in page_function):
            if not (page_function.startswith('async') and '=>' in page_function):
                raise ValueError(
                    f'JavaScript code must start with (...args) => format. Got: {page_function[:50]}...'
                )

        # Build the expression
        if args:
            arg_strs = [json.dumps(arg) for arg in args]
            expression = f'({page_function})({", ".join(arg_strs)})'
        else:
            expression = f'({page_function})()'

        result = await self._client.send(
            'Runtime.evaluate',
            {
                'expression': expression,
                'returnByValue': True,
                'awaitPromise': True,
            },
            session_id=session_id
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

    def _fix_javascript_string(self, js_code: str) -> str:
        """Fix common JavaScript string parsing issues."""
        js_code = js_code.strip()

        # Remove obvious Python string wrapper quotes if they exist
        if (js_code.startswith('"') and js_code.endswith('"')) or \
           (js_code.startswith("'") and js_code.endswith("'")):
            inner = js_code[1:-1]
            if inner.count('"') + inner.count("'") == 0 or '() =>' in inner:
                js_code = inner

        # Fix clearly escaped quotes
        if '\\"' in js_code and js_code.count('\\"') > js_code.count('"'):
            js_code = js_code.replace('\\"', '"')
        if "\\'" in js_code and js_code.count("\\'") > js_code.count("'"):
            js_code = js_code.replace("\\'", "'")

        js_code = js_code.strip()

        if not js_code:
            raise ValueError('JavaScript code is empty after cleaning')

        return js_code

    async def screenshot(self, format: str = 'jpeg', quality: int | None = None) -> str:
        """Take a screenshot of the page.
        
        Args:
            format: Image format ('jpeg', 'png', 'webp')
            quality: Quality 0-100 for JPEG format
            
        Returns:
            Base64-encoded image data
        """
        session_id = await self._ensure_session()

        params = {'format': format}
        if quality is not None and format.lower() == 'jpeg':
            params['quality'] = quality

        result = await self._client.send(
            'Page.captureScreenshot',
            params,
            session_id=session_id
        )

        return result['data']

    async def press(self, key: str) -> None:
        """Press a key on the page.
        
        Args:
            key: Key name or combination (e.g., 'Enter', 'Control+A')
        """
        session_id = await self._ensure_session()

        if '+' in key:
            # Handle key combinations
            parts = key.split('+')
            modifiers = parts[:-1]
            main_key = parts[-1]

            # Calculate modifier bitmask
            modifier_value = 0
            modifier_map = {'Alt': 1, 'Control': 2, 'Meta': 4, 'Shift': 8}
            for mod in modifiers:
                modifier_value |= modifier_map.get(mod, 0)

            # Press modifier keys
            for mod in modifiers:
                code, vk_code = get_key_info(mod)
                params = {'type': 'keyDown', 'key': mod, 'code': code}
                if vk_code is not None:
                    params['windowsVirtualKeyCode'] = vk_code
                await self._client.send('Input.dispatchKeyEvent', params, session_id=session_id)

            # Press main key
            main_code, main_vk_code = get_key_info(main_key)
            main_params = {
                'type': 'keyDown',
                'key': main_key,
                'code': main_code,
                'modifiers': modifier_value,
            }
            if main_vk_code is not None:
                main_params['windowsVirtualKeyCode'] = main_vk_code
            await self._client.send('Input.dispatchKeyEvent', main_params, session_id=session_id)

            main_up_params = {
                'type': 'keyUp',
                'key': main_key,
                'code': main_code,
                'modifiers': modifier_value,
            }
            if main_vk_code is not None:
                main_up_params['windowsVirtualKeyCode'] = main_vk_code
            await self._client.send('Input.dispatchKeyEvent', main_up_params, session_id=session_id)

            # Release modifier keys
            for mod in reversed(modifiers):
                code, vk_code = get_key_info(mod)
                params = {'type': 'keyUp', 'key': mod, 'code': code}
                if vk_code is not None:
                    params['windowsVirtualKeyCode'] = vk_code
                await self._client.send('Input.dispatchKeyEvent', params, session_id=session_id)
        else:
            # Simple key press
            code, vk_code = get_key_info(key)
            down_params = {'type': 'keyDown', 'key': key, 'code': code}
            if vk_code is not None:
                down_params['windowsVirtualKeyCode'] = vk_code
            await self._client.send('Input.dispatchKeyEvent', down_params, session_id=session_id)

            up_params = {'type': 'keyUp', 'key': key, 'code': code}
            if vk_code is not None:
                up_params['windowsVirtualKeyCode'] = vk_code
            await self._client.send('Input.dispatchKeyEvent', up_params, session_id=session_id)

    async def set_viewport_size(self, width: int, height: int) -> None:
        """Set the viewport size.
        
        Args:
            width: Viewport width
            height: Viewport height
        """
        session_id = await self._ensure_session()
        await self._client.send(
            'Emulation.setDeviceMetricsOverride',
            {
                'width': width,
                'height': height,
                'deviceScaleFactor': 1.0,
                'mobile': False,
            },
            session_id=session_id
        )

    async def get_target_info(self) -> dict:
        """Get target information."""
        result = await self._client.send(
            'Target.getTargetInfo',
            {'targetId': self._target_id}
        )
        return result['targetInfo']

    async def get_url(self) -> str:
        """Get the current URL."""
        info = await self.get_target_info()
        return info.get('url', '')

    async def get_title(self) -> str:
        """Get the current page title."""
        info = await self.get_target_info()
        return info.get('title', '')

    async def goto(self, url: str) -> None:
        """Navigate to a URL.
        
        Args:
            url: Target URL
        """
        session_id = await self._ensure_session()
        await self._client.send(
            'Page.navigate',
            {'url': url},
            session_id=session_id
        )

    async def navigate(self, url: str) -> None:
        """Alias for goto."""
        await self.goto(url)

    async def go_back(self) -> None:
        """Navigate back in history."""
        session_id = await self._ensure_session()

        try:
            history = await self._client.send(
                'Page.getNavigationHistory',
                session_id=session_id
            )
            current_index = history['currentIndex']
            entries = history['entries']

            if current_index <= 0:
                raise RuntimeError('Cannot go back - no previous entry in history')

            previous_entry_id = entries[current_index - 1]['id']
            await self._client.send(
                'Page.navigateToHistoryEntry',
                {'entryId': previous_entry_id},
                session_id=session_id
            )
        except Exception as e:
            raise RuntimeError(f'Failed to navigate back: {e}')

    async def go_forward(self) -> None:
        """Navigate forward in history."""
        session_id = await self._ensure_session()

        try:
            history = await self._client.send(
                'Page.getNavigationHistory',
                session_id=session_id
            )
            current_index = history['currentIndex']
            entries = history['entries']

            if current_index >= len(entries) - 1:
                raise RuntimeError('Cannot go forward - no next entry in history')

            next_entry_id = entries[current_index + 1]['id']
            await self._client.send(
                'Page.navigateToHistoryEntry',
                {'entryId': next_entry_id},
                session_id=session_id
            )
        except Exception as e:
            raise RuntimeError(f'Failed to navigate forward: {e}')

    async def get_elements_by_css_selector(self, selector: str) -> list['Element']:
        """Get elements by CSS selector.
        
        Args:
            selector: CSS selector
            
        Returns:
            List of matching Element instances
        """
        session_id = await self._ensure_session()
        from src.openbrowser.actor.element import Element

        # Get document
        doc_result = await self._client.send('DOM.getDocument', session_id=session_id)
        document_node_id = doc_result['root']['nodeId']

        # Query selector all
        result = await self._client.send(
            'DOM.querySelectorAll',
            {'nodeId': document_node_id, 'selector': selector},
            session_id=session_id
        )

        elements = []
        for node_id in result.get('nodeIds', []):
            # Get backend node ID
            node_result = await self._client.send(
                'DOM.describeNode',
                {'nodeId': node_id},
                session_id=session_id
            )
            backend_node_id = node_result['node']['backendNodeId']
            elements.append(Element(self._browser_session, backend_node_id, session_id))

        return elements

    async def get_element_by_prompt(
        self,
        prompt: str,
        llm: 'BaseChatModel | None' = None,
    ) -> 'Element | None':
        """Get an element by natural language prompt using LLM.
        
        Args:
            prompt: Natural language description of the element
            llm: LLM to use (defaults to instance LLM)
            
        Returns:
            Element if found, None otherwise
        """
        from src.openbrowser.agent.views import SystemMessage, UserMessage
        from src.openbrowser.browser.dom.service import DomService

        llm = llm or self._llm
        if not llm:
            raise ValueError('LLM not provided')

        await self._ensure_session()

        # Get DOM state
        dom_service = DomService(self._browser_session)
        dom_state = await dom_service.get_serialized_dom_state()

        system_message = SystemMessage(
            content='''You are an AI created to find an element on a page by a prompt.

Interactive elements are provided in format: [index]<type>text</type>
- index: Numeric identifier for interaction
- type: HTML element type
- text: Element description

Your task is to find the element index that matches the prompt.
Return only the index number, or null if not found.'''
        )

        user_message = UserMessage(
            content=f'''
<browser_state>
{dom_state['element_tree']}
</browser_state>

<prompt>
{prompt}
</prompt>

Return the element index that best matches the prompt, or null if not found.
'''
        )

        class ElementResponse(BaseModel):
            element_highlight_index: int | None

        response = await llm.ainvoke(
            [system_message, user_message],
            output_format=ElementResponse,
        )

        element_index = response.completion.element_highlight_index
        if element_index is None:
            return None

        # Get element from selector map
        selector_map = dom_state.get('selector_map', {})
        if element_index not in selector_map:
            return None

        element_info = selector_map[element_index]
        from src.openbrowser.actor.element import Element
        return Element(
            self._browser_session,
            element_info.get('backend_node_id', 0),
            self._session_id
        )

    async def must_get_element_by_prompt(
        self,
        prompt: str,
        llm: 'BaseChatModel | None' = None,
    ) -> 'Element':
        """Get an element by prompt, raising error if not found.
        
        Args:
            prompt: Natural language description
            llm: LLM to use
            
        Returns:
            Element instance
            
        Raises:
            ValueError: If element not found
        """
        element = await self.get_element_by_prompt(prompt, llm)
        if element is None:
            raise ValueError(f'No element found for prompt: {prompt}')
        return element

    async def extract_content(
        self,
        prompt: str,
        structured_output: type[T],
        llm: 'BaseChatModel | None' = None,
    ) -> T:
        """Extract structured content from the page using LLM.
        
        Args:
            prompt: Description of what content to extract
            structured_output: Pydantic model for output structure
            llm: LLM to use
            
        Returns:
            Structured data as specified model instance
        """
        from src.openbrowser.agent.views import SystemMessage, UserMessage

        llm = llm or self._llm
        if not llm:
            raise ValueError('LLM not provided')

        # Extract page content
        content = await self._extract_page_text()

        system_prompt = '''
You are an expert at extracting structured data from webpage content.

Instructions:
- Extract information relevant to the query
- Only use information from the provided content
- If information is not available, mention that
- Return data in the exact structured format specified
'''.strip()

        prompt_content = f'<query>\n{prompt}\n</query>\n\n<webpage_content>\n{content}\n</webpage_content>'

        try:
            response = await asyncio.wait_for(
                llm.ainvoke(
                    [
                        SystemMessage(content=system_prompt),
                        UserMessage(content=prompt_content),
                    ],
                    output_format=structured_output,
                ),
                timeout=120.0,
            )
            return response.completion
        except Exception as e:
            raise RuntimeError(str(e))

    async def _extract_page_text(self) -> str:
        """Extract text content from the page."""
        session_id = await self._ensure_session()

        result = await self._client.send(
            'Runtime.evaluate',
            {
                'expression': 'document.body.innerText',
                'returnByValue': True,
            },
            session_id=session_id
        )

        return result.get('result', {}).get('value', '')

