"""Screenshot storage service for openbrowser agents.

Performance optimizations:
- Added __slots__ to ScreenshotService class
- Cached screenshots_dir path
- Pre-decode base64 module reference
"""

import base64
from pathlib import Path

import anyio

from openbrowser.observability import observe_debug

# Cache base64 functions at module level for faster access
_b64decode = base64.b64decode
_b64encode = base64.b64encode


class ScreenshotService:
	"""Simple screenshot storage service that saves screenshots to disk.
	
	Performance optimizations:
	- __slots__ for faster attribute access
	- Cached directory path
	- Module-level base64 function references
	"""
	
	__slots__ = ('agent_directory', 'screenshots_dir')

	def __init__(self, agent_directory: str | Path):
		"""Initialize with agent directory path"""
		self.agent_directory = Path(agent_directory) if isinstance(agent_directory, str) else agent_directory

		# Create screenshots subdirectory and cache path
		self.screenshots_dir = self.agent_directory / 'screenshots'
		self.screenshots_dir.mkdir(parents=True, exist_ok=True)

	@observe_debug(ignore_input=True, ignore_output=True, name='store_screenshot')
	async def store_screenshot(self, screenshot_b64: str, step_number: int) -> str:
		"""Store screenshot to disk and return the full path as string"""
		screenshot_filename = f'step_{step_number}.png'
		screenshot_path = self.screenshots_dir / screenshot_filename

		# Decode base64 and save to disk using cached function
		screenshot_data = _b64decode(screenshot_b64)

		async with await anyio.open_file(screenshot_path, 'wb') as f:
			await f.write(screenshot_data)

		return str(screenshot_path)

	@observe_debug(ignore_input=True, ignore_output=True, name='get_screenshot_from_disk')
	async def get_screenshot(self, screenshot_path: str) -> str | None:
		"""Load screenshot from disk path and return as base64"""
		if not screenshot_path:
			return None

		path = Path(screenshot_path)
		if not path.exists():
			return None

		# Load from disk and encode to base64 using cached function
		async with await anyio.open_file(path, 'rb') as f:
			screenshot_data = await f.read()

		return _b64encode(screenshot_data).decode('utf-8')
