"""Custom exceptions for openbrowser.

Performance optimizations:
- Added __slots__ for faster attribute access and reduced memory
"""


class LLMException(Exception):
	"""Exception raised for LLM-related errors.
	
	Attributes:
		status_code: HTTP status code or error code
		message: Error message describing the issue
	"""
	
	__slots__ = ('status_code', 'message')
	
	def __init__(self, status_code: int, message: str):
		self.status_code = status_code
		self.message = message
		super().__init__(f'Error {status_code}: {message}')
