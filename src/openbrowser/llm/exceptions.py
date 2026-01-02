"""LLM exceptions module for handling LLM-specific errors.

This module provides a hierarchy of exception classes for handling errors
that occur when interacting with LLM providers. The exceptions are designed
to capture provider-specific error information while providing a consistent
interface for error handling.

Exception Hierarchy:
    ModelError
        └── ModelProviderError
            ├── ModelRateLimitError
            ├── ModelAuthenticationError
            ├── ModelContextLengthError
            └── ModelNotFoundError
        └── ModelInvalidResponseError
        └── ModelTimeoutError
    LLMException (convenience alias)

Example:
    >>> try:
    ...     response = await llm.ainvoke(messages)
    ... except ModelRateLimitError as e:
    ...     await asyncio.sleep(e.retry_after or 60)
    ... except ModelAuthenticationError:
    ...     raise ValueError("Invalid API key")
"""


class ModelError(Exception):
    """Base exception for all model-related errors.
    
    This is the root exception class for all LLM-related errors. It can be
    used to catch any model error regardless of the specific type.
    
    Example:
        >>> try:
        ...     response = await llm.ainvoke(messages)
        ... except ModelError as e:
        ...     logger.error(f"Model error: {e}")
    """
    pass


class ModelProviderError(ModelError):
    """Exception raised when a model provider returns an error.
    
    This exception is used for errors returned by the LLM provider's API,
    such as server errors, bad requests, or service unavailability.
    
    Args:
        message: Human-readable error message.
        status_code: HTTP status code from the provider (default: 502).
        model: Name of the model that was being used when the error occurred.
    
    Attributes:
        message: The error message.
        status_code: HTTP status code.
        model: The model name, if available.
    
    Example:
        >>> raise ModelProviderError(
        ...     message="Service temporarily unavailable",
        ...     status_code=503,
        ...     model="gpt-4o"
        ... )
    """

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        model: str | None = None,
    ):
        """Initialize ModelProviderError.
        
        Args:
            message: Human-readable error description.
            status_code: HTTP status code from the provider. Defaults to 502
                (Bad Gateway) for generic provider errors.
            model: Optional model identifier that was being used.
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.model = model

    def __str__(self) -> str:
        """Return formatted error string with status code and model.
        
        Returns:
            str: Formatted error message including status code and model name
                if available.
        """
        if self.model:
            return f"[{self.status_code}] {self.model}: {self.message}"
        return f"[{self.status_code}] {self.message}"


class ModelRateLimitError(ModelProviderError):
    """Exception raised when a model provider returns a rate limit error.
    
    This exception indicates that the request was rejected due to rate limiting.
    It includes optional retry information to help implement backoff strategies.
    
    Args:
        message: Human-readable error message.
        status_code: HTTP status code (default: 429 Too Many Requests).
        model: Name of the model.
        retry_after: Suggested wait time in seconds before retrying.
    
    Attributes:
        retry_after: Seconds to wait before retrying, if provided by the API.
    
    Example:
        >>> try:
        ...     response = await llm.ainvoke(messages)
        ... except ModelRateLimitError as e:
        ...     wait_time = e.retry_after or 60
        ...     await asyncio.sleep(wait_time)
    """

    def __init__(
        self,
        message: str,
        status_code: int = 429,
        model: str | None = None,
        retry_after: float | None = None,
    ):
        """Initialize ModelRateLimitError.
        
        Args:
            message: Human-readable error description.
            status_code: HTTP status code. Defaults to 429 (Too Many Requests).
            model: Optional model identifier.
            retry_after: Optional suggested wait time in seconds from the API.
        """
        super().__init__(message, status_code, model)
        self.retry_after = retry_after


class ModelAuthenticationError(ModelProviderError):
    """Exception raised when authentication with a model provider fails.
    
    This exception indicates that the API key or credentials provided
    are invalid, expired, or missing required permissions.
    
    Args:
        message: Human-readable error message.
        status_code: HTTP status code (default: 401 Unauthorized).
        model: Name of the model.
    
    Example:
        >>> raise ModelAuthenticationError(
        ...     message="Invalid API key provided",
        ...     model="gpt-4o"
        ... )
    """

    def __init__(
        self,
        message: str,
        status_code: int = 401,
        model: str | None = None,
    ):
        """Initialize ModelAuthenticationError.
        
        Args:
            message: Human-readable error description.
            status_code: HTTP status code. Defaults to 401 (Unauthorized).
            model: Optional model identifier.
        """
        super().__init__(message, status_code, model)


class ModelContextLengthError(ModelProviderError):
    """Exception raised when the context length exceeds the model's limit.
    
    This exception indicates that the input (messages + expected output)
    exceeds the model's maximum context window. It provides information
    about the limits to help with context management strategies.
    
    Args:
        message: Human-readable error message.
        status_code: HTTP status code (default: 400 Bad Request).
        model: Name of the model.
        max_tokens: The model's maximum token limit.
        requested_tokens: The number of tokens that were requested.
    
    Attributes:
        max_tokens: Maximum tokens the model supports.
        requested_tokens: Number of tokens in the request.
    
    Example:
        >>> try:
        ...     response = await llm.ainvoke(long_messages)
        ... except ModelContextLengthError as e:
        ...     # Truncate messages to fit within limit
        ...     truncated = truncate_to_tokens(messages, e.max_tokens - 1000)
    """

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        model: str | None = None,
        max_tokens: int | None = None,
        requested_tokens: int | None = None,
    ):
        """Initialize ModelContextLengthError.
        
        Args:
            message: Human-readable error description.
            status_code: HTTP status code. Defaults to 400 (Bad Request).
            model: Optional model identifier.
            max_tokens: The maximum token limit for the model.
            requested_tokens: The number of tokens that were requested.
        """
        super().__init__(message, status_code, model)
        self.max_tokens = max_tokens
        self.requested_tokens = requested_tokens


class ModelInvalidResponseError(ModelError):
    """Exception raised when the model returns an invalid or unparseable response.
    
    This exception indicates that the model's response could not be parsed
    or validated, such as when expecting JSON but receiving malformed data.
    
    Args:
        message: Human-readable error message.
        raw_response: The raw response string that failed to parse.
    
    Attributes:
        message: The error message.
        raw_response: The original response for debugging.
    
    Example:
        >>> try:
        ...     result = structured_model.ainvoke(messages)
        ... except ModelInvalidResponseError as e:
        ...     logger.error(f"Failed to parse: {e.raw_response}")
    """

    def __init__(
        self,
        message: str,
        raw_response: str | None = None,
    ):
        """Initialize ModelInvalidResponseError.
        
        Args:
            message: Human-readable error description.
            raw_response: The raw response string that could not be parsed.
        """
        super().__init__(message)
        self.message = message
        self.raw_response = raw_response


class ModelTimeoutError(ModelError):
    """Exception raised when a model request times out.
    
    This exception indicates that the request took longer than the configured
    timeout period. This can happen with complex prompts, slow network connections,
    or overloaded API servers.
    
    Args:
        message: Human-readable error message.
        timeout_seconds: The timeout value that was exceeded.
    
    Attributes:
        message: The error message.
        timeout_seconds: The configured timeout in seconds.
    
    Example:
        >>> try:
        ...     response = await llm.ainvoke(messages)
        ... except ModelTimeoutError as e:
        ...     logger.warning(f"Request timed out after {e.timeout_seconds}s")
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
    ):
        """Initialize ModelTimeoutError.
        
        Args:
            message: Human-readable error description.
            timeout_seconds: The timeout value in seconds that was exceeded.
        """
        super().__init__(message)
        self.message = message
        self.timeout_seconds = timeout_seconds


class ModelNotFoundError(ModelProviderError):
    """Exception raised when the requested model is not found.
    
    This exception indicates that the specified model identifier does not
    exist or is not available for the current API key/account.
    
    Args:
        message: Human-readable error message.
        status_code: HTTP status code (default: 404 Not Found).
        model: Name of the model that was not found.
    
    Example:
        >>> raise ModelNotFoundError(
        ...     message="Model 'gpt-5' does not exist",
        ...     model="gpt-5"
        ... )
    """

    def __init__(
        self,
        message: str,
        status_code: int = 404,
        model: str | None = None,
    ):
        """Initialize ModelNotFoundError.
        
        Args:
            message: Human-readable error description.
            status_code: HTTP status code. Defaults to 404 (Not Found).
            model: The model identifier that was not found.
        """
        super().__init__(message, status_code, model)


class LLMException(Exception):
    """Generic exception for LLM interaction errors.
    
    This is a convenience exception class for backward compatibility and
    simple error handling cases where the specific error type is not important.
    
    Args:
        message: Human-readable error message.
        status_code: Optional HTTP status code.
    
    Attributes:
        message: The error message.
        status_code: HTTP status code, if applicable.
    
    Example:
        >>> raise LLMException("Failed to connect to LLM", status_code=503)
    """

    def __init__(self, message: str, status_code: int | None = None):
        """Initialize LLMException.
        
        Args:
            message: Human-readable error description.
            status_code: Optional HTTP status code for the error.
        """
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return formatted error string with status code if available.
        
        Returns:
            str: Formatted error message, including status code if present.
        """
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message

