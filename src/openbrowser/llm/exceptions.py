"""LLM exceptions module for handling LLM-specific errors."""


class ModelError(Exception):
    """Base exception for all model-related errors."""
    pass


class ModelProviderError(ModelError):
    """Exception raised when a model provider returns an error."""

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        model: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.model = model

    def __str__(self) -> str:
        if self.model:
            return f"[{self.status_code}] {self.model}: {self.message}"
        return f"[{self.status_code}] {self.message}"


class ModelRateLimitError(ModelProviderError):
    """Exception raised when a model provider returns a rate limit error."""

    def __init__(
        self,
        message: str,
        status_code: int = 429,
        model: str | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message, status_code, model)
        self.retry_after = retry_after


class ModelAuthenticationError(ModelProviderError):
    """Exception raised when authentication with a model provider fails."""

    def __init__(
        self,
        message: str,
        status_code: int = 401,
        model: str | None = None,
    ):
        super().__init__(message, status_code, model)


class ModelContextLengthError(ModelProviderError):
    """Exception raised when the context length exceeds the model's limit."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        model: str | None = None,
        max_tokens: int | None = None,
        requested_tokens: int | None = None,
    ):
        super().__init__(message, status_code, model)
        self.max_tokens = max_tokens
        self.requested_tokens = requested_tokens


class ModelInvalidResponseError(ModelError):
    """Exception raised when the model returns an invalid or unparseable response."""

    def __init__(
        self,
        message: str,
        raw_response: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.raw_response = raw_response


class ModelTimeoutError(ModelError):
    """Exception raised when a model request times out."""

    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.timeout_seconds = timeout_seconds


class ModelNotFoundError(ModelProviderError):
    """Exception raised when the requested model is not found."""

    def __init__(
        self,
        message: str,
        status_code: int = 404,
        model: str | None = None,
    ):
        super().__init__(message, status_code, model)


class LLMException(Exception):
    """Generic exception for LLM interaction errors.
    
    This is a convenience alias for backward compatibility.
    """

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message

