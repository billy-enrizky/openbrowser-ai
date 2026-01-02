"""Browser profile configuration following browser-use pattern.

This module provides configuration classes for browser launch parameters,
proxy settings, viewport dimensions, and other browser-related settings.
It follows the browser-use pattern for structured configuration management.

Classes:
    ProxySettings: HTTP/SOCKS proxy configuration for browser traffic.
    ViewportSize: Browser viewport/window size configuration.
    BrowserProfile: Complete browser configuration including launch args,
        security settings, video recording, and DOM processing options.
"""

import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProxySettings(BaseModel):
    """Typed proxy settings for Chromium browser traffic routing.

    Configures HTTP or SOCKS proxy for all browser network requests.
    Supports authentication and bypass rules for specific hosts.

    Attributes:
        server: Proxy URL (e.g., 'http://host:8080' or 'socks5://host:1080').
        bypass: Comma-separated hosts to bypass (e.g., 'localhost,127.0.0.1').
        username: Proxy authentication username.
        password: Proxy authentication password.

    Example:
        >>> proxy = ProxySettings(
        ...     server="http://proxy.example.com:8080",
        ...     bypass="localhost,*.internal",
        ...     username="user",
        ...     password="pass"
        ... )
    """

    server: str | None = Field(default=None, description='Proxy URL, e.g. http://host:8080 or socks5://host:1080')
    bypass: str | None = Field(default=None, description='Comma-separated hosts to bypass, e.g. localhost,127.0.0.1,*.internal')
    username: str | None = Field(default=None, description='Proxy auth username')
    password: str | None = Field(default=None, description='Proxy auth password')


class ViewportSize(BaseModel):
    """Viewport size configuration for browser windows and videos.

    Represents width and height dimensions in pixels. Supports both
    attribute access and dict-like access for compatibility with
    various APIs.

    Attributes:
        width: Width in pixels (must be >= 0).
        height: Height in pixels (must be >= 0).

    Example:
        >>> size = ViewportSize(width=1920, height=1080)
        >>> size['width']  # Dict-like access
        1920
        >>> size.height  # Attribute access
        1080
    """

    width: int = Field(ge=0)
    height: int = Field(ge=0)

    def __getitem__(self, key: str) -> int:
        """Get dimension by key for dict-like access.

        Args:
            key: Either 'width' or 'height'.

        Returns:
            The dimension value as an integer.
        """
        return dict(self)[key]

    def __setitem__(self, key: str, value: int) -> None:
        """Set dimension by key for dict-like access.

        Args:
            key: Either 'width' or 'height'.
            value: The new dimension value.
        """
        setattr(self, key, value)


class BrowserProfile(BaseModel):
    """Browser profile configuration following browser-use pattern.

    Comprehensive configuration class that manages all browser settings including:
    - User data directory for persistent profiles
    - Proxy settings for network traffic routing
    - Window and viewport size configuration
    - Browser launch arguments and executable path
    - Downloads and video recording configuration
    - Security settings (allowed/prohibited domains)
    - DOM processing and element highlighting options

    The profile generates Chrome CLI arguments via get_args() for browser launch.

    Attributes:
        user_data_dir: Path to Chrome user data directory for profile persistence.
        proxy: ProxySettings instance for network proxy configuration.
        window_size: Browser window dimensions when not headless.
        viewport: Viewport size for headless mode.
        headless: Whether to run browser without visible UI.
        executable_path: Custom path to browser executable.
        args: Additional CLI arguments to pass to browser.
        downloads_path: Directory for downloaded files.
        record_video_dir: Directory for session video recordings.
        storage_state: Path or dict for cookie/localStorage persistence.
        disable_security: Disable browser security features (for testing).
        allowed_domains: Whitelist of allowed navigation domains.
        prohibited_domains: Blacklist of blocked navigation domains.
        cross_origin_iframes: Enable cross-origin iframe processing.
        highlight_elements: Highlight interactive elements on page.

    Example:
        >>> profile = BrowserProfile(
        ...     headless=False,
        ...     window_size=ViewportSize(width=1920, height=1080),
        ...     downloads_path="/tmp/downloads",
        ...     allowed_domains=["*.example.com"],
        ... )
        >>> args = profile.get_args()
    """

    model_config = ConfigDict(
        extra='ignore',
        validate_assignment=True,
        revalidate_instances='always',
        from_attributes=True,
        validate_by_name=True,
        validate_by_alias=True,
    )

    # User data directory
    user_data_dir: str | Path | None = Field(
        default=None,
        description='User data directory for Chrome profile. If None, uses temporary directory.',
    )

    # Proxy settings
    proxy: ProxySettings | None = Field(default=None, description='Proxy settings')

    # Window/viewport settings
    window_size: ViewportSize | None = Field(default=None, description='Browser window size when headless=False')
    window_position: ViewportSize | None = Field(
        default=ViewportSize(width=0, height=0),
        description='Window position (x, y) from top left when headless=False',
    )
    viewport: ViewportSize | None = Field(default=None, description='Viewport size for headless mode')

    # Browser launch settings
    headless: bool | None = Field(default=None, description='Whether to run browser in headless mode')
    executable_path: str | Path | None = Field(default=None, description='Path to browser executable')
    args: list[str] = Field(default_factory=list, description='Additional CLI args to pass to browser')

    # Downloads
    downloads_path: str | Path | None = Field(
        default=None,
        description='Directory to save downloads to',
        validation_alias='downloads_dir',
    )

    # Video recording
    record_video_dir: Path | None = Field(
        default=None,
        description='Directory to save video recordings. If set, a video of the session will be recorded.',
        validation_alias='save_recording_path',
    )
    record_video_size: ViewportSize | None = Field(
        default=None, description='Video frame size. If not set, uses viewport size.'
    )
    record_video_framerate: int = Field(default=30, description='Framerate for video recording')
    record_video_format: str = Field(default='mp4', description='Video recording format (e.g., mp4, avi)')

    # Storage state
    storage_state: str | Path | dict | None = Field(
        default=None, description='Storage state file path or dict for cookies/localStorage persistence'
    )

    # Security settings
    disable_security: bool = Field(default=False, description='Disable browser security features')

    # User agent
    user_agent: str | None = Field(default=None, description='Custom user agent string')

    # Profile directory
    profile_directory: str = Field(default='Default', description='Chrome profile directory name')

    # Keep browser alive
    keep_alive: bool | None = Field(default=None, description='Keep browser alive after agent run')

    # Permissions
    permissions: list[str] = Field(
        default_factory=lambda: ['clipboardReadWrite', 'notifications'],
        description='Browser permissions to grant (CDP Browser.grantPermissions).',
    )

    # Security settings
    allowed_domains: list[str] | set[str] | None = Field(
        default=None,
        description='List of allowed domains for navigation e.g. ["*.google.com", "https://example.com"]',
    )
    prohibited_domains: list[str] | set[str] | None = Field(
        default=None,
        description='List of prohibited domains for navigation. Allowed domains take precedence.',
    )
    auto_download_pdfs: bool = Field(default=True, description='Automatically download PDFs when navigating to PDF viewer pages')

    # Browser connection settings
    is_local: bool = Field(default=True, description='Whether this is a local browser instance')

    # DOM processing settings
    cross_origin_iframes: bool = Field(
        default=True,
        description='Enable cross-origin iframe support (OOPIF/Out-of-Process iframes). When False, only same-origin frames are processed.',
    )
    max_iframes: int = Field(
        default=100,
        description='Maximum number of iframe documents to process to prevent crashes.',
    )
    max_iframe_depth: int = Field(
        default=5,
        ge=0,
        description='Maximum depth for cross-origin iframe recursion (default: 5 levels deep).',
    )
    paint_order_filtering: bool = Field(
        default=True,
        description='Enable paint order filtering. Slightly experimental.',
    )
    highlight_elements: bool = Field(
        default=True,
        description='Highlight interactive elements on the page.',
    )
    dom_highlight_elements: bool = Field(
        default=False,
        description='Highlight interactive elements in the DOM (only for debugging purposes).',
    )
    interaction_highlight_color: str = Field(
        default='rgb(255, 127, 39)',
        description='Color to use for highlighting elements during interactions (CSS color string).',
    )
    interaction_highlight_duration: float = Field(
        default=1.0,
        description='Duration in seconds to show interaction highlights.',
    )

    def __init__(self, **kwargs):
        """Initialize BrowserProfile with defaults.

        Creates a temporary user data directory if not provided.
        Resolves and expands any path strings to absolute Path objects.

        Args:
            **kwargs: Profile configuration parameters. See class attributes
                for available options.

        Example:
            >>> profile = BrowserProfile(headless=True)
            >>> profile.user_data_dir  # Auto-generated temp directory
            PosixPath('/tmp/openbrowser-user-data-dir-xyz123')
        """
        # Set default user_data_dir if not provided
        if 'user_data_dir' not in kwargs or kwargs.get('user_data_dir') is None:
            kwargs['user_data_dir'] = tempfile.mkdtemp(prefix='openbrowser-user-data-dir-')

        super().__init__(**kwargs)

        # Ensure user_data_dir is a Path
        if isinstance(self.user_data_dir, str):
            self.user_data_dir = Path(self.user_data_dir).expanduser().resolve()

    def get_args(self) -> list[str]:
        """Get the list of all Chrome CLI launch arguments for this profile.

        Constructs command-line arguments based on profile configuration including
        user data directory, headless mode, window size, proxy, and security settings.

        Returns:
            List of Chrome CLI argument strings ready to pass to subprocess.

        Example:
            >>> profile = BrowserProfile(headless=True, window_size=ViewportSize(width=1920, height=1080))
            >>> args = profile.get_args()
            >>> '--headless=new' in args
            True
        """
        args = []

        # User data directory
        if self.user_data_dir:
            args.append(f'--user-data-dir={self.user_data_dir}')
            args.append(f'--profile-directory={self.profile_directory}')

        # Headless mode
        if self.headless:
            args.append('--headless=new')

        # Window size
        if self.window_size and not self.headless:
            args.append(f'--window-size={self.window_size["width"]},{self.window_size["height"]}')

        # Window position
        if self.window_position and not self.headless:
            args.append(f'--window-position={self.window_position["width"]},{self.window_position["height"]}')

        # Proxy settings
        if self.proxy and self.proxy.server:
            args.append(f'--proxy-server={self.proxy.server}')
            if self.proxy.bypass:
                args.append(f'--proxy-bypass-list={self.proxy.bypass}')

        # User agent
        if self.user_agent:
            args.append(f'--user-agent={self.user_agent}')

        # Security settings
        if self.disable_security:
            args.extend([
                '--disable-site-isolation-trials',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
                '--allow-running-insecure-content',
                '--ignore-certificate-errors',
                '--ignore-ssl-errors',
            ])

        # Additional args
        args.extend(self.args)

        return args

