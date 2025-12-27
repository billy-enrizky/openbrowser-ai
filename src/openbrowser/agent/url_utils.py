"""URL utilities for shortening long URLs in messages.

This module provides functions to replace long URLs with shortened versions
to reduce token usage while preserving the ability to restore them.
"""

import hashlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Regex pattern to match URLs
URL_PATTERN = re.compile(r'https?://[^\s<>"\']+')

# Default limit for URL length before shortening
DEFAULT_URL_LIMIT = 50


def shorten_url(url: str, limit: int = DEFAULT_URL_LIMIT) -> str:
    """Shorten a URL if it exceeds the limit.
    
    Args:
        url: URL to potentially shorten
        limit: Maximum length before shortening
        
    Returns:
        Shortened URL or original if under limit
    """
    if len(url) <= limit:
        return url
    
    # Generate a hash-based suffix
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    
    # Keep the domain and start of path
    try:
        # Extract protocol and domain
        protocol_end = url.find('://') + 3
        domain_end = url.find('/', protocol_end) if '/' in url[protocol_end:] else len(url)
        domain = url[:domain_end]
        
        # Calculate how much of the path we can keep
        remaining = limit - len(domain) - len(url_hash) - 5  # 5 for "/.../."
        
        if remaining > 10:
            path_start = url[domain_end:domain_end + remaining // 2]
            shortened = f"{domain}{path_start}...{url_hash}"
        else:
            shortened = f"{domain[:limit - len(url_hash) - 4]}...{url_hash}"
        
        return shortened
    except Exception:
        # Fallback to simple truncation
        return f"{url[:limit - len(url_hash) - 3]}...{url_hash}"


def replace_urls_in_text(
    text: str,
    limit: int = DEFAULT_URL_LIMIT,
) -> tuple[str, dict[str, str]]:
    """Replace long URLs in text with shortened versions.
    
    Args:
        text: Text containing URLs
        limit: Maximum URL length before shortening
        
    Returns:
        Tuple of (modified_text, replacements_dict)
        The replacements_dict maps shortened URLs to original URLs
    """
    replacements: dict[str, str] = {}
    
    def replace_url(match: re.Match) -> str:
        original_url = match.group(0)
        
        if len(original_url) <= limit:
            return original_url
        
        shortened = shorten_url(original_url, limit)
        
        # Only track if actually shortened
        if shortened != original_url:
            replacements[shortened] = original_url
        
        return shortened
    
    modified_text = URL_PATTERN.sub(replace_url, text)
    
    return modified_text, replacements


def restore_shortened_urls(
    text: str,
    url_replacements: dict[str, str],
) -> str:
    """Restore shortened URLs to their original form.
    
    Args:
        text: Text containing shortened URLs
        url_replacements: Dict mapping shortened URLs to originals
        
    Returns:
        Text with URLs restored
    """
    result = text
    
    for shortened, original in url_replacements.items():
        result = result.replace(shortened, original)
    
    return result


def extract_urls(text: str) -> list[str]:
    """Extract all URLs from text.
    
    Args:
        text: Text to extract URLs from
        
    Returns:
        List of URLs found
    """
    return URL_PATTERN.findall(text)


def normalize_url(url: str) -> str:
    """Normalize a URL for comparison.
    
    Removes trailing slashes, normalizes case, etc.
    
    Args:
        url: URL to normalize
        
    Returns:
        Normalized URL
    """
    url = url.strip()
    
    # Remove trailing slash
    while url.endswith('/'):
        url = url[:-1]
    
    # Lowercase the protocol and domain
    if '://' in url:
        protocol_end = url.find('://') + 3
        domain_end = url.find('/', protocol_end) if '/' in url[protocol_end:] else len(url)
        
        protocol_and_domain = url[:domain_end].lower()
        path = url[domain_end:]
        
        url = protocol_and_domain + path
    
    return url


class URLShortener:
    """Stateful URL shortener that tracks replacements.
    
    Useful for agent sessions where URLs need to be tracked across
    multiple messages.
    """
    
    def __init__(self, limit: int = DEFAULT_URL_LIMIT):
        self.limit = limit
        self.replacements: dict[str, str] = {}
    
    def shorten_in_text(self, text: str) -> str:
        """Shorten URLs in text and track replacements.
        
        Args:
            text: Text to process
            
        Returns:
            Text with shortened URLs
        """
        modified, new_replacements = replace_urls_in_text(text, self.limit)
        self.replacements.update(new_replacements)
        return modified
    
    def restore_in_text(self, text: str) -> str:
        """Restore shortened URLs in text.
        
        Args:
            text: Text with shortened URLs
            
        Returns:
            Text with original URLs
        """
        return restore_shortened_urls(text, self.replacements)
    
    def clear(self) -> None:
        """Clear all tracked replacements."""
        self.replacements.clear()
    
    def get_original_url(self, shortened: str) -> str | None:
        """Get the original URL for a shortened version.
        
        Args:
            shortened: Shortened URL
            
        Returns:
            Original URL or None if not found
        """
        return self.replacements.get(shortened)

