"""Auth profile service -- CRUD + domain cookie filtering + CDP state export."""

import logging
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import AuthProfile
from app.services.kms_service import decrypt_auth_state, encrypt_auth_state

logger = logging.getLogger(__name__)


def filter_cookies_by_domain(cookies: list[dict[str, Any]], domain: str) -> list[dict[str, Any]]:
    """Filter cookies to match the target domain and its subdomains."""
    domain = domain.lower().lstrip(".")
    filtered = []
    for cookie in cookies:
        cookie_domain = cookie.get("domain", "").lower().lstrip(".")
        if cookie_domain == domain or cookie_domain.endswith(f".{domain}"):
            filtered.append(cookie)
    return filtered


async def count_profiles(db: AsyncSession, user_id: str) -> int:
    """Count auth profiles for a user."""
    result = await db.execute(
        select(func.count()).select_from(AuthProfile).where(AuthProfile.user_id == user_id)
    )
    return result.scalar_one()


async def save_profile(
    db: AsyncSession,
    user_id: str,
    domain: str,
    label: str,
    storage_state: dict[str, Any],
) -> AuthProfile:
    """Encrypt and save auth profile.

    Raises RuntimeError if user exceeds MAX_AUTH_PROFILES_PER_USER.
    """
    count = await count_profiles(db, user_id)
    if count >= settings.MAX_AUTH_PROFILES_PER_USER:
        raise RuntimeError(f"Maximum auth profiles ({settings.MAX_AUTH_PROFILES_PER_USER}) reached")

    cookies = storage_state.get("cookies", [])
    filtered_cookies = filter_cookies_by_domain(cookies, domain)
    filtered_state = {
        "cookies": filtered_cookies,
        "origins": storage_state.get("origins", []),
    }

    encrypted_key, encrypted_state = await encrypt_auth_state(user_id, filtered_state)

    profile = AuthProfile(
        user_id=user_id,
        domain=domain,
        label=label,
        encrypted_key=encrypted_key,
        encrypted_state=encrypted_state,
        status="active",
    )
    db.add(profile)
    await db.flush()
    return profile


async def list_profiles(db: AsyncSession, user_id: str) -> list[AuthProfile]:
    """List all auth profiles for a user."""
    result = await db.execute(
        select(AuthProfile)
        .where(AuthProfile.user_id == user_id)
        .order_by(AuthProfile.domain, AuthProfile.label)
    )
    return list(result.scalars().all())


async def get_profile(db: AsyncSession, profile_id: str, user_id: str) -> AuthProfile | None:
    """Get a single auth profile (scoped to user)."""
    result = await db.execute(
        select(AuthProfile).where(
            AuthProfile.id == profile_id,
            AuthProfile.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def load_auth_state(db: AsyncSession, profile_id: str, user_id: str) -> dict[str, Any]:
    """Load and decrypt auth state for use in a browser session.

    Raises ValueError if profile not found or not active.
    """
    profile = await get_profile(db, profile_id, user_id)
    if not profile:
        raise ValueError(f"Auth profile {profile_id} not found")
    if profile.status != "active":
        raise ValueError(f"Auth profile {profile_id} is {profile.status}")
    return await decrypt_auth_state(user_id, profile.encrypted_key, profile.encrypted_state)


async def revoke_profile(db: AsyncSession, profile_id: str, user_id: str) -> bool:
    """Delete an auth profile. Returns True if deleted."""
    profile = await get_profile(db, profile_id, user_id)
    if not profile:
        return False
    await db.delete(profile)
    await db.flush()
    return True


async def update_label(db: AsyncSession, profile_id: str, user_id: str, label: str) -> AuthProfile | None:
    """Update the label of an auth profile."""
    profile = await get_profile(db, profile_id, user_id)
    if not profile:
        return None
    profile.label = label
    await db.flush()
    return profile


async def mark_expired(db: AsyncSession, profile_id: str, user_id: str) -> None:
    """Mark a profile as expired."""
    profile = await get_profile(db, profile_id, user_id)
    if profile:
        profile.status = "expired"
        await db.flush()


async def check_auth_validity(browser_session, domain: str) -> bool:
    """Best-effort check if the loaded auth is still valid.

    Heuristic: navigate to the domain and check for signs of being logged out
    (presence of login form elements, login-related URL patterns).
    Returns True if auth appears valid or check is inconclusive.

    Note: Page.goto() in openbrowser returns None (no response object),
    so we rely on page content heuristics rather than HTTP status codes.
    """
    try:
        import asyncio as _asyncio

        page = await browser_session.get_current_page()
        if not page:
            return True

        await page.goto(f"https://{domain}")
        # Heuristic wait for page load; a CDP Page.loadEventFired listener would
        # be more reliable but is not exposed by openbrowser's page API.
        await _asyncio.sleep(3)

        login_indicators = await page.evaluate("""() => {
            const forms = document.querySelectorAll('form');
            for (const form of forms) {
                const inputs = form.querySelectorAll('input[type="password"], input[name="password"]');
                if (inputs.length > 0) return true;
            }
            const url = window.location.href.toLowerCase();
            return url.includes('/login') || url.includes('/signin') || url.includes('/auth');
        }""")

        if login_indicators:
            return False

        return True
    except Exception as e:
        logger.warning("Auth validity check failed for %s: %s -- treating as valid", domain, e)
        return True
