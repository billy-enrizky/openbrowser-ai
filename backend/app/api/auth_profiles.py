"""Auth profile API endpoints.

POST   /api/v1/auth/profiles/start  -- Launch browser for user login
POST   /api/v1/auth/profiles/save   -- Capture + encrypt auth state
GET    /api/v1/auth/profiles        -- List profiles
GET    /api/v1/auth/profiles/{id}   -- Get single profile
DELETE /api/v1/auth/profiles/{id}   -- Revoke and delete
PATCH  /api/v1/auth/profiles/{id}   -- Update label
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthPrincipal, get_current_user
from app.db.session import get_db_session, is_database_configured
from app.models.schemas import (
    AuthProfileListResponse,
    AuthProfileResponse,
    SaveAuthProfileRequest,
    StartAuthSessionRequest,
    UpdateAuthProfileRequest,
)
from app.services import auth_profile_service
from app.services.agent_service import agent_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/profiles", tags=["auth-profiles"])


def _principal_to_identity(principal: AuthPrincipal | None) -> tuple[str, str | None, str | None]:
    if principal is None:
        return "anonymous-local-user", None, "local"
    return principal.subject, principal.email, principal.username


async def _get_user_id(principal: AuthPrincipal | None, db: AsyncSession) -> str:
    """Resolve principal to user_id (ensures user row exists)."""
    from app.services.chat_service import ChatService

    sub, email, username = _principal_to_identity(principal)
    service = ChatService(db)
    user = await service.ensure_user(cognito_sub=sub, email=email, username=username)
    return user.id


def _profile_to_response(profile) -> AuthProfileResponse:
    return AuthProfileResponse(
        id=profile.id,
        domain=profile.domain,
        label=profile.label,
        status=profile.status,
        last_verified_at=profile.last_verified_at,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


@router.post("/start")
async def start_auth_session(
    req: StartAuthSessionRequest,
    principal: AuthPrincipal | None = Depends(get_current_user),
):
    """Launch a browser session navigated to the domain login page.

    The user logs in via VNC, then calls POST /save to capture state.
    """
    from uuid import uuid4

    task_id = str(uuid4())

    # Create a minimal agent session that just opens a browser
    session = await agent_manager.create_session_with_id(
        task_id=task_id,
        task=f"Navigate to https://{req.domain} and wait for user to log in",
        agent_type="browser",
        max_steps=1,
        use_vision=False,
    )

    # Start VNC + browser but don't run the agent
    from openbrowser import BrowserSession, BrowserProfile

    session.vnc_session = await session._setup_vnc()
    await session._setup_new_browser(BrowserSession, BrowserProfile)

    # Navigate to the domain
    page = await session.browser_session.get_current_page()
    if not page:
        raise HTTPException(status_code=500, detail="Failed to get browser page")
    await page.goto(f"https://{req.domain}")

    vnc_url = session.vnc_session.websocket_url if session.vnc_session else None

    return {
        "task_id": task_id,
        "vnc_url": vnc_url,
        "domain": req.domain,
        "label": req.label,
    }


@router.post("/save", response_model=AuthProfileResponse)
async def save_auth_profile(
    req: SaveAuthProfileRequest,
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Capture cookies/localStorage from an active browser session and save."""
    if not is_database_configured():
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = await _get_user_id(principal, db)

    # Get the browser session
    session = await agent_manager.get_session(req.task_id)
    if not session or not session.browser_session:
        raise HTTPException(status_code=404, detail="No active browser session for this task")

    # Export cookies + localStorage via BrowserSession.export_storage_state()
    # Returns {"cookies": [...], "origins": [...]} -- exact format save_profile expects
    try:
        storage_state = await session.browser_session.export_storage_state()
    except Exception as e:
        logger.exception("Failed to export storage state: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to capture auth state: {e}")

    # Save profile
    try:
        profile = await auth_profile_service.save_profile(
            db=db,
            user_id=user_id,
            domain=req.domain,
            label=req.label,
            storage_state=storage_state,
        )
        await db.commit()
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))

    # Clean up browser session
    try:
        await session._cleanup()
        await agent_manager.remove_session(req.task_id)
    except Exception as e:
        logger.warning("Cleanup error after auth save: %s", e)

    return _profile_to_response(profile)


@router.get("", response_model=AuthProfileListResponse)
async def list_auth_profiles(
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """List all auth profiles for the current user."""
    if not is_database_configured():
        return AuthProfileListResponse(profiles=[])

    user_id = await _get_user_id(principal, db)
    profiles = await auth_profile_service.list_profiles(db, user_id)
    return AuthProfileListResponse(profiles=[_profile_to_response(p) for p in profiles])


@router.get("/{profile_id}", response_model=AuthProfileResponse)
async def get_auth_profile(
    profile_id: str,
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Get a single auth profile."""
    if not is_database_configured():
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = await _get_user_id(principal, db)
    profile = await auth_profile_service.get_profile(db, profile_id, user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Auth profile not found")
    return _profile_to_response(profile)


@router.delete("/{profile_id}", status_code=204)
async def delete_auth_profile(
    profile_id: str,
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Revoke and delete an auth profile."""
    if not is_database_configured():
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = await _get_user_id(principal, db)
    deleted = await auth_profile_service.revoke_profile(db, profile_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Auth profile not found")
    await db.commit()


@router.patch("/{profile_id}", response_model=AuthProfileResponse)
async def update_auth_profile(
    profile_id: str,
    req: UpdateAuthProfileRequest,
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Update the label of an auth profile."""
    if not is_database_configured():
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = await _get_user_id(principal, db)
    profile = await auth_profile_service.update_label(db, profile_id, user_id, req.label)
    if not profile:
        raise HTTPException(status_code=404, detail="Auth profile not found")
    await db.commit()
    return _profile_to_response(profile)
