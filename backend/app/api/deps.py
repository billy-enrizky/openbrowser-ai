"""Shared API dependency helpers."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthPrincipal


def principal_to_identity(principal: AuthPrincipal | None) -> tuple[str, str | None, str | None]:
    """Extract (subject, email, username) from an auth principal."""
    if principal is None:
        return "anonymous-local-user", None, "local"
    return principal.subject, principal.email, principal.username


async def resolve_user_id(principal: AuthPrincipal | None, db: AsyncSession) -> str:
    """Resolve principal to user_id (ensures user row exists)."""
    from app.services.chat_service import ChatService

    sub, email, username = principal_to_identity(principal)
    service = ChatService(db)
    user = await service.ensure_user(cognito_sub=sub, email=email, username=username)
    return user.id
