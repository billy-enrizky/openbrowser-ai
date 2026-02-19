"""API routes for persisted chat conversations/messages."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthPrincipal, get_current_user
from app.db.session import get_db_session, is_database_configured
from app.models.schemas import (
    ChatConversation,
    ChatConversationResponse,
    ChatListResponse,
    ChatMessage,
    CreateChatRequest,
    RenameChatRequest,
    SetActiveChatRequest,
)
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chats", tags=["chats"])


def _principal_to_identity(principal: AuthPrincipal | None) -> tuple[str, str | None, str | None]:
    """Map principal to stable identity. Supports AUTH_ENABLED=false local mode."""
    if principal is None:
        return "anonymous-local-user", None, "local"
    return principal.subject, principal.email, principal.username


def _conversation_to_response(conversation) -> ChatConversation:
    return ChatConversation(
        id=conversation.id,
        title=conversation.title,
        status=conversation.status,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        last_message_at=conversation.last_message_at,
    )


def _message_to_response(message) -> ChatMessage:
    return ChatMessage(
        id=message.id,
        conversation_id=message.conversation_id,
        role=message.role,
        content=message.content,
        task_id=message.task_id,
        metadata=message.metadata_json or {},
        created_at=message.created_at,
    )


def _ensure_db_available() -> None:
    if not is_database_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat persistence is unavailable: DATABASE_URL is not configured",
        )


@router.get("", response_model=ChatListResponse)
async def list_chats(
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """List conversations for the authenticated user."""
    _ensure_db_available()
    sub, email, username = _principal_to_identity(principal)
    service = ChatService(db)
    user = await service.ensure_user(cognito_sub=sub, email=email, username=username)
    conversations = await service.list_conversations(user=user)
    state = await service.get_user_state(user=user)
    await db.commit()

    return ChatListResponse(
        conversations=[_conversation_to_response(c) for c in conversations],
        active_conversation_id=state.active_conversation_id if state else None,
    )


@router.post("", response_model=ChatConversation, status_code=status.HTTP_201_CREATED)
async def create_chat(
    request: CreateChatRequest,
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Create a new conversation."""
    _ensure_db_available()
    sub, email, username = _principal_to_identity(principal)
    service = ChatService(db)
    user = await service.ensure_user(cognito_sub=sub, email=email, username=username)
    title = request.title or "New chat"
    conversation = await service.create_conversation(user=user, title=title)
    await db.commit()
    return _conversation_to_response(conversation)


@router.get("/{conversation_id}", response_model=ChatConversationResponse)
async def get_chat(
    conversation_id: str,
    limit: int = Query(default=500, ge=1, le=2000),
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Get a conversation and its messages."""
    _ensure_db_available()
    sub, email, username = _principal_to_identity(principal)
    service = ChatService(db)
    user = await service.ensure_user(cognito_sub=sub, email=email, username=username)
    conversation = await service.get_conversation(user=user, conversation_id=conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    messages = await service.get_messages(user=user, conversation_id=conversation_id, limit=limit)
    await db.commit()
    return ChatConversationResponse(
        conversation=_conversation_to_response(conversation),
        messages=[_message_to_response(m) for m in messages],
    )


@router.patch("/{conversation_id}", response_model=ChatConversation)
async def rename_chat(
    conversation_id: str,
    request: RenameChatRequest,
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Rename a conversation."""
    _ensure_db_available()
    sub, email, username = _principal_to_identity(principal)
    service = ChatService(db)
    user = await service.ensure_user(cognito_sub=sub, email=email, username=username)
    conversation = await service.rename_conversation(user=user, conversation_id=conversation_id, title=request.title)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    await db.commit()
    return _conversation_to_response(conversation)


@router.delete("/{conversation_id}", response_model=ChatConversation)
async def archive_chat(
    conversation_id: str,
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Archive a conversation."""
    _ensure_db_available()
    sub, email, username = _principal_to_identity(principal)
    service = ChatService(db)
    user = await service.ensure_user(cognito_sub=sub, email=email, username=username)
    conversation = await service.archive_conversation(user=user, conversation_id=conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    await db.commit()
    return _conversation_to_response(conversation)


@router.post("/active")
async def set_active_chat(
    request: SetActiveChatRequest,
    principal: AuthPrincipal | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Set active conversation for current user."""
    _ensure_db_available()
    sub, email, username = _principal_to_identity(principal)
    service = ChatService(db)
    user = await service.ensure_user(cognito_sub=sub, email=email, username=username)
    if request.conversation_id:
        conversation = await service.get_conversation(user=user, conversation_id=request.conversation_id)
        if conversation is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    state = await service.set_active_conversation(user=user, conversation_id=request.conversation_id)
    await db.commit()
    return {"active_conversation_id": state.active_conversation_id}

