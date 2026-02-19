"""Chat persistence service backed by PostgreSQL."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, ConversationMessage, User, UserState


def utc_now() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def conversation_title_from_prompt(prompt: str) -> str:
    """Generate a compact title from the first user prompt."""
    trimmed = " ".join(prompt.split())
    return trimmed[:80] if len(trimmed) > 80 else trimmed


class ChatService:
    """Service layer for user chats and messages."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def ensure_user(self, *, cognito_sub: str, email: str | None, username: str | None) -> User:
        """Get or create user record by Cognito subject."""
        stmt: Select[tuple[User]] = select(User).where(User.cognito_sub == cognito_sub)
        user = (await self.db.execute(stmt)).scalar_one_or_none()
        if user is None:
            user = User(
                id=str(uuid4()),
                cognito_sub=cognito_sub,
                email=email,
                username=username,
            )
            self.db.add(user)
            await self.db.flush()
        else:
            changed = False
            if email and user.email != email:
                user.email = email
                changed = True
            if username and user.username != username:
                user.username = username
                changed = True
            if changed:
                user.updated_at = utc_now()
                await self.db.flush()

        return user

    async def list_conversations(self, *, user: User, limit: int = 100) -> list[Conversation]:
        """List conversations for a user ordered by recency."""
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user.id)
            .where(Conversation.status == "active")
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
        )
        return list((await self.db.execute(stmt)).scalars().all())

    async def create_conversation(self, *, user: User, title: str) -> Conversation:
        """Create a new conversation."""
        now = utc_now()
        conversation = Conversation(
            id=str(uuid4()),
            user_id=user.id,
            title=title[:200] if title else "New chat",
            status="active",
            created_at=now,
            updated_at=now,
            last_message_at=None,
        )
        self.db.add(conversation)
        await self.db.flush()
        return conversation

    async def get_conversation(self, *, user: User, conversation_id: str) -> Conversation | None:
        """Fetch a conversation if it belongs to the user."""
        stmt = select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user.id)
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def get_messages(
        self,
        *,
        user: User,
        conversation_id: str,
        limit: int = 500,
        before: datetime | None = None,
    ) -> list[ConversationMessage]:
        """Get messages for a conversation in ascending time order."""
        stmt = (
            select(ConversationMessage)
            .where(
                ConversationMessage.user_id == user.id,
                ConversationMessage.conversation_id == conversation_id,
            )
            .order_by(ConversationMessage.created_at.asc())
            .limit(limit)
        )
        if before is not None:
            stmt = stmt.where(ConversationMessage.created_at < before)

        return list((await self.db.execute(stmt)).scalars().all())

    async def append_message(
        self,
        *,
        user: User,
        conversation: Conversation,
        role: str,
        content: str,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        """Append a message to a conversation and bump conversation timestamps."""
        now = utc_now()
        msg = ConversationMessage(
            id=str(uuid4()),
            conversation_id=conversation.id,
            user_id=user.id,
            role=role,
            content=content,
            task_id=task_id,
            metadata_json=metadata or {},
            created_at=now,
        )
        self.db.add(msg)
        conversation.updated_at = now
        conversation.last_message_at = now
        await self.db.flush()
        return msg

    async def set_active_conversation(self, *, user: User, conversation_id: str | None) -> UserState:
        """Update active conversation pointer for a user."""
        stmt = select(UserState).where(UserState.user_id == user.id)
        state = (await self.db.execute(stmt)).scalar_one_or_none()
        if state is None:
            state = UserState(
                user_id=user.id,
                active_conversation_id=conversation_id,
                preferences={},
            )
            self.db.add(state)
        else:
            state.active_conversation_id = conversation_id
            state.updated_at = utc_now()

        await self.db.flush()
        return state

    async def get_user_state(self, *, user: User) -> UserState | None:
        """Get user state row."""
        stmt = select(UserState).where(UserState.user_id == user.id)
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def rename_conversation(self, *, user: User, conversation_id: str, title: str) -> Conversation | None:
        """Rename conversation owned by user."""
        conversation = await self.get_conversation(user=user, conversation_id=conversation_id)
        if conversation is None:
            return None
        conversation.title = title[:200] if title else conversation.title
        conversation.updated_at = utc_now()
        await self.db.flush()
        return conversation

    async def archive_conversation(self, *, user: User, conversation_id: str) -> Conversation | None:
        """Archive a conversation."""
        conversation = await self.get_conversation(user=user, conversation_id=conversation_id)
        if conversation is None:
            return None
        conversation.status = "archived"
        conversation.updated_at = utc_now()
        await self.db.flush()
        return conversation

