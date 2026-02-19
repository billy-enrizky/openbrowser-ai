"""Database initialization helpers."""

from app.db.models import Base
from app.db.session import get_engine, is_database_configured


async def init_database() -> None:
    """Create tables if database is configured."""
    if not is_database_configured():
        return

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

