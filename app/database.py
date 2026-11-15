"""Database engine, sessionmaker, and initialization."""

from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from app.config import settings

# Configure database engine options with Supabase / PgBouncer / Cloud support
engine_kwargs: dict = {
    "echo": settings.DEBUG and settings.ENVIRONMENT == "development",
    "future": True,
    "pool_pre_ping": True,
}

if "postgresql" in settings.DATABASE_URL:
    # Supabase free tier connection drops idle sockets after a few minutes; recycle periodically
    engine_kwargs["pool_recycle"] = 300
    connect_args: dict = {}
    # When connecting via Supabase pooler (port 6543 / Supavisor / PgBouncer), disable prepared statement cache
    if any(k in settings.DATABASE_URL for k in ["supabase", "pooler", ":6543"]):
        connect_args["statement_cache_size"] = 0
        connect_args["prepared_statement_cache_size"] = 0
    if connect_args:
        engine_kwargs["connect_args"] = connect_args

# Create async engine
engine = create_async_engine(settings.DATABASE_URL, **engine_kwargs)

# Async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Initialize database extensions and create tables."""
    async with engine.begin() as conn:
        # Enable pg_trgm extension if using PostgreSQL (if permissions permit)
        dialect = conn.dialect.name
        if dialect == "postgresql":
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
            except Exception:
                pass
        # Create all tables registered with SQLModel metadata
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields an active async database session."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
