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
    """Initialize database extensions, create tables, and migrate legacy solo expenses."""
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

        # Idempotent one-time migration: move legacy solo rows from `expenses` -> `self_expenses`
        await conn.execute(
            text(
                """
                INSERT INTO self_expenses (
                    id, user_id, description, amount, currency, category,
                    expense_date, is_deleted, created_at, updated_at
                )
                SELECT
                    e.id, e.payer_id, e.description, e.amount, e.currency, e.category,
                    e.expense_date, e.is_deleted, e.created_at, e.updated_at
                FROM expenses e
                WHERE e.is_deleted = FALSE
                  AND e.group_id IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM expense_splits s
                      WHERE s.expense_id = e.id AND s.user_id != e.payer_id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM self_expenses se WHERE se.id = e.id
                  )
                """
            )
        )
        await conn.execute(
            text(
                """
                UPDATE expenses
                SET is_deleted = TRUE
                WHERE is_deleted = FALSE
                  AND id IN (SELECT id FROM self_expenses)
                """
            )
        )


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields an active async database session."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
