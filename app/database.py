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
    engine_kwargs["pool_recycle"] = 300
    connect_args: dict = {}
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
    """Initialize database extensions, create tables, and ensure unified expense schema."""
    async with engine.begin() as conn:
        dialect = conn.dialect.name
        if dialect == "postgresql":
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
            except Exception:
                pass

        # Create all tables registered with SQLModel metadata
        await conn.run_sync(SQLModel.metadata.create_all)

        # Ensure `is_personal` column exists on pre-existing `expenses` tables
        try:
            await conn.execute(
                text("ALTER TABLE expenses ADD COLUMN is_personal BOOLEAN NOT NULL DEFAULT FALSE;")
            )
        except Exception:
            pass

        # Idempotent migration from legacy `self_expenses` table if it exists in an older DB
        try:
            await conn.execute(
                text(
                    """
                    INSERT INTO expenses (
                        id, group_id, description, amount, currency, payer_id,
                        category, expense_date, is_personal, is_deleted, created_at, updated_at
                    )
                    SELECT
                        se.id, NULL, se.description, se.amount, se.currency, se.user_id,
                        se.category, se.expense_date, TRUE, se.is_deleted, se.created_at, se.updated_at
                    FROM self_expenses se
                    WHERE NOT EXISTS (SELECT 1 FROM expenses e WHERE e.id = se.id)
                    """
                )
            )
        except Exception:
            pass


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields an active async database session."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
