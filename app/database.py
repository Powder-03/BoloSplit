"""Database engine, sessionmaker, and initialization."""

import logging
from collections.abc import AsyncGenerator

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from app.config import settings

logger = logging.getLogger(__name__)

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
    """
    Initialize database extensions, create tables, and run idempotent schema migrations.

    Each DDL / migration step runs in its own transaction so an optional step
    (e.g., pg_trgm extension permissions) never aborts subsequent PostgreSQL statements.
    """
    import app.models  # noqa: F401
    from app.services.activity_service import backfill_missing_activity_logs

    # 1. Optional PostgreSQL pg_trgm extension
    try:
        async with engine.begin() as conn:
            if conn.dialect.name == "postgresql":
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
    except Exception as exc:
        logger.warning("Skipping pg_trgm extension setup: %s", exc)

    # 2. Create all SQLModel tables (users, groups, expenses, activity_logs, etc.)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    # 3. Ensure `is_personal` column & index exist on pre-existing `expenses` tables
    async with engine.begin() as conn:
        if conn.dialect.name == "postgresql":
            await conn.execute(
                text("ALTER TABLE expenses ADD COLUMN IF NOT EXISTS is_personal BOOLEAN NOT NULL DEFAULT FALSE;")
            )
        else:
            cols = {
                row[1]
                for row in (await conn.execute(text("PRAGMA table_info(expenses);"))).all()
            }
            if "is_personal" not in cols:
                await conn.execute(
                    text("ALTER TABLE expenses ADD COLUMN is_personal BOOLEAN NOT NULL DEFAULT FALSE;")
                )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_expenses_is_personal ON expenses (is_personal);")
        )

    # 4. Idempotent migration from legacy `self_expenses` table if it still exists
    async with engine.begin() as conn:
        has_legacy_self = await conn.run_sync(
            lambda sync_conn: sa_inspect(sync_conn).has_table("self_expenses")
        )
        if has_legacy_self:
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

    # 5. Idempotent backfill for pre-existing shared expenses & settlements
    async with async_session_maker() as session:
        await backfill_missing_activity_logs(session)
        await session.commit()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields an active async database session."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
