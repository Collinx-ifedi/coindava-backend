"""
Coindava Waitlist Backend
-------------------------

Production-grade database infrastructure.

Responsibilities:
- Create and configure the SQLAlchemy engine.
- Configure PostgreSQL connection pooling.
- Support local SQLite development/testing.
- Provide request-scoped SQLAlchemy sessions.
- Provide explicit transaction management.
- Verify database connectivity.
- Dispose database resources during shutdown.
- Provide safe database diagnostics.

This module contains database infrastructure only.

Do NOT place:
- Business logic
- API handlers
- Authentication logic
- Point calculations
- Referral logic
- Social verification
- ORM model definitions

in this module.

ORM models belong in models.py.
Application configuration belongs in config.py.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import (
    get_database_url,
    is_production,
    settings,
)


# =============================================================================
# DATABASE URL
# =============================================================================

DATABASE_URL: str = get_database_url().strip()


# =============================================================================
# SQLALCHEMY BASE
# =============================================================================


class Base(DeclarativeBase):
    """
    Base class for all Coindava SQLAlchemy ORM models.

    Models in models.py must inherit from this class.
    """

    pass


# =============================================================================
# DATABASE URL HELPERS
# =============================================================================


def _is_sqlite_url(database_url: str) -> bool:
    """
    Return True when the configured database URL uses SQLite.
    """

    return database_url.startswith("sqlite")


def _is_postgresql_url(database_url: str) -> bool:
    """
    Return True when the configured database URL uses PostgreSQL.
    """

    return database_url.startswith(
        (
            "postgresql://",
            "postgresql+psycopg://",
            "postgresql+psycopg2://",
            "postgres://",
        )
    )


def _validate_database_url(database_url: str) -> None:
    """
    Validate the configured database URL before creating the engine.
    """

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is empty. Configure DATABASE_URL before "
            "starting the Coindava backend."
        )

    if _is_sqlite_url(database_url):
        if is_production():
            raise RuntimeError(
                "SQLite cannot be used in production. "
                "Configure DATABASE_URL with PostgreSQL."
            )

        return

    if not _is_postgresql_url(database_url):
        raise RuntimeError(
            "Unsupported database backend. "
            "Coindava currently supports PostgreSQL and SQLite."
        )


# =============================================================================
# ENGINE CONFIGURATION
# =============================================================================


def _build_engine() -> Engine:
    """
    Build the SQLAlchemy database engine.

    PostgreSQL:
        Uses connection pooling and production-oriented settings.

    SQLite:
        Intended only for local development/testing.

    Returns:
        Configured SQLAlchemy Engine.
    """

    database_url = DATABASE_URL

    _validate_database_url(database_url)

    # -------------------------------------------------------------------------
    # SQLite
    # -------------------------------------------------------------------------

    if _is_sqlite_url(database_url):
        return create_engine(
            database_url,
            connect_args={
                "check_same_thread": False,
            },
            pool_pre_ping=True,
            echo=settings.database_echo,
        )

    # -------------------------------------------------------------------------
    # PostgreSQL
    # -------------------------------------------------------------------------

    if _is_postgresql_url(database_url):
        return create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=settings.database_pool_recycle,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_timeout=settings.database_pool_timeout,
            echo=settings.database_echo,
        )

    # This should be unreachable because _validate_database_url()
    # already rejects unsupported URLs.
    raise RuntimeError(
        "Unable to configure the database engine."
    )


engine: Engine = _build_engine()


# =============================================================================
# POSTGRESQL CONNECTION INITIALIZATION
# =============================================================================


@event.listens_for(engine, "connect")
def _configure_database_connection(
    dbapi_connection: Any,
    connection_record: Any,
) -> None:
    """
    Configure newly-created database connections.

    PostgreSQL connections:
        - Set timezone to UTC.

    SQLite:
        - No special connection configuration is required here.

    All application timestamps should ultimately be treated consistently
    as UTC at the database/application boundary.
    """

    del connection_record

    if not _is_postgresql_url(DATABASE_URL):
        return

    cursor = dbapi_connection.cursor()

    try:
        cursor.execute("SET TIME ZONE 'UTC'")
    finally:
        cursor.close()

    # SET is transactional. Without this commit, the first pool "reset"
    # (a ROLLBACK) would silently undo the timezone change.
    dbapi_connection.commit()


# =============================================================================
# SESSION FACTORY
# =============================================================================


SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    expire_on_commit=False,
)


# =============================================================================
# REQUEST-SCOPED DATABASE SESSION
# =============================================================================


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI database dependency.

    Example:

        @router.get("/example")
        def example(
            db: Session = Depends(get_db),
        ):
            ...

    The session is always closed after the request.

    Mutation endpoints should explicitly commit their successful
    transactions.

    If an exception escapes the request handler, the current transaction
    is rolled back automatically.
    """

    db = SessionLocal()

    try:
        yield db

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


# =============================================================================
# DATABASE HEALTH CHECK
# =============================================================================


def check_database_connection() -> bool:
    """
    Verify that the application can communicate with the database.

    Returns:
        True when the database responds successfully.
        False otherwise.

    No credentials or connection strings are exposed.
    """

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return True

    except SQLAlchemyError:
        return False


# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================


def initialize_database() -> None:
    """
    Create database tables for development/testing environments.

    Production schema management MUST use Alembic migrations.

    Production:

        alembic upgrade head

    Development/testing:

        initialize_database()
    """

    if is_production() and not settings.auto_create_tables:
        raise RuntimeError(
            "initialize_database() cannot be used in production unless "
            "AUTO_CREATE_TABLES=true. Run Alembic migrations instead."
        )

    # Importing models registers every table on Base.metadata. Without this
    # import create_all() would silently create nothing.
    import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


# =============================================================================
# DATABASE SHUTDOWN
# =============================================================================


def close_database() -> None:
    """
    Dispose of all SQLAlchemy connection pools.

    This should be called during application shutdown.
    """

    engine.dispose()


# =============================================================================
# DATABASE LIFESPAN
# =============================================================================


@asynccontextmanager
async def database_lifespan():
    """
    FastAPI lifespan helper for database startup/shutdown.

    Startup:
        Verifies database connectivity.

    Shutdown:
        Disposes the SQLAlchemy engine.

    Example:

        app = FastAPI(lifespan=database_lifespan)
    """

    if not check_database_connection():
        raise RuntimeError(
            "Unable to establish a connection to the Coindava database."
        )

    try:
        yield

    finally:
        close_database()


# =============================================================================
# EXPLICIT TRANSACTION CONTEXT
# =============================================================================


class DatabaseTransaction:
    """
    Explicit database transaction context manager.

    Example:

        with DatabaseTransaction() as db:
            db.add(user)

    Successful execution:
        COMMIT

    Exception:
        ROLLBACK
    """

    def __init__(self) -> None:
        self.db: Session | None = None

    def __enter__(self) -> Session:
        self.db = SessionLocal()

        return self.db

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool:
        if self.db is None:
            return False

        try:
            if exc_type is not None:
                self.db.rollback()
            else:
                self.db.commit()

        except Exception:
            self.db.rollback()
            raise

        finally:
            self.db.close()

        return False


# =============================================================================
# TRANSACTION ROLLBACK HELPER
# =============================================================================


def rollback_session(db: Session) -> None:
    """
    Safely rollback a SQLAlchemy session.

    This helper is intentionally small and infrastructure-only.

    It is useful when service code needs to explicitly recover a session
    after a failed database operation.
    """

    try:
        db.rollback()
    except SQLAlchemyError:
        # The caller owns the session lifecycle.
        # Do not mask the original application/database exception.
        pass


# =============================================================================
# DATABASE INFORMATION
# =============================================================================


def get_database_info() -> dict[str, Any]:
    """
    Return safe database diagnostics.

    Sensitive information such as:
        - username
        - password
        - hostname
        - database name
        - connection string

    is intentionally excluded.
    """

    url: URL = engine.url
    pool = engine.pool

    pool_size = None
    checked_out = None
    overflow = None
    checked_in = None

    # QueuePool exposes these methods.
    if hasattr(pool, "size"):
        pool_size = pool.size()

    if hasattr(pool, "checkedout"):
        checked_out = pool.checkedout()

    if hasattr(pool, "overflow"):
        overflow = pool.overflow()

    if hasattr(pool, "checkedin"):
        checked_in = pool.checkedin()

    return {
        "dialect": url.get_backend_name(),
        "driver": url.get_driver_name(),
        "pool_size": pool_size,
        "checked_out_connections": checked_out,
        "checked_in_connections": checked_in,
        "overflow": overflow,
    }


# =============================================================================
# DATABASE PING
# =============================================================================


def ping_database() -> None:
    """
    Perform a database connectivity check.

    Raises:
        RuntimeError:
            When the database cannot be reached.

    This function is useful during application startup when failure should
    prevent the service from becoming ready.
    """

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    except SQLAlchemyError as exc:
        raise RuntimeError(
            "Coindava database connectivity check failed."
        ) from exc


# =============================================================================
# PUBLIC EXPORTS
# =============================================================================


__all__ = [
    "Base",
    "DATABASE_URL",
    "engine",
    "SessionLocal",
    "get_db",
    "check_database_connection",
    "initialize_database",
    "close_database",
    "database_lifespan",
    "DatabaseTransaction",
    "rollback_session",
    "get_database_info",
    "ping_database",
]