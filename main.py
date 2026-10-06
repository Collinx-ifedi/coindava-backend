"""
Coindava Waitlist Backend
main.py

Production FastAPI application entry point.

Responsibilities:
- Create the FastAPI application
- Configure middleware
- Configure CORS
- Register application routers
- Configure startup/shutdown lifecycle
- Provide health/readiness endpoints
- Provide API metadata
- Handle unexpected application errors
- Keep application wiring separate from business logic

Business logic belongs in:
    auth.py
    users.py
    referrals.py
    points.py
    social.py
    leaderboard.py

Infrastructure belongs in:
    config.py
    database.py
    models.py

IMPORTANT:
- No business logic should be implemented here.
- No database models should be defined here.
- No point/referral/leaderboard calculations should be performed here.
- main.py is application composition and infrastructure wiring only.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import (
    FastAPI,
    Request,
    Response,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from auth import router as auth_router
from config import settings
from database import engine, initialize_database
from leaderboard import router as leaderboard_router
from points import router as points_router
from referrals import router as referrals_router
from social import router as social_router
from users import router as users_router


# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = getattr(
    logging,
    str(
        getattr(
            settings,
            "log_level",
            "INFO",
        )
    ).upper(),
    logging.INFO,
)

logging.basicConfig(
    level=LOG_LEVEL,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "coindava.waitlist"
)


# ============================================================================
# APPLICATION METADATA
# ============================================================================

APP_NAME = str(
    getattr(
        settings,
        "app_name",
        "Coindava Waitlist API",
    )
)

APP_VERSION = str(
    getattr(
        settings,
        "app_version",
        "1.0.0",
    )
)

API_PREFIX = str(
    getattr(
        settings,
        "api_v1_prefix",
        "/api/v1",
    )
)

ENVIRONMENT = str(
    getattr(
        settings,
        "app_env",
        "production",
    )
).lower()

DEBUG = bool(
    getattr(
        settings,
        "debug",
        False,
    )
)

APP_DESCRIPTION = """
Coindava Waitlist API.

Provides authentication, waitlist accounts, referrals,
points, social campaign verification and leaderboard services.

This API does not custody cryptocurrency and does not process
card transactions.
"""


# ============================================================================
# SECURITY CONSTANTS
# ============================================================================

MIN_SECRET_KEY_LENGTH = 32

REQUEST_ID_HEADER = "X-Request-ID"
RESPONSE_TIME_HEADER = "X-Response-Time-Ms"

DEFAULT_PORT = 8000


# ============================================================================
# REQUEST HELPERS
# ============================================================================


def _generate_request_id() -> str:
    """
    Generate a cryptographically random request correlation ID.
    """

    return str(uuid.uuid4())


def _normalize_origin(origin: str) -> str:
    """
    Normalize a configured CORS origin.

    Trailing slashes are removed so that:
        https://coindava.com/
    and:
        https://coindava.com

    are treated consistently.
    """

    return origin.strip().rstrip("/")


def _get_allowed_origins() -> list[str]:
    """
    Read configured CORS origins.

    Primary source:
        settings.cors_origins

    Environment fallback:
        CORS_ORIGINS="https://coindava.com,https://www.coindava.com"

    The production configuration should explicitly define origins.
    """

    configured = getattr(
        settings,
        "cors_origins",
        None,
    )

    if configured:

        if isinstance(
            configured,
            str,
        ):
            origins = configured.split(",")

        else:
            origins = list(configured)

        normalized = [
            _normalize_origin(
                str(origin)
            )
            for origin in origins
            if str(origin).strip()
        ]

        return list(
            dict.fromkeys(normalized)
        )

    environment_value = os.getenv(
        "CORS_ORIGINS",
        "",
    )

    if environment_value:

        origins = [
            _normalize_origin(origin)
            for origin in environment_value.split(",")
            if origin.strip()
        ]

        return list(
            dict.fromkeys(origins)
        )

    if ENVIRONMENT == "production":

        logger.warning(
            "No explicit CORS origins are configured. "
            "Browser requests will be blocked by CORS."
        )

    return []


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================


def _validate_runtime_configuration() -> None:
    """
    Validate essential runtime configuration.

    config.py performs the authoritative environment-specific validation.
    This function exists as an additional application-level safety check.

    It intentionally does not attempt to validate every configuration field.
    """

    # settings.secret_key is a pydantic SecretStr; read the real value
    # explicitly instead of relying on SecretStr's __len__/truthiness.
    secret_holder = getattr(
        settings,
        "secret_key",
        None,
    )

    secret_key = (
        secret_holder.get_secret_value()
        if secret_holder is not None
        else ""
    )

    if not secret_key:
        raise RuntimeError(
            "SECRET_KEY is not configured."
        )

    if len(secret_key) < MIN_SECRET_KEY_LENGTH:
        raise RuntimeError(
            "SECRET_KEY must contain at least "
            f"{MIN_SECRET_KEY_LENGTH} characters."
        )

    database_url = getattr(
        settings,
        "database_url",
        None,
    )

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured."
        )

    # Production must not run against local SQLite.
    if ENVIRONMENT == "production":

        normalized_database_url = str(
            database_url
        ).lower()

        if normalized_database_url.startswith(
            "sqlite"
        ):
            raise RuntimeError(
                "SQLite cannot be used in production. "
                "Configure PostgreSQL."
            )

        if DEBUG:
            raise RuntimeError(
                "DEBUG must be disabled in production."
            )

    # Production frontend should use HTTPS.
    frontend_url = getattr(
        settings,
        "frontend_url",
        None,
    )

    if (
        ENVIRONMENT == "production"
        and frontend_url
    ):

        normalized_frontend_url = str(
            frontend_url
        ).strip().lower()

        if not normalized_frontend_url.startswith(
            "https://"
        ):
            raise RuntimeError(
                "FRONTEND_URL must use HTTPS in production."
            )

    logger.info(
        "Runtime configuration validation passed."
    )


# ============================================================================
# DATABASE HEALTH
# ============================================================================


def check_database_connection() -> bool:
    """
    Perform a lightweight database connectivity check.

    Returns:
        True  -> database is reachable
        False -> database is unavailable

    This function does not initialize tables and does not perform migrations.
    """

    try:

        with engine.connect() as connection:

            connection.execute(
                text("SELECT 1")
            )

        return True

    except SQLAlchemyError:

        logger.exception(
            "Database connectivity check failed."
        )

        return False

    except Exception:

        logger.exception(
            "Unexpected database health-check failure."
        )

        return False


# ============================================================================
# X CONFIGURATION STATUS
# ============================================================================


def _log_social_configuration() -> None:
    """
    Log whether the X integration appears configured.

    Secrets are never logged.
    """

    x_enabled = bool(
        getattr(
            settings,
            "x_enabled",
            False,
        )
    )

    x_client_id = getattr(
        settings,
        "x_client_id",
        None,
    )

    x_client_secret = getattr(
        settings,
        "x_client_secret",
        None,
    )

    x_redirect_uri = getattr(
        settings,
        "x_redirect_uri",
        None,
    )

    if not x_enabled:

        logger.info(
            "X social integration is disabled."
        )

        return

    missing: list[str] = []

    if not x_client_id:
        missing.append(
            "X_CLIENT_ID"
        )

    if not x_client_secret:
        missing.append(
            "X_CLIENT_SECRET"
        )

    if not x_redirect_uri:
        missing.append(
            "X_REDIRECT_URI"
        )

    if missing:

        logger.warning(
            "X social integration is enabled but "
            "not completely configured. Missing: %s",
            ", ".join(missing),
        )

        return

    logger.info(
        "X OAuth configuration detected."
    )


# ============================================================================
# APPLICATION LIFESPAN
# ============================================================================


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Application startup/shutdown lifecycle.

    Startup:
        1. Validate runtime configuration.
        2. Verify database connectivity.
        3. Log enabled integrations.
        4. Mark application startup complete.

    Shutdown:
        1. Dispose SQLAlchemy connection pool.
    """

    logger.info(
        "Starting %s v%s",
        APP_NAME,
        APP_VERSION,
    )

    logger.info(
        "Environment: %s",
        ENVIRONMENT,
    )

    # ------------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------------

    _validate_runtime_configuration()

    # ------------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------------

    if not check_database_connection():

        raise RuntimeError(
            "Database is unavailable. "
            "Application startup aborted."
        )

    logger.info(
        "Database connection verified."
    )

    # ------------------------------------------------------------------------
    # Schema
    #
    # Development/testing/staging always create missing tables.
    # Production only does so with AUTO_CREATE_TABLES=true (first deploy,
    # before Alembic migrations exist). create_all() never alters or drops
    # existing tables.
    # ------------------------------------------------------------------------

    if settings.auto_create_tables or ENVIRONMENT != "production":

        initialize_database()

        logger.info(
            "Database tables verified/created."
        )

    # ------------------------------------------------------------------------
    # Integrations
    # ------------------------------------------------------------------------

    _log_social_configuration()

    # ------------------------------------------------------------------------
    # Startup complete
    # ------------------------------------------------------------------------

    logger.info(
        "Coindava Waitlist API startup complete."
    )

    try:

        yield

    finally:

        # --------------------------------------------------------------------
        # Shutdown
        # --------------------------------------------------------------------

        logger.info(
            "Shutting down Coindava Waitlist API."
        )

        try:

            engine.dispose()

            logger.info(
                "Database connection pool disposed."
            )

        except Exception:

            logger.exception(
                "Error disposing database engine."
            )

        logger.info(
            "Coindava Waitlist API shutdown complete."
        )


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================


# Docs remain useful during development/staging.
# In production they can be disabled through configuration if desired.

DOCS_ENABLED = bool(
    getattr(
        settings,
        "docs_enabled",
        ENVIRONMENT != "production",
    )
)

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    docs_url="/docs" if DOCS_ENABLED else None,
    redoc_url="/redoc" if DOCS_ENABLED else None,
    openapi_url=(
        "/openapi.json"
        if DOCS_ENABLED
        else None
    ),
    lifespan=lifespan,
)


# ============================================================================
# CORS
# ============================================================================


allowed_origins = _get_allowed_origins()

if allowed_origins:

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=[
            "GET",
            "POST",
            "PATCH",
            "PUT",
            "DELETE",
            "OPTIONS",
        ],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Accept",
            "Origin",
            REQUEST_ID_HEADER,
        ],
        expose_headers=[
            REQUEST_ID_HEADER,
            RESPONSE_TIME_HEADER,
        ],
        max_age=600,
    )

    logger.info(
        "CORS configured for %d origin(s).",
        len(allowed_origins),
    )

else:

    logger.warning(
        "No explicit CORS origins configured."
    )


# ============================================================================
# REQUEST ID / RESPONSE TIME MIDDLEWARE
# ============================================================================


@app.middleware("http")
async def request_context_middleware(
    request: Request,
    call_next,
):
    """
    Attach a request correlation ID and measure request duration.

    The request ID can be supplied by trusted upstream infrastructure
    or generated by the API.
    """

    request_id = request.headers.get(
        REQUEST_ID_HEADER
    )

    if not request_id:

        request_id = _generate_request_id()

    request.state.request_id = request_id

    started_at = time.perf_counter()

    try:

        response = await call_next(
            request
        )

    except Exception:

        duration_ms = (
            time.perf_counter()
            - started_at
        ) * 1000

        logger.exception(
            "Unhandled request error | "
            "request_id=%s | "
            "method=%s | "
            "path=%s | "
            "duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )

        raise

    duration_ms = (
        time.perf_counter()
        - started_at
    ) * 1000

    response.headers[
        REQUEST_ID_HEADER
    ] = request_id

    response.headers[
        RESPONSE_TIME_HEADER
    ] = f"{duration_ms:.2f}"

    logger.info(
        "HTTP %s %s -> %s | "
        "request_id=%s | "
        "duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        request_id,
        duration_ms,
    )

    return response


# ============================================================================
# SECURITY HEADERS
# ============================================================================


@app.middleware("http")
async def security_headers_middleware(
    request: Request,
    call_next,
):
    """
    Add baseline HTTP security headers.

    The API does not impose a restrictive Content-Security-Policy because
    CSP is primarily relevant to browser-rendered HTML applications and
    should be configured by the frontend/web layer.
    """

    response = await call_next(
        request
    )

    # Prevent MIME sniffing.
    response.headers[
        "X-Content-Type-Options"
    ] = "nosniff"

    # Prevent framing/clickjacking.
    response.headers[
        "X-Frame-Options"
    ] = "DENY"

    # Reduce referrer information.
    response.headers[
        "Referrer-Policy"
    ] = "strict-origin-when-cross-origin"

    # Disable unnecessary browser capabilities.
    response.headers[
        "Permissions-Policy"
    ] = (
        "camera=(), "
        "microphone=(), "
        "geolocation=()"
    )

    # Prevent browsers/proxies from caching API responses.
    #
    # This is particularly important for:
    # - authenticated user data
    # - leaderboard positions
    # - referral information
    # - session-related responses
    if (
        request.url.path.startswith(
            API_PREFIX
        )
        and request.method != "OPTIONS"
    ):

        response.headers[
            "Cache-Control"
        ] = "no-store"

        response.headers[
            "Pragma"
        ] = "no-cache"

    return response


# ============================================================================
# ROUTER REGISTRATION
# ============================================================================


app.include_router(
    auth_router,
    prefix=API_PREFIX,
)

app.include_router(
    users_router,
    prefix=API_PREFIX,
)

app.include_router(
    referrals_router,
    prefix=API_PREFIX,
)

app.include_router(
    points_router,
    prefix=API_PREFIX,
)

app.include_router(
    social_router,
    prefix=API_PREFIX,
)

app.include_router(
    leaderboard_router,
    prefix=API_PREFIX,
)


# ============================================================================
# ROOT ENDPOINT
# ============================================================================


@app.get(
    "/",
    include_in_schema=False,
)
async def root() -> dict[str, Any]:
    """
    Basic API information.

    Does not expose secrets or infrastructure details.
    """

    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "environment": ENVIRONMENT,
        "api_prefix": API_PREFIX,
    }


# ============================================================================
# LIVENESS ENDPOINT
# ============================================================================


@app.get(
    "/health",
    tags=["System"],
)
async def health() -> dict[str, Any]:
    """
    Kubernetes/Render-style liveness endpoint.

    Liveness answers:

        "Is the API process alive?"

    It deliberately does NOT perform a database query.

    A temporary database outage should not make the application process
    appear dead.
    """

    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
    }


# ============================================================================
# READINESS ENDPOINT
# ============================================================================


@app.get(
    "/ready",
    tags=["System"],
)
async def readiness(
    response: Response,
) -> dict[str, Any]:
    """
    Readiness endpoint.

    Readiness answers:

        "Can this API currently serve requests?"

    Database connectivity is checked here because PostgreSQL is required
    for the waitlist application.
    """

    database_available = (
        check_database_connection()
    )

    if not database_available:

        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

        return {
            "status": "not_ready",
            "service": APP_NAME,
            "version": APP_VERSION,
            "database": "unavailable",
        }

    return {
        "status": "ready",
        "service": APP_NAME,
        "version": APP_VERSION,
        "database": "available",
    }


# ============================================================================
# API SYSTEM INFORMATION
# ============================================================================


@app.get(
    f"{API_PREFIX}/system",
    tags=["System"],
)
async def system_information() -> dict[str, Any]:
    """
    Return public API metadata.

    Never expose:
    - SECRET_KEY
    - database URL
    - X client secret
    - encryption keys
    - session tokens
    - OAuth tokens
    - password hashes
    """

    x_enabled = bool(
        getattr(
            settings,
            "x_enabled",
            False,
        )
    )

    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "environment": ENVIRONMENT,
        "api_prefix": API_PREFIX,
        "features": {
            "authentication": True,
            "waitlist": True,
            "referrals": True,
            "points": True,
            "leaderboard": True,
            "social_verification": x_enabled,
        },
    }


# ============================================================================
# VALIDATION ERROR HANDLER
# ============================================================================


@app.exception_handler(
    RequestValidationError
)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """
    Standardize FastAPI request-validation errors.

    Validation details are returned because they describe the client's
    invalid input rather than server internals.
    """

    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    errors: list[dict[str, Any]] = []

    for error in exc.errors():

        sanitized_error = {
            "type": error.get(
                "type"
            ),
            "loc": error.get(
                "loc"
            ),
            "msg": error.get(
                "msg"
            ),
        }

        errors.append(
            sanitized_error
        )

    return JSONResponse(
        status_code=(
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ),
        content={
            "error": "validation_error",
            "message": (
                "One or more request fields are invalid."
            ),
            "details": errors,
            "request_id": request_id,
        },
    )


# ============================================================================
# UNEXPECTED EXCEPTION HANDLER
# ============================================================================


@app.exception_handler(
    Exception
)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):
    """
    Last-resort exception handler.

    Internal exception details and stack traces are logged server-side
    but NEVER returned to the client.
    """

    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    logger.exception(
        "Unhandled application exception | "
        "request_id=%s | "
        "method=%s | "
        "path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=(
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ),
        content={
            "error": "internal_server_error",
            "message": (
                "An unexpected server error occurred."
            ),
            "request_id": request_id,
        },
    )


# ============================================================================
# LOCAL DEVELOPMENT ENTRY POINT
# ============================================================================


if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            str(DEFAULT_PORT),
        )
    )

    is_development = (
        ENVIRONMENT
        in {
            "development",
            "dev",
            "local",
        }
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=is_development,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )