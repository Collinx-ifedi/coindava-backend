"""
Coindava Waitlist Backend
-------------------------

Production-grade configuration management.

Responsibilities:
- Load configuration exclusively from environment variables.
- Validate required production secrets.
- Provide strongly typed settings.
- Configure PostgreSQL connection settings.
- Configure CORS.
- Configure authentication/session security.
- Configure email verification.
- Configure X/Twitter OAuth.
- Configure waitlist/referral/points behaviour.
- Configure rate limiting and HTTP clients.
- Prevent unsafe production defaults.

This module contains configuration only.

Do NOT place:
- Business logic
- Database queries
- API handlers
- Authentication dependencies
- External API calls
- Background jobs

in this module.
"""

from __future__ import annotations

import json
import secrets
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import (
    AnyHttpUrl,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


# ============================================================================
# Types
# ============================================================================

Environment = Literal[
    "development",
    "testing",
    "staging",
    "production",
]


# ============================================================================
# Settings
# ============================================================================


class Settings(BaseSettings):
    """
    Central application configuration.

    Configuration is loaded from environment variables.

    Example:

        APP_ENV=production
        SECRET_KEY=your-long-random-secret
        DATABASE_URL=postgresql+psycopg://user:password@host/db
        FRONTEND_URL=https://coindava.com
        CORS_ORIGINS=https://coindava.com,https://www.coindava.com
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
        # Blank variables (e.g. "X_REDIRECT_URI=") are treated as unset
        # instead of failing URL/int validation.
        env_ignore_empty=True,
    )

    # =========================================================================
    # Application
    # =========================================================================

    app_name: str = Field(
        default="Coindava Waitlist API",
        validation_alias="APP_NAME",
        min_length=1,
        max_length=200,
    )

    app_version: str = Field(
        default="1.0.0",
        validation_alias="APP_VERSION",
        min_length=1,
        max_length=50,
    )

    app_env: Environment = Field(
        default="development",
        validation_alias="APP_ENV",
    )

    debug: bool = Field(
        default=False,
        validation_alias="DEBUG",
    )

    # =========================================================================
    # API
    # =========================================================================

    api_v1_prefix: str = Field(
        default="/api/v1",
        validation_alias="API_V1_PREFIX",
        min_length=1,
        max_length=100,
    )

    docs_enabled: bool = Field(
        default=True,
        validation_alias="DOCS_ENABLED",
    )

    # =========================================================================
    # Security
    # =========================================================================

    secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(
            secrets.token_urlsafe(64)
        ),
        validation_alias="SECRET_KEY",
    )

    # Encryption key length used by components that derive
    # encryption material from SECRET_KEY.
    encryption_key_length: int = Field(
        default=32,
        validation_alias="ENCRYPTION_KEY_LENGTH",
        ge=16,
        le=64,
    )

    # =========================================================================
    # Authentication / sessions
    # =========================================================================

    access_token_expire_minutes: int = Field(
        default=60 * 24 * 7,
        validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES",
        ge=5,
        le=60 * 24 * 30,
    )

    email_verification_expire_minutes: int = Field(
        default=60 * 24,
        validation_alias="EMAIL_VERIFICATION_EXPIRE_MINUTES",
        ge=5,
        le=60 * 24 * 7,
    )

    # Argon2id cost parameters (defaults follow the OWASP minimum profile:
    # 19 MiB, 2 iterations, 1 lane). Raising them is safe at any time:
    # existing hashes keep their own parameters and are upgraded on login.
    # Memory is used PER concurrent hash, so keep it modest on small hosts.
    argon2_time_cost: int = Field(
        default=2,
        validation_alias="ARGON2_TIME_COST",
        ge=1,
        le=10,
    )

    argon2_memory_cost_kib: int = Field(
        default=19_456,
        validation_alias="ARGON2_MEMORY_COST_KIB",
        ge=8_192,
        le=1_048_576,
    )

    argon2_parallelism: int = Field(
        default=1,
        validation_alias="ARGON2_PARALLELISM",
        ge=1,
        le=8,
    )

    # =========================================================================
    # Database
    # =========================================================================

    database_url: str = Field(
        default="sqlite:///./coindava_waitlist.db",
        validation_alias="DATABASE_URL",
        min_length=1,
    )

    database_pool_size: int = Field(
        default=10,
        validation_alias="DATABASE_POOL_SIZE",
        ge=1,
        le=100,
    )

    database_max_overflow: int = Field(
        default=20,
        validation_alias="DATABASE_MAX_OVERFLOW",
        ge=0,
        le=200,
    )

    database_pool_timeout: int = Field(
        default=30,
        validation_alias="DATABASE_POOL_TIMEOUT",
        ge=1,
        le=300,
    )

    database_pool_recycle: int = Field(
        default=1800,
        validation_alias="DATABASE_POOL_RECYCLE",
        ge=60,
        le=86400,
    )

    database_echo: bool = Field(
        default=False,
        validation_alias="DATABASE_ECHO",
    )

    # Create missing tables at startup.
    #
    # Always on outside production. In production it is OFF unless this is
    # explicitly enabled (useful for the very first deploy, before Alembic
    # migrations exist). create_all() only creates MISSING tables; it never
    # alters existing ones, so move to Alembic before changing the schema.
    auto_create_tables: bool = Field(
        default=False,
        validation_alias="AUTO_CREATE_TABLES",
    )

    # =========================================================================
    # CORS
    # =========================================================================

    # NoDecode stops pydantic-settings from JSON-parsing the raw env value
    # BEFORE parse_cors_origins() runs. Without it, the documented
    # "CORS_ORIGINS=https://a.com,https://b.com" format crashes at startup.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ],
        validation_alias="CORS_ORIGINS",
    )

    cors_allow_credentials: bool = Field(
        default=True,
        validation_alias="CORS_ALLOW_CREDENTIALS",
    )

    # =========================================================================
    # Frontend
    # =========================================================================

    frontend_url: str = Field(
        default="http://localhost:3000",
        validation_alias="FRONTEND_URL",
        min_length=1,
    )

    # =========================================================================
    # Waitlist
    # =========================================================================

    waitlist_enabled: bool = Field(
        default=True,
        validation_alias="WAITLIST_ENABLED",
    )

    waitlist_join_points: int = Field(
        default=100,
        validation_alias="WAITLIST_JOIN_POINTS",
        ge=0,
    )

    email_verified_points: int = Field(
        default=50,
        validation_alias="EMAIL_VERIFIED_POINTS",
        ge=0,
    )

    profile_completed_points: int = Field(
        default=50,
        validation_alias="PROFILE_COMPLETED_POINTS",
        ge=0,
    )

    # =========================================================================
    # Referral system
    # =========================================================================

    referral_enabled: bool = Field(
        default=True,
        validation_alias="REFERRAL_ENABLED",
    )

    referral_code_length: int = Field(
        default=8,
        validation_alias="REFERRAL_CODE_LENGTH",
        ge=6,
        le=32,
    )

    referral_points: int = Field(
        default=200,
        validation_alias="REFERRAL_POINTS",
        ge=0,
    )

    referral_verified_points: int = Field(
        default=100,
        validation_alias="REFERRAL_VERIFIED_POINTS",
        ge=0,
    )

    max_referrals_per_user: int = Field(
        default=10_000,
        validation_alias="MAX_REFERRALS_PER_USER",
        ge=1,
    )

    # =========================================================================
    # Leaderboard
    # =========================================================================

    leaderboard_default_limit: int = Field(
        default=100,
        validation_alias="LEADERBOARD_DEFAULT_LIMIT",
        ge=1,
        le=100,
    )

    leaderboard_max_limit: int = Field(
        default=100,
        validation_alias="LEADERBOARD_MAX_LIMIT",
        ge=1,
        le=500,
    )

    founders_circle_size: int = Field(
        default=100,
        validation_alias="FOUNDERS_CIRCLE_SIZE",
        ge=1,
        le=10_000,
    )

    # =========================================================================
    # X / Twitter
    # =========================================================================

    x_enabled: bool = Field(
        default=False,
        validation_alias="X_ENABLED",
    )

    x_client_id: str | None = Field(
        default=None,
        validation_alias="X_CLIENT_ID",
    )

    x_client_secret: SecretStr | None = Field(
        default=None,
        validation_alias="X_CLIENT_SECRET",
    )

    x_redirect_uri: AnyHttpUrl | None = Field(
        default=None,
        validation_alias="X_REDIRECT_URI",
    )

    x_authorization_url: str = Field(
        default="https://twitter.com/i/oauth2/authorize",
        validation_alias="X_AUTHORIZATION_URL",
    )

    x_token_url: str = Field(
        default="https://api.twitter.com/2/oauth2/token",
        validation_alias="X_TOKEN_URL",
    )

    x_api_base_url: str = Field(
        default="https://api.twitter.com/2",
        validation_alias="X_API_BASE_URL",
    )

    x_scope: str = Field(
        default=(
            "tweet.read "
            "users.read "
            "follows.read "
            "like.read "
            "offline.access"
        ),
        validation_alias="X_SCOPE",
    )

    # Fernet key used to encrypt X OAuth tokens at rest.
    # Generate one with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    social_token_encryption_key: SecretStr | None = Field(
        default=None,
        validation_alias="SOCIAL_TOKEN_ENCRYPTION_KEY",
    )

    # =========================================================================
    # Social challenge points
    # =========================================================================

    x_follow_points: int = Field(
        default=100,
        validation_alias="X_FOLLOW_POINTS",
        ge=0,
    )

    x_campaign_follow_points: int = Field(
        default=25,
        validation_alias="X_CAMPAIGN_FOLLOW_POINTS",
        ge=0,
    )

    x_like_points: int = Field(
        default=10,
        validation_alias="X_LIKE_POINTS",
        ge=0,
    )

    x_repost_points: int = Field(
        default=15,
        validation_alias="X_REPOST_POINTS",
        ge=0,
    )

    x_reply_points: int = Field(
        default=20,
        validation_alias="X_REPLY_POINTS",
        ge=0,
    )

    x_quote_post_points: int = Field(
        default=30,
        validation_alias="X_QUOTE_POST_POINTS",
        ge=0,
    )

    x_create_post_points: int = Field(
        default=150,
        validation_alias="X_CREATE_POST_POINTS",
        ge=0,
    )

    # =========================================================================
    # Social verification
    # =========================================================================

    social_verification_timeout_seconds: float = Field(
        default=10.0,
        validation_alias="SOCIAL_VERIFICATION_TIMEOUT_SECONDS",
        gt=0,
        le=60,
    )

    social_verification_max_retries: int = Field(
        default=2,
        validation_alias="SOCIAL_VERIFICATION_MAX_RETRIES",
        ge=0,
        le=10,
    )

    # =========================================================================
    # HTTP client
    # =========================================================================

    http_timeout_seconds: float = Field(
        default=15.0,
        validation_alias="HTTP_TIMEOUT_SECONDS",
        gt=0,
        le=120,
    )

    # =========================================================================
    # Email
    # =========================================================================

    email_enabled: bool = Field(
        default=False,
        validation_alias="EMAIL_ENABLED",
    )

    email_provider: str = Field(
        default="brevo",
        validation_alias="EMAIL_PROVIDER",
    )

    email_from: str | None = Field(
        default=None,
        validation_alias="EMAIL_FROM",
    )

    # Brevo API key (Brevo dashboard -> SMTP & API -> API keys).
    email_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="EMAIL_API_KEY",
    )

    # SMTP
    smtp_host: str | None = Field(
        default=None,
        validation_alias="SMTP_HOST",
    )

    smtp_port: int = Field(
        default=587,
        validation_alias="SMTP_PORT",
        ge=1,
        le=65_535,
    )

    smtp_username: str | None = Field(
        default=None,
        validation_alias="SMTP_USERNAME",
    )

    smtp_password: SecretStr | None = Field(
        default=None,
        validation_alias="SMTP_PASSWORD",
    )

    smtp_use_tls: bool = Field(
        default=True,
        validation_alias="SMTP_USE_TLS",
    )

    # =========================================================================
    # Rate limiting
    # =========================================================================

    rate_limit_enabled: bool = Field(
        default=True,
        validation_alias="RATE_LIMIT_ENABLED",
    )

    rate_limit_requests: int = Field(
        default=120,
        validation_alias="RATE_LIMIT_REQUESTS",
        ge=1,
    )

    rate_limit_window_seconds: int = Field(
        default=60,
        validation_alias="RATE_LIMIT_WINDOW_SECONDS",
        ge=1,
        le=3600,
    )

    # =========================================================================
    # Logging
    # =========================================================================

    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
    )

    log_json: bool = Field(
        default=True,
        validation_alias="LOG_JSON",
    )

    # =========================================================================
    # Health / observability
    # =========================================================================

    health_check_enabled: bool = Field(
        default=True,
        validation_alias="HEALTH_CHECK_ENABLED",
    )

    # =========================================================================
    # Validators
    # =========================================================================

    @field_validator("database_url")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        """
        Normalize hosted-PostgreSQL URLs.

        Render/Heroku-style providers hand out "postgres://..." URLs.
        SQLAlchemy 2.x rejects the "postgres" dialect name, and a bare
        "postgresql://" would select the psycopg2 driver. This project
        ships psycopg (v3), so both forms are rewritten to
        "postgresql+psycopg://".
        """

        value = value.strip()

        if value.startswith("postgres://"):
            value = "postgresql://" + value[len("postgres://"):]

        if value.startswith("postgresql://"):
            value = "postgresql+psycopg://" + value[len("postgresql://"):]

        return value

    @field_validator("api_v1_prefix")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        """
        Normalize API prefix.

        Examples:
            api/v1   -> /api/v1
            /api/v1/ -> /api/v1
        """

        value = value.strip()

        if not value:
            raise ValueError(
                "API_V1_PREFIX cannot be empty."
            )

        if not value.startswith("/"):
            value = f"/{value}"

        return value.rstrip("/")

    # -------------------------------------------------------------------------
    # CORS
    # -------------------------------------------------------------------------

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        """
        Accept:

            CORS_ORIGINS=https://coindava.com,https://www.coindava.com

        or:

            CORS_ORIGINS=["https://coindava.com"]

        or a native list.
        """

        if value is None:
            return []

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return []

            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "CORS_ORIGINS contains invalid JSON."
                    ) from exc

                if not isinstance(parsed, list):
                    raise ValueError(
                        "CORS_ORIGINS JSON value must be an array."
                    )

                return [
                    str(origin).strip()
                    for origin in parsed
                    if str(origin).strip()
                ]

            return [
                origin.strip()
                for origin in value.split(",")
                if origin.strip()
            ]

        if isinstance(value, (list, tuple, set)):
            return [
                str(origin).strip()
                for origin in value
                if str(origin).strip()
            ]

        raise ValueError(
            "CORS_ORIGINS must be a comma-separated string "
            "or a list of origins."
        )

    # -------------------------------------------------------------------------
    # Log level
    # -------------------------------------------------------------------------

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Normalize and validate logging level."""

        normalized = value.strip().upper()

        allowed = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        if normalized not in allowed:
            raise ValueError(
                "LOG_LEVEL must be one of: "
                + ", ".join(sorted(allowed))
            )

        return normalized

    # -------------------------------------------------------------------------
    # Email provider
    # -------------------------------------------------------------------------

    @field_validator("email_provider")
    @classmethod
    def validate_email_provider(cls, value: str) -> str:
        """Normalize and validate email provider."""

        normalized = value.strip().lower()

        # "brevo" = Brevo transactional email API (default).
        # "smtp"  = any SMTP server, e.g. Brevo's relay smtp-relay.brevo.com.
        allowed = {
            "brevo",
            "smtp",
        }

        if normalized not in allowed:
            raise ValueError(
                "EMAIL_PROVIDER must be one of: "
                + ", ".join(sorted(allowed))
            )

        return normalized

    # -------------------------------------------------------------------------
    # X scopes
    # -------------------------------------------------------------------------

    @field_validator("x_scope")
    @classmethod
    def validate_x_scope(cls, value: str) -> str:
        """Normalize X OAuth scopes."""

        scopes = value.split()

        if not scopes:
            raise ValueError(
                "X_SCOPE must contain at least one OAuth scope."
            )

        # Remove duplicates while preserving order.
        unique_scopes = list(dict.fromkeys(scopes))

        return " ".join(unique_scopes)

    # -------------------------------------------------------------------------
    # Generic URL validation
    # -------------------------------------------------------------------------

    @field_validator(
        "x_authorization_url",
        "x_token_url",
        "x_api_base_url",
    )
    @classmethod
    def validate_external_urls(cls, value: str) -> str:
        """
        Basic URL safety validation for configured external services.

        These URLs are configuration values and are therefore expected
        to be HTTPS in real deployments.
        """

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "Configured external URL cannot be empty."
            )

        if not (
            normalized.startswith("https://")
            or normalized.startswith("http://localhost")
            or normalized.startswith("http://127.0.0.1")
        ):
            raise ValueError(
                "Configured external service URLs must use HTTPS "
                "except local development URLs."
            )

        return normalized.rstrip("/")

    # =========================================================================
    # Cross-field / environment validation
    # =========================================================================

    @model_validator(mode="after")
    def validate_environment(self) -> "Settings":
        """
        Perform cross-field validation and production safety checks.
        """

        # =====================================================================
        # General application checks
        # =====================================================================

        if self.leaderboard_default_limit > self.leaderboard_max_limit:
            raise ValueError(
                "LEADERBOARD_DEFAULT_LIMIT cannot exceed "
                "LEADERBOARD_MAX_LIMIT."
            )

        if self.founders_circle_size > self.leaderboard_max_limit:
            raise ValueError(
                "FOUNDERS_CIRCLE_SIZE cannot exceed "
                "LEADERBOARD_MAX_LIMIT."
            )

        # =====================================================================
        # Production security
        # =====================================================================

        if self.app_env == "production":
            # Debugging must never be enabled in production.
            if self.debug:
                raise ValueError(
                    "DEBUG must be false in production."
                )

            # A generated default SECRET_KEY is acceptable for local
            # development, but not for a production deployment.
            # SECRET_KEY must be set explicitly. The auto-generated fallback
            # changes on every restart and differs between workers.
            if "secret_key" not in self.model_fields_set:
                raise ValueError(
                    "SECRET_KEY must be set explicitly in production."
                )

            secret = self.secret_key.get_secret_value()

            if len(secret) < 32:
                raise ValueError(
                    "SECRET_KEY must contain at least 32 characters "
                    "in production."
                )

            # Prevent the common mistake of deploying with the generated
            # development secret.
            if secret.startswith("CHANGE_ME"):
                raise ValueError(
                    "A real SECRET_KEY must be configured in production."
                )

            # Production database must be PostgreSQL.
            if not self.database_url.startswith(
                (
                    "postgresql://",
                    "postgresql+psycopg://",
                    "postgresql+psycopg2://",
                    "postgres://",
                )
            ):
                raise ValueError(
                    "Production requires PostgreSQL. "
                    "Configure DATABASE_URL with a PostgreSQL connection URL."
                )

            # HTTPS frontend is mandatory.
            if not self.frontend_url.startswith("https://"):
                raise ValueError(
                    "FRONTEND_URL must use HTTPS in production."
                )

            # Wildcard CORS is dangerous when credentials are enabled.
            if "*" in self.cors_origins:
                raise ValueError(
                    "Wildcard CORS origin is not permitted in production."
                )

            if self.cors_allow_credentials and "*" in self.cors_origins:
                raise ValueError(
                    "Wildcard CORS cannot be combined with credentials."
                )

            # Production API should not expose interactive documentation
            # unless explicitly enabled.
            #
            # We do not automatically force this off because a deployment
            # may intentionally expose protected documentation.
            if self.docs_enabled and self.debug:
                raise ValueError(
                    "Interactive API documentation cannot be enabled "
                    "while DEBUG is enabled in production."
                )

        # =====================================================================
        # Testing environment
        # =====================================================================

        if self.app_env == "testing":
            # Tests should not accidentally contact real X services.
            if self.x_enabled:
                raise ValueError(
                    "X_ENABLED should normally be false in testing."
                )

        # =====================================================================
        # X integration
        # =====================================================================

        if self.x_enabled:
            missing_x: list[str] = []

            if not self.x_client_id:
                missing_x.append("X_CLIENT_ID")

            if not self.x_client_secret:
                missing_x.append("X_CLIENT_SECRET")

            if not self.x_redirect_uri:
                missing_x.append("X_REDIRECT_URI")

            if not self.social_token_encryption_key:
                missing_x.append("SOCIAL_TOKEN_ENCRYPTION_KEY")

            if missing_x:
                raise ValueError(
                    "X integration is enabled but required variables "
                    f"are missing: {', '.join(missing_x)}"
                )

            try:
                from cryptography.fernet import Fernet

                Fernet(
                    self.social_token_encryption_key.get_secret_value()
                    .encode("utf-8")
                )
            except Exception as exc:
                raise ValueError(
                    "SOCIAL_TOKEN_ENCRYPTION_KEY is not a valid Fernet key."
                ) from exc

        # =====================================================================
        # Email
        # =====================================================================

        if self.email_enabled:
            missing_email: list[str] = []

            if not self.email_from:
                missing_email.append("EMAIL_FROM")

            if self.email_provider == "smtp":
                if not self.smtp_host:
                    missing_email.append("SMTP_HOST")

                if not self.smtp_username:
                    missing_email.append("SMTP_USERNAME")

                if not self.smtp_password:
                    missing_email.append("SMTP_PASSWORD")

            elif self.email_provider == "brevo":
                if not self.email_api_key:
                    missing_email.append("EMAIL_API_KEY")

            if missing_email:
                raise ValueError(
                    "Email is enabled but required variables are missing: "
                    + ", ".join(missing_email)
                )

        # =====================================================================
        # Referral configuration
        # =====================================================================

        if self.referral_enabled:
            if self.referral_code_length < 6:
                raise ValueError(
                    "REFERRAL_CODE_LENGTH must be at least 6."
                )

            if self.max_referrals_per_user < 1:
                raise ValueError(
                    "MAX_REFERRALS_PER_USER must be at least 1."
                )

        # =====================================================================
        # Waitlist
        # =====================================================================

        if not self.waitlist_enabled:
            # Referral functionality has no useful purpose if the waitlist
            # itself is disabled.
            if self.referral_enabled:
                raise ValueError(
                    "REFERRAL_ENABLED cannot be true when "
                    "WAITLIST_ENABLED is false."
                )

        return self


# ============================================================================
# Cached settings
# ============================================================================


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application's singleton Settings instance.

    Configuration is parsed once per process.
    """

    try:
        return Settings()

    except ValidationError as exc:
        # Report WHICH settings are invalid (names and messages only, never
        # the supplied values) so a failed deploy is diagnosable from logs.
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'settings'}: "
            f"{error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        )

        raise RuntimeError(
            "Invalid Coindava application configuration -> "
            f"{problems}"
        ) from exc


# ============================================================================
# Global settings
# ============================================================================

settings = get_settings()


# ============================================================================
# Environment helpers
# ============================================================================


def is_production() -> bool:
    """Return True when the application runs in production."""

    return settings.app_env == "production"


def is_development() -> bool:
    """Return True when the application runs in development."""

    return settings.app_env == "development"


def is_testing() -> bool:
    """Return True when the application runs in testing."""

    return settings.app_env == "testing"


def is_staging() -> bool:
    """Return True when the application runs in staging."""

    return settings.app_env == "staging"


# ============================================================================
# Database helpers
# ============================================================================


def get_database_url() -> str:
    """
    Return the configured database URL.

    database.py should use this helper rather than reading environment
    variables directly.
    """

    return settings.database_url


# ============================================================================
# Security helpers
# ============================================================================


def get_secret_key() -> str:
    """
    Return the application secret.

    Never log, serialize, or expose this value.
    """

    return settings.secret_key.get_secret_value()


def get_social_encryption_key() -> str | None:
    """
    Return the Fernet key for X token encryption (never log or expose it).
    """

    key = settings.social_token_encryption_key

    return key.get_secret_value() if key else None


# ============================================================================
# Public exports
# ============================================================================

__all__ = [
    "Environment",
    "Settings",
    "settings",
    "get_settings",
    "get_database_url",
    "get_secret_key",
    "get_social_encryption_key",
    "is_production",
    "is_development",
    "is_testing",
    "is_staging",
]