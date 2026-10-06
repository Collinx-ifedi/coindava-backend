"""
Coindava Waitlist Backend
-------------------------

Production-grade SQLAlchemy ORM models.

Core entities:
    User
    EmailVerification
    SessionToken
    Referral
    PointEvent
    XAccount
    OAuthState
    Campaign
    CampaignTask
    SocialAction

Design principles:
    - PostgreSQL-first production schema (also runs on SQLite for local dev)
    - UUID primary keys
    - UTC-aware timestamps
    - Database-level constraints
    - Explicit indexes for common access paths
    - Immutable point-event ledger
    - Idempotent reward events
    - Idempotent social challenge records
    - Foreign-key integrity
    - No plaintext passwords
    - No plaintext X OAuth tokens
    - Deterministic leaderboard support

Enum storage:
    Python enums are stored by VALUE ("active", "pending", ...), never by
    member NAME. All server_default values below are the lowercase values, so
    the native PostgreSQL enum labels must be the values too.

This module contains ORM/schema definitions only.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    false,
    func,
    true,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from database import Base


# =============================================================================
# TYPE HELPERS
# =============================================================================

# JSONB on PostgreSQL, plain JSON elsewhere (SQLite cannot compile JSONB).
JSONType = JSON().with_variant(JSONB(), "postgresql")


def _enum_column(enum_class: type[enum.Enum], name: str) -> Enum:
    """
    Build a native-enum column type that stores enum VALUES.

    By default SQLAlchemy persists enum member NAMES ("ACTIVE"), which would
    not match the lowercase server defaults and the values used in the API.
    """

    return Enum(
        enum_class,
        name=name,
        native_enum=True,
        create_constraint=False,
        validate_strings=True,
        values_callable=lambda members: [member.value for member in members],
    )


# =============================================================================
# ENUMS
# =============================================================================


class UserStatus(str, enum.Enum):
    """Lifecycle state of a waitlist user."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    BANNED = "banned"
    DELETED = "deleted"


class ReferralStatus(str, enum.Enum):
    """Referral lifecycle state."""

    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"


class PointEventType(str, enum.Enum):
    """
    Immutable point-ledger event types.

    This is the single canonical list used by models.py, points.py and
    social.py. Treat the values as stable production identifiers.
    """

    WAITLIST_JOINED = "waitlist_joined"
    EMAIL_VERIFIED = "email_verified"
    PROFILE_COMPLETED = "profile_completed"

    REFERRAL_CREATED = "referral_created"
    REFERRAL_VERIFIED = "referral_verified"

    DAILY_CHALLENGE = "daily_challenge"
    WEEKLY_CHALLENGE = "weekly_challenge"

    STREAK_7_DAYS = "streak_7_days"
    STREAK_14_DAYS = "streak_14_days"
    STREAK_30_DAYS = "streak_30_days"

    SOCIAL_FOLLOW = "social_follow"
    SOCIAL_LIKE = "social_like"
    SOCIAL_REPOST = "social_repost"
    SOCIAL_REPLY = "social_reply"
    SOCIAL_QUOTE = "social_quote"
    SOCIAL_POST = "social_post"

    CAMPAIGN_BONUS = "campaign_bonus"

    MANUAL_ADJUSTMENT = "manual_adjustment"
    ADMIN_ADJUSTMENT = "admin_adjustment"


class SocialPlatform(str, enum.Enum):
    """Supported social platforms."""

    X = "x"


class SocialActionType(str, enum.Enum):
    """Social actions that may earn points."""

    FOLLOW = "follow"
    LIKE = "like"
    REPOST = "repost"
    REPLY = "reply"
    QUOTE_POST = "quote_post"
    CREATE_POST = "create_post"


class SocialActionStatus(str, enum.Enum):
    """Verification state for a social action."""

    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    REVOKED = "revoked"


class CampaignStatus(str, enum.Enum):
    """Campaign lifecycle."""

    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class CampaignTaskType(str, enum.Enum):
    """Types of campaign challenges (mirror SocialActionType)."""

    FOLLOW = "follow"
    LIKE = "like"
    REPOST = "repost"
    REPLY = "reply"
    QUOTE_POST = "quote_post"
    CREATE_POST = "create_post"


# =============================================================================
# COMMON TIMESTAMP MIXIN
# =============================================================================


class TimestampMixin:
    """
    Common creation/update timestamps generated by the database so the
    database remains authoritative for time.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


# =============================================================================
# USER
# =============================================================================


class User(TimestampMixin, Base):
    """
    Core Coindava waitlist user.

    Security:
        - Passwords are stored only as password hashes.
        - X credentials are stored separately in XAccount.

    Points:
        `points` is a materialized balance used for fast leaderboard queries.
        The authoritative audit trail is PointEvent.
    """

    __tablename__ = "users"

    __table_args__ = (
        UniqueConstraint("email", name="uq_users_email"),
        UniqueConstraint("username", name="uq_users_username"),
        UniqueConstraint("referral_code", name="uq_users_referral_code"),
        CheckConstraint("points >= 0", name="ck_users_points_non_negative"),
        # Supports leaderboard ordering.
        Index(
            "ix_users_leaderboard",
            "status",
            "points",
            "created_at",
            "id",
        ),
        Index("ix_users_created_at", "created_at"),
        Index("ix_users_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    email: Mapped[str] = mapped_column(String(320), nullable=False)

    username: Mapped[str] = mapped_column(String(32), nullable=False)

    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    referral_code: Mapped[str] = mapped_column(String(32), nullable=False)

    points: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default="0",
    )

    email_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
    )

    profile_completed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
    )

    # Grants access to /social/admin/*. Never settable through the public API;
    # use `python manage.py make-admin <email>`.
    is_admin: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
    )

    status: Mapped[UserStatus] = mapped_column(
        _enum_column(UserStatus, "user_status"),
        nullable=False,
        default=UserStatus.ACTIVE,
        server_default=UserStatus.ACTIVE.value,
    )

    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    email_verifications: Mapped[list["EmailVerification"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    sessions: Mapped[list["SessionToken"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    referrals_sent: Mapped[list["Referral"]] = relationship(
        foreign_keys="Referral.referrer_id",
        back_populates="referrer",
        passive_deletes=True,
    )

    referral_received: Mapped["Referral | None"] = relationship(
        foreign_keys="Referral.referred_user_id",
        back_populates="referred_user",
        uselist=False,
        passive_deletes=True,
    )

    point_events: Mapped[list["PointEvent"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    x_account: Mapped["XAccount | None"] = relationship(
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    social_actions: Mapped[list["SocialAction"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# =============================================================================
# EMAIL VERIFICATION
# =============================================================================


class EmailVerification(Base):
    """
    Email verification token. Only a SHA-256 hash of the token is stored.
    """

    __tablename__ = "email_verifications"

    __table_args__ = (
        Index("ix_email_verifications_user_id", "user_id"),
        Index("ix_email_verifications_expires_at", "expires_at"),
        Index("ix_email_verifications_token_hash", "token_hash"),
        Index("ix_email_verifications_used_at", "used_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    token_hash: Mapped[str] = mapped_column(String(128), nullable=False)

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    user: Mapped["User"] = relationship(back_populates="email_verifications")


# =============================================================================
# SESSION TOKEN
# =============================================================================


class SessionToken(Base):
    """
    Persistent authentication session. Only a hash of the bearer token is
    stored.
    """

    __tablename__ = "session_tokens"

    __table_args__ = (
        UniqueConstraint("token_hash", name="uq_session_tokens_token_hash"),
        Index("ix_session_tokens_user_id", "user_id"),
        Index("ix_session_tokens_expires_at", "expires_at"),
        Index("ix_session_tokens_revoked_at", "revoked_at"),
        Index(
            "ix_session_tokens_user_active",
            "user_id",
            "revoked_at",
            "expires_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    token_hash: Mapped[str] = mapped_column(String(128), nullable=False)

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    user: Mapped["User"] = relationship(back_populates="sessions")


# =============================================================================
# REFERRAL
# =============================================================================


class Referral(Base):
    """
    Referral relationship between two users.

    Rules enforced by the schema:
        - A referred user can have only one referral.
        - A referral must point to an existing referrer.
    """

    __tablename__ = "referrals"

    __table_args__ = (
        UniqueConstraint("referred_user_id", name="uq_referrals_referred_user"),
        Index("ix_referrals_referrer_id", "referrer_id"),
        Index("ix_referrals_status", "status"),
        Index("ix_referrals_created_at", "created_at"),
        Index("ix_referrals_referrer_status", "referrer_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    referrer_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )

    referred_user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    referral_code: Mapped[str] = mapped_column(String(32), nullable=False)

    status: Mapped[ReferralStatus] = mapped_column(
        _enum_column(ReferralStatus, "referral_status"),
        nullable=False,
        default=ReferralStatus.PENDING,
        server_default=ReferralStatus.PENDING.value,
    )

    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    referrer: Mapped["User"] = relationship(
        foreign_keys=[referrer_id],
        back_populates="referrals_sent",
    )

    referred_user: Mapped["User"] = relationship(
        foreign_keys=[referred_user_id],
        back_populates="referral_received",
    )


# =============================================================================
# POINT EVENT
# =============================================================================


class PointEvent(Base):
    """
    Immutable point ledger event.

    `User.points` is a materialized balance; `PointEvent` is the audit trail.
    `reference_id` provides idempotency for reward-triggering events.
    """

    __tablename__ = "point_events"

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "event_type",
            "reference_id",
            name="uq_point_events_idempotency",
        ),
        Index("ix_point_events_user_id", "user_id"),
        Index("ix_point_events_event_type", "event_type"),
        Index("ix_point_events_created_at", "created_at"),
        Index("ix_point_events_reference_id", "reference_id"),
        Index("ix_point_events_user_created", "user_id", "created_at"),
        CheckConstraint("points <> 0", name="ck_point_events_non_zero"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    event_type: Mapped[PointEventType] = mapped_column(
        _enum_column(PointEventType, "point_event_type"),
        nullable=False,
    )

    points: Mapped[int] = mapped_column(BigInteger, nullable=False)

    reference_id: Mapped[str] = mapped_column(String(255), nullable=False)

    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # NOTE: the attribute is `metadata_json` because `metadata` is reserved
    # by SQLAlchemy's declarative base.
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONType,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    user: Mapped["User"] = relationship(back_populates="point_events")


# =============================================================================
# X ACCOUNT
# =============================================================================


class XAccount(Base):
    """
    Connected X account.

    `access_token` / `refresh_token` hold Fernet-ENCRYPTED values only and
    must never be returned by public API schemas.
    """

    __tablename__ = "x_accounts"

    __table_args__ = (
        UniqueConstraint("user_id", name="uq_x_accounts_user_id"),
        UniqueConstraint(
            "provider",
            "provider_user_id",
            name="uq_x_accounts_provider_user",
        ),
        Index("ix_x_accounts_username", "username"),
        Index("ix_x_accounts_expires_at", "expires_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    provider: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="x",
        server_default="x",
    )

    # The X user ID (a numeric string).
    provider_user_id: Mapped[str] = mapped_column(String(64), nullable=False)

    username: Mapped[str | None] = mapped_column(String(64), nullable=True)

    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    access_token: Mapped[str | None] = mapped_column(Text, nullable=True)

    refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    scopes: Mapped[str | None] = mapped_column(Text, nullable=True)

    connected: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=true(),
    )

    connected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    user: Mapped["User"] = relationship(back_populates="x_account")


# =============================================================================
# OAUTH STATE
# =============================================================================


class OAuthState(Base):
    """
    Short-lived, single-use OAuth 2.0 state + PKCE verifier.

    Only the SHA-256 hash of the state is stored; the PKCE code verifier is
    stored Fernet-encrypted.
    """

    __tablename__ = "oauth_states"

    __table_args__ = (
        UniqueConstraint("state_hash", name="uq_oauth_states_state_hash"),
        Index("ix_oauth_states_user_id", "user_id"),
        Index("ix_oauth_states_expires_at", "expires_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    provider: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="x",
        server_default="x",
    )

    state_hash: Mapped[str] = mapped_column(String(128), nullable=False)

    code_verifier_encrypted: Mapped[str] = mapped_column(Text, nullable=False)

    redirect_uri: Mapped[str] = mapped_column(Text, nullable=False)

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    consumed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


# =============================================================================
# CAMPAIGN
# =============================================================================


class Campaign(TimestampMixin, Base):
    """Coindava social campaign. A campaign contains many CampaignTasks."""

    __tablename__ = "campaigns"

    __table_args__ = (
        UniqueConstraint("slug", name="uq_campaigns_slug"),
        Index("ix_campaigns_status", "status"),
        Index("ix_campaigns_start_at", "start_at"),
        Index("ix_campaigns_end_at", "end_at"),
        Index("ix_campaigns_status_dates", "status", "start_at", "end_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    name: Mapped[str] = mapped_column(String(150), nullable=False)

    slug: Mapped[str] = mapped_column(String(150), nullable=False)

    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[CampaignStatus] = mapped_column(
        _enum_column(CampaignStatus, "campaign_status"),
        nullable=False,
        default=CampaignStatus.DRAFT,
        server_default=CampaignStatus.DRAFT.value,
    )

    start_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    end_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    tasks: Mapped[list["CampaignTask"]] = relationship(
        back_populates="campaign",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="CampaignTask.created_at",
    )


# =============================================================================
# CAMPAIGN TASK
# =============================================================================


class CampaignTask(TimestampMixin, Base):
    """Individual social challenge within a campaign."""

    __tablename__ = "campaign_tasks"

    __table_args__ = (
        Index("ix_campaign_tasks_campaign_id", "campaign_id"),
        Index("ix_campaign_tasks_action_type", "action_type"),
        Index("ix_campaign_tasks_platform", "platform"),
        Index("ix_campaign_tasks_active", "is_active"),
        Index(
            "ix_campaign_tasks_campaign_active",
            "campaign_id",
            "is_active",
        ),
        CheckConstraint(
            "points >= 0",
            name="ck_campaign_tasks_points_non_negative",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    campaign_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(String(150), nullable=False)

    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    action_type: Mapped[CampaignTaskType] = mapped_column(
        _enum_column(CampaignTaskType, "campaign_task_type"),
        nullable=False,
    )

    platform: Mapped[SocialPlatform] = mapped_column(
        _enum_column(SocialPlatform, "social_platform"),
        nullable=False,
        default=SocialPlatform.X,
        server_default=SocialPlatform.X.value,
    )

    # FOLLOW tasks: the X user ID that must be followed.
    target_user_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    # LIKE / REPOST / REPLY / QUOTE tasks: the X post ID.
    target_post_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    # Link shown to the user ("Open on X").
    target_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # CREATE_POST tasks: optional content requirements.
    required_phrase: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    required_hashtag: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    points: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=true(),
    )

    campaign: Mapped["Campaign"] = relationship(back_populates="tasks")

    social_actions: Mapped[list["SocialAction"]] = relationship(
        back_populates="campaign_task",
        passive_deletes=True,
    )


# =============================================================================
# SOCIAL ACTION
# =============================================================================


class SocialAction(Base):
    """
    User participation in a social campaign task.

    Provides idempotency (one row per user+task), verification history and
    fraud-investigation context.
    """

    __tablename__ = "social_actions"

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "campaign_task_id",
            name="uq_social_actions_user_task",
        ),
        Index("ix_social_actions_user_id", "user_id"),
        Index("ix_social_actions_task_id", "campaign_task_id"),
        Index("ix_social_actions_status", "status"),
        Index("ix_social_actions_external_id", "external_id"),
        Index("ix_social_actions_created_at", "created_at"),
        Index("ix_social_actions_user_status", "user_id", "status"),
        CheckConstraint(
            "verification_attempts >= 0",
            name="ck_social_actions_attempts_non_negative",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    campaign_task_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("campaign_tasks.id", ondelete="CASCADE"),
        nullable=False,
    )

    x_account_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("x_accounts.id", ondelete="SET NULL"),
        nullable=True,
    )

    platform: Mapped[SocialPlatform] = mapped_column(
        _enum_column(SocialPlatform, "social_platform"),
        nullable=False,
        default=SocialPlatform.X,
        server_default=SocialPlatform.X.value,
    )

    action_type: Mapped[SocialActionType] = mapped_column(
        _enum_column(SocialActionType, "social_action_type"),
        nullable=False,
    )

    status: Mapped[SocialActionStatus] = mapped_column(
        _enum_column(SocialActionStatus, "social_action_status"),
        nullable=False,
        default=SocialActionStatus.PENDING,
        server_default=SocialActionStatus.PENDING.value,
    )

    # Convenience flag mirrored from `status` (True once VERIFIED).
    verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
    )

    # The X object that proved the action (post ID or target user ID).
    external_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    points_awarded: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    verification_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    failed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONType,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    user: Mapped["User"] = relationship(back_populates="social_actions")

    campaign_task: Mapped["CampaignTask"] = relationship(
        back_populates="social_actions",
    )


# =============================================================================
# MODEL EXPORTS
# =============================================================================


__all__ = [
    "User",
    "UserStatus",
    "EmailVerification",
    "SessionToken",
    "Referral",
    "ReferralStatus",
    "PointEvent",
    "PointEventType",
    "XAccount",
    "OAuthState",
    "Campaign",
    "CampaignStatus",
    "CampaignTask",
    "CampaignTaskType",
    "SocialAction",
    "SocialActionType",
    "SocialActionStatus",
    "SocialPlatform",
]
