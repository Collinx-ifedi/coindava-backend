"""
Coindava Waitlist Backend
-------------------------

points.py

Centralized point-awarding and point-ledger service.

IMPORTANT:
    This module is the ONLY application layer that should modify
    User.points.

Architecture:

    User action
        |
        v
    Application module
        |
        v
    points.py
        |
        +--> validate event
        |
        +--> resolve server-controlled reward
        |
        +--> acquire user row lock
        |
        +--> idempotency check
        |
        +--> create immutable PointEvent
        |
        +--> update User.points
        |
        v
    PostgreSQL transaction

Point principles:

    1. Every point change creates a PointEvent.
    2. PointEvents are immutable after creation.
    3. Reward operations are idempotent.
    4. User.points is a cached aggregate.
    5. The event ledger is the audit source.
    6. Client-provided point amounts are never trusted.
    7. All point mutations occur inside a database transaction.
    8. Concurrent balance updates use row-level locking.
    9. Integrity races use SAVEPOINTs instead of destroying the
       caller's transaction.
    10. Public API endpoints cannot arbitrarily award points.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from auth import CurrentUser
from config import settings
from database import get_db
from models import PointEvent, PointEventType, User, UserStatus


# =============================================================================
# LOGGER
# =============================================================================

logger = logging.getLogger("coindava.points")


# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter(
    prefix="/points",
    tags=["Points"],
)


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_POINT_ADJUSTMENT = 1_000_000

MAX_EVENT_TYPE_LENGTH = 64
MAX_REFERENCE_ID_LENGTH = 128
MAX_DESCRIPTION_LENGTH = 500

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100


# =============================================================================
# POINT EVENT TYPES
# =============================================================================
#
# PointEventType is defined ONCE, in models.py, because it is also the
# PostgreSQL enum behind PointEvent.event_type. Keeping a second copy here
# (as the original code did) let the two lists drift apart, and inserting a
# value missing from the database enum fails at runtime.

# =============================================================================
# DEFAULT REWARD VALUES
# =============================================================================

DEFAULT_POINT_VALUES: dict[str, int] = {
    PointEventType.WAITLIST_JOINED.value: 100,

    PointEventType.EMAIL_VERIFIED.value: 50,

    PointEventType.PROFILE_COMPLETED.value: 50,

    PointEventType.REFERRAL_VERIFIED.value: 200,

    PointEventType.REFERRAL_CREATED.value: 100,

    PointEventType.DAILY_CHALLENGE.value: 25,

    PointEventType.WEEKLY_CHALLENGE.value: 100,

    PointEventType.STREAK_7_DAYS.value: 250,

    PointEventType.STREAK_14_DAYS.value: 500,

    PointEventType.STREAK_30_DAYS.value: 1500,

    PointEventType.SOCIAL_FOLLOW.value: 100,

    PointEventType.SOCIAL_LIKE.value: 10,

    PointEventType.SOCIAL_REPOST.value: 15,

    PointEventType.SOCIAL_REPLY.value: 20,

    PointEventType.SOCIAL_QUOTE.value: 30,

    PointEventType.SOCIAL_POST.value: 150,
}


# =============================================================================
# EVENT CLASSIFICATION
# =============================================================================

SOCIAL_EVENT_TYPES = frozenset(
    {
        PointEventType.SOCIAL_FOLLOW.value,
        PointEventType.SOCIAL_LIKE.value,
        PointEventType.SOCIAL_REPOST.value,
        PointEventType.SOCIAL_REPLY.value,
        PointEventType.SOCIAL_QUOTE.value,
        PointEventType.SOCIAL_POST.value,
    }
)


CONFIGURABLE_REWARD_TYPES = frozenset(
    DEFAULT_POINT_VALUES.keys()
)


# =============================================================================
# TIME HELPERS
# =============================================================================

def utc_now() -> datetime:
    """
    Return a timezone-aware UTC timestamp.
    """

    return datetime.now(timezone.utc)


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_event_type(
    event_type: str | PointEventType,
) -> str:
    """
    Normalize and validate a point event type.
    """

    if isinstance(event_type, Enum):
        value = str(event_type.value)
    else:
        value = str(event_type)

    normalized = value.strip().lower()

    if not normalized:
        raise ValueError(
            "Point event type cannot be empty."
        )

    if len(normalized) > MAX_EVENT_TYPE_LENGTH:
        raise ValueError(
            "Point event type is too long."
        )

    return normalized


def normalize_reference_id(
    reference_id: str | UUID | None,
) -> str | None:
    """
    Normalize an optional event reference ID.
    """

    if reference_id is None:
        return None

    value = str(reference_id).strip()

    if not value:
        return None

    if len(value) > MAX_REFERENCE_ID_LENGTH:
        raise ValueError(
            "Point event reference is too long."
        )

    return value


def normalize_description(
    description: str | None,
) -> str | None:
    """
    Normalize an optional point-event description.
    """

    if description is None:
        return None

    value = description.strip()

    if not value:
        return None

    if len(value) > MAX_DESCRIPTION_LENGTH:
        raise ValueError(
            "Point-event description is too long."
        )

    return value


# =============================================================================
# ACCOUNT VALIDATION
# =============================================================================

def ensure_active_user(
    user: User,
) -> None:
    """
    Ensure an account is permitted to participate in the point system.
    """

    if user.status == UserStatus.SUSPENDED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account suspended.",
        )

    if user.status in {
        UserStatus.BANNED,
        UserStatus.DELETED,
    }:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account unavailable.",
        )


# =============================================================================
# SERVER-CONTROLLED REWARD RESOLUTION
# =============================================================================

def get_configured_reward(
    event_type: str | PointEventType,
) -> int:
    """
    Resolve the server-controlled reward for an event.

    The client cannot choose the reward amount.

    Config.py values take precedence where configured. Otherwise the
    production defaults in this module are used.
    """

    normalized = normalize_event_type(
        event_type
    )

    reward_setting_names: dict[str, str] = {
        PointEventType.WAITLIST_JOINED.value:
            "waitlist_join_points",

        PointEventType.EMAIL_VERIFIED.value:
            "email_verified_points",

        PointEventType.PROFILE_COMPLETED.value:
            "profile_completed_points",

        PointEventType.REFERRAL_VERIFIED.value:
            "referral_points",

        PointEventType.REFERRAL_CREATED.value:
            "referral_verified_points",

        PointEventType.DAILY_CHALLENGE.value:
            "daily_challenge_points",

        PointEventType.WEEKLY_CHALLENGE.value:
            "weekly_challenge_points",

        PointEventType.STREAK_7_DAYS.value:
            "streak_7_points",

        PointEventType.STREAK_14_DAYS.value:
            "streak_14_points",

        PointEventType.STREAK_30_DAYS.value:
            "streak_30_points",

        PointEventType.SOCIAL_FOLLOW.value:
            "x_follow_points",

        PointEventType.SOCIAL_LIKE.value:
            "x_like_points",

        PointEventType.SOCIAL_REPOST.value:
            "x_repost_points",

        PointEventType.SOCIAL_REPLY.value:
            "x_reply_points",

        PointEventType.SOCIAL_QUOTE.value:
            "x_quote_post_points",

        PointEventType.SOCIAL_POST.value:
            "x_create_post_points",
    }

    if normalized not in CONFIGURABLE_REWARD_TYPES:
        raise ValueError(
            f"No configured reward exists for event type: {normalized}"
        )

    setting_name = reward_setting_names.get(
        normalized
    )

    if setting_name is None:
        configured_value = DEFAULT_POINT_VALUES[
            normalized
        ]
    else:
        configured_value = getattr(
            settings,
            setting_name,
            DEFAULT_POINT_VALUES[normalized],
        )

    try:
        points = int(
            configured_value
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid configured reward for event type: {normalized}"
        ) from exc

    if points < 0:
        raise ValueError(
            "Configured reward cannot be negative."
        )

    if points > MAX_POINT_ADJUSTMENT:
        raise ValueError(
            "Configured reward exceeds the maximum allowed adjustment."
        )

    return points


# =============================================================================
# DATABASE USER LOCK
# =============================================================================

def lock_user_for_points(
    db: Session,
    user_id: UUID,
) -> User:
    """
    Lock a user row before modifying User.points.

    PostgreSQL:
        SELECT ... FOR UPDATE

    This prevents concurrent requests from reading the same stale
    User.points value and overwriting one another's increments.
    """

    statement = (
        select(User)
        .where(
            User.id == user_id,
        )
        .with_for_update()
    )

    locked_user = db.scalar(
        statement
    )

    if locked_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    ensure_active_user(
        locked_user
    )

    return locked_user


# =============================================================================
# EXISTING EVENT LOOKUP
# =============================================================================

def get_existing_point_event(
    db: Session,
    *,
    user_id: UUID,
    event_type: str,
    reference_id: str | None,
) -> PointEvent | None:
    """
    Find an existing point event by its logical idempotency key.

    Reference-less events intentionally do not participate in idempotency.
    For production rewards, a reference ID should normally always be used.
    """

    if reference_id is None:
        return None

    return db.scalar(
        select(PointEvent)
        .where(
            PointEvent.user_id == user_id,
            PointEvent.event_type == event_type,
            PointEvent.reference_id == reference_id,
        )
        .limit(1)
    )


# =============================================================================
# POINT LEDGER
# =============================================================================

def award_points(
    db: Session,
    user: User,
    event_type: str | PointEventType,
    reference_id: str | UUID | None = None,
    *,
    points: int | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> PointEvent | None:
    """
    Award points to a user.

    This is the primary internal point-awarding function.

    IMPORTANT:
        This function does NOT commit.

        The caller owns the surrounding transaction.

    Idempotency:
        The logical key is:

            user_id
            event_type
            reference_id

        When reference_id is supplied, repeated requests return the
        existing PointEvent instead of awarding points again.

    Concurrency:
        The User row is locked before modifying User.points.

    Integrity race:
        A SAVEPOINT is used when creating the ledger event so a uniqueness
        race does not destroy the caller's outer transaction.
    """

    normalized_event_type = normalize_event_type(
        event_type
    )

    normalized_reference_id = normalize_reference_id(
        reference_id
    )

    normalized_description = normalize_description(
        description
    )

    # -------------------------------------------------------------------------
    # Production reward operations should normally have a reference ID.
    # -------------------------------------------------------------------------

    if (
        normalized_reference_id is None
        and normalized_event_type
        not in {
            PointEventType.MANUAL_ADJUSTMENT.value,
            PointEventType.ADMIN_ADJUSTMENT.value,
        }
    ):
        raise ValueError(
            "A reference ID is required for idempotent point rewards."
        )

    # PointEvent.reference_id is NOT NULL. Manual/admin adjustments are the
    # only events allowed without a caller-supplied reference, so give them a
    # unique one (they are intentionally not idempotent).
    if normalized_reference_id is None:
        normalized_reference_id = f"adjustment:{uuid4()}"

    # -------------------------------------------------------------------------
    # Resolve server-controlled reward.
    # -------------------------------------------------------------------------

    if points is None:
        resolved_points = get_configured_reward(
            normalized_event_type
        )
    else:
        resolved_points = int(points)

    if resolved_points <= 0:
        raise ValueError(
            "Awarded points must be greater than zero."
        )

    if resolved_points > MAX_POINT_ADJUSTMENT:
        raise ValueError(
            "Point award exceeds the maximum allowed adjustment."
        )

    # -------------------------------------------------------------------------
    # Lock the actual database user.
    #
    # Do not rely on the caller's potentially stale ORM instance.
    # -------------------------------------------------------------------------

    locked_user = lock_user_for_points(
        db,
        user.id,
    )

    # -------------------------------------------------------------------------
    # Idempotency lookup after acquiring the user lock.
    # -------------------------------------------------------------------------

    existing_event = get_existing_point_event(
        db,
        user_id=locked_user.id,
        event_type=normalized_event_type,
        reference_id=normalized_reference_id,
    )

    if existing_event is not None:
        return existing_event

    # -------------------------------------------------------------------------
    # Create immutable ledger event.
    # -------------------------------------------------------------------------

    event = PointEvent(
        user_id=locked_user.id,
        event_type=normalized_event_type,
        reference_id=normalized_reference_id,
        points=resolved_points,
        description=normalized_description,
        metadata_json=metadata or {},
        created_at=utc_now(),
    )

    # -------------------------------------------------------------------------
    # Use a SAVEPOINT.
    #
    # If another transaction inserts the same idempotency event concurrently,
    # PostgreSQL can raise IntegrityError. The nested transaction allows us
    # to recover without rolling back the caller's entire transaction.
    # -------------------------------------------------------------------------

    try:
        with db.begin_nested():
            db.add(event)
            db.flush()

    except IntegrityError:
        existing_event = get_existing_point_event(
            db,
            user_id=locked_user.id,
            event_type=normalized_event_type,
            reference_id=normalized_reference_id,
        )

        if existing_event is not None:
            return existing_event

        logger.exception(
            "Point-event integrity error without recoverable "
            "idempotency event.",
            extra={
                "user_id": str(locked_user.id),
                "event_type": normalized_event_type,
                "reference_id": normalized_reference_id,
            },
        )

        raise

    # -------------------------------------------------------------------------
    # Only update cached balance after the ledger event has successfully
    # entered the transaction.
    # -------------------------------------------------------------------------

    locked_user.points = (
        int(locked_user.points)
        + resolved_points
    )

    db.add(locked_user)

    db.flush()

    return event


# =============================================================================
# POINT DEDUCTION
# =============================================================================

def deduct_points(
    db: Session,
    user: User,
    event_type: str | PointEventType,
    points: int,
    reference_id: str | UUID,
    *,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> PointEvent | None:
    """
    Deduct points from a user's balance.

    A negative PointEvent is written to the ledger.

    Negative balances are prohibited.

    This operation does not commit.
    """

    normalized_event_type = normalize_event_type(
        event_type
    )

    normalized_reference_id = normalize_reference_id(
        reference_id
    )

    if normalized_reference_id is None:
        raise ValueError(
            "A reference ID is required for point deductions."
        )

    normalized_description = normalize_description(
        description
    )

    points = int(points)

    if points <= 0:
        raise ValueError(
            "Points to deduct must be greater than zero."
        )

    if points > MAX_POINT_ADJUSTMENT:
        raise ValueError(
            "Point deduction exceeds the maximum allowed adjustment."
        )

    # -------------------------------------------------------------------------
    # Lock balance.
    # -------------------------------------------------------------------------

    locked_user = lock_user_for_points(
        db,
        user.id,
    )

    # -------------------------------------------------------------------------
    # Idempotency.
    # -------------------------------------------------------------------------

    existing_event = get_existing_point_event(
        db,
        user_id=locked_user.id,
        event_type=normalized_event_type,
        reference_id=normalized_reference_id,
    )

    if existing_event is not None:
        return existing_event

    # -------------------------------------------------------------------------
    # Prevent negative balances.
    # -------------------------------------------------------------------------

    if int(locked_user.points) < points:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Insufficient points.",
        )

    # -------------------------------------------------------------------------
    # Create negative ledger entry.
    # -------------------------------------------------------------------------

    event = PointEvent(
        user_id=locked_user.id,
        event_type=normalized_event_type,
        reference_id=normalized_reference_id,
        points=-points,
        description=normalized_description,
        metadata_json=metadata or {},
        created_at=utc_now(),
    )

    try:
        with db.begin_nested():
            db.add(event)
            db.flush()

    except IntegrityError:
        existing_event = get_existing_point_event(
            db,
            user_id=locked_user.id,
            event_type=normalized_event_type,
            reference_id=normalized_reference_id,
        )

        if existing_event is not None:
            return existing_event

        raise

    locked_user.points = (
        int(locked_user.points)
        - points
    )

    db.add(locked_user)

    db.flush()

    return event


# =============================================================================
# WAITLIST JOIN REWARD
# =============================================================================

def award_waitlist_join_points(
    db: Session,
    user: User,
) -> PointEvent | None:
    """
    Award the initial waitlist reward.

    Idempotency reference:
        user.id
    """

    return award_points(
        db=db,
        user=user,
        event_type=PointEventType.WAITLIST_JOINED,
        reference_id=str(user.id),
        description="Coindava waitlist registration reward.",
    )


# =============================================================================
# EMAIL VERIFICATION REWARD
# =============================================================================

def award_email_verification_points(
    db: Session,
    user: User,
) -> PointEvent | None:
    """
    Award points after successful email verification.

    Idempotency reference:
        user.id
    """

    if not user.email_verified:
        return None

    return award_points(
        db=db,
        user=user,
        event_type=PointEventType.EMAIL_VERIFIED,
        reference_id=str(user.id),
        description="Email verification reward.",
    )


# =============================================================================
# PROFILE COMPLETION REWARD
# =============================================================================

def award_profile_completion_points(
    db: Session,
    user: User,
) -> PointEvent | None:
    """
    Award profile-completion points.

    Idempotency reference:
        user.id
    """

    if not user.profile_completed:
        return None

    return award_points(
        db=db,
        user=user,
        event_type=PointEventType.PROFILE_COMPLETED,
        reference_id=str(user.id),
        description="Profile completion reward.",
    )


# =============================================================================
# REFERRAL REWARD HELPERS
# =============================================================================

def award_referral_verified_points(
    db: Session,
    *,
    referrer: User,
    referral_id: UUID,
    referred_user_id: UUID,
) -> PointEvent | None:
    """
    Award the referrer reward after a referral becomes verified.

    This helper keeps referral-specific point semantics out of the
    referral API layer while still routing the actual mutation through
    award_points().
    """

    return award_points(
        db=db,
        user=referrer,
        event_type=PointEventType.REFERRAL_VERIFIED,
        reference_id=str(referral_id),
        description="Verified referral reward.",
        metadata={
            "referral_id": str(referral_id),
            "referred_user_id": str(referred_user_id),
        },
    )


def award_referred_user_points(
    db: Session,
    *,
    referred_user: User,
    referral_id: UUID,
    referrer_id: UUID,
) -> PointEvent | None:
    """
    Award the referred user's verification reward.

    The deterministic reference prevents duplicate participation rewards.
    """

    reward = get_configured_reward(
        PointEventType.REFERRAL_CREATED
    )

    if reward <= 0:
        return None

    return award_points(
        db=db,
        user=referred_user,
        event_type=PointEventType.REFERRAL_CREATED,
        reference_id=f"{referral_id}:referred_user",
        description="Verified referral participation reward.",
        metadata={
            "referral_id": str(referral_id),
            "referrer_id": str(referrer_id),
        },
    )


# =============================================================================
# SOCIAL REWARD
# =============================================================================

def award_social_points(
    db: Session,
    user: User,
    *,
    action_type: str | PointEventType,
    action_reference: str | UUID,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> PointEvent | None:
    """
    Award points for a verified social action.

    social.py is responsible for proving that the action actually occurred.

    points.py only:
        - validates the reward type
        - resolves the configured reward
        - creates the ledger event
        - updates User.points

    Supported event types:
        social_follow
        social_like
        social_repost
        social_reply
        social_quote
        social_post
    """

    normalized_action = normalize_event_type(
        action_type
    )

    if normalized_action not in SOCIAL_EVENT_TYPES:
        raise ValueError(
            f"Unsupported social reward type: {normalized_action}"
        )

    return award_points(
        db=db,
        user=user,
        event_type=normalized_action,
        reference_id=action_reference,
        description=description
        or f"Verified {normalized_action} reward.",
        metadata=metadata,
    )


# =============================================================================
# CAMPAIGN BONUS
# =============================================================================

def award_campaign_bonus(
    db: Session,
    *,
    user: User,
    campaign_reference: str | UUID,
    points: int | None = None,
    description: str = "Campaign bonus reward.",
    metadata: dict[str, Any] | None = None,
) -> PointEvent | None:
    """
    Award a campaign bonus.

    The campaign system must independently verify eligibility.

    If points is omitted, the configured campaign bonus reward is used
    when one exists.
    """

    if points is None:
        configured = getattr(
            settings,
            "campaign_bonus_points",
            DEFAULT_POINT_VALUES.get(
                PointEventType.CAMPAIGN_BONUS.value,
                0,
            ),
        )

        points = int(
            configured
        )

    if points <= 0:
        raise ValueError(
            "Campaign bonus must be greater than zero."
        )

    return award_points(
        db=db,
        user=user,
        event_type=PointEventType.CAMPAIGN_BONUS,
        reference_id=campaign_reference,
        points=points,
        description=description,
        metadata=metadata,
    )


# =============================================================================
# POINT BALANCE
# =============================================================================

def get_point_balance(
    db: Session,
    user_id: UUID,
) -> int:
    """
    Return the cached User.points balance.
    """

    user = db.get(
        User,
        user_id,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    return int(
        user.points
    )


# =============================================================================
# POINT HISTORY
# =============================================================================

def get_point_history(
    db: Session,
    user_id: UUID,
    *,
    limit: int = DEFAULT_PAGE_SIZE,
    offset: int = 0,
) -> list[PointEvent]:
    """
    Return a user's point history.

    Newest events are returned first.
    """

    if limit < 1 or limit > MAX_PAGE_SIZE:
        raise ValueError(
            f"Limit must be between 1 and {MAX_PAGE_SIZE}."
        )

    if offset < 0:
        raise ValueError(
            "Offset cannot be negative."
        )

    statement = (
        select(PointEvent)
        .where(
            PointEvent.user_id == user_id,
        )
        .order_by(
            PointEvent.created_at.desc(),
            PointEvent.id.desc(),
        )
        .limit(limit)
        .offset(offset)
    )

    return list(
        db.scalars(statement).all()
    )


# =============================================================================
# API RESPONSE MODELS
# =============================================================================

class PointEventResponse(BaseModel):
    """
    Safe point-event representation.

    Metadata is intentionally not returned because it may contain
    internal implementation information.
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    id: UUID
    event_type: str
    reference_id: str | None
    points: int
    description: str | None
    created_at: datetime

    @field_validator("event_type", mode="before")
    @classmethod
    def _event_type_value(cls, value: Any) -> Any:
        """Serialize the enum member as its plain value."""

        if isinstance(value, Enum):
            return str(value.value)

        return value


class PointBalanceResponse(BaseModel):
    """
    Current point balance.
    """

    points: int


class PointHistoryResponse(BaseModel):
    """
    Paginated point history.
    """

    points: int
    events: list[PointEventResponse]


# =============================================================================
# POINT BALANCE ENDPOINT
# =============================================================================

@router.get(
    "/me",
    response_model=PointBalanceResponse,
)
def get_my_points(
    current_user: CurrentUser,
) -> PointBalanceResponse:
    """
    Return the authenticated user's current point balance.
    """

    ensure_active_user(
        current_user
    )

    return PointBalanceResponse(
        points=int(
            current_user.points
        ),
    )


# =============================================================================
# POINT HISTORY ENDPOINT
# =============================================================================

@router.get(
    "/me/history",
    response_model=PointHistoryResponse,
)
def get_my_point_history(
    current_user: CurrentUser,
    db: Session = Depends(get_db),
    limit: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> PointHistoryResponse:
    """
    Return the authenticated user's point-event history.
    """

    ensure_active_user(
        current_user
    )

    events = get_point_history(
        db=db,
        user_id=current_user.id,
        limit=limit,
        offset=offset,
    )

    return PointHistoryResponse(
        points=int(
            current_user.points
        ),
        events=[
            PointEventResponse.model_validate(
                event
            )
            for event in events
        ],
    )


# =============================================================================
# LEDGER BALANCE
# =============================================================================

def calculate_ledger_balance(
    db: Session,
    user_id: UUID,
) -> int:
    """
    Calculate the balance directly from the immutable point ledger.

    This should be used for:
        - audits
        - reconciliation
        - integrity checks
        - administrative tooling

    It should NOT replace User.points for normal leaderboard requests.
    """

    total = db.scalar(
        select(
            func.coalesce(
                func.sum(
                    PointEvent.points
                ),
                0,
            )
        ).where(
            PointEvent.user_id == user_id,
        )
    )

    return int(
        total or 0
    )


# =============================================================================
# BALANCE VERIFICATION
# =============================================================================

def verify_point_balance(
    db: Session,
    user: User,
) -> bool:
    """
    Verify that User.points matches the immutable ledger.
    """

    ledger_balance = calculate_ledger_balance(
        db=db,
        user_id=user.id,
    )

    return (
        ledger_balance
        == int(user.points)
    )


# =============================================================================
# RECONCILIATION RESPONSE
# =============================================================================

class PointReconciliationResponse(BaseModel):
    """
    Point-ledger consistency result.
    """

    user_id: UUID
    cached_points: int
    ledger_points: int
    consistent: bool


def reconcile_user_points(
    db: Session,
    user: User,
) -> PointReconciliationResponse:
    """
    Compare the cached balance with the ledger.

    This function deliberately does not repair discrepancies automatically.

    Silent repair could hide:
        - application bugs
        - race conditions
        - database corruption
        - unauthorized modifications
    """

    ledger_balance = calculate_ledger_balance(
        db=db,
        user_id=user.id,
    )

    cached_balance = int(
        user.points
    )

    return PointReconciliationResponse(
        user_id=user.id,
        cached_points=cached_balance,
        ledger_points=ledger_balance,
        consistent=(
            cached_balance
            == ledger_balance
        ),
    )


# =============================================================================
# ADMIN MANUAL ADJUSTMENT
# =============================================================================

class ManualPointAdjustmentRequest(BaseModel):
    """
    Trusted internal/admin point adjustment request.

    This schema is intentionally not exposed through a public router
    endpoint in this module.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    points: int = Field(
        ge=-MAX_POINT_ADJUSTMENT,
        le=MAX_POINT_ADJUSTMENT,
    )

    reason: str = Field(
        min_length=5,
        max_length=MAX_DESCRIPTION_LENGTH,
    )

    reference_id: str = Field(
        min_length=1,
        max_length=MAX_REFERENCE_ID_LENGTH,
    )


def apply_manual_point_adjustment(
    db: Session,
    *,
    user: User,
    points: int,
    reason: str,
    reference_id: str,
) -> PointEvent | None:
    """
    Apply a trusted manual/admin point adjustment.

    Positive:
        Adds points.

    Negative:
        Removes points.

    The operation is idempotent using reference_id.

    No commit occurs.
    """

    ensure_active_user(
        user
    )

    points = int(
        points
    )

    if points == 0:
        raise ValueError(
            "Point adjustment cannot be zero."
        )

    if abs(points) > MAX_POINT_ADJUSTMENT:
        raise ValueError(
            "Point adjustment exceeds the allowed limit."
        )

    normalized_reference_id = normalize_reference_id(
        reference_id
    )

    if normalized_reference_id is None:
        raise ValueError(
            "Manual adjustments require a reference ID."
        )

    normalized_reason = normalize_description(
        reason
    )

    if normalized_reason is None:
        raise ValueError(
            "Manual adjustments require a reason."
        )

    # -------------------------------------------------------------------------
    # Lock user.
    # -------------------------------------------------------------------------

    locked_user = lock_user_for_points(
        db,
        user.id,
    )

    # -------------------------------------------------------------------------
    # Idempotency.
    # -------------------------------------------------------------------------

    existing_event = get_existing_point_event(
        db,
        user_id=locked_user.id,
        event_type=PointEventType.ADMIN_ADJUSTMENT.value,
        reference_id=normalized_reference_id,
    )

    if existing_event is not None:
        return existing_event

    # -------------------------------------------------------------------------
    # Prevent negative balance.
    # -------------------------------------------------------------------------

    if (
        points < 0
        and int(locked_user.points)
        < abs(points)
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Point adjustment would create a negative balance.",
        )

    # -------------------------------------------------------------------------
    # Create adjustment event.
    # -------------------------------------------------------------------------

    event = PointEvent(
        user_id=locked_user.id,
        event_type=PointEventType.ADMIN_ADJUSTMENT.value,
        reference_id=normalized_reference_id,
        points=points,
        description=normalized_reason,
        metadata_json={
            "source": "admin_adjustment",
        },
        created_at=utc_now(),
    )

    try:
        with db.begin_nested():
            db.add(event)
            db.flush()

    except IntegrityError:
        existing_event = get_existing_point_event(
            db,
            user_id=locked_user.id,
            event_type=PointEventType.ADMIN_ADJUSTMENT.value,
            reference_id=normalized_reference_id,
        )

        if existing_event is not None:
            return existing_event

        raise

    locked_user.points = (
        int(locked_user.points)
        + points
    )

    db.add(locked_user)

    db.flush()

    return event


# =============================================================================
# IMMUTABILITY GUIDANCE
# =============================================================================

def prevent_point_event_mutation() -> None:
    """
    Documentation-only marker for the ledger design.

    PointEvent records should never be edited or deleted after creation.

    To reverse a reward:

        ORIGINAL
            +200 referral_verified

        REVERSAL
            -200 admin_adjustment

    Never mutate the original +200 record.
    """

    return None


# =============================================================================
# SECURITY NOTES
# =============================================================================

"""
SECURITY MODEL
==============

1. User.points is never accepted from the frontend.

   The frontend cannot legitimately submit:

       {
           "points": 50000
       }

   and have that amount credited.


2. Public clients cannot directly invoke award_points().

   Application modules must first establish eligibility.

   Examples:

       auth.py
           -> successful email verification
           -> award_email_verification_points()

       users.py
           -> actual profile completion
           -> award_profile_completion_points()

       referrals.py
           -> verified referral
           -> award_referral_verified_points()

       social.py
           -> independently verified X action
           -> award_social_points()


3. Server-controlled rewards.

   For ordinary rewards:

       points=None

   means points.py resolves the reward from server configuration.

   A caller cannot simply request:

       award_points(..., points=999999)


   Trusted explicit point amounts are reserved for internal operations
   such as campaign bonuses and admin adjustments.


4. Idempotency.

   Every normal reward should provide a reference ID.

   Examples:

       waitlist_joined
           -> user UUID

       email_verified
           -> user UUID

       profile_completed
           -> user UUID

       referral_verified
           -> referral UUID

       referral_created
           -> referral UUID + ":referred_user"

       social_like
           -> X action/post ID

       social_repost
           -> X action/post ID

       social_reply
           -> X reply ID

       social_quote
           -> X quote-post ID

       social_post
           -> X post ID


5. Concurrent balance protection.

   User.points is a cached aggregate.

   Therefore:

       SELECT User
       FOR UPDATE

   occurs before modifying the balance.

   This prevents the classic race:

       Request A reads 100
       Request B reads 100
       Request A writes 200
       Request B writes 150

   where the expected final value should have been 250.


6. Point ledger integrity.

   Every balance mutation has a PointEvent.

   Therefore:

       User.points

   is the fast cached value, while:

       SUM(PointEvent.points)

   is the audit/reconciliation value.


7. No silent reconciliation.

   If:

       User.points = 1200

   but:

       SUM(PointEvent.points) = 1100

   the system should flag the discrepancy.

   It should not silently overwrite one value with the other.


8. Reversals.

   Never edit an existing PointEvent.

   If a 200-point reward was incorrectly issued:

       +200 referral_verified

   should remain unchanged.

   Create:

       -200 admin_adjustment

   with a reason and unique reference ID.


9. Point balances are not financial balances.

   This module represents Coindava waitlist engagement points.

   It does not represent:
       - cash
       - fiat
       - cryptocurrency
       - card balance
       - equity
       - securities
       - deposits
       - guaranteed token allocation


10. Future Founders Circle.

   If Coindava later uses points to determine access to a Founders Circle,
   allocation, benefits, or another campaign, that eligibility calculation
   should be implemented as a separate policy layer.

   The point ledger itself should remain an objective record of earned
   points.
"""


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "router",

    "PointEventType",
    "DEFAULT_POINT_VALUES",

    "award_points",
    "deduct_points",

    "award_waitlist_join_points",
    "award_email_verification_points",
    "award_profile_completion_points",

    "award_referral_verified_points",
    "award_referred_user_points",

    "award_social_points",
    "award_campaign_bonus",

    "get_configured_reward",
    "get_point_balance",
    "get_point_history",

    "calculate_ledger_balance",
    "verify_point_balance",
    "reconcile_user_points",

    "apply_manual_point_adjustment",

    "PointEventResponse",
    "PointBalanceResponse",
    "PointHistoryResponse",
    "PointReconciliationResponse",
    "ManualPointAdjustmentRequest",
]