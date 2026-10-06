"""
Coindava Waitlist Backend
-------------------------

referrals.py

Responsibilities:
    - Create referral relationships.
    - Validate referral codes.
    - Prevent self-referrals.
    - Prevent multiple referral ownership.
    - Verify referrals.
    - Award referral points exclusively through points.py.
    - Return referral statistics.
    - Maintain idempotency.
    - Prevent referral reward manipulation.
    - Support registration-time referral attachment.
    - Support post-registration referral application where permitted.

Security principles:
    - Referral codes are server-controlled.
    - Users cannot choose or replace their referrer after a referral
      relationship has been established.
    - A user cannot refer themselves.
    - A referred account can only have one referral relationship.
    - Referral rewards are awarded only after verification.
    - Point mutations are delegated exclusively to points.py.
    - Referral mutations participate in the caller's transaction.
    - This module does not commit inside transaction-owned helpers.
    - Database uniqueness constraints remain the final concurrency guard.

Referral flow:

    User A
       |
       | shares referral code
       v
    User B registers
       |
       v
    Referral(status=PENDING)
       |
       | User B verifies email
       v
    Referral(status=VERIFIED)
       |
       +--> User A receives referral points
       |
       +--> User B receives referred-user points

Important transaction rule:

    Public route handlers own commits.

    Internal functions such as:
        attach_registration_referral()
        verify_referral()
        create_referral()

    DO NOT commit or rollback the caller's transaction.

This allows referral operations to participate safely in larger workflows,
especially registration and email verification.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Final
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from auth import CurrentUser
from config import settings
from database import get_db
from models import (
    Referral,
    ReferralStatus,
    User,
    UserStatus,
)

# =============================================================================
# LOGGER
# =============================================================================

logger = logging.getLogger("coindava.referrals")


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_REFERRAL_CODE_LENGTH: Final[int] = 32
DEFAULT_PAGE_SIZE: Final[int] = 50
MAX_PAGE_SIZE: Final[int] = 100


# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter(
    prefix="/referrals",
    tags=["Referrals"],
)


# =============================================================================
# TIME HELPERS
# =============================================================================

def utc_now() -> datetime:
    """
    Return the current timezone-aware UTC timestamp.
    """
    return datetime.now(timezone.utc)


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_referral_code(
    referral_code: str,
) -> str:
    """
    Normalize a referral code before lookup.

    Coindava-generated referral codes should use one consistent format.
    The normalization here intentionally does not alter case so that the
    database remains the authoritative source of truth.

    Args:
        referral_code: User-supplied referral code.

    Returns:
        Whitespace-trimmed referral code.
    """

    return referral_code.strip()


# =============================================================================
# ACCOUNT VALIDATION
# =============================================================================

def ensure_active_user(
    user: User,
) -> None:
    """
    Ensure a user is allowed to participate in the referral system.

    Suspended, banned, and deleted accounts cannot create or receive
    referrals.
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
# RESPONSE MODELS
# =============================================================================

class ReferralCreateRequest(BaseModel):
    """
    Request for applying a referral code after registration.

    Registration-time referral attachment is preferred because it prevents
    users from waiting until after registration to select a referrer.

    This endpoint exists for legitimate post-registration flows where the
    account has not yet been referral-locked.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    referral_code: str = Field(
        min_length=1,
        max_length=MAX_REFERRAL_CODE_LENGTH,
    )


class ReferralResponse(BaseModel):
    """
    Safe representation of a referral relationship.

    No private information belonging to the other participant is exposed.
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    id: UUID
    referral_code: str
    status: ReferralStatus
    referrer_id: UUID
    referred_user_id: UUID
    created_at: datetime
    verified_at: datetime | None


class ReferralStatsResponse(BaseModel):
    """
    Authenticated user's referral statistics.
    """

    referral_code: str
    total_referrals: int
    pending_referrals: int
    verified_referrals: int
    rejected_referrals: int


class ReferralDashboardResponse(BaseModel):
    """
    Referral information required by the frontend dashboard.
    """

    referral_code: str
    referral_link: str
    total_referrals: int
    pending_referrals: int
    verified_referrals: int
    rejected_referrals: int


class MessageResponse(BaseModel):
    """
    Generic API message.
    """

    message: str


# =============================================================================
# SERIALIZATION
# =============================================================================

def referral_to_response(
    referral: Referral,
) -> ReferralResponse:
    """
    Convert an ORM referral into a safe API response.
    """

    return ReferralResponse(
        id=referral.id,
        referral_code=referral.referral_code,
        status=referral.status,
        referrer_id=referral.referrer_id,
        referred_user_id=referral.referred_user_id,
        created_at=referral.created_at,
        verified_at=referral.verified_at,
    )


# =============================================================================
# REFERRAL LOOKUPS
# =============================================================================

def get_referrer_by_code(
    db: Session,
    referral_code: str,
) -> User | None:
    """
    Find the active user owning a referral code.

    Referral codes are server-generated and unique at the database level.
    """

    normalized_code = normalize_referral_code(
        referral_code
    )

    if not normalized_code:
        return None

    return db.scalar(
        select(User).where(
            User.referral_code == normalized_code,
            User.status == UserStatus.ACTIVE,
        )
    )


def get_referral_for_user(
    db: Session,
    user_id: UUID,
) -> Referral | None:
    """
    Return the referral relationship through which a user joined.

    The database schema should enforce uniqueness on
    Referral.referred_user_id.
    """

    return db.scalar(
        select(Referral).where(
            Referral.referred_user_id == user_id,
        )
    )


def get_referral_for_user_for_update(
    db: Session,
    user_id: UUID,
) -> Referral | None:
    """
    Retrieve a user's referral relationship with a row lock where supported.

    PostgreSQL uses SELECT ... FOR UPDATE.

    SQLite does not provide equivalent row-level locking, so the database
    uniqueness constraint remains the final concurrency protection there.
    """

    statement = (
        select(Referral)
        .where(
            Referral.referred_user_id == user_id,
        )
        .with_for_update()
    )

    return db.scalar(statement)


def get_referral_by_id(
    db: Session,
    referral_id: UUID,
) -> Referral | None:
    """
    Retrieve a referral by primary key.
    """

    return db.get(
        Referral,
        referral_id,
    )


# =============================================================================
# CREATE REFERRAL
# =============================================================================

def create_referral(
    db: Session,
    *,
    referrer: User,
    referred_user: User,
    referral_code: str,
) -> Referral:
    """
    Create a referral relationship.

    This function:
        - validates both accounts
        - prevents self-referrals
        - verifies ownership of the referral code
        - prevents multiple referral ownership
        - creates a PENDING referral

    This function does NOT:
        - award points
        - commit
        - rollback

    The caller owns the surrounding transaction.
    """

    ensure_active_user(
        referrer
    )

    ensure_active_user(
        referred_user
    )

    # -------------------------------------------------------------------------
    # Self-referral protection
    # -------------------------------------------------------------------------

    if referrer.id == referred_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot refer yourself.",
        )

    normalized_code = normalize_referral_code(
        referral_code
    )

    if not normalized_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Referral code is required.",
        )

    # -------------------------------------------------------------------------
    # Verify code ownership.
    # -------------------------------------------------------------------------

    if referrer.referral_code != normalized_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid referral code.",
        )

    # -------------------------------------------------------------------------
    # Existing referral check.
    #
    # Use a row lock where supported. The database uniqueness constraint on
    # referred_user_id remains the ultimate protection against races.
    # -------------------------------------------------------------------------

    existing_referral = get_referral_for_user_for_update(
        db,
        referred_user.id,
    )

    if existing_referral is not None:
        if existing_referral.referrer_id == referrer.id:
            # Idempotent behavior.
            return existing_referral

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This account already has a referral relationship.",
        )

    # -------------------------------------------------------------------------
    # Create referral.
    # -------------------------------------------------------------------------

    referral = Referral(
        referrer_id=referrer.id,
        referred_user_id=referred_user.id,
        referral_code=normalized_code,
        status=ReferralStatus.PENDING,
    )

    db.add(referral)

    # Do not catch IntegrityError here with db.rollback().
    #
    # Rolling back inside this helper would destroy the caller's transaction,
    # which is particularly dangerous during registration or email
    # verification.
    #
    # The caller should flush/commit and handle IntegrityError at the
    # transaction boundary.
    db.flush()

    return referral


# =============================================================================
# APPLY REFERRAL CODE
# =============================================================================

@router.post(
    "/apply",
    response_model=ReferralResponse,
    status_code=status.HTTP_201_CREATED,
)
def apply_referral_code(
    payload: ReferralCreateRequest,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
) -> ReferralResponse:
    """
    Apply a referral code to the authenticated account.

    Once a referral relationship exists, the referrer cannot be replaced.

    No referral points are awarded at this stage.
    """

    ensure_active_user(
        current_user
    )

    if not settings.referral_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Referral system is currently unavailable.",
        )

    normalized_code = normalize_referral_code(
        payload.referral_code
    )

    if not normalized_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Referral code is required.",
        )

    # -------------------------------------------------------------------------
    # Check whether this user already has a referral.
    # -------------------------------------------------------------------------

    existing_referral = get_referral_for_user_for_update(
        db,
        current_user.id,
    )

    if existing_referral is not None:
        if existing_referral.referral_code == normalized_code:
            # Idempotent response.
            return referral_to_response(
                existing_referral
            )

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This account already has a referral relationship.",
        )

    # -------------------------------------------------------------------------
    # Resolve referrer.
    # -------------------------------------------------------------------------

    referrer = get_referrer_by_code(
        db,
        normalized_code,
    )

    if referrer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid referral code.",
        )

    # -------------------------------------------------------------------------
    # Create relationship.
    # -------------------------------------------------------------------------

    try:
        referral = create_referral(
            db=db,
            referrer=referrer,
            referred_user=current_user,
            referral_code=normalized_code,
        )

        db.commit()
        db.refresh(referral)

    except HTTPException:
        db.rollback()
        raise

    except IntegrityError:
        db.rollback()

        # A concurrent request may have won the race.
        existing_referral = get_referral_for_user(
            db,
            current_user.id,
        )

        if existing_referral is not None:
            if existing_referral.referral_code == normalized_code:
                return referral_to_response(
                    existing_referral
                )

            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This account already has a referral relationship.",
            )

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Referral relationship could not be created.",
        )

    except SQLAlchemyError:
        db.rollback()

        logger.exception(
            "Database error while applying referral code.",
            extra={
                "user_id": str(current_user.id),
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to apply referral code.",
        )

    return referral_to_response(
        referral
    )


# =============================================================================
# REGISTRATION-TIME REFERRAL
# =============================================================================

def attach_registration_referral(
    db: Session,
    *,
    referred_user: User,
    referral_code: str | None,
) -> Referral | None:
    """
    Attach a referral during registration.

    Intended usage from auth.py:

        user = User(...)
        db.add(user)
        db.flush()

        attach_registration_referral(
            db=db,
            referred_user=user,
            referral_code=payload.referral_code,
        )

        db.commit()

    No commit or rollback occurs here.

    Invalid referral codes raise HTTPException so registration can fail
    atomically instead of creating an account with a silently discarded
    referral.
    """

    if not referral_code:
        return None

    if not settings.referral_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Referral codes are currently unavailable.",
        )

    ensure_active_user(
        referred_user
    )

    normalized_code = normalize_referral_code(
        referral_code
    )

    if not normalized_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Referral code is required.",
        )

    # -------------------------------------------------------------------------
    # Resolve referrer.
    # -------------------------------------------------------------------------

    referrer = get_referrer_by_code(
        db,
        normalized_code,
    )

    if referrer is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid referral code.",
        )

    # -------------------------------------------------------------------------
    # Self-referral protection.
    # -------------------------------------------------------------------------

    if referrer.id == referred_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot refer yourself.",
        )

    # -------------------------------------------------------------------------
    # Existing referral protection.
    # -------------------------------------------------------------------------

    existing_referral = get_referral_for_user_for_update(
        db,
        referred_user.id,
    )

    if existing_referral is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This account already has a referral relationship.",
        )

    return create_referral(
        db=db,
        referrer=referrer,
        referred_user=referred_user,
        referral_code=normalized_code,
    )


# =============================================================================
# VERIFY REFERRAL
# =============================================================================

def verify_referral(
    db: Session,
    *,
    referred_user: User,
) -> Referral | None:
    """
    Verify a referral after the referred user has completed the required
    verification condition.

    Current verification condition:
        referred_user.email_verified is True.

    Rewards:
        - Referrer receives settings.referral_points.
        - Referred user receives settings.referral_verified_points.

    Point mutations are delegated exclusively to points.py.

    Idempotency:
        - Already VERIFIED referrals return immediately.
        - Point events use deterministic reference IDs.
        - The point ledger therefore remains the final duplicate-reward
          protection.

    Transaction behavior:
        - No commit.
        - No rollback.

    The caller owns the transaction.
    """

    if not settings.referral_enabled:
        return None

    ensure_active_user(
        referred_user
    )

    # -------------------------------------------------------------------------
    # Lock the referral row when supported.
    # -------------------------------------------------------------------------

    referral = get_referral_for_user_for_update(
        db,
        referred_user.id,
    )

    if referral is None:
        return None

    # -------------------------------------------------------------------------
    # Idempotency.
    # -------------------------------------------------------------------------

    if referral.status == ReferralStatus.VERIFIED:
        return referral

    # Rejected referrals are terminal.
    if referral.status == ReferralStatus.REJECTED:
        return referral

    # -------------------------------------------------------------------------
    # Verification requirement.
    # -------------------------------------------------------------------------

    if not referred_user.email_verified:
        return referral

    # -------------------------------------------------------------------------
    # Resolve referrer.
    # -------------------------------------------------------------------------

    referrer = db.get(
        User,
        referral.referrer_id,
    )

    if referrer is None:
        referral.status = ReferralStatus.REJECTED
        return referral

    # A suspended referrer should not receive a reward while suspended.
    # The relationship is retained as rejected rather than automatically
    # resurrected later.
    if referrer.status != UserStatus.ACTIVE:
        referral.status = ReferralStatus.REJECTED
        return referral

    # -------------------------------------------------------------------------
    # Mark relationship verified before awarding points.
    #
    # The point ledger uses deterministic reference IDs so the reward cannot
    # be duplicated if verification is retried.
    # -------------------------------------------------------------------------

    referral.status = ReferralStatus.VERIFIED
    referral.verified_at = utc_now()

    db.add(referral)

    # -------------------------------------------------------------------------
    # Award referral-owner reward.
    # -------------------------------------------------------------------------

    from points import award_points

    if settings.referral_points > 0:
        award_points(
            db=db,
            user=referrer,
            event_type="referral_verified",
            reference_id=str(referral.id),
            points=settings.referral_points,
            description="Verified referral reward.",
            metadata={
                "referral_id": str(referral.id),
                "referred_user_id": str(referred_user.id),
            },
        )

    # -------------------------------------------------------------------------
    # Award referred-user participation reward.
    # -------------------------------------------------------------------------

    if settings.referral_verified_points > 0:
        award_points(
            db=db,
            user=referred_user,
            event_type="referral_created",
            reference_id=f"{referral.id}:referred_user",
            points=settings.referral_verified_points,
            description="Verified referral participation reward.",
            metadata={
                "referral_id": str(referral.id),
                "referrer_id": str(referrer.id),
            },
        )

    return referral


# =============================================================================
# VERIFY CURRENT USER REFERRAL
# =============================================================================

@router.post(
    "/me/verify",
    response_model=ReferralResponse | None,
)
def verify_my_referral(
    current_user: CurrentUser,
    db: Session = Depends(get_db),
) -> ReferralResponse | None:
    """
    Verify the current user's referral if the verification condition has
    been satisfied.

    In the final production flow this should normally be called internally
    immediately after successful email verification.

    The endpoint remains useful during the waitlist integration phase.
    """

    ensure_active_user(
        current_user
    )

    try:
        referral = verify_referral(
            db=db,
            referred_user=current_user,
        )

        db.commit()

        if referral is None:
            return None

        db.refresh(referral)

        return referral_to_response(
            referral
        )

    except HTTPException:
        db.rollback()
        raise

    except IntegrityError:
        db.rollback()

        logger.exception(
            "Integrity error while verifying referral.",
            extra={
                "user_id": str(current_user.id),
            },
        )

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Referral verification could not be completed.",
        )

    except SQLAlchemyError:
        db.rollback()

        logger.exception(
            "Database error while verifying referral.",
            extra={
                "user_id": str(current_user.id),
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to verify referral.",
        )


# =============================================================================
# MY REFERRAL
# =============================================================================

@router.get(
    "/me",
    response_model=ReferralResponse | None,
)
def get_my_referral(
    current_user: CurrentUser,
    db: Session = Depends(get_db),
) -> ReferralResponse | None:
    """
    Return the referral relationship through which the current user joined.
    """

    ensure_active_user(
        current_user
    )

    referral = get_referral_for_user(
        db,
        current_user.id,
    )

    if referral is None:
        return None

    return referral_to_response(
        referral
    )


# =============================================================================
# REFERRAL STATISTICS
# =============================================================================

def get_referral_statistics(
    db: Session,
    user_id: UUID,
) -> dict[str, int]:
    """
    Return referral counts for a user.

    Aggregation is performed in the database rather than loading all
    referrals into Python memory.
    """

    counts_statement = (
        select(
            Referral.status,
            func.count(Referral.id),
        )
        .where(
            Referral.referrer_id == user_id,
        )
        .group_by(
            Referral.status,
        )
    )

    rows = db.execute(
        counts_statement
    ).all()

    total_referrals = 0
    pending_referrals = 0
    verified_referrals = 0
    rejected_referrals = 0

    for referral_status, count in rows:
        count = int(count)

        total_referrals += count

        if referral_status == ReferralStatus.PENDING:
            pending_referrals = count

        elif referral_status == ReferralStatus.VERIFIED:
            verified_referrals = count

        elif referral_status == ReferralStatus.REJECTED:
            rejected_referrals = count

    return {
        "total_referrals": total_referrals,
        "pending_referrals": pending_referrals,
        "verified_referrals": verified_referrals,
        "rejected_referrals": rejected_referrals,
    }


@router.get(
    "/me/stats",
    response_model=ReferralStatsResponse,
)
def get_my_referral_stats(
    current_user: CurrentUser,
    db: Session = Depends(get_db),
) -> ReferralStatsResponse:
    """
    Return referral statistics for the authenticated user.
    """

    ensure_active_user(
        current_user
    )

    statistics = get_referral_statistics(
        db=db,
        user_id=current_user.id,
    )

    return ReferralStatsResponse(
        referral_code=current_user.referral_code,
        **statistics,
    )


# =============================================================================
# REFERRAL DASHBOARD
# =============================================================================

@router.get(
    "/me/dashboard",
    response_model=ReferralDashboardResponse,
)
def get_my_referral_dashboard(
    current_user: CurrentUser,
    db: Session = Depends(get_db),
) -> ReferralDashboardResponse:
    """
    Return referral information required by the frontend dashboard.
    """

    ensure_active_user(
        current_user
    )

    statistics = get_referral_statistics(
        db=db,
        user_id=current_user.id,
    )

    frontend_url = settings.frontend_url.rstrip("/")

    referral_link = (
        f"{frontend_url}"
        f"/?ref={current_user.referral_code}"
    )

    return ReferralDashboardResponse(
        referral_code=current_user.referral_code,
        referral_link=referral_link,
        **statistics,
    )


# =============================================================================
# REFERRAL LIST
# =============================================================================

@router.get(
    "/me/list",
    response_model=list[ReferralResponse],
)
def list_my_referrals(
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
) -> list[ReferralResponse]:
    """
    Return referrals generated by the authenticated user.

    Pagination is enforced at the API layer.
    """

    ensure_active_user(
        current_user
    )

    statement = (
        select(Referral)
        .where(
            Referral.referrer_id == current_user.id,
        )
        .order_by(
            Referral.created_at.desc(),
            Referral.id.desc(),
        )
        .limit(limit)
        .offset(offset)
    )

    referrals = db.scalars(
        statement
    ).all()

    return [
        referral_to_response(
            referral
        )
        for referral in referrals
    ]


# =============================================================================
# INTERNAL EMAIL-VERIFICATION HOOK
# =============================================================================

def verify_user_referral_after_email_verification(
    db: Session,
    user: User,
) -> Referral | None:
    """
    Verify a user's referral immediately after successful email verification.

    Intended integration:

        user.email_verified = True

        verify_user_referral_after_email_verification(
            db,
            user,
        )

        db.commit()

    This function deliberately does not commit.

    The email-verification transaction therefore becomes atomic:

        email_verified
            +
        referral VERIFIED
            +
        referral points
            +
        referred-user points

    all succeed or all roll back together.
    """

    return verify_referral(
        db=db,
        referred_user=user,
    )


# =============================================================================
# MAINTENANCE
# =============================================================================

def reject_orphaned_referrals(
    db: Session,
) -> int:
    """
    Mark pending referrals as rejected when their referrer has been
    permanently removed or banned.

    Intended for scheduled maintenance.

    Unlike the original implementation, this function does not commit
    internally. The caller owns the transaction.
    """

    statement = (
        select(Referral)
        .where(
            Referral.status == ReferralStatus.PENDING,
        )
    )

    referrals = db.scalars(
        statement
    ).all()

    rejected_count = 0

    for referral in referrals:
        referrer = db.get(
            User,
            referral.referrer_id,
        )

        if referrer is None:
            referral.status = ReferralStatus.REJECTED
            rejected_count += 1
            continue

        if referrer.status in {
            UserStatus.BANNED,
            UserStatus.DELETED,
        }:
            referral.status = ReferralStatus.REJECTED
            rejected_count += 1

    return rejected_count


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "router",
    "ReferralCreateRequest",
    "ReferralResponse",
    "ReferralStatsResponse",
    "ReferralDashboardResponse",
    "MessageResponse",
    "create_referral",
    "attach_registration_referral",
    "verify_referral",
    "verify_user_referral_after_email_verification",
    "get_referral_statistics",
    "get_referral_for_user",
    "get_referral_for_user_for_update",
    "get_referral_by_id",
    "get_referrer_by_code",
    "reject_orphaned_referrals",
]