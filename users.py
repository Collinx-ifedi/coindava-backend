"""
Coindava Waitlist Backend
-------------------------

users.py

Responsibilities:
    - Retrieve the authenticated user's profile.
    - Update permitted profile fields.
    - Return waitlist membership information.
    - Expose the authenticated user's referral code.
    - Complete the user's profile.
    - Award profile-completion points through points.py.
    - Expose a privacy-safe public user profile.
    - Keep protected account fields server-controlled.

Security:
    - Protected endpoints require authentication.
    - Users can only modify their own profile.
    - Sensitive ORM fields are never returned.
    - Email is not exposed through public profiles.
    - Referral codes are not exposed through public profiles.
    - Points cannot be modified by clients.
    - Account status cannot be modified by clients.
    - Referral codes cannot be modified by clients.
    - Profile completion is idempotent.
    - Profile completion uses row-level locking to prevent concurrent
      duplicate rewards.
    - Database changes are transactional.
    - Database uniqueness remains the final authority for usernames.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from auth import CurrentUser
from database import get_db
from models import User, UserStatus


# =============================================================================
# LOGGER
# =============================================================================

logger = logging.getLogger("coindava.users")


# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


# =============================================================================
# CONSTANTS
# =============================================================================

USERNAME_MIN_LENGTH = 3
USERNAME_MAX_LENGTH = 32

USERNAME_PATTERN = re.compile(
    r"^[a-z0-9_]+$",
)


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================


class UserProfileResponse(BaseModel):
    """
    Private authenticated-user profile.

    This schema is intentionally different from the public profile schema.
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    id: UUID
    username: str
    email: str

    points: int
    email_verified: bool
    profile_completed: bool

    status: str
    referral_code: str

    created_at: datetime
    updated_at: datetime


class PublicUserProfileResponse(BaseModel):
    """
    Privacy-safe public user profile.

    Deliberately excludes:
        - email
        - referral_code
        - account internals
        - authentication information
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    id: UUID
    username: str
    points: int
    profile_completed: bool
    created_at: datetime


class UserProfileUpdateRequest(BaseModel):
    """
    Fields a user is allowed to update.

    Protected fields such as:
        - email
        - password
        - password_hash
        - points
        - email_verified
        - status
        - referral_code
        - profile_completed

    cannot be changed through this endpoint.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    username: str | None = Field(
        default=None,
        min_length=USERNAME_MIN_LENGTH,
        max_length=USERNAME_MAX_LENGTH,
    )

    @field_validator("username")
    @classmethod
    def validate_username(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.lower()

        if not USERNAME_PATTERN.fullmatch(
            normalized
        ):
            raise ValueError(
                "Username may contain only lowercase letters, "
                "numbers, and underscores."
            )

        return normalized


class WaitlistStatusResponse(BaseModel):
    """
    Authenticated user's waitlist status.
    """

    user_id: UUID
    username: str
    joined_at: datetime

    email_verified: bool

    points: int
    referral_code: str

    profile_completed: bool
    account_status: str


class ProfileCompletionResponse(BaseModel):
    """
    Result of a profile-completion operation.
    """

    message: str
    profile_completed: bool
    points: int


class ReferralCodeResponse(BaseModel):
    """
    Authenticated user's referral code.
    """

    referral_code: str


class MessageResponse(BaseModel):
    """
    Generic API message.
    """

    message: str


# =============================================================================
# RESPONSE MAPPERS
# =============================================================================


def to_user_profile(
    user: User,
) -> UserProfileResponse:
    """
    Convert a User ORM object into a private authenticated-user response.

    Sensitive fields are intentionally excluded.
    """

    return UserProfileResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        points=int(user.points),
        email_verified=bool(
            user.email_verified
        ),
        profile_completed=bool(
            user.profile_completed
        ),
        status=user.status.value,
        referral_code=user.referral_code,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


def to_public_user_profile(
    user: User,
) -> PublicUserProfileResponse:
    """
    Convert a User ORM object into a privacy-safe public profile.
    """

    return PublicUserProfileResponse(
        id=user.id,
        username=user.username,
        points=int(user.points),
        profile_completed=bool(
            user.profile_completed
        ),
        created_at=user.created_at,
    )


def to_waitlist_status(
    user: User,
) -> WaitlistStatusResponse:
    """
    Build the authenticated user's waitlist summary.
    """

    return WaitlistStatusResponse(
        user_id=user.id,
        username=user.username,
        joined_at=user.created_at,
        email_verified=bool(
            user.email_verified
        ),
        points=int(user.points),
        referral_code=user.referral_code,
        profile_completed=bool(
            user.profile_completed
        ),
        account_status=user.status.value,
    )


# =============================================================================
# ACCOUNT STATUS
# =============================================================================


def ensure_active_user(
    user: User,
) -> None:
    """
    Enforce account status for user-management operations.

    CurrentUser already performs this check, but keeping the check here
    provides defence in depth for functions called internally.
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
# DATABASE HELPERS
# =============================================================================


def get_user_by_id(
    db: Session,
    user_id: UUID,
) -> User | None:
    """
    Retrieve a user by UUID.

    No sensitive relationships are eagerly loaded.
    """

    return db.scalar(
        select(User).where(
            User.id == user_id,
        )
    )


def get_user_for_update(
    db: Session,
    user_id: UUID,
) -> User | None:
    """
    Retrieve a user using a row-level lock.

    Used for operations where duplicate concurrent updates could produce
    inconsistent state or duplicate rewards.
    """

    statement = (
        select(User)
        .where(
            User.id == user_id,
        )
        .with_for_update()
    )

    return db.scalar(
        statement
    )


# =============================================================================
# GET CURRENT USER PROFILE
# =============================================================================


@router.get(
    "/me",
    response_model=UserProfileResponse,
)
def get_my_profile(
    current_user: CurrentUser,
) -> UserProfileResponse:
    """
    Return the authenticated user's private profile.

    Endpoint:
        GET /api/v1/users/me
    """

    ensure_active_user(
        current_user,
    )

    return to_user_profile(
        current_user,
    )


# =============================================================================
# UPDATE CURRENT USER PROFILE
# =============================================================================


@router.patch(
    "/me",
    response_model=UserProfileResponse,
)
def update_my_profile(
    payload: UserProfileUpdateRequest,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
) -> UserProfileResponse:
    """
    Update permitted profile fields.

    Currently supported:
        - username

    Email changes should use a dedicated email-change workflow that verifies
    the new email address.
    """

    ensure_active_user(
        current_user,
    )

    updates = payload.model_dump(
        exclude_unset=True,
    )

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No profile fields were provided.",
        )

    new_username = updates.get(
        "username"
    )

    if new_username is None:
        return to_user_profile(
            current_user,
        )

    # -------------------------------------------------------------------------
    # No-op update.
    # -------------------------------------------------------------------------

    if new_username == current_user.username:
        return to_user_profile(
            current_user,
        )

    # -------------------------------------------------------------------------
    # Verify username is available.
    #
    # The database unique constraint remains the final authority against
    # concurrent requests.
    # -------------------------------------------------------------------------

    existing_user = db.scalar(
        select(User.id)
        .where(
            User.username == new_username,
            User.id != current_user.id,
        )
        .limit(1)
    )

    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username is already taken.",
        )

    current_user.username = new_username

    try:
        db.add(
            current_user,
        )

        db.commit()

        db.refresh(
            current_user,
        )

    except IntegrityError:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username is already taken.",
        )

    except SQLAlchemyError:
        db.rollback()

        logger.exception(
            "Database error while updating user profile",
            extra={
                "user_id": str(
                    current_user.id
                ),
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to update profile.",
        )

    return to_user_profile(
        current_user,
    )


# =============================================================================
# GET WAITLIST STATUS
# =============================================================================


@router.get(
    "/me/waitlist",
    response_model=WaitlistStatusResponse,
)
def get_my_waitlist_status(
    current_user: CurrentUser,
) -> WaitlistStatusResponse:
    """
    Return the authenticated user's waitlist status.

    Endpoint:
        GET /api/v1/users/me/waitlist
    """

    ensure_active_user(
        current_user,
    )

    return to_waitlist_status(
        current_user,
    )


# =============================================================================
# COMPLETE PROFILE
# =============================================================================


@router.post(
    "/me/complete-profile",
    response_model=ProfileCompletionResponse,
)
def complete_my_profile(
    current_user: CurrentUser,
    db: Session = Depends(get_db),
) -> ProfileCompletionResponse:
    """
    Mark the authenticated user's profile as complete.

    The operation is idempotent.

    A database row lock prevents two concurrent requests from both seeing
    profile_completed=False and attempting to award the same reward.

    Point awarding is delegated to points.py.
    """

    ensure_active_user(
        current_user,
    )

    # -------------------------------------------------------------------------
    # Lock the user row.
    # -------------------------------------------------------------------------

    user = get_user_for_update(
        db,
        current_user.id,
    )

    if user is None:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    ensure_active_user(
        user,
    )

    # -------------------------------------------------------------------------
    # Idempotency.
    # -------------------------------------------------------------------------

    if user.profile_completed:
        db.rollback()

        return ProfileCompletionResponse(
            message="Profile already completed.",
            profile_completed=True,
            points=int(user.points),
        )

    # -------------------------------------------------------------------------
    # Mark completion.
    # -------------------------------------------------------------------------

    user.profile_completed = True

    db.add(
        user
    )

    try:
        db.flush()

        # ---------------------------------------------------------------------
        # Import locally to prevent circular imports.
        # ---------------------------------------------------------------------

        from points import award_points

        # ---------------------------------------------------------------------
        # The points service owns:
        #
        #   - PointEvent creation
        #   - idempotency
        #   - User.points mutation
        #   - point-event validation
        #
        # The reference ID is deterministic, so the same logical event
        # cannot legitimately receive a second reward.
        # ---------------------------------------------------------------------

        award_points(
            db=db,
            user=user,
            event_type="profile_completed",
            reference_id=str(
                user.id
            ),
        )

        db.commit()

        db.refresh(
            user
        )

    except IntegrityError:
        db.rollback()

        logger.exception(
            "Integrity error while completing profile",
            extra={
                "user_id": str(
                    user.id
                ),
            },
        )

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Profile completion could not be recorded.",
        )

    except HTTPException:
        db.rollback()
        raise

    except SQLAlchemyError:
        db.rollback()

        logger.exception(
            "Database error while completing profile",
            extra={
                "user_id": str(
                    user.id
                ),
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to complete profile.",
        )

    except Exception:
        db.rollback()

        logger.exception(
            "Unexpected error while completing profile",
            extra={
                "user_id": str(
                    user.id
                ),
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to complete profile.",
        )

    return ProfileCompletionResponse(
        message="Profile completed successfully.",
        profile_completed=True,
        points=int(
            user.points
        ),
    )


# =============================================================================
# GET MY REFERRAL CODE
# =============================================================================


@router.get(
    "/me/referral-code",
    response_model=ReferralCodeResponse,
)
def get_my_referral_code(
    current_user: CurrentUser,
) -> ReferralCodeResponse:
    """
    Return the authenticated user's referral code.

    Referral codes are intentionally unavailable through public profiles.
    """

    ensure_active_user(
        current_user,
    )

    return ReferralCodeResponse(
        referral_code=current_user.referral_code,
    )


# =============================================================================
# GET PUBLIC USER PROFILE
# =============================================================================


@router.get(
    "/{user_id}",
    response_model=PublicUserProfileResponse,
)
def get_public_user_profile(
    user_id: UUID,
    db: Session = Depends(get_db),
) -> PublicUserProfileResponse:
    """
    Retrieve a privacy-safe public profile by user UUID.

    Public profiles deliberately do not expose:
        - email
        - referral code
        - account status
        - authentication information
        - internal security fields
    """

    user = get_user_by_id(
        db,
        user_id,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    # Suspended, banned, and deleted users should not be publicly discoverable.
    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    return to_public_user_profile(
        user,
    )


# =============================================================================
# PROFILE COMPLETION REQUIREMENTS
# =============================================================================


def is_profile_complete(
    user: User,
) -> bool:
    """
    Determine whether the user's required profile fields are populated.

    This helper is intentionally conservative.

    Registration already guarantees:
        - email
        - username
        - password

    Email verification is deliberately NOT treated as profile completion.
    """

    return bool(
        user.email
        and user.username
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Router
    "router",

    # Schemas
    "UserProfileResponse",
    "PublicUserProfileResponse",
    "UserProfileUpdateRequest",
    "WaitlistStatusResponse",
    "ProfileCompletionResponse",
    "ReferralCodeResponse",
    "MessageResponse",

    # Helpers
    "get_user_by_id",
    "ensure_active_user",
    "is_profile_complete",

    # Endpoints
    "get_my_profile",
    "update_my_profile",
    "get_my_waitlist_status",
    "complete_my_profile",
    "get_my_referral_code",
    "get_public_user_profile",
]