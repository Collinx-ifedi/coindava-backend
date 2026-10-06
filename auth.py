"""
Coindava Waitlist Backend
-------------------------

Authentication and account-security module.

Responsibilities:
    - User registration
    - Password hashing and verification
    - Login
    - Session creation
    - Session validation
    - Session revocation
    - Email verification
    - Authentication dependencies
    - Account-status enforcement

Security principles:
    - Passwords are never stored in plaintext.
    - Session tokens are never stored in plaintext.
    - Email-verification tokens are never stored in plaintext.
    - Authentication secrets are generated using cryptographically secure
      randomness.
    - Passwords are hashed using Argon2id.
    - Authentication errors avoid unnecessary account enumeration.
    - Database mutations are transactional.
    - Expired/revoked sessions cannot authenticate.
    - Authentication secrets are never logged.
    - Authentication business logic is kept separate from referrals,
      points, leaderboard, and social modules.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Annotated

from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    status,
)
from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    field_validator,
)
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from email_service import send_verification_email
from models import (
    EmailVerification,
    SessionToken,
    User,
    UserStatus,
)


# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


# =============================================================================
# SECURITY CONSTANTS
# =============================================================================

SESSION_TOKEN_BYTES = 48
EMAIL_VERIFICATION_TOKEN_BYTES = 32

MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 128

USERNAME_MIN_LENGTH = 3
USERNAME_MAX_LENGTH = 32

USERNAME_PATTERN = re.compile(
    r"^[a-z0-9_]+$"
)


# =============================================================================
# PASSWORD HASHING (Argon2id)
# =============================================================================
#
# How login stays correct (read this before touching this section):
#
#   * The plaintext password is NEVER compared to the stored hash directly.
#     argon2's verify() reads the salt and cost parameters embedded in the
#     stored hash string ("$argon2id$v=19$m=...,t=...,p=...$salt$digest"),
#     re-derives a digest from the submitted password with exactly those
#     values, and compares the digests in constant time.
#
#   * Because the parameters live inside each hash, changing the ARGON2_*
#     settings later does NOT invalidate existing passwords. Old hashes keep
#     verifying with their own parameters and are silently upgraded to the
#     new parameters the next time that user logs in (see login()).
#
#   * Argon2 has no 72-byte input limit (unlike bcrypt), so no pre-hashing
#     step is needed and every character of the password counts.
#
#   * Registration and login pass the password through the SAME function
#     (_prepare_password) and nothing else in the request path trims or
#     alters it, so the value that was hashed is always the value verified.
# =============================================================================

PASSWORD_HASHER = PasswordHasher(
    time_cost=settings.argon2_time_cost,
    memory_cost=settings.argon2_memory_cost_kib,
    parallelism=settings.argon2_parallelism,
)


def _prepare_password(password: str) -> str:
    """
    Canonicalize Unicode (NFKC) before hashing/verifying.

    The same visible password can be encoded differently by different
    devices (for example "é" as one code point on Windows, or as "e" plus a
    combining accent on macOS/iOS). NFKC maps both to the same string, so a
    password set on one device still works on another. ASCII passwords are
    unaffected. Whitespace is deliberately NOT stripped.
    """

    return unicodedata.normalize("NFKC", password)


# =============================================================================
# INTERNAL EXCEPTIONS
# =============================================================================


class AuthenticationError(Exception):
    """
    Internal authentication failure.

    This exception is intentionally not exposed directly to API clients.
    """

    pass


# =============================================================================
# TIME
# =============================================================================


def utc_now() -> datetime:
    """
    Return the current timezone-aware UTC timestamp.
    """

    return datetime.now(timezone.utc)


def as_aware_utc(value: datetime) -> datetime:
    """
    Treat naive datetimes as UTC.

    PostgreSQL (timestamptz) returns aware values; SQLite returns naive ones.
    This keeps comparisons with utc_now() valid in both.
    """

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value


# =============================================================================
# NORMALIZATION
# =============================================================================


def normalize_email(email: str) -> str:
    """
    Normalize an email address.

    Only whitespace trimming and lowercase normalization are performed.

    Provider-specific transformations such as Gmail dot removal or
    plus-address stripping are intentionally avoided.
    """

    return email.strip().lower()


def normalize_username(username: str) -> str:
    """
    Normalize usernames.

    Usernames are treated as case-insensitive.
    """

    return username.strip().lower()


def normalize_referral_code(
    referral_code: str | None,
) -> str | None:
    """
    Normalize an optional referral code.
    """

    if referral_code is None:
        return None

    normalized = referral_code.strip()

    return normalized or None


# =============================================================================
# SECURE TOKEN FUNCTIONS
# =============================================================================


def generate_secure_token(
    byte_length: int,
) -> str:
    """
    Generate a cryptographically secure URL-safe token.
    """

    if byte_length <= 0:
        raise ValueError(
            "Token byte length must be greater than zero."
        )

    return secrets.token_urlsafe(
        byte_length
    )


def hash_token(
    token: str,
) -> str:
    """
    SHA-256 hash a randomly generated authentication token.

    SHA-256 is appropriate for these tokens because they are generated from
    high-entropy cryptographically secure randomness.

    This function MUST NOT be used for passwords.
    """

    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


# =============================================================================
# PASSWORD FUNCTIONS
# =============================================================================


def validate_password_strength(
    password: str,
) -> str:
    """
    Validate password strength.

    Requirements:
        - 10-128 characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one number
        - At least one special character
    """

    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"Password must contain at least "
            f"{MIN_PASSWORD_LENGTH} characters."
        )

    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(
            f"Password cannot exceed "
            f"{MAX_PASSWORD_LENGTH} characters."
        )

    if not any(
        character.isupper()
        for character in password
    ):
        raise ValueError(
            "Password must contain at least one uppercase letter."
        )

    if not any(
        character.islower()
        for character in password
    ):
        raise ValueError(
            "Password must contain at least one lowercase letter."
        )

    if not any(
        character.isdigit()
        for character in password
    ):
        raise ValueError(
            "Password must contain at least one number."
        )

    if not any(
        not character.isalnum()
        for character in password
    ):
        raise ValueError(
            "Password must contain at least one special character."
        )

    return password


def hash_password(
    password: str,
) -> str:
    """
    Hash a password with Argon2id (random salt generated per call).
    """

    return PASSWORD_HASHER.hash(
        _prepare_password(password)
    )


def verify_password(
    plain_password: str,
    password_hash: str,
) -> bool:
    """
    Verify a plaintext password against a stored Argon2 hash.

    Returns False for a wrong password and for a corrupt/unsupported hash
    (fail closed). It never raises for bad credentials.
    """

    try:
        return PASSWORD_HASHER.verify(
            password_hash,
            _prepare_password(plain_password),
        )
    except (
        VerifyMismatchError,
        VerificationError,
        InvalidHashError,
    ):
        return False


def password_needs_rehash(
    password_hash: str,
) -> bool:
    """
    True when the stored hash was made with weaker/older parameters than the
    current settings and should be upgraded at the next successful login.
    """

    try:
        return PASSWORD_HASHER.check_needs_rehash(
            password_hash
        )
    except InvalidHashError:
        return False


# =============================================================================
# DUMMY PASSWORD HASH
# =============================================================================
#
# Used when a login email does not exist so a missing account costs the same
# time as a wrong password (no account-enumeration timing side-channel).
# Generated at import time with the CURRENT parameters.
# =============================================================================

DUMMY_PASSWORD_HASH = hash_password(
    secrets.token_urlsafe(24)
)


# =============================================================================
# REQUEST / RESPONSE SCHEMAS
# =============================================================================


class RegisterRequest(BaseModel):
    """
    User registration payload.
    """

    email: EmailStr

    # Optional: when omitted a unique username is derived from the email.
    username: str | None = Field(
        default=None,
        min_length=USERNAME_MIN_LENGTH,
        max_length=USERNAME_MAX_LENGTH,
    )

    password: str = Field(
        min_length=MIN_PASSWORD_LENGTH,
        max_length=MAX_PASSWORD_LENGTH,
    )

    referral_code: str | None = Field(
        default=None,
        max_length=32,
    )

    @field_validator("email")
    @classmethod
    def validate_email(
        cls,
        value: str,
    ) -> str:
        # EmailStr is NOT a str subclass in pydantic v2 and cannot be called
        # (EmailStr("x") raises TypeError), so return the plain string.
        return normalize_email(
            str(value)
        )

    @field_validator("username")
    @classmethod
    def validate_username(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None or not value.strip():
            return None

        value = normalize_username(
            value
        )

        if not USERNAME_PATTERN.fullmatch(
            value
        ):
            raise ValueError(
                "Username may contain only "
                "letters, numbers, and underscores."
            )

        return value

    @field_validator("password")
    @classmethod
    def validate_password(
        cls,
        value: str,
    ) -> str:
        return validate_password_strength(
            value
        )

    @field_validator("referral_code")
    @classmethod
    def validate_referral_code(
        cls,
        value: str | None,
    ) -> str | None:
        return normalize_referral_code(
            value
        )


class LoginRequest(BaseModel):
    """
    Login payload.
    """

    email: EmailStr

    password: str = Field(
        min_length=1,
        max_length=MAX_PASSWORD_LENGTH,
    )

    @field_validator("email")
    @classmethod
    def normalize_login_email(
        cls,
        value: str,
    ) -> str:
        return normalize_email(
            str(value)
        )


class VerifyEmailRequest(BaseModel):
    """
    Email verification payload.
    """

    token: str = Field(
        min_length=20,
        max_length=512,
    )


class UserResponse(BaseModel):
    """
    Safe user representation.

    Sensitive fields such as password_hash are never returned.
    """

    id: str
    email: str
    username: str
    points: int
    email_verified: bool
    profile_completed: bool
    status: str
    created_at: datetime


class AuthResponse(BaseModel):
    """
    Successful authentication response.
    """

    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: UserResponse


class MessageResponse(BaseModel):
    """
    Generic API message.
    """

    message: str


# =============================================================================
# USER SERIALIZATION
# =============================================================================


def user_to_response(
    user: User,
) -> UserResponse:
    """
    Convert a User ORM object into a safe API response.
    """

    return UserResponse(
        id=str(user.id),
        email=user.email,
        username=user.username,
        points=int(user.points),
        email_verified=bool(
            user.email_verified
        ),
        profile_completed=bool(
            user.profile_completed
        ),
        status=user.status.value,
        created_at=user.created_at,
    )


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================


def create_session(
    db: Session,
    user: User,
) -> tuple[str, SessionToken, datetime]:
    """
    Create an authentication session.

    Returns:
        plaintext access token
        ORM SessionToken
        expiration timestamp

    Only the SHA-256 hash is persisted.
    """

    plaintext_token = generate_secure_token(
        SESSION_TOKEN_BYTES
    )

    token_hash = hash_token(
        plaintext_token
    )

    expires_at = (
        utc_now()
        + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    )

    session = SessionToken(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
    )

    db.add(session)

    return (
        plaintext_token,
        session,
        expires_at,
    )


def get_session_by_token(
    db: Session,
    token: str,
) -> SessionToken | None:
    """
    Resolve a plaintext bearer token into an active session.
    """

    if not token:
        return None

    token_hash = hash_token(
        token
    )

    statement = (
        select(SessionToken)
        .where(
            SessionToken.token_hash == token_hash,
            SessionToken.revoked_at.is_(None),
        )
    )

    session = db.scalar(
        statement
    )

    if session is None:
        return None

    if as_aware_utc(session.expires_at) <= utc_now():
        return None

    return session


def revoke_session(
    db: Session,
    session: SessionToken,
) -> None:
    """
    Revoke one authentication session.
    """

    if session.revoked_at is None:
        session.revoked_at = utc_now()

    db.add(session)


def revoke_all_user_sessions(
    db: Session,
    user: User,
) -> int:
    """
    Revoke all active sessions belonging to a user.

    Returns the number of sessions revoked.
    """

    statement = (
        select(SessionToken)
        .where(
            SessionToken.user_id == user.id,
            SessionToken.revoked_at.is_(None),
        )
    )

    sessions = db.scalars(
        statement
    ).all()

    now = utc_now()

    for session in sessions:
        session.revoked_at = now

    return len(sessions)


# =============================================================================
# EMAIL VERIFICATION
# =============================================================================


def create_email_verification(
    db: Session,
    user: User,
) -> str:
    """
    Create a new email verification token.

    Existing unused tokens for the same user are invalidated.

    The plaintext token is returned only to the caller so that the email
    delivery layer can send it. It is never persisted.
    """

    now = utc_now()

    existing_statement = (
        select(EmailVerification)
        .where(
            EmailVerification.user_id == user.id,
            EmailVerification.used_at.is_(None),
        )
    )

    existing_tokens = db.scalars(
        existing_statement
    ).all()

    for verification in existing_tokens:
        verification.used_at = now

    plaintext_token = generate_secure_token(
        EMAIL_VERIFICATION_TOKEN_BYTES
    )

    verification = EmailVerification(
        user_id=user.id,
        token_hash=hash_token(
            plaintext_token
        ),
        expires_at=(
            now
            + timedelta(
                minutes=settings.email_verification_expire_minutes
            )
        ),
    )

    db.add(verification)

    return plaintext_token


def verify_email_token(
    db: Session,
    token: str,
) -> User | None:
    """
    Validate and consume an email verification token.

    Returns:
        User on success.
        None on invalid/expired/already-used token.
    """

    if not token:
        return None

    token_hash = hash_token(
        token
    )

    statement = (
        select(EmailVerification)
        .where(
            EmailVerification.token_hash == token_hash,
            EmailVerification.used_at.is_(None),
        )
    )

    verification = db.scalar(
        statement
    )

    if verification is None:
        return None

    now = utc_now()

    if as_aware_utc(verification.expires_at) <= now:
        return None

    user = db.get(
        User,
        verification.user_id,
    )

    if user is None:
        return None

    if user.status in {
        UserStatus.BANNED,
        UserStatus.DELETED,
    }:
        return None

    verification.used_at = now
    user.email_verified = True

    db.add(verification)
    db.add(user)

    return user


# =============================================================================
# AUTHORIZATION HEADER
# =============================================================================


def _extract_bearer_token(
    request: Request,
) -> str | None:
    """
    Extract a Bearer token from the Authorization header.

    Expected format:

        Authorization: Bearer <token>
    """

    authorization = request.headers.get(
        "Authorization"
    )

    if not authorization:
        return None

    scheme, separator, token = (
        authorization.partition(" ")
    )

    if (
        not separator
        or scheme.lower() != "bearer"
    ):
        return None

    token = token.strip()

    if not token:
        return None

    return token


# =============================================================================
# CURRENT USER DEPENDENCY
# =============================================================================


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    """
    Resolve the authenticated user from the Bearer token.

    Raises:
        401 for missing/invalid/expired authentication.
        403 for unavailable accounts.
    """

    token = _extract_bearer_token(
        request
    )

    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    session = get_session_by_token(
        db,
        token,
    )

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication session.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    user = db.get(
        User,
        session.user_id,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication session.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

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

    return user


CurrentUser = Annotated[
    User,
    Depends(get_current_user),
]


# =============================================================================
# REGISTRATION
# =============================================================================


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    payload: RegisterRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> AuthResponse:
    """
    Register a new Coindava account.

    Authentication is responsible for creating:
        - User
        - Session
        - Email verification record

    Referral attribution remains in referrals.py.

    Point attribution remains in points.py.
    """

    email = normalize_email(
        str(payload.email)
    )

    # The username is optional: derive a unique one from the email when the
    # client did not send it.
    username = (
        normalize_username(payload.username)
        if payload.username
        else generate_username(db, email)
    )

    # -------------------------------------------------------------------------
    # Pre-checks.
    #
    # These improve UX, while database unique constraints remain the final
    # authority against race conditions.
    # -------------------------------------------------------------------------

    existing_email = db.scalar(
        select(User.id)
        .where(
            User.email == email
        )
        .limit(1)
    )

    if existing_email is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "An account with these registration "
                "details already exists."
            ),
        )

    existing_username = db.scalar(
        select(User.id)
        .where(
            User.username == username
        )
        .limit(1)
    )

    if existing_username is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "An account with these registration "
                "details already exists."
            ),
        )

    # -------------------------------------------------------------------------
    # Generate unique referral code.
    # -------------------------------------------------------------------------

    referral_code = generate_referral_code(
        db
    )

    # -------------------------------------------------------------------------
    # Create user.
    # -------------------------------------------------------------------------

    user = User(
        email=email,
        username=username,
        password_hash=hash_password(
            payload.password
        ),
        referral_code=referral_code,
        points=0,
        email_verified=False,
        profile_completed=False,
        status=UserStatus.ACTIVE,
    )

    db.add(user)

    try:
        db.flush()

        # ---------------------------------------------------------------------
        # Referral attribution + waitlist-join reward.
        #
        # Imported lazily: referrals.py and points.py import from this module
        # (CurrentUser), so a module-level import here would be circular.
        #
        # Both helpers participate in THIS transaction (they never commit),
        # so an invalid referral code fails the whole registration atomically
        # instead of silently creating an un-referred account.
        # ---------------------------------------------------------------------

        from points import award_waitlist_join_points
        from referrals import attach_registration_referral

        attach_registration_referral(
            db=db,
            referred_user=user,
            referral_code=payload.referral_code,
        )

        award_waitlist_join_points(
            db,
            user,
        )

        # ---------------------------------------------------------------------
        # Authentication session.
        # ---------------------------------------------------------------------

        (
            access_token,
            _session,
            expires_at,
        ) = create_session(
            db,
            user,
        )

        # ---------------------------------------------------------------------
        # Email verification.
        # ---------------------------------------------------------------------

        verification_token = (
            create_email_verification(
                db,
                user,
            )
        )

        # ---------------------------------------------------------------------
        # Commit atomically.
        # ---------------------------------------------------------------------

        db.commit()

    except HTTPException:
        db.rollback()
        raise

    except IntegrityError:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "An account with these registration "
                "details already exists."
            ),
        )

    except SQLAlchemyError:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account creation could not be completed.",
        )

    # -------------------------------------------------------------------------
    # The verification token is never returned to the client. It is emailed
    # after the response is sent (BackgroundTasks), so a slow mail provider
    # cannot delay or fail registration. Never log the token.
    # -------------------------------------------------------------------------

    background_tasks.add_task(
        send_verification_email,
        email,
        verification_token,
    )

    db.refresh(user)

    return AuthResponse(
        access_token=access_token,
        expires_at=expires_at,
        user=user_to_response(
            user
        ),
    )


# =============================================================================
# LOGIN
# =============================================================================


@router.post(
    "/login",
    response_model=AuthResponse,
)
def login(
    payload: LoginRequest,
    db: Session = Depends(get_db),
) -> AuthResponse:
    """
    Authenticate a user and create a session.

    Invalid credentials intentionally use the same response regardless of
    whether the email exists.
    """

    email = normalize_email(
        str(payload.email)
    )

    user = db.scalar(
        select(User)
        .where(
            User.email == email
        )
        .limit(1)
    )

    # -------------------------------------------------------------------------
    # Timing-resistant password verification.
    # -------------------------------------------------------------------------

    if user is None:
        verify_password(
            payload.password,
            DUMMY_PASSWORD_HASH,
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    if not verify_password(
        payload.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    # -------------------------------------------------------------------------
    # Account status.
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Transparent hash upgrade (only after the password was proven correct).
    # Committed together with the session below.
    # -------------------------------------------------------------------------

    if password_needs_rehash(
        user.password_hash
    ):
        user.password_hash = hash_password(
            payload.password
        )

    # -------------------------------------------------------------------------
    # Create session.
    # -------------------------------------------------------------------------

    (
        access_token,
        _session,
        expires_at,
    ) = create_session(
        db,
        user,
    )

    user.last_login_at = utc_now()

    db.add(user)

    try:
        db.commit()

    except SQLAlchemyError:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication could not be completed.",
        )

    db.refresh(user)

    return AuthResponse(
        access_token=access_token,
        expires_at=expires_at,
        user=user_to_response(
            user
        ),
    )


# =============================================================================
# CURRENT USER
# =============================================================================


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_authenticated_user(
    current_user: CurrentUser,
) -> UserResponse:
    """
    Return the currently authenticated user.
    """

    return user_to_response(
        current_user
    )


# =============================================================================
# EMAIL VERIFICATION
# =============================================================================


@router.post(
    "/verify-email",
    response_model=MessageResponse,
)
def verify_email(
    payload: VerifyEmailRequest,
    db: Session = Depends(get_db),
) -> MessageResponse:
    """
    Verify a user's email address.

    Points for email verification belong to points.py.
    """

    user = verify_email_token(
        db,
        payload.token,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token.",
        )

    try:
        # Lazy imports (circular otherwise; see register()).
        from points import award_email_verification_points
        from referrals import (
            verify_user_referral_after_email_verification,
        )

        # Email-verification reward. Idempotent (reference = user id).
        award_email_verification_points(
            db,
            user,
        )

        # If this user was referred, verifying the email verifies the
        # referral and pays both sides. Idempotent as well.
        verify_user_referral_after_email_verification(
            db,
            user,
        )

        # email_verified + rewards + referral status commit atomically.
        db.commit()

    except HTTPException:
        db.rollback()
        raise

    except SQLAlchemyError:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email verification could not be completed.",
        )

    return MessageResponse(
        message="Email address verified successfully."
    )


# =============================================================================
# RESEND EMAIL VERIFICATION
# =============================================================================


@router.post(
    "/resend-verification",
    response_model=MessageResponse,
)
def resend_verification(
    current_user: CurrentUser,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> MessageResponse:
    """
    Generate a replacement email-verification token.

    Actual delivery belongs to the email integration layer.

    The raw token is never returned by the API.
    """

    if current_user.email_verified:
        return MessageResponse(
            message=(
                "If email verification is required, "
                "a verification email will be sent."
            )
        )

    token = create_email_verification(
        db,
        current_user,
    )

    try:
        db.commit()

    except SQLAlchemyError:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unable to generate a verification request."
            ),
        )

    background_tasks.add_task(
        send_verification_email,
        current_user.email,
        token,
    )

    return MessageResponse(
        message=(
            "If email verification is required, "
            "a verification email will be sent."
        )
    )


# =============================================================================
# LOGOUT
# =============================================================================


@router.post(
    "/logout",
    response_model=MessageResponse,
)
def logout(
    request: Request,
    db: Session = Depends(get_db),
) -> MessageResponse:
    """
    Revoke the current authentication session.
    """

    token = _extract_bearer_token(
        request
    )

    if token is None:
        return MessageResponse(
            message="Logged out successfully."
        )

    session = get_session_by_token(
        db,
        token,
    )

    if session is None:
        return MessageResponse(
            message="Logged out successfully."
        )

    revoke_session(
        db,
        session,
    )

    try:
        db.commit()

    except SQLAlchemyError:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout could not be completed.",
        )

    return MessageResponse(
        message="Logged out successfully."
    )


# =============================================================================
# LOGOUT ALL DEVICES
# =============================================================================


@router.post(
    "/logout-all",
    response_model=MessageResponse,
)
def logout_all(
    current_user: CurrentUser,
    db: Session = Depends(get_db),
) -> MessageResponse:
    """
    Revoke every active session belonging to the user.
    """

    revoke_all_user_sessions(
        db,
        current_user,
    )

    try:
        db.commit()

    except SQLAlchemyError:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sessions could not be revoked.",
        )

    return MessageResponse(
        message="All active sessions have been revoked."
    )


# =============================================================================
# USERNAME GENERATION
# =============================================================================


def generate_username(
    db: Session,
    email: str,
) -> str:
    """
    Derive a unique username from an email address.

    "Jane.Doe+promo@example.com" -> "jane_doe_promo" (+ numeric suffix when
    that name is already taken). Only the local part is used, so the domain
    is never exposed on the public leaderboard.
    """

    local_part = email.split("@", 1)[0].lower()

    base = re.sub(
        r"[^a-z0-9_]+",
        "_",
        local_part,
    ).strip("_")

    if len(base) < USERNAME_MIN_LENGTH:
        base = f"user_{base}".strip("_")

    # Leave room for a "_1234" suffix inside the 32-character limit.
    base = base[: USERNAME_MAX_LENGTH - 5]

    candidate = base

    for _attempt in range(25):
        exists = db.scalar(
            select(User.id)
            .where(User.username == candidate)
            .limit(1)
        )

        if exists is None:
            return candidate

        candidate = f"{base}_{secrets.randbelow(10_000):04d}"

    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Unable to allocate a username. Please choose one.",
    )


# =============================================================================
# REFERRAL CODE GENERATION
# =============================================================================


def generate_referral_code(
    db: Session,
) -> str:
    """
    Generate a cryptographically random unique referral code.

    The database unique constraint remains the final protection against
    concurrent collision races.
    """

    alphabet = (
        "ABCDEFGHJKLMNPQRSTUVWXYZ"
        "abcdefghijkmnopqrstuvwxyz"
        "23456789"
    )

    for _ in range(20):
        code = "".join(
            secrets.choice(alphabet)
            for _ in range(
                settings.referral_code_length
            )
        )

        existing = db.scalar(
            select(User.id)
            .where(
                User.referral_code == code
            )
            .limit(1)
        )

        if existing is None:
            return code

    raise RuntimeError(
        "Unable to generate a unique referral code."
    )


# =============================================================================
# MAINTENANCE — SESSION CLEANUP
# =============================================================================


def cleanup_expired_sessions(
    db: Session,
) -> int:
    """
    Delete expired authentication sessions.

    Intended for a scheduled maintenance task.

    This should NOT run on every request.
    """

    now = utc_now()

    statement = delete(
        SessionToken
    ).where(
        SessionToken.expires_at <= now
    )

    result = db.execute(
        statement
    )

    count = int(
        result.rowcount or 0
    )

    if count:
        db.commit()

    return count


# =============================================================================
# MAINTENANCE — EMAIL VERIFICATION CLEANUP
# =============================================================================


def cleanup_expired_email_verifications(
    db: Session,
) -> int:
    """
    Delete expired or already-consumed email verification records.

    Intended for a scheduled maintenance task.
    """

    now = utc_now()

    statement = delete(
        EmailVerification
    ).where(
        (
            EmailVerification.expires_at <= now
        )
        |
        (
            EmailVerification.used_at.is_not(None)
        )
    )

    result = db.execute(
        statement
    )

    count = int(
        result.rowcount or 0
    )

    if count:
        db.commit()

    return count


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Router
    "router",

    # Schemas
    "RegisterRequest",
    "LoginRequest",
    "VerifyEmailRequest",
    "UserResponse",
    "AuthResponse",
    "MessageResponse",

    # Dependencies
    "CurrentUser",
    "get_current_user",

    # Password security
    "hash_password",
    "verify_password",
    "password_needs_rehash",
    "validate_password_strength",

    # Session management
    "create_session",
    "get_session_by_token",
    "revoke_session",
    "revoke_all_user_sessions",

    # Email verification
    "create_email_verification",
    "verify_email_token",

    # Referral code
    "generate_referral_code",

    # Maintenance
    "cleanup_expired_sessions",
    "cleanup_expired_email_verifications",
]