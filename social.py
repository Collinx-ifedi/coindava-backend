"""
Coindava Waitlist Backend
social.py

Production X/Twitter social verification module.

Responsibilities
----------------
- X OAuth 2.0 Authorization Code + PKCE
- Server-side, one-time OAuth state storage
- Connect an authenticated Coindava user to X
- Encrypt X access/refresh tokens at rest
- Refresh X access tokens
- Verify X follows
- Verify X likes
- Verify X reposts
- Verify X replies
- Verify X quote posts
- Verify user-created campaign posts
- Award points only after successful verification
- Enforce idempotency
- Prevent duplicate social rewards
- Expose campaign/task endpoints
- Restrict campaign-task administration

Security model
--------------
- Coindava never asks users for X passwords.
- OAuth authorization is performed by X.
- OAuth state is stored server-side and consumed exactly once.
- PKCE verifier is stored encrypted server-side.
- X access/refresh tokens are encrypted at rest.
- The frontend never controls reward values.
- Point awards are delegated to points.py.
- Social verification is fail-closed.
- SocialAction and PointEvent records are treated as audit records.

Important
---------
The exact X API capabilities available to an application depend on the
X developer product/access level and granted scopes. This module therefore
treats provider authorization/rate-limit failures explicitly rather than
pretending that every endpoint is universally available.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlencode
from uuid import UUID

import httpx
from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from auth import get_current_user
from config import get_social_encryption_key, settings
from database import get_db
from models import (
    Campaign,
    CampaignStatus,
    CampaignTask,
    OAuthState,
    PointEventType,
    SocialAction,
    SocialActionStatus,
    SocialPlatform,
    User,
    UserStatus,
    XAccount,
)
from points import award_points


logger = logging.getLogger("coindava.social")

router = APIRouter(
    prefix="/social",
    tags=["Social"],
)


# ============================================================================
# CONSTANTS
# ============================================================================

X_API_BASE = "https://api.x.com/2"
X_AUTHORIZE_URL = "https://twitter.com/i/oauth2/authorize"
X_TOKEN_URL = "https://api.x.com/2/oauth2/token"

DEFAULT_HTTP_TIMEOUT = 15.0

OAUTH_STATE_TTL_SECONDS = 10 * 60
TOKEN_REFRESH_SKEW_SECONDS = 60

X_PAGE_SIZE = 100

# Returned when the user's X authorization is missing/expired. Deliberately
# NOT 401: the frontend treats 401 as "your Coindava session expired" and
# logs the user out, which an expired X token must not do.
X_AUTH_ERROR_STATUS = status.HTTP_424_FAILED_DEPENDENCY
X_MAX_PAGINATION_PAGES = 10

SUPPORTED_SOCIAL_ACTIONS = {
    "follow",
    "like",
    "repost",
    "reply",
    "quote_post",
    "create_post",
}

X_DEFAULT_SCOPES = (
    "tweet.read "
    "users.read "
    "follows.read "
    "like.read "
    "offline.access"
)


# ============================================================================
# ENUMS
# ============================================================================


class SocialActionType(str, Enum):
    FOLLOW = "follow"
    LIKE = "like"
    REPOST = "repost"
    REPLY = "reply"
    QUOTE = "quote_post"
    POST = "create_post"


class OAuthProvider(str, Enum):
    X = "x"


# ============================================================================
# SCHEMAS
# ============================================================================


class XConnectResponse(BaseModel):
    authorization_url: str
    state: str


class XAccountResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    provider: str
    provider_user_id: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    connected: bool


class SocialVerificationRequest(BaseModel):
    """
    Kept for frontend compatibility.

    The task ID is normally supplied in the route itself. The remaining
    fields are optional and are not trusted for reward calculation.
    """

    task_id: Optional[str] = None
    post_id: Optional[str] = None
    target_user_id: Optional[str] = None


class SocialVerificationResponse(BaseModel):
    success: bool
    verified: bool
    action: str
    points_awarded: int
    message: str


class XCallbackResponse(BaseModel):
    success: bool
    provider: str
    provider_user_id: str
    username: Optional[str] = None
    message: str


class CreateCampaignTaskRequest(BaseModel):
    campaign_id: str
    action_type: SocialActionType
    target_user_id: Optional[str] = None
    target_post_id: Optional[str] = None
    points: int = Field(default=0, ge=0, le=100_000)
    description: Optional[str] = Field(
        default=None,
        max_length=500,
    )

    # Optional campaign-post verification requirements.
    required_phrase: Optional[str] = Field(
        default=None,
        max_length=500,
    )
    required_hashtag: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    @field_validator("target_user_id", "target_post_id")
    @classmethod
    def normalize_target(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        value = value.strip()

        return value or None

    @field_validator("required_hashtag")
    @classmethod
    def normalize_hashtag(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        value = value.strip()

        if value and not value.startswith("#"):
            value = f"#{value}"

        return value or None


class CampaignTaskResponse(BaseModel):
    id: str
    campaign_id: str
    name: Optional[str] = None
    action_type: str
    target_user_id: Optional[str] = None
    target_post_id: Optional[str] = None
    target_url: Optional[str] = None
    points: int
    description: Optional[str] = None
    required_phrase: Optional[str] = None
    required_hashtag: Optional[str] = None
    completed: bool = False


class CreateCampaignRequest(BaseModel):
    name: str = Field(min_length=3, max_length=150)
    slug: str = Field(min_length=3, max_length=150, pattern=r"^[a-z0-9-]+$")
    description: Optional[str] = Field(default=None, max_length=1000)
    activate: bool = True


class CampaignResponse(BaseModel):
    id: str
    name: str
    slug: str
    status: str


# ============================================================================
# TIME / IDENTIFIER HELPERS
# ============================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _string(value: Any) -> Optional[str]:
    if value is None:
        return None

    return str(value)


def _enum_value(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)

    return str(value)


def _is_expired(value: Optional[datetime]) -> bool:
    if value is None:
        return False

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value <= utc_now()


def _require_active_user(user: User) -> None:
    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is not active.",
        )


def _parse_uuid(value: str, not_found_detail: str) -> UUID:
    """
    Parse a path/body identifier.

    Malformed IDs are reported as 404 (not 500): from the client's point of
    view such a resource simply does not exist.
    """

    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=not_found_detail,
        ) from exc


# ============================================================================
# FERNET TOKEN ENCRYPTION
# ============================================================================


def _get_fernet() -> Fernet:
    """
    Return the dedicated encryption cipher.

    SOCIAL_TOKEN_ENCRYPTION_KEY must be a valid Fernet key. It is
    independent of SECRET_KEY so token encryption and application signing
    never share key material.
    """

    key = get_social_encryption_key()

    if not key:
        raise RuntimeError(
            "SOCIAL_TOKEN_ENCRYPTION_KEY is not configured."
        )

    try:
        return Fernet(key.encode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "SOCIAL_TOKEN_ENCRYPTION_KEY is invalid."
        ) from exc


def encrypt_token(token: str) -> str:
    if not token:
        raise ValueError(
            "Cannot encrypt an empty token."
        )

    encrypted = _get_fernet().encrypt(
        token.encode("utf-8")
    )

    return encrypted.decode("utf-8")


def decrypt_token(encrypted_token: str) -> str:
    if not encrypted_token:
        raise ValueError(
            "Encrypted token is empty."
        )

    try:
        decrypted = _get_fernet().decrypt(
            encrypted_token.encode("utf-8")
        )

        return decrypted.decode("utf-8")

    except InvalidToken as exc:
        raise RuntimeError(
            "Unable to decrypt stored social token."
        ) from exc


# ============================================================================
# OAUTH / PKCE HELPERS
# ============================================================================


def _generate_oauth_state() -> str:
    return secrets.token_urlsafe(48)


def _hash_oauth_state(state: str) -> str:
    """
    Hash OAuth state before persistence.

    The raw state is returned only to the browser so it can be sent back
    to X. The database only stores the SHA-256 digest.
    """

    return hashlib.sha256(
        state.encode("utf-8")
    ).hexdigest()


def _generate_code_verifier() -> str:
    """
    RFC 7636-compatible PKCE verifier.
    """

    return secrets.token_urlsafe(64)


def _generate_code_challenge(
    code_verifier: str,
) -> str:
    digest = hashlib.sha256(
        code_verifier.encode("ascii")
    ).digest()

    return (
        base64.urlsafe_b64encode(digest)
        .decode("ascii")
        .rstrip("=")
    )


def _get_x_client_secret() -> Optional[str]:
    """
    Return the real X client secret.

    settings.x_client_secret is a pydantic SecretStr; calling str() on it
    returns the masked placeholder "**********", which makes every token
    exchange fail with an authentication error.
    """

    secret = getattr(
        settings,
        "x_client_secret",
        None,
    )

    if secret is None:
        return None

    return secret.get_secret_value() or None


def _get_x_client_id() -> str:
    if not getattr(settings, "x_enabled", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="X integration is not enabled.",
        )

    client_id = getattr(
        settings,
        "x_client_id",
        None,
    )

    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="X OAuth is not configured.",
        )

    return str(client_id)


def _get_x_redirect_uri() -> str:
    redirect_uri = getattr(
        settings,
        "x_redirect_uri",
        None,
    )

    if not redirect_uri:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="X OAuth redirect URI is not configured.",
        )

    return str(redirect_uri)


def _get_x_scopes() -> str:
    configured = getattr(
        settings,
        "x_scope",
        None,
    )

    if configured:
        return str(configured)

    return X_DEFAULT_SCOPES


# ============================================================================
# MODEL COMPATIBILITY HELPERS
# ============================================================================


def _task_action_type(task: CampaignTask) -> str:
    """
    Current architecture uses action_type.

    The fallback exists because earlier model revisions used task_type.
    """

    value = getattr(
        task,
        "action_type",
        None,
    )

    if value is None:
        value = getattr(
            task,
            "task_type",
            None,
        )

    return _enum_value(value)


def _task_target_user_id(
    task: CampaignTask,
) -> Optional[str]:
    value = getattr(
        task,
        "target_user_id",
        None,
    )

    if value is None:
        value = getattr(
            task,
            "target",
            None,
        )

    return _string(value)


def _task_target_post_id(
    task: CampaignTask,
) -> Optional[str]:
    value = getattr(
        task,
        "target_post_id",
        None,
    )

    if value is None:
        value = getattr(
            task,
            "target_tweet_id",
            None,
        )

    return _string(value)


def _task_description(
    task: CampaignTask,
) -> Optional[str]:
    return getattr(
        task,
        "description",
        None,
    )


def _task_required_phrase(
    task: CampaignTask,
) -> Optional[str]:
    return getattr(
        task,
        "required_phrase",
        None,
    )


def _task_required_hashtag(
    task: CampaignTask,
) -> Optional[str]:
    return getattr(
        task,
        "required_hashtag",
        None,
    )


def _social_action_task_id(
    action: SocialAction,
) -> Any:
    """
    Support both campaign_task_id and the earlier task_id naming.
    """

    value = getattr(
        action,
        "campaign_task_id",
        None,
    )

    if value is None:
        value = getattr(
            action,
            "task_id",
            None,
        )

    return value


def _build_social_action_kwargs(
    *,
    user: User,
    account: XAccount,
    task: CampaignTask,
    action_type: SocialActionType,
    external_id: Optional[str],
) -> dict[str, Any]:
    """
    Build SocialAction constructor kwargs for a VERIFIED action.
    """

    return {
        "user_id": user.id,
        "campaign_task_id": task.id,
        "x_account_id": account.id,
        "platform": SocialPlatform.X,
        "action_type": action_type.value,
        "status": SocialActionStatus.VERIFIED,
        "external_id": external_id,
        "verified": True,
        "verified_at": utc_now(),
    }


# ============================================================================
# OAUTH STATE STORAGE
# ============================================================================


def _create_oauth_state(
    *,
    db: Session,
    user: User,
    code_verifier: str,
    redirect_uri: str,
) -> str:
    """
    Create a cryptographically random OAuth state and persist only its hash.

    The PKCE verifier is encrypted before persistence.

    State expires after a short period and is consumed exactly once.
    """

    raw_state = _generate_oauth_state()

    state_hash = _hash_oauth_state(
        raw_state
    )

    expires_at = (
        utc_now()
        + timedelta(
            seconds=OAUTH_STATE_TTL_SECONDS
        )
    )

    oauth_state = OAuthState(
        user_id=user.id,
        provider=OAuthProvider.X.value,
        state_hash=state_hash,
        code_verifier_encrypted=encrypt_token(
            code_verifier
        ),
        redirect_uri=redirect_uri,
        expires_at=expires_at,
    )

    db.add(oauth_state)

    try:
        db.commit()

    except SQLAlchemyError:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to initialize X authorization.",
        )

    return raw_state


def _consume_oauth_state(
    *,
    db: Session,
    raw_state: str,
) -> OAuthState:
    """
    Atomically consume an OAuth state.

    This prevents replaying the same OAuth callback.
    """

    state_hash = _hash_oauth_state(
        raw_state
    )

    oauth_state = db.scalar(
        select(OAuthState)
        .where(
            OAuthState.state_hash == state_hash
        )
        .with_for_update()
    )

    if not oauth_state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state.",
        )

    if oauth_state.consumed_at is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth authorization has already been used.",
        )

    if _is_expired(
        oauth_state.expires_at
    ):
        oauth_state.consumed_at = utc_now()

        db.add(oauth_state)
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth authorization expired. Please reconnect X.",
        )

    oauth_state.consumed_at = utc_now()

    db.add(oauth_state)

    try:
        db.flush()

    except SQLAlchemyError:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to consume OAuth authorization.",
        )

    return oauth_state


# ============================================================================
# X HTTP CLIENT
# ============================================================================


async def _x_request(
    *,
    method: str,
    url: str,
    access_token: str,
    params: Optional[dict[str, Any]] = None,
    json_body: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(
            timeout=DEFAULT_HTTP_TIMEOUT,
            follow_redirects=False,
        ) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_body,
            )

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="X API request timed out.",
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to reach X API.",
        ) from exc

    if response.status_code == 401:
        raise HTTPException(
            status_code=X_AUTH_ERROR_STATUS,
            detail="X authorization expired or is invalid.",
        )

    if response.status_code == 403:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="X denied the requested operation.",
        )

    if response.status_code == 404:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Requested X resource was not found.",
        )

    if response.status_code == 429:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="X API rate limit reached. Please try again later.",
        )

    if response.status_code >= 400:
        try:
            provider_body = response.json()
        except ValueError:
            provider_body = {}

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "message": "X API returned an error.",
                "provider_status": response.status_code,
                "provider_error": provider_body,
            },
        )

    if not response.content:
        return {}

    try:
        return response.json()

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="X API returned an invalid response.",
        ) from exc


# ============================================================================
# X TOKEN REFRESH
# ============================================================================


async def _refresh_x_token(
    *,
    db: Session,
    account: XAccount,
) -> str:
    if not account.refresh_token:
        account.connected = False
        db.add(account)

        try:
            db.commit()
        except SQLAlchemyError:
            db.rollback()

        raise HTTPException(
            status_code=X_AUTH_ERROR_STATUS,
            detail="X account requires reconnection.",
        )

    try:
        refresh_token = decrypt_token(
            account.refresh_token
        )

    except RuntimeError as exc:
        account.connected = False
        db.add(account)

        try:
            db.commit()
        except SQLAlchemyError:
            db.rollback()

        raise HTTPException(
            status_code=X_AUTH_ERROR_STATUS,
            detail="Stored X authorization is invalid. Please reconnect X.",
        ) from exc

    client_id = _get_x_client_id()

    client_secret = _get_x_client_secret()

    form_data = {
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
        "client_id": client_id,
    }

    auth = None

    if client_secret:
        auth = (
            client_id,
            client_secret,
        )

    try:
        async with httpx.AsyncClient(
            timeout=DEFAULT_HTTP_TIMEOUT,
            follow_redirects=False,
        ) as client:
            response = await client.post(
                X_TOKEN_URL,
                data=form_data,
                auth=auth,
                headers={
                    "Content-Type": (
                        "application/x-www-form-urlencoded"
                    )
                },
            )

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="X token refresh timed out.",
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to refresh X authorization.",
        ) from exc

    if response.status_code >= 400:
        account.connected = False
        db.add(account)

        try:
            db.commit()
        except SQLAlchemyError:
            db.rollback()

        raise HTTPException(
            status_code=X_AUTH_ERROR_STATUS,
            detail="X authorization expired. Please reconnect X.",
        )

    try:
        token_data = response.json()

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="X returned an invalid token response.",
        ) from exc

    access_token = token_data.get(
        "access_token"
    )

    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="X did not return a new access token.",
        )

    new_refresh_token = token_data.get(
        "refresh_token"
    )

    try:
        expires_in = int(
            token_data.get(
                "expires_in",
                7200,
            )
        )
    except (TypeError, ValueError):
        expires_in = 7200

    account.access_token = encrypt_token(
        access_token
    )

    if new_refresh_token:
        account.refresh_token = encrypt_token(
            new_refresh_token
        )

    account.expires_at = (
        utc_now()
        + timedelta(
            seconds=max(
                60,
                expires_in,
            )
        )
    )

    account.connected = True

    db.add(account)

    try:
        db.commit()
        db.refresh(account)

    except SQLAlchemyError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to persist refreshed X authorization.",
        ) from exc

    return access_token


async def _get_x_access_token(
    *,
    db: Session,
    account: XAccount,
) -> str:
    if not account.connected:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X account is not connected.",
        )

    if not account.access_token:
        raise HTTPException(
            status_code=X_AUTH_ERROR_STATUS,
            detail="X access token is missing. Please reconnect X.",
        )

    if (
        account.expires_at
        and account.expires_at
        <= utc_now()
        + timedelta(
            seconds=TOKEN_REFRESH_SKEW_SECONDS
        )
    ):
        return await _refresh_x_token(
            db=db,
            account=account,
        )

    try:
        return decrypt_token(
            account.access_token
        )

    except RuntimeError as exc:
        account.connected = False
        db.add(account)

        try:
            db.commit()
        except SQLAlchemyError:
            db.rollback()

        raise HTTPException(
            status_code=X_AUTH_ERROR_STATUS,
            detail="Stored X authorization is invalid. Please reconnect X.",
        ) from exc


# ============================================================================
# X ACCOUNT LOOKUP
# ============================================================================


def _get_user_x_account(
    db: Session,
    user_id: Any,
) -> XAccount:
    account = db.scalar(
        select(XAccount)
        .where(
            XAccount.user_id == user_id,
            XAccount.provider == OAuthProvider.X.value,
            XAccount.connected.is_(True),
        )
    )

    if not account:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Connect your X account first.",
        )

    return account


# ============================================================================
# OAUTH CONNECT
# ============================================================================


@router.get(
    "/x/connect",
    response_model=XConnectResponse,
)
def connect_x(
    current_user: User = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
) -> XConnectResponse:
    """
    Start X OAuth 2.0 + PKCE.

    The state and PKCE verifier are stored server-side.
    """

    _require_active_user(current_user)

    client_id = _get_x_client_id()
    redirect_uri = _get_x_redirect_uri()

    code_verifier = _generate_code_verifier()
    code_challenge = _generate_code_challenge(
        code_verifier
    )

    raw_state = _create_oauth_state(
        db=db,
        user=current_user,
        code_verifier=code_verifier,
        redirect_uri=redirect_uri,
    )

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": _get_x_scopes(),
        "state": raw_state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }

    authorization_url = (
        f"{X_AUTHORIZE_URL}?{urlencode(params)}"
    )

    return XConnectResponse(
        authorization_url=authorization_url,
        state=raw_state,
    )


# ============================================================================
# OAUTH CALLBACK
# ============================================================================


async def _complete_x_oauth(
    *,
    code: str,
    state: str,
    db: Session,
) -> XCallbackResponse:
    """
    Complete X OAuth.

    The state is consumed before exchanging the authorization code.
    """

    oauth_state = _consume_oauth_state(
        db=db,
        raw_state=state,
    )

    user = db.scalar(
        select(User).where(
            User.id == oauth_state.user_id
        )
    )

    if not user:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Coindava user not found.",
        )

    _require_active_user(user)

    client_id = _get_x_client_id()

    redirect_uri = str(
        oauth_state.redirect_uri
    )

    code_verifier = decrypt_token(
        oauth_state.code_verifier_encrypted
    )

    token_payload = {
        "code": code,
        "grant_type": "authorization_code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }

    client_secret = _get_x_client_secret()

    auth = None

    if client_secret:
        auth = (
            client_id,
            client_secret,
        )

    try:
        async with httpx.AsyncClient(
            timeout=DEFAULT_HTTP_TIMEOUT,
            follow_redirects=False,
        ) as client:
            response = await client.post(
                X_TOKEN_URL,
                data=token_payload,
                auth=auth,
                headers={
                    "Content-Type": (
                        "application/x-www-form-urlencoded"
                    )
                },
            )

    except httpx.TimeoutException as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="X OAuth token exchange timed out.",
        ) from exc

    except httpx.RequestError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to communicate with X.",
        ) from exc

    if response.status_code >= 400:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X OAuth authorization failed.",
        )

    try:
        tokens = response.json()

    except ValueError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="X returned an invalid OAuth response.",
        ) from exc

    access_token = tokens.get(
        "access_token"
    )

    refresh_token = tokens.get(
        "refresh_token"
    )

    if not access_token:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="X did not return an access token.",
        )

    try:
        expires_in = int(
            tokens.get(
                "expires_in",
                7200,
            )
        )
    except (TypeError, ValueError):
        expires_in = 7200

    # ------------------------------------------------------------------
    # Fetch authenticated X identity.
    # ------------------------------------------------------------------

    x_user = await _x_request(
        method="GET",
        url=f"{X_API_BASE}/users/me",
        access_token=access_token,
        params={
            "user.fields": (
                "id,name,username,profile_image_url"
            )
        },
    )

    x_data = x_user.get("data")

    if not x_data or not x_data.get("id"):
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to retrieve X account information.",
        )

    provider_user_id = str(
        x_data["id"]
    )

    # ------------------------------------------------------------------
    # Prevent the same X account being connected to another user.
    # ------------------------------------------------------------------

    existing = db.scalar(
        select(XAccount)
        .where(
            XAccount.provider
            == OAuthProvider.X.value,
            XAccount.provider_user_id
            == provider_user_id,
        )
        .with_for_update()
    )

    if (
        existing
        and str(existing.user_id)
        != str(user.id)
    ):
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This X account is already connected "
                "to another Coindava account."
            ),
        )

    # ------------------------------------------------------------------
    # Reuse existing account or create it.
    # ------------------------------------------------------------------

    if existing:
        account = existing

    else:
        account = XAccount(
            user_id=user.id,
            provider=OAuthProvider.X.value,
            provider_user_id=provider_user_id,
        )

    account.access_token = encrypt_token(
        access_token
    )

    if refresh_token:
        account.refresh_token = encrypt_token(
            refresh_token
        )

    account.expires_at = (
        utc_now()
        + timedelta(
            seconds=max(
                60,
                expires_in,
            )
        )
    )

    account.username = x_data.get(
        "username"
    )

    account.display_name = x_data.get(
        "name"
    )

    account.connected = True

    db.add(account)

    try:
        db.commit()
        db.refresh(account)

    except IntegrityError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This X account could not be connected "
                "because it is already in use."
            ),
        ) from exc

    except SQLAlchemyError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to save X account connection.",
        ) from exc

    return XCallbackResponse(
        success=True,
        provider=OAuthProvider.X.value,
        provider_user_id=provider_user_id,
        username=account.username,
        message="X account connected successfully.",
    )


def _frontend_redirect(
    outcome: str,
    message: Optional[str] = None,
) -> RedirectResponse:
    """
    Send the browser back to the Coindava site with the OAuth outcome.

    X redirects the user's BROWSER to the callback, so returning JSON would
    leave the user staring at raw API output. The site reads x_status /
    x_message from the URL and shows a toast.
    """

    query = {"x_status": outcome}

    if message:
        query["x_message"] = message[:200]

    base = settings.frontend_url.rstrip("/")

    return RedirectResponse(
        url=f"{base}/?{urlencode(query)}#waitlist",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get(
    "/x/callback",
    include_in_schema=False,
)
async def x_callback(
    code: Optional[str] = Query(default=None, max_length=4096),
    state: Optional[str] = Query(default=None, max_length=4096),
    error: Optional[str] = Query(default=None, max_length=200),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    """
    OAuth redirect target registered with X (X_REDIRECT_URI).

    Intentionally unauthenticated: the browser arrives from X without the
    Coindava bearer token. The one-time server-side `state` identifies the
    user instead.
    """

    if error or not code or not state:
        return _frontend_redirect(
            "error",
            "X authorization was cancelled or denied.",
        )

    try:
        await _complete_x_oauth(
            code=code,
            state=state,
            db=db,
        )

    except HTTPException as exc:
        db.rollback()

        message = (
            exc.detail
            if isinstance(exc.detail, str)
            else "X connection failed."
        )

        return _frontend_redirect("error", message)

    except Exception:
        db.rollback()

        logger.exception("Unexpected error during X OAuth callback.")

        return _frontend_redirect(
            "error",
            "X connection failed. Please try again.",
        )

    return _frontend_redirect("connected")


# ============================================================================
# X ACCOUNT STATUS
# ============================================================================


@router.get(
    "/x/account",
    response_model=XAccountResponse,
)
def get_x_account(
    current_user: User = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
) -> XAccountResponse:
    _require_active_user(current_user)

    account = db.scalar(
        select(XAccount).where(
            XAccount.user_id
            == current_user.id,
            XAccount.provider
            == OAuthProvider.X.value,
        )
    )

    if not account:
        return XAccountResponse(
            provider=OAuthProvider.X.value,
            provider_user_id="",
            username=None,
            display_name=None,
            connected=False,
        )

    return XAccountResponse(
        provider=OAuthProvider.X.value,
        provider_user_id=str(
            account.provider_user_id
        ),
        username=account.username,
        display_name=account.display_name,
        connected=bool(
            account.connected
        ),
    )


# ============================================================================
# X USER LOOKUP
# ============================================================================


async def _get_x_user_by_username(
    *,
    access_token: str,
    username: str,
) -> Optional[dict[str, Any]]:
    username = (
        username
        .lstrip("@")
        .strip()
    )

    if not username:
        return None

    result = await _x_request(
        method="GET",
        url=(
            f"{X_API_BASE}/users/"
            f"by/username/{username}"
        ),
        access_token=access_token,
        params={
            "user.fields": (
                "id,name,username"
            )
        },
    )

    return result.get("data")


# ============================================================================
# PAGINATION HELPER
# ============================================================================


async def _paginate_x_get(
    *,
    access_token: str,
    url: str,
    params: dict[str, Any],
    item_key: str = "data",
    max_pages: int = X_MAX_PAGINATION_PAGES,
) -> list[dict[str, Any]]:
    """
    Safely paginate an X endpoint.

    The caller remains responsible for respecting API access/rate limits.
    """

    results: list[dict[str, Any]] = []

    current_params = dict(params)

    for _ in range(max_pages):
        response = await _x_request(
            method="GET",
            url=url,
            access_token=access_token,
            params=current_params,
        )

        page_items = response.get(
            item_key,
            [],
        )

        if isinstance(page_items, list):
            results.extend(
                item
                for item in page_items
                if isinstance(item, dict)
            )

        meta = response.get(
            "meta",
            {},
        )

        next_token = meta.get(
            "next_token"
        )

        if not next_token:
            break

        current_params[
            "pagination_token"
        ] = next_token

    return results


# ============================================================================
# VERIFY FOLLOW
# ============================================================================


async def _verify_follow(
    *,
    db: Session,
    account: XAccount,
    target_user_id: str,
) -> bool:
    """
    Verify that the connected X account follows target_user_id.
    """

    access_token = await _get_x_access_token(
        db=db,
        account=account,
    )

    users = await _paginate_x_get(
        access_token=access_token,
        url=(
            f"{X_API_BASE}/users/"
            f"{account.provider_user_id}/following"
        ),
        params={
            "max_results": X_PAGE_SIZE,
            "user.fields": "id,username",
        },
    )

    return any(
        str(user.get("id"))
        == str(target_user_id)
        for user in users
    )


# ============================================================================
# VERIFY LIKE
# ============================================================================


async def _verify_like(
    *,
    db: Session,
    account: XAccount,
    post_id: str,
) -> bool:
    """
    Verify whether the connected X user liked post_id.
    """

    access_token = await _get_x_access_token(
        db=db,
        account=account,
    )

    posts = await _paginate_x_get(
        access_token=access_token,
        url=(
            f"{X_API_BASE}/users/"
            f"{account.provider_user_id}/liked_tweets"
        ),
        params={
            "max_results": X_PAGE_SIZE,
            "tweet.fields": "id",
        },
    )

    return any(
        str(post.get("id"))
        == str(post_id)
        for post in posts
    )


# ============================================================================
# VERIFY REPOST
# ============================================================================


async def _verify_repost(
    *,
    db: Session,
    account: XAccount,
    post_id: str,
) -> bool:
    """
    Verify whether the connected X user reposted post_id.
    """

    access_token = await _get_x_access_token(
        db=db,
        account=account,
    )

    users = await _paginate_x_get(
        access_token=access_token,
        url=(
            f"{X_API_BASE}/tweets/"
            f"{post_id}/retweeted_by"
        ),
        params={
            "max_results": X_PAGE_SIZE,
            "user.fields": "id",
        },
    )

    return any(
        str(user.get("id"))
        == str(account.provider_user_id)
        for user in users
    )


# ============================================================================
# VERIFY REPLY
# ============================================================================


async def _verify_reply(
    *,
    db: Session,
    account: XAccount,
    target_post_id: str,
) -> Optional[dict[str, Any]]:
    """
    Find a reply authored by the connected X account in the target
    conversation.

    Returns the matching post instead of only True/False so the exact
    external X post ID can be recorded in the audit trail.
    """

    access_token = await _get_x_access_token(
        db=db,
        account=account,
    )

    if not account.username:
        return None

    query = (
        f"conversation_id:{target_post_id} "
        f"from:{account.username} "
        "-is:retweet"
    )

    response = await _x_request(
        method="GET",
        url=f"{X_API_BASE}/tweets/search/recent",
        access_token=access_token,
        params={
            "query": query,
            "max_results": X_PAGE_SIZE,
            "tweet.fields": (
                "id,author_id,conversation_id,"
                "referenced_tweets,created_at"
            ),
        },
    )

    posts = response.get(
        "data",
        [],
    )

    for post in posts:
        if str(
            post.get("author_id")
        ) != str(
            account.provider_user_id
        ):
            continue

        if str(
            post.get("conversation_id")
        ) != str(target_post_id):
            continue

        references = post.get(
            "referenced_tweets",
            [],
        )

        if any(
            str(reference.get("id"))
            == str(target_post_id)
            and reference.get("type")
            == "replied_to"
            for reference in references
        ):
            return post

    return None


# ============================================================================
# VERIFY QUOTE POST
# ============================================================================


async def _verify_quote(
    *,
    db: Session,
    account: XAccount,
    target_post_id: str,
) -> Optional[dict[str, Any]]:
    """
    Find a quote post authored by the connected X account that references
    the target post.
    """

    access_token = await _get_x_access_token(
        db=db,
        account=account,
    )

    if not account.username:
        return None

    query = (
        f"from:{account.username} "
        f"url:{target_post_id}"
    )

    response = await _x_request(
        method="GET",
        url=f"{X_API_BASE}/tweets/search/recent",
        access_token=access_token,
        params={
            "query": query,
            "max_results": X_PAGE_SIZE,
            "tweet.fields": (
                "id,author_id,referenced_tweets,"
                "created_at,text"
            ),
        },
    )

    posts = response.get(
        "data",
        [],
    )

    for post in posts:
        if str(
            post.get("author_id")
        ) != str(
            account.provider_user_id
        ):
            continue

        references = post.get(
            "referenced_tweets",
            [],
        )

        for reference in references:
            if (
                reference.get("type")
                == "quoted"
                and str(
                    reference.get("id")
                )
                == str(target_post_id)
            ):
                return post

    return None


# ============================================================================
# VERIFY USER-CREATED COINDAVA POST
# ============================================================================


async def _verify_coindava_post(
    *,
    db: Session,
    account: XAccount,
    required_phrase: Optional[str] = None,
    required_hashtag: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Find a recent public post created by the connected X account.

    Optional campaign requirements:
    - phrase
    - hashtag
    """

    access_token = await _get_x_access_token(
        db=db,
        account=account,
    )

    if not account.username:
        return None

    query_parts = [
        f"from:{account.username}",
        "-is:retweet",
    ]

    if required_hashtag:
        hashtag = (
            required_hashtag
            .strip()
            .lstrip("#")
        )

        if hashtag:
            query_parts.append(
                f"#{hashtag}"
            )

    query = " ".join(
        query_parts
    )

    response = await _x_request(
        method="GET",
        url=f"{X_API_BASE}/tweets/search/recent",
        access_token=access_token,
        params={
            "query": query,
            "max_results": X_PAGE_SIZE,
            "tweet.fields": (
                "id,text,author_id,created_at"
            ),
        },
    )

    posts = response.get(
        "data",
        [],
    )

    normalized_phrase = (
        required_phrase.casefold().strip()
        if required_phrase
        else None
    )

    normalized_hashtag = (
        required_hashtag.casefold().strip()
        if required_hashtag
        else None
    )

    for post in posts:
        if str(
            post.get("author_id")
        ) != str(
            account.provider_user_id
        ):
            continue

        text_value = str(
            post.get("text", "")
        )

        if normalized_phrase:
            if (
                normalized_phrase
                not in text_value.casefold()
            ):
                continue

        if normalized_hashtag:
            hashtag_without_hash = (
                normalized_hashtag
                .lstrip("#")
            )

            if (
                f"#{hashtag_without_hash}"
                not in text_value.casefold()
            ):
                continue

        return post

    return None


# ============================================================================
# SOCIAL ACTION IDEMPOTENCY
# ============================================================================


def _social_action_exists(
    db: Session,
    *,
    user_id: Any,
    task_id: Any,
    action_type: str,
) -> bool:
    """
    Check whether this user already completed the task.

    The unique constraint (user_id, campaign_task_id) remains the
    authoritative concurrency protection; this is an early fast-path.
    """

    return (
        db.scalar(
            select(SocialAction.id).where(
                SocialAction.user_id == user_id,
                SocialAction.campaign_task_id == task_id,
                SocialAction.verified.is_(True),
            )
        )
        is not None
    )


def _record_social_action(
    db: Session,
    *,
    user: User,
    account: XAccount,
    task: CampaignTask,
    action_type: SocialActionType,
    external_id: Optional[str],
) -> SocialAction:
    kwargs = _build_social_action_kwargs(
        user=user,
        account=account,
        task=task,
        action_type=action_type,
        external_id=external_id,
    )

    action = SocialAction(
        **kwargs
    )

    db.add(action)

    return action


# ============================================================================
# CAMPAIGN VALIDATION
# ============================================================================


def _campaign_is_active(
    campaign: Campaign,
) -> bool:
    """
    A campaign is active when its status is ACTIVE and the current time is
    inside its optional [start_at, end_at) window.
    """

    start_at = getattr(campaign, "start_at", None)
    end_at = getattr(campaign, "end_at", None)

    # Not started yet (start_at is still in the future).
    if start_at is not None and not _is_expired(start_at):
        return False

    # Already ended.
    if end_at is not None and _is_expired(end_at):
        return False

    return _campaign_status_active(campaign)


def _campaign_status_active(
    campaign: Campaign,
) -> bool:
    """
    Support status-based campaign models and an optional active flag.
    """

    active = getattr(
        campaign,
        "active",
        None,
    )

    if active is not None:
        return bool(active)

    campaign_status = getattr(
        campaign,
        "status",
        None,
    )

    if campaign_status is None:
        return True

    return _enum_value(
        campaign_status
    ).lower() == "active"


def _validate_campaign_task(
    task: CampaignTask,
    campaign: Campaign,
) -> SocialActionType:
    if not _campaign_is_active(
        campaign
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This campaign is not active.",
        )

    action_value = _task_action_type(
        task
    ).lower()

    if action_value not in SUPPORTED_SOCIAL_ACTIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported social task.",
        )

    try:
        action_type = SocialActionType(
            action_value
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported social task.",
        ) from exc

    target_user_id = _task_target_user_id(
        task
    )

    target_post_id = _task_target_post_id(
        task
    )

    if action_type == SocialActionType.FOLLOW:
        if not target_user_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Campaign configuration error: "
                    "follow task has no target user."
                ),
            )

    elif action_type in {
        SocialActionType.LIKE,
        SocialActionType.REPOST,
        SocialActionType.REPLY,
        SocialActionType.QUOTE,
    }:
        if not target_post_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Campaign configuration error: "
                    "social task has no target post."
                ),
            )

    elif action_type == SocialActionType.POST:
        # POST tasks can optionally use phrase/hashtag requirements.
        pass

    return action_type


# ============================================================================
# CENTRAL SOCIAL VERIFICATION ENGINE
# ============================================================================


async def verify_social_task(
    *,
    db: Session,
    user: User,
    task: CampaignTask,
    campaign: Optional[Campaign] = None,
) -> SocialVerificationResponse:
    """
    Verify one campaign task and award its configured reward exactly once.

    The frontend cannot specify the reward amount.

    The task's server-side points value is the only amount passed to
    points.py.
    """

    _require_active_user(user)

    if campaign is None:
        campaign = db.scalar(
            select(Campaign).where(
                Campaign.id
                == task.campaign_id
            )
        )

    if not campaign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Campaign not found.",
        )

    action_type = _validate_campaign_task(
        task,
        campaign,
    )

    account = _get_user_x_account(
        db=db,
        user_id=user.id,
    )

    # Fast idempotency path.
    if _social_action_exists(
        db,
        user_id=user.id,
        task_id=task.id,
        action_type=action_type.value,
    ):
        return SocialVerificationResponse(
            success=True,
            verified=True,
            action=action_type.value,
            points_awarded=0,
            message=(
                "This task has already been completed."
            ),
        )

    verified = False
    external_id: Optional[str] = None

    # ------------------------------------------------------------------
    # FOLLOW
    # ------------------------------------------------------------------

    if action_type == SocialActionType.FOLLOW:
        verified = await _verify_follow(
            db=db,
            account=account,
            target_user_id=(
                _task_target_user_id(task)
                or ""
            ),
        )

        external_id = (
            _task_target_user_id(task)
        )

    # ------------------------------------------------------------------
    # LIKE
    # ------------------------------------------------------------------

    elif action_type == SocialActionType.LIKE:
        verified = await _verify_like(
            db=db,
            account=account,
            post_id=(
                _task_target_post_id(task)
                or ""
            ),
        )

        external_id = (
            _task_target_post_id(task)
        )

    # ------------------------------------------------------------------
    # REPOST
    # ------------------------------------------------------------------

    elif action_type == SocialActionType.REPOST:
        verified = await _verify_repost(
            db=db,
            account=account,
            post_id=(
                _task_target_post_id(task)
                or ""
            ),
        )

        external_id = (
            _task_target_post_id(task)
        )

    # ------------------------------------------------------------------
    # REPLY
    # ------------------------------------------------------------------

    elif action_type == SocialActionType.REPLY:
        reply_post = await _verify_reply(
            db=db,
            account=account,
            target_post_id=(
                _task_target_post_id(task)
                or ""
            ),
        )

        if reply_post:
            verified = True
            external_id = _string(
                reply_post.get("id")
            )

    # ------------------------------------------------------------------
    # QUOTE
    # ------------------------------------------------------------------

    elif action_type == SocialActionType.QUOTE:
        quote_post = await _verify_quote(
            db=db,
            account=account,
            target_post_id=(
                _task_target_post_id(task)
                or ""
            ),
        )

        if quote_post:
            verified = True
            external_id = _string(
                quote_post.get("id")
            )

    # ------------------------------------------------------------------
    # USER POST
    # ------------------------------------------------------------------

    elif action_type == SocialActionType.POST:
        post = await _verify_coindava_post(
            db=db,
            account=account,
            required_phrase=(
                _task_required_phrase(task)
            ),
            required_hashtag=(
                _task_required_hashtag(task)
            ),
        )

        if post:
            verified = True
            external_id = _string(
                post.get("id")
            )

    if not verified:
        return SocialVerificationResponse(
            success=True,
            verified=False,
            action=action_type.value,
            points_awarded=0,
            message=(
                "The required X action could not "
                "be verified yet."
            ),
        )

    # ------------------------------------------------------------------
    # Resolve canonical point event type.
    # ------------------------------------------------------------------

    point_event_map = {
        SocialActionType.FOLLOW:
            PointEventType.SOCIAL_FOLLOW,

        SocialActionType.LIKE:
            PointEventType.SOCIAL_LIKE,

        SocialActionType.REPOST:
            PointEventType.SOCIAL_REPOST,

        SocialActionType.REPLY:
            PointEventType.SOCIAL_REPLY,

        SocialActionType.QUOTE:
            PointEventType.SOCIAL_QUOTE,

        SocialActionType.POST:
            PointEventType.SOCIAL_POST,
    }

    event_type = point_event_map[
        action_type
    ]

    # ------------------------------------------------------------------
    # Server-controlled reward.
    # ------------------------------------------------------------------

    task_points = int(
        getattr(
            task,
            "points",
            0,
        )
        or 0
    )

    if task_points < 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Campaign task has an invalid reward.",
        )

    if task_points > 100_000:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Campaign task reward exceeds the allowed maximum.",
        )

    point_reference = (
        f"social_task:{task.id}"
    )

    # ------------------------------------------------------------------
    # Record action + award points.
    #
    # The points.py implementation is responsible for its own
    # idempotent point ledger handling.
    #
    # We do not commit the SocialAction before award_points succeeds.
    # ------------------------------------------------------------------

    try:
        action = _record_social_action(
            db=db,
            user=user,
            account=account,
            task=task,
            action_type=action_type,
            external_id=external_id,
        )

        db.flush()

        point_event = award_points(
            db=db,
            user=user,
            event_type=event_type,
            # 0 -> None: fall back to the configured reward for this action.
            points=task_points or None,
            reference_id=point_reference,
            description=(
                f"Completed X "
                f"{action_type.value} task."
            ),
            metadata={
                "task_id": str(task.id),
                "campaign_id": str(
                    task.campaign_id
                ),
                "action_type": (
                    action_type.value
                ),
                "external_id": external_id,
            },
        )

        # award_points() returns the PointEvent (or None), not an int.
        points_awarded = (
            int(point_event.points)
            if point_event is not None
            else 0
        )

        action.points_awarded = points_awarded

        db.add(action)

        db.commit()

    except IntegrityError:
        db.rollback()

        # A concurrent request may have won the race.
        if _social_action_exists(
            db,
            user_id=user.id,
            task_id=task.id,
            action_type=action_type.value,
        ):
            return SocialVerificationResponse(
                success=True,
                verified=True,
                action=action_type.value,
                points_awarded=0,
                message=(
                    "This task has already been completed."
                ),
            )

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Social verification conflicted "
                "with another request."
            ),
        )

    except HTTPException:
        db.rollback()
        raise

    except ValueError as exc:
        # e.g. a task with no usable reward configuration.
        db.rollback()

        logger.error("Invalid social reward configuration: %s", exc)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Campaign task has an invalid reward.",
        ) from exc

    except SQLAlchemyError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to record social verification.",
        ) from exc

    return SocialVerificationResponse(
        success=True,
        verified=True,
        action=action_type.value,
        points_awarded=int(
            points_awarded
        ),
        message=(
            "Social task verified successfully."
        ),
    )


# ============================================================================
# TASK LOOKUP
# ============================================================================


def _get_task_or_404(
    db: Session,
    task_id: str,
    *,
    for_update: bool = False,
) -> CampaignTask:
    parsed = _parse_uuid(task_id, "Social task not found.")

    statement = select(CampaignTask).where(CampaignTask.id == parsed)

    if for_update:
        statement = statement.with_for_update()

    task = db.scalar(statement)

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Social task not found.",
        )

    return task


# ============================================================================
# VERIFY TASK ENDPOINT
# ============================================================================


@router.post(
    "/tasks/{task_id}/verify",
    response_model=SocialVerificationResponse,
)
async def verify_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SocialVerificationResponse:
    """
    Verify an X campaign task.

    All reward values come from the persisted campaign task.
    """

    _require_active_user(current_user)

    task = _get_task_or_404(db, task_id)

    if not task.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This task is no longer available.",
        )

    campaign = db.scalar(
        select(Campaign).where(Campaign.id == task.campaign_id)
    )

    if not campaign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Campaign not found.",
        )

    return await verify_social_task(
        db=db,
        user=current_user,
        task=task,
        campaign=campaign,
    )


@router.post(
    "/verify",
    response_model=SocialVerificationResponse,
)
async def verify_task_by_body(
    payload: SocialVerificationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SocialVerificationResponse:
    """
    Body-style alias: POST /social/verify {"task_id": "..."}.

    Only task_id is used. Every other field is ignored and the reward is
    always the server-side value.
    """

    if not payload.task_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="task_id is required.",
        )

    return await verify_task(
        task_id=payload.task_id,
        current_user=current_user,
        db=db,
    )


# ============================================================================
# USER SOCIAL ACTION HISTORY
# ============================================================================


@router.get("/actions/me")
def get_my_social_actions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    _require_active_user(current_user)

    actions = db.scalars(
        select(SocialAction)
        .where(SocialAction.user_id == current_user.id)
        .order_by(desc(SocialAction.created_at))
        .limit(limit)
    ).all()

    return {
        "actions": [
            {
                "id": str(action.id),
                "task_id": str(action.campaign_task_id),
                "action_type": _enum_value(action.action_type),
                "external_id": action.external_id,
                "verified": bool(action.verified),
                "points_awarded": int(action.points_awarded or 0),
                "verified_at": (
                    action.verified_at.isoformat()
                    if action.verified_at
                    else None
                ),
            }
            for action in actions
        ]
    }


# ============================================================================
# CAMPAIGN TASK LIST
# ============================================================================


def _serialize_campaign_task(
    task: CampaignTask,
    *,
    completed: bool = False,
) -> CampaignTaskResponse:
    return CampaignTaskResponse(
        id=str(task.id),
        campaign_id=str(task.campaign_id),
        name=task.name,
        action_type=_task_action_type(task),
        target_user_id=_task_target_user_id(task),
        target_post_id=_task_target_post_id(task),
        target_url=task.target_url,
        points=int(task.points or 0),
        description=_task_description(task),
        required_phrase=_task_required_phrase(task),
        required_hashtag=_task_required_hashtag(task),
        completed=completed,
    )


@router.get("/tasks")
def get_available_social_tasks(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    campaign_id: Optional[str] = Query(default=None, max_length=64),
) -> dict[str, Any]:
    """
    Active tasks from active campaigns, each flagged with whether the
    current user has already completed it.
    """

    _require_active_user(current_user)

    query = (
        select(CampaignTask, Campaign)
        .join(Campaign, Campaign.id == CampaignTask.campaign_id)
        .where(CampaignTask.is_active.is_(True))
    )

    if campaign_id:
        query = query.where(
            CampaignTask.campaign_id
            == _parse_uuid(campaign_id, "Campaign not found.")
        )

    rows = db.execute(
        query.order_by(CampaignTask.created_at, CampaignTask.id)
    ).all()

    completed_ids = {
        row
        for row in db.scalars(
            select(SocialAction.campaign_task_id).where(
                SocialAction.user_id == current_user.id,
                SocialAction.verified.is_(True),
            )
        ).all()
    }

    return {
        "tasks": [
            _serialize_campaign_task(
                task,
                completed=task.id in completed_ids,
            ).model_dump()
            for task, campaign in rows
            if _campaign_is_active(campaign)
        ]
    }


# ============================================================================
# ADMIN AUTHORIZATION
# ============================================================================


def _require_admin(user: User) -> None:
    """
    Restrict campaign management to administrators (User.is_admin).

    The flag cannot be set through any public endpoint; use
    `python manage.py make-admin <email>`.
    """

    if not bool(getattr(user, "is_admin", False)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required.",
        )


def _validate_task_creation_payload(
    payload: CreateCampaignTaskRequest,
) -> None:
    action = payload.action_type

    if action == SocialActionType.FOLLOW:
        if not payload.target_user_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Follow tasks require target_user_id.",
            )

    elif action in {
        SocialActionType.LIKE,
        SocialActionType.REPOST,
        SocialActionType.REPLY,
        SocialActionType.QUOTE,
    }:
        if not payload.target_post_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="This task requires target_post_id.",
            )

    elif action == SocialActionType.POST:
        if not payload.required_phrase and not payload.required_hashtag:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "POST tasks should define at least "
                    "a required phrase or hashtag."
                ),
            )


# ============================================================================
# ADMIN: CAMPAIGNS
# ============================================================================


@router.post(
    "/admin/campaigns",
    response_model=CampaignResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_campaign(
    payload: CreateCampaignRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CampaignResponse:
    """Create a campaign (admin only). Tasks are attached to campaigns."""

    _require_active_user(current_user)
    _require_admin(current_user)

    campaign = Campaign(
        name=payload.name,
        slug=payload.slug,
        description=payload.description,
        status=(
            CampaignStatus.ACTIVE
            if payload.activate
            else CampaignStatus.DRAFT
        ),
    )

    db.add(campaign)

    try:
        db.commit()
        db.refresh(campaign)

    except IntegrityError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A campaign with this slug already exists.",
        ) from exc

    except SQLAlchemyError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to create campaign.",
        ) from exc

    return CampaignResponse(
        id=str(campaign.id),
        name=campaign.name,
        slug=campaign.slug,
        status=_enum_value(campaign.status),
    )


# ============================================================================
# ADMIN: CAMPAIGN TASKS
# ============================================================================


@router.post(
    "/admin/tasks",
    response_model=CampaignTaskResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_campaign_task(
    payload: CreateCampaignTaskRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CampaignTaskResponse:
    """
    Create a social campaign task (admin only).

    Point values are stored here and are server-controlled during
    verification.
    """

    _require_active_user(current_user)
    _require_admin(current_user)
    _validate_task_creation_payload(payload)

    campaign = db.scalar(
        select(Campaign).where(
            Campaign.id
            == _parse_uuid(payload.campaign_id, "Campaign not found.")
        )
    )

    if not campaign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Campaign not found.",
        )

    if not _campaign_is_active(campaign):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tasks can only be created for an active campaign.",
        )

    action_value = payload.action_type.value

    task = CampaignTask(
        campaign_id=campaign.id,
        name=(
            payload.description
            or f"X {action_value.replace('_', ' ')} task"
        )[:150],
        description=payload.description,
        action_type=action_value,
        platform=SocialPlatform.X,
        target_user_id=payload.target_user_id,
        target_post_id=payload.target_post_id,
        required_phrase=payload.required_phrase,
        required_hashtag=payload.required_hashtag,
        points=int(payload.points),
    )

    db.add(task)

    try:
        db.commit()
        db.refresh(task)

    except IntegrityError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Unable to create campaign task because it conflicts "
                "with an existing record."
            ),
        ) from exc

    except SQLAlchemyError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to create campaign task.",
        ) from exc

    return _serialize_campaign_task(task)


@router.delete("/admin/tasks/{task_id}")
def deactivate_campaign_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Deactivate a task instead of deleting it (completed SocialAction rows
    reference it and are audit records).
    """

    _require_active_user(current_user)
    _require_admin(current_user)

    task = _get_task_or_404(db, task_id, for_update=True)

    task.is_active = False

    db.add(task)

    try:
        db.commit()

    except SQLAlchemyError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to deactivate campaign task.",
        ) from exc

    return {
        "success": True,
        "message": "Campaign task deactivated.",
        "task_id": str(task.id),
    }


# ============================================================================
# OAUTH STATE CLEANUP
# ============================================================================


def cleanup_expired_oauth_states(
    db: Session,
) -> int:
    """
    Mark expired OAuth states as consumed.

    Actual row deletion should normally be handled by a scheduled cleanup
    job once the retention period has elapsed.

    This function intentionally does not hard-delete rows because OAuth
    state records can be useful for short-lived security auditing.
    """

    now = utc_now()

    states = db.scalars(
        select(OAuthState)
        .where(
            OAuthState.expires_at <= now,
            OAuthState.consumed_at.is_(None),
        )
        .limit(1000)
    ).all()

    count = 0

    for oauth_state in states:
        oauth_state.consumed_at = now
        db.add(oauth_state)
        count += 1

    if count:
        try:
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            raise

    return count


# ============================================================================
# PUBLIC INTERNAL HELPERS
# ============================================================================


__all__ = [
    "router",
    "SocialActionType",
    "OAuthProvider",
    "XConnectResponse",
    "XAccountResponse",
    "SocialVerificationRequest",
    "SocialVerificationResponse",
    "XCallbackResponse",
    "CreateCampaignTaskRequest",
    "CampaignTaskResponse",
    "CreateCampaignRequest",
    "CampaignResponse",
    "encrypt_token",
    "decrypt_token",
    "verify_social_task",
    "cleanup_expired_oauth_states",
]