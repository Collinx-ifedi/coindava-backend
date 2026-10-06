"""
Coindava Waitlist Backend
leaderboard.py

Production leaderboard service.

Responsibilities:
- Global leaderboard
- Current user's rank
- Top-N leaderboard
- Paginated leaderboard
- User position lookup
- Tie-safe ranking
- Active-user filtering
- Efficient database queries
- Server-side ranking only

Canonical ranking rules:
1. Higher points rank higher.
2. If two users have equal points, the user with the earlier
   account creation timestamp ranks higher.
3. If creation timestamps are equal, UUID is used as the
   deterministic final tie-breaker.

Security rules:
- Only active users participate in the leaderboard.
- Public endpoints expose only leaderboard-safe information.
- Private fields such as email, password hashes, session tokens,
  referral codes and social tokens are never serialized.
- The frontend never calculates authoritative rank.
- Rank is always calculated by the backend.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import and_, asc, desc, func, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from models import User, UserStatus


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/leaderboard",
    tags=["Leaderboard"],
)


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100
TOP_100_LIMIT = 100
MAX_AROUND_RADIUS = 10


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class LeaderboardUser(BaseModel):
    """
    Public leaderboard representation.

    Intentionally excludes:
    - email
    - password hash
    - session tokens
    - referral code
    - account status
    - X credentials
    - private profile information
    """

    rank: int
    user_id: str
    username: str
    points: int
    joined_at: datetime
    is_top_100: bool

    model_config = ConfigDict(
        from_attributes=True,
    )


class LeaderboardResponse(BaseModel):
    """
    Paginated leaderboard response.
    """

    leaderboard: list[LeaderboardUser]
    total_users: int
    limit: int
    offset: int
    has_more: bool


class MyLeaderboardResponse(BaseModel):
    """
    Exact leaderboard position for a specific user.
    """

    user_id: str
    username: str
    points: int
    rank: int
    total_users: int
    is_top_100: bool
    percentile: float


class LeaderboardAroundMeResponse(BaseModel):
    """
    Current user plus nearby leaderboard positions.
    """

    user: LeaderboardUser
    above: list[LeaderboardUser]
    below: list[LeaderboardUser]


class Top100Response(BaseModel):
    """
    Top 100 leaderboard response.
    """

    leaderboard: list[LeaderboardUser]
    total_users: int
    top_100_count: int


class LeaderboardStatsResponse(BaseModel):
    """
    Public aggregate leaderboard statistics.
    """

    total_users: int
    highest_points: int
    average_points: float
    top_100_size: int


class Top100StatusResponse(BaseModel):
    """
    Lightweight dashboard response.
    """

    rank: int
    points: int
    is_top_100: bool


# ============================================================================
# USER VALIDATION
# ============================================================================


def _require_active_user(user: User) -> None:
    """
    Ensure the supplied user is eligible for leaderboard participation.

    Inactive, suspended, pending or otherwise non-active users are excluded
    from ranking.
    """

    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is not active.",
        )


# ============================================================================
# CANONICAL RANKING ORDER
# ============================================================================


def _ranking_order():
    """
    Return the canonical leaderboard ordering.

    Ordering:
        1. points DESC
        2. created_at ASC
        3. id ASC

    Every leaderboard query MUST use this ordering.

    Keeping the ordering centralized prevents subtle differences between:
    - global leaderboard
    - top 100
    - rank calculation
    - users-around-me
    - public user position
    """

    return (
        desc(User.points),
        asc(User.created_at),
        asc(User.id),
    )


# ============================================================================
# ACTIVE USER COUNT
# ============================================================================


def _active_user_count(db: Session) -> int:
    """
    Return the number of active users participating in the leaderboard.
    """

    count = db.scalar(
        select(func.count(User.id)).where(
            User.status == UserStatus.ACTIVE,
        )
    )

    return int(count or 0)


# ============================================================================
# USER REFRESH
# ============================================================================


def _get_fresh_active_user(
    db: Session,
    user: User,
) -> User:
    """
    Retrieve the latest database state for an authenticated user.

    This is important because the User object supplied by FastAPI's
    dependency may have been loaded before another transaction awarded
    points.

    Leaderboard calculations must operate on the current database values.
    """

    user_id = user.id

    fresh_user = db.scalar(
        select(User).where(
            User.id == user_id,
            User.status == UserStatus.ACTIVE,
        )
    )

    if fresh_user is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is not active.",
        )

    return fresh_user


# ============================================================================
# EXACT RANK CALCULATION
# ============================================================================


def get_user_rank(
    db: Session,
    user: User,
) -> int:
    """
    Calculate the user's exact deterministic leaderboard rank.

    A user ranks ahead of the target user when:

        another.points > target.points

    OR

        another.points == target.points
        AND another.created_at < target.created_at

    OR

        another.points == target.points
        AND another.created_at == target.created_at
        AND another.id < target.id

    Rank is therefore:

        number_of_users_ahead + 1

    This implementation deliberately uses scalar values rather than
    comparing User columns against themselves, avoiding ambiguous SQL
    expressions such as:

        User.points == user.points
    """

    _require_active_user(user)

    points = int(user.points or 0)
    created_at = user.created_at
    user_id = user.id

    ahead_condition = or_(
        User.points > points,

        and_(
            User.points == points,
            User.created_at < created_at,
        ),

        and_(
            User.points == points,
            User.created_at == created_at,
            User.id < user_id,
        ),
    )

    users_ahead = db.scalar(
        select(func.count(User.id)).where(
            User.status == UserStatus.ACTIVE,
            ahead_condition,
        )
    )

    return int(users_ahead or 0) + 1


# ============================================================================
# BACKWARD-COMPATIBLE RANK FUNCTION
# ============================================================================


def calculate_user_rank(
    db: Session,
    user: User,
) -> int:
    """
    Backward-compatible alias for callers that use calculate_user_rank().

    The authoritative implementation is get_user_rank().
    """

    return get_user_rank(
        db=db,
        user=user,
    )


# ============================================================================
# PERCENTILE
# ============================================================================


def _calculate_percentile(
    rank: int,
    total_users: int,
) -> float:
    """
    Calculate the user's leaderboard percentile.

    Interpretation:

        rank 1 among 100 users -> 100.00
        rank 50 among 100 users -> 50.51
        rank 100 among 100 users -> 0.00

    A single user receives 100.00.

    The value is clamped to [0, 100] to protect the public API from
    accidental numerical anomalies.
    """

    if total_users <= 1:
        return 100.0

    percentile = (
        (total_users - rank)
        / (total_users - 1)
    ) * 100.0

    return round(
        max(
            0.0,
            min(
                100.0,
                percentile,
            ),
        ),
        2,
    )


# ============================================================================
# PUBLIC SERIALIZATION
# ============================================================================


def _serialize_user(
    user: User,
    rank: int,
) -> LeaderboardUser:
    """
    Convert a User ORM object into a safe public leaderboard object.

    Never return the ORM object directly from a public endpoint.
    """

    return LeaderboardUser(
        rank=rank,
        user_id=str(user.id),
        username=user.username,
        points=int(user.points or 0),
        joined_at=user.created_at,
        is_top_100=rank <= TOP_100_LIMIT,
    )


# ============================================================================
# PAGINATED LEADERBOARD
# ============================================================================


def get_leaderboard(
    db: Session,
    *,
    limit: int = DEFAULT_PAGE_SIZE,
    offset: int = 0,
) -> LeaderboardResponse:
    """
    Return a page of the global leaderboard.

    Ranking is entirely server-side.

    The query retrieves one additional record so that has_more can be
    determined without requiring a second page-count query.
    """

    limit = min(
        max(int(limit), 1),
        MAX_PAGE_SIZE,
    )

    offset = max(
        int(offset),
        0,
    )

    total_users = _active_user_count(db)

    users = db.scalars(
        select(User)
        .where(
            User.status == UserStatus.ACTIVE,
        )
        .order_by(
            *_ranking_order(),
        )
        .offset(offset)
        .limit(limit + 1)
    ).all()

    has_more = len(users) > limit

    users = users[:limit]

    leaderboard = [
        _serialize_user(
            user=user,
            rank=offset + index + 1,
        )
        for index, user in enumerate(users)
    ]

    return LeaderboardResponse(
        leaderboard=leaderboard,
        total_users=total_users,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


# ============================================================================
# TOP 100
# ============================================================================


def get_top_100(
    db: Session,
) -> Top100Response:
    """
    Return the current top 100 active users.
    """

    total_users = _active_user_count(db)

    users = db.scalars(
        select(User)
        .where(
            User.status == UserStatus.ACTIVE,
        )
        .order_by(
            *_ranking_order(),
        )
        .limit(TOP_100_LIMIT)
    ).all()

    leaderboard = [
        _serialize_user(
            user=user,
            rank=index + 1,
        )
        for index, user in enumerate(users)
    ]

    return Top100Response(
        leaderboard=leaderboard,
        total_users=total_users,
        top_100_count=len(leaderboard),
    )


# ============================================================================
# CURRENT USER POSITION
# ============================================================================


def get_my_leaderboard_position(
    db: Session,
    user: User,
) -> MyLeaderboardResponse:
    """
    Return the current user's exact leaderboard position.

    The user is refreshed from the database first so that the calculation
    reflects the latest points value.
    """

    _require_active_user(user)

    fresh_user = _get_fresh_active_user(
        db=db,
        user=user,
    )

    rank = get_user_rank(
        db=db,
        user=fresh_user,
    )

    total_users = _active_user_count(db)

    percentile = _calculate_percentile(
        rank=rank,
        total_users=total_users,
    )

    return MyLeaderboardResponse(
        user_id=str(fresh_user.id),
        username=fresh_user.username,
        points=int(fresh_user.points or 0),
        rank=rank,
        total_users=total_users,
        is_top_100=rank <= TOP_100_LIMIT,
        percentile=percentile,
    )


# ============================================================================
# USERS AROUND CURRENT USER
# ============================================================================


def get_users_around_me(
    db: Session,
    user: User,
    *,
    radius: int = 2,
) -> LeaderboardAroundMeResponse:
    """
    Return users immediately above and below the current user.

    Example:

        rank 137

        above:
            135
            136

        user:
            137

        below:
            138
            139

    The same canonical ordering used by the global leaderboard is used here.
    """

    _require_active_user(user)

    fresh_user = _get_fresh_active_user(
        db=db,
        user=user,
    )

    radius = min(
        max(int(radius), 1),
        MAX_AROUND_RADIUS,
    )

    rank = get_user_rank(
        db=db,
        user=fresh_user,
    )

    # ------------------------------------------------------------------------
    # Users above
    # ------------------------------------------------------------------------

    above_offset = max(
        rank - radius - 1,
        0,
    )

    above_users = db.scalars(
        select(User)
        .where(
            User.status == UserStatus.ACTIVE,
        )
        .order_by(
            *_ranking_order(),
        )
        .offset(above_offset)
        .limit(radius),
    ).all()

    above: list[LeaderboardUser] = []

    for index, candidate in enumerate(above_users):
        candidate_rank = (
            above_offset
            + index
            + 1
        )

        if candidate.id == fresh_user.id:
            continue

        if candidate_rank < rank:
            above.append(
                _serialize_user(
                    user=candidate,
                    rank=candidate_rank,
                )
            )

    # ------------------------------------------------------------------------
    # Users below
    # ------------------------------------------------------------------------

    below_users = db.scalars(
        select(User)
        .where(
            User.status == UserStatus.ACTIVE,
        )
        .order_by(
            *_ranking_order(),
        )
        .offset(rank)
        .limit(radius),
    ).all()

    below = [
        _serialize_user(
            user=candidate,
            rank=rank + index + 1,
        )
        for index, candidate in enumerate(below_users)
    ]

    current_user_entry = _serialize_user(
        user=fresh_user,
        rank=rank,
    )

    return LeaderboardAroundMeResponse(
        user=current_user_entry,
        above=above,
        below=below,
    )


# ============================================================================
# PUBLIC USER POSITION
# ============================================================================


def _parse_user_id(user_id: str) -> UUID:
    """
    Validate a public user UUID before sending it to the database.

    This provides:
    - consistent 400 responses
    - early rejection of malformed IDs
    - cleaner database queries
    """

    try:
        return UUID(str(user_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID.",
        ) from exc


def get_public_user_position(
    db: Session,
    user_id: str,
) -> MyLeaderboardResponse:
    """
    Return the public leaderboard position of an active user.
    """

    parsed_user_id = _parse_user_id(user_id)

    user = db.scalar(
        select(User).where(
            User.id == parsed_user_id,
            User.status == UserStatus.ACTIVE,
        )
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Leaderboard user not found.",
        )

    rank = get_user_rank(
        db=db,
        user=user,
    )

    total_users = _active_user_count(db)

    percentile = _calculate_percentile(
        rank=rank,
        total_users=total_users,
    )

    return MyLeaderboardResponse(
        user_id=str(user.id),
        username=user.username,
        points=int(user.points or 0),
        rank=rank,
        total_users=total_users,
        is_top_100=rank <= TOP_100_LIMIT,
        percentile=percentile,
    )


# ============================================================================
# FAST TOP-100 ENDPOINT
# ============================================================================


@router.get(
    "/top-100",
    response_model=Top100Response,
)
def top_100(
    db: Session = Depends(get_db),
) -> Top100Response:
    """
    Public endpoint for the current top 100 users.
    """

    try:
        return get_top_100(
            db=db,
        )

    except HTTPException:
        raise

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load leaderboard.",
        ) from exc


# ============================================================================
# GLOBAL LEADERBOARD ENDPOINT
# ============================================================================


@router.get(
    "",
    response_model=LeaderboardResponse,
)
def leaderboard(
    limit: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description="Number of leaderboard users to return.",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of leaderboard users to skip.",
    ),
    db: Session = Depends(get_db),
) -> LeaderboardResponse:
    """
    Public paginated leaderboard.
    """

    try:
        return get_leaderboard(
            db=db,
            limit=limit,
            offset=offset,
        )

    except HTTPException:
        raise

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load leaderboard.",
        ) from exc


# ============================================================================
# CURRENT USER RANK ENDPOINT
# ============================================================================


@router.get(
    "/me",
    response_model=MyLeaderboardResponse,
)
def my_rank(
    current_user: User = Depends(
        get_current_user,
    ),
    db: Session = Depends(get_db),
) -> MyLeaderboardResponse:
    """
    Return the authenticated user's current leaderboard position.
    """

    try:
        return get_my_leaderboard_position(
            db=db,
            user=current_user,
        )

    except HTTPException:
        raise

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to calculate your leaderboard position.",
        ) from exc


# ============================================================================
# USERS AROUND ME ENDPOINT
# ============================================================================


@router.get(
    "/me/around",
    response_model=LeaderboardAroundMeResponse,
)
def around_me(
    radius: int = Query(
        default=2,
        ge=1,
        le=MAX_AROUND_RADIUS,
        description="Number of users to return above and below you.",
    ),
    current_user: User = Depends(
        get_current_user,
    ),
    db: Session = Depends(get_db),
) -> LeaderboardAroundMeResponse:
    """
    Return the authenticated user and nearby leaderboard positions.
    """

    try:
        return get_users_around_me(
            db=db,
            user=current_user,
            radius=radius,
        )

    except HTTPException:
        raise

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load nearby leaderboard positions.",
        ) from exc


# ============================================================================
# PUBLIC USER POSITION ENDPOINT
# ============================================================================


@router.get(
    "/user/{user_id}",
    response_model=MyLeaderboardResponse,
)
def get_user_position(
    user_id: str,
    db: Session = Depends(get_db),
) -> MyLeaderboardResponse:
    """
    Return the public leaderboard position of an active user.

    Only leaderboard-safe information is returned.
    """

    try:
        return get_public_user_position(
            db=db,
            user_id=user_id,
        )

    except HTTPException:
        raise

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to calculate leaderboard position.",
        ) from exc


# ============================================================================
# CURRENT USER TOP-100 STATUS
# ============================================================================


@router.get(
    "/me/top-100",
    response_model=Top100StatusResponse,
)
def my_top_100_status(
    current_user: User = Depends(
        get_current_user,
    ),
    db: Session = Depends(get_db),
) -> Top100StatusResponse:
    """
    Lightweight endpoint for the dashboard.

    Example:

        {
            "rank": 42,
            "points": 1850,
            "is_top_100": true
        }
    """

    try:
        fresh_user = _get_fresh_active_user(
            db=db,
            user=current_user,
        )

        rank = get_user_rank(
            db=db,
            user=fresh_user,
        )

        return Top100StatusResponse(
            rank=rank,
            points=int(fresh_user.points or 0),
            is_top_100=rank <= TOP_100_LIMIT,
        )

    except HTTPException:
        raise

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to calculate leaderboard status.",
        ) from exc


# ============================================================================
# LEADERBOARD STATISTICS
# ============================================================================


@router.get(
    "/stats",
    response_model=LeaderboardStatsResponse,
)
def leaderboard_stats(
    db: Session = Depends(get_db),
) -> LeaderboardStatsResponse:
    """
    Public aggregate leaderboard statistics.

    No private user information is returned.
    """

    try:
        total_users = _active_user_count(db)

        highest_points = db.scalar(
            select(
                func.max(User.points),
            ).where(
                User.status == UserStatus.ACTIVE,
            )
        )

        average_points = db.scalar(
            select(
                func.avg(User.points),
            ).where(
                User.status == UserStatus.ACTIVE,
            )
        )

        return LeaderboardStatsResponse(
            total_users=total_users,
            highest_points=int(
                highest_points or 0,
            ),
            average_points=round(
                float(
                    average_points or 0,
                ),
                2,
            ),
            top_100_size=min(
                total_users,
                TOP_100_LIMIT,
            ),
        )

    except HTTPException:
        raise

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load leaderboard statistics.",
        ) from exc


# ============================================================================
# INTERNAL LEADERBOARD HELPERS
# ============================================================================


def is_user_in_top_100(
    db: Session,
    user: User,
) -> bool:
    """
    Internal helper used by other backend modules.

    Returns True when the supplied active user currently occupies a
    top-100 position.
    """

    _require_active_user(user)

    fresh_user = _get_fresh_active_user(
        db=db,
        user=user,
    )

    rank = get_user_rank(
        db=db,
        user=fresh_user,
    )

    return rank <= TOP_100_LIMIT


def get_rank_and_points(
    db: Session,
    user: User,
) -> tuple[int, int]:
    """
    Internal helper returning the user's current rank and points.

    Useful for:
    - dashboard services
    - referral summaries
    - admin analytics
    - notification systems
    """

    _require_active_user(user)

    fresh_user = _get_fresh_active_user(
        db=db,
        user=user,
    )

    rank = get_user_rank(
        db=db,
        user=fresh_user,
    )

    return (
        rank,
        int(fresh_user.points or 0),
    )


# ============================================================================
# LEADERBOARD HEALTH / CONSISTENCY HELPERS
# ============================================================================


def validate_leaderboard_position(
    db: Session,
    user: User,
) -> bool:
    """
    Validate that an active user's calculated rank falls inside the valid
    leaderboard range.

    This is an internal integrity helper and is intentionally not exposed
    as a public endpoint.
    """

    _require_active_user(user)

    total_users = _active_user_count(db)

    if total_users <= 0:
        return False

    rank = get_user_rank(
        db=db,
        user=user,
    )

    return 1 <= rank <= total_users


def get_leaderboard_snapshot(
    db: Session,
    *,
    limit: int = TOP_100_LIMIT,
) -> list[LeaderboardUser]:
    """
    Internal snapshot helper.

    Returns a deterministic leaderboard snapshot without exposing ORM
    objects to callers.
    """

    limit = min(
        max(int(limit), 1),
        MAX_PAGE_SIZE,
    )

    users = db.scalars(
        select(User)
        .where(
            User.status == UserStatus.ACTIVE,
        )
        .order_by(
            *_ranking_order(),
        )
        .limit(limit),
    ).all()

    return [
        _serialize_user(
            user=user,
            rank=index + 1,
        )
        for index, user in enumerate(users)
    ]