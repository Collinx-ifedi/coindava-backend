"""
Admin CLI.

    python manage.py make-admin you@example.com
    python manage.py seed-campaign
"""

from __future__ import annotations

import sys

from sqlalchemy import select

from auth import normalize_email
from database import SessionLocal, initialize_database
from models import Campaign, CampaignStatus, User


def make_admin(email: str) -> None:
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.email == normalize_email(email)))
        if not user:
            sys.exit(f"No user with email {email}")
        user.is_admin = True
        db.commit()
        print(f"{user.email} is now an admin.")


def seed_campaign() -> None:
    with SessionLocal() as db:
        if db.scalar(select(Campaign).where(Campaign.slug == "launch")):
            print("Campaign 'launch' already exists.")
            return
        db.add(Campaign(name="Launch", slug="launch", status=CampaignStatus.ACTIVE))
        db.commit()
        print("Created active campaign 'launch'. Add tasks via POST /api/v1/social/admin/tasks.")


if __name__ == "__main__":
    initialize_database()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "make-admin" and len(sys.argv) == 3:
        make_admin(sys.argv[2])
    elif cmd == "seed-campaign":
        seed_campaign()
    else:
        sys.exit(__doc__)
