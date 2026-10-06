# Coindava Waitlist Backend

Ten core Python modules for the Coindava waitlist game. The architecture keeps X integration isolated from referrals, points, and leaderboard logic.

## Build order
config -> database -> models -> main -> auth -> users -> referrals -> points -> social -> leaderboard.

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

## Production requirements
Use Alembic migrations; do not rely on startup table creation. Put SECRET_KEY and X secrets in Render secret environment variables. Use a KMS/secret manager for token encryption, Redis-backed rate limiting, transactional email, monitoring, backups, and comprehensive integration tests before public launch.

The X callback uses server-side OAuth state and PKCE verifier storage; users are returned to the frontend after successful connection.
