"""
Coindava Waitlist Backend
-------------------------

email_service.py

Outbound transactional email (currently: email-address verification).

Providers (EMAIL_PROVIDER):
    brevo     -> Brevo transactional email API (EMAIL_API_KEY)   [default]
    smtp      -> any SMTP server with STARTTLS; Brevo's relay also works:
                 SMTP_HOST=smtp-relay.brevo.com SMTP_PORT=587

Brevo requirement: the address in EMAIL_FROM must be a verified sender (or
belong to an authenticated domain) in your Brevo account, otherwise Brevo
rejects the message.

Rules:
    - Functions are synchronous and are meant to be run through FastAPI
      BackgroundTasks, so a slow mail provider never delays an API response.
    - Delivery failures are logged and swallowed. A failed email must never
      roll back or fail account creation; the user can request a new email
      with POST /auth/resend-verification.
    - Verification tokens are NEVER logged.
"""

from __future__ import annotations

import html
import logging
import re
import smtplib
from email.message import EmailMessage
from urllib.parse import quote

import httpx

from config import settings

logger = logging.getLogger("coindava.email")

HTTP_TIMEOUT_SECONDS = 15.0


def build_verification_link(token: str) -> str:
    """
    Link embedded in the verification email.

    The frontend reads `verify_token` on load and calls
    POST /auth/verify-email.
    """

    frontend_url = settings.frontend_url.rstrip("/")

    return f"{frontend_url}/?verify_token={quote(token, safe='')}#waitlist"


def _render_verification_email(link: str) -> tuple[str, str, str]:
    """Return (subject, plain-text body, HTML body)."""

    subject = "Verify your Coindava email"

    text = (
        "Welcome to the Coindava waitlist.\n\n"
        "Confirm your email address to earn your verification points:\n\n"
        f"{link}\n\n"
        "If you did not create a Coindava account, you can ignore this "
        "email.\n"
    )

    safe_link = html.escape(link, quote=True)

    body = (
        "<div style=\"font-family:Arial,sans-serif;max-width:480px;\">"
        "<h2>Welcome to the Coindava waitlist</h2>"
        "<p>Confirm your email address to earn your verification points.</p>"
        f"<p><a href=\"{safe_link}\" "
        "style=\"display:inline-block;padding:12px 20px;background:#5d7aff;"
        "color:#ffffff;text-decoration:none;border-radius:8px;\">"
        "Verify email</a></p>"
        "<p style=\"color:#666;font-size:13px;\">If you did not create a "
        "Coindava account, you can ignore this email.</p>"
        "</div>"
    )

    return subject, text, body


def _send_via_smtp(
    to_email: str,
    subject: str,
    text: str,
    body: str,
) -> None:
    message = EmailMessage()
    message["From"] = settings.email_from
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(text)
    message.add_alternative(body, subtype="html")

    with smtplib.SMTP(
        settings.smtp_host,
        settings.smtp_port,
        timeout=HTTP_TIMEOUT_SECONDS,
    ) as server:
        if settings.smtp_use_tls:
            server.starttls()

        if settings.smtp_username and settings.smtp_password:
            server.login(
                settings.smtp_username,
                settings.smtp_password.get_secret_value(),
            )

        server.send_message(message)


BREVO_SEND_URL = "https://api.brevo.com/v3/smtp/email"

_SENDER_PATTERN = re.compile(r"^\s*(?P<name>.*?)\s*<(?P<email>[^<>\s]+)>\s*$")


def parse_sender(value: str) -> dict[str, str]:
    """
    Split EMAIL_FROM into Brevo's {"name", "email"} sender object.

    Accepts  'Coindava <no-reply@coindava.com>'  or  'no-reply@coindava.com'.
    """

    match = _SENDER_PATTERN.match(value or "")

    if match:
        name = match.group("name").strip().strip('"')
        sender = {"email": match.group("email")}

        if name:
            sender["name"] = name

        return sender

    return {"email": (value or "").strip(), "name": "Coindava"}


def _send_via_brevo(
    to_email: str,
    subject: str,
    text: str,
    body: str,
) -> None:
    response = httpx.post(
        BREVO_SEND_URL,
        headers={
            "api-key": settings.email_api_key.get_secret_value(),
            "accept": "application/json",
            "content-type": "application/json",
        },
        json={
            "sender": parse_sender(settings.email_from),
            "to": [{"email": to_email}],
            "subject": subject,
            "htmlContent": body,
            "textContent": text,
        },
        timeout=HTTP_TIMEOUT_SECONDS,
    )

    if response.status_code >= 400:
        # Brevo answers {"code": "...", "message": "..."} (for example an
        # unverified sender or an invalid API key). Log that, never the
        # request body, so the verification token cannot leak into logs.
        try:
            detail = response.json().get("message", "")
        except Exception:
            detail = ""

        raise RuntimeError(
            f"Brevo rejected the email (HTTP {response.status_code}): {detail}"
        )


def send_verification_email(to_email: str, token: str) -> None:
    """
    Send the email-verification message.

    Never raises: failures are logged (without the token or recipient's
    secrets) and swallowed.
    """

    if not settings.email_enabled:
        logger.warning(
            "EMAIL_ENABLED is false; verification email was not sent."
        )
        return

    subject, text, body = _render_verification_email(
        build_verification_link(token)
    )

    provider = settings.email_provider

    try:
        if provider == "smtp":
            _send_via_smtp(to_email, subject, text, body)

        elif provider == "brevo":
            _send_via_brevo(to_email, subject, text, body)

        else:
            logger.warning(
                "Email provider '%s' has no sender implementation; "
                "verification email was not sent.",
                provider,
            )
            return

    except Exception:
        # logger.exception would include the request body/URL only for
        # httpx errors, never the token; the token is not in any message.
        logger.exception(
            "Failed to send verification email via %s.",
            provider,
        )


__all__ = [
    "build_verification_link",
    "send_verification_email",
]
