"""
SARAL — SMS Notification Service (Twilio)
==========================================
Credentials are loaded exclusively from backend/.env — never hardcoded.

.env format (see backend/.env):
    TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxx
    TWILIO_AUTH_TOKEN=your_token_here
    TWILIO_FROM_NUMBER=+1XXXXXXXXXX
    OWNER_PHONE_NUMBER=+919497685546
"""

import os
import logging
from pathlib import Path

# ── Load .env from the same directory as this file ─────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass   # dotenv optional — fall back to actual environment variables

logger = logging.getLogger("saral.sms")

# ── Twilio credentials (read from environment) ──────────────────────────────
ACCOUNT_SID   = os.getenv("TWILIO_ACCOUNT_SID", "")
AUTH_TOKEN    = os.getenv("TWILIO_AUTH_TOKEN",  "")
TWILIO_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")

# ── Demo recipient (read from environment, with fallback) ───────────────────
OWNER_PHONE_NUMBER = os.getenv("OWNER_PHONE_NUMBER", "+919497685546")
OWNER_NAME         = "Vehicle Owner"

# ── Fine amounts by violation type ─────────────────────────────────────────
FINE_MAP: dict = {
    "illegal parking":      500,
    "unauthorized parking": 500,
    "no helmet":            500,
    "helmet violation":     500,
    "parking":              500,
    "helmet":               500,
}
DEFAULT_FINE = 500


def _get_fine(violation_type: str) -> int:
    """Return the fine amount for a given violation type string."""
    key = (violation_type or "").lower().strip()
    for pattern, amount in FINE_MAP.items():
        if pattern in key:
            return amount
    return DEFAULT_FINE


def _get_client():
    """
    Return a configured Twilio REST client.
    Returns None and logs a warning if credentials are missing/placeholder.
    """
    if not ACCOUNT_SID or not AUTH_TOKEN or not TWILIO_NUMBER:
        logger.warning(
            "[SMS] Twilio credentials not set in backend/.env — SMS skipped. "
            "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER."
        )
        return None
    try:
        from twilio.rest import Client
        return Client(ACCOUNT_SID, AUTH_TOKEN)
    except ImportError:
        logger.error("[SMS] twilio not installed. Run: pip install twilio")
        return None
    except Exception as e:
        logger.error("[SMS] Could not create Twilio client: %s", e)
        return None


# ── Public API ──────────────────────────────────────────────────────────────

def send_approval_sms(
    plate: str,
    violation_type: str,
    location: str,
    fine: int | None = None,
) -> bool:
    """
    Send an approval SMS to OWNER_PHONE_NUMBER.
    Returns True on success, False otherwise. Never raises.
    """
    client = _get_client()
    if not client:
        return False

    fine      = fine if fine is not None else _get_fine(violation_type)
    plate_str = (plate or "UNKNOWN").upper()

    body = f"SARAL Alert: Violation Approved. Fine Rs{fine}. Plate {plate_str}."
    body = body[:120]   # safety cap

    try:
        msg = client.messages.create(
            body=body,
            from_=TWILIO_NUMBER,
            to=OWNER_PHONE_NUMBER,
        )
        logger.info("[SMS] Approval SMS sent  SID=%s  to=%s", msg.sid, OWNER_PHONE_NUMBER)
        return True
    except Exception as e:
        logger.error("[SMS] Failed to send approval SMS: %s", e)
        return False


def send_rejection_sms(plate: str) -> bool:
    """
    Send a rejection SMS to OWNER_PHONE_NUMBER.
    Returns True on success, False otherwise. Never raises.
    """
    client = _get_client()
    if not client:
        return False

    plate_str = (plate or "UNKNOWN").upper()

    body = f"SARAL Alert: Violation Rejected. No fine issued. Plate {plate_str}."
    body = body[:120]   # safety cap

    try:
        msg = client.messages.create(
            body=body,
            from_=TWILIO_NUMBER,
            to=OWNER_PHONE_NUMBER,
        )
        logger.info("[SMS] Rejection SMS sent  SID=%s  to=%s", msg.sid, OWNER_PHONE_NUMBER)
        return True
    except Exception as e:
        logger.error("[SMS] Failed to send rejection SMS: %s", e)
        return False
