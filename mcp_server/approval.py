"""Approval binding (Developer B, task B2).

Flow:
  1. The agent returns a proposed action. Its proposed_action_id is compute_action_id(...),
     a hash of the exact payload (email, departments, draft subject and body).
  2. Only after the human clicks Approve does the APPLICATION call issue_approval_token().
     The token is an HMAC over the action id with a server-side secret, so the model
     cannot fabricate one. issue_approval_token is NOT exposed as an MCP tool.
  3. The MCP write tool re-computes the id from the payload and verifies the token.

Fails closed: without APPROVAL_SECRET nothing can be issued or verified.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

TOKEN_TTL_SECONDS = 900  # an approval is valid for 15 minutes


class MissingSecretError(RuntimeError):
    pass


def _secret() -> bytes:
    value = os.environ.get("APPROVAL_SECRET", "")
    if not value:
        raise MissingSecretError("APPROVAL_SECRET is not set")
    return value.encode("utf-8")


def compute_action_id(email_id: str, primary_department: str, secondary_department: str | None,
                      draft_subject: str, draft_body: str) -> str:
    """Hash of the exact payload; any change to the draft or routing changes the id."""
    canonical = json.dumps(
        {"email_id": email_id, "primary_department": primary_department,
         "secondary_department": secondary_department,
         "draft_subject": draft_subject, "draft_body": draft_body},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sign(action_id: str, issued_at: int) -> str:
    return hmac.new(_secret(), f"{action_id}|{issued_at}".encode("utf-8"), hashlib.sha256).hexdigest()


def issue_approval_token(action_id: str, now: float | None = None) -> str:
    issued_at = int(now if now is not None else time.time())
    return f"{issued_at}.{_sign(action_id, issued_at)}"


def verify_approval_token(action_id: str, token: str, now: float | None = None) -> str | None:
    """Return None when valid, otherwise an error code."""
    try:
        issued_raw, signature = (token or "").split(".", 1)
        issued_at = int(issued_raw)
    except ValueError:
        return "malformed_token"
    if not hmac.compare_digest(signature, _sign(action_id, issued_at)):
        return "invalid_token"
    current = now if now is not None else time.time()
    if current - issued_at > TOKEN_TTL_SECONDS or issued_at > current + 60:
        return "expired_token"
    return None