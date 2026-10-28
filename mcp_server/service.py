"""Approved-draft persistence (Developer B, task B2).

This is the ONLY code path that writes drafts. It re-checks approval itself, so a UI
checkbox or a model instruction alone can never cause a write.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from agent.schemas import ApprovalWriteRequest, ContractError, MCPWriteFailure, MCPWriteSuccess
from mcp_server.approval import MissingSecretError, compute_action_id, verify_approval_token


def storage_dir() -> Path:
    path = Path(os.environ.get("SMARTINBOX_STORAGE_DIR") or Path(__file__).resolve().parent.parent / "storage")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append(filename: str, record: dict) -> None:
    with open(storage_dir() / filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def audit(event: str, action_id: str | None, email_id: str | None, reason: str) -> None:
    """Audit log WITHOUT draft content."""
    _append("audit_log.jsonl", {"time": datetime.now(timezone.utc).isoformat(), "event": event,
                                "action_id": action_id, "email_id": email_id, "reason": reason})


def _already_saved(action_id: str) -> bool:
    path = storage_dir() / "approved_drafts.jsonl"
    if not path.exists():
        return False
    with open(path, encoding="utf-8") as f:
        return any(json.loads(line).get("action_id") == action_id for line in f if line.strip())


def _fail(code: str, message: str, action_id: str | None, email_id: str | None) -> dict:
    audit("write_blocked", action_id, email_id, code)
    return MCPWriteFailure(error=ContractError(code=code, message=message)).model_dump()


def save_approved_draft(raw_request: dict | ApprovalWriteRequest) -> dict:
    data = raw_request.model_dump() if isinstance(raw_request, ApprovalWriteRequest) else dict(raw_request)
    try:
        request = ApprovalWriteRequest(**data)
    except (ValidationError, TypeError):
        return _fail("invalid_request", "The write request is malformed.",
                     data.get("action_id"), data.get("email_id"))

    expected_id = compute_action_id(request.email_id, request.primary_department,
                                    request.secondary_department, request.draft_subject,
                                    request.draft_body)
    if expected_id != request.action_id:
        return _fail("payload_mismatch", "The draft differs from the approved action.",
                     request.action_id, request.email_id)
    try:
        token_error = verify_approval_token(request.action_id, request.approval_token)
    except MissingSecretError:
        return _fail("server_misconfigured", "Approval cannot be verified.",
                     request.action_id, request.email_id)
    if token_error:
        return _fail(token_error, "A valid human approval is required before saving.",
                     request.action_id, request.email_id)
    if _already_saved(request.action_id):
        return _fail("already_saved", "This approved action was already saved.",
                     request.action_id, request.email_id)

    record_id = f"DRAFT-{request.action_id[:12]}"
    _append("approved_drafts.jsonl", {
        "record_id": record_id, "action_id": request.action_id, "email_id": request.email_id,
        "primary_department": request.primary_department,
        "secondary_department": request.secondary_department,
        "draft_subject": request.draft_subject, "draft_body": request.draft_body,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    })
    audit("write_saved", request.action_id, request.email_id, "approved")
    return MCPWriteSuccess(record_id=record_id).model_dump()


def record_rejection(action_id: str, email_id: str) -> None:
    """The reviewer rejected the proposal: nothing is persisted, only an audit event."""
    audit("action_rejected", action_id, email_id, "human_rejected")