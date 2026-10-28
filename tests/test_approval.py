import json

import pytest

from mcp_server import service
from mcp_server.approval import compute_action_id, issue_approval_token
from mcp_server.client import call_save_draft, list_server_tools

PAYLOAD = dict(email_id="E005", primary_department="customer_support", secondary_department=None,
               draft_subject="Your refund status",
               draft_body="Your refund for ORD-1002 is pending; we cannot confirm it yet.")


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("APPROVAL_SECRET", "test-secret-value")
    monkeypatch.setenv("SMARTINBOX_STORAGE_DIR", str(tmp_path))
    return tmp_path


def request(payload=PAYLOAD, token=None, action_id=None):
    aid = action_id or compute_action_id(**payload)
    return {**payload, "action_id": aid, "approval_token": token if token is not None else issue_approval_token(aid)}


def saved_lines(tmp_path):
    path = tmp_path / "approved_drafts.jsonl"
    return path.read_text().splitlines() if path.exists() else []


def audit_events(tmp_path):
    path = tmp_path / "audit_log.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_action_id_is_deterministic_and_payload_bound():
    base = compute_action_id(**PAYLOAD)
    assert base == compute_action_id(**PAYLOAD)
    for field, value in [("draft_body", "changed"), ("draft_subject", "changed"),
                         ("primary_department", "finance"), ("secondary_department", "finance"),
                         ("email_id", "E006")]:
        assert compute_action_id(**{**PAYLOAD, field: value}) != base


def test_valid_approval_saves_once(env):
    result = service.save_approved_draft(request())
    assert result["status"] == "saved" and result["record_id"].startswith("DRAFT-")
    assert len(saved_lines(env)) == 1


@pytest.mark.parametrize("token", ["", "garbage", "123.abc"])
def test_missing_or_malformed_token_never_writes(env, token):
    result = service.save_approved_draft(request(token=token))
    assert result["status"] == "error" and not saved_lines(env)


def test_forged_token_is_rejected(env):
    aid = compute_action_id(**PAYLOAD)
    result = service.save_approved_draft(request(token="1700000000." + "0" * 64, action_id=aid))
    assert result["error"]["code"] in {"invalid_token", "expired_token"} and not saved_lines(env)


def test_token_for_another_action_is_rejected(env):
    other = compute_action_id(**{**PAYLOAD, "draft_body": "different"})
    result = service.save_approved_draft(request(token=issue_approval_token(other)))
    assert result["error"]["code"] == "invalid_token" and not saved_lines(env)


def test_draft_changed_after_approval_is_blocked(env):
    approved = request()
    approved["draft_body"] = "Your refund has been completed."  # tampered after approval
    result = service.save_approved_draft(approved)
    assert result["error"]["code"] == "payload_mismatch" and not saved_lines(env)


def test_expired_token_is_rejected(env):
    aid = compute_action_id(**PAYLOAD)
    result = service.save_approved_draft(request(token=issue_approval_token(aid, now=1_000_000), action_id=aid))
    assert result["error"]["code"] == "expired_token" and not saved_lines(env)


def test_replay_is_blocked(env):
    req = request()
    assert service.save_approved_draft(req)["status"] == "saved"
    assert service.save_approved_draft(req)["error"]["code"] == "already_saved"
    assert len(saved_lines(env)) == 1


def test_rejected_action_never_persists(env):
    service.record_rejection(compute_action_id(**PAYLOAD), "E005")
    assert not saved_lines(env)
    assert audit_events(env)[-1]["event"] == "action_rejected"


def test_fails_closed_without_secret(env, monkeypatch):
    req = request()
    monkeypatch.delenv("APPROVAL_SECRET")
    result = service.save_approved_draft(req)
    assert result["error"]["code"] == "server_misconfigured" and not saved_lines(env)


def test_malformed_request_is_rejected(env):
    assert service.save_approved_draft({"email_id": "E005"})["error"]["code"] == "invalid_request"
    assert service.save_approved_draft({**request(), "extra": "x"})["error"]["code"] == "invalid_request"
    assert not saved_lines(env)


def test_audit_never_contains_draft_text(env):
    service.save_approved_draft(request())
    service.save_approved_draft(request(token="bad"))
    assert "pending" not in (env / "audit_log.jsonl").read_text()


def test_mcp_server_exposes_only_the_save_tool():
    assert list_server_tools() == ["save_draft_mcp"]


def test_mcp_roundtrip_with_and_without_approval(env):
    assert call_save_draft(request())["status"] == "saved"
    assert len(saved_lines(env)) == 1
    forged = call_save_draft(request(payload={**PAYLOAD, "email_id": "E006"}, token="1.bad"))
    assert forged["status"] == "error" and len(saved_lines(env)) == 1