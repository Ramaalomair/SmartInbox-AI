import inspect
import json

from agent import tools
from agent.schemas import ToolResult
from agent.tools import (AUDIT_LOG, bind_requester, lookup_invoice, lookup_order,
                         lookup_ticket, record_evidence)

DENIED = "not_found_or_unauthorized"


def assert_denied(res: dict) -> None:
    assert res["status"] == "error" and res["data"] is None
    assert res["error"]["code"] == DENIED


def test_owner_can_read_order_e005():
    r = lookup_order("ORD-1002", "sender5@example.com")
    assert r["status"] == "success" and r["error"] is None
    assert r["data"]["refund_status"] == "pending"


def test_unauthorized_sender_e032_gets_no_data():
    assert_denied(lookup_order("ORD-1004", "unknown.person@example.com"))


def test_not_found_and_unauthorized_are_indistinguishable():
    not_found = lookup_order("ORD-9999", "sender5@example.com")
    unauthorized = lookup_order("ORD-1004", "sender5@example.com")
    assert not_found == unauthorized


def test_unlinked_records_are_denied():
    assert_denied(lookup_order("ORD-1009", "sender1@example.com"))
    assert_denied(lookup_ticket("T-103", "sender16@example.com"))


def test_invoice_with_two_authorized_contacts():
    for sender in ("sender3@example.com", "sender27@example.com"):
        assert lookup_invoice("INV-301", sender)["status"] == "success"
    assert_denied(lookup_invoice("INV-301", "sender8@example.com"))


def test_ticket_ownership_and_blank_resolution():
    ok = lookup_ticket("T-100", "sender11@example.com")
    assert ok["status"] == "success" and ok["data"]["resolution"] is None
    assert_denied(lookup_ticket("T-101", "sender11@example.com"))
    assert lookup_ticket("T-102", "sender28@example.com")["data"]["resolution"]


def test_missing_confirmed_date_is_explicit():
    r = lookup_order("ORD-1005", "sender26@example.com")
    assert r["data"]["confirmed_delivery_date"] is None
    assert "unavailable" in record_evidence(r["data"]).excerpt


def test_missing_id_and_unsupported_type():
    assert lookup_order("", "sender1@example.com")["error"]["code"] == "missing_information"
    assert tools._lookup("customer", "X-1", "sender1@example.com").error.code == "unsupported_record_type"


def test_sender_match_is_case_and_space_insensitive():
    assert lookup_order(" ord-1001 ", " SENDER1@Example.com ")["status"] == "success"


def test_sensitive_fields_are_never_returned():
    for res in (lookup_order("ORD-1002", "sender5@example.com"),
                lookup_invoice("INV-301", "sender3@example.com"),
                lookup_ticket("T-100", "sender11@example.com")):
        assert not {"customer_email", "authorized_contacts", "amount_sar"} & set(res["data"])


def test_results_match_the_shared_contract():
    for res in (lookup_order("ORD-1002", "sender5@example.com"),
                lookup_order("ORD-1004", "nobody@example.com")):
        ToolResult(**res)  # extra="forbid": fails on any unexpected key


def test_llm_visible_tools_have_no_identity_parameter():
    bound = bind_requester("sender5@example.com")
    for name, expected in (("lookup_order", {"order_id"}), ("lookup_invoice", {"invoice_id"}),
                           ("lookup_ticket", {"ticket_id"})):
        assert set(inspect.signature(getattr(bound, name)).parameters) == expected
    assert_denied(bound.lookup_order("ORD-1004"))  # bound identity cannot be overridden


def test_audit_logs_denial_without_record_content():
    before = len(AUDIT_LOG)
    lookup_order("ORD-1004", "unknown.person@example.com")
    event = AUDIT_LOG[-1]
    assert len(AUDIT_LOG) == before + 1
    assert event["outcome"] == "denied" and event["reason"] == "not_authorized"
    assert "completed" not in json.dumps(event)