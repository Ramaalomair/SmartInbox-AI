"""Secure read-only record lookup (Developer B, task B1).

Implements lookup_order / lookup_invoice / lookup_ticket from the AgentTools protocol in
agent/schemas.py. Authorization is enforced HERE, in code. The requester identity is
bound by the application through bind_requester(); the LLM never sees or supplies it.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from agent.schemas import ContractError, Evidence, ToolResult

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

logger = logging.getLogger("smartinbox.audit")
AUDIT_LOG: list[dict] = []  # in-memory copy, used by tests and the UI trace

# record_type -> (csv file, id column, owner column, fields the model may see)
_SPECS = {
    "order": ("orders.csv", "order_id", "customer_email",
              ("order_id", "status", "refund_status", "confirmed_delivery_date")),
    "invoice": ("invoices.csv", "invoice_id", "authorized_contacts",
                ("invoice_id", "payment_status")),  # amount_sar is deliberately withheld
    "ticket": ("tickets.csv", "ticket_id", "customer_email",
               ("ticket_id", "status", "category", "notes", "resolution")),
}
_ID_PATTERNS = {
    "order": re.compile(r"^ORD-\d{4}$"),
    "invoice": re.compile(r"^INV-\d{3}$"),
    "ticket": re.compile(r"^T-\d{3}$"),
}


def _error(code: str, message: str) -> ToolResult:
    return ToolResult(status="error", data=None, error=ContractError(code=code, message=message))


def _denied() -> ToolResult:
    """One identical response for 'not found' and 'not authorized' (no existence leak)."""
    return _error(
        "not_found_or_unauthorized",
        "No accessible record was found. Ask the sender to verify their identity "
        "through an approved channel.",
    )


@lru_cache(maxsize=None)
def _load(filename: str, id_column: str) -> dict[str, dict]:
    with open(DATA_DIR / filename, newline="", encoding="utf-8-sig") as f:
        return {row[id_column]: row for row in csv.DictReader(f)}


def _owners(raw: str) -> set[str]:
    return {part.strip().lower() for part in (raw or "").split(";") if part.strip()}


def _audit(requester: str, record_type: str, record_id: str, outcome: str, reason: str) -> None:
    """Log the attempt WITHOUT any protected record content."""
    event = {"tool": f"lookup_{record_type}", "requester": requester, "record_id": record_id,
             "outcome": outcome, "reason": reason}
    AUDIT_LOG.append(event)
    logger.info(json.dumps(event))


def _lookup(record_type: str, record_id: str, requester_email: str) -> ToolResult:
    rtype = (record_type or "").strip().lower()
    requester = (requester_email or "").strip().lower()
    spec = _SPECS.get(rtype)
    if spec is None:
        return _error("unsupported_record_type", "Use order, invoice or ticket.")

    rid = (record_id or "").strip().upper()
    if not rid:
        return _error("missing_information", "A record ID is required.")

    filename, id_col, owner_col, visible_fields = spec
    if not _ID_PATTERNS[rtype].match(rid):
        _audit(requester, rtype, rid, "denied", "malformed_id")
        return _denied()

    row = _load(filename, id_col).get(rid)
    if row is None:
        _audit(requester, rtype, rid, "denied", "not_found")
        return _denied()
    if not requester or requester not in _owners(row[owner_col]):
        _audit(requester, rtype, rid, "denied", "not_authorized")
        return _denied()

    data = {field: (row.get(field) or None) for field in visible_fields}
    _audit(requester, rtype, rid, "allowed", "authorized")
    return ToolResult(status="success", data=data, error=None)


# --- Public tool functions (match the AgentTools protocol; return plain dicts) ---------------

def lookup_order(order_id: str, requester_email: str) -> dict:
    return _lookup("order", order_id, requester_email).model_dump()


def lookup_invoice(invoice_id: str, requester_email: str) -> dict:
    return _lookup("invoice", invoice_id, requester_email).model_dump()


def lookup_ticket(ticket_id: str, requester_email: str) -> dict:
    return _lookup("ticket", ticket_id, requester_email).model_dump()


def record_evidence(data: dict) -> Evidence:
    """Build a short grounded Evidence item from a successful lookup's data."""
    source_id = next(v for k, v in data.items() if k.endswith("_id"))
    excerpt = "; ".join(
        f"{k}={'unavailable' if v is None else v}" for k, v in data.items() if k != "notes"
    )[:240]
    return Evidence(source_id=source_id, excerpt=excerpt)


class BoundTools:
    """The tools the LLM may call. No method has an identity parameter, so an email
    (or an injected instruction) cannot make the model look up records as someone else."""

    def __init__(self, requester_email: str):
        self._requester_email = requester_email

    def lookup_order(self, order_id: str) -> dict:
        return lookup_order(order_id, self._requester_email)

    def lookup_invoice(self, invoice_id: str) -> dict:
        return lookup_invoice(invoice_id, self._requester_email)

    def lookup_ticket(self, ticket_id: str) -> dict:
        return lookup_ticket(ticket_id, self._requester_email)


def bind_requester(requester_email: str) -> BoundTools:
    return BoundTools(requester_email)