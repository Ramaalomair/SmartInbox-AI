
import re

from agent.schemas import EmailInput, Intent


TOOL_BY_INTENT = {
    "order_status": "lookup_order",
    "return_request": "lookup_order",
    "refund_status": "lookup_order",
    "invoice_status": "lookup_invoice",
    "billing_dispute": "lookup_invoice",
    "technical_issue": "lookup_ticket",
    "service_outage": "lookup_ticket",
}


# Record ID formats must match the backend validation rules.
ID_PATTERNS = {
    "lookup_order": r"\bORD-\d{4}\b",
    "lookup_invoice": r"\bINV-\d{3}\b",
    "lookup_ticket": r"\bT-\d{3}\b",
}


def select_record_tools(
    email: EmailInput,
    intent: Intent
) -> list[dict]:
    """Select read-only record lookups based on intent and record IDs."""

    text = f"{email.subject}\n{email.body}"
    selected = []

    # Suspicious requests must not trigger record lookups.
    if intent == "suspicious_request":
        return []

    for tool_name, pattern in ID_PATTERNS.items():
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        # Remove duplicate IDs while preserving their original order.
        unique_ids = dict.fromkeys(
            record_id.upper()
            for record_id in matches
        )

        for record_id in unique_ids:
            if (
                tool_name == TOOL_BY_INTENT.get(intent)
                or intent == "multi_intent"
            ):
                selected.append({
                    "tool": tool_name,
                    "record_id": record_id
                })

    return selected
