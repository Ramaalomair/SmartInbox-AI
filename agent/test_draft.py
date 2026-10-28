
from agent.draft import generate_draft
from agent.schemas import EmailInput, Evidence


def main():
    """Test draft generation using supplied policy evidence."""

    email = EmailInput(
        email_id="TEST-DRAFT-001",
        sender="customer@example.com",
        subject="Delayed order",
        body=(
            "My order ORD-1005 has not arrived. "
            "Can you confirm the exact delivery date?"
        ),
        language="en"
    )

    evidence = [
        Evidence(
            source_id="customer_support_policy.md",
            excerpt=(
                "Order status must be checked via lookup_order "
                "before making delivery-status claims. "
                "Do not promise a delivery date unless the order "
                "record explicitly contains a confirmed date."
            ),
            score=0.95
        )
    ]

    # No successful authorized record lookup is available.
    verified_records = []

    result = generate_draft(
        email=email,
        evidence=evidence,
        verified_records=verified_records
    )

    print("\nDraft Subject:", result["draft_subject"])
    print("\nDraft Body:\n", result["draft_body"])
    print("\nRequires Approval:", result["requires_approval"])


if __name__ == "__main__":
    main()
