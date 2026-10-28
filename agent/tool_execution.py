
from agent.schemas import EmailInput, AgentTools


def execute_record_lookups(
    selections: list[dict],
    email: EmailInput,
    tools: AgentTools
) -> list[dict]:
    """Execute read-only lookups using backend-bound requester identity."""

    results = []

    # The backend must bind a trusted requester identity before calling this.
    # The email body must never control the requester identity.
    _ = email

    for selection in selections:
        tool_name = selection["tool"]
        record_id = selection["record_id"]

        try:
            if tool_name == "lookup_order":
                response = tools.lookup_order(
                    order_id=record_id
                )

            elif tool_name == "lookup_invoice":
                response = tools.lookup_invoice(
                    invoice_id=record_id
                )

            elif tool_name == "lookup_ticket":
                response = tools.lookup_ticket(
                    ticket_id=record_id
                )

            else:
                response = {
                    "status": "error",
                    "data": None,
                    "error": {
                        "code": "unknown_tool",
                        "message": "The requested tool is not available."
                    }
                }

            if (
                not isinstance(response, dict)
                or response.get("status") not in ("success", "error")
            ):
                raise ValueError("Invalid tool response.")

        except Exception:
            # Never expose internal backend exceptions to the LLM.
            response = {
                "status": "error",
                "data": None,
                "error": {
                    "code": "tool_execution_failed",
                    "message": "The record lookup could not be completed."
                }
            }

        results.append({
            "tool": tool_name,
            "record_id": record_id,
            "result": response
        })

    return results
