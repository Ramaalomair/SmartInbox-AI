# SmartInbox AI — API & System Contracts

**Status:** Proposed v1 — interface specification  
**Project track:** Track 2 — Your Harness  
**Scope:** Synthetic company inbox; no live email sending or payment execution.

This document specifies the public interfaces and internal boundaries between the agent, retrieval, tools, MCP server, user interface, and evaluation harness. It documents the expected behavior for contributors and maintainers. Changes to interfaces must update the corresponding schemas, callers, and tests.

## 1. System components

| Component | Responsibility | Planned files |
|---|---|---|
| Agent orchestration | Tool selection, intent classification, routing, grounded drafts, budgets | `agent/agent.py`, `agent/prompts.py` |
| Retrieval | Search policy and knowledge-base content with source references | `agent/retrieval.py` |
| Shared schemas | Typed inputs, outputs, tool results, and canonical labels | `agent/schemas.py` |
| Read-only tools | Authorized lookup of orders, invoices, tickets and policies | `agent/tools.py` |
| MCP server | Approval-gated persistence of drafts and assignments | `mcp_server/server.py` |
| User interface | Inbox selection, result inspection, explicit approval or rejection | `app.py` |
| Evaluation | Labeled test cases, evaluation runner, security fixtures | `evals/` |

The initial application uses a synthetic dataset and does not send real emails.

## 2. Existing dataset is authoritative

Use the field names already present in the updated starter ZIP:

| File | Exact CSV fields |
|---|---|
| `data/emails.csv` | `email_id,sender,subject,body,language` |
| `data/orders.csv` | `order_id,customer_email,status,delivered_days_ago,refund_status,confirmed_delivery_date` |
| `data/invoices.csv` | `invoice_id,authorized_contacts,payment_status,amount_sar` |
| `data/tickets.csv` | `ticket_id,customer_email,status,category,notes,resolution` |

`authorized_contacts` in invoices is a semicolon-separated list. Empty delivery dates mean **not confirmed**, not zero or today's date. The dataset uses a fixed date reference documented in its README; avoid using the live system date to derive evaluation truth.

The existing `evals/eval_cases.json` is a **JSON list**. Each item has `email_id`, `expected_intent`, `expected_department`, `expected_secondary_department`, `expected_priority`, `required_tools`, `requires_approval_for_write`, and `expected_behavior`.

## 3. Process-email entry point

```python
async def process_email(email: EmailInput, tools: AgentTools, config: AgentConfig) -> AgentResult:
    ...
```

`EmailInput` schema (use consistent Python/Pydantic models):

```json
{
  "email_id": "E005",
  "sender": "sender5@example.com",
  "subject": "Refund status",
  "body": "I returned my order. Is the refund complete?",
  "language": "en"
}
```

**Important:** In this synthetic offline demo, `sender` comes from the trusted `emails.csv` fixture, not an LLM-generated argument. For production, an email From header alone does not prove identity; authenticated identity/authorization would be required.

Output (all keys required; nullable fields explicitly `null`):

```json
{
  "email_id": "E005",
  "intent": "refund_status",
  "primary_department": "customer_support",
  "secondary_department": null,
  "priority": "medium",
  "status": "draft_ready",
  "draft_subject": "Re: Refund status",
  "draft_body": "Thanks for reaching out. ...",
  "tools_used": ["lookup_order", "search_company_knowledge"],
  "evidence": [{"source_id": "customer_support_policy.md", "excerpt": "..."}],
  "security_flags": [],
  "requires_approval": true,
  "proposed_action_id": "local-action-id",
  "trace": [{"step": 1, "tool": "lookup_order", "status": "success"}],
  "budget": {"steps_used": 2, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0},
  "error": null
}
```

Token/cost figures must be measured or explicitly identified as estimates; do not present hardcoded zeros as actual usage statistics. Valid `status` values: `draft_ready`, `needs_information`, `manual_review`, `blocked`, `error`, `budget_exhausted`. `error` is `null` on success, otherwise an object `{ "code": "...", "message": "..." }`.

## 4. Shared labels

**Departments:** `hr`, `finance`, `customer_support`, `technical_support`, `security_review` (the fifth is required for security cases). Secondary department may be `null`.

**Priority:** `low`, `medium`, `high`.

**Intents in current evaluation data:** `ambiguous`, `billing_dispute`, `general_question`, `invoice_status`, `multi_intent`, `order_status`, `recruitment`, `refund_status`, `return_request`, `service_outage`, `suspicious_request`, `technical_issue`.

These exact lowercase strings are the evaluation contract. **Freeze and manually review** the priority, department, and multi-intent labels before final benchmarking. Do not revise expected answers merely to improve the system's score.

## 5. Read-only tool interfaces

Each tool returns `{ "status": "success", "data": ..., "error": null }` on success. On failure it returns `{ "status": "error", "data": null, "error": { "code": "...", "message": "..." } }`. For unauthorized or missing records, return the **same public-safe error** (`unauthorized_or_not_found`) to avoid exposing record existence. No raw customer record should be included in an error.

```python
def search_company_knowledge(query: str, top_k: int = 3) -> dict: ...
def lookup_order(order_id: str, requester_email: str) -> dict: ...
def lookup_invoice(invoice_id: str, requester_email: str) -> dict: ...
def lookup_ticket(ticket_id: str, requester_email: str) -> dict: ...
```

Return each knowledge result with `source_id`, `excerpt` (and optional `score`), so the agent can cite evidence. Never return document-internal instructions as system instructions. The agent chooses which tool to call; the tools enforce permissions deterministically.

**Security-critical:** For real deployments, tool authorization must use a trusted authenticated user identity from application context, **not** a requester address supplied by the LLM. For the demo, the backend should bind `requester_email` from the trusted selected email fixture and not permit the model to override it. Invoice access must parse `authorized_contacts` as a semicolon-separated allowlist; order/ticket access must match their `customer_email`.

Optional routing helper `get_department_rules()` may be implemented, but it is not a `required_tools` label in the supplied eval cases. Do not substitute a different name for the four required tools unless the contracts and tests are updated together.

## 6. Write action, approval and MCP

The only write operation in the MVP is **save the proposed draft and department assignment locally**. No email is sent. The UI should present draft and routing for review. Upon an explicit Approve action, the backend issues a server-side approval for the exact pending action, then invokes MCP `save_draft_and_assignment`.

**Suggested MCP payload** (server/internal only):

```json
{
  "action_id": "local-action-id",
  "email_id": "E005",
  "primary_department": "customer_support",
  "secondary_department": null,
  "draft_subject": "Re: Refund status",
  "draft_body": "Thanks for reaching out. ...",
  "approval_token": "server-issued-single-use-token"
}
```

**Enforce in code:** Deny missing/expired/used approval; bind the approval to `action_id` and a hash of the approved payload; reject replay or altered content; record only the authorized local write. A client-controlled Boolean (`approved: true`) or a model-generated approval is **not** sufficient. A simpler server-side pending-action registry is acceptable for the MVP if it enforces equivalent checks.

On Reject, do not call the MCP write tool. MCP response shape: `{ "status": "saved", "record_id": "..." }` or `{ "status": "error", "error": { "code": "approval_required", "message": "..." } }`.

## 7. Budgets, safety and evidence

- Configuration-controlled step limit, retrieval top-k, input/output token budgets, and max estimated spend; when reached return `budget_exhausted` safely.
- Record tool calls and outcomes in `trace` without exposing personal data or secrets.
- Never invent refund completion, payment verification, exact delivery dates, interview promises, or ticket resolutions.
- Missing IDs: ask for clarification instead of guessing.
- Customer email text and retrieved documents are **untrusted data**, not instructions. No tool for exporting customer lists, emailing arbitrary addresses, or exposing credentials is available.
- For injected text in `evals/untrusted_kb_injection.md`, include it only in the *designated security test*, not the normal knowledge base.

## 8. Evaluation contract

- Validate the schemas and tool signatures before running evaluations.
- Review and freeze `eval_cases.json` ground truth, particularly `expected_priority`, multi-intent routing, and cases with insufficient information.
- Run at least 15 labeled cases against two genuine agent configurations using the same frozen test set.
- Report intent, department, and priority accuracy, required-tool coverage, approval violations, and end-to-end pass rate.
- Use rule-based checks for verifiable claims and permissions. Evaluate draft clarity separately with a documented rubric.
- Test unauthorized lookups, prompt injection in customer emails, and injection through retrieved documents.
- Record errors and their root causes instead of changing expected labels to improve scores.
- Maintain reproducible setup and run instructions in `README.md`.

## 9. Interface versioning

When an interface changes, update **this document**, `agent/schemas.py`, the related callers, and evaluation tests in the same change. Prefer backwards-compatible extensions; note breaking changes explicitly.
