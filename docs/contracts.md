# SmartInbox AI — API & System Contracts

**Status:** Implemented MVP — current interface and security contract  
**Project track:** Track 2 — Your Harness  
**Scope:** Synthetic company inbox; no live email sending or payment execution.

This document describes the implemented MVP interfaces and security boundaries between the agent, retrieval, read-only tools, MCP write service, UI, and evaluation harness. It also records remaining evaluation limitations. Changes to interfaces must update the corresponding schemas, callers, and tests.

## 1. System components

| Component | Responsibility | Current files |
|---|---|---|
| Agent orchestration | Classification, routing, deterministic record-tool selection, grounded drafts, budgets | `agent/orchestrator.py`, `agent/classification.py`, `agent/tool_selection.py`, `agent/tool_execution.py`, `agent/draft.py`, `agent/llm.py` |
| Retrieval | TF-IDF baseline and multilingual E5 + RRF hybrid policy retrieval with source references | `agent/retrieval.py`, `agent/hybrid_retrieval.py`, `agent/integrated_tools.py` |
| Shared schemas | Typed inputs, outputs, tool results, and canonical labels | `agent/schemas.py` |
| Read-only tools | Authorized lookup of synthetic orders, invoices, and tickets | `agent/tools.py`, `agent/integrated_tools.py` |
| MCP server | Approval-gated draft and routing persistence | `mcp_server/server.py`, `mcp_server/approval.py`, `mcp_server/service.py`, `mcp_server/client.py` |
| User interface | Inbox selection, result inspection, explicit approval or rejection | `app.py` |
| Evaluation | Labeled cases, RAG benchmarking, agent evaluation, security tests, preserved reports | `evals/cases/`, `evals/run_evals.py`, `evals/evaluate_retrieval.py`, `evals/test_security.py`, `evals/reports/`, `tests/` |

The application uses synthetic company records and does not send real emails, execute payments, or issue refunds. Live Agent mode calls an external model provider through OpenRouter; Mock mode is available for UI demonstrations. `agent/agent.py` also retains a standalone function-calling demonstration; the application processing path uses `agent/orchestrator.py`.

## 2. Synthetic dataset and evaluation cases

Use the following fixture fields as the dataset contract:

| File | Exact CSV fields |
|---|---|
| `data/emails.csv` | `email_id,sender,subject,body,language` |
| `data/orders.csv` | `order_id,customer_email,status,delivered_days_ago,refund_status,confirmed_delivery_date` |
| `data/invoices.csv` | `invoice_id,authorized_contacts,payment_status,amount_sar` |
| `data/tickets.csv` | `ticket_id,customer_email,status,category,notes,resolution` |

`authorized_contacts` in invoices is a semicolon-separated list. Empty delivery dates mean **not confirmed**, not zero or today's date. The dataset uses a fixed date reference documented in its README; avoid using the live system date to derive evaluation truth.

The `evals/cases/eval_cases.json` is a **JSON list**. Each item has `email_id`, `expected_intent`, `expected_department`, `expected_secondary_department`, `expected_priority`, `required_tools`, `requires_approval_for_write`, and `expected_behavior`.

## 3. Process-email entry point

```python
async def process_email(email: EmailInput, tools: AgentTools, config: AgentConfig | None = None) -> AgentResult:
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

**Important:** In this local synthetic demo, `sender` comes from the selected `emails.csv` fixture, not an LLM-generated argument. For production, an email From header alone does not prove identity; authenticated identity/authorization would be required.

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
  "proposed_action_id": null,
  "trace": [{"step": 2, "tool": "search_company_knowledge", "status": "success"}, {"step": 3, "tool": "lookup_order", "status": "success"}],
  "budget": {"steps_used": 4, "input_tokens": null, "output_tokens": null, "estimated_cost_usd": null},
  "error": null
}
```

A `draft_ready` result declares `requires_approval=true` but does not itself authorize persistence. `proposed_action_id` may be `null` in the agent result; the approval flow derives and validates a content-bound action identifier at write time. Token/cost figures must be measured or explicitly identified as estimates; missing values should remain `null`, not invented.  Valid `status` values: `draft_ready`, `needs_information`, `manual_review`, `blocked`, `error`, `budget_exhausted`. `error` is `null` on success, otherwise an object `{ "code": "...", "message": "..." }`.

## 4. Shared labels

**Departments:** `hr`, `finance`, `customer_support`, `technical_support`, `security_review` (the fifth is required for security cases). Secondary department may be `null`.

**Priority:** `low`, `medium`, `high`.

**Intents in current evaluation data:** `ambiguous`, `billing_dispute`, `general_question`, `invoice_status`, `multi_intent`, `order_status`, `recruitment`, `refund_status`, `return_request`, `service_outage`, `suspicious_request`, `technical_issue`.

These exact lowercase strings are the evaluation contract. The expected priority, department, and multi-intent labels should be reviewed and frozen before comparing configurations. Do not revise expected answers merely to improve the system's score.

## 5. Read-only tool interfaces

Each tool returns `{ "status": "success", "data": ..., "error": null }` on success. On failure it returns `{ "status": "error", "data": null, "error": { "code": "...", "message": "..." } }`. For unauthorized or missing records, return the **same public-safe error** (`not_found_or_unauthorized`) to avoid exposing record existence. No raw customer record should be included in an error.

```python
def search_company_knowledge(query: str, top_k: int = 3) -> dict: ...
def lookup_order(order_id: str) -> dict: ...
def lookup_invoice(invoice_id: str) -> dict: ...
def lookup_ticket(ticket_id: str) -> dict: ...
```

Return each knowledge result with `source_id`, `excerpt`, and (when available) `score`, so the agent can cite evidence. Retrieved text is data, not an instruction source. The implemented orchestration selects record lookups using deterministic code after LLM classification; the tools enforce permissions at the backend boundary. The separate `agent/agent.py::run_agent` remains an optional function-calling example and is not the main UI workflow.

**Security-critical:** For real deployments, tool authorization must use a trusted authenticated user identity from application context, **not** a requester address supplied by the LLM. For the demo, the backend should bind `requester_email` from the trusted selected email fixture and not permit the model to override it. Invoice access must parse `authorized_contacts` as a semicolon-separated allowlist; order/ticket access must match their `customer_email`.

The evaluation tool labels are `search_company_knowledge`, `lookup_order`, `lookup_invoice`, and `lookup_ticket`. If tool names or signatures change, update the schemas, callers, and evaluation expectations together.

## 6. Write action, approval and MCP

The only write operation in the MVP is **save the proposed draft and department assignment locally**. No email is sent. The UI should present draft and routing for review. Upon an explicit Approve action, the application obtains an approval token bound to the exact draft/action and invokes the MCP tool `save_draft_mcp`. The MCP service calls `save_approved_draft` after validating the signed token and payload.

**MCP write payload** (server/internal only; illustrative synthetic values):

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

**Enforced by the implementation:** Deny missing, malformed, expired, forged, or replayed approval tokens; bind approval to the content-derived `action_id` and approved payload; reject altered content; record only the authorized local write. A client-controlled Boolean (`approved: true`) or model-generated approval is never sufficient. Approved drafts are stored under the configured storage directory (by default `storage/approved_drafts.jsonl`); audit events use `storage/audit_log.jsonl`. These runtime JSONL files are ignored by Git.

On Reject, do not call the MCP write tool. A successful write returns a saved record identifier; unsuccessful writes return a structured error. No outbound email is sent.

## 7. Budgets, safety and evidence

- Configuration-controlled step limit, retrieval top-k, input/output token budgets, and max estimated spend; when reached return `budget_exhausted` safely.
- Record tool calls and outcomes in `trace` without exposing personal data or secrets.
- Never invent refund completion, payment verification, exact delivery dates, interview promises, or ticket resolutions.
- Missing IDs: ask for clarification instead of guessing.
- Customer email text and retrieved documents are **untrusted data**, not instructions. No tool for exporting customer lists, emailing arbitrary addresses, or exposing credentials is available.
- For injected text in `evals/cases/untrusted_kb_injection.md`, include it only in the *designated security test*, not the normal knowledge base.

## 8. Evaluation contract

- Validate the schemas and tool signatures before running evaluations.
- Review and freeze `evals/cases/eval_cases.json` ground truth, particularly `expected_priority`, multi-intent routing, and cases with insufficient information.
- The current B4 evaluation runs 34 synthetic labeled emails through the real `process_email` implementation. Comparing two genuine agent configurations on the same frozen cases is a possible additional experiment, not a result established by the existing reports.
- Report intent, primary and secondary department, priority, required-tool coverage, and approval declaration separately. Structural agreement is not proof of safe approval enforcement or factual correctness of drafts.
- Keep deterministic permission/approval regression tests distinct from LLM-dependent prompt-injection integration tests. Evaluate semantic correctness and draft clarity separately with a documented rubric; the current B4 report marks semantic behavior `not_assessed`.
- Test unauthorized lookups, prompt injection in customer emails, and injection through retrieved documents.
- Record errors and their root causes instead of changing expected labels to improve scores.
- Preserve existing evidence in `evals/reports/agent/`, `evals/reports/rag/`, and `evals/reports/security/`. `evals/run_evals.py` writes reports and calls OpenRouter; do not rerun it merely to reorganize files or when the provider quota is exhausted.
- Maintain reproducible setup and run instructions in `README.md`.

### Current evaluation evidence (preserved reports)

| Evaluation | Evidence | Scope / result |
|---|---|---|
| Agent (B4) | `evals/reports/agent/results.json`, `results.csv` | 34/34 completed, 0 execution errors, 26/34 structural all-check passes (76.47%); semantic behavior not assessed |
| Retrieval | `evals/reports/rag/rag_evaluation.txt` | Hybrid RRF Hit@3 = 93.75%, MRR@3 = 0.9062 across 16 cases |
| Security | `evals/reports/security/security_evaluation.xml` | Last saved full run: 40 tests, 0 failures, 0 errors. Five LLM-dependent tests cannot be verified while OpenRouter responds with HTTP 429; 35 other tests passed in the later offline run. |

The security XML preserves an earlier passing run; it should not be described as the result of a later quota-limited run. Results may change with a new model, policy text, fixtures, or code revision.

## 9. Interface versioning

When an interface changes, update **this document**, `agent/schemas.py`, the related callers, and evaluation tests in the same change. Prefer backwards-compatible extensions; note breaking changes explicitly.
