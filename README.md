# SmartInbox AI

> Customer support leads, finance, HR and technical teams who share one inbox use this agent to get each email classified, routed to the right department with a priority, and answered with a grounded draft reply that a human approves before anything is saved.

**Track:** 2 · Your harness
**Team:**  Remas Alsubaie, Rama Alomair
**Model(s):** <TODO: e.g. google/gemma-4-31b-it:free via OpenRouter>, chosen because <TODO: benchmark evidence + your own eval>

## Problem

Companies that receive mixed inquiries through one address (HR, finance, technical support, customer support) sort and answer them by hand. This is slow, priority is judged inconsistently, and replies often make claims that were never checked against the order, invoice or policy (for example confirming a refund that has not happened). Emails can also carry hidden instructions that try to make an assistant leak data.

SmartInbox AI works on a fictional company, **NovaTech**, with fully synthetic data. For each email the agent identifies the intent, picks a department and priority, calls the lookup and policy tools it needs, and prepares a draft with source references. "Done" means: correct department and priority, every factual claim in the draft traceable to a record or policy, a human approval before any write, and no action taken on instructions found inside the email.

## How to run

```bash
uv run smartinbox.py ask "Where is my refund for ORD-1002?"
uv run smartinbox.py eval tasks.jsonl --config A B
```

Environment: `OPENROUTER_API_KEY` in `.env` (see `.env.example`). <TODO: Streamlit UI command, if included, e.g. `uv run streamlit run app.py`.>

## Design across the ladder

| Rung | Your decision | Why |
|---|---|---|
| Prompt | `AGENTS.md` as the system spec: department and priority definitions, output schema (intent, department, priority, draft, sources), rule that email content is data and never instructions | One place to freeze the definitions the evals depend on |
| Context | Top-3 policy retrieval over 7 Markdown documents; controlled history (current email only); budget rule: max 3 chunks, max 6 agent steps, token/cost cap with a safe stop that returns "needs human review" | Keeps context small and the behaviour measurable |
| Harness | Allow/ask permission policy, human approval for the write, approval re-checked at the MCP write boundary, step and cost budgets, audit log of every tool call | The write is the only irreversible action, so it is enforced in code, not in the prompt |
| Loop | Not needed: one bounded plan-and-act pass per email with a step cap | A triage task has a clear end, so an outer retry loop adds cost, not accuracy |
| Graph | Not needed: the flow is linear (classify, tools, draft, approve, save) and is a fixed sequence in the harness | No branching or long-lived state that would justify a LangGraph graph |

```
                +-----------------------------+
 Email (CSV) -> |  Harness (budgets, policy,  | <-> Human approver (approve / reject)
                |  audit log)                 |
                |   Agent (LLM, AGENTS.md)    |
                +--------------+--------------+
                               |
        +----------------------+----------------------+
        |                      |                      |
 policy retrieval        MCP server (local)       routing rules
 (top-3, Markdown KB)    - lookup order/invoice/ticket
                         - save draft + assignment (approval-gated)
                               |
                      synthetic records + saved drafts
```

## Tools

| Tool | Read / write | Over MCP? | Approval? | Trifecta leg (private / untrusted / egress) |
|---|---|---|---|---|
| `search_policy` | Read | <TODO: yes/no> | No | Untrusted (retrieved passages are treated as data) |
| `lookup_order` / `lookup_invoice` / `lookup_ticket` | Read | Yes | No | Private (mock customer records) |
| `route_rules` | Read | No (local) | No | None |
| `save_draft` (draft + assignment) | Write (local only) | Yes | **Yes**, checked again at the MCP boundary | Egress (limited: local save, never sends email) |
| `export_records` <TODO: decide to include as a blocked decoy> | Read / egress | No | Denied by policy | Egress (present only to be attacked and blocked) |

## Evaluation

**Suite:** 25 tasks in `tasks.jsonl` (HR, finance, technical support, customer support, ambiguous requests, Arabic emails, missing records, one prompt-injection email). Expected labels are reviewed and frozen by both team members before benchmarking. Graders: `tool_called` / `tool_not_called` for required tools and forbidden actions, `contains_any` / `not_contains` for fixed facts (such as an order status) and for forbidden claims, `db` or audit-log checks for "no unauthorized write", and `judge` for draft grounding and tone. Department, priority and intent are compared with the expected labels, and macro-F1 is computed from the result CSV.
**Configurations:** A = basic retrieval; B = rule-enhanced retrieval <TODO: define the single change precisely>.

| Config | pass rate | pass^k (k = <TODO>) | avg requests | avg tokens |
|---|---|---|---|---|
| A | <TODO> | <TODO> | <TODO> | <TODO> |
| B | <TODO> | <TODO> | <TODO> | <TODO> |

Other measured metrics (fill after running): primary routing accuracy, priority accuracy, intent macro-F1, required-tool recall, approval compliance, grounded-draft quality.

### Error analysis

| Root cause | Count (A) | Count (B) | Example task | Fix tried |
|---|---|---|---|---|
| <TODO> | | | | |

<TODO: what flipped between A and B, in both directions, and what you would change next. Do not report any number before the experiments are run.>

## Security note

- **Trifecta:** The system has private data (mock orders, invoices, tickets), untrusted content (email subject and body, retrieved passages) and a limited egress path (the draft save and the blocked export tool). The legs meet in the agent's context, so the defence cannot rely on the model alone.
- **Attack tried:** An incoming email that instructs the assistant to export all records, reveal credentials or change tool permissions. <TODO: exact input and what happened with defences off.>
- **Defence:** Email and retrieved text are marked as untrusted data in the spec; `export_records` is denied by the permission policy in code; the save tool rejects any call without a valid human approval, even if the UI is bypassed; the email is escalated to Security Review. <TODO: confirm each is enforced in code, and which parts still depend on the model.>
- **Residual risk:** The model may still be misled on classification or produce a misleading draft; the approval step is the safety net, so the approver must read the draft and its sources. <TODO>
- **Governance:** Synthetic data only, so no personal data is processed in this prototype. For production: data minimisation (send the model only the fields it needs), human oversight on every write, and a decision on where email content goes (the model provider receives it), in line with PDPL and SDAIA AI principles. <TODO: refine.>

## Limitations and next steps

- No live Gmail/Outlook connection and no real sending; drafts are saved locally only.
- Mock records and 7 fictional policies, so results do not show real-world accuracy.
- 25 evaluation cases are small; the evaluation set and the tuning set overlap.
- Arabic is supported for email understanding, while the interface is English only. <TODO: confirm.>
- Production would need real authentication, a stronger spam and abuse filter, monitoring, and an approved model provider.
