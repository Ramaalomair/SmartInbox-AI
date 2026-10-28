# SmartInbox AI

> SmartInbox AI helps customer support, finance, HR, and technical teams triage a shared inbox: it classifies incoming emails, routes them by department and priority, consults synthetic records and company policies, and prepares draft replies for human review before approved local saving.

**Track:** 2 · Your harness  
**Team:** Remas Alsubaie, Rama Alomair  
**Program Repository:** [SDAIA Academy – Agentic AI](https://github.com/SDAIAAcademy/agentic-ai)  
**Model(s):** `nvidia/nemotron-3.5-super-vl:free` via OpenRouter (OpenAI-compatible API). This is the model used for the reported final evaluation; the evaluation below documents measured performance, not a comparative model benchmark.

## Problem

Companies receiving HR, finance, technical support, and customer inquiries through a shared inbox often classify and route messages manually. This slows responses, makes prioritization inconsistent, and risks replies that assert unverified facts—for example, claiming a refund was completed without checking the order. Incoming emails and retrieved documents can also contain prompt-injection attempts.

SmartInbox AI uses a fictional company, **NovaTech**, and synthetic email, order, invoice, and ticket records. The agent classifies each email, selects departments and priority, performs relevant lookups and knowledge retrieval, and attempts to prepare a grounded draft. The intended completion criteria are correct routing, verifiable claims, appropriate escalation, and no unauthorized write. These criteria are **not all fully established** by the structural evaluation reported below.

## How to run

From the project root, activate the project's Python environment and install the dependencies required by your local setup. Configure `OPENROUTER_API_KEY` in `.env`. Approval-token configuration may also be required for approved writes; consult `mcp_server/approval.py` and the project's configuration before enabling that flow.

**Run the existing evaluation suite:**

```bash
python -m evals.run_evals --outdir evals/reports/final_evaluation
```

**Resume an interrupted evaluation without rerunning completed cases:**

```bash
python -m evals.run_evals --outdir evals/reports/final_evaluation --resume
```

**Run security tests:**

```bash
python -m pytest evals/test_security.py -v
```

**Run approval and access-control tests:**

```bash
python -m pytest tests/test_approval.py tests/test_access_control.py -v
```

**User interface:** `app.py` contains the application UI and an agent execution path. Confirm its framework and entry-point command in the source before publishing a launch command; the earlier `uv run smartinbox.py ...` commands were placeholders and do not match the current file tree.

## Design across the ladder

| Rung | Your decision | Why |
|---|---|---|
| Prompt | `AGENTS.md` documents agent behavior; `agent/classification.py` defines structured intent, department, and priority classification and includes instructions to treat email content as untrusted. | Consistent labels and a clear trust boundary for incoming messages. |
| Context | Markdown knowledge base with **8 documents**; retrieval components include `agent/retrieval.py`, `agent/hybrid_retrieval.py`, and `agent/rag.py`. Configured retrieval top-k is 3; the agent has a bounded context and output budget. | Retrieve relevant policy context without supplying the entire knowledge base to every request. |
| Harness | `agent/orchestrator.py` coordinates classification, tool use, draft generation, and status handling. `agent/schemas.py` sets limits, including `max_steps=6`, `max_input_tokens=12000`, `max_output_tokens=6000`, and `max_estimated_cost_usd=0.10` in the evaluated configuration. | Limit cost and execution; stop safely when the budget or workflow cannot proceed. |
| Loop | A bounded per-email orchestration flow rather than an open-ended autonomous loop. | Email triage has a defined scope and should not continue indefinitely. |
| Graph | The current implementation is code-orchestrated; no dedicated graph framework is evidenced in the current file structure. | Keeps the prototype simpler while supporting explicit checks and handoffs. |

```text
Incoming email (synthetic CSV / UI)
              |
              v
      App / Agent Harness ----------------> Human reviewer
      (orchestrator, budgets,                | approve / reject
       classification, audit)                |
              |                              v
      +-------+--------+               Approval token
      |                |                     |
      v                v                     v
Knowledge retrieval   Record lookups     MCP save_draft_mcp
(8 Markdown docs)     (synthetic CSVs)   (server + service)
      |                |                     |
      +-------+--------+                     v
              |                       Local approved drafts
              v                       and audit log
        Grounded draft
```

## Tools

| Tool / component | Read / write | Over MCP? | Approval? | Trifecta leg (private / untrusted / egress) |
|---|---|---|---|---|
| Knowledge retrieval (`agent/retrieval.py`, `agent/hybrid_retrieval.py`) | Read | Not established from current evidence | No write approval | Untrusted retrieved content |
| Record lookup components (`agent/integrated_tools.py`, `agent/mock_tools.py`, `agent/tool_execution.py`) | Read | Verify individual lookup transport in code | Access control required; no write approval | Private synthetic records |
| Classification and routing (`agent/classification.py`, `agent/tool_selection.py`) | Decision / read | No MCP requirement established | No | Untrusted email input |
| `save_draft_mcp` (`mcp_server/server.py`) | Write, local | **Yes** | **Yes**; approval token verified by the service | Limited local write / egress-like action |
| Audit logging (`mcp_server/service.py`) | Write, local | Service-side | System-controlled | Audit trail |

The original template's `export_records` decoy is **not listed as an implemented tool** here because the provided source listing does not establish its existence.

## Evaluation

**Suite:** **34 synthetic email cases** in `evals/eval_cases.json`, run by `evals/run_evals.py`. The suite checks email ID, intent, primary and secondary departments, priority, required tools, and approval declaration. Results are saved after each case, allowing interrupted runs to resume.

**Configurations:** A = earlier baseline evaluation; B = updated classification/routing instructions and execution budget. These are **successive development snapshots**, not a controlled one-variable A/B experiment. Reported results therefore show observed changes, not causal attribution to a single modification.

| Config | Structural pass rate | pass^k | Avg requests | Avg tokens |
|---|---:|---|---|---|
| A · Baseline | 20/34 (**58.82%**) | Not measured | Not measured | Not measured |
| B · Final | 26/34 (**76.47%**) | Not measured | Not measured | Not measured |

| Structural check | A · Baseline | B · Final |
|---|---:|---:|
| Email ID | Not available | 34/34 (100.00%) |
| Intent | 94.12% | 31/34 (91.18%) |
| Primary department | 88.24% | 32/34 (94.12%) |
| Secondary department | 94.12% | 33/34 (97.06%) |
| Priority | 73.53% | 30/34 (88.24%) |
| Required tools | 91.18% | 31/34 (91.18%) |
| Approval declared | 79.41% | 30/34 (88.24%) |

**Final execution:** 34/34 cases completed, **0 execution errors**, with **330.457 seconds** total reported latency. The supplementary `approval_or_safe_handoff_rate` was **100%**, but this combines approval declarations with certain safe manual-review handoffs; it is **not** proof that every email was approved or that all security behaviors passed.

**Important evaluation boundary:** `behavior_cases_not_assessed = 34`. The current runner does not automatically grade whether draft text is factually grounded, whether refusals are semantically correct, or whether all expected behaviors occurred. These require separate behavioral checks and/or human review.

### Retrieval evaluation

**Suite:** 16 retrieval cases in `evals/retrieval_cases.json`, evaluated with `python -m evals.evaluate_retrieval`. `Hit@3` measures whether a relevant result appears in the first three results; `MRR@3` rewards higher ranking of the first relevant result.

| Retrieval method | Hit@3 | MRR@3 | Cases with hit |
|---|---:|---:|---:|
| TF-IDF | 68.75% | 0.6146 | 11/16 |
| Semantic E5 | 93.75% | 0.8646 | 15/16 |
| Hybrid RRF | **93.75%** | **0.9062** | **15/16** |

Hybrid RRF has the highest measured MRR@3, while Semantic E5 and Hybrid RRF tie on Hit@3. Case `R002` failed for all three retrieval methods. These are retrieval-only metrics, separate from the 34-email structural evaluation.

### Error analysis

| Root cause / metric | Baseline | Final | Interpretation / fix tried |
|---|---:|---:|---|
| Overall structural failures | 14/34 | 8/34 | Improved classification rules and execution configuration; failures remain. |
| Priority mismatches | 9/34 | 4/34 | Clarified HIGH/MEDIUM/LOW criteria, including operational impact rather than urgency words alone. |
| Primary-department mismatches | 4/34 | 2/34 | Refined routing instructions. |
| Intent mismatches | 2/34 | 3/34 | Small regression; further prompt changes may trade off performance elsewhere. |
| Required-tool mismatches | 3/34 | 3/34 | No aggregate improvement; examine individual cases before further tuning. |
| Approval-declaration mismatches | 7/34 | 4/34 | Separate strict approval declaration from manual-review safe handoff. |

**Next evaluation change:** Add a behavioral grader and manual review for high-risk cases rather than optimizing solely for structural accuracy. Retest on an independent holdout suite to reduce tuning-set bias.

## Security note

- **Trifecta:** The agent receives untrusted email content and retrieved policy text, can query private-like synthetic order/invoice/ticket records, and has a restricted local write path through the MCP service. These capabilities intersect at the orchestration layer.
- **Attack tried:** The project contains injection fixtures (`evals/injection_case.json`, `evals/untrusted_kb_injection.md`) and tests for unauthorized lookup, indistinguishable unauthorized/unknown record responses, data minimization, unapproved writes, tampered payloads, prompt injection, cross-customer record access, and permission overrides (`evals/test_security.py`).
- **Defence:** Structured routing and untrusted-input instructions are prompt-level controls. The separate MCP approval service (`mcp_server/approval.py`, `mcp_server/service.py`) provides code-level approval-token verification for local draft writes; access-control tests exercise record boundaries.
- **Observed security result:** The provided Pytest JUnit XML reports **40/40 tests passing**, **0 failures**, **0 errors**, **0 skipped**, in **46.214 seconds**. Breakdown: **13** access-control tests (`tests/test_access_control.py`), **16** approval/MCP tests (`tests/test_approval.py`), and **11** security/injection tests (`evals/test_security.py`). This demonstrates the listed checks passed in that run, not comprehensive security assurance. No unprotected-before/after attack comparison has been established.
- **Residual risk:** Model-generated classifications and drafts can still be incorrect or misleading. The structural evaluation does not establish resistance to every prompt injection or verify every draft's grounding. Human review remains necessary.
- **Governance:** The prototype uses synthetic data. Production would require privacy and data-minimization controls, access logging, explicit authorization, retention rules, human oversight, and a review of provider-side data processing and applicable Saudi PDPL / SDAIA guidance.

## Limitations and next steps

- No confirmed live email-provider integration or real message sending; the documented write path saves approved drafts locally.
- Evaluations use synthetic NovaTech data and a small 34-case suite; the results are not production accuracy estimates.
- The baseline and final suites reflect iterative development, not a controlled model benchmark.
- All **34 expected behaviors remain unassessed by the structural runner**; passing the 40 dedicated tests does not substitute for full per-email semantic grading. Add behavioral and end-to-end testing, especially for security-sensitive cases.
- UI entry point and installation instructions should be verified against `app.py` and the project's dependency files before release.
- Authentication, monitoring, rate-limit handling, production secrets management, and data-governance controls need deployment-level review.
