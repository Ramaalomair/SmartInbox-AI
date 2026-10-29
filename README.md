# SmartInbox AI

> SmartInbox AI helps support teams classify and route incoming emails, retrieve relevant information, and prepare replies for human approval.

**Track:** 2 · Your harness  
**Team:** Remas Alsubaie, Rama Alomair  
**Model(s):** `nvidia/nemotron-3.5-super-vl:free` via OpenRouter, used in both A/B configurations to compare retrieval methods under the same model.

## Problem

Teams handling shared inboxes receive customer support, finance, HR, and technical requests. Manually classifying emails, assigning priorities, and finding the right information takes time and can lead to inconsistent responses. Replies may also include unverified claims or expose information to unauthorized requesters. SmartInbox AI classifies each email, routes it to the appropriate department, retrieves relevant company policies and authorized records, and prepares a draft for human review. The project uses synthetic emails and company data. Success means accurate classification and routing, correct tool usage, and adherence to the approval requirement.

## How to run

Set `OPENROUTER_API_KEY` in `.env` and run the following commands from the project root:

```bash
uv sync
uv run python -m evals.run_evals --config A --repeats 2 --outdir evals/reports/config_A
uv run python -m evals.run_evals --config B --repeats 2 --outdir evals/reports/config_B
```

## Design across the ladder

| Rung | Your decision | Why |
|---|---|---|
| Prompt | Structured classification and drafting instructions | Consistent intent, routing, and draft outputs. |
| Context | Company knowledge retrieval using TF-IDF (A) or Hybrid RRF (B) | Supply relevant policy context for each email. |
| Harness | Orchestrator, tool access controls, approval checks, and execution budgets | Control tool use and prevent unauthorized actions. |
| Loop | Bounded processing flow for each email | Each request has a defined endpoint. |
| Graph | Code-based workflow rather than a separate graph framework | The workflow is handled by the existing orchestrator. |

```text
                 Incoming email
                       |
                       v
                Agent / Harness
                       |
                       v
                  LLM (OpenRouter)
                       |
             Classification & routing
                       |
             +---------+---------+
             |                   |
             v                   v
      Knowledge retrieval   Record lookup tools
      (company policies)    (synthetic records)
             |                   |
             +---------+---------+
                       |
                       v
                  Draft reply
                       |
                       v
                 Human approval
                       |
                       v
              MCP-approved local save
```

## Tools

| Tool | Read / write | Over MCP? | Approval? | Trifecta leg (private / untrusted / egress) |
|---|---|---|---|---|
| `search_company_knowledge` | Read | No | No | Untrusted retrieved content |
| `lookup_order` | Read | No | Access-controlled | Private synthetic records |
| `lookup_invoice` | Read | No | Access-controlled | Private synthetic records |
| `lookup_ticket` | Read | No | Access-controlled | Private synthetic records |
| `save_draft_mcp` | Write | Yes | Yes | Local write action |

## Evaluation

**Suite:** 34 synthetic email tasks in `evals/eval_cases.json`, covering support, finance, HR, technical requests, and security-sensitive cases; 2 repetitions per task (68 attempts per configuration). **Graders:** structural checks for email ID, intent, primary and secondary departments, priority, required tools, and approval declaration.

**Configurations:** A = TF-IDF retrieval (baseline); B = Hybrid RRF retrieval (TF-IDF + multilingual E5). Retrieval method is the one configuration change.

| Config | pass rate | pass^k (k = 2) | avg requests | avg tokens |
|---|---:|---:|---:|---:|
| A | 69.12% | 67.65% | 1.79 | 1,738.56 |
| B | **75.00%** | **73.53%** | 1.88 | 1,767.34 |

### Error analysis

| Root cause | Count (A) | Count (B) | Example task | Fix tried |
|---|---:|---:|---|---|
| Incorrect intent classification | 4 | 6 | E004, E024 | Hybrid RRF retrieval |
| Incorrect primary department routing | 4 | 4 | E024 | Hybrid RRF retrieval |
| Incorrect secondary department routing | 2 | 2 | E031 | Hybrid RRF retrieval |
| Incorrect priority classification | 7 | 7 | E012, E013 | Hybrid RRF retrieval |
| Missing required tools | 10 | 6 | E018, E033 | Hybrid RRF retrieval |
| Incorrect approval declaration | 14 | 8 | E019, E033 | Hybrid RRF retrieval |

*Counts are failed checks across 68 attempts per configuration; one attempt can fail multiple checks.*

**A → B:** E018, E019, and E033 changed from failing to passing both repetitions. E004 changed from passing to failing. Next, we would improve intent classification and investigate persistent routing and priority errors.

## Security note

- **Trifecta:** Untrusted incoming emails and retrieved text meet access to synthetic private records and a restricted local write tool in the agent workflow.
- **Attack tried:** Prompt-injection and unauthorized-access cases tested whether the agent would perform unauthorized lookups, reveal records, or write without approval. The security suite passed 40/40 tests; behavior without defenses was not separately measured.
- **Defence:** Tool-level access checks and code-enforced approval verification for local draft saving, alongside instructions to treat incoming content as untrusted.
- **Residual risk:** Incorrect classification, unsupported draft statements, and new injection patterns remain possible.
- **Governance:** The prototype uses synthetic data. Production use would require data minimization, human oversight, and compliance review under applicable Saudi PDPL and SDAIA principles.

## Limitations and next steps

- The A/B evaluation covers 34 synthetic tasks, not production email traffic.
- Graders check structural outcomes; they do not independently score draft quality or factual grounding.
- The system does not establish live email sending; approved drafts are saved locally.
- Next steps include improving persistent classification and routing errors, adding draft-quality checks, and testing with a broader dataset.
