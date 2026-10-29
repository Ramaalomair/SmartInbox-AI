
# SmartInbox AI

> SmartInbox AI helps support teams classify and route incoming emails, retrieve relevant information, and prepare replies for human approval.

**Track:** 2 · Your harness  
**Team:** Remas Alsubaie, Rama Alomair  
**Course:** [SDAIA Academy — Agentic AI](https://github.com/SDAIAAcademy/agentic-ai)  
**Model(s):** `nvidia/nemotron-3.5-super-vl:free` via OpenRouter, used in both A/B configurations to compare retrieval methods under the same model.

## Problem

Teams handling shared inboxes receive customer support, finance, HR, and technical requests. Manually classifying emails, assigning priorities, and finding the right information takes time and can lead to inconsistent responses. Replies may also include unverified claims or expose information to unauthorized requesters.

SmartInbox AI classifies each email, routes it to the appropriate department, retrieves relevant company policies and authorized records, and prepares a draft for human review.

The project uses synthetic emails and company data. Success means accurate classification and routing, correct tool usage, and adherence to the approval requirement.

## How to run

Set `OPENROUTER_API_KEY` in a `.env` file in the project root.

Install the project dependencies:

```bash
uv sync
```

Launch the Streamlit interface:

```bash
uv run streamlit run app.py
```

Run the agent A/B evaluation:

```bash
uv run python -m evals.run_evals --config A --repeats 2 --outdir evals/reports/config_A
uv run python -m evals.run_evals --config B --repeats 2 --outdir evals/reports/config_B
```

Run the behavioral evaluation using the saved real-agent outputs:

```bash
uv run python -m evals.evaluate_behavior --all
```

To reevaluate all cases with the current LLM Judge rubric and authorized ground-truth fixtures:

```bash
uv run python -m evals.evaluate_behavior --all --refresh
```

The behavioral evaluation uses OpenRouter and requires an available judge model, configured using `OPENROUTER_JUDGE_MODEL` or `OPENROUTER_MODEL`.

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
              Approval-checked local save via MCP
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

### Agent A/B Evaluation

**Suite:** 34 synthetic email tasks in `evals/cases/eval_cases.json`, covering support, finance, HR, technical requests, and security-sensitive cases; 2 repetitions per task (68 attempts per configuration).

**Graders:** Structural checks for email ID, intent, primary and secondary departments, priority, required tools, and approval declaration.

**Configurations:** A = TF-IDF retrieval (baseline); B = Hybrid RRF retrieval (TF-IDF + multilingual E5). Retrieval method is the one configuration change.

| Config | Pass rate | pass^k (k = 2) | Avg requests | Avg tokens |
|---|---:|---:|---:|---:|
| A | 69.12% | 67.65% | 1.79 | 1,738.56 |
| B | **75.00%** | **73.53%** | 1.88 | 1,767.34 |

### Error Analysis

| Root cause | Count (A) | Count (B) | Example task | Fix tried |
|---|---:|---:|---|---|
| Incorrect intent classification | 4 | 6 | E004, E024 | Hybrid RRF retrieval |
| Incorrect primary department routing | 4 | 4 | E024 | Hybrid RRF retrieval |
| Incorrect secondary department routing | 2 | 2 | E031 | Hybrid RRF retrieval |
| Incorrect priority classification | 7 | 7 | E012, E013 | Hybrid RRF retrieval |
| Missing required tools | 10 | 6 | E018, E033 | Hybrid RRF retrieval |
| Approval declaration check failure | 14 | 8 | E019, E033 | Hybrid RRF retrieval |

*Counts are failed checks across 68 attempts per configuration; one attempt can fail multiple checks.*

**What flipped between A and B:** Switching from TF-IDF to Hybrid RRF improved three tasks (E018, E019, and E033), which changed from failing to passing both repetitions. These improvements were related to required tool usage and approval declaration checks. However, E004 regressed from passing to failing due to incorrect intent classification.

Overall, Configuration B increased pass² from 67.65% to 73.53%, although intent classification errors increased from 4 to 6 attempts.

**What we would change next:** Improve intent classification for ambiguous requests, refine priority and department-routing rules, and strengthen validation of required tool usage and approval declarations. Then rerun the A/B evaluation to measure the impact.

### Behavioral Evaluation

We evaluated 34 saved real-agent outputs using `openai/gpt-4.1-mini` through OpenRouter as an LLM-as-a-Judge, supplemented by deterministic approval checks.

The judge assessed five dimensions:

1. Groundedness
2. Task completion
3. Safety
4. Language
5. Expected behavior

To verify factual claims, the judge used retrieved evidence and authorized synthetic company records from `orders.csv`, `invoices.csv`, and `tickets.csv`.

| Metric | Result |
|---|---:|
| Emails evaluated | 34 |
| Total LLM judgments | 170 |
| PASS | 169 |
| FAIL | 0 |
| REVIEW | 1 |
| Pass rate (excluding REVIEW) | 100% |
| Deterministic checks passed | 68/68 |

**Unresolved case:** E007 remains marked `REVIEW` for groundedness because the available evidence does not verify the claim that NovaTech carries wireless mice.

During evaluation development, the judge initially flagged some correct order and invoice statements as unsupported because the saved evidence contained policy excerpts but not the relevant database records. After authorized ground-truth records were supplied, those claims were verified.

The final evaluation combines rule-based checks with LLM judgments. Results describe performance on the tested synthetic cases and do not establish perfect agent accuracy. LLM judgments can be imperfect, and unresolved evidence is not automatically treated as a pass.

Behavioral evaluation reports are saved under `evals/reports/behavior/`.

## Security Note

- **Trifecta:** Untrusted incoming emails and retrieved text meet access to synthetic private records and a restricted local write tool in the agent workflow.
- **Attack tried:** Prompt-injection and unauthorized-access cases tested whether the agent would perform unauthorized lookups, reveal records, or write without approval. The security suite passed 40/40 tests; behavior without defenses was not separately measured.
- **Defence:** Tool-level access checks and code-enforced approval verification for local draft saving, alongside instructions to treat incoming content as untrusted.
- **Residual risk:** Incorrect classification, unsupported draft statements, and new injection patterns remain possible.
- **Governance:** The prototype uses synthetic data. Production use would require data minimization, human oversight, and compliance review under applicable Saudi PDPL and SDAIA principles.

## Limitations and Next Steps

- The A/B evaluation covers 34 synthetic tasks, not production email traffic.
- Behavioral evaluation uses model-based judgments and authorized synthetic records; it does not guarantee complete factual accuracy or production safety.
- The behavioral evaluation assesses one saved real-agent output per email, while the A/B structural evaluation uses two repetitions per email per configuration.
- The system does not support live email sending; approved drafts are saved locally.
- Next steps include improving classification and routing consistency, expanding factual verification for product inquiries, and evaluating performance on a broader dataset.

## Course Reference

This project was developed as part of the **SDAIA Academy Agentic AI course**, following the Track 2 capstone approach.

**Course repository:** [SDAIAAcademy/agentic-ai](https://github.com/SDAIAAcademy/agentic-ai)
