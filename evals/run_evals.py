"""B4: Evaluate the real SmartInbox AI email agent against labelled fixtures.

Run from repository root:
    uv run python -m evals.run_evals
    uv run python -m evals.run_evals --email-id E001

This runner never calls the mock agent or the write/approval endpoints.
Behavioral requirements requiring semantic judgment are marked 'not_assessed'.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.schemas import AgentResult, EmailInput, EvalCase
from agent.tools import bind_requester

ROOT = Path(__file__).resolve().parents[1]
EMAILS_PATH = ROOT / "data" / "emails.csv"
CASES_PATH = ROOT / "evals" / "eval_cases.json"
REPORT_DIR = ROOT / "evals" / "reports"


def load_emails(path: Path) -> dict[str, EmailInput]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        emails = [EmailInput.model_validate(row) for row in csv.DictReader(file)]
    if len({e.email_id for e in emails}) != len(emails):
        raise ValueError("Duplicate email_id in emails.csv")
    return {e.email_id: e for e in emails}


def load_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("eval_cases.json must contain a JSON array")
    # Ignore optional metadata (e.g. test_tag) without weakening the shared contract.
    allowed = set(EvalCase.model_fields)
    cases = [EvalCase.model_validate({k: v for k, v in item.items() if k in allowed}) for item in raw]
    if len({c.email_id for c in cases}) != len(cases):
        raise ValueError("Duplicate email_id in eval_cases.json")
    return cases


def evaluate(case: EvalCase, result: AgentResult) -> dict[str, Any]:
    actual_tools = set(result.tools_used) | {step.tool for step in result.trace}
    required_tools = set(case.required_tools)
    checks = {
        "email_id": result.email_id == case.email_id,
        "intent": result.intent == case.expected_intent,
        "primary_department": result.primary_department == case.expected_department,
        "secondary_department": result.secondary_department == case.expected_secondary_department,
        "priority": result.priority == case.expected_priority,
        "required_tools": required_tools.issubset(actual_tools),
        # This is only a declared approval requirement; enforcement needs separate security tests.
        "approval_declared": (not case.requires_approval_for_write) or result.requires_approval,
    }
    # No draft may be considered approved/saved by the runner. No write tools are called here.
    return {
        "checks": checks,
        "missing_tools": sorted(required_tools - actual_tools),
        "expected_behavior": case.expected_behavior,
        "behavior_assessment": "not_assessed",
        "behavior_note": "Requires dedicated assertions or human review; label alone is not proof.",
        "passed_structural_checks": all(checks.values()),
    }


def run_one(case: EvalCase, emails: dict[str, EmailInput], process_email: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "email_id": case.email_id,
        "expected_intent": case.expected_intent,
        "expected_department": case.expected_department,
        "expected_secondary_department": case.expected_secondary_department,
        "expected_priority": case.expected_priority,
        "required_tools": case.required_tools,
        "expected_behavior": case.expected_behavior,
        "passed_structural_checks": False,
        "behavior_assessment": "not_assessed",
        "error": None,
    }
    if case.email_id not in emails:
        row["error"] = "Missing email fixture in data/emails.csv"
        return row

    email = emails[case.email_id]
    start = time.perf_counter()
    try:
        # Requester is bound from the trusted fixture, never from email body instructions.
        raw_result = process_email(email, bind_requester(email.sender))
        result = AgentResult.model_validate(raw_result)
        row.update(evaluate(case, result))
        row.update({
            "actual_intent": result.intent,
            "actual_department": result.primary_department,
            "actual_secondary_department": result.secondary_department,
            "actual_priority": result.priority,
            "status": result.status,
            "tools_used": result.tools_used,
            "security_flags": result.security_flags,
            "requires_approval": result.requires_approval,
            "budget": result.budget.model_dump(mode="json"),
            "agent_error": result.error.model_dump(mode="json") if result.error else None,
            "result": result.model_dump(mode="json"),
        })
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["latency_seconds"] = round(time.perf_counter() - start, 3)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in rows if r.get("checks") is not None]
    check_names = (
        "email_id", "intent", "primary_department", "secondary_department",
        "priority", "required_tools", "approval_declared"
    )
    accuracy = {
        name: round(sum(bool(r["checks"][name]) for r in completed) / len(completed), 4)
        if completed else None
        for name in check_names
    }
    return {
        "total_cases": len(rows),
        "completed": len(completed),
        "errors": len(rows) - len(completed),
        "structural_passes": sum(bool(r.get("passed_structural_checks")) for r in rows),
        "structural_pass_rate": round(
            sum(bool(r.get("passed_structural_checks")) for r in rows) / len(rows), 4
        ) if rows else None,
        "accuracy_by_check_on_completed_cases": accuracy,
        "behavior_cases_not_assessed": len(rows),
        "total_latency_seconds": round(sum(r.get("latency_seconds", 0) for r in rows), 3),
    }


def save_reports(rows: list[dict[str, Any]], summary: dict[str, Any], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "real_process_email",
        "summary": summary,
        "cases": rows,
    }
    (outdir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = [
        "email_id", "expected_intent", "actual_intent", "expected_department",
        "actual_department", "expected_secondary_department", "actual_secondary_department",
        "expected_priority", "actual_priority", "required_tools", "tools_used",
        "status", "security_flags", "requires_approval", "passed_structural_checks",
        "checks", "missing_tools", "expected_behavior", "behavior_assessment",
        "latency_seconds", "error", "agent_error",
    ]
    with (outdir / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key), ensure_ascii=False)
                if isinstance(row.get(key), (list, dict)) else row.get(key)
                for key in columns
            })


def main() -> int:
    parser = argparse.ArgumentParser(description="B4 SmartInbox AI evaluation runner")
    parser.add_argument("--email-id", action="append", help="Evaluate only selected email ID(s)")
    parser.add_argument("--outdir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    try:
        from agent.agent import process_email
    except ImportError as exc:
        print(f"ERROR: Real process_email is not available: {exc}", file=sys.stderr)
        print("Ask Developer A to expose process_email(email, tools) -> AgentResult.", file=sys.stderr)
        return 2

    emails = load_emails(EMAILS_PATH)
    cases = load_cases(CASES_PATH)
    if args.email_id:
        selected = set(args.email_id)
        unknown = selected - {case.email_id for case in cases}
        if unknown:
            parser.error(f"Unknown case ID(s): {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.email_id in selected]

    rows = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.email_id} ...", flush=True)
        row = run_one(case, emails, process_email)
        rows.append(row)
        outcome = "ERROR" if row["error"] else "PASS" if row["passed_structural_checks"] else "FAIL"
        print(f"  {outcome} ({row.get('latency_seconds', 0):.2f}s)", flush=True)

    summary = summarize(rows)
    save_reports(rows, summary, args.outdir)
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nReports: {args.outdir / 'results.json'} and {args.outdir / 'results.csv'}")
    return 1 if summary["errors"] or summary["structural_passes"] != summary["total_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
