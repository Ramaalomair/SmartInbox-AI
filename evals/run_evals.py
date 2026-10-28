"""B4: Evaluate the real SmartInbox AI agent with checkpointing and resume.

Run from repository root:
    python -m evals.run_evals
    python -m evals.run_evals --email-id E001
    python -m evals.run_evals --resume
    python -m evals.run_evals --resume-from evals/reports/results_full_34.json

No mock agent or write/approval endpoints are invoked.
Semantic behavior remains not_assessed.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.schemas import AgentResult, EmailInput, EvalCase
from agent.tools import bind_requester
from agent.integrated_tools import IntegratedAgentTools

ROOT = Path(__file__).resolve().parents[1]
EMAILS_PATH = ROOT / "data" / "emails.csv"
CASES_PATH = ROOT / "evals" / "cases" / "eval_cases.json"
REPORT_DIR = ROOT / "evals" / "reports" / "agent"

CHECK_NAMES = (
    "email_id", "intent", "primary_department", "secondary_department",
    "priority", "required_tools", "approval_declared",
)
CSV_COLUMNS = [
    "email_id", "expected_intent", "actual_intent", "expected_department",
    "actual_department", "expected_secondary_department", "actual_secondary_department",
    "expected_priority", "actual_priority", "required_tools", "tools_used",
    "status", "security_flags", "requires_approval", "approval_or_safe_handoff", "passed_structural_checks",
    "checks", "missing_tools", "expected_behavior", "behavior_assessment",
    "latency_seconds", "error", "agent_error",
]


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
        "approval_declared": (not case.requires_approval_for_write) or result.requires_approval,
    }
    # Diagnostic only: a manual review with no proposed action is a handoff,
    # NOT proof of approval or of correct security behavior.
    approval_or_safe_handoff = (
        not case.requires_approval_for_write
        or result.requires_approval
        or (
            result.status == "manual_review"
            and result.draft_subject is None
            and result.draft_body is None
            and result.proposed_action_id is None
        )
    )
    return {
        "checks": checks,
        "approval_or_safe_handoff": approval_or_safe_handoff,
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
        tools = IntegratedAgentTools(bind_requester(email.sender))
        raw_result = asyncio.run(process_email(email, tools))
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


def is_execution_error(row: dict[str, Any]) -> bool:
    """Do not score agent/runtime failures as incorrect classifications."""
    return bool(row.get("error")) or row.get("status") == "error"


def is_reusable(row: dict[str, Any]) -> bool:
    """Keep successful evaluations, including legitimate manual-review outcomes."""
    return not is_execution_error(row) and isinstance(row.get("checks"), dict)


def is_limit_error(row: dict[str, Any]) -> bool:
    """Recognize explicit provider quota errors; orchestrator may hide the 429."""
    msg = str(row.get("error") or "") + " " + str(row.get("agent_error") or "")
    return any(x in msg.lower() for x in (
        "ratelimiterror", "rate limit", "429", "free-models-per-day", "quota exceeded"
    ))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in rows if is_reusable(r)]
    accuracy = {
        name: round(sum(bool(r["checks"].get(name)) for r in completed) / len(completed), 4)
        if completed else None
        for name in CHECK_NAMES
    }
    passes = sum(bool(r.get("passed_structural_checks")) for r in completed)
    return {
        "total_cases": len(rows),
        "completed": len(completed),
        "errors": sum(is_execution_error(r) for r in rows),
        "structural_passes": passes,
        "approval_or_safe_handoff_rate": round(
            sum(bool(r.get("approval_or_safe_handoff")) for r in completed) / len(completed), 4
        ) if completed else None,
        "structural_pass_rate": round(passes / len(completed), 4) if completed else None,
        "accuracy_by_check_on_completed_cases": accuracy,
        "behavior_cases_not_assessed": sum(
            r.get("behavior_assessment") == "not_assessed" for r in rows
        ),
        "total_latency_seconds": round(sum(r.get("latency_seconds", 0) for r in rows), 3),
    }


def atomic_write(path: Path, content: str) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def save_reports(rows: list[dict[str, Any]], summary: dict[str, Any], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "real_process_email",
        "summary": summary,
        "cases": rows,
    }
    atomic_write(outdir / "results.json", json.dumps(report, ensure_ascii=False, indent=2))
    temp = outdir / "results.csv.tmp"
    with temp.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key), ensure_ascii=False)
                if isinstance(row.get(key), (list, dict)) else row.get(key)
                for key in CSV_COLUMNS
            })
    os.replace(temp, outdir / "results.csv")


def load_previous(path: Path, case_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("runner") != "real_process_email" or not isinstance(report.get("cases"), list):
        raise ValueError("Invalid B4 report format for resume")
    previous = {}
    for row in report["cases"]:
        case_id = row.get("email_id")
        if case_id in case_ids and is_reusable(row):
            previous[case_id] = row
    return previous


def main() -> int:
    parser = argparse.ArgumentParser(description="B4 SmartInbox AI evaluation runner")
    parser.add_argument("--email-id", action="append", help="Evaluate selected email ID(s)")
    parser.add_argument("--outdir", type=Path, default=REPORT_DIR)
    parser.add_argument("--resume", action="store_true", help="Reuse successful cases from outdir/results.json")
    parser.add_argument("--resume-from", type=Path, help="Reuse successful cases from a specified JSON report")
    args = parser.parse_args()

    if args.resume and args.resume_from:
        parser.error("Use either --resume or --resume-from, not both")

    try:
        from agent.orchestrator import process_email
    except ImportError as exc:
        print(f"ERROR: Real process_email is not available: {exc}", file=sys.stderr)
        return 2

    emails = load_emails(EMAILS_PATH)
    cases = load_cases(CASES_PATH)
    if args.email_id:
        selected = set(args.email_id)
        unknown = selected - {case.email_id for case in cases}
        if unknown:
            parser.error(f"Unknown case ID(s): {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.email_id in selected]

    resume_path = args.resume_from or (args.outdir / "results.json" if args.resume else None)
    previous = load_previous(resume_path, {c.email_id for c in cases}) if resume_path else {}
    if previous:
        print(f"Reusing {len(previous)} successful cases from {resume_path}")

    rows: list[dict[str, Any]] = []
    stopped = False
    for i, case in enumerate(cases, 1):
        if case.email_id in previous:
            row = previous[case.email_id]
            print(f"[{i}/{len(cases)}] {case.email_id} ... REUSED", flush=True)
        else:
            print(f"[{i}/{len(cases)}] {case.email_id} ...", flush=True)
            row = run_one(case, emails, process_email)
            outcome = "ERROR" if is_execution_error(row) else (
                "PASS" if row["passed_structural_checks"] else "FAIL"
            )
            print(f"  {outcome} ({row.get('latency_seconds', 0):.2f}s)", flush=True)
        rows.append(row)
        save_reports(rows, summarize(rows), args.outdir)  # checkpoint after every case

        # The orchestrator currently masks HTTP 429 as orchestration_failed.
        # Stop on any orchestration failure to avoid wasting remaining API calls.
        agent_code = (row.get("agent_error") or {}).get("code")
        if is_execution_error(row) and (
            is_limit_error(row) or agent_code == "orchestration_failed"
        ):
            print("STOP: API/orchestration failure. Results saved; resume later.", file=sys.stderr)
            stopped = True
            break

    summary = summarize(rows)
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nReports: {args.outdir / 'results.json'} and {args.outdir / 'results.csv'}")
    if stopped:
        print("Run again with --resume when the API is available.")
        return 2
    return 1 if summary["errors"] or summary["structural_passes"] != summary["total_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
