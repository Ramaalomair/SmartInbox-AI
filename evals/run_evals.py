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
from agent.llm import start_usage_tracking, get_usage, stop_usage_tracking

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
    "repeat_index", "request_count", "input_tokens", "output_tokens", "total_tokens",
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


def run_one(
    case: EvalCase,
    emails: dict[str, EmailInput],
    process_email: Any,
    retrieval_method: str = "hybrid",
) -> dict[str, Any]:
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
    usage_token = start_usage_tracking()
    try:
        tools = IntegratedAgentTools(
            bind_requester(email.sender), retrieval_method=retrieval_method
        )
        raw_result = asyncio.run(process_email(email, tools))
        result = AgentResult.model_validate(raw_result)
        row.update(evaluate(case, result))
        # The orchestrator owns a nested usage context. Its budget is captured
        # before that context is reset, so use it for provider token counts.
        row["input_tokens"] = result.budget.input_tokens
        row["output_tokens"] = result.budget.output_tokens
        row["total_tokens"] = (
            result.budget.input_tokens + result.budget.output_tokens
            if result.budget.input_tokens is not None
            and result.budget.output_tokens is not None
            else None
        )
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
        usage = get_usage()
        # The orchestrator reuses this tracking context, so its LLM calls
        # are counted here before the context is reset.
        row["request_count"] = usage.get("request_count")
        row.setdefault("input_tokens", None)
        row.setdefault("output_tokens", None)
        row.setdefault("total_tokens", None)
        stop_usage_tracking(usage_token)
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
    measured_requests = [r["request_count"] for r in rows if r.get("request_count") is not None]
    measured_tokens = [r["total_tokens"] for r in rows if r.get("total_tokens") is not None]
    return {
        "avg_requests": round(sum(measured_requests) / len(measured_requests), 2) if len(measured_requests) == len(rows) and rows else None,
        "avg_tokens": round(sum(measured_tokens) / len(measured_tokens), 2) if len(measured_tokens) == len(rows) and rows else None,
        "request_measurements": len(measured_requests),
        "token_measurements": len(measured_tokens),
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


def load_previous(
    path: Path, case_ids: set[str], expected_config: str
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("runner") != "real_process_email" or not isinstance(report.get("cases"), list):
        raise ValueError("Invalid B4 report format for resume")
    saved_config = report.get("summary", {}).get("config")
    if saved_config != expected_config:
        raise ValueError(
            f"Report config={saved_config!r}, requested config={expected_config!r}; refusing to mix runs"
        )
    previous = {}
    for row in report["cases"]:
        case_id = row.get("email_id")
        if case_id in case_ids and is_reusable(row):
            previous[case_id] = row
    return previous



def summarize_repeated(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """pass^k = fraction of cases passing all k independent attempts."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["email_id"], []).append(row)
    complete = {key: group for key, group in groups.items() if len(group) == k}
    all_passes = sum(
        all(not is_execution_error(r) and r.get("passed_structural_checks") for r in group)
        for group in complete.values()
    )
    request_values = [r.get("request_count") for r in rows]
    token_values = [r.get("total_tokens") for r in rows]
    return {
        "k": k,
        "cases_completed_all_repeats": len(complete),
        "total_attempts": len(rows),
        "pass^k": round(all_passes / len(complete), 4) if complete else None,
        "cases_passing_all_repeats": all_passes,
        "attempt_pass_rate": round(sum(bool(r.get("passed_structural_checks")) and not is_execution_error(r) for r in rows) / len(rows), 4) if rows else None,
        "avg_requests": round(sum(request_values) / len(rows), 2) if rows and all(v is not None for v in request_values) else None,
        "avg_tokens": round(sum(token_values) / len(rows), 2) if rows and all(v is not None for v in token_values) else None,
        "request_measurements": sum(v is not None for v in request_values),
        "token_measurements": sum(v is not None for v in token_values),
        "errors": sum(is_execution_error(r) for r in rows),
    }

def main() -> int:
    parser = argparse.ArgumentParser(description="B4 SmartInbox AI evaluation runner")
    parser.add_argument("--email-id", action="append", help="Evaluate selected email ID(s)")
    parser.add_argument("--repeats", type=int, default=1, help="Independent attempts per email (e.g., 3 for pass^3)")
    parser.add_argument("--config", choices=["A", "B"], required=True, help="A = TF-IDF; B = Hybrid RRF")
    parser.add_argument("--outdir", type=Path, default=REPORT_DIR)
    parser.add_argument("--resume", action="store_true", help="Reuse successful cases from outdir/results.json")
    parser.add_argument("--resume-from", type=Path, help="Reuse successful cases from a specified JSON report")
    args = parser.parse_args()
    retrieval_method = {"A": "tfidf", "B": "hybrid"}[args.config]

    def report_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        result = summarize_repeated(rows, args.repeats) if args.repeats > 1 else summarize(rows)
        return {**result, "config": args.config, "retrieval_method": retrieval_method}
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

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

    if args.repeats > 1:
        # Resume by (email_id, repeat_index), not email_id alone.
        resume_path = args.resume_from or (args.outdir / "results.json" if args.resume else None)
        previous_attempts: dict[tuple[str, int], dict[str, Any]] = {}
        if resume_path is not None:
            if not resume_path.is_file():
                parser.error(f"Resume report not found: {resume_path}")
            report = json.loads(resume_path.read_text(encoding="utf-8"))
            if report.get("runner") != "real_process_email" or not isinstance(report.get("cases"), list):
                parser.error("Invalid evaluation report format")
            saved_config = report.get("summary", {}).get("config")
            if saved_config != args.config:
                parser.error(
                    f"Report config={saved_config!r}, requested config={args.config!r}; refusing to mix runs"
                )
            previous_k = report.get("summary", {}).get("k")
            if previous_k != args.repeats:
                parser.error(f"Report repeats={previous_k!r}, requested repeats={args.repeats}; refusing to mix runs")
            selected_ids = {case.email_id for case in cases}
            for old_row in report["cases"]:
                case_id = old_row.get("email_id")
                index = old_row.get("repeat_index")
                if case_id not in selected_ids:
                    continue
                if not isinstance(index, int) or not 1 <= index <= args.repeats:
                    parser.error(f"Invalid repeat_index for {case_id}: {index!r}")
                key = (case_id, index)
                if key in previous_attempts:
                    parser.error(f"Duplicate attempt in saved report: {key}")
                if is_reusable(old_row):
                    previous_attempts[key] = old_row
            print(f"Resuming: {len(previous_attempts)} completed attempts preserved from {resume_path}")

        # Rebuild the report in canonical case/repeat order, retrying only failed
        # or missing attempts. Save a checkpoint after each new attempt.
        rows: list[dict[str, Any]] = []
        for i, case in enumerate(cases, 1):
            for repeat_index in range(1, args.repeats + 1):
                key = (case.email_id, repeat_index)
                if key in previous_attempts:
                    row = previous_attempts[key]
                    print(f"[{i}/{len(cases)}] {case.email_id} attempt {repeat_index}/{args.repeats} REUSED", flush=True)
                else:
                    print(f"[{i}/{len(cases)}] {case.email_id} attempt {repeat_index}/{args.repeats}", flush=True)
                    row = run_one(case, emails, process_email, retrieval_method=retrieval_method)
                    row["repeat_index"] = repeat_index
                rows.append(row)
                save_reports(rows, report_summary(rows), args.outdir)
                if is_execution_error(row):
                    print(f"ERROR in {case.email_id} attempt {repeat_index}: {row.get('error') or row.get('agent_error')}", file=sys.stderr)
                    print(
                        f"Checkpoint saved. Retry later with --config {args.config} "
                        f"--repeats {args.repeats} --outdir {args.outdir} --resume.",
                        file=sys.stderr,
                    )
                    return 2
        summary = report_summary(rows)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nReports: {args.outdir / 'results.json'} and {args.outdir / 'results.csv'}")
        return 0 if summary["cases_completed_all_repeats"] == len(cases) else 2

    resume_path = args.resume_from or (args.outdir / "results.json" if args.resume else None)
    previous = load_previous(resume_path, {c.email_id for c in cases}, args.config) if resume_path else {}
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
            row = run_one(case, emails, process_email, retrieval_method=retrieval_method)
            outcome = "ERROR" if is_execution_error(row) else (
                "PASS" if row["passed_structural_checks"] else "FAIL"
            )
            print(f"  {outcome} ({row.get('latency_seconds', 0):.2f}s)", flush=True)
        rows.append(row)
        save_reports(rows, report_summary(rows), args.outdir)  # checkpoint after every case

        # The orchestrator currently masks HTTP 429 as orchestration_failed.
        # Stop on any orchestration failure to avoid wasting remaining API calls.
        agent_code = (row.get("agent_error") or {}).get("code")
        if is_execution_error(row) and (
            is_limit_error(row) or agent_code == "orchestration_failed"
        ):
            print("STOP: API/orchestration failure. Results saved; resume later.", file=sys.stderr)
            stopped = True
            break

    summary = report_summary(rows)
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nReports: {args.outdir / 'results.json'} and {args.outdir / 'results.csv'}")
    if stopped:
        print("Run again with --resume when the API is available.")
        return 2
    return 1 if summary["errors"] or summary["structural_passes"] != summary["total_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())