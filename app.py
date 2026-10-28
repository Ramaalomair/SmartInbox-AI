"""SmartInbox AI - Streamlit shared inbox (Developer B, task B3).

Run:  uv run streamlit run app.py

Uses the real agent (agent/agent.py -> process_email) when it exists, otherwise a clearly
labelled MOCK agent so the UI and approval flow can be built and tested in parallel.
Approval is enforced server-side (mcp_server): this UI only asks for it.
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent.schemas import AgentResult, BudgetUsage, EmailInput, ToolTrace
from agent.tools import bind_requester, record_evidence
from mcp_server.approval import compute_action_id, issue_approval_token
from mcp_server.client import call_save_draft
from mcp_server.service import record_rejection, storage_dir

try:  # Developer A's agent: process_email(email, tools) -> AgentResult
    from agent.agent import process_email as real_process_email
except ImportError:
    real_process_email = None

ROOT = Path(__file__).resolve().parent
ID_RE = re.compile(r"\b(ORD-\d{4}|INV-\d{3}|T-\d{3})\b", re.I)
SUSPICIOUS_RE = re.compile(
    r"(export|ignore (all |your |previous )?(instructions|polic)|password|otp|credentials|"
    r"كلمات? (المرور|السر)|بيانات (الاعتماد|الموظفين))", re.I)

st.set_page_config(page_title="SmartInbox AI", page_icon="📥", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_data
def load_emails() -> list[EmailInput]:
    with open(ROOT / "data" / "emails.csv", newline="", encoding="utf-8-sig") as f:
        return [EmailInput(**{k: row[k] for k in EmailInput.model_fields}) for row in csv.DictReader(f)]


@st.cache_data
def load_labels() -> dict[str, dict]:
    path = ROOT / "evals" / "eval_cases.json"
    return {c["email_id"]: c for c in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}


# ----------------------------------------------------------------------------- mock agent
def mock_process_email(email: EmailInput) -> AgentResult:
    """UI-development stand-in. Real record lookups, but labels come from the eval file."""
    tools = bind_requester(email.sender)  # identity is bound by the app, not by the email text
    label = load_labels().get(email.email_id, {})
    text = f"{email.subject}\n{email.body}"
    trace, evidence, notes, unverified = [], [], [], []

    for rid in dict.fromkeys(m.upper() for m in ID_RE.findall(text)):
        fn = (tools.lookup_order if rid.startswith("ORD") else
              tools.lookup_invoice if rid.startswith("INV") else tools.lookup_ticket)
        res = fn(rid)
        trace.append(ToolTrace(step=len(trace) + 1, tool=fn.__name__, status=res["status"]))
        if res["status"] == "success":
            item = record_evidence(res["data"])
            evidence.append(item)
            notes.append(item.excerpt)
        else:
            unverified.append(rid)

    department = label.get("expected_department", "customer_support")
    flags = ["possible_injection"] if SUSPICIOUS_RE.search(text) else []
    intent = label.get("expected_intent", "ambiguous")

    if department == "security_review":
        status, subject, body = "manual_review", None, None
    else:
        if notes:
            summary = "Here is what we could verify: " + "; ".join(notes) + "."
        elif intent in ("order_status", "invoice_status") and not unverified:
            summary = "Could you please share your order or invoice number so we can check it?"
        else:
            summary = "Thank you for contacting NovaTech. Your message was routed to the right team."
        if unverified:
            summary += " For some references we could not verify your access; please authenticate through an approved channel."
        if email.language == "ar":
            summary = "شكراً لتواصلكم مع نوفاتك.\n\n" + summary
        status = "needs_information" if "share your order" in summary else "draft_ready"
        subject = f"Re: {email.subject}"
        body = f"Hello,\n\n{summary}\n\nNovaTech Support"

    secondary = label.get("expected_secondary_department")
    action_id = (compute_action_id(email.email_id, department, secondary, subject, body)
                 if body is not None else None)
    return AgentResult(
        email_id=email.email_id, intent=intent, primary_department=department,
        secondary_department=secondary, priority=label.get("expected_priority", "medium"),
        status=status, draft_subject=subject, draft_body=body,
        tools_used=sorted({t.tool for t in trace}), evidence=evidence, security_flags=flags,
        requires_approval=True, proposed_action_id=action_id, trace=trace,
        budget=BudgetUsage(steps_used=len(trace), input_tokens=None, output_tokens=None,
                           estimated_cost_usd=None),
        error=None)


def run_agent(email: EmailInput) -> tuple[AgentResult, str]:
    if real_process_email is not None and not st.session_state.get("force_mock", False):
        return real_process_email(email, bind_requester(email.sender)), "real"
    return mock_process_email(email), "mock"


# ----------------------------------------------------------------------------- state
st.session_state.setdefault("results", {})    # email_id -> (AgentResult, "real" | "mock")
st.session_state.setdefault("decisions", {})  # email_id -> {"decision": ..., "result": ...}
emails = load_emails()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📥 SmartInbox AI")
    st.caption("NovaTech (fictional) · synthetic data only")
    choice = st.selectbox("Email", [f"{e.email_id} · {e.subject}" for e in emails])
    email = next(e for e in emails if e.email_id == choice.split(" · ")[0])
    if real_process_email is None:
        st.warning("MOCK agent: agent/agent.py was not found.")
    else:
        st.checkbox("Use mock agent", key="force_mock")
    if not os.environ.get("APPROVAL_SECRET"):
        st.error("APPROVAL_SECRET is not set: approvals are disabled. Add it to .env.")
    if st.button("Reset session"):
        st.session_state.results.clear()
        st.session_state.decisions.clear()
        st.rerun()

# ----------------------------------------------------------------------------- inbox
with st.expander("Shared inbox", expanded=False):
    rows = []
    for e in emails:
        decision = st.session_state.decisions.get(e.email_id, {}).get("decision")
        state = decision or ("analyzed" if e.email_id in st.session_state.results else "new")
        rows.append({"id": e.email_id, "from": e.sender, "subject": e.subject,
                     "lang": e.language, "state": state})
    st.dataframe(rows, width="stretch", hide_index=True)

# ----------------------------------------------------------------------------- email
left, right = st.columns([2, 3])
with left:
    st.subheader(email.subject)
    st.caption(f"{email.email_id} · from {email.sender} · {email.language}")
    st.text(email.body)
    if st.button("Run agent", type="primary", key="run"):
        with st.spinner("Analyzing..."):
            st.session_state.results[email.email_id] = run_agent(email)
        st.session_state.decisions.pop(email.email_id, None)

# ----------------------------------------------------------------------------- analysis
entry = st.session_state.results.get(email.email_id)
with right:
    if entry is None:
        st.info("Select an email and press **Run agent**.")
        st.stop()

    result, mode = entry
    if mode == "mock":
        st.warning("MOCK agent: routing labels come from the eval file; lookups are real.")

    def pretty(value: str | None) -> str:
        return value.replace("_", " ") if value else "—"

    row1, row2 = st.columns(2), st.columns(2)
    for col, (label, value) in zip(row1 + row2, [
            ("Intent", result.intent), ("Priority", result.priority),
            ("Department", result.primary_department),
            ("Secondary department", result.secondary_department)]):
        col.caption(label)
        col.markdown(f"**{pretty(value)}**")  # wraps instead of being cut off like st.metric
    st.caption(f"Status: **{result.status}** · tools: {', '.join(result.tools_used) or 'none'}")
    if result.security_flags:
        st.error("Security flags: " + ", ".join(result.security_flags))
    if result.error:
        st.error(f"{result.error.code}: {result.error.message}")

    st.markdown("#### Draft reply")
    has_draft = result.draft_body is not None
    subject = st.text_input("Subject", value=result.draft_subject or "",
                            key=f"subj_{email.email_id}", disabled=not has_draft)
    body = st.text_area("Body", value=result.draft_body or "", height=220,
                        key=f"body_{email.email_id}", disabled=not has_draft)
    if not has_draft:
        st.info("No draft was produced: this email needs manual review.")

    with st.expander(f"Evidence ({len(result.evidence)})", expanded=True):
        for item in result.evidence:
            st.markdown(f"**{item.source_id}**" + (f" · score {item.score:.2f}" if item.score is not None else ""))
            st.caption(item.excerpt)
        if not result.evidence:
            st.caption("No evidence items.")
    with st.expander("Trace and budget"):
        st.dataframe([t.model_dump() for t in result.trace], hide_index=True)
        st.json(result.budget.model_dump())

    # ------------------------------------------------------------------------- approval
    st.markdown("#### Human approval")
    decision = st.session_state.decisions.get(email.email_id)
    if decision:
        out = decision.get("result") or {}
        if decision["decision"] == "approved" and out.get("status") == "saved":
            st.success(f"Approved and saved: {out['record_id']}")
        elif decision["decision"] == "approved":
            st.error(f"Write blocked: {out['error']['code']}")
        else:
            st.warning("Rejected: nothing was saved.")
    else:
        action_id = compute_action_id(email.email_id, result.primary_department,
                                      result.secondary_department, subject, body)
        st.caption(f"Action ID `{action_id[:12]}` is bound to exactly this text and routing.")
        can_approve = has_draft and result.requires_approval and bool(os.environ.get("APPROVAL_SECRET"))
        a, b = st.columns(2)
        if a.button("✅ Approve and save", disabled=not can_approve, key="approve"):
            payload = {"action_id": action_id, "email_id": email.email_id,
                       "primary_department": result.primary_department,
                       "secondary_department": result.secondary_department,
                       "draft_subject": subject, "draft_body": body,
                       "approval_token": issue_approval_token(action_id)}
            with st.spinner("Saving through the MCP server..."):
                out = call_save_draft(payload)
            st.session_state.decisions[email.email_id] = {"decision": "approved", "result": out}
            st.rerun()
        if b.button("❌ Reject", key="reject"):
            record_rejection(action_id, email.email_id)
            st.session_state.decisions[email.email_id] = {"decision": "rejected", "result": None}
            st.rerun()

    with st.expander("Audit log (latest 8)"):
        path = storage_dir() / "audit_log.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()[-8:] if path.exists() else []
        st.code("\n".join(lines) or "empty", language="json")