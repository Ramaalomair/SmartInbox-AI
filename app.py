
"""SmartInbox AI - Streamlit shared inbox.

Run:
    python -m streamlit run app.py

The application supports the real AI agent and a mock mode for testing.
Drafts are only saved through MCP after explicit human approval.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv

from agent.schemas import (
    AgentResult,
    BudgetUsage,
    EmailInput,
    ToolTrace,
    AgentConfig,
)

from agent.tools import bind_requester, record_evidence
from agent.integrated_tools import IntegratedAgentTools

from mcp_server.approval import (
    compute_action_id,
    issue_approval_token,
)
from mcp_server.client import call_save_draft
from mcp_server.service import record_rejection, storage_dir


load_dotenv()


# Import the real agent.
try:
    from agent.agent import process_email as real_process_email
except ImportError:
    real_process_email = None


ROOT = Path(__file__).resolve().parent

ID_RE = re.compile(
    r"\b(ORD-\d{4}|INV-\d{3}|T-\d{3})\b",
    re.IGNORECASE,
)

SUSPICIOUS_RE = re.compile(
    r"(export|ignore (all |your |previous )?"
    r"(instructions|polic)|password|otp|credentials|"
    r"كلمات? (المرور|السر)|بيانات (الاعتماد|الموظفين))",
    re.IGNORECASE,
)


st.set_page_config(
    page_title="SmartInbox AI",
    page_icon="📥",
    layout="wide",
)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

@st.cache_data
def load_emails() -> list[EmailInput]:
    """Load synthetic emails from the project dataset."""

    path = ROOT / "data" / "emails.csv"

    with open(
        path,
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return [
            EmailInput(
                **{
                    key: row[key]
                    for key in EmailInput.model_fields
                }
            )
            for row in csv.DictReader(file)
        ]


@st.cache_data
def load_labels() -> dict[str, dict]:
    """Load evaluation labels for mock mode only."""

    path = ROOT / "evals" / "eval_cases.json"

    if not path.exists():
        return {}

    cases = json.loads(
        path.read_text(encoding="utf-8")
    )

    return {
        case["email_id"]: case
        for case in cases
    }


# ---------------------------------------------------------------------
# Mock agent
# ---------------------------------------------------------------------

def mock_process_email(email: EmailInput) -> AgentResult:
    """Run the mock agent for UI development and testing."""

    tools = bind_requester(email.sender)

    label = load_labels().get(
        email.email_id,
        {},
    )

    text = f"{email.subject}\n{email.body}"

    trace = []
    evidence = []
    notes = []
    unverified = []

    record_ids = dict.fromkeys(
        match.upper()
        for match in ID_RE.findall(text)
    )

    for record_id in record_ids:

        if record_id.startswith("ORD"):
            tool_name = "lookup_order"
            lookup_fn = tools.lookup_order

        elif record_id.startswith("INV"):
            tool_name = "lookup_invoice"
            lookup_fn = tools.lookup_invoice

        else:
            tool_name = "lookup_ticket"
            lookup_fn = tools.lookup_ticket

        response = lookup_fn(record_id)

        trace.append(
            ToolTrace(
                step=len(trace) + 1,
                tool=tool_name,
                status=response["status"],
            )
        )

        if response["status"] == "success":

            item = record_evidence(
                response["data"]
            )

            evidence.append(item)
            notes.append(item.excerpt)

        else:
            unverified.append(record_id)

    department = label.get(
        "expected_department",
        "customer_support",
    )

    secondary_department = label.get(
        "expected_secondary_department"
    )

    intent = label.get(
        "expected_intent",
        "ambiguous",
    )

    priority = label.get(
        "expected_priority",
        "medium",
    )

    security_flags = (
        ["possible_injection"]
        if SUSPICIOUS_RE.search(text)
        else []
    )

    if department == "security_review":

        status = "manual_review"
        subject = None
        body = None

    else:

        if notes:
            summary = (
                "Here is what we could verify: "
                + "; ".join(notes)
                + "."
            )

        elif (
            intent in ("order_status", "invoice_status")
            and not unverified
        ):
            summary = (
                "Could you please share your order "
                "or invoice number so we can check it?"
            )

        else:
            summary = (
                "Thank you for contacting NovaTech. "
                "Your message was routed to the right team."
            )

        if unverified:
            summary += (
                " For some references, we could not verify "
                "your access. Please authenticate through "
                "an approved channel."
            )

        if email.language == "ar":
            summary = (
                "شكراً لتواصلكم مع نوفاتك.\n\n"
                + summary
            )

        status = (
            "needs_information"
            if "share your order" in summary
            else "draft_ready"
        )

        subject = f"Re: {email.subject}"

        body = (
            "Hello,\n\n"
            f"{summary}\n\n"
            "NovaTech Support"
        )

    action_id = (
        compute_action_id(
            email.email_id,
            department,
            secondary_department,
            subject,
            body,
        )
        if body is not None
        else None
    )

    return AgentResult(
        email_id=email.email_id,
        intent=intent,
        primary_department=department,
        secondary_department=secondary_department,
        priority=priority,
        status=status,
        draft_subject=subject,
        draft_body=body,
        tools_used=sorted(
            {item.tool for item in trace}
        ),
        evidence=evidence,
        security_flags=security_flags,
        requires_approval=(
            body is not None
            and status == "draft_ready"
        ),
        proposed_action_id=action_id,
        trace=trace,
        budget=BudgetUsage(
            steps_used=len(trace),
            input_tokens=None,
            output_tokens=None,
            estimated_cost_usd=None,
        ),
        error=None,
    )


# ---------------------------------------------------------------------
# Real agent integration
# ---------------------------------------------------------------------

def run_agent(
    email: EmailInput,
) -> tuple[AgentResult, str]:
    """Run the real agent with authorized backend tools."""

    use_mock = st.session_state.get(
        "force_mock",
        False,
    )

    if real_process_email is not None and not use_mock:

        # The synthetic dataset uses fixture sender identities.
        # Production must bind an authenticated requester identity.
        bound_tools = bind_requester(
            email.sender
        )

        # Combine hybrid RAG with authorized record lookups.
        tools = IntegratedAgentTools(
            bound_tools=bound_tools
        )

        config = AgentConfig(
            max_steps=6,
            retrieval_top_k=3,
            max_input_tokens=12000,
            max_output_tokens=2000,
            max_estimated_cost_usd=0.10,
        )

        # Run the asynchronous orchestrator from Streamlit.
        result = asyncio.run(
            real_process_email(
                email=email,
                tools=tools,
                config=config,
            )
        )

        return result, "real"

    return mock_process_email(email), "mock"


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

st.session_state.setdefault(
    "results",
    {},
)

st.session_state.setdefault(
    "decisions",
    {},
)

emails = load_emails()


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:

    st.title("📥 SmartInbox AI")

    st.caption(
        "NovaTech (fictional) · synthetic data only"
    )

    choice = st.selectbox(
        "Email",
        [
            f"{email.email_id} · {email.subject}"
            for email in emails
        ],
    )

    email = next(
        item
        for item in emails
        if item.email_id == choice.split(" · ")[0]
    )

    if real_process_email is None:

        st.warning(
            "MOCK agent: the real agent is unavailable."
        )

    else:

        st.checkbox(
            "Use mock agent",
            key="force_mock",
        )

    if not os.environ.get("APPROVAL_SECRET"):

        st.error(
            "APPROVAL_SECRET is not set: approvals are disabled. "
            "Add it to .env."
        )

    if st.button("Reset session"):

        st.session_state.results.clear()
        st.session_state.decisions.clear()

        st.rerun()


# ---------------------------------------------------------------------
# Shared inbox
# ---------------------------------------------------------------------

with st.expander(
    "Shared inbox",
    expanded=False,
):

    rows = []

    for item in emails:

        decision = st.session_state.decisions.get(
            item.email_id,
            {},
        ).get("decision")

        state = decision or (
            "analyzed"
            if item.email_id in st.session_state.results
            else "new"
        )

        rows.append({
            "id": item.email_id,
            "from": item.sender,
            "subject": item.subject,
            "lang": item.language,
            "state": state,
        })

    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------
# Selected email
# ---------------------------------------------------------------------

left, right = st.columns([2, 3])

with left:

    st.subheader(email.subject)

    st.caption(
        f"{email.email_id} · from {email.sender} "
        f"· {email.language}"
    )

    st.text(email.body)

    if st.button(
        "Run agent",
        type="primary",
        key="run",
    ):

        with st.spinner("Analyzing..."):

            st.session_state.results[email.email_id] = (
                run_agent(email)
            )

        # A new analysis invalidates previous UI decisions.
        st.session_state.decisions.pop(
            email.email_id,
            None,
        )


# ---------------------------------------------------------------------
# Agent results
# ---------------------------------------------------------------------

entry = st.session_state.results.get(
    email.email_id
)

with right:

    if entry is None:

        st.info(
            "Select an email and press **Run agent**."
        )

        st.stop()

    result, mode = entry

    if mode == "mock":

        st.warning(
            "MOCK agent: routing labels come from "
            "the evaluation file; lookups are real."
        )

    def pretty(value: str | None) -> str:
        """Format labels for display."""

        return (
            value.replace("_", " ")
            if value
            else "—"
        )

    row1 = st.columns(2)
    row2 = st.columns(2)

    values = [
        ("Intent", result.intent),
        ("Priority", result.priority),
        ("Department", result.primary_department),
        (
            "Secondary department",
            result.secondary_department,
        ),
    ]

    for column, (label, value) in zip(
        row1 + row2,
        values,
    ):

        column.caption(label)

        column.markdown(
            f"**{pretty(value)}**"
        )

    st.caption(
        f"Status: **{result.status}** "
        f"· tools: {', '.join(result.tools_used) or 'none'}"
    )

    if result.security_flags:

        st.error(
            "Security flags: "
            + ", ".join(result.security_flags)
        )

    if result.error:

        st.error(
            f"{result.error.code}: "
            f"{result.error.message}"
        )

    # -----------------------------------------------------------------
    # Draft reply
    # -----------------------------------------------------------------

    st.markdown("#### Draft reply")

    has_draft = (
        result.status == "draft_ready"
        and result.draft_body is not None
        and result.draft_subject is not None
    )

    subject = st.text_input(
        "Subject",
        value=result.draft_subject or "",
        key=f"subj_{email.email_id}",
        disabled=not has_draft,
    )

    body = st.text_area(
        "Body",
        value=result.draft_body or "",
        height=220,
        key=f"body_{email.email_id}",
        disabled=not has_draft,
    )

    if not has_draft:

        st.info(
            "No approved-ready draft was produced. "
            "This email may need manual review "
            "or additional information."
        )

    # -----------------------------------------------------------------
    # Evidence
    # -----------------------------------------------------------------

    with st.expander(
        f"Evidence ({len(result.evidence)})",
        expanded=True,
    ):

        for item in result.evidence:

            source_label = (
                f"**{item.source_id}**"
            )

            if item.score is not None:
                source_label += (
                    f" · score {item.score:.2f}"
                )

            st.markdown(source_label)
            st.caption(item.excerpt)

        if not result.evidence:

            st.caption(
                "No evidence items."
            )

    # -----------------------------------------------------------------
    # Trace and budget
    # -----------------------------------------------------------------

    with st.expander("Trace and budget"):

        st.dataframe(
            [
                item.model_dump()
                for item in result.trace
            ],
            hide_index=True,
        )

        st.json(
            result.budget.model_dump()
        )

    # -----------------------------------------------------------------
    # Human approval
    # -----------------------------------------------------------------

    st.markdown("#### Human approval")

    decision = st.session_state.decisions.get(
        email.email_id
    )

    if decision:

        outcome = (
            decision.get("result")
            or {}
        )

        if (
            decision["decision"] == "approved"
            and outcome.get("status") == "saved"
        ):

            st.success(
                "Approved and saved: "
                f"{outcome['record_id']}"
            )

        elif decision["decision"] == "approved":

            error_code = outcome.get(
                "error",
                {},
            ).get(
                "code",
                "unknown_error",
            )

            st.error(
                f"Write blocked: {error_code}"
            )

        else:

            st.warning(
                "Rejected: nothing was saved."
            )

    else:

        # Bind the action ID to the current draft content.
        action_id = compute_action_id(
            email.email_id,
            result.primary_department,
            result.secondary_department,
            subject,
            body,
        )

        st.caption(
            f"Action ID `{action_id[:12]}` "
            "is bound to exactly this text and routing."
        )

        can_approve = (
            has_draft
            and result.requires_approval
            and bool(
                os.environ.get(
                    "APPROVAL_SECRET"
                )
            )
        )

        approve_column, reject_column = st.columns(2)

        if approve_column.button(
            "✅ Approve and save",
            disabled=not can_approve,
            key="approve",
        ):

            payload = {
                "action_id": action_id,
                "email_id": email.email_id,
                "primary_department": (
                    result.primary_department
                ),
                "secondary_department": (
                    result.secondary_department
                ),
                "draft_subject": subject,
                "draft_body": body,
                "approval_token": (
                    issue_approval_token(
                        action_id
                    )
                ),
            }

            with st.spinner(
                "Saving through the MCP server..."
            ):

                outcome = call_save_draft(
                    payload
                )

            st.session_state.decisions[email.email_id] = {
                "decision": "approved",
                "result": outcome,
            }

            st.rerun()

        if reject_column.button(
            "❌ Reject",
            key="reject",
        ):

            record_rejection(
                action_id,
                email.email_id,
            )

            st.session_state.decisions[email.email_id] = {
                "decision": "rejected",
                "result": None,
            }

            st.rerun()

    # -----------------------------------------------------------------
    # Audit log
    # -----------------------------------------------------------------

    with st.expander(
        "Audit log (latest 8)"
    ):

        path = (
            storage_dir()
            / "audit_log.jsonl"
        )

        lines = (
            path.read_text(
                encoding="utf-8"
            ).splitlines()[-8:]
            if path.exists()
            else []
        )

        st.code(
            "\n".join(lines) or "empty",
            language="json",
        )
