"""SmartInbox AI - Streamlit shared inbox.

Run:
    python -m streamlit run app.py --server.fileWatcherType none

The application supports the real AI agent and a mock mode for testing.
Drafts are only saved through MCP after explicit human approval.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import html
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
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Data loading
# ============================================================

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


# ============================================================
# Mock agent
# ============================================================

def mock_process_email(email: EmailInput) -> AgentResult:
    """Run the mock agent for UI development and testing."""

    tools = bind_requester(email.sender)

    label_data = load_labels().get(
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
            item = record_evidence(response["data"])
            evidence.append(item)
            notes.append(item.excerpt)
        else:
            unverified.append(record_id)

    department = label_data.get(
        "expected_department",
        "customer_support",
    )

    secondary_department = label_data.get(
        "expected_secondary_department"
    )

    intent = label_data.get(
        "expected_intent",
        "ambiguous",
    )

    priority = label_data.get(
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


# ============================================================
# Real agent integration
# ============================================================

def run_agent(
    email: EmailInput,
) -> tuple[AgentResult, str]:
    """Run the real agent with authorized backend tools."""

    use_mock = st.session_state.get(
        "force_mock",
        False,
    )

    if real_process_email is not None and not use_mock:

        # Synthetic sender identities are used for the demo only.
        # Production must use authenticated requester identities.
        bound_tools = bind_requester(email.sender)

        # Combine hybrid RAG with authorized backend lookups.
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

        result = asyncio.run(
            real_process_email(
                email=email,
                tools=tools,
                config=config,
            )
        )

        return result, "real"

    return mock_process_email(email), "mock"


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --si-accent: #4f7cff;
        --si-accent-soft: rgba(79, 124, 255, 0.12);
        --si-border: rgba(140, 150, 170, 0.22);
        --si-border-strong: rgba(140, 150, 170, 0.34);
        --si-surface: rgba(140, 150, 170, 0.06);
        --si-surface-2: rgba(140, 150, 170, 0.10);
        --si-success: #22a06b;
        --si-warning: #c88a12;
        --si-danger: #d9534f;
        --si-neutral: #7a8699;
        --si-radius: 10px;
    }

    html, body, [class*="css"], .stApp,
    .stMarkdown, .stTextInput, .stTextArea, .stButton,
    [data-testid="stMetric"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont,
            'Segoe UI', Roboto, 'Noto Sans Arabic', sans-serif;
    }

    code, pre, .si-mono {
        font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular,
            Menlo, monospace !important;
    }

    #MainMenu, footer, [data-testid="stDecoration"] {
        visibility: hidden;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }

    h1, h2, h3, h4 {
        letter-spacing: -0.015em;
        font-weight: 600;
    }

    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--si-border);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .si-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.25rem;
    }

    .si-brand-mark {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        background: var(--si-accent);
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
        flex-shrink: 0;
    }

    .si-brand-name {
        font-weight: 650;
        font-size: 1.05rem;
        line-height: 1.2;
    }

    .si-brand-sub {
        font-size: 0.75rem;
        opacity: 0.6;
    }

    .si-section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.55;
        margin: 0.4rem 0 0.5rem;
    }

    .si-status-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.82rem;
        padding: 0.55rem 0.7rem;
        border: 1px solid var(--si-border);
        border-radius: 8px;
        background: var(--si-surface);
        margin-top: 0.5rem;
    }

    .si-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .si-footnote {
        font-size: 0.74rem;
        opacity: 0.55;
        line-height: 1.5;
    }

    /* ---------- Page header ---------- */

    .si-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 1rem;
        flex-wrap: wrap;
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--si-border);
    }

    .si-breadcrumb {
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        opacity: 0.55;
        margin-bottom: 0.35rem;
    }

    .si-title {
        font-weight: 650;
        font-size: 1.75rem;
        line-height: 1.2;
        letter-spacing: -0.02em;
        margin: 0;
    }

    .si-subtitle {
        font-size: 0.9rem;
        opacity: 0.65;
        margin-top: 0.35rem;
    }

    .si-env {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 0.35rem 0.7rem;
        border: 1px solid var(--si-border);
        border-radius: 6px;
        background: var(--si-surface);
        white-space: nowrap;
    }

    /* ---------- Section headings ---------- */

    .si-section {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 0.25rem 0 0.75rem;
    }

    .si-section-title {
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    .si-section-hint {
        font-size: 0.78rem;
        opacity: 0.55;
    }

    /* ---------- Cards ---------- */

    .si-card {
        background: var(--si-surface);
        border: 1px solid var(--si-border);
        border-radius: var(--si-radius);
        padding: 1.25rem 1.35rem;
        margin-bottom: 1rem;
    }

    .si-email-subject {
        font-size: 1.15rem;
        font-weight: 600;
        line-height: 1.4;
        margin-bottom: 0.9rem;
        overflow-wrap: anywhere;
    }

    .si-meta-grid {
        display: grid;
        grid-template-columns: 88px 1fr;
        row-gap: 0.35rem;
        font-size: 0.83rem;
    }

    .si-meta-key {
        opacity: 0.55;
    }

    .si-meta-val {
        overflow-wrap: anywhere;
        font-weight: 500;
    }

    .si-message {
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        unicode-bidi: plaintext;
        text-align: start;
        line-height: 1.7;
        font-size: 0.93rem;
    }

    .si-divider {
        border-top: 1px solid var(--si-border);
        margin: 1rem 0;
    }

    /* ---------- Analysis summary ---------- */

    .si-summary {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: center;
        padding: 0.85rem 1rem;
        border: 1px solid var(--si-border);
        border-radius: var(--si-radius);
        background: var(--si-surface);
        margin-bottom: 1rem;
    }

    .si-summary-title {
        font-weight: 600;
        font-size: 0.92rem;
    }

    .si-note {
        font-size: 0.78rem;
        opacity: 0.6;
        margin-top: 0.15rem;
    }

    .si-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border-radius: 6px;
        padding: 0.28rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        white-space: nowrap;
        border: 1px solid transparent;
    }

    .si-badge::before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: currentColor;
    }

    .si-tone-success {
        color: var(--si-success);
        background: rgba(34, 160, 107, 0.12);
        border-color: rgba(34, 160, 107, 0.28);
    }

    .si-tone-warning {
        color: var(--si-warning);
        background: rgba(200, 138, 18, 0.12);
        border-color: rgba(200, 138, 18, 0.28);
    }

    .si-tone-danger {
        color: var(--si-danger);
        background: rgba(217, 83, 79, 0.12);
        border-color: rgba(217, 83, 79, 0.28);
    }

    .si-tone-neutral {
        color: var(--si-neutral);
        background: rgba(122, 134, 153, 0.12);
        border-color: rgba(122, 134, 153, 0.28);
    }

    .si-tone-accent {
        color: var(--si-accent);
        background: var(--si-accent-soft);
        border-color: rgba(79, 124, 255, 0.28);
    }

    /* ---------- Classification grid ---------- */

    .si-kv-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        border: 1px solid var(--si-border);
        border-radius: var(--si-radius);
        overflow: hidden;
        margin-bottom: 1rem;
    }

    .si-kv {
        padding: 0.8rem 1rem;
        border-bottom: 1px solid var(--si-border);
    }

    .si-kv:nth-child(odd) {
        border-right: 1px solid var(--si-border);
    }

    .si-kv:nth-last-child(-n+2) {
        border-bottom: none;
    }

    .si-kv-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        opacity: 0.55;
        margin-bottom: 0.25rem;
    }

    .si-kv-value {
        font-size: 0.95rem;
        font-weight: 600;
        overflow-wrap: anywhere;
    }

    /* ---------- Evidence ---------- */

    .si-evidence {
        border: 1px solid var(--si-border);
        border-left: 3px solid var(--si-accent);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.65rem;
        background: var(--si-surface);
    }

    .si-evidence-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.4rem;
    }

    .si-evidence-source {
        font-size: 0.8rem;
        font-weight: 600;
        overflow-wrap: anywhere;
    }

    .si-evidence-text {
        font-size: 0.86rem;
        line-height: 1.6;
        opacity: 0.8;
        overflow-wrap: anywhere;
        unicode-bidi: plaintext;
    }

    .si-subhead {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        opacity: 0.55;
        margin: 1.1rem 0 0.5rem;
    }

    /* ---------- Streamlit widgets ---------- */

    .stButton > button {
        border-radius: 8px;
        min-height: 2.6rem;
        font-weight: 600;
        font-size: 0.88rem;
        letter-spacing: 0.005em;
        transition: border-color 0.15s ease, background 0.15s ease;
    }

    [data-testid="stMetric"] {
        border: 1px solid var(--si-border);
        border-radius: var(--si-radius);
        padding: 0.9rem 1.1rem;
        background: var(--si-surface);
    }

    [data-testid="stMetricLabel"] p {
        font-size: 0.72rem !important;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        opacity: 0.6;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 650;
    }

    [data-testid="stExpander"] {
        border-radius: var(--si-radius);
        overflow: hidden;
    }

    [data-testid="stExpander"] summary p {
        font-weight: 600;
        font-size: 0.88rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1.25rem;
        border-bottom: 1px solid var(--si-border);
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 0.1rem;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab"] p {
        font-size: 0.88rem;
    }

    .stTextInput input, .stTextArea textarea {
        border-radius: 8px;
        font-size: 0.92rem;
        unicode-bidi: plaintext;
    }

    [data-testid="stAlert"] {
        border-radius: var(--si-radius);
    }

    /* ---------- Shared inbox ---------- */

    .st-key-si_inbox_list {
        border: 1px solid var(--si-border);
        border-radius: var(--si-radius);
        padding: 0.35rem !important;
        gap: 0.25rem !important;
    }

    [class*="st-key-si_row_"] {
        position: relative;
        gap: 0 !important;
    }

    .si-row {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.65rem 0.8rem;
        border: 1px solid transparent;
        border-radius: 8px;
        transition: background 0.12s ease, border-color 0.12s ease;
    }

    [class*="st-key-si_row_"]:hover .si-row {
        background: var(--si-surface-2);
        border-color: var(--si-border);
    }

    .si-row.is-selected {
        background: var(--si-accent-soft);
        border-color: rgba(79, 124, 255, 0.35);
        box-shadow: inset 3px 0 0 var(--si-accent);
    }

    .si-row-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        margin-top: 0.42rem;
        flex-shrink: 0;
    }

    .si-row-main {
        flex: 1;
        min-width: 0;
    }

    .si-row-top {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        font-size: 0.78rem;
    }

    .si-row-sender {
        opacity: 0.7;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        min-width: 0;
    }

    .si-row-id {
        margin-left: auto;
        font-size: 0.72rem;
        opacity: 0.45;
        flex-shrink: 0;
    }

    .si-row-subject {
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.12rem 0 0.1rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .si-row.is-new .si-row-subject {
        font-weight: 650;
    }

    .si-row-preview {
        font-size: 0.8rem;
        opacity: 0.55;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .si-row-side {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0.3rem;
        flex-shrink: 0;
        min-width: 78px;
    }

    .si-row-status {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.72rem;
        font-weight: 600;
        opacity: 0.8;
    }

    .si-row-pri {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        opacity: 0.55;
    }

    .si-row-pri.is-high {
        color: var(--si-danger);
        opacity: 0.9;
    }

    /* Invisible full-row button that makes each item clickable. */
    [class*="st-key-si_open_"] {
        position: absolute !important;
        inset: 0;
        z-index: 2;
        margin: 0 !important;
    }

    [class*="st-key-si_open_"] > div,
    [class*="st-key-si_open_"] .stButton {
        width: 100%;
        height: 100%;
    }

    [class*="st-key-si_open_"] button {
        width: 100%;
        height: 100%;
        min-height: 0;
        opacity: 0;
        cursor: pointer;
    }

    [class*="st-key-si_open_"] button:focus-visible {
        opacity: 1;
        color: transparent;
        background: transparent;
        outline: 2px solid var(--si-accent);
        outline-offset: -2px;
    }

    .si-empty {
        font-size: 0.85rem;
        opacity: 0.55;
        text-align: center;
        padding: 2rem 0;
    }

    @media (max-width: 900px) {
        .si-row-side {
            min-width: 0;
        }

        .si-row-preview {
            display: none;
        }
    }

    @media (max-width: 900px) {
        .si-title {
            font-size: 1.4rem;
        }

        .si-kv-grid {
            grid-template-columns: 1fr;
        }

        .si-kv:nth-child(odd) {
            border-right: none;
        }

        .si-kv:nth-last-child(2) {
            border-bottom: 1px solid var(--si-border);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helpers
# ============================================================

def safe(value: object) -> str:
    """Escape untrusted email content before rendering HTML."""
    return html.escape(
        str(value if value is not None else ""),
        quote=True,
    )


def label(value: str | None) -> str:
    """Humanize internal enum values."""
    return (
        value.replace("_", " ").title()
        if value
        else "Not assigned"
    )


def status_tone(value: str | None) -> str:
    """Map a status value to a badge color tone (display only)."""

    key = (value or "").lower().replace(" ", "_")

    return {
        "draft_ready": "success",
        "approved": "success",
        "saved": "success",
        "needs_information": "warning",
        "rejected": "warning",
        "manual_review": "danger",
        "error": "danger",
        "failed": "danger",
    }.get(key, "neutral")


def badge(text: str, tone: str = "neutral") -> str:
    """Render a small status badge."""
    return (
        f'<span class="si-badge si-tone-{tone}">'
        f"{safe(text)}</span>"
    )


def section(title: str, hint: str = "") -> None:
    """Render a section heading with an optional hint."""

    hint_html = (
        f'<span class="si-section-hint">{safe(hint)}</span>'
        if hint
        else ""
    )

    st.markdown(
        f'<div class="si-section">'
        f'<span class="si-section-title">{safe(title)}</span>'
        f"{hint_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


INBOX_STATUSES = ["All", "New", "Analyzed", "Approved", "Rejected"]

INBOX_STATUS_COLOR = {
    "New": "var(--si-accent)",
    "Analyzed": "var(--si-neutral)",
    "Approved": "var(--si-success)",
    "Rejected": "var(--si-danger)",
}


def inbox_status(item: EmailInput) -> str:
    """Return the inbox processing state from existing session data."""

    decision = st.session_state.decisions.get(
        item.email_id,
        {},
    ).get("decision")

    if decision in ("approved", "rejected"):
        return decision.title()

    if item.email_id in st.session_state.results:
        return "Analyzed"

    return "New"


def inbox_priority(item: EmailInput) -> str | None:
    """Return the agent-assigned priority once a message is analyzed."""

    entry = st.session_state.results.get(item.email_id)

    return entry[0].priority if entry else None


def preview_text(body: str, limit: int = 120) -> str:
    """Collapse whitespace and shorten the body for list previews."""

    text = " ".join((body or "").split())

    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def select_email(email_id: str) -> None:
    """Make an inbox item the active message."""
    st.session_state.selected_email_id = email_id


# ============================================================
# Session state
# ============================================================

st.session_state.setdefault("results", {})
st.session_state.setdefault("decisions", {})
st.session_state.setdefault("force_mock", False)

emails = load_emails()

# The inbox list is the primary selector; default to the first message.
if st.session_state.get("selected_email_id") not in {
    item.email_id for item in emails
}:
    st.session_state.selected_email_id = emails[0].email_id

email = next(
    item
    for item in emails
    if item.email_id == st.session_state.selected_email_id
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="si-brand">'
        '<div class="si-brand-mark">SI</div>'
        '<div>'
        '<div class="si-brand-name">SmartInbox AI</div>'
        '<div class="si-brand-sub">NovaTech Workspace</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="si-section-label">Configuration</div>',
        unsafe_allow_html=True,
    )

    if real_process_email is None:
        st.warning(
            "Real agent unavailable. Mock mode is active."
        )
    else:
        st.toggle(
            "Use mock agent",
            key="force_mock",
            help=(
                "Mock mode uses evaluation labels. "
                "Do not use its predictions for model evaluation."
            ),
        )

    if not os.environ.get("APPROVAL_SECRET"):
        st.error(
            "APPROVAL_SECRET is missing. "
            "Draft approvals are disabled."
        )
    else:
        st.markdown(
            '<div class="si-status-row">'
            '<span class="si-dot" '
            'style="background:var(--si-success)"></span>'
            'Human approval enabled'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button(
        "Reset session",
        use_container_width=True,
        help=(
            "Clear UI session state; saved drafts "
            "and audit logs are not deleted."
        ),
    ):
        st.session_state.results.clear()
        st.session_state.decisions.clear()

        for key in list(st.session_state.keys()):
            if key.startswith(("subj_", "body_")):
                del st.session_state[key]

        st.rerun()

    st.markdown(
        '<div class="si-footnote" style="margin-top:0.75rem">'
        'Prototype environment using synthetic data. '
        'Saving a draft never sends an email.'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Dashboard header
# ============================================================

mode_label = (
    "Mock agent"
    if real_process_email is None or st.session_state.force_mock
    else "Live agent"
)

mode_color = (
    "var(--si-warning)"
    if mode_label == "Mock agent"
    else "var(--si-success)"
)

st.markdown(
    f'<div class="si-header">'
    f'<div>'
    f'<div class="si-breadcrumb">Customer Operations / Inbox</div>'
    f'<div class="si-title">Inbox Command Center</div>'
    f'<div class="si-subtitle">'
    f'Analyze messages, verify evidence, and approve '
    f'AI-generated drafts before they are saved.'
    f'</div>'
    f'</div>'
    f'<span class="si-env">'
    f'<span class="si-dot" style="background:{mode_color}"></span>'
    f'{safe(mode_label)}'
    f'</span>'
    f'</div>',
    unsafe_allow_html=True,
)


# ============================================================
# Dashboard metrics
# ============================================================

analyzed_count = len(st.session_state.results)

approved_count = sum(
    1
    for decision in st.session_state.decisions.values()
    if (
        decision.get("decision") == "approved"
        and (decision.get("result") or {}).get("status") == "saved"
    )
)

rejected_count = sum(
    1
    for decision in st.session_state.decisions.values()
    if decision.get("decision") == "rejected"
)

metric_columns = st.columns(4)

titles = [
    "Inbox messages",
    "Analyzed",
    "Saved drafts",
    "Rejected",
]

numbers = [
    len(emails),
    analyzed_count,
    approved_count,
    rejected_count,
]

for column, title, number in zip(
    metric_columns,
    titles,
    numbers,
):
    column.metric(title, number)


st.markdown(
    "<div style='height:0.75rem'></div>",
    unsafe_allow_html=True,
)


# ============================================================
# Shared inbox
# ============================================================

statuses = {item.email_id: inbox_status(item) for item in emails}

status_counts = {
    name: (
        len(emails)
        if name == "All"
        else sum(1 for value in statuses.values() if value == name)
    )
    for name in INBOX_STATUSES
}

section("Shared inbox", f"{len(emails)} messages")

search_column, filter_column = st.columns([1.4, 1], gap="medium")

query = search_column.text_input(
    "Search inbox",
    placeholder="Search by subject, sender, or email ID",
    key="inbox_query",
    label_visibility="collapsed",
)

if hasattr(st, "segmented_control"):
    status_filter = filter_column.segmented_control(
        "Status filter",
        INBOX_STATUSES,
        default="All",
        format_func=lambda name: f"{name} {status_counts[name]}",
        key="inbox_status_filter",
        label_visibility="collapsed",
    ) or "All"
else:
    status_filter = filter_column.radio(
        "Status filter",
        INBOX_STATUSES,
        format_func=lambda name: f"{name} {status_counts[name]}",
        key="inbox_status_filter",
        horizontal=True,
        label_visibility="collapsed",
    )

needle = query.strip().lower()

visible = [
    item
    for item in emails
    if (status_filter == "All" or statuses[item.email_id] == status_filter)
    and (
        not needle
        or needle in item.subject.lower()
        or needle in item.sender.lower()
        or needle in item.email_id.lower()
    )
]

with st.container(height=400, border=False, key="si_inbox_list"):

    if not visible:
        st.markdown(
            '<div class="si-empty">No messages match the current filters.</div>',
            unsafe_allow_html=True,
        )

    for item in visible:

        status = statuses[item.email_id]
        priority = inbox_priority(item)

        row_classes = "si-row"
        if item.email_id == email.email_id:
            row_classes += " is-selected"
        if status == "New":
            row_classes += " is-new"

        priority_html = (
            f'<span class="si-row-pri'
            f'{" is-high" if priority.lower() in ("high", "urgent") else ""}">'
            f"{safe(label(priority))}</span>"
            if priority
            else ""
        )

        with st.container(key=f"si_row_{item.email_id}"):

            st.markdown(
                f'<div class="{row_classes}">'
                f'<span class="si-row-dot" '
                f'style="background:{INBOX_STATUS_COLOR[status]}"></span>'
                f'<div class="si-row-main">'
                f'<div class="si-row-top">'
                f'<span class="si-row-sender">{safe(item.sender)}</span>'
                f'<span class="si-row-id si-mono">{safe(item.email_id)}</span>'
                f'</div>'
                f'<div class="si-row-subject" dir="auto">{safe(item.subject)}</div>'
                f'<div class="si-row-preview" dir="auto">'
                f'{safe(preview_text(item.body))}'
                f'</div>'
                f'</div>'
                f'<div class="si-row-side">'
                f'<span class="si-row-status">{safe(status)}</span>'
                f'{priority_html}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.button(
                f"Open {item.email_id}",
                key=f"si_open_{item.email_id}",
                on_click=select_email,
                args=(item.email_id,),
                use_container_width=True,
            )


st.markdown(
    "<div class='si-divider' style='margin:1.5rem 0'></div>",
    unsafe_allow_html=True,
)


# ============================================================
# Workspace layout
# ============================================================

left, right = st.columns(
    [1, 1.22],
    gap="large",
)


# ============================================================
# Incoming email panel
# ============================================================

with left:

    section("Incoming email", email.email_id)

    st.markdown(
        f'<div class="si-card">'
        f'<div class="si-email-subject">'
        f'{safe(email.subject)}'
        f'</div>'
        f'<div class="si-meta-grid">'
        f'<span class="si-meta-key">From</span>'
        f'<span class="si-meta-val">{safe(email.sender)}</span>'
        f'<span class="si-meta-key">Message ID</span>'
        f'<span class="si-meta-val si-mono">{safe(email.email_id)}</span>'
        f'<span class="si-meta-key">Language</span>'
        f'<span class="si-meta-val">{safe(email.language.upper())}</span>'
        f'</div>'
        f'<div class="si-divider"></div>'
        f'<div class="si-message">'
        f'{safe(email.body)}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "Analyze with agent",
        type="primary",
        use_container_width=True,
        key="run",
    ):

        # Reset editors to display the latest generated draft.
        for key in (
            f"subj_{email.email_id}",
            f"body_{email.email_id}",
        ):
            st.session_state.pop(key, None)

        with st.spinner(
            "Classifying, searching knowledge, "
            "and checking authorized records..."
        ):
            try:
                st.session_state.results[email.email_id] = (
                    run_agent(email)
                )

                # A new analysis invalidates previous UI decisions.
                st.session_state.decisions.pop(
                    email.email_id,
                    None,
                )

            except Exception:
                # Avoid exposing credentials or stack traces in the UI.
                st.error(
                    "Analysis could not complete. "
                    "Check the local terminal for details."
                )

    st.caption(
        "The agent may propose a reply. "
        "Human approval is required before saving."
    )


# ============================================================
# Agent workspace
# ============================================================

with right:

    section("Agent workspace")

    entry = st.session_state.results.get(
        email.email_id
    )

    if entry is None:

        st.info(
            "Select a message and click "
            "**Analyze with agent** to see its "
            "classification, evidence, and draft."
        )

    else:

        result, mode = entry

        if mode == "mock":
            st.warning(
                "Mock preview. Classification labels "
                "come from evaluation data; "
                "backend lookups are real."
            )

        status_text = label(result.status)

        st.markdown(
            f'<div class="si-summary">'
            f'<div>'
            f'<div class="si-summary-title">Analysis result</div>'
            f'<div class="si-note">'
            f'{safe(len(result.tools_used))} tool(s) used'
            f' &nbsp;|&nbsp; {safe(mode.title())} mode'
            f'</div>'
            f'</div>'
            f'{badge(status_text, status_tone(result.status))}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        priority_tone = {
            "urgent": "danger",
            "high": "danger",
            "medium": "warning",
            "low": "neutral",
        }.get((result.priority or "").lower(), "neutral")

        st.markdown(
            f'<div class="si-kv-grid">'
            f'<div class="si-kv">'
            f'<div class="si-kv-label">Intent</div>'
            f'<div class="si-kv-value">{safe(label(result.intent))}</div>'
            f'</div>'
            f'<div class="si-kv">'
            f'<div class="si-kv-label">Priority</div>'
            f'<div class="si-kv-value">'
            f'{badge(label(result.priority), priority_tone)}'
            f'</div>'
            f'</div>'
            f'<div class="si-kv">'
            f'<div class="si-kv-label">Primary team</div>'
            f'<div class="si-kv-value">'
            f'{safe(label(result.primary_department))}'
            f'</div>'
            f'</div>'
            f'<div class="si-kv">'
            f'<div class="si-kv-label">Secondary team</div>'
            f'<div class="si-kv-value">'
            f'{safe(label(result.secondary_department))}'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
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

        # ----------------------------------------------------
        # Workspace tabs
        # ----------------------------------------------------

        draft_tab, evidence_tab, diagnostics_tab = st.tabs(
            [
                "Draft and approval",
                f"Evidence ({len(result.evidence)})",
                "Agent trace",
            ]
        )

        # ----------------------------------------------------
        # Draft and approval tab
        # ----------------------------------------------------

        with draft_tab:

            has_draft = (
                result.status == "draft_ready"
                and result.draft_body is not None
                and result.draft_subject is not None
            )

            if not has_draft:
                st.info(
                    "No review-ready draft. "
                    "This message may need more information "
                    "or manual review."
                )

            subject = st.text_input(
                "Reply subject",
                value=result.draft_subject or "",
                key=f"subj_{email.email_id}",
                disabled=not has_draft,
            )

            body = st.text_area(
                "Reply body",
                value=result.draft_body or "",
                height=230,
                key=f"body_{email.email_id}",
                disabled=not has_draft,
            )

            st.markdown(
                '<div class="si-subhead">Human approval</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                "Review and edit the draft before saving. "
                "Approval is tied to the exact final text "
                "and routing."
            )

            decision = st.session_state.decisions.get(
                email.email_id
            )

            if decision:

                outcome = decision.get("result") or {}

                if (
                    decision["decision"] == "approved"
                    and outcome.get("status") == "saved"
                ):
                    st.success(
                        "Approved and saved. Record: "
                        f"{outcome['record_id']}"
                    )

                elif decision["decision"] == "approved":

                    error = outcome.get("error") or {}

                    st.error(
                        "Write blocked: "
                        + str(
                            error.get(
                                "code",
                                "unknown_error",
                            )
                        )
                    )

                else:
                    st.warning(
                        "Rejected. No draft was saved."
                    )

            else:

                # Bind action ID to the exact edited draft.
                action_id = compute_action_id(
                    email.email_id,
                    result.primary_department,
                    result.secondary_department,
                    subject,
                    body,
                )

                with st.expander(
                    "Approval action details",
                    expanded=False,
                ):
                    st.code(
                        action_id,
                        language="text",
                    )

                    st.caption(
                        "Changing the message changes its action ID."
                    )

                can_approve = (
                    has_draft
                    and result.requires_approval
                    and bool(os.environ.get("APPROVAL_SECRET"))
                    and bool(subject.strip())
                    and bool(body.strip())
                )

                approve_column, reject_column = st.columns(2)

                if approve_column.button(
                    "Approve and save",
                    type="primary",
                    use_container_width=True,
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
                            issue_approval_token(action_id)
                        ),
                    }

                    with st.spinner(
                        "Saving approved draft through MCP..."
                    ):
                        outcome = call_save_draft(payload)

                    st.session_state.decisions[
                        email.email_id
                    ] = {
                        "decision": "approved",
                        "result": outcome,
                    }

                    st.rerun()

                if reject_column.button(
                    "Reject draft",
                    use_container_width=True,
                    key="reject",
                    disabled=not has_draft,
                ):

                    record_rejection(
                        action_id,
                        email.email_id,
                    )

                    st.session_state.decisions[
                        email.email_id
                    ] = {
                        "decision": "rejected",
                        "result": None,
                    }

                    st.rerun()

        # ----------------------------------------------------
        # Evidence tab
        # ----------------------------------------------------

        with evidence_tab:

            if not result.evidence:
                st.info(
                    "No supporting evidence was returned "
                    "for this message."
                )

            for index, item in enumerate(
                result.evidence,
                start=1,
            ):

                score_html = (
                    badge(f"Relevance {item.score:.2f}", "accent")
                    if item.score is not None
                    else ""
                )

                st.markdown(
                    f'<div class="si-evidence">'
                    f'<div class="si-evidence-head">'
                    f'<span class="si-evidence-source si-mono">'
                    f'{index}. {safe(item.source_id)}'
                    f'</span>'
                    f'{score_html}'
                    f'</div>'
                    f'<div class="si-evidence-text">'
                    f'{safe(item.excerpt)}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------
        # Trace and budget tab
        # ----------------------------------------------------

        with diagnostics_tab:

            st.markdown(
                '<div class="si-subhead" style="margin-top:0.4rem">'
                'Tools invoked</div>',
                unsafe_allow_html=True,
            )

            if result.tools_used:
                st.markdown(
                    " ".join(
                        badge(tool, "neutral")
                        for tool in result.tools_used
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("None")

            st.markdown(
                '<div class="si-subhead">Execution trace</div>',
                unsafe_allow_html=True,
            )

            if result.trace:
                st.dataframe(
                    [
                        item.model_dump()
                        for item in result.trace
                    ],
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.caption(
                    "No tool calls were recorded."
                )

            st.markdown(
                '<div class="si-subhead">Budget usage</div>',
                unsafe_allow_html=True,
            )

            st.json(result.budget.model_dump())

            with st.expander(
                "Audit log (latest 8 events)"
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
                    "\n".join(lines)
                    or "No audit events recorded.",
                    language="json",
                )