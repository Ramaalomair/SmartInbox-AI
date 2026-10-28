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
        --si-border: rgba(140, 150, 170, 0.20);
        --si-surface: rgba(140, 150, 170, 0.05);
        --si-surface-2: rgba(140, 150, 170, 0.10);
        --si-success: #22a06b;
        --si-warning: #c88a12;
        --si-danger: #d9534f;
        --si-neutral: #7a8699;
        --si-radius: 10px;
    }

    html, body, .stApp, .stMarkdown, .stTextInput,
    .stTextArea, .stButton {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont,
            'Segoe UI', Roboto, 'Noto Sans Arabic', sans-serif;
    }

    code, pre, .si-mono {
        font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular,
            Menlo, monospace !important;
    }

    /* Keep the Streamlit header out of the way without hiding controls. */
    [data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu, footer, [data-testid="stDecoration"],
    .stAppDeployButton, [data-testid="stAppDeployButton"] {
        display: none !important;
    }

    .block-container {
        padding-top: 3.25rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--si-border);
    }

    .si-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 1.25rem;
    }

    .si-brand-mark {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: var(--si-accent);
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.8rem;
        flex-shrink: 0;
    }

    .si-brand-name {
        font-weight: 650;
        font-size: 1rem;
        line-height: 1.2;
    }

    .si-brand-sub {
        font-size: 0.74rem;
        opacity: 0.55;
    }

    .si-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.55;
        margin: 0.25rem 0 0.5rem;
    }

    .si-status-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
        opacity: 0.8;
        margin: 0.25rem 0 0.75rem;
    }

    .si-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        flex-shrink: 0;
        display: inline-block;
    }

    .si-footnote {
        font-size: 0.74rem;
        opacity: 0.5;
        line-height: 1.5;
        margin-top: 0.75rem;
    }

    /* ---------- Page header ---------- */

    .si-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        padding-bottom: 0.9rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--si-border);
    }

    .si-title {
        font-weight: 650;
        font-size: 1.35rem;
        letter-spacing: -0.015em;
        line-height: 1.2;
    }

    .si-subtitle {
        font-size: 0.82rem;
        opacity: 0.55;
        margin-top: 0.15rem;
    }

    .si-mode {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.74rem;
        font-weight: 500;
        padding: 0.3rem 0.65rem;
        border: 1px solid var(--si-border);
        border-radius: 999px;
        white-space: nowrap;
        cursor: default;
    }

    .si-pane-title {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    .si-pane-hint {
        font-size: 0.75rem;
        font-weight: 400;
        opacity: 0.5;
    }

    /* ---------- Inbox list ---------- */

    .st-key-si_inbox_list {
        border: 1px solid var(--si-border);
        border-radius: var(--si-radius);
        padding: 0.3rem !important;
        gap: 0.2rem !important;
    }

    [class*="st-key-si_row_"] {
        position: relative;
        gap: 0 !important;
        flex-shrink: 0;
    }

    /* Streamlit gives markdown blocks a negative bottom margin, which
       makes each row shorter than its content and lets highlights
       overlap the next email. Reset it inside inbox rows only. */
    [class*="st-key-si_row_"] [data-testid="stElementContainer"],
    [class*="st-key-si_row_"] .stMarkdown,
    [class*="st-key-si_row_"] [data-testid="stMarkdownContainer"] {
        margin: 0 !important;
        padding: 0 !important;
    }

    .si-row {
        display: flex;
        align-items: flex-start;
        gap: 0.65rem;
        padding: 0.6rem 0.7rem;
        border: 1px solid transparent;
        border-radius: 8px;
        transition: background 0.12s ease, border-color 0.12s ease;
    }

    [class*="st-key-si_row_"]:hover .si-row {
        background: var(--si-surface-2);
    }

    .si-row.is-selected {
        background: var(--si-accent-soft);
        border-color: rgba(79, 124, 255, 0.30);
    }

    .si-row .si-dot {
        margin-top: 0.4rem;
    }

    .si-row-main {
        flex: 1;
        min-width: 0;
    }

    .si-row-line {
        display: flex;
        align-items: baseline;
        gap: 0.5rem;
        min-width: 0;
    }

    .si-row-sender {
        flex: 1;
        min-width: 0;
        font-size: 0.76rem;
        opacity: 0.65;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .si-row-status {
        font-size: 0.7rem;
        font-weight: 500;
        opacity: 0.6;
        flex-shrink: 0;
    }

    .si-row-subject {
        font-size: 0.87rem;
        font-weight: 500;
        margin: 0.1rem 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .si-row.is-new .si-row-subject {
        font-weight: 650;
    }

    .si-row-preview {
        flex: 1;
        min-width: 0;
        font-size: 0.78rem;
        opacity: 0.5;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .si-row-pri {
        font-size: 0.66rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        opacity: 0.55;
        flex-shrink: 0;
    }

    .si-row-pri.is-high {
        color: var(--si-danger);
        opacity: 0.9;
    }

    /* Invisible full-row button that makes each inbox item clickable. */
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
        font-size: 0.82rem;
        opacity: 0.5;
        text-align: center;
        padding: 2rem 0.5rem;
    }

    /* ---------- Selected email ---------- */

    .si-mail {
        border: 1px solid var(--si-border);
        border-radius: var(--si-radius);
        background: var(--si-surface);
        padding: 1rem 1.15rem;
        margin-bottom: 0.75rem;
    }

    .si-mail-subject {
        font-size: 1.1rem;
        font-weight: 600;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }

    .si-mail-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem 0.9rem;
        font-size: 0.78rem;
        margin: 0.35rem 0 0.75rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--si-border);
    }

    .si-mail-meta span {
        opacity: 0.6;
    }

    .si-mail-meta b {
        font-weight: 500;
        opacity: 0.9;
        overflow-wrap: anywhere;
    }

    .si-mail-body {
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        unicode-bidi: plaintext;
        text-align: start;
        line-height: 1.65;
        font-size: 0.9rem;
        max-height: 200px;
        overflow-y: auto;
    }

    /* ---------- AI analysis ---------- */

    .si-section {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.55;
        margin: 1rem 0 0.45rem;
    }

    .si-analysis {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        border: 1px solid var(--si-border);
        border-radius: var(--si-radius);
        overflow: hidden;
    }

    .si-cell {
        padding: 0.65rem 0.85rem;
        border-right: 1px solid var(--si-border);
        min-width: 0;
    }

    .si-cell:last-child {
        border-right: none;
    }

    .si-cell-label {
        font-size: 0.68rem;
        opacity: 0.5;
        margin-bottom: 0.2rem;
    }

    .si-cell-value {
        font-size: 0.86rem;
        font-weight: 600;
        overflow-wrap: anywhere;
    }

    .si-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border-radius: 999px;
        padding: 0.12rem 0.55rem;
        font-size: 0.74rem;
        font-weight: 600;
        white-space: nowrap;
    }

    .si-badge::before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: currentColor;
    }

    .si-tone-success { color: var(--si-success); background: rgba(34, 160, 107, 0.12); }
    .si-tone-warning { color: var(--si-warning); background: rgba(200, 138, 18, 0.12); }
    .si-tone-danger  { color: var(--si-danger);  background: rgba(217, 83, 79, 0.12); }
    .si-tone-neutral { color: var(--si-neutral); background: rgba(122, 134, 153, 0.14); }

    .si-placeholder {
        border: 1px dashed var(--si-border);
        border-radius: var(--si-radius);
        padding: 1.1rem;
        font-size: 0.84rem;
        opacity: 0.6;
        text-align: center;
        margin-top: 0.75rem;
    }

    .si-hint {
        font-size: 0.76rem;
        opacity: 0.5;
        margin: 0.35rem 0 0.5rem;
    }

    /* ---------- Evidence ---------- */

    .si-evidence {
        border-left: 2px solid var(--si-accent);
        padding: 0.15rem 0 0.15rem 0.75rem;
        margin-bottom: 0.7rem;
    }

    .si-evidence-source {
        font-size: 0.76rem;
        font-weight: 600;
        overflow-wrap: anywhere;
    }

    .si-evidence-score {
        font-weight: 400;
        opacity: 0.55;
        margin-left: 0.4rem;
    }

    .si-evidence-text {
        font-size: 0.82rem;
        line-height: 1.55;
        opacity: 0.75;
        overflow-wrap: anywhere;
        unicode-bidi: plaintext;
        margin-top: 0.15rem;
    }

    /* ---------- Streamlit widgets ---------- */

    .stButton > button {
        border-radius: 8px;
        min-height: 2.5rem;
        font-weight: 600;
        font-size: 0.87rem;
    }

    /* Green primary actions instead of Streamlit's default red. */
    .stButton button[kind="primary"],
    [data-testid="stBaseButton-primary"] {
        background: #1f9d63 !important;
        border-color: #1f9d63 !important;
        color: #ffffff !important;
    }

    .stButton button[kind="primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background: #23b06f !important;
        border-color: #23b06f !important;
    }

    .stButton button[kind="primary"]:active,
    [data-testid="stBaseButton-primary"]:active {
        background: #1a8754 !important;
        border-color: #1a8754 !important;
    }

    .stButton button[kind="primary"]:disabled,
    [data-testid="stBaseButton-primary"]:disabled {
        background: rgba(31, 157, 99, 0.25) !important;
        border-color: transparent !important;
        color: rgba(255, 255, 255, 0.5) !important;
    }

    .stTextInput input, .stTextArea textarea {
        border-radius: 8px;
        font-size: 0.9rem;
        unicode-bidi: plaintext;
    }

    [data-testid="stExpander"] {
        border-radius: var(--si-radius);
    }

    [data-testid="stExpander"] summary p {
        font-size: 0.85rem;
        font-weight: 500;
    }

    [data-testid="stAlert"] {
        border-radius: var(--si-radius);
    }

    @media (max-width: 1100px) {
        .si-analysis {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .si-cell:nth-child(2) {
            border-right: none;
        }

        .si-cell:nth-child(-n+2) {
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


def badge(text: str, tone: str = "neutral") -> str:
    """Render a small status badge."""
    return (
        f'<span class="si-badge si-tone-{tone}">'
        f"{safe(text)}</span>"
    )


def status_tone(value: str | None) -> str:
    """Map an agent status to a badge color tone (display only)."""
    return {
        "draft_ready": "success",
        "needs_information": "warning",
        "manual_review": "danger",
    }.get((value or "").lower(), "neutral")


def priority_tone(value: str | None) -> str:
    """Map a priority to a badge color tone (display only)."""
    return {
        "urgent": "danger",
        "high": "danger",
        "medium": "warning",
    }.get((value or "").lower(), "neutral")


INBOX_FILTERS = ["All", "New", "Analyzed", "Approved", "Rejected"]

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


def preview_text(body: str, limit: int = 110) -> str:
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
st.session_state.setdefault("dev_mode", False)

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

approval_enabled = bool(os.environ.get("APPROVAL_SECRET"))


# ============================================================
# Sidebar: settings only
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

    st.markdown(
        '<div class="si-label">Settings</div>',
        unsafe_allow_html=True,
    )

    if real_process_email is None:
        st.caption(
            "Live agent unavailable. Running in mock mode."
        )
    else:
        st.toggle(
            "Use mock agent",
            key="force_mock",
            help=(
                "Mock mode uses evaluation labels for classification. "
                "Do not use its predictions for model evaluation."
            ),
        )

    st.toggle(
        "Developer mode",
        key="dev_mode",
        help="Show agent trace, budget usage, and audit details.",
    )

    if approval_enabled:
        st.markdown(
            '<div class="si-status-row">'
            '<span class="si-dot" '
            'style="background:var(--si-success)"></span>'
            'Human approval enabled'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.error(
            "APPROVAL_SECRET is missing. "
            "Draft approvals are disabled."
        )

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
        '<div class="si-footnote">'
        'Synthetic data only. Approved drafts are saved, '
        'never sent.'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Page header with mode indicator
# ============================================================

is_mock = real_process_email is None or st.session_state.force_mock

mode_label = "Mock agent" if is_mock else "Live agent"
mode_color = "var(--si-warning)" if is_mock else "var(--si-success)"
mode_tip = (
    "Classification comes from evaluation labels; "
    "backend lookups are real."
    if is_mock
    else "Classification and drafts come from the live AI agent."
)

st.markdown(
    f'<div class="si-header">'
    f'<div>'
    f'<div class="si-title">SmartInbox</div>'
    f'<div class="si-subtitle">'
    f'AI-assisted replies with human approval'
    f'</div>'
    f'</div>'
    f'<span class="si-mode" title="{safe(mode_tip)}">'
    f'<span class="si-dot" style="background:{mode_color}"></span>'
    f'{safe(mode_label)}'
    f'</span>'
    f'</div>',
    unsafe_allow_html=True,
)


# ============================================================
# Two-column layout
# ============================================================

inbox_col, detail_col = st.columns([1, 1.55], gap="large")


# ============================================================
# Left: inbox list
# ============================================================

with inbox_col:

    statuses = {item.email_id: inbox_status(item) for item in emails}

    st.markdown(
        f'<div class="si-pane-title">Inbox'
        f'<span class="si-pane-hint">{len(emails)} messages</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Search inbox",
        placeholder="Search subject, sender, or ID",
        key="inbox_query",
        label_visibility="collapsed",
    )

    # Pills wrap onto a new line in narrow columns instead of overflowing.
    if hasattr(st, "pills"):
        status_filter = st.pills(
            "Status",
            INBOX_FILTERS,
            default="All",
            key="inbox_status_filter",
            label_visibility="collapsed",
        ) or "All"
    else:
        status_filter = st.radio(
            "Status",
            INBOX_FILTERS,
            key="inbox_status_filter",
            horizontal=True,
            label_visibility="collapsed",
        )

    needle = query.strip().lower()

    visible = [
        item
        for item in emails
        if (
            status_filter == "All"
            or statuses[item.email_id] == status_filter
        )
        and (
            not needle
            or needle in item.subject.lower()
            or needle in item.sender.lower()
            or needle in item.email_id.lower()
        )
    ]

    with st.container(height=560, border=False, key="si_inbox_list"):

        if not visible:
            st.markdown(
                '<div class="si-empty">'
                'No messages match the current filters.'
                '</div>',
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

            priority_html = ""
            if priority:
                high = priority.lower() in ("high", "urgent")
                priority_html = (
                    f'<span class="si-row-pri'
                    f'{" is-high" if high else ""}">'
                    f"{safe(label(priority))}</span>"
                )

            with st.container(key=f"si_row_{item.email_id}"):

                st.markdown(
                    f'<div class="{row_classes}">'
                    f'<span class="si-dot" '
                    f'style="background:{INBOX_STATUS_COLOR[status]}">'
                    f'</span>'
                    f'<div class="si-row-main">'
                    f'<div class="si-row-line">'
                    f'<span class="si-row-sender">{safe(item.sender)}</span>'
                    f'<span class="si-row-status">{safe(status)}</span>'
                    f'</div>'
                    f'<div class="si-row-subject" dir="auto">'
                    f'{safe(item.subject)}'
                    f'</div>'
                    f'<div class="si-row-line">'
                    f'<span class="si-row-preview" dir="auto">'
                    f'{safe(preview_text(item.body))}'
                    f'</span>'
                    f'{priority_html}'
                    f'</div>'
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


# ============================================================
# Right: selected email, analysis, reply, approval
# ============================================================

with detail_col:

    # --------------------------------------------------------
    # Selected email
    # --------------------------------------------------------

    st.markdown(
        f'<div class="si-mail">'
        f'<div class="si-mail-subject" dir="auto">'
        f'{safe(email.subject)}'
        f'</div>'
        f'<div class="si-mail-meta">'
        f'<span>From <b>{safe(email.sender)}</b></span>'
        f'<span>ID <b class="si-mono">{safe(email.email_id)}</b></span>'
        f'<span>Language <b>{safe(email.language.upper())}</b></span>'
        f'</div>'
        f'<div class="si-mail-body">{safe(email.body)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "Analyze with AI",
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

    entry = st.session_state.results.get(email.email_id)

    if entry is None:

        st.markdown(
            '<div class="si-placeholder">'
            'Run the analysis to classify this message '
            'and draft a suggested reply.'
            '</div>',
            unsafe_allow_html=True,
        )

    else:

        result, mode = entry

        # ----------------------------------------------------
        # AI analysis
        # ----------------------------------------------------

        department = label(result.primary_department)
        if result.secondary_department:
            department += f" + {label(result.secondary_department)}"

        st.markdown(
            f'<div class="si-section">AI analysis</div>'
            f'<div class="si-analysis">'
            f'<div class="si-cell">'
            f'<div class="si-cell-label">Intent</div>'
            f'<div class="si-cell-value">{safe(label(result.intent))}</div>'
            f'</div>'
            f'<div class="si-cell">'
            f'<div class="si-cell-label">Priority</div>'
            f'<div class="si-cell-value">'
            f'{badge(label(result.priority), priority_tone(result.priority))}'
            f'</div>'
            f'</div>'
            f'<div class="si-cell">'
            f'<div class="si-cell-label">Department</div>'
            f'<div class="si-cell-value">{safe(department)}</div>'
            f'</div>'
            f'<div class="si-cell">'
            f'<div class="si-cell-label">Status</div>'
            f'<div class="si-cell-value">'
            f'{badge(label(result.status), status_tone(result.status))}'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if result.security_flags:
            st.warning(
                "Security review: "
                + ", ".join(label(flag) for flag in result.security_flags)
            )

        if result.error:
            st.error(
                f"{result.error.code}: "
                f"{result.error.message}"
            )

        # ----------------------------------------------------
        # Suggested reply and human approval
        # ----------------------------------------------------

        has_draft = (
            result.status == "draft_ready"
            and result.draft_body is not None
            and result.draft_subject is not None
        )

        decision = st.session_state.decisions.get(email.email_id)

        st.markdown(
            '<div class="si-section">Suggested reply</div>',
            unsafe_allow_html=True,
        )

        action_id = None

        if not has_draft:

            st.markdown(
                '<div class="si-placeholder" style="margin-top:0">'
                'No reply was drafted. This message needs more '
                'information or a manual review.'
                '</div>',
                unsafe_allow_html=True,
            )

        else:

            subject = st.text_input(
                "Subject",
                value=result.draft_subject or "",
                key=f"subj_{email.email_id}",
                disabled=decision is not None,
            )

            body = st.text_area(
                "Message",
                value=result.draft_body or "",
                height=200,
                key=f"body_{email.email_id}",
                disabled=decision is not None,
            )

            # Bind the action ID to the exact edited draft.
            action_id = compute_action_id(
                email.email_id,
                result.primary_department,
                result.secondary_department,
                subject,
                body,
            )

        if decision:

            outcome = decision.get("result") or {}

            if (
                decision["decision"] == "approved"
                and outcome.get("status") == "saved"
            ):
                st.success(
                    "Approved and saved as draft "
                    f"{outcome['record_id']}. Nothing was sent."
                )

            elif decision["decision"] == "approved":

                error = outcome.get("error") or {}

                st.error(
                    "Save was blocked: "
                    + str(
                        error.get(
                            "code",
                            "unknown_error",
                        )
                    )
                )

            else:
                st.info("Draft rejected. Nothing was saved.")

        elif has_draft:

            st.markdown(
                '<div class="si-hint">'
                'Review and edit before approving. Approval saves '
                'this exact text as a draft; it is never sent.'
                '</div>',
                unsafe_allow_html=True,
            )

            can_approve = (
                result.requires_approval
                and approval_enabled
                and bool(subject.strip())
                and bool(body.strip())
            )

            approve_column, reject_column = st.columns(2)

            if approve_column.button(
                "Approve & Save",
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
                "Reject",
                use_container_width=True,
                key="reject",
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
        # Supporting evidence
        # ----------------------------------------------------

        if result.evidence:

            st.markdown(
                "<div style='height:0.5rem'></div>",
                unsafe_allow_html=True,
            )

            with st.expander(
                f"Supporting evidence ({len(result.evidence)})"
            ):

                for item in result.evidence:

                    score_html = (
                        f'<span class="si-evidence-score">'
                        f'relevance {item.score:.2f}</span>'
                        if item.score is not None
                        else ""
                    )

                    st.markdown(
                        f'<div class="si-evidence">'
                        f'<div class="si-evidence-source si-mono">'
                        f'{safe(item.source_id)}{score_html}'
                        f'</div>'
                        f'<div class="si-evidence-text">'
                        f'{safe(item.excerpt)}'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ----------------------------------------------------
        # Developer mode: diagnostics only
        # ----------------------------------------------------

        if st.session_state.dev_mode:

            with st.expander("Developer details"):

                st.caption(
                    f"Agent mode: {mode}. "
                    + (
                        "Classification labels come from evaluation "
                        "data; backend lookups are real."
                        if mode == "mock"
                        else "Results come from the live agent."
                    )
                )

                st.markdown("**Tools invoked**")
                st.write(", ".join(result.tools_used) or "None")

                st.markdown("**Execution trace**")
                if result.trace:
                    st.dataframe(
                        [item.model_dump() for item in result.trace],
                        hide_index=True,
                        width="stretch",
                    )
                else:
                    st.caption("No tool calls were recorded.")

                st.markdown("**Budget usage**")
                st.json(result.budget.model_dump())

                if action_id:
                    st.markdown("**Approval action ID**")
                    st.code(action_id, language="text")
                    st.caption(
                        "Changing the draft changes its action ID."
                    )

                st.markdown("**Audit log (latest 8 events)**")

                path = storage_dir() / "audit_log.jsonl"

                lines = (
                    path.read_text(encoding="utf-8").splitlines()[-8:]
                    if path.exists()
                    else []
                )

                st.code(
                    "\n".join(lines) or "No audit events recorded.",
                    language="json",
                )