
"""Shared Pydantic contracts for SmartInbox AI (docs/contracts.md, proposed v1).

These models validate data shapes; actual access control, approval, budgets,
and untrusted-content handling MUST be enforced in service/tool code.
"""
from __future__ import annotations

from typing import Any, Literal, Protocol, TypeAlias
from pydantic import BaseModel, ConfigDict, Field

Department: TypeAlias = Literal[
    "hr", "finance", "customer_support", "technical_support", "security_review"
]
Priority: TypeAlias = Literal["low", "medium", "high"]
Intent: TypeAlias = Literal[
    "ambiguous", "billing_dispute", "general_question", "invoice_status",
    "multi_intent", "order_status", "recruitment", "refund_status",
    "return_request", "service_outage", "suspicious_request", "technical_issue",
]
AgentStatus: TypeAlias = Literal[
    "draft_ready", "needs_information", "manual_review", "blocked", "error", "budget_exhausted"
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmailInput(StrictModel):
    email_id: str
    sender: str  # Trusted fixture in demo; NOT an authenticated identity in production.
    subject: str
    body: str
    language: str


class ContractError(StrictModel):
    code: str
    message: str


class Evidence(StrictModel):
    source_id: str
    excerpt: str
    score: float | None = None


class ToolTrace(StrictModel):
    step: int = Field(ge=1)
    tool: str
    status: str


class BudgetUsage(StrictModel):
    steps_used: int = Field(ge=0)
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    estimated_cost_usd: float | None = Field(ge=0)


class AgentResult(StrictModel):
    # All keys required; nullable fields must be explicitly null.
    email_id: str
    intent: Intent
    primary_department: Department
    secondary_department: Department | None
    priority: Priority
    status: AgentStatus
    draft_subject: str | None
    draft_body: str | None
    tools_used: list[str]
    evidence: list[Evidence]
    security_flags: list[str]
    requires_approval: bool
    proposed_action_id: str | None
    trace: list[ToolTrace]
    budget: BudgetUsage
    error: ContractError | None


class ToolResult(StrictModel):
    status: Literal["success", "error"]
    data: Any | None
    error: ContractError | None


class KnowledgeHit(StrictModel):
    source_id: str
    excerpt: str
    score: float | None = None


class ApprovalWriteRequest(StrictModel):
    # Internal/server-side only: a model MUST NOT fabricate approval_token.
    action_id: str
    email_id: str
    primary_department: Department
    secondary_department: Department | None
    draft_subject: str
    draft_body: str
    approval_token: str


class MCPWriteSuccess(StrictModel):
    status: Literal["saved"] = "saved"
    record_id: str


class MCPWriteFailure(StrictModel):
    status: Literal["error"] = "error"
    error: ContractError


class AgentConfig(StrictModel):
    max_steps: int = Field(default=6, ge=1)
    retrieval_top_k: int = Field(default=3, ge=1)
    max_input_tokens: int = Field(default=12000, ge=1)
    max_output_tokens: int = Field(default=6000, ge=1)
    max_estimated_cost_usd: float = Field(default=0.10, gt=0)


class AgentTools(Protocol):
    """Agreed read-only tool signatures, implemented by the backend."""

    def search_company_knowledge(self, query: str, top_k: int = 3) -> dict: ...
    def lookup_order(self, order_id: str, requester_email: str) -> dict: ...
    def lookup_invoice(self, invoice_id: str, requester_email: str) -> dict: ...
    def lookup_ticket(self, ticket_id: str, requester_email: str) -> dict: ...


class EvalCase(StrictModel):
    email_id: str
    expected_intent: Intent
    expected_department: Department
    expected_secondary_department: Department | None
    expected_priority: Priority
    required_tools: list[str]
    requires_approval_for_write: bool
    expected_behavior: str
