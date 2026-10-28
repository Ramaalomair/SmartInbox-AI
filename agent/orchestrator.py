
from agent.classification import classify_email

from agent.schemas import (
    EmailInput,
    AgentTools,
    AgentConfig,
    AgentResult,
    BudgetUsage,
    ContractError,
    Evidence,
    ToolTrace,
)


async def process_email(
    email: EmailInput,
    tools: AgentTools,
    config: AgentConfig
) -> AgentResult:
    """Classify an email and retrieve policy evidence."""

    tools_used = []
    evidence = []
    trace = []
    steps_used = 0

    try:
        if tools is None:
            raise ValueError("AgentTools implementation is required.")

        # Step 1: Classify the email.
        if steps_used >= config.max_steps:
            raise RuntimeError("Step budget exhausted.")

        classification = classify_email(email)
        steps_used += 1

        # Step 2: Retrieve company policies.
        if steps_used >= config.max_steps:
            return AgentResult(
                email_id=email.email_id,
                intent=classification.intent,
                primary_department=classification.primary_department,
                secondary_department=classification.secondary_department,
                priority=classification.priority,
                status="budget_exhausted",
                draft_subject=None,
                draft_body=None,
                tools_used=tools_used,
                evidence=evidence,
                security_flags=[],
                requires_approval=False,
                proposed_action_id=None,
                trace=trace,
                budget=BudgetUsage(
                    steps_used=steps_used,
                    input_tokens=None,
                    output_tokens=None,
                    estimated_cost_usd=None
                ),
                error=None
            )

        query = f"{email.subject} {email.body}"

        result = tools.search_company_knowledge(
            query=query,
            top_k=config.retrieval_top_k
        )

        steps_used += 1
        tools_used.append("search_company_knowledge")

        trace.append(
            ToolTrace(
                step=steps_used,
                tool="search_company_knowledge",
                status=result["status"]
            )
        )

        if result["status"] == "success":
            for hit in result["data"]:
                evidence.append(
                    Evidence(
                        source_id=hit["source_id"],
                        excerpt=hit["excerpt"],
                        score=hit.get("score")
                    )
                )

        return AgentResult(
            email_id=email.email_id,
            intent=classification.intent,
            primary_department=classification.primary_department,
            secondary_department=classification.secondary_department,
            priority=classification.priority,
            status="needs_information",
            draft_subject=None,
            draft_body=None,
            tools_used=tools_used,
            evidence=evidence,
            security_flags=[],
            requires_approval=False,
            proposed_action_id=None,
            trace=trace,
            budget=BudgetUsage(
                steps_used=steps_used,
                input_tokens=None,
                output_tokens=None,
                estimated_cost_usd=None
            ),
            error=None
        )

    except Exception as error:
        print(f"Orchestrator error: {type(error).__name__}: {error}")

        return AgentResult(
            email_id=email.email_id,
            intent="ambiguous",
            primary_department="security_review",
            secondary_department=None,
            priority="medium",
            status="error",
            draft_subject=None,
            draft_body=None,
            tools_used=tools_used,
            evidence=evidence,
            security_flags=[],
            requires_approval=False,
            proposed_action_id=None,
            trace=trace,
            budget=BudgetUsage(
                steps_used=steps_used,
                input_tokens=None,
                output_tokens=None,
                estimated_cost_usd=None
            ),
            error=ContractError(
                code="orchestration_failed",
                message="Unable to process the email."
            )
        )
