
from agent.classification import classify_email
from agent.tool_selection import select_record_tools
from agent.tool_execution import execute_record_lookups
from agent.draft import generate_draft

from agent.llm import (
    start_usage_tracking,
    get_usage,
    stop_usage_tracking,
)

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


# Current LLM calls use up to 1000 output tokens each.
LLM_CALL_MAX_OUTPUT_TOKENS = 1000


async def process_email(
    email: EmailInput,
    tools: AgentTools,
    config: AgentConfig | None = None
) -> AgentResult:
    """Process an email with grounded drafting and usage tracking."""

    if config is None:
        config = AgentConfig()

    tools_used = []
    evidence = []
    trace = []
    security_flags = []

    verified_records = []
    draft_subject = None
    draft_body = None
    requires_approval = False

    steps_used = 0
    classification = None
    status = "needs_information"
    error_info = None

    # Isolate usage counters for this email-processing request.
    usage_token = start_usage_tracking()

    def current_budget() -> BudgetUsage:
        """Build budget information from the current LLM context."""

        usage = get_usage()

        return BudgetUsage(
            steps_used=steps_used,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            estimated_cost_usd=usage["estimated_cost_usd"],
        )

    def build_result() -> AgentResult:
        """Build the final result using the shared contract."""

        return AgentResult(
            email_id=email.email_id,
            intent=(
                classification.intent
                if classification else "ambiguous"
            ),
            primary_department=(
                classification.primary_department
                if classification else "security_review"
            ),
            secondary_department=(
                classification.secondary_department
                if classification else None
            ),
            priority=(
                classification.priority
                if classification else "medium"
            ),
            status=status,
            draft_subject=draft_subject,
            draft_body=draft_body,
            tools_used=tools_used,
            evidence=evidence,
            security_flags=security_flags,
            requires_approval=requires_approval,
            proposed_action_id=None,
            trace=trace,
            budget=current_budget(),
            error=error_info,
        )

    def budget_exceeded() -> bool:
        """Check cumulative provider-reported usage against limits."""

        usage = get_usage()

        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        cost = usage["estimated_cost_usd"]

        if (
            input_tokens is not None
            and input_tokens > config.max_input_tokens
        ):
            return True

        if (
            output_tokens is not None
            and output_tokens > config.max_output_tokens
        ):
            return True

        if (
            cost is not None
            and cost > config.max_estimated_cost_usd
        ):
            return True

        return False

    def can_start_llm_call() -> bool:
        """Check whether enough reported budget remains for a call."""

        if steps_used >= config.max_steps:
            return False

        if budget_exceeded():
            return False

        usage = get_usage()

        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        cost = usage["estimated_cost_usd"]

        if (
            input_tokens is not None
            and input_tokens >= config.max_input_tokens
        ):
            return False

        if (
            output_tokens is not None
            and (
                config.max_output_tokens - output_tokens
                < LLM_CALL_MAX_OUTPUT_TOKENS
            )
        ):
            return False

        if (
            cost is not None
            and cost >= config.max_estimated_cost_usd
        ):
            return False

        return True

    try:
        if tools is None:
            raise ValueError("AgentTools implementation is required.")

        # Step 1: Classify the incoming email.
        if not can_start_llm_call():
            status = "budget_exhausted"
            return build_result()

        steps_used += 1
        classification = classify_email(email)

        # Check provider-reported usage after classification.
        if budget_exceeded():
            status = "budget_exhausted"
            return build_result()

        # Suspicious requests require human security review.
        if classification.intent == "suspicious_request":
            security_flags.append("suspicious_request")
            status = "manual_review"
            return build_result()

        # Step 2: Retrieve relevant company policies.
        if steps_used >= config.max_steps:
            status = "budget_exhausted"
            return build_result()

        steps_used += 1
        tools_used.append("search_company_knowledge")

        try:
            query = f"{email.subject} {email.body}"

            search_result = tools.search_company_knowledge(
                query=query,
                top_k=config.retrieval_top_k,
            )

            if search_result.get("status") != "success":
                raise ValueError("Knowledge retrieval failed.")

            for hit in search_result.get("data") or []:
                evidence.append(
                    Evidence(
                        source_id=hit["source_id"],
                        excerpt=hit["excerpt"],
                        score=hit.get("score"),
                    )
                )

            trace.append(
                ToolTrace(
                    step=steps_used,
                    tool="search_company_knowledge",
                    status="success",
                )
            )

        except Exception:
            trace.append(
                ToolTrace(
                    step=steps_used,
                    tool="search_company_knowledge",
                    status="error",
                )
            )

            status = "manual_review"
            return build_result()

        if not evidence:
            status = "needs_information"
            return build_result()

        # Step 3: Identify required read-only record lookups.
        selections = select_record_tools(
            email=email,
            intent=classification.intent,
        )

        # Reserve one step for draft generation.
        remaining_lookup_steps = max(
            0,
            config.max_steps - steps_used - 1,
        )

        selected_to_execute = selections[:remaining_lookup_steps]
        skipped_count = len(selections) - len(selected_to_execute)

        if skipped_count > 0:
            status = "budget_exhausted"
            return build_result()

        # Execute selected record lookups.
        if selected_to_execute:
            lookup_results = execute_record_lookups(
                selections=selected_to_execute,
                email=email,
                tools=tools,
            )

            for lookup in lookup_results:
                steps_used += 1

                tool_name = lookup["tool"]
                tool_result = lookup["result"]

                tools_used.append(tool_name)

                tool_status = (
                    "success"
                    if tool_result.get("status") == "success"
                    else "error"
                )

                trace.append(
                    ToolTrace(
                        step=steps_used,
                        tool=tool_name,
                        status=tool_status,
                    )
                )

                if tool_status == "success":
                    # Authorization must be enforced by backend tools.
                    record_data = tool_result.get("data")

                    if record_data is None:
                        status = "manual_review"
                    else:
                        verified_records.append(
                            {
                                "tool": tool_name,
                                "data": record_data,
                            }
                        )
                else:
                    # Never draft claims based on failed lookups.
                    status = "manual_review"

        if status == "manual_review":
            return build_result()

        # Step 4: Generate a grounded response draft.
        if not can_start_llm_call():
            status = "budget_exhausted"
            return build_result()

        steps_used += 1

        try:
            draft_result = generate_draft(
                email=email,
                evidence=evidence,
                verified_records=verified_records,
            )

            generated_subject = draft_result.get("draft_subject")
            generated_body = draft_result.get("draft_body")

            if not generated_subject or not generated_body:
                raise ValueError("Draft output is incomplete.")

            # Check total usage before accepting the draft.
            if budget_exceeded():
                status = "budget_exhausted"
                return build_result()

            draft_subject = generated_subject
            draft_body = generated_body
            requires_approval = True
            status = "draft_ready"

        except Exception as exc:
            # Preserve evidence and the tool trace on LLM failure.
            print(
                f"Draft generation failed: {type(exc).__name__}"
            )

            draft_subject = None
            draft_body = None
            requires_approval = False

            status = "manual_review"
            error_info = ContractError(
                code="draft_generation_failed",
                message="Unable to generate a grounded draft.",
            )

        return build_result()

    except Exception as exc:
        # Avoid exposing internal exception details to callers.
        print(
            f"Orchestrator error: {type(exc).__name__}"
        )

        status = "error"
        draft_subject = None
        draft_body = None
        requires_approval = False

        error_info = ContractError(
            code="orchestration_failed",
            message="Unable to complete email processing.",
        )

        return build_result()

    finally:
        # Clean up usage tracking even when processing fails.
        stop_usage_tracking(usage_token)
