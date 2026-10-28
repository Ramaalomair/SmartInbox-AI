
import asyncio

from agent.orchestrator import process_email
from agent.schemas import EmailInput, AgentConfig
from agent.tools import bind_requester
from agent.integrated_tools import IntegratedAgentTools


async def main():
    """Test classification, retrieval, and record lookup."""

    
    email = EmailInput(
    email_id="TEST-ORDER",
    sender="sender26@example.com",
    subject="Missing order",
    body="My order ORD-1005 has not arrived. Can you check its status?",
    language="en"
)


    tools = IntegratedAgentTools(
    bind_requester(email.sender)
)

    result = await process_email(
        email=email,
        tools=tools,
        config=AgentConfig(
            max_steps=6,
            retrieval_top_k=3,
            max_input_tokens=12000,
            max_output_tokens=2000,
            max_estimated_cost_usd=0.10
        )
    )

    print("\nAgent Result:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
