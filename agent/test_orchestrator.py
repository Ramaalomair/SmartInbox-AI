
import asyncio

from agent.mock_tools import MockAgentTools
from agent.orchestrator import process_email
from agent.schemas import EmailInput, AgentConfig


async def main():
    """Test classification, retrieval, and record lookup."""

    
    email = EmailInput(
     email_id="TEST-POLICY",
     sender="customer@example.com",
     subject="Return policy question",
     body="What is the standard return window for a product?",
     language="en"
    )


    tools = MockAgentTools()

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
