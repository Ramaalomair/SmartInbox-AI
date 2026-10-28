
import asyncio

from agent.mock_tools import MockAgentTools
from agent.orchestrator import process_email
from agent.schemas import EmailInput, AgentConfig


async def main():
    """Test classification and policy retrieval integration."""

    email = EmailInput(
        email_id="TEST-001",
        sender="customer@example.com",
        subject="Delayed order",
        body="My order has not arrived yet.",
        language="en"
    )

    tools = MockAgentTools()

    result = await process_email(
        email=email,
        tools=tools,
        config=AgentConfig()
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
