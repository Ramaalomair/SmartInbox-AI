
from agent.hybrid_retrieval import HybridKnowledgeRetriever


class MockAgentTools:
    """Local read-only tools for agent integration tests."""

    def __init__(self):
        self.retriever = HybridKnowledgeRetriever()

    def search_company_knowledge(
        self,
        query: str,
        top_k: int = 3
    ) -> dict:
        """Search company policies using hybrid retrieval."""

        results = self.retriever.search(
            query=query,
            top_k=top_k
        )

        return {
            "status": "success",
            "data": results,
            "error": None
        }

    def lookup_order(
        self,
        order_id: str,
        requester_email: str
    ) -> dict:
        return {
            "status": "error",
            "data": None,
            "error": {
                "code": "tool_not_implemented",
                "message": "Order lookup is not available yet."
            }
        }

    def lookup_invoice(
        self,
        invoice_id: str,
        requester_email: str
    ) -> dict:
        return {
            "status": "error",
            "data": None,
            "error": {
                "code": "tool_not_implemented",
                "message": "Invoice lookup is not available yet."
            }
        }

    def lookup_ticket(
        self,
        ticket_id: str,
        requester_email: str
    ) -> dict:
        return {
            "status": "error",
            "data": None,
            "error": {
                "code": "tool_not_implemented",
                "message": "Ticket lookup is not available yet."
            }
        }


if __name__ == "__main__":
    tools = MockAgentTools()

    result = tools.search_company_knowledge(
        query="What is the standard return window?",
        top_k=3
    )

    print("Status:", result["status"])

    for hit in result["data"]:
        print(hit["source_id"], hit["score"])
