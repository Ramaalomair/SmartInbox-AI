from agent.hybrid_retrieval import HybridKnowledgeRetriever
from agent.retrieval import KnowledgeRetriever


class IntegratedAgentTools:
    """Combine configurable knowledge search with authorized backend lookups."""

    def __init__(self, bound_tools, retrieval_method="hybrid"):
        # The application must provide tools bound to a trusted identity.
        self._bound_tools = bound_tools
        if retrieval_method == "tfidf":
            self._retriever = KnowledgeRetriever()
        elif retrieval_method == "hybrid":
            self._retriever = HybridKnowledgeRetriever()
        else:
            raise ValueError(f"Unsupported retrieval method: {retrieval_method}")

    def search_company_knowledge(self, query: str, top_k: int = 3) -> dict:
        """Retrieve company policies using the configured search method."""
        try:
            results = self._retriever.search(query=query, top_k=top_k)
            return {"status": "success", "data": results, "error": None}
        except Exception:
            return {
                "status": "error",
                "data": None,
                "error": {
                    "code": "retrieval_failed",
                    "message": "Knowledge retrieval could not be completed.",
                },
            }

    def lookup_order(self, order_id: str) -> dict:
        """Retrieve an order using the bound requester identity."""
        return self._bound_tools.lookup_order(order_id)

    def lookup_invoice(self, invoice_id: str) -> dict:
        """Retrieve an invoice using the bound requester identity."""
        return self._bound_tools.lookup_invoice(invoice_id)

    def lookup_ticket(self, ticket_id: str) -> dict:
        """Retrieve a ticket using the bound requester identity."""
        return self._bound_tools.lookup_ticket(ticket_id)
