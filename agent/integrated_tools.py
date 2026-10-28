
from agent.hybrid_retrieval import HybridKnowledgeRetriever


class IntegratedAgentTools:
    """Combine hybrid knowledge search with authorized backend lookups."""

    def __init__(self, bound_tools):
        # The application must provide tools bound to a trusted identity.
        self._bound_tools = bound_tools
        self._retriever = HybridKnowledgeRetriever()

    def search_company_knowledge(
        self,
        query: str,
        top_k: int = 3
    ) -> dict:
        """Retrieve company policies using hybrid search."""

        try:
            results = self._retriever.search(
                query=query,
                top_k=top_k
            )

            return {
                "status": "success",
                "data": results,
                "error": None
            }

        except Exception:
            return {
                "status": "error",
                "data": None,
                "error": {
                    "code": "retrieval_failed",
                    "message": "Knowledge retrieval could not be completed."
                }
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
