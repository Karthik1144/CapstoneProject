"""
Query Router: maps the user's explicit chat mode to a routing decision.

There are exactly two independent modes:
- GENERAL: always answers from general knowledge, never touches uploaded files.
- FILES:   always searches uploaded files, never silently falls back to
           general knowledge.

There is no "Auto" mode and no similarity-threshold guessing - the user's
selection is the routing decision. This keeps behavior predictable and easy
to reason about (what you pick is what runs).
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryMode(str, Enum):
    """User-selected query mode."""
    GENERAL = "general"
    FILES = "files"


class RouteType(str, Enum):
    """Routing decision."""
    GENERAL = "general"
    RAG = "rag"


@dataclass
class RoutingDecision:
    """Result of routing decision (mostly useful for the debug panel)."""
    route: RouteType
    reason: str
    confidence: float = 1.0
    best_score: Optional[float] = None
    retrieved_chunks: Optional[int] = None


class QueryRouter:
    """Maps the selected QueryMode straight to a RouteType. No guessing."""

    def __init__(self, vector_store, config):
        """
        Args:
            vector_store: ChromaDB or FAISS vector store (used only to report
                whether any files are indexed, for the sidebar hint).
            config: Config dict (kept for interface compatibility).
        """
        self.vector_store = vector_store
        self.config = config

    def route(self, query: str, mode: QueryMode) -> RoutingDecision:
        """
        Args:
            query: User's question (unused - routing is decided purely by
                mode now, no preview search is performed here anymore, since
                that meant every query paid for two embedding + retrieval
                round-trips instead of one).
            mode: GENERAL or FILES - always honored exactly as selected.
        """
        if mode == QueryMode.GENERAL:
            return RoutingDecision(
                route=RouteType.GENERAL,
                reason="General Chat mode: answering from general knowledge only, files are ignored",
                confidence=1.0,
            )

        # mode == QueryMode.FILES - the actual retrieval happens once, inside
        # SimpleRAGSystem.query(). retrieved_chunks gets filled in by the
        # caller afterwards from the real response, not guessed here.
        return RoutingDecision(
            route=RouteType.RAG,
            reason="Chat with Files mode: searching your uploaded files only",
            confidence=1.0,
        )

    def has_indexed_content(self) -> bool:
        """Check if knowledge base has any indexed content."""
        try:
            if hasattr(self.vector_store, 'get_collection_count'):
                count = self.vector_store.get_collection_count()
            elif hasattr(self.vector_store, '_collection'):
                count = self.vector_store._collection.count()
            else:
                result = self.vector_store.similarity_search("", k=1)
                count = len(result.chunks) if result and hasattr(result, 'chunks') else 0
            return count > 0
        except Exception as e:
            logger.warning(f"Error checking indexed content: {e}")
            return False

    def get_routing_explanation(self, decision: RoutingDecision) -> str:
        """Get human-readable explanation of routing decision."""
        return f"**Route**: {decision.route.value.upper()} | **Reason**: {decision.reason}"