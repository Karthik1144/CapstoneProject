"""
Query Router: Intelligently routes between General Chat and RAG pipelines.

Implements the decision logic from SmartRAG specification:
- Explicit mode overrides
- Empty knowledge base fast path
- Similarity-based routing with threshold
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryMode(str, Enum):
    """User-selected query mode"""
    AUTO = "auto"
    GENERAL = "general"
    FILES = "files"


class RouteType(str, Enum):
    """Routing decision"""
    GENERAL = "general"
    RAG = "rag"


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    route: RouteType
    reason: str
    confidence: float  # 0.0 to 1.0
    best_score: Optional[float] = None
    retrieved_chunks: Optional[int] = None


class QueryRouter:
    """Routes queries between General Chat and RAG pipelines"""
    
    def __init__(self, vector_store, config):
        """
        Args:
            vector_store: ChromaDB or FAISS vector store
            config: Config dict with query_routing settings
        """
        self.vector_store = vector_store
        self.config = config
        
        # Extract routing config with defaults
        routing_cfg = config.get('query_routing', {})
        self.relevance_threshold = routing_cfg.get('relevance_threshold', 0.5)
        self.retrieval_top_k = routing_cfg.get('retrieval_top_k', 5)
        
        logger.info(f"QueryRouter initialized with threshold={self.relevance_threshold}")
    
    def route(self, query: str, mode: QueryMode) -> RoutingDecision:
        """
        Route a query to either General Chat or RAG pipeline.
        
        Args:
            query: User's question
            mode: Auto / General / Files
        
        Returns:
            RoutingDecision with route type and reasoning
        """
        
        # Step 1: Explicit modes take precedence
        if mode == QueryMode.GENERAL:
            return RoutingDecision(
                route=RouteType.GENERAL,
                reason="User selected General Chat mode",
                confidence=1.0
            )
        
        # Step 2: Check if knowledge base is empty
        if not self.has_indexed_content():
            return RoutingDecision(
                route=RouteType.GENERAL,
                reason="No files indexed - using General Chat",
                confidence=1.0
            )
        
        # Step 3: Files mode always uses RAG
        if mode == QueryMode.FILES:
            return RoutingDecision(
                route=RouteType.RAG,
                reason="User selected Chat with Files mode",
                confidence=1.0
            )
        
        # Step 4: Auto mode - use relevance threshold
        if mode == QueryMode.AUTO:
            return self._auto_route(query)
        
        # Fallback (shouldn't reach here)
        return RoutingDecision(
            route=RouteType.GENERAL,
            reason="Unknown mode - defaulting to General Chat",
            confidence=0.5
        )
    
    def _auto_route(self, query: str) -> RoutingDecision:
        """
        Auto mode routing: perform similarity search and compare to threshold.
        """
        try:
            # Retrieve top-k chunks
            chunks = self.vector_store.similarity_search(
                query,
                k=self.retrieval_top_k
            )
            
            if not chunks or not chunks.chunks:
                return RoutingDecision(
                    route=RouteType.GENERAL,
                    reason="No similar documents found",
                    confidence=0.8,
                    best_score=0.0,
                    retrieved_chunks=0
                )
            
            # Get best score from chunks
            best_score = chunks.chunks[0].score if hasattr(chunks.chunks[0], 'score') else 0.7
            
            # Compare to threshold
            if best_score >= self.relevance_threshold:
                return RoutingDecision(
                    route=RouteType.RAG,
                    reason=f"Relevance score {best_score:.3f} meets threshold {self.relevance_threshold}",
                    confidence=min(best_score, 1.0),
                    best_score=best_score,
                    retrieved_chunks=len(chunks.chunks)
                )
            else:
                return RoutingDecision(
                    route=RouteType.GENERAL,
                    reason=f"Relevance score {best_score:.3f} below threshold {self.relevance_threshold}",
                    confidence=1.0 - best_score,
                    best_score=best_score,
                    retrieved_chunks=len(chunks.chunks)
                )
        
        except Exception as e:
            logger.warning(f"Auto-routing failed: {e} - falling back to General Chat")
            return RoutingDecision(
                route=RouteType.GENERAL,
                reason=f"Routing error: {str(e)} - using General Chat as fallback",
                confidence=0.3
            )
    
    def has_indexed_content(self) -> bool:
        """Check if knowledge base has any indexed content"""
        try:
            # Try to get collection count or check if any documents exist
            if hasattr(self.vector_store, 'get_collection_count'):
                count = self.vector_store.get_collection_count()
            elif hasattr(self.vector_store, '_collection'):
                count = self.vector_store._collection.count()
            else:
                # Fallback: try to search for empty query
                result = self.vector_store.similarity_search("", k=1)
                count = len(result.chunks) if result and hasattr(result, 'chunks') else 0
            
            return count > 0
        except Exception as e:
            logger.warning(f"Error checking indexed content: {e}")
            return False
    
    def get_routing_explanation(self, decision: RoutingDecision) -> str:
        """Get human-readable explanation of routing decision"""
        return f"**Route**: {decision.route.value.upper()} | **Reason**: {decision.reason}"