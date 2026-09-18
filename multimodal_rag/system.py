"""
Main multimodal RAG system implementation.
Simple traditional system using Ollama + basic processors.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Import new config schema
try:
    from config_schema import SmartRAGConfig, ConfigLoader, load_config
    USE_NEW_CONFIG = True
except ImportError:
    USE_NEW_CONFIG = False
    SmartRAGConfig = Any
    logging.warning("config_schema not found, using legacy config loading")


from .base import (
    QueryRequest,
    QueryResponse,
    DocumentChunk,
    ProcessingResult,
    OllamaLLM
)

logger = logging.getLogger(__name__)


class SimpleRAGSystem:
    """Simple RAG system using available components."""

    def __init__(self, config: Union[Dict[str, Any], SmartRAGConfig]):

        # Handle both new SmartRAGConfig and legacy dict
        if USE_NEW_CONFIG and isinstance(config, SmartRAGConfig):
            self.config = config.to_dict()
            self._typed_config = config
        else:
            self.config = config
            self._typed_config = None

        logger.info("Initializing Simple RAG System...")

        try:
            # Initialize LLM
            self.llm = OllamaLLM(self.config)
            logger.info("LLM initialized")

            # Initialize processors
            from .processors import (
                DocumentProcessorManager,
                ImageProcessorManager,
                AudioProcessorManager
            )

            self.document_processor = DocumentProcessorManager(self.config)
            self.image_processor = ImageProcessorManager(self.config)
            self.audio_processor = AudioProcessorManager(self.config)

            logger.info("Processors initialized")

            # Initialize vector store
            from .vector_stores.chroma_store import ChromaVectorStore

            self.vector_store = ChromaVectorStore(config)

            logger.info("Vector store initialized")
            logger.info("Simple RAG System initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Simple RAG System: {e}")

            self.llm = None
            self.document_processor = None
            self.image_processor = None
            self.audio_processor = None
            self.vector_store = None

    def is_available(self) -> bool:
        """Check if the system is available."""

        return (
            self.llm is not None
            and hasattr(self.llm, "is_available")
            and self.llm.is_available()
        )

    def ingest_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Ingest a file into the system."""

        if not self.is_available():
            return ProcessingResult(
                chunks=[],
                success=False,
                error_message="System not available"
            )

        try:
            file_path = Path(file_path)

            logger.info(f"Ingesting file: {file_path}")

            # Determine processor based on file type
            if (
                self.document_processor
                and self.document_processor.can_process(file_path)
            ):
                result = self.document_processor.process_file(file_path)

            elif (
                self.image_processor
                and self.image_processor.can_process(file_path)
            ):
                result = self.image_processor.extract_content(file_path)

            elif (
                self.audio_processor
                and self.audio_processor.can_process(file_path)
            ):
                result = self.audio_processor.extract_content(file_path)

            else:
                return ProcessingResult(
                    chunks=[],
                    success=False,
                    error_message=f"Unsupported file type: {file_path.suffix}"
                )

            # Store chunks in vector database
            if (
                result.success
                and result.chunks
                and self.vector_store
            ):
                self.vector_store.add_documents(result.chunks)

                logger.info(
                    f"Added {len(result.chunks)} chunks to vector store"
                )

            return result

        except Exception as e:

            logger.error(
                f"Error ingesting file {file_path}: {e}"
            )

            return ProcessingResult(
                chunks=[],
                success=False,
                error_message=str(e)
            )

    def query(
        self,
        query: Union[str, QueryRequest]
    ) -> QueryResponse:
        """Process a query and return response."""

        if not self.is_available():

            return QueryResponse(
                answer="System not available",
                sources=[],
                query=query if isinstance(query, str) else query.query,
                confidence_score=0.0
            )

        try:

            if isinstance(query, str):

                query_text = query
                query_obj = QueryRequest(query)

            else:

                query_text = query.query
                query_obj = query

            logger.info(
                f"Processing query: {query_text}"
            )

            mode = getattr(
                query_obj,
                "mode",
                "auto"
            ) or "auto"

            context = ""
            sources = []
            is_conversational = False

            # ============================================================
            # GENERAL CHAT
            # ============================================================

            if mode == "general":

                # General Chat never touches the vector store.
                logger.info(
                    "Mode=general: skipping file/vector search entirely"
                )

            # ============================================================
            # CHAT WITH FILES
            # ============================================================

            elif mode == "files":

                # Chat with Files always searches the knowledge base.
                if self.vector_store:

                    retrieval_result = (
                        self.vector_store.similarity_search(
                            query_text,
                            k=self.config.get(
                                "retrieval",
                                {}
                            ).get(
                                "top_k",
                                5
                            )
                        )
                    )

                    relevant_chunks = (
                        self._filter_relevant_context(
                            retrieval_result.chunks,
                            query_text
                        )
                    )

                    if relevant_chunks:

                        context = self._build_context(
                            relevant_chunks
                        )

                        sources = relevant_chunks

                    logger.info(
                        f"Mode=files: retrieved "
                        f"{len(relevant_chunks)} relevant chunks"
                    )

                else:

                    logger.info(
                        "Mode=files: no vector store available"
                    )

            # ============================================================
            # LEGACY / AUTO MODE
            # ============================================================

            else:

                is_conversational = (
                    self._is_conversational_query(
                        query_text
                    )
                )

                if (
                    not is_conversational
                    and self.vector_store
                ):

                    retrieval_result = (
                        self.vector_store.similarity_search(
                            query_text,
                            k=self.config.get(
                                "retrieval",
                                {}
                            ).get(
                                "top_k",
                                5
                            )
                        )
                    )

                    relevant_chunks = (
                        self._filter_relevant_context(
                            retrieval_result.chunks,
                            query_text
                        )
                    )

                    if relevant_chunks:

                        context = self._build_context(
                            relevant_chunks
                        )

                        sources = relevant_chunks

                    logger.info(
                        f"Retrieved "
                        f"{len(relevant_chunks)} relevant chunks "
                        f"for query"
                    )

                else:

                    logger.info(
                        f"Treating as conversational query: "
                        f"{query_text}"
                    )

            # ============================================================
            # GENERATE RESPONSE
            # ============================================================

            response = self._generate_contextual_response(
                query_text,
                context,
                is_conversational,
                mode=mode,
                **query_obj.generation_params
            )

            return QueryResponse(
                answer=response,
                sources=sources,
                query=query_text,
                confidence_score=(
                    0.8 if sources else 0.9
                )
            )

        except Exception as e:

            logger.error(
                f"Error processing query: {e}"
            )

            return QueryResponse(
                answer=f"Error processing query: {str(e)}",
                sources=[],
                query=(
                    query_text
                    if "query_text" in locals()
                    else str(query)
                ),
                confidence_score=0.0
            )

    # ================================================================
    # KNOWLEDGE BASE
    # ================================================================

    def has_indexed_content(self) -> bool:
        """Check if knowledge base has indexed content."""

        try:
            if self.vector_store is None:
                return False

            # ChromaVectorStore uses `collection`
            collection = getattr(
                self.vector_store,
                "collection",
                None
            )

            if collection is not None:
                return collection.count() > 0

            # Backward compatibility
            collection = getattr(
                self.vector_store,
                "_collection",
                None
            )

            if collection is not None:
                return collection.count() > 0

            return False

        except Exception as e:
            logger.warning(
                f"Error checking indexed content: {e}"
            )
            return False

    def get_indexed_files(self) -> List[Dict]:
        """Get list of indexed files with metadata."""

        try:
            files = []

            if self.vector_store is None:
                return files

            # ChromaVectorStore uses `collection`
            collection = getattr(
                self.vector_store,
                "collection",
                None
            )

            # Backward compatibility
            if collection is None:
                collection = getattr(
                    self.vector_store,
                    "_collection",
                    None
                )

            if collection is None:
                return files

            all_docs = collection.get()

            sources_dict = {}

            metadatas = (
                all_docs.get("metadatas", [])
                if all_docs
                else []
            )

            for metadata in metadatas:

                if (
                    isinstance(metadata, dict)
                    and "source_file" in metadata
                ):

                    source = metadata["source_file"]

                    if source not in sources_dict:
                        sources_dict[source] = 0

                    sources_dict[source] += 1

            for source, chunk_count in sources_dict.items():

                # Display only the actual filename
                source_name = (
                    str(source)
                    .replace("\\", "/")
                    .split("/")[-1]
                )

                file_ext = (
                    source_name.split(".")[-1].upper()
                    if "." in source_name
                    else "UNKNOWN"
                )

                files.append({
                    "name": source_name,
                    "chunk_count": chunk_count,
                    "type": file_ext
                })

            return sorted(
                files,
                key=lambda x: x["name"].lower()
            )

        except Exception as e:

            logger.error(
                f"Error getting indexed files: {e}"
            )

            return []

    def delete_file(self, filename: str) -> bool:
        """Delete all chunks for a specific source file."""

        try:

            if self.vector_store is None:
                return False

            collection = getattr(
                self.vector_store,
                "collection",
                None
            )

            if collection is None:
                collection = getattr(
                    self.vector_store,
                    "_collection",
                    None
                )

            if collection is None:
                return False

            # Support both the displayed filename and stored temp path
            all_docs = collection.get()

            metadatas = (
                all_docs.get("metadatas", [])
                if all_docs
                else []
            )

            ids_to_delete = []

            for index, metadata in enumerate(metadatas):

                if not isinstance(metadata, dict):
                    continue

                source = metadata.get("source_file", "")

                source_name = (
                    str(source)
                    .replace("\\", "/")
                    .split("/")[-1]
                )

                if (
                    source == filename
                    or source_name == filename
                ):

                    ids = all_docs.get("ids", [])

                    if index < len(ids):
                        ids_to_delete.append(ids[index])

            if ids_to_delete:
                collection.delete(
                    ids=ids_to_delete
                )

                logger.info(
                    f"Deleted {filename} from knowledge base"
                )

                return True

            logger.warning(
                f"No chunks found for {filename}"
            )

            return False

        except Exception as e:

            logger.error(
                f"Error deleting file {filename}: {e}"
            )

            return False

    # ================================================================
    # QUERY HELPERS
    # ================================================================

    def _is_conversational_query(
        self,
        query_text: str
    ) -> bool:
        """Determine if a query is conversational."""

        conversational_patterns = [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "what is your name",
            "who are you",
            "thanks",
            "thank you",
            "bye",
            "goodbye",
            "see you",
            "nice to meet you",
            "how do you do"
        ]

        query_lower = (
            query_text
            .lower()
            .strip()
        )

        for pattern in conversational_patterns:

            if (
                query_lower == pattern
                or query_lower.startswith(pattern)
            ):

                return True

        if (
            len(query_lower) <= 10
            and not any(
                c in query_lower
                for c in [
                    "?",
                    "what",
                    "how",
                    "when",
                    "where",
                    "why"
                ]
            )
        ):

            return True

        return False

    def _filter_relevant_context(
        self,
        chunks: List[DocumentChunk],
        query_text: str,
        min_score: float = 0.1
    ) -> List[DocumentChunk]:
        """Filter chunks to only include relevant ones."""

        if not chunks:
            return []

        relevant_chunks = []

        # Limit to top 3 most relevant chunks
        for chunk in chunks[:3]:

            # Skip extremely short chunks
            if len(chunk.content) > 50:
                relevant_chunks.append(chunk)

        return relevant_chunks

    def _build_context(
        self,
        chunks: List[DocumentChunk]
    ) -> str:
        """Build context string with source attribution."""

        context_parts = []

        for i, chunk in enumerate(chunks):

            source_file = (
                chunk.source_file
                or chunk.metadata.get(
                    "filename",
                    "Unknown Document"
                )
            )

            if (
                source_file
                and source_file != "Unknown Document"
            ):

                source_name = (
                    source_file
                    .split("/")[-1]
                    .split("\\")[-1]
                )

            else:

                source_name = (
                    chunk.metadata.get(
                        "filename",
                        "Unknown Document"
                    )
                )

            page_info = ""

            if chunk.page_number:

                page_info = (
                    f" (Page {chunk.page_number})"
                )

            elif "page_number" in chunk.metadata:

                page_info = (
                    f" (Page "
                    f"{chunk.metadata['page_number']})"
                )

            context_parts.append(
                f"[{i + 1}] From "
                f"'{source_name}'{page_info}:\n"
                f"{chunk.content}"
            )

        return "\n\n".join(context_parts)

    # ================================================================
    # RESPONSE GENERATION
    # ================================================================

    def _generate_contextual_response(
        self,
        query_text: str,
        context: str,
        is_conversational: bool,
        mode: str = "auto",
        **kwargs
    ) -> str:

        """Generate response with appropriate prompting."""

        if mode == "general":

            system_prompt = (
                "You are a helpful, knowledgeable AI assistant. "
                "Answer the user's question directly using your own "
                "general knowledge. Do not mention or reference any "
                "uploaded documents or files, even if some exist."
            )

            full_prompt = (
                f"{system_prompt}\n\n"
                f"User: {query_text}\n"
                f"Assistant:"
            )

        elif mode == "files":

            if context and context.strip():

                full_prompt = f"""
You are a helpful AI assistant that answers strictly from the user's uploaded files.

Use ONLY the provided context to answer.

Always mention the specific document name(s) where you found the information.

Do not add outside/general knowledge.

Context from documents:
{context}

Question:
{query_text}

Answer (include specific document names in your response):
"""

            else:

                full_prompt = f"""
You are a helpful AI assistant restricted to the user's uploaded files.

No relevant information was found in the uploaded documents for this question.

Clearly tell the user you couldn't find anything relevant in their files, and suggest they rephrase the question or upload a document that covers it.

Do NOT answer from general knowledge.

Question:
{query_text}

Answer:
"""

        elif is_conversational:

            system_prompt = (
                "You are a helpful AI assistant. "
                "Respond naturally and conversationally to greetings "
                "and casual interactions. Keep responses brief and friendly."
            )

            full_prompt = (
                f"{system_prompt}\n\n"
                f"User: {query_text}\n"
                f"Assistant:"
            )

        else:

            if context and context.strip():

                full_prompt = f"""
You are a helpful AI assistant.

Use the provided context to answer the user's question.

Always mention the specific document name(s) where you found the information.

If the context doesn't contain relevant information, say so and provide what general help you can.

Context from documents:
{context}

Question:
{query_text}

Answer (include specific document names in your response):
"""

            else:

                full_prompt = f"""
You are a helpful AI assistant.

The user asked a question but no relevant documents were found.

Provide a helpful general response.

Question:
{query_text}

Answer:
"""

        # Generate response using LLM
        response = self.llm.generate_response(
            prompt=full_prompt,
            context="",
            **kwargs
        )

        return response

    # ================================================================
    # SYSTEM STATUS
    # ================================================================

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""

        try:

            return {
                "system_type": "simple_traditional",

                "llm_available": self.is_available(),

                "processors_available": {
                    "documents": (
                        self.document_processor is not None
                    ),
                    "images": (
                        self.image_processor is not None
                    ),
                    "audio": (
                        self.audio_processor is not None
                    )
                },

                "vector_store_available": (
                    self.vector_store is not None
                ),

                "model_name": (
                    self.config
                    .get("models", {})
                    .get(
                        "llm_model",
                        "unknown"
                    )
                )
            }

        except Exception as e:

            logger.error(
                f"Error getting system status: {e}"
            )

            return {
                "system_type": "simple_traditional",
                "error": str(e),
                "llm_available": False
            }


# ====================================================================
# MAIN MULTIMODAL RAG SYSTEM
# ====================================================================

class MultimodalRAGSystem:
    """Main multimodal RAG system."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        **overrides
    ):

        # Load configuration
        if USE_NEW_CONFIG:

            try:

                typed_config = load_config(
                    config_path=config_path,
                    **overrides
                )

                config = typed_config.to_dict()

                logger.info(
                    "✅ Using validated configuration schema"
                )

            except Exception as e:

                logger.warning(
                    f"Failed to load with new config system: "
                    f"{e}, falling back to legacy"
                )

                config = self._load_config_legacy(
                    config_path,
                    config_dict
                )

        else:

            config = self._load_config_legacy(
                config_path,
                config_dict
            )

        logger.info(
            "Initializing multimodal RAG system..."
        )

        # Try enhanced system first
        try:

            from .enhanced_system import (
                EnhancedMultimodalRAGSystem
            )

            self._system = (
                EnhancedMultimodalRAGSystem(config)
            )

            if self._system.is_available():

                self.system_type = "traditional"

                logger.info(
                    "Enhanced Multimodal RAG System "
                    "initialized successfully"
                )

            else:

                logger.warning(
                    "Enhanced system initialized "
                    "but LLM not available"
                )

                self._system = None
                self.system_type = "none"

        except Exception as e:

            logger.error(
                f"Failed to initialize Enhanced system: {e}"
            )

            # Fallback to simple system
            try:

                self._system = SimpleRAGSystem(config)

                if self._system.is_available():

                    self.system_type = "traditional"

                    logger.info(
                        "Simple RAG System "
                        "initialized as fallback"
                    )

                else:

                    logger.error(
                        "Simple RAG System not available"
                    )

                    self._system = None
                    self.system_type = "none"

            except Exception as e2:

                logger.error(
                    f"Failed to initialize fallback system: {e2}"
                )

                self._system = None
                self.system_type = "none"

    # ================================================================
    # CONFIGURATION
    # ================================================================

    def _load_config_legacy(
        self,
        config_path: Optional[Union[str, Path]],
        config_dict: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:

        import yaml

        if config_dict:

            return config_dict

        elif config_path:

            return self._load_config(config_path)

        else:

            return self._get_default_config()

    def _load_config(
        self,
        config_path: Union[str, Path]
    ) -> Dict[str, Any]:

        import yaml

        try:

            with open(
                config_path,
                "r",
                encoding="utf-8"
            ) as file:

                config = yaml.safe_load(file)

            logger.info(
                f"Loaded configuration from {config_path}"
            )

            return config

        except Exception as e:

            logger.error(
                f"Failed to load config from "
                f"{config_path}: {str(e)}"
            )

            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:

        return {

            "system": {
                "name": "SmartRAG System",
                "offline_mode": True,
                "debug": False,
                "log_level": "INFO"
            },

            "models": {
                "llm_type": "ollama",
                "llm_model": "llama3.1:8b",
                "ollama_host": "http://localhost:11434",
                "embedding_model": "nomic-embed-text",
                "embedding_dimension": 768,
                "vision_model": "Salesforce/blip-image-captioning-base",
                "whisper_model": "base"
            },

            "vector_store": {
                "type": "chromadb",
                "persist_directory": "./vector_db",
                "collection_name": "multimodal_documents",
                "embedding_dimension": 768,
                "ollama_host": "http://localhost:11434"
            },

            "processing": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "max_image_size": [1024, 1024],
                "ocr_enabled": True,
                "batch_size": 32,
                "store_original_images": True,
                "image_preprocessing": "resize",
                "audio_sample_rate": 16000,
                "max_audio_duration": 300
            },

            "retrieval": {
                "top_k": 5,
                "similarity_threshold": 0.7,
                "rerank_enabled": False
            },

            "generation": {
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                "max_new_tokens": 1024,
                "seed": None
            }
        }

    # ================================================================
    # DELEGATION
    # ================================================================

    def ingest_file(
        self,
        file_path: Union[str, Path]
    ) -> ProcessingResult:

        if self._system:

            return self._system.ingest_file(
                file_path
            )

        logger.error(
            "No active system available"
        )

        return ProcessingResult(
            chunks=[],
            success=False,
            error_message="No active system"
        )

    def is_available(self) -> bool:

        return (
            self._system is not None
            and self._system.is_available()
        )

    def query(
        self,
        query: Union[str, QueryRequest]
    ) -> QueryResponse:

        if not self._system:

            logger.error(
                "No active system available"
            )

            return QueryResponse(
                answer=(
                    "No active system available. "
                    "Please check Ollama is running and "
                    "llama3.1:8b model is available."
                ),
                sources=[],
                query=str(query),
                confidence_score=0.0
            )

        return self._system.query(query)

    def get_system_stats(self) -> Dict[str, Any]:

        if (
            self._system
            and hasattr(
                self._system,
                "get_system_status"
            )
        ):

            stats = (
                self._system
                .get_system_status()
            )

            stats["wrapper_system_type"] = (
                self.system_type
            )

            return stats

        return {
            "wrapper_system_type": self.system_type,
            "active_system": None,
            "error": "No active system"
        }

    def has_indexed_content(self) -> bool:

        if (
            self._system
            and hasattr(
                self._system,
                "has_indexed_content"
            )
        ):

            return self._system.has_indexed_content()

        return False

    def get_indexed_files(self) -> List[Dict]:

        if (
            self._system
            and hasattr(
                self._system,
                "get_indexed_files"
            )
        ):

            return self._system.get_indexed_files()

        return []

    def delete_file(
        self,
        filename: str
    ) -> bool:

        if (
            self._system
            and hasattr(
                self._system,
                "delete_file"
            )
        ):

            return self._system.delete_file(
                filename
            )

        return False

    @property
    def vector_store(self):

        if (
            self._system
            and hasattr(
                self._system,
                "vector_store"
            )
        ):

            return self._system.vector_store

        return None

    def clear_knowledge_base(self) -> bool:
        """Delete every indexed chunk from the vector store."""

        try:

            vs = self.vector_store

            if vs is None:
                return False

            collection = getattr(
                vs,
                "collection",
                None
            )

            if collection is None:
                collection = getattr(
                    vs,
                    "_collection",
                    None
                )

            if collection is None:
                return False

            results = collection.get()

            ids = results.get("ids", [])

            if ids:
                collection.delete(ids=ids)

            logger.info(
                "Cleared knowledge base"
            )

            return True

        except Exception as e:

            logger.error(
                f"Error clearing knowledge base: {e}"
            )

            return False


# Backward compatibility alias
MultimodalRAG = MultimodalRAGSystem