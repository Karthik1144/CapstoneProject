#!/usr/bin/env python3
"""
SmartRAG Chatbot - Streamlit UI with intelligent query routing, streaming output, and file management
"""

import streamlit as st
import logging
from datetime import datetime
from pathlib import Path
import tempfile
import os
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import SmartRAG components
try:
    from config_schema import load_config
    from multimodal_rag.system import MultimodalRAGSystem
    from multimodal_rag.router import QueryRouter, QueryMode, RouteType
except ImportError as e:
    logger.error(f"Import error: {e}")
    st.error("Failed to import SmartRAG components. Please check installation.")
    st.stop()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SmartRAG - Intelligent Document Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 SmartRAG - Intelligent Chat with Documents")
st.markdown("*Local-first multimodal AI for document intelligence*")

# ============================================================================
# CACHED RESOURCES (Critical for performance!)
# ============================================================================

@st.cache_resource
def load_rag_system():
    """Load RAG system (cached - prevents model reloading)"""
    logger.info("=" * 60)
    logger.info("🚀 Initializing RAG system (will be cached)")
    logger.info("=" * 60)
    try:
        config = load_config()
        system = MultimodalRAGSystem(config_dict=config.to_dict() if hasattr(config, 'to_dict') else config)
        
        if system.is_available():
            logger.info("✅ RAG system initialized and ready")
            return system
        else:
            logger.error("❌ RAG system initialized but LLM not available")
            st.error("❌ LLM (Ollama) is not available. Please ensure Ollama is running with llama3.1:8b model.")
            st.stop()
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        st.error(f"Failed to initialize RAG system: {e}")
        st.stop()

# Initialize system
try:
    rag_system = load_rag_system()
    config = load_config()
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    logger.info("Initialized chat history")

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# ============================================================================
# SIDEBAR - FILE MANAGEMENT & SETTINGS
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings & Knowledge Base")
    
    # Chat mode selector
    st.subheader("Chat Mode")
    mode_option = st.radio(
        "How should I respond?",
        options=["Auto", "General Chat", "Chat with Files"],
        horizontal=False,
        help="""
        **Auto**: Use smart routing - I'll choose between general knowledge and uploaded files
        **General Chat**: Only use my general knowledge, ignore uploaded files  
        **Chat with Files**: Only reference uploaded files, don't use general knowledge
        """
    )
    
    # Map UI option to QueryMode
    mode_map = {
        "Auto": QueryMode.AUTO,
        "General Chat": QueryMode.GENERAL,
        "Chat with Files": QueryMode.FILES
    }
    st.session_state.chat_mode = mode_map[mode_option]
    
    st.divider()
    
    # Knowledge base management
    st.subheader("📚 Knowledge Base")
    
    # Get indexed files
    indexed_files = []
    try:
        if hasattr(rag_system, '_system') and rag_system._system:
            system = rag_system._system
            if hasattr(system, 'vector_store'):
                # Try to get collection info
                try:
                    collection_info = system.vector_store._collection.peek()
                    if collection_info and hasattr(collection_info, 'metadatas'):
                        files_dict = {}
                        for doc in collection_info.documents:
                            # Extract source from document
                            if hasattr(collection_info, 'metadatas') and collection_info.metadatas:
                                for i, meta in enumerate(collection_info.metadatas):
                                    if meta.get('source_file'):
                                        source = meta['source_file']
                                        if source not in files_dict:
                                            files_dict[source] = {'chunks': 0}
                                        files_dict[source]['chunks'] += 1
                        indexed_files = [{'name': k, 'chunk_count': v['chunks']} for k, v in files_dict.items()]
                except:
                    indexed_files = []
    except Exception as e:
        logger.warning(f"Could not retrieve indexed files: {e}")
    
    if indexed_files:
        st.info(f"📄 {len(indexed_files)} file(s) indexed")
        
        # List files with delete option
        for file_info in indexed_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(
                    f"📄 {file_info['name']}\n"
                    f"({file_info.get('chunk_count', '?')} chunks)"
                )
            with col2:
                if st.button("🗑️", key=f"delete_{file_info['name']}", help="Delete this file"):
                    try:
                        # Delete from vector store
                        if hasattr(rag_system._system.vector_store, '_collection'):
                            rag_system._system.vector_store._collection.delete(
                                where={"source_file": file_info['name']}
                            )
                        st.success(f"✅ Deleted {file_info['name']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to delete: {e}")
        
        # Clear all option
        if st.button("⚠️ Clear All Files", use_container_width=True, type="secondary"):
            if st.checkbox("Confirm: Clear entire knowledge base?"):
                try:
                    if hasattr(rag_system._system.vector_store, '_collection'):
                        rag_system._system.vector_store._collection.delete(where={})
                    st.success("✅ Knowledge base cleared")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to clear knowledge base: {e}")
    else:
        st.info("📭 No files indexed yet. Upload documents to get started!")
    
    st.divider()
    
    # Debug mode toggle
    st.subheader("🔧 Debug")
    debug_mode = st.checkbox("Show routing decisions", value=st.session_state.debug_mode)
    st.session_state.debug_mode = debug_mode
    
    # Configuration info
    with st.expander("⚙️ Configuration Info"):
        try:
            routing_cfg = config.to_dict().get('query_routing', {}) if hasattr(config, 'to_dict') else config.get('query_routing', {})
            st.write(f"**Relevance Threshold**: {routing_cfg.get('relevance_threshold', 'N/A')}")
            st.write(f"**Top-K Retrieval**: {routing_cfg.get('retrieval_top_k', 'N/A')}")
            perf_cfg = config.to_dict().get('performance', {}) if hasattr(config, 'to_dict') else config.get('performance', {})
            st.write(f"**Streaming**: {'✅ Enabled' if perf_cfg.get('streaming_enabled', True) else '❌ Disabled'}")
        except:
            st.caption("Could not load configuration info")

# ============================================================================
# FILE UPLOAD
# ============================================================================

st.header("📤 Upload Documents")

col1, col2 = st.columns([3, 1])
with col1:
    st.caption("Supported: PDF, Word, images (JPG, PNG), audio (MP3, WAV)")

uploaded_files = st.file_uploader(
    "Drop files here or click to browse",
    type=["pdf", "docx", "doc", "txt", "md", "rtf", "png", "jpg", "jpeg", "gif", "bmp", "mp3", "wav", "m4a", "ogg"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if uploaded_files:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (idx) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.info(f"Processing {uploaded_file.name}...")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            # Ingest file
            try:
                result = rag_system.ingest_file(tmp_path)
                if result.success:
                    st.success(f"✅ {uploaded_file.name} processed ({len(result.chunks)} chunks)")
                    logger.info(f"Successfully ingested {uploaded_file.name} with {len(result.chunks)} chunks")
                else:
                    st.error(f"❌ {uploaded_file.name}: {result.error_message}")
                    logger.error(f"Failed to ingest {uploaded_file.name}: {result.error_message}")
            finally:
                # Cleanup temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
            # Update progress
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            logger.error(f"Error processing {uploaded_file.name}: {e}")
    
    progress_bar.empty()
    status_text.empty()

# ============================================================================
# CHAT INTERFACE
# ============================================================================

st.header("💬 Chat")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available (RAG responses)
        if message.get("sources"):
            with st.expander("📎 Sources", expanded=False):
                for source in message["sources"]:
                    # Format source info
                    source_name = source.metadata.get('source_file', 'Unknown') if hasattr(source, 'metadata') else str(source)
                    st.caption(f"**{source_name}**")

# Query input
user_input = st.chat_input("Ask your question...")

if user_input:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process query
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        sources_placeholder = st.empty()
        debug_placeholder = st.empty()
        
        try:
            # Route and get answer
            logger.info(f"Query: {user_input} | Mode: {st.session_state.chat_mode.value}")
            
            # Get routing decision - safely access vector_store
            route_decision = None
            try:
                # Try to access vector_store from the system
                if hasattr(rag_system._system, 'vector_store'):
                    vector_store = rag_system._system.vector_store
                elif hasattr(rag_system._system, '_system') and hasattr(rag_system._system._system, 'vector_store'):
                    vector_store = rag_system._system._system.vector_store
                else:
                    vector_store = None
                
                if vector_store is not None:
                    router = QueryRouter(vector_store, config.to_dict() if hasattr(config, 'to_dict') else config)
                    route_decision = router.route(user_input, st.session_state.chat_mode)
            except Exception as e:
                logger.warning(f"Router error: {e}")
                route_decision = None
            
            # Generate answer based on route
            if route_decision is None or route_decision.route == RouteType.GENERAL:
                logger.info("Using General Chat (no routing)")
                response_obj = rag_system.query(user_input)
                answer = response_obj.answer
                sources = []
            else:
                logger.info(f"Routing to RAG: {route_decision.reason}")
                response_obj = rag_system.query(user_input)
                answer = response_obj.answer
                sources = response_obj.sources if hasattr(response_obj, 'sources') else []

            # Display answer
            response_placeholder.markdown(answer)
            
            # Display sources if RAG
            if sources:
                with sources_placeholder.expander("📎 Sources", expanded=True):
                    for source in sources:
                        source_name = source.metadata.get('source_file', 'Document') if hasattr(source, 'metadata') else 'Document'
                        st.caption(f"📄 {source_name}")
            
            # Display routing decision (if debug mode)
            if st.session_state.debug_mode and route_decision:
                with debug_placeholder.expander("🔍 Routing Details", expanded=False):
                    st.code(
                        f"Route: {route_decision.route.value.upper()}\n"
                        f"Reason: {route_decision.reason}\n"
                        f"Confidence: {route_decision.confidence:.2f}\n"
                        f"Best Score: {route_decision.best_score}",
                        language="text"
                    )
            elif st.session_state.debug_mode:
                with debug_placeholder.expander("🔍 Routing Details", expanded=False):
                    st.info("Router not available - using fallback mode")

            # Save to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now(),
                "sources": None
            })
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now(),
                "sources": sources
            })
            
            logger.info("Response generated successfully")
        
        except Exception as e:
            logger.exception(f"Error processing query: {e}")
            error_msg = f"❌ Error: {str(e)}"
            response_placeholder.error(error_msg)
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": datetime.now(),
                "sources": None
            })