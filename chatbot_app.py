#!/usr/bin/env python3
"""
SmartRAG Chatbot - Streamlit UI with two independent chat modes,
reproducible-generation controls, and file management.
"""

import streamlit as st
import logging
from datetime import datetime
import tempfile
import os

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
    from multimodal_rag.base import QueryRequest
except ImportError as e:
    logger.error(f"Import error: {e}")
    st.error("Failed to import SmartRAG components. Please check installation.")
    st.stop()

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="SmartRAG - Intelligent Document Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; max-width: 1000px; }
        .smartrag-subtitle { color: var(--text-color); opacity: 0.65; margin-top: -0.6rem; }
        .mode-badge {
            display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
            font-size: 0.75rem; font-weight: 600; margin-bottom: 0.3rem;
        }
        .mode-badge.general { background: #e8f0fe; color: #1a56db; }
        .mode-badge.files { background: #eafaf0; color: #0e7d3f; }
        div[data-testid="stChatMessage"] { border-radius: 14px; }
        .stStatusWidget-empty-example {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 SmartRAG")
st.markdown('<p class="smartrag-subtitle">Local-first multimodal AI for document intelligence</p>', unsafe_allow_html=True)

# ============================================================================
# CACHED RESOURCES (Critical for performance!)
# ============================================================================

@st.cache_resource
def load_rag_system():
    """Load RAG system (cached - prevents model reloading)"""
    logger.info("Initializing RAG system (will be cached)")
    try:
        config = load_config()
        system = MultimodalRAGSystem(config_dict=config.to_dict() if hasattr(config, 'to_dict') else config)

        if system.is_available():
            logger.info("RAG system initialized and ready")
            return system
        else:
            logger.error("RAG system initialized but LLM not available")
            st.error("❌ LLM (Ollama) is not available. Please ensure Ollama is running with llama3.1:8b model.")
            st.stop()
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        st.error(f"Failed to initialize RAG system: {e}")
        st.stop()


def _get_vector_store(system):
    """Best-effort lookup of the underlying vector store, however deep it's nested."""
    for path in (
        lambda s: s._system.vector_store,
        lambda s: s._system._system.vector_store,
    ):
        try:
            vs = path(system)
            if vs is not None:
                return vs
        except AttributeError:
            continue
    return None


def get_indexed_files(system):
    """Return [{'name': ..., 'chunk_count': ...}] for everything currently indexed."""
    vector_store = _get_vector_store(system)
    if vector_store is None or not hasattr(vector_store, "_collection"):
        return []
    try:
        all_docs = vector_store._collection.get()
        counts = {}
        metadatas = all_docs.get("metadatas") if isinstance(all_docs, dict) else getattr(all_docs, "metadatas", None)
        for meta in (metadatas or []):
            source = (meta or {}).get("source_file")
            if source:
                counts[source] = counts.get(source, 0) + 1
        return [{"name": k, "chunk_count": v} for k, v in sorted(counts.items())]
    except Exception as e:
        logger.warning(f"Could not retrieve indexed files: {e}")
        return []


# Initialize system
try:
    rag_system = load_rag_system()
    config = load_config()
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else config
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

if "use_fixed_seed" not in st.session_state:
    st.session_state.use_fixed_seed = False

if "seed_value" not in st.session_state:
    st.session_state.seed_value = 42

if "temperature" not in st.session_state:
    st.session_state.temperature = float(config_dict.get("generation", {}).get("temperature", 0.7))

# ============================================================================
# SIDEBAR - MODE, KNOWLEDGE BASE, SETTINGS
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")

    # --- Chat mode: exactly two, fully independent options -----------------
    st.subheader("Chat Mode")
    mode_option = st.radio(
        "How should I answer?",
        options=["💬 General Chat", "🗂️ Chat with Files"],
        horizontal=False,
        label_visibility="collapsed",
    )

    mode_map = {
        "💬 General Chat": QueryMode.GENERAL,
        "🗂️ Chat with Files": QueryMode.FILES,
    }
    st.session_state.chat_mode = mode_map[mode_option]

    if st.session_state.chat_mode == QueryMode.GENERAL:
        st.caption("Answers only from general knowledge. Your uploaded files are never read in this mode.")
    else:
        st.caption("Answers only from your uploaded files. If nothing relevant is found, I'll say so rather than guessing.")

    st.divider()

    # --- Knowledge base management ------------------------------------------
    st.subheader("📚 Knowledge Base")
    indexed_files = get_indexed_files(rag_system)

    if indexed_files:
        st.caption(f"{len(indexed_files)} file(s) indexed")
        for file_info in indexed_files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"📄 **{file_info['name']}** · {file_info['chunk_count']} chunks")
            with col2:
                if st.button("🗑️", key=f"delete_{file_info['name']}", help="Delete this file"):
                    try:
                        rag_system.delete_file(file_info["name"])
                        st.success(f"Deleted {file_info['name']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete: {e}")

        with st.popover("⚠️ Clear all files", use_container_width=True):
            st.write("This removes every indexed file. This can't be undone.")
            if st.button("Yes, clear everything", type="primary"):
                try:
                    vector_store = _get_vector_store(rag_system)
                    if vector_store is not None and hasattr(vector_store, "_collection"):
                        vector_store._collection.delete(where={})
                    st.success("Knowledge base cleared")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear knowledge base: {e}")
    else:
        st.info("No files indexed yet — upload documents below to use Chat with Files.")

    st.divider()

    # --- Generation settings (incl. reproducible seed) ----------------------
    with st.expander("🎛️ Generation settings"):
        st.session_state.temperature = st.slider(
            "Creativity (temperature)", min_value=0.0, max_value=1.5,
            value=st.session_state.temperature, step=0.1,
            help="Lower = more focused/deterministic wording. Higher = more varied."
        )
        st.session_state.use_fixed_seed = st.checkbox(
            "Use a fixed seed (reproducible answers)",
            value=st.session_state.use_fixed_seed,
            help="When on, the same question + seed reliably produces the same answer. "
                 "When off, each answer can vary slightly, as usual."
        )
        if st.session_state.use_fixed_seed:
            st.session_state.seed_value = st.number_input(
                "Seed", min_value=0, max_value=2_147_483_647,
                value=st.session_state.seed_value, step=1,
            )

    # --- Debug ---------------------------------------------------------------
    with st.expander("🔧 Debug"):
        st.session_state.debug_mode = st.checkbox("Show routing details", value=st.session_state.debug_mode)

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ============================================================================
# FILE UPLOAD
# ============================================================================

with st.expander("📤 Upload documents", expanded=not indexed_files):
    st.caption("Supported: PDF, Word, TXT/MD, images (JPG, PNG), audio (MP3, WAV)")
    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=["pdf", "docx", "doc", "txt", "md", "rtf", "png", "jpg", "jpeg", "gif", "bmp", "mp3", "wav", "m4a", "ogg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.info(f"Processing {uploaded_file.name}...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                try:
                    result = rag_system.ingest_file(tmp_path)
                    if result.success:
                        st.success(f"✅ {uploaded_file.name} processed ({len(result.chunks)} chunks)")
                    else:
                        st.error(f"❌ {uploaded_file.name}: {result.error_message}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

                progress_bar.progress((idx + 1) / len(uploaded_files))
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                logger.error(f"Error processing {uploaded_file.name}: {e}")

        progress_bar.empty()
        status_text.empty()
        st.rerun()

# ============================================================================
# CHAT INTERFACE
# ============================================================================

st.header("💬 Chat")

if not st.session_state.chat_history:
    st.caption("Ask anything — pick a mode in the sidebar first.")

for message in st.session_state.chat_history:
    avatar = "🧑" if message["role"] == "user" else "📚"
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "assistant" and message.get("mode_label"):
            badge_class = "files" if message.get("mode") == "files" else "general"
            st.markdown(f'<span class="mode-badge {badge_class}">{message["mode_label"]}</span>', unsafe_allow_html=True)
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Sources", expanded=False):
                for source in message["sources"]:
                    source_name = source.metadata.get('source_file', 'Unknown') if hasattr(source, 'metadata') else str(source)
                    st.caption(f"📄 {source_name}")

user_input = st.chat_input("Ask your question...")

if user_input:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="📚"):
        badge_placeholder = st.empty()
        response_placeholder = st.empty()
        sources_placeholder = st.empty()
        debug_placeholder = st.empty()

        try:
            mode = st.session_state.chat_mode
            mode_label = "💬 General Chat" if mode == QueryMode.GENERAL else "🗂️ Chat with Files"
            badge_class = "general" if mode == QueryMode.GENERAL else "files"
            badge_placeholder.markdown(f'<span class="mode-badge {badge_class}">{mode_label}</span>', unsafe_allow_html=True)

            logger.info(f"Query: {user_input} | Mode: {mode.value}")

            # Routing decision is purely informational now (for the debug
            # panel) - the mode passed into the request below is what
            # actually and independently controls behavior.
            route_decision = None
            try:
                vector_store = _get_vector_store(rag_system)
                router = QueryRouter(vector_store, config_dict)
                route_decision = router.route(user_input, mode)
            except Exception as e:
                logger.warning(f"Router preview failed (non-fatal): {e}")

            generation_params = {"temperature": st.session_state.temperature}
            if st.session_state.use_fixed_seed:
                generation_params["seed"] = int(st.session_state.seed_value)

            request = QueryRequest(
                query=user_input,
                mode=mode.value,
                generation_params=generation_params,
            )
            response_obj = rag_system.query(request)
            answer = response_obj.answer
            sources = response_obj.sources if hasattr(response_obj, "sources") else []

            response_placeholder.markdown(answer)

            if sources:
                with sources_placeholder.expander("📎 Sources", expanded=True):
                    for source in sources:
                        source_name = source.metadata.get('source_file', 'Document') if hasattr(source, 'metadata') else 'Document'
                        st.caption(f"📄 {source_name}")

            if st.session_state.debug_mode:
                with debug_placeholder.expander("🔍 Routing details", expanded=False):
                    if route_decision:
                        lines = [
                            f"Mode: {mode.value.upper()}",
                            f"Route: {route_decision.route.value.upper()}",
                            f"Reason: {route_decision.reason}",
                        ]
                        if route_decision.retrieved_chunks is not None:
                            lines.append(f"Chunks found: {route_decision.retrieved_chunks}")
                        if st.session_state.use_fixed_seed:
                            lines.append(f"Seed: {st.session_state.seed_value}")
                        lines.append(f"Temperature: {st.session_state.temperature}")
                        st.code("\n".join(lines), language="text")
                    else:
                        st.info("Router preview unavailable this turn (query still ran normally).")

            st.session_state.chat_history.append({
                "role": "user", "content": user_input, "timestamp": datetime.now(), "sources": None,
            })
            st.session_state.chat_history.append({
                "role": "assistant", "content": answer, "timestamp": datetime.now(),
                "sources": sources, "mode": mode.value, "mode_label": mode_label,
            })

        except Exception as e:
            logger.exception(f"Error processing query: {e}")
            error_msg = f"❌ Error: {str(e)}"
            response_placeholder.error(error_msg)
            st.session_state.chat_history.append({
                "role": "assistant", "content": error_msg, "timestamp": datetime.now(), "sources": None,
            })