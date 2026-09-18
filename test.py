"""
Standalone regression tests for the two fixes:
  1. Ollama options must use `num_predict`, never `max_tokens`.
  2. General Chat and Chat with Files must be genuinely independent:
     - mode='general' must NEVER call vector_store.similarity_search
     - mode='files' must ALWAYS call it, even for greeting-like text

No live Ollama/Chroma server is needed - everything is mocked/faked.
Run with: python3 test_fixes.py
"""

import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List


def _install_fake_ollama():
    """Install a fake `ollama` module so base.py's OllamaLLM can import it."""
    fake = types.ModuleType("ollama")
    fake.calls = []

    def fake_chat(model, messages, options=None, stream=False):
        fake.calls.append({"model": model, "messages": messages, "options": options or {}})
        return {"message": {"content": "fake response"}}

    def fake_list():
        Model = types.SimpleNamespace(model="llama3.1:8b")
        return types.SimpleNamespace(models=[Model])

    fake.chat = fake_chat
    fake.list = fake_list
    sys.modules["ollama"] = fake
    return fake


def test_num_predict_used_not_max_tokens():
    fake_ollama = _install_fake_ollama()
    sys.path.insert(0, "/home/claude/repo")
    from multimodal_rag.base import OllamaLLM

    llm = OllamaLLM({"models": {"llm_model": "llama3.1:8b", "ollama_host": "http://localhost:11434"}})
    llm.generate_response("hello", context="", max_tokens=2048, temperature=0.7)

    assert fake_ollama.calls, "ollama.chat was never called"
    options = fake_ollama.calls[-1]["options"]
    assert "num_predict" in options, f"num_predict missing from options: {options}"
    assert options["num_predict"] == 2048, f"num_predict not wired through correctly: {options}"
    assert "max_tokens" not in options, f"stale/ignored max_tokens key still being sent: {options}"
    print("PASS: generate_response sends num_predict (not max_tokens) to Ollama")


def test_seed_only_included_when_provided():
    fake_ollama = _install_fake_ollama()
    from multimodal_rag.base import OllamaLLM

    llm = OllamaLLM({"models": {"llm_model": "llama3.1:8b"}})

    llm.generate_response("hello", context="")
    assert "seed" not in fake_ollama.calls[-1]["options"], "seed should be absent when not requested"

    llm.generate_response("hello", context="", seed=42)
    assert fake_ollama.calls[-1]["options"].get("seed") == 42, "seed should be passed through when given"
    print("PASS: seed is only sent to Ollama when explicitly requested")


@dataclass
class FakeChunk:
    content: str = "some relevant document text that is definitely long enough"
    metadata: Dict[str, Any] = field(default_factory=lambda: {"source_file": "test.pdf"})
    source_file: str = "test.pdf"
    page_number: int = None


@dataclass
class FakeRetrievalResult:
    chunks: List[FakeChunk]


class FakeVectorStore:
    def __init__(self):
        self.similarity_search_calls = []

    def similarity_search(self, query, k=5):
        self.similarity_search_calls.append(query)
        return FakeRetrievalResult(chunks=[FakeChunk()])


class FakeLLM:
    def is_available(self):
        return True

    def generate_response(self, prompt, context="", **kwargs):
        return f"[fake answer for prompt of length {len(prompt)}]"


def _make_bare_simple_rag_system(vector_store, llm):
    """Build a SimpleRAGSystem instance without running its heavy __init__
    (which needs real processors/Chroma). We only need query()'s logic."""
    from multimodal_rag.system import SimpleRAGSystem

    system = SimpleRAGSystem.__new__(SimpleRAGSystem)
    system.config = {"retrieval": {"top_k": 5}}
    system.llm = llm
    system.vector_store = vector_store
    system.document_processor = None
    system.image_processor = None
    system.audio_processor = None
    return system


def test_general_mode_never_touches_vector_store():
    from multimodal_rag.base import QueryRequest

    vs = FakeVectorStore()
    system = _make_bare_simple_rag_system(vs, FakeLLM())

    # Even a question that clearly sounds like it's about an uploaded file:
    response = system.query(QueryRequest(query="what does my document say about Python?", mode="general"))

    assert len(vs.similarity_search_calls) == 0, "General Chat must never call similarity_search"
    assert response.sources == [], "General Chat must never return file sources"
    print("PASS: General Chat mode never touches the vector store")


def test_files_mode_always_searches_even_for_greetings():
    from multimodal_rag.base import QueryRequest

    vs = FakeVectorStore()
    system = _make_bare_simple_rag_system(vs, FakeLLM())

    # A pure greeting - the OLD heuristic would have treated this as
    # "conversational" and skipped retrieval entirely.
    response = system.query(QueryRequest(query="hi there", mode="files"))

    assert len(vs.similarity_search_calls) == 1, "Chat with Files must always search, even for greetings"
    assert len(response.sources) == 1, "Chat with Files should surface the retrieved chunk as a source"
    print("PASS: Chat with Files mode always searches, even for greeting-like text")


def test_files_mode_says_so_when_nothing_relevant_found():
    from multimodal_rag.base import QueryRequest

    class EmptyVectorStore(FakeVectorStore):
        def similarity_search(self, query, k=5):
            self.similarity_search_calls.append(query)
            return FakeRetrievalResult(chunks=[])

    vs = EmptyVectorStore()
    system = _make_bare_simple_rag_system(vs, FakeLLM())
    response = system.query(QueryRequest(query="anything", mode="files"))

    assert len(vs.similarity_search_calls) == 1
    assert response.sources == []
    print("PASS: Chat with Files still searches (and returns no sources) when the KB has nothing relevant")


if __name__ == "__main__":
    tests = [
        test_num_predict_used_not_max_tokens,
        test_seed_only_included_when_provided,
        test_general_mode_never_touches_vector_store,
        test_files_mode_always_searches_even_for_greetings,
        test_files_mode_says_so_when_nothing_relevant_found,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR: {t.__name__}: {e}")

    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)