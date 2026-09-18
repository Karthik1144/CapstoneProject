"""
Minimal script to run SmartRAG's MultimodalRAGSystem:
ingest a file and ask it a question.

Usage:
    python run_smartrag.py path/to/document.pdf "your question here"
"""
import sys
from multimodal_rag.system import MultimodalRAGSystem

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_smartrag.py <file_to_ingest> \"<question>\"")
        sys.exit(1)

    file_path = sys.argv[1]
    question = sys.argv[2]

    print("Initializing SmartRAG (loading config.yaml)...")
    system = MultimodalRAGSystem(config_path="config.yaml")

    print(f"Ingesting: {file_path}")
    result = system.ingest_file(file_path)
    print("Ingest result:", result)

    print(f"\nQuerying: {question}")
    response = system.query(question)
    print("\n=== ANSWER ===")
    print(response.answer if hasattr(response, "answer") else response)

if __name__ == "__main__":
    main()
