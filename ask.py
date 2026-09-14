"""
ask.py — Interactive terminal Q&A for Document Copilot
-------------------------------------------------------
Usage:
    uv run python ask.py

Type your question and press Enter. Type 'exit' or 'quit' to stop.
"""

from app.services.chat_service import get_rag_chain

def main():
    print("\n=== Document Copilot — Interactive Q&A ===")
    print("  Initializing RAG pipeline (loading models)...\n")
    
    rag_chain = get_rag_chain()
    
    print("  ✅ Ready! Ask anything about the document.")
    print("  Type 'exit' to quit.\n")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n🔍 Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if not question:
            continue
        
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        
        print("\n⏳ Thinking...\n")
        answer = rag_chain.invoke(question)
        print(f"🤖 Answer:\n{answer}")
        print("\n" + "-" * 50)

if __name__ == "__main__":
    main()
