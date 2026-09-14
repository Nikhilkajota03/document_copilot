from app.services.chat_service import get_rag_chain

def main():
    print("Initializing Full RAG Generation Pipeline...")
    rag_chain = get_rag_chain()
    
    query = "What is the exact name of the Registrant as specified in its charter?"
    print(f"\nUser Query: '{query}'\n")
    
    print("Generating Answer...\n")
    print("-" * 40)
    
    # Run the full RAG pipeline
    answer = rag_chain.invoke(query)
    
    print(" your answer--->",answer)
    print("-" * 40)

if __name__ == "__main__":
    main()
