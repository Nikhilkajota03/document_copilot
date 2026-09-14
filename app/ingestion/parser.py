import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_core.documents import Document

# 1. Load the API keys from your .env file
load_dotenv()

def parse_document(file_path: str) -> list[Document]:
    """
    Takes a PDF file path, sends it to LlamaParse, and returns 
    a list of LangChain Document objects.
    """
    print(f"Parsing document: {file_path}")
    
    # 2. Initialize the parser
    # We tell it to return the results in Markdown format so tables stay intact.
    parser = LlamaParse(
        result_type="markdown",
        verbose=True
    )
    
    # 3. Perform the actual parsing (this calls the LlamaCloud API)
    parsed_docs = parser.load_data(file_path)
    
    # 4. Convert LlamaIndex documents into LangChain documents
    langchain_docs = []
    for doc in parsed_docs:
        langchain_docs.append(
            Document(
                page_content=doc.text,
                metadata={"source": file_path}
            )
        )
        
    print(f"Successfully parsed into {len(langchain_docs)} markdown blocks!")
    print(f"this is the returned langchain doc{langchain_docs}")
    return langchain_docs

# --- Quick Test ---
if __name__ == "__main__":
    # Test it on your 10-K document
    pdf_path = "../../data/uploads/TSLA-Q4-2025-Update.pdf"
    
    # Ensure the file exists before parsing
    if os.path.exists(pdf_path):
        docs = parse_document(pdf_path)
        
        # Print the first 500 characters of the parsed output so you can see it!
        if docs:
            print("\n--- Snippet of Parsed Output ---")
            print(docs[0].page_content[:500])
    else:
        print(f"Could not find file at {pdf_path}")
