from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .parser import parse_document  # Import our parser from the other file

def split_markdown_documents(docs: list[Document]) -> list[Document]:
    """
    Takes a list of large LangChain Documents (like full pages) and splits them 
    intelligently using Markdown headers. It then ensures no chunk is too large.
    """
    
    print("Splitting documents based on Markdown headers...")
    
    # 1. Define which headers we want the splitter to track and turn into metadata
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    # Initialize the Markdown Splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # Keep the headers in the text so the LLM still sees them
    )
    
    # 2. Split by Headers
    # We combine all the text from the incoming documents first, 
    # then split it perfectly along the markdown headers.
    md_splits = []
    for doc in docs:
        # MarkdownHeaderTextSplitter expects raw text, not Document objects, so we pass page_content
        splits = markdown_splitter.split_text(doc.page_content)
        
        # Bring over the original metadata (like the source file path) 
        # and combine it with the new header metadata
        for split in splits:
            split.metadata.update(doc.metadata)
            md_splits.append(split)
            
    print(f"Created {len(md_splits)} chunks after Markdown splitting.")
    
    # 3. Secondary Splitting (Safety Net)
    # What if a section under a header is STILL 5,000 words long? 
    # We use a character splitter as a safety net to chunk it down further.
    chunk_size = 1000
    chunk_overlap = 200
    
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    # Split the markdown chunks
    final_splits = char_splitter.split_documents(md_splits)
    
    print(f"Created {len(final_splits)} final chunks after character splitting safety net.")
    
    return final_splits

# --- Quick Test ---
if __name__ == "__main__":
    
    pdf_path = "data/uploads/TSLA-Q4-2025-Update.pdf"
    
    print("--- STEP 1: PARSING ---")
    raw_docs = parse_document(pdf_path)
    
    print("\n--- STEP 2: SPLITTING ---")
    final_chunks = split_markdown_documents(raw_docs)
    
    if final_chunks:
        print("\n--- Example of a Split Chunk ---")
        print("METADATA:")
        print(final_chunks[5].metadata) # Let's look at chunk #5
        print("\nCONTENT:")
        print(final_chunks[5].page_content[:500])
