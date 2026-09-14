import pandas as pd
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --------------------------------------------------------------
# 1. Prepare Document Data
# --------------------------------------------------------------
raw_data = [
    {
        "title": "Avatar",
        "year": 2009,
        "budget": 237000000,
        "box_office": 2923000000,
        "director": "James Cameron",
        "description": "A paraplegic Marine dispatched to Pandora on a unique mission becomes torn between following his orders and protecting his home."
    },
    {
        "title": "Avengers: Endgame",
        "year": 2019,
        "budget": 356000000,
        "box_office": 2799000000,
        "director": "Anthony Russo, Joe Russo",
        "description": "After the devastating events of Infinity War, the universe is in ruins. The remaining allies assemble once more."
    },
    {
        "title": "Titanic",
        "year": 1997,
        "budget": 200000000,
        "box_office": 2264000000,
        "director": "James Cameron",
        "description": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic."
    },
    {
        "title": "The Dark Knight",
        "year": 2008,
        "budget": 185000000,
        "box_office": 1006000000,
        "director": "Christopher Nolan",
        "description": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest tests."
    }
]

docs = []
for item in raw_data:
    content = f"Movie Title: {item['title']}. Directed by {item['director']}. Plot: {item['description']}"
    metadata = {
        "title": item["title"],
        "year": item["year"],
        "budget": item["budget"],
        "box_office": item["box_office"],
        "director": item["director"]
    }
    docs.append(Document(page_content=content, metadata=metadata))

# Index documents into Chroma Vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(docs, embeddings)

# --------------------------------------------------------------
# 2. Define AttributeInfo Schema
# --------------------------------------------------------------
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="The exact title of the movie",
        type="string",
    ),
    AttributeInfo(
        name="director",
        description="The name of the director(s) who made the movie",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The release year of the movie as an integer",
        type="integer",
    ),
    AttributeInfo(
        name="budget",
        description="The total financial budget of the movie in USD (e.g. 200000000 for $200M)",
        type="integer or float",
    ),
    AttributeInfo(
        name="box_office",
        description="The total revenue earned at the global box office in USD (e.g. 1000000000 for $1B)",
        type="integer or float",
    ),
]

document_content_description = "A brief summary and plot description of a movie"

# --------------------------------------------------------------
# 3. Instantiate SelfQueryRetriever
# --------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True  # Enables logging to see the query constructed by LLM
)