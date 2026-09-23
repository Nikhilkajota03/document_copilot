import os
from dotenv import load_dotenv

load_dotenv()

# Fix environment variables if they were set as LANGSMITH_* instead of LANGCHAIN_*
if "LANGSMITH_API_KEY" in os.environ and "LANGCHAIN_API_KEY" not in os.environ:
    os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
if "LANGSMITH_TRACING" in os.environ:
    os.environ["LANGCHAIN_TRACING_V2"] = os.environ["LANGSMITH_TRACING"]
if "LANGSMITH_ENDPOINT" in os.environ:
    os.environ["LANGCHAIN_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"]
if "LANGSMITH_PROJECT" in os.environ:
    os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]

from langsmith.evaluation import evaluate
from langchain_groq import ChatGroq

from app.services.chat_service import get_rag_chain

# 1. Define our Ground Truth Dataset manually based on TSLA-Q4-2025-Update.pdf
dataset = [
    {
        "inputs": {"query": "What was Tesla's GAAP operating income in 2025?"},
        "outputs": {"answer": "$4.4B GAAP operating income in 2025; $1.4B in Q4"}
    },
    {
        "inputs": {"query": "When does Tesla plan to unveil the Gen 3 version of Optimus?"},
        "outputs": {"answer": "In Q1 of 2025 (this year), they plan to unveil the Gen 3 version of Optimus, which will include major upgrades from version 2.5."}
    },
    {
        "inputs": {"query": "What is the targeted improvement in performance for AI5 relative to AI4?"},
        "outputs": {"answer": "Tesla is targeting a 50x improvement in performance for AI5 relative to AI4 thanks to 10x raw compute, 9x memory capacity and 5x hardened block quantization."}
    }
]

# 2. Define the Target Function (Our RAG Pipeline)
def predict_rag_answer(inputs: dict) -> dict:
    print(f"\n[Evaluating Question] {inputs['query']}")
    rag_chain = get_rag_chain()
    # The RAG chain returns a string answer
    result = rag_chain.invoke(inputs["query"])
    return {"answer": result}

# 3. Define the Evaluator (The "Judge" LLM)
def correctness_evaluator(run, example):
    # We use Groq (Llama 3.3 70B) as our judge
    judge_llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.0
    )
    
    prediction = run.outputs["answer"]
    reference = example.outputs["answer"]
    input_query = example.inputs["query"]
    
    prompt = f"""
    You are an expert evaluator. 
    Question: {input_query}
    Expected Answer: {reference}
    Actual Answer: {prediction}
    
    Is the actual answer correct and semantically equivalent to the expected answer? 
    Answer exactly 'YES' or 'NO' followed by a brief reason.
    """
    
    result = judge_llm.invoke(prompt).content.strip().upper()
    score = 1 if result.startswith("YES") else 0
    return {"key": "correctness", "score": score, "comment": result}

from langsmith import Client

def main():
    print("Starting LangSmith Evaluation...")
    
    # 3. Create the Dataset in LangSmith
    client = Client()
    dataset_name = "Tesla Q4 Manual Eval"
    
    # Check if dataset exists, if not create it
    if not client.has_dataset(dataset_name=dataset_name):
        print(f"Creating dataset '{dataset_name}'...")
        client.create_dataset(dataset_name=dataset_name, description="Manual QA pairs for Tesla Q4 report")
        for item in dataset:
            client.create_example(inputs=item["inputs"], outputs=item["outputs"], dataset_name=dataset_name)
    else:
        print(f"Using existing dataset '{dataset_name}'")

    # 4. Run the Evaluation!
    experiment_results = evaluate(
        predict_rag_answer,
        data=dataset_name,
        evaluators=[correctness_evaluator],
        experiment_prefix="tesla-q4-eval-",
        description="Initial manual dataset evaluation for Tesla Q4 report."
    )
    
    print("\n=========================================")
    print("Evaluation complete!")
    print("Check your LangSmith dashboard 'Datasets & Testing' tab for the full report and grades.")
    print("=========================================")

if __name__ == "__main__":
    main()
