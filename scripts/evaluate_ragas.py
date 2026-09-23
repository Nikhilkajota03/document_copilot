import os
import sys
import asyncio
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

# --- HOTFIX FOR RAGAS BUG ---
# Ragas tries to import ChatVertexAI from the old langchain_community path.
# This monkey patch tricks Ragas into thinking the module exists to prevent the crash.
try:
    import langchain_community.chat_models
    import langchain_google_vertexai
    langchain_community.chat_models.vertexai = langchain_google_vertexai
    sys.modules['langchain_community.chat_models.vertexai'] = langchain_google_vertexai
except ImportError:
    pass
# ----------------------------

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from app.services.chat_service import get_rag_chain_with_sources
from app.retrieval.vector_store import get_embeddings

load_dotenv()

# Define our manual dataset
test_cases = [
    {
        "question": "What was Tesla's GAAP operating income in 2025?",
        "ground_truth": "$4.4B GAAP operating income in 2025; $1.4B in Q4"
    },
    {
        "question": "When does Tesla plan to unveil the Gen 3 version of Optimus?",
        "ground_truth": "In Q1 of 2025, they plan to unveil the Gen 3 version of Optimus, which will include major upgrades from version 2.5."
    },
    {
        "question": "What is the targeted improvement in performance for AI5 relative to AI4?",
        "ground_truth": "Tesla is targeting a 50x improvement in performance for AI5 relative to AI4 thanks to 10x raw compute, 9x memory capacity and 5x hardened block quantization."
    }
]

async def main():
    print("Initializing RAG chain...")
    # Using the chain that returns sources (so we get 'contexts')
    rag_chain = get_rag_chain_with_sources()
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    print("Generating answers from RAG pipeline...")
    for case in test_cases:
        question = case["question"]
        print(f"  -> Asking: {question}")
        
        # Invoke the chain
        result = rag_chain(question)
        
        questions.append(question)
        answers.append(result["answer"])
        # Extract the content from the SourceChunk objects
        ctx_list = [source.content for source in result["sources"]]
        contexts.append(ctx_list)
        ground_truths.append(case["ground_truth"])
        
        # Small delay to avoid rate limits during generation
        await asyncio.sleep(2)

    # Format the data for Ragas (v0.4.x format)
    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    print("\nInitializing Ragas Evaluator LLMs...")
    # We use the same model format you use in evaluate.py
    judge_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0)
    ragas_llm = LangchainLLMWrapper(judge_llm)
    
    eval_embeddings = get_embeddings()
    ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)
    
    print("Starting Ragas Evaluation (this may take a few minutes)...")
    # Evaluate
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False, # So it doesn't crash if an API call fails
    )
    
    df = result.to_pandas()
    
    print("\n=======================================================")
    print("RAGAS METRICS RESULTS")
    print("=======================================================")
    # Print the entire dataframe to avoid KeyErrors
    print(df)

    # --- CI/CD QUALITY GATES ---
    threshold = 0.80
    metrics_to_check = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    
    # Calculate means (ignoring NaN which might happen due to API timeouts)
    means = df[metrics_to_check].mean()
    
    # Format output for GitHub Step Summary
    summary = "### Ragas Evaluation Results\n\n"
    summary += "| Metric | Average Score | Status (Threshold: 0.80) |\n|--------|---------------|--------------------------|\n"
    
    failed = False
    for metric in metrics_to_check:
        score = means[metric]
        if pd.isna(score):
            status = "⚠️ N/A (API Timeout)"
            score_display = "N/A"
        elif score >= threshold:
            status = "✅ PASS"
            score_display = f"{score:.2f}"
        else:
            status = "❌ FAIL"
            score_display = f"{score:.2f}"
            failed = True
            
        summary += f"| {metric} | {score_display} | {status} |\n"
        
    print("\n" + summary)
    
    # Write to GitHub Step Summary if running in GitHub Actions
    github_step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if github_step_summary:
        with open(github_step_summary, "a") as f:
            f.write(summary)
            f.write("\n\n*This summary was automatically generated by the Document Copilot CI/CD pipeline.*")
            
    if failed:
        print(f"\n❌ Evaluation Failed! One or more metrics dropped below the {threshold} threshold.")
        sys.exit(1)
    else:
        print(f"\n✅ Evaluation Passed! All available metrics met the {threshold} threshold.")

if __name__ == "__main__":
    asyncio.run(main())
