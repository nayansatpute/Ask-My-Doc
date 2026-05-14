import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../retrieval'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../generation'))

from hybrid import hybrid_search
from reranker import rerank
from llm_client import generate_answer
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Golden dataset — questions you know the answers to from your document
golden_dataset = [
    {
        "question": "What is regularization in deep learning?",
        "ground_truth": "Regularization is a technique used in Deep Learning and Machine Learning to prevent overfitting."
    },
    {
        "question": "What is early stopping?",
        "ground_truth": "Early stopping is a technique that stops training when validation loss starts increasing to prevent overfitting."
    },
    {
        "question": "What is data augmentation?",
        "ground_truth": "Data augmentation is a technique to increase training data by applying transformations to existing data."
    },
]

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics import faithfulness, answer_relevancy

# Configure Ragas to use Groq + local embeddings
groq_llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
))
hf_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
))

faithfulness.llm = groq_llm
faithfulness.embeddings = hf_embeddings
answer_relevancy.llm = groq_llm
answer_relevancy.embeddings = hf_embeddings

def run_evaluation():
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in golden_dataset:
        query = item["question"]
        results = hybrid_search(query, n_results=5)
        reranked = rerank(query, results, top_n=3)
        answer = generate_answer(query, reranked)

        questions.append(query)
        answers.append(answer)
        contexts.append([r["text"] for r in reranked])
        ground_truths.append(item["ground_truth"])
        print(f"✅ Evaluated: {query}")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    print("\n📊 Ragas Evaluation Results:")
    print(results)
    return results

if __name__ == "__main__":
    run_evaluation()