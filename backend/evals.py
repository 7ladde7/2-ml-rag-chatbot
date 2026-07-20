import random
random.seed(42)
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_recall,
    context_precision,
    faithfulness,
    answer_relevancy
)
from .baseline import get_baseline_rag_agent
from .agent import get_rag_agent
from .utils import (
    get_doc_chunks_by_url,
    llm,
    embeddings,
)

def run_ragas_eval(agent, doc_url, selected_chunks):
    data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    for chunk in selected_chunks:
        prompt = f"""
Convert the following text into a clear factual question:

TEXT:
{chunk.page_content}

QUESTION:
"""
        question = llm.invoke(prompt).content

        ground_truth_prompt = f"""
You are a strict evaluator. Answer the question accurately. Ground your answer based on the provided context. Be concise.

CONTEXT:
{chunk.page_content}

QUESTION:
{question}

ANSWER:
"""
        ground_truth = llm.invoke(ground_truth_prompt).content

        response = agent.invoke(
            {
                "messages": [{
                    "role": "user",
                    "content": f"""
{question}
Link: {doc_url}
"""
                }]
            },
        )

        messages = response["messages"]
        answer = messages[-1].content

        # для метрик мы должны вытащить выбранные агентом документы
        retrieved_contexts = []
        for msg in reversed(messages):
            if msg.type == "tool" and msg.name == "retrieve_context":
                parsed = json.loads(msg.content)
                if isinstance(parsed, list):
                    retrieved_contexts = parsed


        data["user_input"].append(question)
        data["response"].append(answer)
        data["retrieved_contexts"].append(retrieved_contexts)
        data["reference"].append(ground_truth)

    results = evaluate(
        dataset=Dataset.from_dict(data),
        metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings
    )

    print("\n----------\n")
    print(results)
    print("\n----------\n")

if __name__ == "__main__":
    # случайная научная статья для тестирования RAG
    doc_url = "https://arxiv.org/pdf/2307.05979"

    # сколько чанков будем искать через RAGAS для снятия метрик
    num_chunks = 25

    chunks = get_doc_chunks_by_url(doc_url)
    selected_chunks = random.sample(chunks, min(num_chunks, len(chunks)))

    run_ragas_eval(
        get_baseline_rag_agent(),
        doc_url,
        selected_chunks
    )

    run_ragas_eval(
        get_rag_agent(),
        doc_url,
        selected_chunks
    )
