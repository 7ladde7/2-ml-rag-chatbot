from langchain.agents import create_agent
from .utils import llm, get_doc_chunks_by_url, retrieve_bm25

def get_baseline_rag_agent():
    """Простой RAG для бейзлайна (BM25)"""

    def retrieve_context(query: str, doc_url: str):
        """Retrieves relevant contexts from a document"""

        chunks = get_doc_chunks_by_url(doc_url)
        docs = retrieve_bm25(query, chunks)
        return [doc.page_content for doc in docs]

    return create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt="You are a RAG chatbot. You MUST retrieve context through the tool before answering a question and ground your answers on it. Be concise.",
    )