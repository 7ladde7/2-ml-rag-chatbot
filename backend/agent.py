import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.agents import create_agent
from .utils import (
    get_doc_chunks_by_url,
    llm,
    embeddings,
    retrieve_bm25,
    reranker,
)

dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(dim)
vector_stores_by_doc_url = {}

def retrieve_context(query: str, doc_url: str):
    """Retrieves relevant contexts from a document"""

    chunks = get_doc_chunks_by_url(doc_url)
    bm25_docs = retrieve_bm25(query, chunks)

    # кэш
    if doc_url not in vector_stores_by_doc_url:
        vector_stores_by_doc_url[doc_url] = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vector_stores_by_doc_url[doc_url].add_documents(documents=chunks)

    vector_store = vector_stores_by_doc_url[doc_url]
    dense_docs = vector_store.similarity_search(query, k=4)

    # склеиваем результаты
    retrieved_docs = list(set(doc.page_content for doc in (bm25_docs + dense_docs)))
    retrieved_docs.sort(
        key=lambda doc: reranker.predict([(query, doc)])[0],
        reverse=True
    )

    return retrieved_docs[:4]

def check_relevance(query: str, contexts: list[str]):
    """Return 'yes' or 'no' if context is useful"""

    return llm.invoke(f"""
QUESTION: {query}

CONTEXT:
{contexts}

Is this relevant? Answer 'yes' or 'no'.
""").content

def rewrite_query(query: str):
    """Rewrite query for better retrieval"""

    return llm.invoke(f"""
Rewrite this query to improve retrieval (one variant):
{query}
""").content

def get_rag_agent():
    return create_agent(
        model=llm,
        tools=[retrieve_context, check_relevance, rewrite_query],
        system_prompt=f"""
You are a RAG agent.

You can:
- retrieve context
- check relevance
- rewrite queries

Strategy:
Step 1. Retrieve context
Step 2. Check relevance
Step 3. If not relevant, rewrite and retry
Step 4. If relevant, answer

Use Step 3 only once. If after retry no relevant context is found, say:
"I could not find relevant information in the document." and stop.

You decide when to use each tool.

Be concise.""")