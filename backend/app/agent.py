import faiss
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.agents import create_agent
from langchain.messages import AnyMessage, AIMessage, ToolCall, ToolMessage
from pydantic import BaseModel
from hashlib import md5
import json
import operator

from app.utils import (
    get_doc_chunks_by_url,
    llm,
    embeddings,
    retrieve_bm25,
    reranker,
)

dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(dim)
vector_stores_by_doc_url = {}

class ParsedRequest(BaseModel):
    query: str
    doc_url: str

class RelevanceChecker(BaseModel):
    relevant: bool

class RAGState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_input: str
    query: str
    doc_url: str
    contexts: list[str]
    relevant: bool
    retries: int

def parse_request_node(state: RAGState):
    parser = llm.with_structured_output(ParsedRequest)

    result = parser.invoke(
        f"""
Extract:

1. User question
2. Document URL

User request:

{state["messages"]}
"""
    )

    return {
        "query": result.query,
        "doc_url": result.doc_url,
        "retries": 0,
    }

def retrieve_node(state: RAGState):
    doc_url = state["doc_url"]
    query = state["query"]

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

    # делаем переранжирование
    retrieved_docs.sort(
        key=lambda doc: reranker.predict([(query, doc)])[0],
        reverse=True
    )

    contexts = retrieved_docs[:4]

    # отдаем топ-4 и добавим информацию в чат в виде вызова тулы для наглядности
    tool_call_id=md5((doc_url + query).encode('utf-8')).hexdigest()
    return {
        "contexts": contexts,
        "messages": [
            ToolCall(
              name="retrieve_context",
              args={"query": query, "doc_url": doc_url},
              id=tool_call_id,
            ),
            ToolMessage(
              name="retrieve_context",
              tool_call_id=tool_call_id,
              content=json.dumps(contexts),
            )
        ]
    }

def relevance_node(state: RAGState):
    checker = llm.with_structured_output(RelevanceChecker)

    response = checker.invoke(
        f"""
QUESTION:
{state["query"]}

CONTEXT:
{state["contexts"]}

Is the context relevant to the question?
"""
    )

    return {
        "relevant": response.relevant
    }

def rewrite_node(state: RAGState):
    response = llm.invoke(
        f"""
Rewrite this query to improve retrieval.

Return only the rewritten query.

Query:
{state["query"]}
"""
    )

    return {
        "query": response.content.strip(),
        "retries": state["retries"] + 1,
    }

def answer_node(state: RAGState):
    response = llm.invoke(
        f"""
Answer the user's question using ONLY the provided context.

QUESTION:
{state["query"]}

CONTEXT:
{state["contexts"]}
""")

    return {
        "messages": [
            AIMessage(response.content)
        ]
    }

def fail_node(state: RAGState):
    return {
        "messages": [
            AIMessage("I could not find relevant information in the document.")
        ]
    }

def route_after_relevance(state: RAGState):
    if state["relevant"]:
        return "answer"

    if state["retries"] >= 1:
        return "fail"

    return "rewrite"

def get_rag_agent():
    graph = StateGraph(RAGState)

    graph.add_node("parse_request", parse_request_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check_relevance", relevance_node)
    graph.add_node("rewrite_query", rewrite_node)
    graph.add_node("answer", answer_node)
    graph.add_node("fail", fail_node)

    graph.add_edge(START, "parse_request")
    graph.add_edge("parse_request", "retrieve")
    graph.add_edge("retrieve", "check_relevance")

    # делаем ли еще одну попытку ретрива
    graph.add_conditional_edges(
        "check_relevance",
        route_after_relevance,
        {
            "answer": "answer",
            "rewrite": "rewrite_query",
            "fail": "fail",
        },
    )

    graph.add_edge("rewrite_query", "retrieve")

    graph.add_edge("answer", END)
    graph.add_edge("fail", END)

    return graph.compile()