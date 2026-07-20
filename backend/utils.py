import os
import requests
from hashlib import md5
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openrouter import ChatOpenRouter
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatOpenRouter(model="openai/gpt-5.4-mini")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)

def get_doc_chunks_by_url(doc_url):
    os.makedirs("assets", exist_ok=True)

    # кэш
    cache_path = os.path.join(
        "assets",
        f"{md5(doc_url.encode('utf-8')).hexdigest()}.pdf"
    )

    if not os.path.exists(cache_path):
        response = requests.get(doc_url)

        if response.status_code == 200:
            with open(cache_path, "wb") as f:
                f.write(response.content)

    loader = PyMuPDFLoader(cache_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        add_start_index=True,
    )

    chunks = splitter.split_documents(docs)
    return chunks

def retrieve_bm25(query, chunks):
    bm25_docs = BM25Retriever.from_documents(
        chunks,
        k=4,
        bm25_variant="plus",
    ).invoke(query)

    return bm25_docs