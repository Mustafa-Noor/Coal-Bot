import os
import asyncio
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from modules.doc_loader import load_pdf
from modules.text_splitter import semantic_split

FAISS_PERSIST_DIR = "data/faiss_index"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def save_faiss_index(db):
    os.makedirs(FAISS_PERSIST_DIR, exist_ok=True)
    db.save_local(FAISS_PERSIST_DIR)

def load_faiss_index():
    return FAISS.load_local(FAISS_PERSIST_DIR, EMBEDDING_MODEL)

def get_or_build_vector_store(docs):
    db = FAISS.from_documents(docs, EMBEDDING_MODEL)
    save_faiss_index(db)
    return db

def load_faiss_vector_store():
    if os.path.exists(FAISS_PERSIST_DIR) and os.listdir(FAISS_PERSIST_DIR):
        return load_faiss_index()
    else:
        pdf_path = "doc/coal_book.pdf"
        docs = asyncio.run(load_pdf(pdf_path))
        chunks = semantic_split(docs)
        return get_or_build_vector_store(chunks)
