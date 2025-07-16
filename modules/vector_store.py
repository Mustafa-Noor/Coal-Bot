import os
os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"

from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from modules.doc_loader import load_pdf
from modules.text_splitter import semantic_split
from chromadb.config import Settings
import asyncio


CHROMA_PERSIST_DIR = "data/chroma_db"

CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=CHROMA_PERSIST_DIR,
)

EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_or_build_vector_store(docs):
    db = Chroma.from_documents(
        documents=docs,
        embedding=EMBEDDING_MODEL,
        persist_directory=CHROMA_PERSIST_DIR,
        client_settings=CHROMA_SETTINGS  # ✅ Added this line
    )
    db.persist()
    return db

def load_chroma_vector_store():
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        return Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=EMBEDDING_MODEL,
            client_settings=CHROMA_SETTINGS  # ✅ Added this line
        )
    else:
        pdf_path = "doc/coal_book.pdf"
        docs = asyncio.run(load_pdf(pdf_path))
        chunks = semantic_split(docs)
        return get_or_build_vector_store(chunks)
