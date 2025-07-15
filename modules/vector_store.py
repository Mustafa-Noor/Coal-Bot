import os
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PERSIST_DIR = "data/chroma_db"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_or_build_vector_store(docs):
    if os.path.exists(CHROMA_PERSIST_DIR):
        return Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=EMBEDDING_MODEL,
        )
    else:
        db = Chroma.from_documents(
            documents=docs,
            embedding=EMBEDDING_MODEL,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        db.persist()
        return db

def load_chroma_vector_store():
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        return Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=EMBEDDING_MODEL
        )
    else:
        # 📄 Load + embed on the fly (for Streamlit Cloud)
        pdf_path = "docs/coal_book.pdf"
        docs = load_pdf(pdf_path)                # load from PDF
        chunks = semantic_split(docs)            # split
        return get_or_build_vector_store(chunks) # create + return
