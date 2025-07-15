from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def print_embeddings(chunks):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.embed_documents(texts)
    for i, emb in enumerate(embeddings):
        print(f"Embedding {i+1}: {emb[:10]}...")  # Print first 10 values for brevity

# --- Example usage ---
if __name__ == "__main__":
    chunks = [
        Document(page_content="Chunk about AI."),
        Document(page_content="Chunk about machine learning."),
    ]
    print_embeddings(chunks)