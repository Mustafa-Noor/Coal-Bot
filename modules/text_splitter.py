from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.schema import Document

load_dotenv()  # Load environment variables from .env file

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def semantic_split(documents):
    chunker = SemanticChunker(embedder)
    chunks = chunker.split_documents(documents)
    return chunks



# # --- Test code ---
# if __name__ == "__main__":
#     docs = [
#         Document(page_content="This is the first test document. It contains some text about AI."),
#         Document(page_content="This is the second test document. It discusses machine learning and deep learning."),
#     ]
#     result_chunks = semantic_split(docs)
#     for i, chunk in enumerate(result_chunks):
#         print(f"Chunk {i+1}: {chunk.page_content}\n")