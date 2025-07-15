from langchain_community.document_loaders import PyPDFLoader

async def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = []
    async for page in loader.alazy_load():
        documents.append(page)
    return documents