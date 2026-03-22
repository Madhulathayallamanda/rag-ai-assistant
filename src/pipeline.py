from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.embeddings import load_embeddings
from src.vector_store import build_vector_store, query_vector_store
from src.rag_chain import load_llm, generate_response

def run_rag_pipeline(file_path, query):
    # Load document
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = load_embeddings()

    # Vector store
    vectorstore = build_vector_store(chunks, embeddings)

    # Retrieve
    docs = query_vector_store(vectorstore, query)

    # Generate
    llm = load_llm()
    answer = generate_response(llm, query, docs)

    return answer
