from langchain.vectorstores import FAISS

def build_vector_store(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)

def query_vector_store(vectorstore, query):
    return vectorstore.similarity_search(query, k=3)
