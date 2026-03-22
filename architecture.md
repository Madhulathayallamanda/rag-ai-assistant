# System Architecture

This project follows a modular RAG pipeline:

User Query → Retriever → Vector DB → Context → LLM → Response

## Components:
- Loader: Parses documents
- Splitter: Breaks text into chunks
- Embeddings: Converts text to vectors
- Vector DB: Stores embeddings (FAISS)
- Retriever: Finds relevant context
- Generator: Produces final answer

## Scalability:
- Can integrate with APIs (FastAPI)
- Replace FAISS with Pinecone
- Add caching layer
