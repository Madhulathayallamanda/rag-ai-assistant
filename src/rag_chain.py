from langchain.chat_models import ChatOpenAI

def load_llm():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def generate_response(llm, query, docs):
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    You are an AI assistant. Answer ONLY from the context below.
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """
    
    return llm.predict(prompt)
