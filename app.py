from src.pipeline import run_rag_pipeline
from src.utils import format_output

if __name__ == "__main__":
    file_path = "sample.pdf"
    query = "What is this document about?"

    answer = run_rag_pipeline(file_path, query)
    print(format_output(answer))
