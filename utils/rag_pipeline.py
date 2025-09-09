from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os


def run_rag_query(query: str) -> str:
    """
    Run a clean, modern RAG pipeline using HuggingFace and Chroma.
    """
    # Load environment variables
    hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_api_key:
        return "Error: HuggingFace API token not found. Please set it in Streamlit Secrets."

    # Initialize embeddings (explicit model name to avoid warnings)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load vector database
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    # Initialize LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=hf_api_key,
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    # Create RAG pipeline
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    # Run query
    result = qa_chain.invoke({"query": query})
    return result.get("result", "No answer found.")
