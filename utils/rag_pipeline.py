import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from utils.guardrails_setup import validate_output

def run_rag_query(query: str) -> str:
    """
    Runs the RAG query using LangChain and applies guardrails validation.
    """
    try:
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load vector store
        persist_dir = "vector_store"
        if not os.path.exists(persist_dir):
            return "⚠️ Error: Vector store not found. Please build it before running the app."

        vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

        # Load LLM
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.2, "max_length": 512}
        )

        # Create RAG chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

        # Run query
        result = qa_chain.run(query)

        # Validate response with guardrails
        return validate_output(result)

    except Exception as e:
        return f"❌ An error occurred: {str(e)}"
