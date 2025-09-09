import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

def run_rag_query(query: str):
    """Run a simple RAG pipeline using HuggingFace and Chroma."""

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings()

    # Load documents from local Chroma database
    # Make sure you have a folder called 'chroma_db' in your project
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    # Initialize HuggingFace LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=os.getenv("HF_API_TOKEN")
    )

    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    # Run the query
    result = qa_chain.run(query)
    return result
