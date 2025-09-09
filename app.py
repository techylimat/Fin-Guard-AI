import streamlit as st
from utils.rag_pipeline import run_rag_query

st.set_page_config(page_title="FinGuard AI", page_icon="🤖", layout="wide")

# Header
st.title("🤖 FinGuard AI - Financial RAG Assistant")
st.subheader("Ask questions about financial documents using AI-powered retrieval")

# Query input
user_query = st.text_input("Enter your question:", placeholder="e.g., What are the latest financial trends?")

# Button to trigger query
if st.button("Search"):
    if not user_query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Searching and generating answer..."):
            response = run_rag_query(user_query)
            st.success("Answer:")
            st.write(response)

# Footer
st.markdown("---")
st.caption("FinGuard AI © 2025 - Powered by Streamlit and LangChain")
