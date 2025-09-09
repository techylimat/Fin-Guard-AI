import streamlit as st
from dotenv import load_dotenv
from utils.rag_pipeline import run_rag_query

# Load environment variables
load_dotenv()

st.title("FinGuard AI - Simple RAG App")

st.write("Ask a question and get answers powered by your documents!")

# Input field
user_query = st.text_input("Enter your query:")

# When user submits
if st.button("Search"):
    if user_query.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Searching..."):
            response = run_rag_query(user_query)
            st.success("Answer:")
            st.write(response)
