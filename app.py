import os
import streamlit as st
from dotenv import load_dotenv
from utils.rag_pipeline import run_rag_query

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(
    page_title="FinGuard AI",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for beauty
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            font-size: 3em;
            font-weight: 700;
        }
        .stButton>button {
            background-color: #27ae60;
            color: white;
            border-radius: 10px;
            padding: 0.5em 1.5em;
            font-size: 1.2em;
            font-weight: 600;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #219150;
        }
        .response-box {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #ddd;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>💰 FinGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>AI-powered fraud detection using Adaptive RAG and Guardrails for secure financial insights.</p>", unsafe_allow_html=True)

# User input
st.subheader("🔹 Enter a transaction description or financial query:")
query = st.text_area("")

# Analyze button
if st.button("🔍 Analyze Transaction"):
    if query.strip():
        with st.spinner("Analyzing... Please wait"):
            response = run_rag_query(query)

        st.markdown("### ✅ Analysis Result")
        st.markdown(f"<div class='response-box'>{response}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a transaction description or query first.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Built with ❤️ using Streamlit & LangChain</p>", unsafe_allow_html=True)
