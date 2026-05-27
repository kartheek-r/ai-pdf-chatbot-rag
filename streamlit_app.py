import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

# Load API Key
api_key = st.secrets["GOOGLE_API_KEY"]
# Streamlit Title
st.set_page_config(page_title="AI PDF Chatbot")
st.title("📄 AI PDF Chatbot using RAG")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    # Save uploaded PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF Uploaded Successfully!")

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store vectors
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    # User question
    query = st.text_input(
        "Ask a question from the PDF"
    )

    if query:

        # Similarity search
        docs = vector_store.similarity_search(query)

        # Create context
        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        # Final prompt
        final_prompt = f"""
Answer the question using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

        # Gemini Model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )

        # Generate response
        with st.spinner("Generating answer..."):

            response = llm.invoke(final_prompt)

        # Show answer
        st.subheader("Answer")

        st.write(response.content)
