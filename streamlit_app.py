import streamlit as st
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Streamlit UI
st.set_page_config(page_title="AI PDF Chatbot")
st.title("📄 AI PDF Chatbot using RAG")

# API Key
api_key = st.secrets["GOOGLE_API_KEY"]

# Configure Gemini
genai.configure(api_key=api_key)

# Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    # Save PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF Uploaded Successfully!")

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Question Input
    query = st.text_input(
        "Ask a question from the PDF"
    )

    if query:

        docs = vector_store.similarity_search(query)

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = f"""
Answer the question using the provided context only.

Context:
{context}

Question:
{query}

Answer:
"""

        with st.spinner("Generating answer..."):

            response = model.generate_content(final_prompt)

        st.subheader("Answer")

        st.write(response.text)
