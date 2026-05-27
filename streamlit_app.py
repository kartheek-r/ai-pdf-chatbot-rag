
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

# Streamlit Page
st.set_page_config(page_title="AI PDF Chatbot")

st.title("📄 AI PDF Chatbot using RAG")

# Read API Key from Streamlit Secrets
api_key = st.secrets["GOOGLE_API_KEY"]

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

    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    # User Query
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

        # Gemini Model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key
        )

        with st.spinner("Generating answer..."):

            response = llm.invoke(final_prompt)

        st.subheader("Answer")

        st.write(response.content)
