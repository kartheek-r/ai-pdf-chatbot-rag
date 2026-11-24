import os
import tempfile
import streamlit as st
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

MODEL = "gemini-3.5-flash"

st.set_page_config(page_title="AI PDF Chatbot")
st.title("📄 AI PDF Chatbot using RAG")


def get_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return os.getenv("GOOGLE_API_KEY")


api_key = get_api_key()
if not api_key:
    st.error("Set GOOGLE_API_KEY in Streamlit Secrets or as an environment variable.")
    st.stop()

client = genai.Client(api_key=api_key)


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner="Indexing PDF...")
def build_index(pdf_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        path = f.name
    try:
        docs = PyPDFLoader(path).load()
    finally:
        os.remove(path)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings())


uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    vector_store = build_index(uploaded_file.getvalue())
    st.success("PDF ready!")

    query = st.text_input("Ask a question from the PDF")

    if query:
        docs = vector_store.similarity_search(query, k=4)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I couldn't find that in the PDF."

Context:
{context}

Question: {query}
Answer:"""

        with st.spinner("Generating answer..."):
            try:
                response = client.models.generate_content(
                    model=MODEL, contents=prompt
                )
                st.subheader("Answer")
                st.write(response.text)

                pages = sorted({d.metadata.get("page", 0) + 1 for d in docs})
                st.caption("Sources: page " + ", ".join(map(str, pages)))
            except Exception as e:
                st.error(f"Gemini error: {e}")
