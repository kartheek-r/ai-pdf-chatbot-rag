import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load API key
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# Title
st.title("📄 AI PDF Chatbot using RAG")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF Uploaded Successfully!")

    loader = PyPDFLoader("temp.pdf")

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    query = st.text_input(
        "Ask a question from the PDF"
    )

    if query:

        docs = vector_store.similarity_search(query)

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3
        )

        prompt_template = """
        Answer the question using the provided context.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = load_qa_chain(
            llm,
            chain_type="stuff",
            prompt=prompt
        )

        with st.spinner("Generating answer..."):

            response = chain.run(
                input_documents=docs,
                question=query
            )

        st.subheader("Answer")

        st.write(response)
