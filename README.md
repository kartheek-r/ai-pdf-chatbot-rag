# AI PDF Chatbot using RAG

An AI-powered PDF chatbot built using Streamlit, LangChain, FAISS, and Google Gemini API.

## Features
- Upload PDF files
- Ask questions from PDFs
- Context-aware AI answers
- Vector similarity search using FAISS

## Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- Gemini API

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Architecture

PDF
↓
Text Chunking
↓
Embeddings
↓
FAISS Vector Store
↓
Similarity Search
↓
Gemini API Response
