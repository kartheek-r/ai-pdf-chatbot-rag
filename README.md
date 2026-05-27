## 🤖 AI PDF Chatbot using RAG (Retrieval-Augmented Generation)

> An AI-powered PDF chatbot built with **Streamlit, LangChain, FAISS, HuggingFace Embeddings, and Google Gemini API**. Upload any PDF and ask questions — get intelligent, context-based answers instantly.

---

## 📌 Project Overview

Reading through long PDFs manually is time-consuming. This chatbot lets you **upload any PDF and chat with it** — the app extracts content, understands it semantically, and answers your questions accurately using RAG (Retrieval-Augmented Generation).

Unlike basic keyword search, RAG retrieves the most **semantically relevant chunks** from your document and feeds them to Gemini to generate precise, context-aware answers.

---

## ✨ Features

- 📤 Upload any PDF file directly via the UI
- 🔍 Semantic search using FAISS vector store
- 🧠 Context-aware answers powered by **Google Gemini 1.5 Flash**
- ✂️ Smart text chunking with overlap for better context retention
- 💬 Interactive Q&A interface built with Streamlit
- ⚡ Fast similarity search using HuggingFace sentence embeddings

---

## 🏗️ How It Works

```
PDF Upload
    ↓
PyPDFLoader  →  Extract text from PDF
    ↓
RecursiveCharacterTextSplitter  →  Split into chunks (500 tokens, 50 overlap)
    ↓
HuggingFace Embeddings (all-MiniLM-L6-v2)  →  Convert chunks to vectors
    ↓
FAISS Vector Store  →  Store & index all vectors
    ↓
User Query  →  Similarity Search  →  Retrieve top relevant chunks
    ↓
Gemini 1.5 Flash + PromptTemplate  →  Generate answer from context
    ↓
Answer displayed in Streamlit UI
```

---

## 📂 Project Structure

```
ai-pdf-chatbot-rag/
│
├── streamlit_app.py       # Main application code
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore file
├── chatbot-ui.jpeg        # App UI screenshot
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| 🖥️ Frontend / UI | Streamlit |
| 📄 PDF Loader | LangChain PyPDFLoader |
| ✂️ Text Splitting | RecursiveCharacterTextSplitter |
| 🔢 Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| 🗃️ Vector Store | FAISS |
| 🤖 LLM | Google Gemini 1.5 Flash |
| 🔗 Orchestration | LangChain QA Chain |
| 🔐 Env Management | python-dotenv |

---

## 🖼️ App UI

![Chatbot UI](chatbot-ui.jpeg)

---

## ▶️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/kartheek-r/ai-pdf-chatbot-rag.git
cd ai-pdf-chatbot-rag
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Key
Create a `.env` file in the root folder:
```
GOOGLE_API_KEY=your_google_gemini_api_key_here
```
> Get your free API key at [Google AI Studio](https://aistudio.google.com/)

### 5. Run the App
```bash
streamlit run streamlit_app.py
```

---

## 📦 Requirements

```
streamlit
langchain
langchain-community
langchain-google-genai
faiss-cpu
sentence-transformers
pypdf2
python-dotenv
google-generativeai
```

---

## 💡 Use Cases

- 📚 Chat with research papers and academic documents
- 📋 Extract key information from lengthy reports
- 📜 Query legal or policy documents quickly
- 🧾 Summarize and Q&A on invoices or contracts
- 🎓 Study from textbooks by asking topic-specific questions

---

## 🔮 Future Scope

- Support for **multiple PDF uploads** simultaneously
- Add **chat history** to maintain conversation context
- Deploy to **Streamlit Cloud** for public access
- Add **source highlighting** to show which page the answer came from
- Support for **Word documents and web URLs**

---

## 🙌 Conclusion

This project demonstrates how to build a **production-ready RAG pipeline** from scratch using modern LLM tools. It combines semantic search, vector embeddings, and generative AI to create a genuinely useful document Q&A tool.

---

## 👨‍💻 Author

**Ryalampadu Kartheek**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kartheek-ryalampadu)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:kartheekryalampadu@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/kartheek-r)
