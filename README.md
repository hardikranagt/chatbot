# 🛡️ Policy Assistant — Multi-Mode AI Chatbot

A full-stack AI chatbot that demonstrates three distinct GenAI architectures side-by-side, letting you compare **RAG**, **Hybrid (RAG + Web Search)**, and **Standalone LLM** approaches in a single interface.

Built with **Ollama** (local LLMs), **ChromaDB** (vector store), **FastAPI** (backend), and **React + Vite** (frontend).

---

## ✨ Features

| Mode | How it works |
|------|-------------|
| **📄 RAG** | Answers strictly from internal documents (Privacy Policy & Terms of Service) stored in a ChromaDB vector store. Refuses to answer off-topic questions. |
| **🔀 Hybrid** | Combines internal document retrieval with live Google web search (via Serper API). Prefers internal docs and clearly labels web-sourced information. |
| **🧠 Standalone LLM** | Direct conversation with llama3.1 — no retrieval, no guardrails. Pure frozen-model responses from training data. |

### Additional Highlights

- **Closed-domain safety** — RAG mode uses a similarity threshold gate + system prompt guardrails to prevent hallucination
- **Local-first** — All LLM inference runs locally via Ollama (no OpenAI API needed)
- **Modern chat UI** — Dark theme, chat bubbles, animated typing indicator, mode switcher tabs
- **Source attribution** — Shows which documents/pages/web results were used for each answer

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│                  Frontend                    │
│            React + Vite (port 5173)          │
│  ┌─────────┬──────────┬──────────────────┐   │
│  │   RAG   │  Hybrid  │  Standalone LLM  │   │
│  └────┬────┴────┬─────┴────────┬─────────┘   │
└───────┼─────────┼──────────────┼─────────────┘
        │         │              │
        ▼         ▼              ▼
┌─────────────────────────────────────────────┐
│              FastAPI Backend (port 8000)      │
│                                              │
│  POST /chat  { question, mode }              │
│                                              │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  │
│  │ RAG Mode │  │Hybrid Mode │  │Standalone│  │
│  │          │  │            │  │          │  │
│  │ ChromaDB │  │ ChromaDB + │  │  Direct  │  │
│  │ retrieval│  │ Serper API │  │  Ollama  │  │
│  └────┬─────┘  └─────┬──────┘  └────┬─────┘  │
│       └──────────────┼──────────────┘        │
│                      ▼                       │
│              Ollama (llama3.1)                │
│              localhost:11434                  │
└──────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
rag-chatbot/
├── backend/
│   ├── main.py              # FastAPI app with /chat endpoint (3 modes)
│   ├── ollama_client.py     # Ollama API wrapper (embed, RAG, hybrid, standalone)
│   ├── retriever.py         # ChromaDB retrieval with similarity threshold
│   ├── web_search.py        # Serper API integration for web search
│   ├── ingest.py            # PDF → chunks → ChromaDB ingestion pipeline
│   ├── generate_pdfs.py     # Extracts TSX content → PDF files
│   ├── requirements.txt     # Python dependencies
│   ├── data/                # Source PDFs (privacy policy, terms & conditions)
│   └── .env                 # SERPER_API_KEY (not committed)
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Chat UI with mode tab switcher
│   │   ├── index.css        # Dark theme, chat bubbles, animations
│   │   └── main.jsx         # React entry point
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- [Ollama](https://ollama.ai) installed and running
- Python 3.10+
- Node.js 18+

### 1. Pull the required Ollama models

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

Generate the source PDFs and build the vector index:

```bash
python generate_pdfs.py
python ingest.py
```

(Optional) For Hybrid mode, create a `.env` file:

```bash
echo "SERPER_API_KEY=your_key_here" > .env
```

Start the API server:

```bash
uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** and start chatting!

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Ollama + llama3.1 (local) |
| **Embeddings** | nomic-embed-text via Ollama |
| **Vector Store** | ChromaDB (persistent, cosine similarity) |
| **Backend** | FastAPI + Python |
| **Frontend** | React + Vite |
| **Web Search** | Serper API (Google Search) |
| **PDF Processing** | pypdf + fpdf2 |
| **Text Splitting** | LangChain RecursiveCharacterTextSplitter |

---

## 📄 License

This project is for educational and demonstration purposes.
