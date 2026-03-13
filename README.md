# 🧠 IntelliRAG — Context-Aware Document Intelligence

A professional **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit** and **LangChain**. Upload PDFs or text files and ask natural-language questions — IntelliRAG retrieves the most relevant passages and generates accurate, cited answers using **Llama 3.3 70B** via Groq.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 **PDF & TXT Support** | Upload and process multiple documents at once |
| 🔍 **Semantic Search** | FAISS vector database with HuggingFace embeddings for meaning-based retrieval |
| 🤖 **Llama 3.3 70B** | State-of-the-art LLM via Groq for fast, intelligent responses |
| 💬 **Chat Memory** | Retains conversation history for natural follow-up questions |
| 📎 **Source Citations** | Every answer shows the exact document passages it was derived from |
| 📥 **Export Chat** | Download the full conversation as a text file |
| 🎨 **Messenger-style UI** | Custom dark theme with right-aligned user bubbles and left-aligned bot responses |

---

## 🏗️ How It Works (RAG Pipeline)

```
Upload Document → Text Extraction → Chunking → Embedding → FAISS Vector Store
                                                                    ↓
                    User Question → Semantic Retrieval (top-3 chunks) → LLM generates answer
```

1. **Text Extraction** — `pypdf` extracts text from PDFs; plain `.txt` files are read directly.
2. **Chunking** — `RecursiveCharacterTextSplitter` splits text into 1000-character chunks with 200-character overlap to preserve context across boundaries.
3. **Embedding** — `all-MiniLM-L6-v2` (HuggingFace) converts each chunk into a 384-dimensional vector.
4. **Indexing** — FAISS stores the vectors for fast similarity search.
5. **Retrieval** — When the user asks a question, the top 3 most semantically similar chunks are retrieved.
6. **Generation** — The retrieved chunks + question are passed to Llama 3.3 70B, which generates a grounded answer.

---

## 📁 Project Structure

```
simple_rag_chatbot/
├── .streamlit/
│   ├── config.toml          # Streamlit theme (dark mode, colors, font)
│   └── secrets.toml         # API keys (gitignored — never committed)
├── documents/
│   └── sample.txt           # Sample document for quick testing
├── .env                     # Local API key fallback (gitignored)
├── .gitignore               # Files excluded from git
├── app.py                   # Main application — all logic lives here
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

### File-by-File Breakdown

| File | Purpose |
|---|---|
| **`app.py`** | The entire application: UI (Streamlit + custom CSS), document processing, RAG chain, chat logic, and export feature. Single-file architecture keeps the project simple and deployable. |
| **`.streamlit/config.toml`** | Configures the dark theme (`#0e1117` background, `#3b82f6` primary accent, sans-serif font). Streamlit reads this automatically on startup. |
| **`.streamlit/secrets.toml`** | Stores `GROQ_API_KEY` securely. Used by `st.secrets` in production (e.g., Streamlit Cloud). **Never committed to git.** |
| **`.env`** | Fallback for local development — `os.getenv("GROQ_API_KEY")` reads from here if `secrets.toml` is unavailable. |
| **`requirements.txt`** | All Python packages needed. Install with `pip install -r requirements.txt`. |
| **`documents/sample.txt`** | A sample text file to test the chatbot without uploading your own documents. |
| **`.gitignore`** | Prevents secrets (`.env`, `secrets.toml`), cache (`__pycache__`), and OS files from being tracked. |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- A free [Groq API key](https://console.groq.com/)

### 1. Clone & Install

```bash
git clone https://github.com/kushc/simple_chatbot_RAG.git
cd simple_chatbot_RAG
pip install -r requirements.txt
```

### 2. Set Your API Key

**Option A** — Create `.env` in the project root:
```
GROQ_API_KEY=gsk_your_key_here
```

**Option B** — Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

### 3. Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload a document in the sidebar, hit **Process Documents**, and start chatting!

---

## 🌐 Deploy to Streamlit Cloud

1. Push to GitHub (see steps below).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your repo → select `app.py` as the main file.
4. Add `GROQ_API_KEY` in **Settings → Secrets**.
5. Deploy!

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit + Custom CSS |
| LLM | Llama 3.3 70B (via Groq) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Framework | LangChain |
| PDF Parsing | pypdf |

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).
