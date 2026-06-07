# Multi-Agent Data & Document Analysis System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-00A86B?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Google%20Gemini-LLM-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-009FDA?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade, multi-agent AI system for intelligent document and data analysis.**  
Ask questions about PDFs, Word documents, CSVs, and Excel files — in natural language.

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Configuration](#-configuration) · [Deployment](#-deployment)

</div>

---

## Overview

This system routes every user query through a specialized pipeline of LLM agents built with **LangGraph**. Depending on what you ask, it either:

- **Searches** your uploaded documents semantically using FAISS vector embeddings
- **Computes** answers from tabular data by generating and executing sandboxed Pandas code
- **Reasons** through conceptual and hypothetical questions using general knowledge

All answers are fact-checked, source-cited, and hallucination-guarded before being shown to you.

---

## Features

### Multi-Agent Pipeline

| Agent | Role |
|-------|------|
| **Router** | Classifies every query: `data_computation`, `document_grounded`, `hybrid`, or `direct` reasoning |
| **Retrieval** | Semantic FAISS search across uploaded PDFs and Word documents |
| **Analysis** | Extracts only facts relevant to the core question from retrieved chunks |
| **Writer** | Drafts a direct, structured answer with per-line source citations |
| **Reviewer** | Fact-checks the draft against source evidence; rewrites if needed |
| **Data Analyst** | Generates sandboxed Pandas code to query CSV/Excel/extracted tables |
| **Math Critic** | Validates mathematical and logical correctness of the generated code |
| **Data Writer** | Formats computed results with inline citations and no hallucinations |
| **Data Critic** | Catches any fabricated numbers in the final data answer |
| **Conversational** | Handles reasoning, preferences, and hypothetical questions directly |

### Intelligent Query Understanding

- **Answer-mode classification**: Distinguishes between questions that need document evidence vs. general reasoning — financial keywords alone do not trigger a document lookup
- **Core question extraction**: The router restates *what* the user actually wants answered before any pipeline runs
- **Mandatory inline citations**: Every factual line from uploads ends with `(Source: filename.ext)` — no grouped footnotes

### Document Processing

| Format | Capabilities |
|--------|-------------|
| **PDF** | Text extraction, table extraction to CSV, image analysis via Gemini Vision |
| **Word (.docx)** | Full text + embedded table extraction |
| **CSV / Excel** | Direct Pandas ingestion with multi-sheet support |
| **Images** (PNG, JPG, WEBP) | Gemini Vision description |
| **TXT** | Raw text ingestion |

### Security & Reliability

- Sandboxed `exec()` with strict module whitelist — generated code cannot access `os`, `sys`, `subprocess`, etc.
- Automatic fallback: if table analysis fails, the system falls back to document search
- Hallucination guard loop: Data Critic can reject and re-trigger the Data Writer up to 2 times
- Session persistence: last 5 sessions per uploaded file set saved to disk

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    Router Agent                         │
│  Determines: intent · answer_mode · core_question       │
│              requires_uploads · urgency                 │
└──────────┬──────────────────┬──────────────────┬────────┘
           │                  │                  │
    data_computation   document_grounded     direct/general
           │                  │                  │
           ▼                  ▼                  ▼
   ┌───────────────┐  ┌───────────────┐  ┌────────────────┐
   │ Data Analyst  │  │   Retrieval   │  │ Conversational │
   │ (Pandas code) │  │  Agent (FAISS)│  │    Agent       │
   └───────┬───────┘  └───────┬───────┘  └────────────────┘
           │                  │
           ▼                  ▼
   ┌───────────────┐  ┌───────────────┐
   │  Math Critic  │  │   Analysis    │
   └───────┬───────┘  │    Agent      │
           │          └───────┬───────┘
           ▼                  ▼
   ┌───────────────┐  ┌───────────────┐
   │  Data Writer  │  │  Writer Agent │
   └───────┬───────┘  └───────┬───────┘
           │                  │
           ▼                  ▼
   ┌───────────────┐  ┌───────────────┐
   │  Data Critic  │  │   Reviewer    │
   │ (loop ≤ 2x)   │  │    Agent      │
   └───────┬───────┘  └───────┬───────┘
           │                  │
           └────────┬─────────┘
                    ▼
              Final Answer
         (cited · verified · on-point)
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| **UI** | Streamlit (dark mode, session management) |
| **Orchestration** | LangGraph (directed state graph) |
| **LLM** | Google Gemini 3 Flash (via `langchain-google-genai`) |
| **Embeddings** | `sentence-transformers` / `all-MiniLM-L6-v2` |
| **Vector Store** | FAISS (in-session, per-upload) |
| **Table Extraction** | pdfplumber, PyMuPDF |
| **Image Analysis** | Gemini Vision |
| **Data Processing** | Pandas, NumPy, openpyxl |

---

## Quick Start

### Prerequisites

- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/apikey)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Prabhdeep17/Multi-Agent-System-final.git
cd Multi-Agent-System-final

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_key_here

# 5. Run the app
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### First Use

1. Enter your **Gemini API Key** in the sidebar (or set it in `.env` — no UI entry needed)
2. Click **Upload** and select one or more files (PDF, CSV, Excel, Word, TXT, or images)
3. Click **Upload & Chat**
4. Ask questions in the chat box

---

## Configuration

Copy `.env.example` to `.env` and configure:

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Retrieval tuning
CHUNK_SIZE=1000          # Characters per text chunk
CHUNK_OVERLAP=200        # Overlap between chunks
TOP_K_RETRIEVAL=5        # Number of chunks retrieved per query
SIMILARITY_THRESHOLD=0.7 # Minimum cosine similarity

# Agent tuning
AGENT_TEMPERATURE=0.7
MAX_TOKENS=2048
```

> The sidebar API key field overrides `.env` at runtime for quick key switching.

---

## Project Structure

```
Multi-Agent-System-final/
│
├── app.py                  # Streamlit UI, session management, query dispatch
├── agents_graph.py         # All agents + LangGraph pipeline
├── vector_store.py         # FAISS indexing, document chunking, search
├── multimodal_processor.py # PDF/Word/image processing, table extraction
│
├── requirements.txt
├── .env.example
│
└── scratch/                # Development test scripts (not production)
    ├── test_hardcore.py
    ├── test_images.py
    ├── test_suite.py
    └── fix_api.py
```

---

## Example Queries

| Query type | Example |
|-----------|---------|
| **Document fact** | "What is VantaGroup's Q4 2025 revenue target?" |
| **Table computation** | "What was the total sales for the Asia-Pacific region?" |
| **Hybrid** | "Based on the report, which segment shows the highest growth risk?" |
| **Reasoning** | "Would you rather invest in a company with 30% revenue growth or 30% EBITDA growth?" |
| **Summarization** | "Summarize the key risks mentioned in the report" |
| **Cross-document** | "Compare the figures from the CSV with the targets in the PDF" |

---

## Deployment

### Streamlit Community Cloud

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New app**
3. Select your fork and set **Main file path** to `app.py`
4. Click **Advanced settings** → **Secrets** and add:

```toml
GOOGLE_API_KEY = "AIzaSy..."
```

5. Click **Deploy**

### Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t multi-agent-system .
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key multi-agent-system
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) & [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- [Google Gemini](https://deepmind.google/technologies/gemini/) for LLM and vision capabilities
- [Streamlit](https://streamlit.io/) for the UI framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [pdfplumber](https://github.com/jsvine/pdfplumber) & [PyMuPDF](https://github.com/pymupdf/PyMuPDF) for document processing

---

<div align="center">
Built with LangGraph · Gemini · Streamlit
</div>
