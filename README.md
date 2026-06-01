# Multi-Agent PDF Analysis System

A modular, agent-driven application that enables semantic interaction with PDF documents through a web interface using **AgentScope**, **Google Gemini**, and **Streamlit**.

## 🚀 Features

### **5 Specialized Agents**
- **RAG Agent**: Question answering with evidence retrieval
- **Summarization Agent**: Single & multi-document summarization with map-reduce
- **Comparator Agent**: Cross-document comparison and contrast
- **Timeline Agent**: Chronological event arrangement
- **Aggregator Agent**: Evidence consolidation and deduplication

### **Dynamic Orchestration**
- Automatic intent detection from natural language
- Dynamic agent routing and chaining
- Reasoning trace visualization
- Multi-agent workflows (e.g., RAG → Comparator → Response)

### **Advanced Features**
- Multi-document upload and indexing
- Citation tracking with metadata (doc name, page, chunk ID, similarity score)
- Interactive PDF viewer with page navigation
- Citation-to-PDF navigation
- Cross-document search and retrieval
- Persistent vector database (ChromaDB)


## 🛠️ Installation

### 1. Clone and Setup

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
MISTRAL_API_KEY=your_actual_mistral_api_key_here
```

**To get a Mistral API key:**
1. Go to [Mistral AI Console](https://console.mistral.ai/)
2. Create an account or sign in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

## 🎯 Usage

### Start the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Workflow

1. **Upload PDFs**: Use the sidebar to upload one or more PDF documents
2. **Process Documents**: Click "Process Documents" to index the PDFs
3. **Ask Questions**: Use natural language to interact with your documents

### Example Queries

#### **Simple Q&A (RAG Agent)**
```
What are the main findings in section 3?
Who are the authors of this paper?
```

#### **Summarization (Summarization Agent)**
```
Summarize this document
Give me a summary of all uploaded documents
```

#### **Comparison (RAG → Comparator Agent)**
```
Compare the methodologies in doc1.pdf and doc2.pdf
What are the differences between these two approaches?
```

#### **Timeline (RAG → Timeline Agent)**
```
Create a timeline of events from these documents
Arrange the findings chronologically
```

#### **Aggregation (RAG → Aggregator Agent)**
```
What evidence supports this claim across all documents?
Consolidate all findings about topic X
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────┐
│                Streamlit Frontend               │
│  (Multi-file upload, Chat, PDF Viewer)         │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│            Agent Orchestrator                   │
│       (Intent Detection & Routing)              │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐   ┌─────────────────────┐
│  RAG Agent   │   │ Summarization Agent │
└──────┬───────┘   └─────────────────────┘
       │
       ├─────► Comparator Agent
       ├─────► Timeline Agent
       └─────► Aggregator Agent
```

### Directory Structure

```
Smile/
├── agents/
│   ├── __init__.py
│   ├── rag_agent.py              # Question answering
│   ├── summarization_agent.py    # Document summarization
│   ├── comparator_agent.py       # Cross-doc comparison
│   ├── timeline_agent.py         # Chronological arrangement
│   └── aggregator_agent.py       # Evidence aggregation
├── components/
│   ├── __init__.py
│   └── pdf_viewer.py             # PDF display & navigation
├── config.py                     # Configuration management
├── pdf_processor.py              # PDF extraction & chunking
├── vector_store_manager.py       # ChromaDB integration
├── planner.py                    # Intent detection & orchestration
├── app.py                        # Main Streamlit application
├── requirements.txt              # Dependencies
├── .env.example                  # Environment template
└── README.md                     # This file
```











