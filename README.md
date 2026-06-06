# Multi-Agent Data & Document Analysis System

A state-of-the-art, agent-driven application that enables semantic interaction with PDF documents and CSV/Excel data using **LangGraph**, **Google Gemini**, and **Streamlit**.

## 🚀 Features

### **Multi-Agent Pipeline**
- **Router Agent**: Classifies query intent and urgency.
- **Data Analyst Agent**: Writes secure, sandboxed Pandas/Python code to extract tabular data.
- **Math Critic Agent**: Acts as a Senior Data Scientist to verify the Analyst’s mathematical logic.
- **Data Writer Agent**: Drafts a clean, strict, hallucination-free answer.
- **Retrieval Agent**: Fetches relevant semantic chunks from PDFs using FAISS.
- **Reviewer Agent**: Fact-checks draft vs. source chunks to ensure 100% accuracy.

### **Advanced Architecture**
- Dynamic LangGraph routing.
- Secure `exec()` sandbox with strict module whitelisting.
- In-memory FAISS indexing for real-time document interaction.
- Bulletproof Pandas integration (automatic DataFrame injection and sanitization).
- Beautiful Dark Mode Streamlit UI.

## 🛠️ Installation & Local Usage

1. **Clone the repo**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app locally:**
   ```bash
   python -m streamlit run app.py
   ```
4. **Enter your API Key:** Use the sidebar Settings in the UI to securely input your Google Gemini API Key.

## ☁️ Streamlit Cloud Deployment

This application is natively configured for **Streamlit Community Cloud**.

1. Fork or push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and create a new app.
3. Select this repository and set the main file path to `app.py`.
4. Click **Advanced settings...** and paste your API key in the Secrets box:
   ```toml
   GOOGLE_API_KEY = "AIzaSy..."
   ```
5. Click **Deploy!**

## 🏗️ Architecture Stack
- **Frontend**: Streamlit
- **Orchestration**: LangGraph
- **LLM**: Google Gemini 1.5 Flash (via `langchain-google-genai`)
- **Vector Store**: FAISS & HuggingFace Embeddings (`sentence-transformers`)
- **Data**: Pandas & Numpy
