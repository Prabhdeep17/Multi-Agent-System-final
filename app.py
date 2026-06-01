# """
# Multi-Agent PDF Analysis System - Main Streamlit Application

# Features:
# - Multi-file PDF upload
# - Natural language interaction
# - Automatic agent routing
# - Citation highlighting
# - Document navigation
# """
# import streamlit as st
# from pathlib import Path
# import shutil
# from datetime import datetime
# from pdf_processor import PDFProcessor
# from vector_store_manager import VectorStoreManager
# from planner import AgentOrchestrator
# from components.pdf_viewer import PDFViewer
# from config import Config

# # Page configuration
# st.set_page_config(
#     page_title="Multi-Agent PDF Analysis",
#     page_icon="📚",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .stChatMessage {
#         background-color: #383a3e;
#         color: #ffffff;
#         border-radius: 10px;
#         padding: 10px;
#         margin-bottom: 10px;
#     }
#     .reasoning-trace {
#         background-color: #e8f4f8;
#         color: #000000;
#         border-left: 4px solid #1f77b4;
#         padding: 10px;
#         margin: 10px 0;
#         font-size: 0.9em;
#     }
# </style>
# """, unsafe_allow_html=True)


# def initialize_session_state():
#     """Initialize session state variables"""
#     if 'vector_store' not in st.session_state:
#         st.session_state.vector_store = None
#     if 'orchestrator' not in st.session_state:
#         st.session_state.orchestrator = None
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []
#     if 'uploaded_docs' not in st.session_state:
#         st.session_state.uploaded_docs = {}
#     if 'active_pdf' not in st.session_state:
#         st.session_state.active_pdf = None
#     if 'active_page' not in st.session_state:
#         st.session_state.active_page = 1
#     if 'processing_complete' not in st.session_state:
#         st.session_state.processing_complete = False


# def initialize_system():
#     """Initialize vector store and orchestrator"""
#     if st.session_state.vector_store is None:
#         with st.spinner("Initializing system..."):
#             st.session_state.vector_store = VectorStoreManager()
#             st.session_state.orchestrator = AgentOrchestrator(
#                 st.session_state.vector_store
#             )
            
#             # Auto-enable chat if DB already has documents
#             stats = st.session_state.vector_store.get_stats()
#             if stats.get('total_chunks', 0) > 0:
#                 st.session_state.processing_complete = True
                
#         st.success("✅ System initialized successfully!")


# def process_uploaded_pdfs(uploaded_files):
#     """
#     Process and index uploaded PDFs
    
#     Args:
#         uploaded_files: List of uploaded file objects
#     """
#     if not uploaded_files:
#         return
    
#     pdf_processor = PDFProcessor()
    
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     for idx, uploaded_file in enumerate(uploaded_files):
#         # Save uploaded file
#         pdf_path = Path(Config.PDF_STORAGE_DIR) / uploaded_file.name
        
#         # Skip if already processed
#         if uploaded_file.name in st.session_state.uploaded_docs:
#             continue
        
#         status_text.text(f"Processing {uploaded_file.name}...")
        
#         # Save file
#         with open(pdf_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         try:
#             # Process PDF
#             result = pdf_processor.process_pdf(str(pdf_path))
            
#             # Index chunks
#             index_result = st.session_state.vector_store.index_documents(
#                 result['chunks']
#             )
            
#             if index_result['status'] == 'success':
#                 # Store in session state
#                 st.session_state.uploaded_docs[uploaded_file.name] = {
#                     'path': str(pdf_path),
#                     'doc_name': result['doc_name'],
#                     'page_count': result['pdf_info']['page_count'],
#                     'chunks_indexed': index_result['chunks_indexed']
#                 }
                
#                 status_text.success(f"✅ {uploaded_file.name} indexed successfully!")
#             else:
#                 status_text.error(f"❌ Error indexing {uploaded_file.name}")
        
#         except Exception as e:
#             status_text.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
#         progress_bar.progress((idx + 1) / len(uploaded_files))
    
#     progress_bar.empty()
#     status_text.empty()
#     st.session_state.processing_complete = True


# def display_chat_history():
#     """Display chat history"""
#     for message in st.session_state.chat_history:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
            
#             # Display reasoning trace if available
#             if "reasoning_trace" in message and message["reasoning_trace"]:
#                 with st.expander("🔍 Reasoning Trace"):
#                     for step in message["reasoning_trace"]:
#                         st.markdown(f"- {step}")
            
#             # Display agent chain if available
#             if "agent_chain" in message and message["agent_chain"]:
#                 st.caption(f"**Agent Chain:** {' → '.join(message['agent_chain'])}")


# def handle_user_query(query: str):
#     """
#     Handle user query and generate response
    
#     Args:
#         query: User query string
#     """
#     # Add user message to history
#     st.session_state.chat_history.append({
#         "role": "user",
#         "content": query
#     })
    
#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(query)
    
#     # Generate response
#     with st.chat_message("assistant"):
#         with st.spinner("Processing..."):
#             # Execute orchestrator
#             result = st.session_state.orchestrator.execute(query)
            
#             # Display answer
#             st.markdown(result["answer"])
            
#             # Display reasoning trace
#             if result.get("reasoning_trace"):
#                 with st.expander("🔍 Reasoning Trace"):
#                     for step in result["reasoning_trace"]:
#                         st.markdown(f"- {step}")
            
#             # Display agent chain
#             if result.get("agent_chain"):
#                 st.caption(f"**Agent Chain:** {' → '.join(result['agent_chain'])}")
            
#             # Add to history
#             st.session_state.chat_history.append({
#                 "role": "assistant",
#                 "content": result["answer"],
#                 "reasoning_trace": result.get("reasoning_trace", []),
#                 "agent_chain": result.get("agent_chain", []),
#                 "evidence": result.get("evidence", [])
#             })
            
#             # Store evidence for citation viewer
#             if result.get("evidence"):
#                 st.session_state.last_evidence = result["evidence"]


# def main():
#     """Main application"""
#     # Initialize
#     initialize_session_state()
#     initialize_system()
    
    
    
#     # Sidebar
#     with st.sidebar:
#         st.header("📁 Document Manager")
        
#         # File uploader
#         uploaded_files = st.file_uploader(
#             "Upload PDF documents",
#             type=["pdf"],
#             accept_multiple_files=True,
#             key="pdf_uploader"
#         )
        
#         # Process button
#         if uploaded_files:
#             if st.button("Process Documents", type="primary"):
#                 initialize_system()
#                 process_uploaded_pdfs(uploaded_files)
        
    
#     # Main chat interface
#     if not st.session_state.processing_complete:
#         with st.chat_message("assistant"):
#             st.markdown("👋 Hello! I am your Multi-Agent AI Assistant. Please upload your PDF documents in the sidebar so I can analyze them for you.")
#     else:
#         # Display chat history
#         display_chat_history()
        
#         # Chat input is full width at the bottom
#         if query := st.chat_input("Ask a question about your documents..."):
#             handle_user_query(query)


# if __name__ == "__main__":
#     main()


































"""
Multi-Agent PDF Analysis System
"""
import streamlit as st
from pathlib import Path
from pdf_processor import PDFProcessor
from vector_store_manager import VectorStoreManager
from planner import AgentOrchestrator
from config import Config

st.set_page_config(
    page_title="DocMind — PDF Analysis",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Lora:ital@0;1&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* hide chrome */
header[data-testid="stHeader"]  { display: none; }
[data-testid="stSidebar"]        { display: none; }
.stDeployButton                  { display: none; }
footer                           { display: none; }
#MainMenu                        { display: none; }

/* page background */
.stApp { background: #f7f7f5; }

/* centre column */
.block-container {
    max-width: 720px !important;
    padding: 48px 24px 120px !important;
}

/* ── top wordmark ── */
.wordmark {
    font-family: 'Lora', serif;
    font-size: 18px;
    color: #111;
    letter-spacing: -0.2px;
    margin-bottom: 40px;
}
.wordmark span { color: #999; font-style: italic; }

/* ── upload card ── */
.upload-card {
    background: #ffffff;
    border: 1px solid #e5e5e3;
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    margin-bottom: 16px;
}
.upload-card .uc-title {
    font-size: 17px;
    font-weight: 500;
    color: #111;
    margin-bottom: 6px;
}
.upload-card .uc-sub {
    font-size: 13px;
    color: #999;
    margin-bottom: 24px;
}

/* file uploader — strip default chrome */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #f7f7f5 !important;
    border: 1.5px dashed #d1d1ce !important;
    border-radius: 8px !important;
    padding: 20px !important;
}

/* process button */
.stButton > button {
    background: #111 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 28px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.8 !important; }

/* ── indexed doc pills ── */
.doc-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: #fff;
    border: 1px solid #e5e5e3;
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #444;
}
.doc-row .dr-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── ready hint ── */
.ready-hint {
    text-align: center;
    padding: 56px 0 8px;
    font-size: 14px;
    color: #bbb;
}

/* ── chat bubbles ── */
[data-testid="stChatMessage"]        { background: transparent !important; border: none !important; padding: 0 !important; }
[data-testid="stChatMessageContent"] { background: transparent !important; }

.u-bubble {
    background: #111;
    color: #f0f0ee;
    border-radius: 18px 18px 4px 18px;
    padding: 11px 16px;
    font-size: 14px;
    line-height: 1.6;
    max-width: 540px;
    margin-left: auto;
    margin-bottom: 2px;
}
.a-bubble {
    background: #fff;
    border: 1px solid #e5e5e3;
    border-radius: 4px 18px 18px 18px;
    padding: 14px 18px;
    font-size: 14px;
    line-height: 1.7;
    color: #222;
    max-width: 620px;
    margin-bottom: 2px;
}
.agent-chain {
    font-size: 11px;
    color: #bbb;
    margin-top: 6px;
    padding-left: 2px;
}

/* reasoning expander */
[data-testid="stExpander"] {
    background: #fff !important;
    border: 1px solid #e5e5e3 !important;
    border-radius: 8px !important;
    margin-top: 6px !important;
}
[data-testid="stExpander"] summary { color: #aaa !important; font-size: 12px !important; }
[data-testid="stExpander"] p       { color: #777 !important; font-size: 13px !important; }

/* chat input */
[data-testid="stChatInput"] {
    background: #2a2b2f !important;
    border: 1px solid #555 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2) !important;
}
[data-testid="stChatInput"] textarea { font-size: 14px !important; color: #ffffff !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #888 !important; }

/* spinner */
.stSpinner > div { border-top-color: #111 !important; }

/* progress bar */
.stProgress > div > div { background: #111 !important; }
</style>
""", unsafe_allow_html=True)


# ── session state ─────────────────────────────────────────────────────────────

def init_state():
    for k, v in {
        "vector_store": None,
        "orchestrator": None,
        "chat_history": [],
        "uploaded_docs": {},
        "processing_complete": False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def init_system():
    if st.session_state.vector_store is None:
        with st.spinner("Initialising…"):
            st.session_state.vector_store = VectorStoreManager()
            st.session_state.orchestrator = AgentOrchestrator(
                st.session_state.vector_store
            )
            stats = st.session_state.vector_store.get_stats()
            if stats.get("total_chunks", 0) > 0:
                st.session_state.processing_complete = True


# ── PDF processing ────────────────────────────────────────────────────────────

def process_pdfs(uploaded_files):
    if not uploaded_files:
        return
    processor = PDFProcessor()
    bar = st.progress(0)
    msg = st.empty()
    for i, f in enumerate(uploaded_files):
        if f.name in st.session_state.uploaded_docs:
            continue
        msg.text(f"Processing {f.name}…")
        path = Path(Config.PDF_STORAGE_DIR) / f.name
        with open(path, "wb") as fp:
            fp.write(f.getbuffer())
        try:
            result = processor.process_pdf(str(path))
            idx = st.session_state.vector_store.index_documents(result["chunks"])
            if idx["status"] == "success":
                st.session_state.uploaded_docs[f.name] = {
                    "path": str(path),
                    "doc_name": result["doc_name"],
                    "page_count": result["pdf_info"]["page_count"],
                }
        except Exception as e:
            msg.error(f"Error with {f.name}: {e}")
        bar.progress((i + 1) / len(uploaded_files))
    bar.empty()
    msg.empty()
    st.session_state.processing_complete = True


# ── chat ──────────────────────────────────────────────────────────────────────

def render_history():
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            css = "u-bubble" if m["role"] == "user" else "a-bubble"
            st.markdown(f'<div class="{css}">{m["content"]}</div>', unsafe_allow_html=True)
            if m["role"] == "assistant":
                if m.get("agent_chain"):
                    st.markdown(
                        f'<div class="agent-chain">{" · ".join(m["agent_chain"])}</div>',
                        unsafe_allow_html=True,
                    )
                if m.get("reasoning_trace"):
                    with st.expander("Reasoning trace"):
                        for step in m["reasoning_trace"]:
                            st.markdown(f"— {step}")


def handle_query(query: str):
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(f'<div class="u-bubble">{query}</div>', unsafe_allow_html=True)
    with st.chat_message("assistant"):
        with st.spinner(""):
            result = st.session_state.orchestrator.execute(query)
        st.markdown(f'<div class="a-bubble">{result["answer"]}</div>', unsafe_allow_html=True)
        if result.get("agent_chain"):
            st.markdown(
                f'<div class="agent-chain">{" · ".join(result["agent_chain"])}</div>',
                unsafe_allow_html=True,
            )
        if result.get("reasoning_trace"):
            with st.expander("Reasoning trace"):
                for step in result["reasoning_trace"]:
                    st.markdown(f"— {step}")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "reasoning_trace": result.get("reasoning_trace", []),
            "agent_chain": result.get("agent_chain", []),
        })


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    init_state()
    init_system()

    # wordmark
    st.markdown('<div class="wordmark">Multi-Agent System</div>', unsafe_allow_html=True)

    # ── upload zone (always visible until docs are loaded) ──
    if not st.session_state.processing_complete:
        st.markdown("""
        <div class="upload-card">
            <div class="uc-title">Upload your documents</div>
            <div class="uc-sub">PDF files only — multiple files supported</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "pdf",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            if st.button("Analyse documents"):
                init_system()
                process_pdfs(uploaded_files)

    else:
        # ── indexed doc list (compact, no numbers) ──
        for name in st.session_state.uploaded_docs:
            st.markdown(
                f'<div class="doc-row">📄 <span class="dr-name">{name}</span></div>',
                unsafe_allow_html=True,
            )

        # allow adding more docs
        with st.expander("+ Add more documents"):
            more = st.file_uploader(
                "pdf2",
                type=["pdf"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if more and st.button("Process"):
                process_pdfs(more)

        # ── conversation ──
        if not st.session_state.chat_history:
            st.markdown('<div class="ready-hint">Ask anything about your documents</div>', unsafe_allow_html=True)
        else:
            render_history()

        if query := st.chat_input("Ask a question…"):
            handle_query(query)


if __name__ == "__main__":
    main()