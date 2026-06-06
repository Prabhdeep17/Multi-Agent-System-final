"""
MultiAgent System (Dark Theme) - Session Manager
"""
import streamlit as st
from pathlib import Path
import json
import uuid
import shutil
import os
from datetime import datetime

st.set_page_config(
    page_title="MultiAgent System",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Hide chrome */
.stDeployButton, footer, #MainMenu { display: none; }

/* Dark app background */
.stApp { background: #0f0f0f !important; }
.block-container { max-width: 780px !important; padding: 36px 24px 120px !important; }

/* All text in main area */
.stApp p, .stApp span, .stApp label, .stApp div,
.stApp h1, .stApp h2, .stApp h3, .stApp li { color: #e5e5e5 !important; }

/* Sidebar — slightly lighter dark */
[data-testid="stSidebar"] {
    background: #161616 !important;
    border-right: 1px solid #2a2a2a !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #cccccc !important; }

[data-testid="stSidebar"] .stButton > button {
    background: #2563eb !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    width: 100% !important;
    font-weight: 500 !important;
    padding: 10px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1d4ed8 !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
}
[data-testid="stChatMessage"] p { color: #e5e5e5 !important; }

/* Chat input */
[data-testid="stChatInput"] {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: #e5e5e5 !important;
    font-size: 14px !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #666 !important; }

/* Info / success / error boxes */
[data-testid="stAlert"] {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
}
[data-testid="stAlert"] p { color: #e5e5e5 !important; }

/* Status widget */
[data-testid="stStatusWidget"] {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
}
[data-testid="stStatusWidget"] p,
[data-testid="stStatusWidget"] span { color: #ccc !important; }

/* Expanders */
[data-testid="stExpander"] {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary span { color: #ccc !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1a1a1a !important;
    border: 1px dashed #333 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span { color: #aaa !important; }

/* Divider */
hr { border-color: #2a2a2a !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f0f0f; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 3px; }

/* Meta tags */
.meta-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.tag {
    border-radius: 20px; padding: 3px 11px;
    font-size: 11px; font-weight: 500; border: 1px solid;
    font-family: 'DM Sans', sans-serif;
}
.tag-blue   { background: #1e3a5f; color: #93c5fd; border-color: #2563eb; }
.tag-green  { background: #14291e; color: #6ee7b7; border-color: #16a34a; }
.tag-yellow { background: #2d2010; color: #fcd34d; border-color: #d97706; }
.tag-red    { background: #2d1010; color: #fca5a5; border-color: #dc2626; }
.tag-gray   { background: #1f1f1f; color: #888; border-color: #333; }

.page-title {
    font-size: 26px;
    font-weight: 600;
    color: #f0f0f0 !important;
    margin-bottom: 2px;
}
.page-sub {
    font-size: 13px;
    color: #666 !important;
    margin-bottom: 28px;
}
/* Beautiful Shimmer Loading */
.shimmer-text {
    font-size: 15px;
    font-weight: 500;
    background: linear-gradient(90deg, #555 0%, #fff 50%, #555 100%);
    background-size: 200% auto;
    color: transparent;
    -webkit-background-clip: text;
    animation: shimmer 1.5s linear infinite;
}
@keyframes shimmer {
    to { background-position: 200% center; }
}

/* Sidebar Active Button Outline */
[data-testid="stSidebar"] button[kind="primary"] {
    border: 2px solid #6c9fff !important;
    box-shadow: 0 0 10px rgba(108, 159, 255, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session Management (Disk) ─────────────────────────────────────────────────
SESSIONS_DIR = Path("sessions")
SESSION_DOCS_DIR = Path("session_docs")
SESSION_FAISS_DIR = Path("session_faiss")

for d in [SESSIONS_DIR, SESSION_DOCS_DIR, SESSION_FAISS_DIR]:
    d.mkdir(exist_ok=True)

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False
if "loaded_pdfs" not in st.session_state:
    st.session_state.loaded_pdfs = []

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "vector_store": None,
    "langgraph_app": None,
    "uploader_key": 0
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────

def save_session_meta(session_id, meta):
    with open(SESSIONS_DIR / f"{session_id}.json", "w") as f:
        json.dump(meta, f)

def load_all_sessions():
    sessions = []
    for f in SESSIONS_DIR.glob("*.json"):
        with open(f, "r") as file:
            sessions.append(json.load(file))
    # Sort by created_at descending (newest first)
    return sorted(sessions, key=lambda x: x.get("created_at", ""), reverse=True)

def enforce_session_limit():
    sessions = load_all_sessions()
    if len(sessions) > 5:
        # Delete the oldest sessions beyond 5
        for s in sessions[5:]:
            sid = s["id"]
            (SESSIONS_DIR / f"{sid}.json").unlink(missing_ok=True)
            shutil.rmtree(SESSION_DOCS_DIR / sid, ignore_errors=True)
            shutil.rmtree(SESSION_FAISS_DIR / sid, ignore_errors=True)

def update_session_history():
    if st.session_state.active_session_id:
        meta_file = SESSIONS_DIR / f"{st.session_state.active_session_id}.json"
        if meta_file.exists():
            with open(meta_file, "r") as f:
                data = json.load(f)
            data["history"] = st.session_state.chat_history
            save_session_meta(st.session_state.active_session_id, data)

def delete_session(session_id):
    """Delete a session's metadata, documents, and FAISS index."""
    (SESSIONS_DIR / f"{session_id}.json").unlink(missing_ok=True)
    shutil.rmtree(SESSION_DOCS_DIR / session_id, ignore_errors=True)
    shutil.rmtree(SESSION_FAISS_DIR / session_id, ignore_errors=True)
    # If the deleted session was active, reset state
    if st.session_state.active_session_id == session_id:
        st.session_state.active_session_id = None
        st.session_state.chat_history = []
        st.session_state.loaded_pdfs = []
        st.session_state.vector_store = None
        st.session_state.langgraph_app = None
        st.session_state.system_ready = False

def switch_to_session(session_id):
    meta_file = SESSIONS_DIR / f"{session_id}.json"
    if not meta_file.exists(): return
    with open(meta_file, "r") as f:
        meta = json.load(f)
        
    st.session_state.active_session_id = session_id
    st.session_state.chat_history = meta.get("history", [])
    st.session_state.loaded_pdfs = meta.get("files", [])
    
    from vector_store import load_faiss_index
    from agents_graph import build_graph
    
    faiss_path = str(SESSION_FAISS_DIR / session_id)
    vs = load_faiss_index(faiss_path)
    
    st.session_state.vector_store = vs
    st.session_state.langgraph_app = build_graph(vs, session_id)
    st.session_state.system_ready = True


def add_new_docs(uploaded_files):
    from vector_store import build_faiss_index, load_and_chunk_docs
    from agents_graph import build_graph
    
    # Generate a fixed sequential Session name to use as the session ID and Folder name
    existing = load_all_sessions()
    highest = 0
    for s in existing:
        title = s.get("title", "")
        if title.startswith("Session "):
            try:
                num = int(title.replace("Session ", ""))
                if num > highest: highest = num
            except: pass
    
    session_number = highest + 1
    session_id = f"Session_{session_number}"
    chat_name = f"Session {session_number}"
    
    # Isolate storage for this specific session using the clean name
    doc_dir = SESSION_DOCS_DIR / session_id
    faiss_dir = SESSION_FAISS_DIR / session_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    file_names = []
    for f in uploaded_files:
        dest = doc_dir / f.name
        dest.write_bytes(f.getbuffer())
        file_names.append(f.name)
        
    api_key = st.session_state.get("api_key")
    # Build a fresh memory index JUST for these newly uploaded files
    docs = load_and_chunk_docs(str(doc_dir), session_id=session_id, api_key=api_key)
    if not docs:
        vs = None
    else:
        vs = build_faiss_index(docs, str(faiss_dir))
    
    app = build_graph(vs, session_id)
    
    # Save the new session
    meta = {
        "id": session_id,
        "title": chat_name,
        "files": file_names,
        "created_at": datetime.now().isoformat(),
        "history": [{
            "role": "assistant",
            "content": "Hey! The new session is created and the file is uploaded. How may I help you?"
        }]
    }
    save_session_meta(session_id, meta)
    
    # Keep it at 5 max
    enforce_session_limit()
    
    # Switch to the brand new session instantly
    st.session_state.active_session_id = session_id
    st.session_state.chat_history = [{
        "role": "assistant",
        "content": "Welcome the file is uploaded sucessfuly, lets go"
    }]
    st.session_state.loaded_pdfs = file_names
    st.session_state.vector_store = vs
    st.session_state.langgraph_app = app
    st.session_state.system_ready = True
    st.session_state.uploader_key = st.session_state.get('uploader_key', 1) + 1
    st.rerun()


def render_meta(result: dict):
    urgency = result.get("urgency", "medium")
    urgency_cls = {"high": "tag-red", "medium": "tag-yellow", "low": "tag-green"}.get(urgency, "tag-yellow")
    verified = "✅ Verified" if result.get("is_approved") else "⚠️ Corrected"
    intent = result.get("intent", "").replace("_", " ").title()
    chunks = result.get("retrieved_count", 0)

    if result.get("reviewer_notes") and result["reviewer_notes"] != "Accurate":
        with st.expander("🔍 Reviewer Notes"):
            st.markdown(result["reviewer_notes"])
    if result.get("source_citations"):
        with st.expander("📚 Sources"):
            st.markdown(result["source_citations"])


def render_history():
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("meta"):
                # Only show tags for document_search (RAG) queries
                if m["meta"].get("intent") not in ["data_analysis", "general"]:
                    render_meta(m["meta"])


def handle_query(query: str):
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    active_key = st.session_state.get("api_key")
    if not active_key:
        msg = "⚠️ **Please enter your Gemini API Key in the Settings sidebar to continue.**"
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.chat_history.append({"role": "assistant", "content": msg, "meta": {"intent": "system", "urgency": "high"}})
        return

    from agents_graph import run_query

    with st.chat_message("assistant"):
        with st.status("Running pipeline…", expanded=True) as status:
            try:
                result = None
                for event in run_query(
                    st.session_state.langgraph_app,
                    query,
                    api_key=active_key,
                    session_id=st.session_state.active_session_id,
                    chat_history=st.session_state.chat_history,
                    uploaded_files=st.session_state.loaded_pdfs
                ):
                    if "status" in event:
                        status.update(label=event["status"])
                        st.markdown(f"✅ {event['status']}")
                    elif "final_answer" in event:
                        result = event
                status.update(label="Done", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Error", state="error")
                error_str = str(e)
                
                # Determine the exact reason for the error
                if "503" in error_str or "UNAVAILABLE" in error_str or "high demand" in error_str:
                    reason = "Google's Gemini API is currently overloaded with too many requests. This is a temporary issue on Google's servers, not a bug in your code."
                elif "429" in error_str or "quota" in error_str.lower():
                    reason = "You have exceeded your Gemini API rate limit or quota. Please wait a minute before trying again, or check your Google Cloud billing."
                elif "API_KEY" in error_str or "key" in error_str.lower() and "invalid" in error_str.lower():
                    reason = "The API key provided is invalid, missing, or lacks the necessary permissions to access the Gemini model."
                elif "context length" in error_str.lower() or "too large" in error_str.lower() or "token" in error_str.lower():
                    reason = "The documents you uploaded contain too much text or data for the model to process in a single request. Try uploading smaller files or fewer files at once."
                else:
                    reason = "An unexpected failure occurred during data processing or AI execution. This could be due to a syntax error generated by the AI in the background, or an unhandled edge case in the document structure."

                error_msg = f"**System Error:**\n`{error_str}`\n\n**Why this happened:**\n{reason}\n\n*Please try again or adjust your request based on the reason above.*"
                
                result = {
                    "final_answer": error_msg,
                    "intent": "general",
                    "error": True
                }

        st.markdown(result["final_answer"])
        # Only show tags for document_search (RAG) queries
        if result.get("intent") not in ["data_analysis", "general"]:
            render_meta(result)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["final_answer"],
        "meta": result,
    })
    
    # Save the updated history for this session to disk!
    update_session_history()


if "initialized" not in st.session_state:
    st.session_state.initialized = True
    if not st.session_state.active_session_id:
        sessions = load_all_sessions()
        if sessions:
            switch_to_session(sessions[0]["id"])


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Multi Agent System")
    
    st.markdown("**Settings**")
    st.text_input("Gemini API Key", type="password", key="api_key", help="Enter your Gemini API Key to use the application.", placeholder="AIzaSy...")
    st.divider()
    
    st.markdown("**Upload Documents**")

    if "last_processed_files" not in st.session_state:
        st.session_state.last_processed_files = []

    uploaded = st.file_uploader("Upload Documents", type=["pdf", "csv", "xlsx", "xls", "docx", "txt", "png", "jpg", "jpeg", "webp"], accept_multiple_files=True, label_visibility="collapsed", key=f"uploader_{st.session_state.get('uploader_key', 1)}")
    
    if uploaded:
        current_file_names = sorted([f.name for f in uploaded])
        if st.button("Upload & Chat", key="btn_upload_chat"):
            if not st.session_state.get("api_key"):
                msg = "⚠️ **Please enter your Gemini API Key in the Settings sidebar before uploading documents.**"
                st.session_state.chat_history.append({"role": "assistant", "content": msg, "meta": {"intent": "system"}})
                st.rerun()
            else:
                st.session_state.last_processed_files = current_file_names
                add_new_docs(uploaded)
        
    st.caption("(Maintaining last 5 sessions on each uploaded set of files)")

    st.divider()
    
    st.markdown("### Recent Sessions")
    all_sessions = load_all_sessions()
    
    if not all_sessions:
        st.markdown("<span style='color:#888; font-size:13px;'>No sessions yet. Upload a document to begin.</span>", unsafe_allow_html=True)
    else:
        for idx, s in enumerate(all_sessions):
            is_active = st.session_state.active_session_id == s["id"]
            # Keep the name permanently fixed to whatever it was originally created as
            base_label = s.get("title", f"Session {len(all_sessions) - idx}")
            label = f"🟢 {base_label}" if is_active else base_label
            
            col1, col2 = st.columns([5, 1])
            with col1:
                # Active session is a primary (colored) button, inactive are secondary (gray)
                if st.button(label, key=f"sel_{idx}_{s['id'][:8]}", use_container_width=True, type="primary" if is_active else "secondary"):
                    if not is_active:
                        switch_to_session(s["id"])
                        st.rerun()
            with col2:
                if st.button("−", key=f"del_{idx}_{s['id'][:8]}", help="Delete this session"):
                    delete_session(s["id"])
                    remaining = load_all_sessions()
                    if remaining:
                        switch_to_session(remaining[0]["id"])
                    st.rerun()

    st.divider()

# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state.system_ready:
    if st.session_state.get("chat_history"):
        render_history()
    else:
        st.markdown("""
        <div class="gemini-greeting" style="text-align: center; margin-top: 10vh;">
            <div style="font-size: 38px; font-weight: 500; background: -webkit-linear-gradient(45deg, #7cacf8, #e0949d, #b993ee); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.3;">
                Ask questions about your uploaded documents
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

if st.session_state.loaded_pdfs:
    files_str = ", ".join(st.session_state.loaded_pdfs)
    st.markdown(f"""
    <style>
    .file-pill {{
        background-color: rgba(24, 24, 37, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(137, 180, 250, 0.3); 
        padding: 4px 12px; 
        border-radius: 8px; font-size: 11px; font-weight: 500; color: #bac2de; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.3); white-space: nowrap;
        overflow: hidden; text-overflow: ellipsis; max-width: 200px;
        display: inline-block; transition: max-width 0.3s ease, background-color 0.2s ease;
        cursor: default;
    }}
    .file-pill:hover {{
        max-width: 90vw;
        background-color: rgba(30, 30, 46, 0.95);
        white-space: normal;
        word-wrap: break-word;
    }}
    </style>
    <div style='position: fixed; top: 60px; left: 50%; transform: translateX(-50%); z-index: 999999; text-align: center;'>
        <span class='file-pill'>
            Active Files: {files_str}
        </span>
    </div>
    """, unsafe_allow_html=True)

render_history()

greeting_placeholder = st.empty()

if not st.session_state.chat_history:
    greeting_placeholder.markdown("""
    <div class="gemini-greeting" style="text-align: center; margin-top: 10vh;">
        <div style="font-size: 38px; font-weight: 500; background: -webkit-linear-gradient(45deg, #7cacf8, #e0949d, #b993ee); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.3;">
            Ask questions about your uploaded documents
        </div>
    </div>
    """, unsafe_allow_html=True)

if query := st.chat_input("Ask a question about your documents…"):
    greeting_placeholder.empty()
    handle_query(query)
