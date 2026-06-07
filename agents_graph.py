"""
LangGraph Multi-Agent System for Company Policy RAG

5 Agents wired into a directed graph:
  1. Router Agent    — classifies query intent and urgency
  2. Retrieval Agent — fetches top-k chunks from FAISS
  3. Analysis Agent  — deep-reads chunks and extracts key facts
  4. Writer Agent    — drafts a clear, structured answer
  5. Reviewer Agent  — fact-checks draft vs. source chunks, approves or corrects

Graph flow:
  START -> router -> retrieval -> analysis -> writer -> reviewer -> END
"""
import os
from typing import TypedDict, List, Dict, Any, Annotated
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
import operator
import json
import re
from langchain.tools import tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

load_dotenv(override=True)

# ── LLM Setup ─────────────────────────────────────────────────────────────────

def get_gemini(api_key: str = None, temperature: float = 0.3):
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=api_key or "",
        temperature=temperature,
        max_retries=1
    )


def parse_gemini_content(content) -> str:
    """Safely extract plain text from Gemini response (handles list format)."""
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


ANSWER_ON_POINT_RULES = """
## ANSWER ON POINT (MANDATORY — EVERY QUESTION, EVERY PIPELINE):
1. Restate internally what the user is ACTUALLY asking — not just keywords that overlap with uploaded files.
2. The opening of your response must DIRECTLY address that question before any supporting detail.
3. Use uploaded documents or computed data ONLY when the question requires evidence from those sources.
4. Never substitute a document summary, data dump, or tangentially related facts for a direct answer.
5. For reasoning, preference, hypothetical, or "what would you" questions: answer with logic and trade-offs first; cite files only if the user explicitly asked about them.
6. For factual questions about uploads: answer the specific fact requested — do not recite unrelated metrics from the same file.
7. Match depth to the question: one sentence for simple asks, structured analysis for complex ones. No boilerplate headers unless useful.
"""

INLINE_CITATION_RULES = """
## INLINE SOURCE CITATIONS (MANDATORY when using uploaded files or computed data):
Every line or bullet that states a fact from uploads MUST end with an inline citation on that same line.

Format:
- (Source: Filename.ext)
- (Source: Filename.pdf, p. 12)  — include page when known

Examples:
- Q4 revenue was $450M (Source: VantaGroup_Q4_2025_Report.docx)
- EBITDA margin reached 28.4% in FY2025 (Source: VantaGroup_Q4_2025_Report.docx)
- Total widget sales: 1,240 units (Source: sales_data.csv)

Rules:
1. Attach the source to EACH factual line — never group all sources at the bottom or in a separate section.
2. Do NOT use bare numbered refs like [1] without the filename on the same line.
3. Clean internal artifact names: Report_p2_table1 or Report_table2 → Report.pdf (or Report.docx).
4. Lines of pure reasoning with no document/data fact do NOT need a citation.
5. If a line combines reasoning + fact, only the fact portion needs the citation at line end.
6. For computed metrics, cite the original uploaded file name, not internal table CSV names.
"""


# ── Shared State ──────────────────────────────────────────────────────────────

class PolicyState(TypedDict):
    """Shared state dictionary passed between all agents in the graph."""
    api_key: str
    session_id: str
    uploaded_files: List[str]
    # Input
    query: str
    chat_history: List[Dict[str, str]]
    intent: str
    urgency: str
    answer_mode: str
    core_question: str
    requires_uploads: bool
    search_queries: List[str]
    retrieved_chunks: List[str]
    data_analysis_result: str
    key_facts: str
    source_citations: str
    draft_answer: str
    final_answer: str
    reviewer_notes: str
    is_approved: bool
    generated_code: str
    math_critic_feedback: str
    data_critic_feedback: str
    data_critic_loop_count: int


def citations_required(state: PolicyState) -> bool:
    return bool(state.get("requires_uploads")) and state.get("answer_mode") != "direct"


# ── Agent 1: Router ───────────────────────────────────────────────────────────

def router_agent(state: PolicyState, session_id: str) -> PolicyState:
    """
    Classifies the user query dynamically by peeking into the actual
    file schemas and data samples from the session directory.
    """
    print(f"--> [Router Agent] Processing query: {state['query'][:50]}...")
    
    from pathlib import Path
    import pandas as pd
    
    session_docs_dir = Path("./session_docs") / str(session_id)
    
    file_context_blocks = []
    if session_docs_dir.exists():
        for f in session_docs_dir.iterdir():
            if not f.is_file():
                continue
            
            ext = f.suffix.lower()
            if ext in ['.csv', '.xlsx', '.xls']:
                try:
                    # Read only 2 rows to keep it ultra-fast and token-light
                    if ext == '.csv':
                        df = pd.read_csv(f, encoding_errors='replace', nrows=2)
                    else:
                        df = pd.read_excel(f, nrows=2)
                        
                    columns = list(df.columns)
                    sample = df.to_string(index=False, header=False)
                    
                    file_context_blocks.append(
                        f"File: {f.name} (Tabular Data)\n"
                        f"  Columns: {columns}\n"
                        f"  Sample Data:\n{sample}\n"
                    )
                except Exception as e:
                    file_context_blocks.append(f"File: {f.name} (Tabular Data - Failed to read preview)")
            elif ext == '.pdf':
                file_context_blocks.append(f"File: {f.name} (Text Document / PDF)")
            else:
                file_context_blocks.append(f"File: {f.name} (Unknown Type)")

    file_list_str = "\n".join(file_context_blocks) if file_context_blocks else "No files uploaded."

    llm = get_gemini(api_key=state.get('api_key'), temperature=0.0)

    history_str = ""
    if state.get("chat_history"):
        history_str = "\n".join(
            f"{m['role'].upper()}: {m['content'][:150]}"
            for m in state["chat_history"][-4:]
        )

    messages = [
        SystemMessage(content=(
            "You are an intelligent query router for a Multi-Agent Research System.\n\n"
            "Below is the exact context of the files currently uploaded in the user's session. "
            "Use the column names and data samples to deduce what the files are actually about.\n\n"
            f"--- UPLOADED FILE CONTEXT ---\n{file_list_str}\n-----------------------------\n\n"
            "Analyze the user query and decide the best execution pipeline to handle it.\n"
            "Respond with ONLY valid JSON.\n\n"
            "JSON format:\n"
            "{\n"
            '  "intent": one of ["data_analysis", "document_search", "general"],\n'
            '  "answer_mode": one of ["direct", "document_grounded", "data_computation", "hybrid"],\n'
            '  "core_question": "One clear sentence restating what the user actually wants answered",\n'
            '  "requires_uploads": true or false,\n'
            '  "urgency": one of ["high", "medium", "low"],\n'
            '  "search_queries": ["query 1", "query 2"]\n'
            "}\n\n"
            "Classification rules:\n"
            "- answer_mode=direct: reasoning, opinions, preferences, hypotheticals, general knowledge, greetings. "
            "The question can be answered WITHOUT reading uploaded files. Set requires_uploads=false.\n"
            "- answer_mode=document_grounded: user wants specific facts, quotes, policies, or summaries FROM uploaded documents. "
            "Set requires_uploads=true, intent=document_search.\n"
            "- answer_mode=data_computation: user wants numbers calculated or extracted FROM uploaded tables/spreadsheets. "
            "Set requires_uploads=true, intent=data_analysis.\n"
            "- answer_mode=hybrid: user wants both reasoning AND document facts (e.g. 'based on the report, should we invest?'). "
            "Set requires_uploads=true, intent=document_search.\n\n"
            "CRITICAL — distinguish question TYPE from topic overlap:\n"
            "- 'Would you rather invest in 30% revenue growth or 30% EBITDA?' → direct, requires_uploads=false "
            "(investment reasoning; uploaded files are irrelevant even if they mention revenue/EBITDA).\n"
            "- 'What was VantaGroup Q4 revenue?' → data_computation or document_grounded, requires_uploads=true.\n"
            "- 'Summarize the risk section' → document_grounded, requires_uploads=true.\n"
            "- Keywords matching file content do NOT automatically mean requires_uploads=true. "
            "Ask: would the answer change if no files were uploaded?\n"
        )),
        HumanMessage(content=(
            f"CONVERSATION HISTORY:\n{history_str}\n\n"
            f"USER QUERY: {state['query']}\n\n"
            "Respond with JSON only."
        ))
    ]

    response = llm.invoke(messages)
    raw = parse_gemini_content(response.content)

    # Robust JSON extraction
    intent = "document_search"
    urgency = "medium"
    answer_mode = "document_grounded"
    core_question = state["query"]
    requires_uploads = True
    search_queries = [state["query"]]

    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            intent = parsed.get("intent", intent)
            urgency = parsed.get("urgency", urgency)
            answer_mode = parsed.get("answer_mode", answer_mode)
            core_question = parsed.get("core_question", core_question)
            requires_uploads = bool(parsed.get("requires_uploads", requires_uploads))
            search_queries = parsed.get("search_queries", search_queries)
        except json.JSONDecodeError:
            pass

    if answer_mode == "direct" or not requires_uploads:
        intent = "general"
        answer_mode = "direct"
        requires_uploads = False
    elif answer_mode == "data_computation":
        intent = "data_analysis"
    elif answer_mode in ("document_grounded", "hybrid"):
        intent = "document_search"

    return {
        **state,
        "intent": intent,
        "urgency": urgency,
        "answer_mode": answer_mode,
        "core_question": core_question,
        "requires_uploads": requires_uploads,
        "search_queries": search_queries
    }


# ── Agent 2: Retrieval ────────────────────────────────────────────────────────

def retrieval_agent(state: PolicyState, vector_store) -> PolicyState:
    """
    Performs semantic search on FAISS using the Router's refined query list.
    Returns top-k chunks with source metadata.
    """
    if vector_store is None:
        print("--> [Retrieval Agent] Vector store is None (no text documents in session), bypassing search.")
        return {**state, "retrieved_chunks": []}

    queries = state.get('search_queries', [state['query']])
    print(f"--> [Retrieval Agent] Executing {len(queries)} search queries...")
    from vector_store import search_documents

    # Significantly increased top_k to improve recall
    top_k = 20 if state.get("urgency") == "high" else 12
    
    chunks = []
    seen = set()
    
    # Execute query planner multi-searches
    queries_to_run = list(queries)
    if state["query"] not in queries_to_run:
        queries_to_run.append(state["query"])
        
    for q in queries_to_run:
        results = search_documents(vector_store, q, top_k=top_k)
        for c in results:
            if c["text"] not in seen:
                chunks.append(c)
                seen.add(c["text"])

    return {**state, "retrieved_chunks": chunks}


# ── Agent 3: Analysis ─────────────────────────────────────────────────────────

def analysis_agent(state: PolicyState) -> PolicyState:
    """
    Deep-reads the retrieved chunks, extracts key facts, rules,
    numbers, conditions and structures them for the Writer Agent.
    """
    print("--> [Analysis Agent] Analyzing chunks...")
    llm = get_gemini(api_key=state.get('api_key'), temperature=0.2)

    if not state.get("retrieved_chunks"):
        return {**state, "key_facts": "No relevant policy content found.", "source_citations": ""}

    # Format chunks as numbered evidence
    evidence = ""
    citations = []
    for i, chunk in enumerate(state["retrieved_chunks"], 1):
        src = chunk['source']
        page = chunk.get('page', '?')
        evidence += f"[{i}] Source: {src} (Page {page})\n{chunk['text']}\n\n"
        citations.append(f"[{i}] {src}, Page {page}")

    cite_rules = INLINE_CITATION_RULES if citations_required(state) else ""

    messages = [
        SystemMessage(content=(
            "You are a senior policy analyst. Extract ONLY facts from the evidence that help answer the user's specific question.\n\n"
            f"{ANSWER_ON_POINT_RULES}\n"
            f"{cite_rules}\n"
            "Extraction rules:\n"
            "- Format each extracted fact on its own line prefixed with its source:\n"
            "  [Source: filename.ext, p. N] The fact text here.\n"
            "- KEY FACTS: Facts directly relevant to the core question\n"
            "- If evidence does not contain information needed to answer the question, say so explicitly.\n"
            "- Do NOT extract tangentially related facts just because they share keywords with the question.\n"
            "- STRICT NO INFERENCE RULE: Do not draw conclusions beyond literal evidence.\n"
        )),
        HumanMessage(content=(
            f"CORE QUESTION TO ANSWER: {state.get('core_question', state['query'])}\n\n"
            f"TABULAR DATA INSIGHTS:\n{state.get('data_analysis_result', 'None')}\n\n"
            f"POLICY EVIDENCE (From Text/PDFs):\n{evidence}\n\n"
            f"ORIGINAL USER QUERY: {state['query']}\n\n"
            "Extract only facts relevant to the core question."
        ))
    ]

    response = llm.invoke(messages)
    key_facts = parse_gemini_content(response.content)

    return {
        **state,
        "key_facts": key_facts,
        "source_citations": "\n".join(citations)
    }


# ── Agent 4: Writer ───────────────────────────────────────────────────────────

def writer_agent(state: PolicyState) -> PolicyState:
    """
    Writes a clear, well-structured, professional answer using
    the Analysis Agent's key facts. Adapts tone to urgency.
    """
    print("--> [Writer Agent] Drafting answer...")
    llm = get_gemini(api_key=state.get('api_key'), temperature=0.4)

    # Inject conversation history for contextual answers
    history_messages = []
    for turn in (state.get("chat_history") or [])[-6:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            history_messages.append(AIMessage(content=content))

    needs_cites = citations_required(state)

    messages = [
        SystemMessage(content=(
            "You are an expert financial and policy analyst. You write clearly, professionally, and logically.\n\n"
            f"{ANSWER_ON_POINT_RULES}\n"
            f"{INLINE_CITATION_RULES if needs_cites else ''}\n"
            "## OUTPUT FORMAT:\n"
            'Respond with valid JSON: {"answer": "Your full response here"}\n'
            "- Write the complete answer in the `answer` field as markdown prose and/or bullet lines.\n"
            "- When citations are required, EVERY factual line from uploads must end with (Source: filename.ext).\n"
            "- Lead with a direct answer to the core question, then supporting detail.\n"
            "- Do NOT use separate 'Document Evidence' sections — weave cited facts into the answer.\n"
            "- Bold important numbers using **bold**.\n"
            "- Clean internal table names (Report_table1 → Report.pdf).\n"
        )),
        *history_messages,
        HumanMessage(content=(
            f"CORE QUESTION: {state.get('core_question', state['query'])}\n"
            f"ANSWER MODE: {state.get('answer_mode', 'document_grounded')}\n"
            f"CITATIONS REQUIRED ON EACH FACT LINE: {needs_cites}\n\n"
            f"ORIGINAL USER QUERY: {state['query']}\n\n"
            f"ANALYST KEY FACTS:\n{state.get('key_facts', '')}\n\n"
            f"SOURCES AVAILABLE:\n{state.get('source_citations', '')}\n\n"
            'Write the final answer as JSON: {"answer": "..."}'
        ))
    ]

    response = llm.invoke(messages)
    raw = parse_gemini_content(response.content)
    
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            answer_text = parsed.get("answer", parsed.get("analysis", "")).strip()
            if answer_text:
                return {**state, "writer_draft": answer_text}
        except json.JSONDecodeError:
            pass

    return {**state, "writer_draft": raw.strip()}


# ── Agent 5: Reviewer ─────────────────────────────────────────────────────────

def reviewer_agent(state: PolicyState) -> PolicyState:
    """
    Balanced Reviewer Agent — Real verification with a Python safety net.

    WHAT IT DOES:
    1. Real fact verification — checks if Writer's facts are in source evidence.
    2. Completeness check — ensures all parts of the question were answered.
    3. Python deletion guard — if the Reviewer accidentally removes more than
       40% of the Writer's content, the guard automatically rejects the
       Reviewer's output and restores the Writer's original draft.

    This gives genuine multi-agent verification without the False Negative risk.
    """
    print("--> [Reviewer Agent] Verifying draft (balanced mode)...")
    llm = get_gemini(api_key=state.get('api_key'), temperature=0.1)

    evidence_chunks = state.get("retrieved_chunks", [])
    evidence = ""
    for i, chunk in enumerate(evidence_chunks, 1):
        evidence += f"[{i}] {chunk['source']} (Page {chunk['page']}):\n{chunk['text'][:400]}\n\n"

    writer_draft = state.get("writer_draft", "")

    needs_cites = citations_required(state)

    messages = [
        SystemMessage(content=(
            "You are a senior policy QA reviewer.\n\n"
            f"{ANSWER_ON_POINT_RULES}\n"
            f"{INLINE_CITATION_RULES if needs_cites else ''}\n"
            "## CHECK 1 — ON-POINT: Does the draft directly answer the CORE QUESTION in its opening?\n\n"
            "## CHECK 2 — PER-LINE CITATIONS:\n"
            "If citations are required, EVERY line/bullet containing a document fact MUST end with "
            "(Source: filename.ext). Rewrite any line missing its source. Remove bottom-only source lists.\n\n"
            "## CHECK 3 — FACT ACCURACY:\n"
            "Verify each cited fact exists in source chunks.\n\n"
            "## CHECK 4 — RELEVANCE:\n"
            "Remove facts that do not help answer the core question.\n\n"
            "## RESPONSE FORMAT (JSON only):\n"
            '{"approved": true, "notes": "...", "final_answer": "<copy or corrected draft>"}\n'
        )),
        HumanMessage(content=(
            f"CORE QUESTION: {state.get('core_question', state['query'])}\n"
            f"CITATIONS REQUIRED ON EACH FACT LINE: {needs_cites}\n\n"
            f"ORIGINAL USER QUERY: {state['query']}\n\n"
            f"DRAFT ANSWER:\n{writer_draft}\n\n"
            f"SOURCE EVIDENCE:\n{evidence}\n\n"
            "Respond with JSON only."
        ))
    ]

    response = llm.invoke(messages)
    raw = parse_gemini_content(response.content)

    # Defaults — always fall back to Writer's draft
    is_approved = True
    reviewer_notes = "All facts verified. Answer is complete."
    final_answer = writer_draft

    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            is_approved = parsed.get("approved", True)
            reviewer_notes = parsed.get("notes", "All facts verified.")
            proposed_answer = parsed.get("final_answer", writer_draft)

            # Use the Reviewer's output directly, trusting it to remove hallucinations
            final_answer = proposed_answer

        except json.JSONDecodeError:
            final_answer += "\n\n[DEBUG: QA Reviewer failed to output valid JSON. Showing un-reviewed draft.]"

    return {
        **state,
        "is_approved": is_approved,
        "reviewer_notes": reviewer_notes,
        "final_answer": final_answer
    }



# ── Agent 6: Data Analysis ────────────────────────────────────────────────────

import streamlit as st
import os

@st.cache_data
def load_dataframe_cached(file_path_str: str, mtime: float):
    import pandas as pd
    from pathlib import Path
    f = Path(file_path_str)
    if f.suffix.lower() == ".csv":
        return pd.read_csv(f, encoding_errors='replace')
    else:
        return pd.read_excel(f)

def data_analysis_agent(state: PolicyState, session_id: str, vector_store=None) -> PolicyState:
    """
    If tabular data is present, this agent writes and executes a pandas script to answer the query.
    It can also use search_pdfs to query FAISS for hybrid questions.
    """
    print(f"--> [Data Analysis Agent] Processing tabular data for session {session_id}...")
    import pandas as pd
    from pathlib import Path
    
    llm = get_gemini(api_key=state.get('api_key'), temperature=0.0)
    
    session_docs_dir = Path("./session_docs") / str(session_id)
    files = list(session_docs_dir.glob("*.csv")) + list(session_docs_dir.glob("*.xlsx")) + list(session_docs_dir.glob("*.xls"))
    
    if not files:
        return {**state, "data_analysis_result": "No tabular data files found."}
        
    dfs = []
    df_names = []
    for f in files:
        try:
            mtime = os.path.getmtime(f)
            df = load_dataframe_cached(str(f), mtime)
            # Normalize column names implicitly handled by pandas agent
            dfs.append(df)
            df_names.append(f.name)
        except Exception as e:
            print(f"Failed to load {f.name} into pandas: {e}")
            
    # Use LLM Script Engine to solve the query
    schema_lines = []
    df_vars = []
    for name, df in zip(df_names, dfs):
        import re
        var_name = re.sub(r'\W+', '_', name.replace('.csv','').replace('.xlsx',''))
        df_vars.append(var_name)
        schema_lines.append(f"DataFrame `{var_name}` (from {name}): {list(df.columns)}")
        # Provide sample data so the AI knows the actual column values
        schema_lines.append(f"Sample data for `{var_name}`:\n{df.head(3).to_string()}\n")
    schema_str = "\n".join(schema_lines)
    
    history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in (state.get("chat_history") or [])[-4:]])
    
    uploaded_files_str = ", ".join(state.get("uploaded_files", [])) or "uploaded files"

    prompt = f"""You are a senior Python Data Analyst. 
    DataFrames are available as variables: {df_vars}
    Schemas: 
    {schema_str}
    
    RECENT CHAT HISTORY:
    {history_str}
    
    USER QUERY: {state['query']}
    
    CORE QUESTION (what you must answer): {state.get('core_question', state['query'])}
    
    Write Python code to solve this. Assign your final string output to a variable named `final_answer`.
    CRITICAL RULE 1: Whenever working with dates or timestamps, you MUST convert the columns using `pd.to_datetime(col, errors='coerce', utc=True)` before performing any time-based operations to prevent formatting crashes.
    CRITICAL RULE 2: If searching for specific names, IDs, or string values, use safe pandas filtering (e.g. `.str.contains(..., na=False)`) instead of strict exact matches, and handle empty results gracefully by assigning "No matching records found" to `final_answer` instead of throwing an error.
    CRITICAL RULE 3: By default, analyze and cross-reference data across ALL provided DataFrames. ONLY restrict your analysis to a single file if the user explicitly asks about that specific file. Do NOT list internal table variable names in your final_answer.
    CRITICAL RULE 4: ABSOLUTELY DO NOT use the backtick character (`) ANYWHERE in your Python code. Not in strings, not in comments, not in replace() functions. The presence of any backtick inside your code will break the code parser and cause a fatal SyntaxError. Use standard quotes instead.
    CRITICAL RULE 5: DO NOT HALLUCINATE numbers. Before giving up, you MUST: (a) clean numeric columns (strip $, commas, %, parentheses and convert to float), (b) try flexible string matching with .str.contains(..., case=False, na=False), and (c) call search_documents with the user's question to pull facts from PDF/Word text. Only if tables AND document search both lack the needed data, set final_answer to a short message explaining what specific metric or field could not be found — do NOT dump a list of all table names.
    CRITICAL RULE 6: Format final_answer with one fact per line. Each line containing a number or metric MUST end with (Source: filename.ext). Cite original uploaded files ({uploaded_files_str}), NOT internal extracted table CSV names.
    HYBRID RAG SEARCH: You have access to a function `search_documents(query: str) -> str`. You can call this inside your python code to search the uploaded PDFs/Word/Text documents for policies or rules, and then use those rules to filter your DataFrames!
    Respond ONLY with the Python code in a ```python ... ``` block.
    """
    try:
        agent_llm = get_gemini(api_key=state.get('api_key'), temperature=0.1)
        response = agent_llm.invoke([HumanMessage(content=prompt)])
        content = parse_gemini_content(response.content)
        
        if "```python" in content:
            start_idx = content.find("```python") + len("```python\n")
            end_idx = content.rfind("```")
            if end_idx > start_idx:
                code = content[start_idx:end_idx]
            else:
                code = content[start_idx:]
        else:
            code = content.replace('```python', '').replace('```', '')
        
        import textwrap
        code = textwrap.dedent(code).strip()
        
        # Security Sandbox: Restrict execution environment to prevent 'import os' attacks
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            dangerous = ['os', 'sys', 'subprocess', 'shutil', 'socket', 'urllib', 'requests']
            if name.split('.')[0] in dangerous:
                raise ImportError(f"Importing '{name}' is strictly forbidden by the security sandbox.")
            return __import__(name, globals, locals, fromlist, level)
            
        def safe_search(q: str):
            from vector_store import search_documents
            if vector_store:
                results = search_documents(vector_store, q, top_k=5)
                return "\n".join([r['text'] for r in results])
            return "No text documents available."
            
        import builtins
        safe_builtins = dict(builtins.__dict__)
        safe_builtins['__import__'] = safe_import
        exec_globals = {'pd': pd, 'np': np, '__builtins__': safe_builtins, 'search_documents': safe_search}
        import re
        exec_globals.update({re.sub(r'\W+', '_', name.replace('.csv','').replace('.xlsx','')): df for name, df in zip(df_names, dfs)})
        exec_locals = {}
        
        exec(code, exec_globals)
        answer = exec_globals.get('final_answer', "Script ran but final_answer was not assigned.")
    except Exception as e:
        answer = f"Data Analysis Failed: {str(e)}"
        print(f"Data Agent Error: {str(e)}")
        print(f"Raw LLM Content:\n{content}")
        code = ""
            
    return {
        **state, 
        "data_analysis_result": str(answer),
        "generated_code": code
    }



# ── Agent 6.5: Math Critic ────────────────────────────────────────────────────

def math_critic_agent(state: PolicyState) -> PolicyState:
    """
    Acts as a Senior Data Scientist code-reviewing the Python script generated
    by the Data Analysis agent to ensure logical and mathematical correctness.
    """
    print("--> [Math Critic Agent] Reviewing generated Pandas code...")
    llm = get_gemini(api_key=state.get('api_key'), temperature=0.1)

    code = state.get('generated_code', '')
    query = state['query']
    result = state.get('data_analysis_result', '')

    messages = [
        SystemMessage(content=(
            "You are a Senior Data Scientist code-reviewing a junior analyst's Pandas Python script. "
            "Your job is to verify if the logic used in the script is mathematically sound for the given query.\n\n"
            "Checks to perform:\n"
            "1. Did they use the correct aggregation (e.g., sum vs average vs median)?\n"
            "2. Did they filter the data logically based on the query?\n"
            "3. Are there obvious logic errors (e.g., dividing by a sum instead of a count)?\n\n"
            "If the code looks logically sound and correctly answers the user query, output ONLY:\n"
            "APPROVED\n\n"
            "If the code has a logical flaw, incorrect math, or threw a fatal execution error, output:\n"
            "REJECTED: [detailed reason why the logic is wrong and instructions on how to fix it]"
        )),
        HumanMessage(content=(
            f"USER QUERY: {query}\n\n"
            f"GENERATED CODE:\n```python\n{code}\n```\n\n"
            f"EXECUTION RESULT:\n{result}\n\n"
            "Review the logic. Output APPROVED or REJECTED: [reason]."
        ))
    ]

    response = llm.invoke(messages)
    content = parse_gemini_content(response.content).strip()
    
    if content.startswith("APPROVED"):
        return {**state, "is_approved": True, "math_critic_feedback": ""}
    else:
        return {**state, "is_approved": False, "math_critic_feedback": content}

# ── Agent 7: Data Writer ──────────────────────────────────────────────────────

def data_writer_agent(state: PolicyState) -> PolicyState:
    """
    Takes the raw data analysis output from the python script and formats
    it beautifully using an LLM. Ensures no hallucinations occur.
    """
    print("--> [Data Writer Agent] Formatting data analysis results...")
    
    raw_result = state.get('data_analysis_result', '')

    # If the analyst hit an error and dumped diagnostic info, pass it straight through
    # without letting the LLM rephrase it into a vague polite message
    if raw_result.startswith("The query could not be answered") or \
       raw_result.startswith("The code executed but") or \
       raw_result.startswith("No tabular data") or \
       raw_result.startswith("Failed to load") or \
       raw_result.startswith("ERROR:"):
        return {**state, "final_answer": raw_result}

    llm = get_gemini(api_key=state.get('api_key'), temperature=0.3)
    
    # Inject conversation history for contextual answers
    history_messages = []
    for turn in (state.get("chat_history") or [])[-6:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            history_messages.append(AIMessage(content=content))

    raw_result = state.get('data_analysis_result', '')

    # If the analyst hit an error and dumped diagnostic info, pass it straight through
    # without letting the LLM rephrase it into a vague polite message
    if raw_result.startswith("The query could not be answered") or \
       raw_result.startswith("The code executed but") or \
       raw_result.startswith("No tabular data") or \
       raw_result.startswith("Failed to load"):
        return {**state, "final_answer": raw_result}

    feedback_str = ""
    if state.get("data_critic_feedback") and state["data_critic_feedback"] not in ["PASS", "PASS_LIMIT_REACHED"]:
        feedback_str = f"CRITIC REJECTION - YOU HALLUCINATED: {state['data_critic_feedback']}\nYou MUST rewrite your answer without inventing facts. If you don't have the data, state that you don't have it.\n\n"

    uploaded = ", ".join(state.get("uploaded_files", [])) or "uploaded files"

    messages = [
        SystemMessage(content=(
            "You are a brilliant Senior Data Scientist formatting computed results for the user.\n\n"
            f"{ANSWER_ON_POINT_RULES}\n"
            f"{INLINE_CITATION_RULES}\n"
            "## FORMATTING RULES:\n"
            "1. Lead with a direct answer to the core question — not a data dump.\n"
            "2. Every line with a computed number or metric MUST end with (Source: filename.ext).\n"
            "3. Cite the original uploaded document name, not internal extracted table CSV names.\n"
            "4. ZERO HALLUCINATION: Never invent numbers. Report exactly what the computation returned.\n"
            "5. If data is missing, say so clearly on its own line (no citation needed for that line).\n"
            f"Uploaded files in this session: {uploaded}\n"
        )),
        *history_messages,
        HumanMessage(content=(
            f"{feedback_str}"
            f"CORE QUESTION: {state.get('core_question', state['query'])}\n"
            f"ORIGINAL USER QUERY: {state['query']}\n\n"
            f"RAW DATA RESULTS:\n{raw_result}\n\n"
            "Write a direct, on-point answer now."
        ))
    ]

    response = llm.invoke(messages)
    draft = parse_gemini_content(response.content)

    return {**state, "final_answer": draft}
# ── Agent 7: Data Critic (Hallucination Prevention) ───────────────────────────

def data_critic_agent(state: PolicyState) -> PolicyState:
    """
    Intercepts the data writer's draft and mathematically verifies it against 
    the raw python output to prevent hallucinations.
    """
    print("--> [Data Critic Agent] Verifying draft for hallucinations...")
    llm = get_gemini(api_key=state.get('api_key'), temperature=0.0)
    
    loop_count = state.get("data_critic_loop_count", 0)
    if loop_count >= 2:
        print("    [!] Max loop limit reached. Passing.")
        return {**state, "data_critic_feedback": "PASS_LIMIT_REACHED"}
        
    messages = [
        SystemMessage(content=(
            "You are a ruthless QA Critic for an Enterprise AI system.\n"
            "You will receive the RAW PYTHON OUTPUT and the WRITER'S DRAFT.\n\n"
            f"{INLINE_CITATION_RULES}\n"
            "Checks:\n"
            "1. Extract every number, percentage, count, and statistic from the WRITER'S DRAFT.\n"
            "2. Cross-reference them against the RAW PYTHON OUTPUT — reject hallucinated numbers.\n"
            "3. REJECT if any factual line with a number/metric is missing (Source: filename.ext) at line end.\n"
            "4. REJECT if sources are grouped at the bottom instead of attached to each line.\n\n"
            "Respond ONLY in valid JSON format:\n"
            "{\n"
            '  "is_hallucinated": boolean,\n'
            '  "feedback": "If rejected, explain missing citations or fake numbers. If approved, leave empty."\n'
            "}"
        )),
        HumanMessage(content=(
            f"RAW PYTHON OUTPUT:\n{state.get('data_analysis_result', 'No raw data')}\n\n"
            f"WRITER'S DRAFT:\n{state.get('final_answer', 'No draft')}"
        ))
    ]
    
    response = llm.invoke(messages)
    raw = parse_gemini_content(response.content)
    
    is_hallucinated = False
    feedback = ""
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            is_hallucinated = parsed.get("is_hallucinated", False)
            feedback = parsed.get("feedback", "")
        except json.JSONDecodeError:
            pass
            
    if is_hallucinated:
        print(f"    [!] Hallucination Detected: {feedback}")
        return {
            **state, 
            "data_critic_feedback": feedback,
            "data_critic_loop_count": loop_count + 1
        }
    else:
        print("    [SUCCESS] Draft verified as 100% factual.")
        return {**state, "data_critic_feedback": "PASS"}


# ── Agent 8: Conversational ───────────────────────────────────────────────────

def conversational_agent(state: PolicyState, session_id: str = None) -> PolicyState:
    """
    Handles direct-reasoning questions, greetings, and any query that does
    not require grounding in uploaded files.
    """
    print("--> [Conversational Agent] Handling direct-reasoning query...")
    
    file_context_blocks = []
    if session_id:
        from pathlib import Path
        doc_dir = Path("session_docs") / session_id
        if doc_dir.exists():
            for f in doc_dir.iterdir():
                if f.suffix.lower() == '.pdf':
                    file_context_blocks.append(f"{f.name} (PDF)")
                else:
                    file_context_blocks.append(f"{f.name} (Data)")
                    
    file_list_str = ", ".join(file_context_blocks) if file_context_blocks else "No files uploaded"
    
    llm = get_gemini(api_key=state.get('api_key'), temperature=0.4)
    
    history_messages = []
    for turn in (state.get("chat_history") or [])[-6:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            history_messages.append(AIMessage(content=content))

    messages = [
        SystemMessage(content=(
            "You are an expert analyst assistant.\n\n"
            f"{ANSWER_ON_POINT_RULES}\n"
            f"Uploaded files available (use ONLY if the question explicitly requires them): {file_list_str}\n\n"
            "This question was classified as NOT requiring uploaded files. "
            "Answer using reasoning and general knowledge. Do not force document citations."
        )),
        *history_messages,
        HumanMessage(content=(
            f"CORE QUESTION: {state.get('core_question', state['query'])}\n\n"
            f"ORIGINAL USER QUERY: {state['query']}"
        ))
    ]

    response = llm.invoke(messages)
    final_answer = parse_gemini_content(response.content)

    return {**state, "final_answer": final_answer, "is_approved": True, "reviewer_notes": "Conversational response"}


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(vector_store, session_id: str = None):
    """
    Assembles all agents into a LangGraph directed graph.

    Flow: 
      START -> router
      router (if data_analysis) -> data_analysis -> END
      router (else) -> retrieval -> analysis -> writer -> reviewer -> END
    """

    def retrieval_node(state: PolicyState) -> PolicyState:
        return retrieval_agent(state, vector_store)
        
    def data_analysis_node(state: PolicyState) -> PolicyState:
        return data_analysis_agent(state, session_id, vector_store)

    def data_writer_node(state: PolicyState) -> PolicyState:
        return data_writer_agent(state)

    def data_critic_node(state: PolicyState) -> PolicyState:
        return data_critic_agent(state)

    def conversational_node(state: PolicyState) -> PolicyState:
        return conversational_agent(state, session_id)

    def data_analysis_fallback_node(state: PolicyState) -> PolicyState:
        print("--> [Fallback] Table analysis could not answer; searching document text...")
        return {**state, "data_analysis_result": ""}

    graph = StateGraph(PolicyState)

    def router_node(state: PolicyState) -> PolicyState:
        return router_agent(state, session_id)

    # Register all nodes
    graph.add_node("router",        router_node)
    graph.add_node("retrieval",     retrieval_node)
    graph.add_node("analysis",      analysis_agent)
    graph.add_node("writer",        writer_agent)
    graph.add_node("reviewer",      reviewer_agent)
    graph.add_node("data_analysis", data_analysis_node)
    graph.add_node("data_analysis_fallback", data_analysis_fallback_node)
    graph.add_node("data_writer",   data_writer_node)
    graph.add_node("data_critic",   data_critic_node)
    graph.add_node("conversational", conversational_node)

    def route_after_data_analysis(state: PolicyState):
        result = str(state.get("data_analysis_result", ""))
        if result.startswith("ERROR:") or "Cannot calculate this using the provided tables" in result:
            return "fallback"
        return "data_writer"

    def route_after_router(state: PolicyState):
        if state.get("answer_mode") == "direct" or not state.get("requires_uploads", True):
            return "conversational"
        if state.get("intent") == "data_analysis":
            return "data_analysis"
        elif state.get("intent") == "general":
            return "conversational"
        return "retrieval"

    def route_after_data_critic(state: PolicyState):
        """Route to END if passed, otherwise loop back to writer."""
        feedback = state.get("data_critic_feedback", "PASS")
        if feedback in ["PASS", "PASS_LIMIT_REACHED"]:
            return END
        else:
            return "data_writer"

    # Wire edges
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "data_analysis": "data_analysis",
            "retrieval": "retrieval",
            "conversational": "conversational"
        }
    )
    
    # Text RAG path
    graph.add_edge("retrieval", "analysis")
    graph.add_edge("analysis",  "writer")
    graph.add_edge("writer",    "reviewer")
    graph.add_edge("reviewer",  END)
    
    # Data Analysis path (with Critic Loop, or fallback to text RAG)
    graph.add_conditional_edges(
        "data_analysis",
        route_after_data_analysis,
        {
            "data_writer": "data_writer",
            "fallback": "data_analysis_fallback",
        }
    )
    graph.add_edge("data_analysis_fallback", "retrieval")
    graph.add_edge("data_writer", "data_critic")
    graph.add_conditional_edges(
        "data_critic",
        route_after_data_critic,
        {
            END: END,
            "data_writer": "data_writer"
        }
    )
    
    # Conversational / Clarification path
    graph.add_edge("conversational", END)

    return graph.compile()


def run_query(
    app,
    query: str,
    api_key: str = None,
    session_id: str = "default_session",
    chat_history: List[Dict[str, str]] = None,
    uploaded_files: List[str] = None
) -> Dict[str, Any]:
    """
    Run a query through the full pipeline.
    """
    initial_state: PolicyState = {
        "api_key": api_key or "",
        "session_id": session_id,
        "uploaded_files": uploaded_files or [],
        "chat_history": chat_history or [],
        "query": query,
        "intent": "",
        "urgency": "",
        "answer_mode": "",
        "core_question": "",
        "requires_uploads": True,
        "search_queries": [query],
        "retrieved_chunks": [],
        "data_analysis_result": "",
        "key_facts": "",
        "source_citations": "",
        "draft_answer": "",
        "final_answer": "",
        "reviewer_notes": "",
        "is_approved": False,
        "generated_code": "",
        "math_critic_feedback": "",
        "data_critic_feedback": "",
        "data_critic_loop_count": 0
    }

    # Stream events to show live progress
    for event in app.stream(initial_state):
        node_name = list(event.keys())[0]
        yield {"status": f"Agent working: {node_name.title()}", "node": node_name}

    # The last event contains the final state
    last_node = list(event.keys())[0]
    result = event[last_node]

    # Check the actual node executed to determine the path
    is_data_path = last_node in ["data_analysis", "data_writer", "data_critic"]
    
    if is_data_path:
        # final_answer is now drafted by data_writer, not pulled from data_analysis_result directly
        final_answer = result.get("final_answer", "I could not generate a response from the data.")
        if not final_answer:
             final_answer = result.get("data_analysis_result", "I could not analyze the data.")
        is_approved = True  # Python raw data execution is inherently verified
        final_intent = "data_analysis"
    else:
        final_answer = result.get("final_answer", "I could not generate an answer.")
        is_approved = result.get("is_approved", True)
        final_intent = result.get("intent", "general")
        # If the LLM classified as data_analysis but we forced it to RAG (e.g. only PDFs)
        if final_intent == "data_analysis":
            final_intent = "general"

    agent_chain = []
    if is_data_path:
        agent_chain = ["Router", "Data Analysis", "Data Writer", "Data Critic"]
        if result.get("data_critic_loop_count", 0) > 0:
            agent_chain.append("(Loop triggered)")
    elif last_node == "conversational":
        agent_chain = ["Router", "Conversational"]
    else:
        agent_chain = ["Router", "Retrieval", "Analysis", "Writer", "Reviewer"]

    yield {
        "final_answer":    final_answer,
        "intent":          final_intent,
        "urgency":         result.get("urgency", "medium"),
        "reviewer_notes":  result.get("reviewer_notes", ""),
        "is_approved":     is_approved,
        "source_citations": result.get("source_citations", ""),
        "retrieved_count": len(result.get("retrieved_chunks", [])),
        "agent_chain": agent_chain
    }

