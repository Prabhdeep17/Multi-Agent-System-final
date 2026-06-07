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
            '  "urgency": one of ["high", "medium", "low"],\n'
            '  "search_queries": ["query 1", "query 2"] (if document_search, provide semantic variations)\n'
            "}\n\n"
            "Rules:\n"
            "- Choose 'data_analysis' if the query involves math, calculations, counting, financial metrics (revenue, EBITDA, margins, profits, growth, loss), analyzing trends, or referencing tables/data. If the query asks for ANY financial numbers from a Q-report or financial statement, you MUST choose data_analysis. Also choose this for HYBRID questions requiring both text policies and math.\n"
            "- Choose 'document_search' if the query explicitly asks ONLY for text facts, policies, paragraphs, rules, or semantic knowledge found in the Text Documents (PDFs/Word) AND does NOT require analyzing numerical data or financial metrics.\n"
            "- Choose 'general' ONLY if the CURRENT USER QUERY is a casual greeting completely unrelated to the files. IGNORE the conversation history when deciding the intent. If the current query asks a question, requests a summary, or mentions any topic that could potentially be in the documents, you MUST choose document_search!\n"
            "CRITICAL RULE: Never choose 'general' for questions about data, policies, rules, or summaries. 'general' is ONLY for 'hi', 'hello', or asking what you are.\n"
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
    intent = "document_search"  # Default fallback
    urgency = "medium"
    search_queries = [state["query"]]

    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            intent = parsed.get("intent", "document_search")
            urgency = parsed.get("urgency", "medium")
            search_queries = parsed.get("search_queries", [state["query"]])
        except json.JSONDecodeError:
            pass

    # PYTHON-LEVEL OVERRIDE: Never allow 'general' if the user is asking a question!
    # This prevents the LLM from getting confused by chat history and hallucinating.
    query_lower = state['query'].lower()
    question_words = ["?", "how", "what", "where", "when", "why", "who", "which", "explain", "summarize", "tell me", "detail", "does", "is", "are", "can", "could", "would", "should"]
    if intent == "general" and any(w in query_lower for w in question_words):
        intent = "document_search"

    return {
        **state,
        "intent": intent,
        "urgency": urgency,
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
        evidence += f"[{i}] Source: {chunk['source']} (Page {chunk['page']})\n{chunk['text']}\n\n"
        citations.append(f"[{i}] {chunk['source']}, Page {chunk['page']}")

    messages = [
        SystemMessage(content=(
            "You are a senior policy analyst. Your job is to extract every relevant fact, "
            "rule, condition, number, date, and exception from the provided policy evidence.\n\n"
            "Be exhaustive. Structure your analysis as:\n"
            "- KEY RULES: The main policy rules that apply\n"
            "- NUMBERS & LIMITS: Any quantities, days, amounts, percentages\n"
            "- CONDITIONS & EXCEPTIONS: When rules apply or don't apply\n"
            "- HIERARCHY & OVERRIDES: Specifically note where a specific exception overrides a general rule. Do not let general rules contradict specific exceptions.\n"
            "- IMPORTANT NOTES: Any warnings, deadlines, or special cases\n"
            "- CRITICAL CONTEXT FIREWALL: Treat every uploaded file as an independent context silo. You are strictly forbidden from fabricating, assuming, or weaving narrative connections between separate files unless there is an explicit textual cross-reference connecting them by name or unique ID within the source data.\n"
            "- TABULAR VS. TEXTUAL ISOLATION RULE: If one file contains numerical data tables and another contains general narratives, keep their operations completely separate. Do NOT infer or state that a person mentioned in a text file was responsible for, processed, calculated, or monitored rows in a data table unless that text file explicitly mentions those exact data parameters or SKU IDs.\n"
            "- STRICT NO INFERENCE RULE (OVERREACH PREVENTION): You must act as a strict, literal legal clerk. Do NOT draw conclusions, infer intent, read between the lines, or combine separate facts into a new narrative unless the text explicitly does so. If the text says A and B, do not claim A caused B unless explicitly stated. Never overreach beyond the literal evidence.\n\n"
            "Always reference evidence numbers [1], [2] etc."
        )),
        HumanMessage(content=(
            f"TABULAR DATA INSIGHTS (Warning - May Contain Hallucinations):\n{state.get('data_analysis_result', 'None')}\n\n"
            f"POLICY EVIDENCE (From Text/PDFs):\n{evidence}\n\n"
            f"USER QUESTION: {state['query']}\n"
            f"QUERY INTENT: {state.get('intent', 'general')}\n\n"
            "Extract all relevant facts from the tabular data and evidence above. If the query asks about both, combine them intelligently."
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

    tone = "clear and urgent" if state.get("urgency") == "high" else "professional and helpful"

    if len(state.get("uploaded_files", [])) > 1:
        file_rule = f"- STRICT RULE FOR MULTIPLE FILES ({', '.join(state['uploaded_files'])}): NEVER group or list these file names at the beginning of your answer. You MUST use inline citations for EVERY piece of data you provide (e.g., 'Revenue grew by 10% (Source: data.csv)'). Failure to cite the specific file for each fact is unacceptable.\n"
    elif len(state.get("uploaded_files", [])) == 1:
        file_rule = "- There is only one file uploaded. Do NOT mention its name.\n"
    else:
        file_rule = ""

    messages = [
        SystemMessage(content=(
            f"You are an expert HR policy communication specialist. Write in a {tone} tone.\n\n"
            "## FACTS-ONLY CONTRACT (MOST IMPORTANT RULE):\n"
            "You are STRICTLY FORBIDDEN from writing any fact, number, date, name, amount, or policy rule "
            "that is not explicitly present in the ANALYST KEY FACTS provided below.\n"
            "- Do NOT add context from general knowledge.\n"
            "- Do NOT infer or extrapolate beyond what is written. Do NOT draw your own conclusions.\n"
            "- Do NOT fill gaps with assumptions or create causal relationships not explicitly stated.\n"
            "- OVERREACH PREVENTION: Do not synthesize or editorialize facts. Stick strictly to the literal text.\n"
            "- If a fact is not in the key facts → it does not exist for this answer.\n"
            "- NO CONCLUDING NARRATIVES: Do not write concluding, summarizing, or bridging sentences at the end of paragraphs (e.g. 'These metrics further support...'). End abruptly after stating the hard facts.\n"
            "- STRICT NO-SYNTHESIS RULE: Do not try to combine facts from Document A with facts from Document B to form a cohesive argument. Present facts from different documents as completely separate, disconnected bullet points or distinct paragraphs.\n\n"
            "## FORMAT RULES (STRICT):\n"
            "- Your ENTIRE output MUST be a strict list of independent bullet points.\n"
            "- Do NOT write an introductory paragraph. Do NOT write a concluding paragraph.\n"
            f"{file_rule}"
            "- Bold important numbers/limits using **bold**\n"
            "- Reference sources inline: [1], [2]\n"
            "- If a specific exception exists in the key facts, explicitly state that it overrides the general rule.\n"
            "- If the key facts indicate that no relevant information was found, simply state that the provided documents do not contain the answer to the user's question. Use your own natural wording, be concise, and do NOT add any filler text or tell the user to contact anyone.\n"
            "- CROSS-DOCUMENT RULE: If the key facts involve both a spreadsheet and a text policy, "
            "only state connections that are EXPLICITLY written in the key facts. "
            "Do not create narrative bridges between unrelated documents.\n"
            "- DOCUMENT NAMES NOT NUMBERS: Never use terms like 'Document 1', 'File A', or 'the first file'. Always refer to the exact source document name (e.g. 'According to policy.pdf...').\n"
            "- ORIGINAL PDF CITAIONS: If a data source has an internal extracted name like 'Report_p2_table1.csv' or 'Report_table1', you MUST clean it up and cite the original document name instead (e.g. 'Report.pdf'). Do NOT expose internal '_table' or '_p2' artifacts to the user.\n"
        )),
        *history_messages,
        HumanMessage(content=(
            f"USER QUESTION: {state['query']}\n\n"
            f"ANALYST KEY FACTS:\n{state.get('key_facts', '')}\n\n"
            f"SOURCES AVAILABLE: {state.get('source_citations', '')}\n\n"
            "Write the final answer now."
        ))
    ]

    response = llm.invoke(messages)
    draft = parse_gemini_content(response.content)

    return {**state, "draft_answer": draft}


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

    writer_draft = state.get("draft_answer", "")

    messages = [
        SystemMessage(content=(
            "You are a senior policy QA reviewer. You perform TWO checks:\n\n"
            "## CHECK 1 — FACT ACCURACY:\n"
            "For each fact in the draft, verify it exists in the source evidence chunks.\n"
            "- If a fact IS in the source evidence → it is VALID. Do not touch it.\n"
            "- If a fact is completely ABSENT from ALL source chunks → flag it for removal.\n\n"
            "## CHECK 1.2 — STRICT BULLET POINT STRUCTURE:\n"
            "The Writer is FORBIDDEN from writing introductory paragraphs, bridging paragraphs, or concluding paragraphs. The entire response MUST be a simple list of independent bullet points. If you see ANY conversational text, introductions, or conclusions outside of the bullet points, you MUST delete them entirely.\n\n"
            "## CHECK 1.5 — OVERREACH PREVENTION:\n"
            "Verify that the Writer did not draw conclusions, assume intent, or infer causal links that are not explicitly stated in the text. If the Writer inferred a connection that isn't literal, rewrite the answer to remove the inference.\n\n"
            "## CHECK 1.6 — NO CROSS-DOCUMENT SYNTHESIS:\n"
            "If the Writer attempts to build a unified argument by connecting a fact from Document A to a fact from Document B, that is an unauthorized synthesis! You MUST rewrite the answer to completely isolate the facts into unrelated, separate bullet points. Additionally, if the Writer adds any summarizing or concluding sentence at the end of a paragraph, delete it entirely.\n\n"
            "## CHECK 2 — COMPLETENESS:\n"
            "Did the Writer answer ALL parts of the user's question?\n"
            "If any part of the question was missed, add the missing information from the source evidence.\n\n"
            "## CRITICAL RULE — NO CROSS-DOCUMENT PANIC:\n"
            "Corporate policy documents share standard terms: '1 year recovery', 'clawback', "
            "'settling-in bonus', 'relocation bonus'. If Document A and Document B both mention "
            "'one year recovery' — this is completely NORMAL. Do NOT flag this as a hallucination. "
            "Only flag something if it is 100% absent from ALL source evidence chunks provided.\n\n"
            "## RESPONSE FORMAT (JSON only):\n"
            "{\"approved\": true, \"notes\": \"All facts verified. Answer is complete.\", "
            "\"final_answer\": \"<copy draft exactly>\"}\n"
            "OR if you found a genuine error OR a missing part:\n"
            "{\"approved\": false, \"notes\": \"<what specifically is wrong or missing>\", "
            "\"final_answer\": \"<corrected/completed version>\"}\n"
        )),
        HumanMessage(content=(
            f"USER QUESTION: {state['query']}\n\n"
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
            pass  # Parsing failed, defaults already set to Writer's draft

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
    
    prompt = f"""You are a senior Python Data Analyst. 
    DataFrames are available as variables: {df_vars}
    Schemas: 
    {schema_str}
    
    RECENT CHAT HISTORY:
    {history_str}
    
    USER QUERY: {state['query']}
    
    Write Python code to solve this. Assign your final string output to a variable named `final_answer`.
    CRITICAL RULE 1: Whenever working with dates or timestamps, you MUST convert the columns using `pd.to_datetime(col, errors='coerce', utc=True)` before performing any time-based operations to prevent formatting crashes.
    CRITICAL RULE 2: If searching for specific names, IDs, or string values, use safe pandas filtering (e.g. `.str.contains(..., na=False)`) instead of strict exact matches, and handle empty results gracefully by assigning "No matching records found" to `final_answer` instead of throwing an error.
    CRITICAL RULE 3: By default, you MUST analyze and cross-reference data across ALL provided DataFrames. If there are MULTIPLE DataFrames, explicitly mention their names in your `final_answer`. If there is only ONE DataFrame provided, do NOT mention its name. ONLY restrict your analysis to a single file if the user explicitly asks about that specific file.
    CRITICAL RULE 4: ABSOLUTELY DO NOT use the backtick character (`) ANYWHERE in your Python code. Not in strings, not in comments, not in replace() functions. The presence of any backtick inside your code will break the code parser and cause a fatal SyntaxError. Use standard quotes instead.
    CRITICAL RULE 5: DO NOT HALLUCINATE. If the user asks a theoretical question that cannot be calculated using the provided DataFrames, set `final_answer = 'ERROR: Cannot calculate this using the provided tables.'` Do NOT invent numbers or facts inside the python script.
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

    if len(state.get("uploaded_files", [])) > 1:
        file_rule = f"4. STRICT FILE CITATIONS ({', '.join(state['uploaded_files'])}): NEVER list these file names at the beginning of your answer. You MUST use inline citations for EVERY piece of data you provide (e.g., 'Revenue grew by 10% (Source: data.csv)'). Failure to cite the specific file for each metric is unacceptable.\n"
    elif len(state.get("uploaded_files", [])) == 1:
        file_rule = "4. EXPLICIT FILE NAMES: There is only one file uploaded. Do NOT mention its name in your answer.\n"
    else:
        file_rule = ""

    messages = [
        SystemMessage(content=(
            "You are a brilliant, highly intelligent Senior Data Scientist. When presented with raw mathematical results or data extracts, DO NOT just regurgitate the numbers like a mindless robot.\n\n"
            "## ADAPTIVE FORMATTING RULES:\n"
            "1. ADAPT TO THE QUERY: Do not use the exact same template for every answer. If the user asks a simple question (e.g., 'How many leads?'), provide a short, natural 1-2 sentence answer. If the question is complex (e.g., 'Analyze the funnel'), provide a deeper, structured breakdown.\n"
            "2. ORGANIC INSIGHTS: Provide business context and interpretation naturally woven into your response. Do not force a hardcoded 'Analyst Insight:' header at the bottom of every message.\n"
            "3. STRICT RELEVANCE: Answer ONLY what the user explicitly asked. Do NOT add unsolicited explanations, tutorials (like 'Why this works'), or business strategy unless specifically requested. If the user asks for code and output, give ONLY code and output.\n"
            f"{file_rule}"
            "5. NO BOILERPLATE: Never output 'Status: Active', file paths, column lists, or 'System Notes'.\n"
            "6. ZERO HALLUCINATION CONTRACT: You must NEVER invent, assume, or hallucinate numbers to fulfill a user's hypothetical scenario. If a result is 0 (e.g., 0 conversions, 0 sales), report exactly 0. If data is missing to answer a strategy question, state that the data is missing. Never fabricate statistics.\n"
            "7. DOCUMENT NAMES NOT TABLE NAMES: Never use terms like 'Table 1', 'Table 2', or 'the dataframe' in your final answer. Always refer to the exact source document name.\n"
            "8. ORIGINAL PDF CITAIONS: If a data source has an internal extracted name like 'Report_p2_table1.csv' or 'Report_table1', you MUST clean it up and cite the original document name instead (e.g. 'Report.pdf'). Do NOT expose internal '_table' or '_p2' artifacts to the user."
        )),
        *history_messages,
        HumanMessage(content=(
            f"{feedback_str}"
            f"USER QUESTION: {state['query']}\n\n"
            f"RAW DATA RESULTS:\n{raw_result}\n\n"
            "Write a direct, clean answer now."
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
            "You are a ruthless QA Critic for an Enterprise AI system. Your ONLY job is to verify that the Writer Agent did not hallucinate numbers or statistics.\n"
            "You will receive the RAW PYTHON OUTPUT and the WRITER'S DRAFT.\n"
            "1. Extract every single number, percentage, count, and statistic from the WRITER'S DRAFT.\n"
            "2. Cross-reference them against the RAW PYTHON OUTPUT.\n"
            "3. If the WRITER'S DRAFT contains ANY numbers or statistical claims that are NOT present in or immediately deducible from the RAW PYTHON OUTPUT, you must REJECT IT.\n\n"
            "Respond ONLY in valid JSON format:\n"
            "{\n"
            '  "is_hallucinated": boolean,\n'
            '  "feedback": "If hallucinated, list the exact numbers that are fake and tell the writer to remove them. If not hallucinated, leave empty."\n'
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
    Handles chit-chat and clarification when the user asks about something
    out of context (e.g. asking about PDFs when only CSVs are uploaded).
    """
    print("--> [Conversational Agent] Handling general/clarification query...")
    
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

    if len(state.get("uploaded_files", [])) > 1:
        file_rule = f"STRICT RULE FOR MULTIPLE FILES ({', '.join(state['uploaded_files'])}): NEVER list these file names at the beginning of your answer. You MUST use inline citations for EVERY piece of info you provide (e.g., 'You have 5 vacation days (Source: policy.docx)').\n"
    elif len(state.get("uploaded_files", [])) == 1:
        file_rule = "If the user's question relates to the uploaded files, do NOT mention the file name.\n"
    else:
        file_rule = ""

    messages = [
        SystemMessage(content=(
            "You are a helpful conversational AI assistant.\n"
            f"The user has currently uploaded the following files: {file_list_str}\n\n"
            f"{file_rule}"
            "If the user asks about a document type (like a PDF or Policy document) that they did NOT upload, "
            "but they DID upload other files (like CSVs), politely point this out.\n"
            "If the user's question is completely unrelated to the system, just respond conversationally and helpfully.\n"
            "Keep your responses concise and natural."
        )),
        *history_messages,
        HumanMessage(content=state['query'])
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
    graph.add_node("data_writer",   data_writer_node)
    graph.add_node("data_critic",   data_critic_node)
    graph.add_node("conversational", conversational_node)

    def route_after_router(state: PolicyState):
        """
        Since the LLM Router now has Universal Context Injection, 
        we trust it completely to pick the correct pipeline.
        We have removed the hardcoded file-type guards.
        """
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
    
    # Data Analysis path (with Critic Loop)
    graph.add_edge("data_analysis", "data_writer")
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

