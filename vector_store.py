"""
FAISS Vector Store for LangGraph Multi-Agent RAG System
Handles PDF loading, chunking, embedding, and similarity search.
"""
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pandas as pd

load_dotenv(override=True)

FAISS_INDEX_PATH = "./faiss_index"
DATA_DIR = "./documents"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "375"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def get_embeddings():
    """Return the local HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )


def _load_excel(file_path: Path) -> List[Document]:
    """Helper to load an Excel file as a list of LangChain Documents."""
    try:
        # Read all sheets; returns a dict {sheet_name: dataframe}
        sheets = pd.read_excel(file_path, sheet_name=None)
        docs = []
        for sheet_name, df in sheets.items():
            df = df.fillna("")
            content = f"--- Sheet: {sheet_name} ---\n" + df.to_string(index=False)
            docs.append(Document(page_content=content, metadata={"source": str(file_path)}))
        return docs
    except Exception as e:
        print(f"Error loading Excel file {file_path}: {e}")
        return []

def _load_csv_with_fallback(file_path: Path) -> List[Document]:
    """Helper to load a CSV file with fallback encodings to prevent RuntimeErrors."""
    try:
        loader = CSVLoader(str(file_path), autodetect_encoding=True)
        return loader.load()
    except Exception as e:
        print(f"Fallback: CSVLoader failed for {file_path}, trying pandas. Error: {e}")
        try:
            df = pd.read_csv(file_path, encoding_errors='replace')
            df = df.fillna("")
            docs = []
            for idx, row in df.iterrows():
                content = "\n".join([f"{col}: {val}" for col, val in row.items()])
                docs.append(Document(page_content=content, metadata={"source": str(file_path), "row": idx}))
            return docs
        except Exception as inner_e:
            print(f"Total failure loading CSV {file_path}: {inner_e}")
            return []

def load_and_chunk_docs(data_dir: str = DATA_DIR, session_id: str = None, api_key: str = None) -> List:
    """
    Load all files, run them through multimodal processor to extract tables/images,
    and split text into FAISS chunks.
    """
    path = Path(data_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_docs = []
    # Include docx and txt now
    files = list(path.glob("*.*"))

    if not files:
        return []

    from multimodal_processor import process_document

    for file_path in files:
        ext = file_path.suffix.lower()
        
        # We skip vector embeddings for tables, but the multimodal_processor
        # will convert PDF tables into CSVs in this folder. We don't want to
        # embed those CSVs either.
        if ext in [".csv", ".xlsx", ".xls"]:
            continue

        if ext not in [".pdf", ".docx", ".doc", ".txt"]:
            continue

        print(f"  Multimodal Processing: {file_path.name}")
        text_content = process_document(str(file_path), session_id or "default", api_key)
        
        if not text_content.strip():
            continue

        # Wrap raw text into a Langchain Document
        doc = Document(page_content=text_content, metadata={"source_file": file_path.name})
        chunks = splitter.split_documents([doc])
        
        all_docs.extend(chunks)
        print(f"  >> {len(chunks)} chunks from {file_path.name}")

    return all_docs


def build_faiss_index(docs: List, save_path: str = FAISS_INDEX_PATH):
    """
    Build a FAISS index from documents and save to disk.

    Args:
        docs: List of LangChain Document objects
        save_path: Directory path to save the FAISS index

    Returns:
        FAISS vector store instance
    """
    print("Building FAISS index...")
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(save_path)
    print(f"FAISS index saved to: {save_path}")
    return vector_store


def load_faiss_index(index_path: str = FAISS_INDEX_PATH):
    """
    Load an existing FAISS index from disk.

    Args:
        index_path: Path to the saved FAISS index

    Returns:
        FAISS vector store instance or None if not found
    """
    if not Path(index_path).exists():
        return None
    print("Loading existing FAISS index...")
    embeddings = get_embeddings()
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


def get_or_build_index(data_dir: str = DATA_DIR, index_path: str = FAISS_INDEX_PATH, session_id: str = None, api_key: str = None):
    """
    Load existing index or build a new one from documents.
    """
    existing = load_faiss_index(index_path)
    if existing:
        return existing

    docs = load_and_chunk_docs(data_dir, session_id, api_key)
    if not docs:
        print(f"No textual documents found for FAISS in '{data_dir}/'. Returning None.")
        return None

    return build_faiss_index(docs, index_path)


def search_documents(
    vector_store,
    query: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform semantic similarity search and return formatted results.

    Args:
        vector_store: FAISS vector store instance
        query: Search query string
        top_k: Number of results to return

    Returns:
        List of dicts with text, source, page, and score
    """
    if vector_store is None:
        return []

    results = vector_store.similarity_search_with_score(query, k=top_k)

    formatted = []
    for doc, score in results:
        formatted.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source_file", "Unknown"),
            "page": doc.metadata.get("page", 0) + 1,
            "score": float(score)
        })

    return formatted


def add_documents_to_index(
    vector_store,
    file_paths: List[str],
    session_id: str = None,
    api_key: str = None,
    index_path: str = FAISS_INDEX_PATH
):
    """
    Add new documents to an existing FAISS index.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    from multimodal_processor import process_document

    new_docs = []
    for path in file_paths:
        file_path = Path(path)
        ext = file_path.suffix.lower()
        
        if ext in [".csv", ".xlsx", ".xls"]:
            continue
            
        if ext not in [".pdf", ".docx", ".doc", ".txt"]:
            continue
            
        print(f"  Multimodal Processing (Add): {file_path.name}")
        text_content = process_document(str(file_path), session_id or "default", api_key)
        
        if not text_content.strip():
            continue
            
        doc = Document(page_content=text_content, metadata={"source_file": file_path.name})
        chunks = splitter.split_documents([doc])
        new_docs.extend(chunks)

    if new_docs:
        vector_store.add_documents(new_docs)
        vector_store.save_local(index_path)
    return vector_store
