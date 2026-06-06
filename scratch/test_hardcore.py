import os
import time
from pathlib import Path
import pandas as pd
from fpdf import FPDF

# 1. Setup mock session
SESSION_ID = "hardcore_test_999"
SESSION_DIR = Path("./session_docs") / SESSION_ID
SESSION_DIR.mkdir(parents=True, exist_ok=True)

print("--- 1. Generating Mock Data ---")
# Create a PDF with a Policy Rule, a Data Table, and an Image
class MyPDF(FPDF):
    pass

pdf = MyPDF()
pdf.add_page()
pdf.set_font("Arial", size=15)
pdf.cell(200, 10, txt="Corporate Expense Policy", ln=1, align='C')
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Rule 1: The absolute maximum allowed travel expense is $500.", ln=1)
pdf.cell(200, 10, txt="Rule 2: Anyone exceeding this limit is marked as VIOLATION.", ln=1)
pdf.ln(10)

# Add a table to the PDF using standard strings (pdfplumber can usually parse basic aligned text, or we can use fpdf cell grid)
pdf.cell(200, 10, txt="Below is the Internal Audit Table:", ln=1)
pdf.set_font("Courier", size=12)
pdf.cell(50, 10, txt="Employee", border=1)
pdf.cell(50, 10, txt="Department", border=1)
pdf.cell(50, 10, txt="Expense", border=1, ln=1)

data = [
    ("Alice", "Sales", "450"),
    ("Bob", "IT", "600"),
    ("Charlie", "HR", "300")
]
for row in data:
    pdf.cell(50, 10, txt=row[0], border=1)
    pdf.cell(50, 10, txt=row[1], border=1)
    pdf.cell(50, 10, txt=row[2], border=1, ln=1)

pdf_path = SESSION_DIR / "policy_audit.pdf"
pdf.output(str(pdf_path))

# Create an external CSV
csv_data = pd.DataFrame({
    "Employee": ["Dave", "Eve", "Frank"],
    "Department": ["Sales", "Sales", "IT"],
    "Expense": [550, 200, 800]
})
csv_path = SESSION_DIR / "external_expenses.csv"
csv_data.to_csv(csv_path, index=False)

print(f"Generated: {pdf_path.name}")
print(f"Generated: {csv_path.name}")


print("\n--- 2. Running Multimodal Processor ---")
from multimodal_processor import process_document
from vector_store import build_faiss_index
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Process PDF
text_content = process_document(str(pdf_path), SESSION_ID, os.getenv("GOOGLE_API_KEY"))
print(f"Extracted Text from PDF:\n{text_content[:200]}...\n")

# Check if tables were extracted
csvs = list(SESSION_DIR.glob("*.csv"))
print(f"CSVs now in session directory: {[f.name for f in csvs]}")

# Build FAISS index for the text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc = Document(page_content=text_content, metadata={"source_file": pdf_path.name})
chunks = splitter.split_documents([doc])
faiss_dir = Path("./faiss_index") / SESSION_ID
vs = build_faiss_index(chunks, str(faiss_dir))

print("\n--- 3. Running Hybrid Multi-Agent Query ---")
from agents_graph import build_graph, run_query

app = build_graph(vs, SESSION_ID)

query = "Search the documents to find the maximum allowed travel expense limit. Then, look at ALL the expense data (both internal and external) and tell me the names of every employee who violated this limit."

print(f"User Query: {query}")
print("Executing LangGraph...\n")

for event in run_query(app, query, session_id=SESSION_ID):
    if "status" in event:
        print(f"STATUS: {event['status']}")
    elif "final_answer" in event:
        print("\n" + "="*50)
        print("FINAL ANSWER:")
        print(event["final_answer"])
        print("\nExecution Path:")
        print(" -> ".join(event.get("agent_chain", [])))
        print("="*50)
    else:
        print(f"Raw State Result: {event.keys()}")
