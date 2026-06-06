import os
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
import base64
from io import BytesIO

# Try importing image processing
try:
    from PIL import Image
except ImportError:
    Image = None

# Try importing PDF libraries
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# Try importing DOCX
try:
    from docx import Document
except ImportError:
    Document = None

# ── Gemini Vision Helper ──────────────────────────────────────────────────────

def _describe_image_with_gemini(pil_img, api_key: str) -> str:
    """Sends a PIL image to Gemini Vision to get a text description."""
    if not api_key:
        return "[Image detected, but no API key provided to analyze it]"
        
    try:
        from langchain_core.messages import HumanMessage
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Convert PIL to base64
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=api_key,
            temperature=0.2,
            max_retries=1
        )
        
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image in detail. If it is a chart or graph, extract the key data points and trends. If it contains text, OCR it."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
            ]
        )
        
        response = llm.invoke([msg])
        content = response.content
        if isinstance(content, list):
            # Extract text blocks if it's a list
            text_blocks = [item['text'] for item in content if item.get('type') == 'text']
            return "\n".join(text_blocks)
        return str(content)
    except Exception as e:
        print(f"Error analyzing image with Gemini: {e}")
        return f"[Image analysis failed: {str(e)}]"

# ── Processing Engines ────────────────────────────────────────────────────────

def _process_pdf(file_path: Path, session_dir: Path, api_key: str) -> str:
    """Extracts text, saves tables to CSVs, and describes images from a PDF."""
    full_text = []
    
    # 1. Extract Tables using pdfplumber
    if pdfplumber:
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables, 1):
                        if not table: continue
                        # Clean empty rows
                        clean_table = [[str(cell).strip() if cell else "" for cell in row] for row in table]
                        # Only save if it looks like a real table (more than 1 row, multiple cols)
                        if len(clean_table) > 1 and len(clean_table[0]) > 1:
                            df = pd.DataFrame(clean_table[1:], columns=clean_table[0])
                            csv_name = f"{file_path.stem}_p{page_num}_table{table_idx}.csv"
                            csv_path = session_dir / csv_name
                            df.to_csv(csv_path, index=False)
                            full_text.append(f"\n[A data table was extracted from page {page_num} and saved as {csv_name} for Pandas analysis]\n")
        except Exception as e:
            print(f"pdfplumber error on {file_path.name}: {e}")
            
    # 2. Extract Text and Images using PyMuPDF (fitz)
    if fitz and Image:
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Text
                text = page.get_text()
                if text.strip():
                    full_text.append(text)
                    
                # Images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list, 1):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
                        # Ignore tiny images (icons, lines)
                        if pil_img.width > 100 and pil_img.height > 100:
                            desc = _describe_image_with_gemini(pil_img, api_key)
                            full_text.append(f"\n[Image {img_index} from Page {page_num+1} Description: {desc}]\n")
                    except Exception as img_e:
                        print(f"Error extracting image {img_index} on page {page_num+1}: {img_e}")
        except Exception as e:
            print(f"PyMuPDF error on {file_path.name}: {e}")
            
    return "\n".join(full_text)

def _process_docx(file_path: Path, session_dir: Path, api_key: str) -> str:
    """Extracts text and saves tables from Word documents."""
    full_text = []
    if not Document:
        return ""
        
    try:
        doc = Document(file_path)
        
        # Paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
                
        # Tables -> CSV
        for t_idx, table in enumerate(doc.tables, 1):
            data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                data.append(row_data)
                
            if len(data) > 1:
                df = pd.DataFrame(data[1:], columns=data[0])
                csv_name = f"{file_path.stem}_table{t_idx}.csv"
                csv_path = session_dir / csv_name
                df.to_csv(csv_path, index=False)
                full_text.append(f"\n[A data table was extracted and saved as {csv_name} for Pandas analysis]\n")
                
    except Exception as e:
        print(f"python-docx error on {file_path.name}: {e}")
        
    return "\n".join(full_text)

def _process_txt(file_path: Path) -> str:
    """Reads raw text from TXT files."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"txt read error on {file_path.name}: {e}")
        return ""

# ── Main Entrypoint ───────────────────────────────────────────────────────────

def process_document(file_path_str: str, session_id: str, api_key: str) -> str:
    """
    Analyzes the document, extracts tables to CSVs inside the session folder,
    extracts images and summarizes them via Gemini, and returns the final
    assembled text content (including image descriptions) to be used by RAG.
    """
    file_path = Path(file_path_str)
    session_dir = Path("./session_docs") / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    
    ext = file_path.suffix.lower()
    
    if ext in [".png", ".jpg", ".jpeg", ".webp"]:
        if Image:
            try:
                pil_img = Image.open(file_path).convert("RGB")
                desc = _describe_image_with_gemini(pil_img, api_key)
                return f"[Image File Description: {desc}]\n"
            except Exception as e:
                print(f"Error processing image file: {e}")
                return ""
        return ""
    elif ext == ".pdf":
        return _process_pdf(file_path, session_dir, api_key)
    elif ext in [".docx", ".doc"]:
        return _process_docx(file_path, session_dir, api_key)
    elif ext == ".txt":
        return _process_txt(file_path)
    elif ext in [".csv", ".xlsx", ".xls"]:
        # Do not extract text from CSV/Excel, leave them fully to the Pandas Agent
        return ""
    else:
        return ""
