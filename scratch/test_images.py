import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF
from multimodal_processor import process_document
from vector_store import build_faiss_index
from langchain_core.documents import Document as LC_Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agents_graph import build_graph, run_query

SESSION_ID = "image_test_777"
SESSION_DIR = Path("./session_docs") / SESSION_ID
SESSION_DIR.mkdir(parents=True, exist_ok=True)

def generate_image():
    print("--- Generating Test Image ---")
    img_path = SESSION_DIR / "dummy_diagram.png"
    img = Image.new('RGB', (400, 200), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10,10), "Project X Architecture", fill=(255,255,0))
    d.text((10,40), "Contains 3 main servers.", fill=(255,255,255))
    img.save(img_path)
    return img_path

def test_image_pipeline():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("FAIL: No API key found.")
        return

    img_path = generate_image()
    
    print("\n--- Processing Files (Triggering Vision Model) ---")
    text = process_document(str(img_path), SESSION_ID, api_key)
    
    print("\n[EXTRACTED TEXT FROM PDF + IMAGES]")
    print(text)
    
    # Build FAISS
    print("\n--- Building Vector Index ---")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc = LC_Document(page_content=text, metadata={"source": "image_test"})
    chunks = splitter.split_documents([doc])
    faiss_dir = Path("./faiss_index") / SESSION_ID
    vs = build_faiss_index(chunks, str(faiss_dir))
    
    app = build_graph(vs, SESSION_ID)
    
    query = "According to the architectural diagram in the document, how many main servers does Project X contain?"
    
    print(f"\n======================================")
    print(f"QUERY: {query}")
    try:
        for event in run_query(app, query, api_key=api_key, session_id=SESSION_ID):
            if "status" in event:
                print(f"STATUS: {event['status']}")
            elif "final_answer" in event:
                print("\nFINAL ANSWER:")
                print(event["final_answer"])
                print("\nPATH: ", " -> ".join(event.get("agent_chain", [])))
            elif "error" in event:
                print(f"PIPELINE ERROR CAUGHT: {event.get('final_answer')}")
    except Exception as e:
        print(f"CRITICAL CRASH: {e}")

if __name__ == "__main__":
    test_image_pipeline()
