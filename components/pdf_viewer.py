"""
PDF Viewer Component for Multi-Agent PDF Analysis System

Handles:
- PDF display in Streamlit
- Citation highlighting and navigation
- Page-by-page PDF rendering
"""
import streamlit as st
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from PIL import Image
import io


class PDFViewer:
    """PDF viewer with citation highlighting capabilities"""
    
    def __init__(self):
        """Initialize PDF viewer"""
        pass
    
    @staticmethod
    def pdf_to_base64(pdf_path: str) -> str:
        """
        Convert PDF to base64 for embedding
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Base64 encoded PDF
        """
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        return base64.b64encode(pdf_bytes).decode('utf-8')
    
    @staticmethod
    def render_pdf_with_pdfjs(pdf_path: str, page_number: int = 1):
        """
        Render PDF using PDF.js in an iframe
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page to display (1-indexed)
        """
        pdf_base64 = PDFViewer.pdf_to_base64(pdf_path)
        
        # Create PDF.js viewer HTML
        pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{pdf_base64}#page={page_number}"
            width="100%"
            height="800px"
            type="application/pdf"
            style="border: 1px solid #ccc; border-radius: 5px;"
        </iframe>
        """
        
        st.markdown(pdf_display, unsafe_allow_html=True)
    
    @staticmethod
    def convert_pdf_page_to_image(pdf_path: str, page_number: int) -> Image:
        """
        Convert PDF page to image
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)
            
        Returns:
            PIL Image object
        """
        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]  # Convert to 0-indexed
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        
        # Convert to PIL Image
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        
        doc.close()
        
        return img
    
    @staticmethod
    def render_pdf_as_images(pdf_path: str, page_number: int = 1):
        """
        Render PDF page as image (alternative to PDF.js)
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page to display (1-indexed)
        """
        try:
            img = PDFViewer.convert_pdf_page_to_image(pdf_path, page_number)
            st.image(img, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering PDF page: {str(e)}")
    
    @staticmethod
    def display_citation_navigator(
        evidence_list: List[Dict[str, Any]],
        pdf_paths: Dict[str, str]
    ):
        """
        Display citation navigator with clickable citations
        
        Args:
            evidence_list: List of evidence dicts with metadata
            pdf_paths: Dict mapping doc_name to file path
        """
        if not evidence_list:
            st.info("No citations available")
            return
        
        st.subheader("📚 Citations & Evidence")
        
        for idx, evidence in enumerate(evidence_list, 1):
            doc_name = evidence.get('doc_name', 'Unknown')
            page_num = evidence.get('page_number', 0)
            chunk_id = evidence.get('chunk_id', 0)
            similarity = evidence.get('similarity_score', 0)
            text = evidence.get('text', '')
            
            # Create expander for each citation
            with st.expander(
                f"[{idx}] {doc_name} - Page {page_num} (Score: {similarity:.3f})",
                expanded=(idx == 1)  # Expand first citation by default
            ):
                st.markdown(f"**Document:** {doc_name}")
                st.markdown(f"**Page:** {page_num}")
                st.markdown(f"**Chunk ID:** {chunk_id}")
                st.markdown(f"**Similarity Score:** {similarity:.3f}")
                st.markdown("**Evidence Text:**")
                st.text_area(
                    "Evidence",
                    value=text,
                    height=150,
                    key=f"evidence_{idx}",
                    label_visibility="collapsed"
                )
                
                # Add button to view in PDF
                if doc_name in pdf_paths:
                    if st.button(f"View in PDF 📄", key=f"view_pdf_{idx}"):
                        st.session_state.active_pdf = pdf_paths[doc_name]
                        st.session_state.active_page = page_num
                        st.rerun()
    
    @staticmethod
    def display_pdf_viewer(pdf_path: str, page_number: int = 1, use_images: bool = True):
        """
        Display PDF viewer
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page to display
            use_images: If True, use image rendering; else use PDF.js
        """
        if not Path(pdf_path).exists():
            st.error(f"PDF file not found: {pdf_path}")
            return
        
        st.subheader(f"📄 {Path(pdf_path).stem}")
        
        # Get total pages
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        # Page navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            page_num = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=page_number,
                key="pdf_page_input"
            )
        
        # Render PDF
        if use_images:
            PDFViewer.render_pdf_as_images(pdf_path, page_num)
        else:
            PDFViewer.render_pdf_with_pdfjs(pdf_path, page_num)
        
        st.caption(f"Page {page_num} of {total_pages}")
