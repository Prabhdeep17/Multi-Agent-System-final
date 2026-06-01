"""
PDF Processing Module for Multi-Agent PDF Analysis System

Handles:
- PDF text extraction with page tracking
- Intelligent text chunking with overlap
- Document metadata management
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config


@dataclass
class DocumentMetadata:
    """Metadata for document chunks"""
    doc_name: str
    page_number: int
    chunk_id: int
    total_chunks: int
    chunk_text: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "total_chunks": self.total_chunks
        }


class PDFExtractor:
    """Extract text from PDFs with page tracking"""
    
    @staticmethod
    def extract_text_with_pages(pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with page numbers
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dicts with 'page_number' and 'text' keys
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        pages_data = []
        
        try:
            doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():  # Only include pages with text
                    pages_data.append({
                        "page_number": page_num + 1,  # 1-indexed for user display
                        "text": text
                    })
            
            doc.close()
            
        except Exception as e:
            raise RuntimeError(f"Error extracting text from {pdf_path}: {str(e)}")
        
        if not pages_data:
            raise ValueError(f"No text content found in {pdf_path}")
        
        return pages_data
    
    @staticmethod
    def get_pdf_info(pdf_path: str) -> Dict[str, Any]:
        """Get PDF metadata (page count, title, etc.)"""
        doc = fitz.open(pdf_path)
        info = {
            "page_count": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "file_name": Path(pdf_path).name
        }
        doc.close()
        return info


class IntelligentChunker:
    """Split text into semantic chunks with metadata"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize chunker
        
        Args:
            chunk_size: Size of each chunk (defaults to Config.CHUNK_SIZE)
            chunk_overlap: Overlap between chunks (defaults to Config.CHUNK_OVERLAP)
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_document(
        self,
        pages_data: List[Dict[str, Any]],
        doc_name: str
    ) -> List[DocumentMetadata]:
        """
        Chunk document pages into smaller segments with metadata
        
        Args:
            pages_data: List of dicts with page_number and text
            doc_name: Name of the document
            
        Returns:
            List of DocumentMetadata objects
        """
        all_chunks = []
        chunk_counter = 0
        
        # Process each page
        for page_data in pages_data:
            page_num = page_data["page_number"]
            page_text = page_data["text"]
            
            # Split page text into chunks
            page_chunks = self.text_splitter.split_text(page_text)
            
            for chunk_text in page_chunks:
                chunk_metadata = DocumentMetadata(
                    doc_name=doc_name,
                    page_number=page_num,
                    chunk_id=chunk_counter,
                    total_chunks=0,  # Will be updated after all chunks are created
                    chunk_text=chunk_text
                )
                all_chunks.append(chunk_metadata)
                chunk_counter += 1
        
        # Update total_chunks for all chunks
        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.total_chunks = total
        
        return all_chunks
    
    def chunk_multiple_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, List[DocumentMetadata]]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of dicts with 'name' and 'pages_data' keys
            
        Returns:
            Dict mapping doc_name to list of DocumentMetadata
        """
        all_doc_chunks = {}
        
        for doc in documents:
            doc_name = doc["name"]
            pages_data = doc["pages_data"]
            
            chunks = self.chunk_document(pages_data, doc_name)
            all_doc_chunks[doc_name] = chunks
        
        return all_doc_chunks


class PDFProcessor:
    """High-level PDF processing pipeline"""
    
    def __init__(self):
        """Initialize PDF processor"""
        self.extractor = PDFExtractor()
        self.chunker = IntelligentChunker()
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict with pdf_info, pages_data, and chunks
        """
        pdf_path = Path(pdf_path)
        doc_name = pdf_path.stem
        
        # Extract text with pages
        pages_data = self.extractor.extract_text_with_pages(str(pdf_path))
        
        # Get PDF info
        pdf_info = self.extractor.get_pdf_info(str(pdf_path))
        
        # Chunk the document
        chunks = self.chunker.chunk_document(pages_data, doc_name)
        
        return {
            "doc_name": doc_name,
            "pdf_info": pdf_info,
            "pages_data": pages_data,
            "chunks": chunks
        }
    
    def process_multiple_pdfs(
        self,
        pdf_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple PDF files
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of processing results for each PDF
        """
        results = []
        
        for pdf_path in pdf_paths:
            try:
                result = self.process_pdf(pdf_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
                # Continue with other PDFs
                continue
        
        return results
