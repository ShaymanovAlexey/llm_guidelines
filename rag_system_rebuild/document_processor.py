import re
import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Handles document ingestion, preprocessing, and chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        # Preprocess the text
        cleaned_text = self.preprocess_text(text)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(cleaned_text)
        
        # Add metadata to each chunk
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                'chunk_id': i,
                'chunk_size': len(chunk),
                'total_chunks': len(chunks)
            })
            
            chunk_docs.append({
                'text': chunk,
                'metadata': chunk_metadata
            })
        
        return chunk_docs
    
    def process_file(self, file_content: str, filename: str) -> List[Dict[str, Any]]:
        """Process a file and return chunked documents."""
        metadata = {
            'source': filename,
            'type': 'file'
        }
        return self.chunk_text(file_content, metadata)
    
    def process_text(self, text: str, source: str = "pasted_text") -> List[Dict[str, Any]]:
        """Process pasted text and return chunked documents."""
        metadata = {
            'source': source,
            'type': 'text'
        }
        return self.chunk_text(text, metadata)
    
    def process_pdf(self, file_content: bytes, filename: str) -> List[Dict[str, Any]]:
        """Process a PDF file and return chunked documents."""
        try:
            # Convert bytes to BytesIO for PDF reader
            from io import BytesIO
            pdf_stream = BytesIO(file_content)
            
            # Try to import PyPDF2 first, then fallback to pypdf
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(pdf_stream)
            except ImportError:
                try:
                    import pypdf
                    pdf_reader = pypdf.PdfReader(pdf_stream)
                except ImportError:
                    raise ImportError("Neither PyPDF2 nor pypdf is installed. Please install one of them: pip install PyPDF2 or pip install pypdf")
            
            # Extract text from all pages
            text_content = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
            
            if not text_content.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            # Create metadata
            metadata = {
                'source': filename,
                'type': 'pdf',
                'total_pages': len(pdf_reader.pages),
                'file_size': len(file_content)
            }
            
            # Process the extracted text
            return self.chunk_text(text_content, metadata)
            
        except Exception as e:
            raise Exception(f"Error processing PDF {filename}: {str(e)}")
    
    def process_file_by_extension(self, file_content: bytes, filename: str) -> List[Dict[str, Any]]:
        """Process a file based on its extension and return chunked documents."""
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension == '.pdf':
            return self.process_pdf(file_content, filename)
        elif file_extension in ['.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm']:
            # For text-based files, decode and process as text
            try:
                text_content = file_content.decode('utf-8')
                return self.process_file(text_content, filename)
            except UnicodeDecodeError:
                try:
                    text_content = file_content.decode('latin-1')
                    return self.process_file(text_content, filename)
                except UnicodeDecodeError:
                    raise ValueError(f"Could not decode file {filename} with UTF-8 or Latin-1 encoding")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}") 