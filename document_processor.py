"""
Document processing module for medical documents.
Handles PDF/text ingestion, chunking, and metadata extraction.
"""
import re
from typing import List, Dict, Optional
from datetime import datetime
import PyPDF2
import pdfplumber
from pathlib import Path
import tiktoken

import config


class MedicalDocument:
    """Represents a medical document with metadata"""
    
    def __init__(self, content: str, metadata: Dict):
        self.content = content
        self.metadata = metadata
        self.chunks = []
    
    def __repr__(self):
        return f"MedicalDocument(id={self.metadata.get('doc_id', 'unknown')}, pages={self.metadata.get('num_pages', 0)})"


class DocumentProcessor:
    """Process medical documents for Q&A system"""
    
    def __init__(self, chunk_size: int = config.CHUNK_SIZE, 
                 chunk_overlap: int = config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def load_pdf(self, file_path: str) -> MedicalDocument:
        """
        Load a PDF file and extract text with metadata.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            MedicalDocument object
        """
        file_path = Path(file_path)
        
        # Extract text using pdfplumber for better accuracy
        full_text = ""
        num_pages = 0
        
        try:
            with pdfplumber.open(file_path) as pdf:
                num_pages = len(pdf.pages)
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n\n"
        except Exception as e:
            print(f"Error reading PDF with pdfplumber: {e}")
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    num_pages = len(pdf_reader.pages)
                    for page in pdf_reader.pages:
                        full_text += page.extract_text() + "\n\n"
            except Exception as e2:
                print(f"Error reading PDF with PyPDF2: {e2}")
                raise
        
        # Extract metadata
        metadata = {
            'doc_id': file_path.stem,
            'filename': file_path.name,
            'file_path': str(file_path),
            'num_pages': num_pages,
            'extracted_date': datetime.now().isoformat(),
            'doc_type': self._infer_document_type(full_text)
        }
        
        # Try to extract dates from content
        dates = self._extract_dates(full_text)
        if dates:
            metadata['document_dates'] = dates
        
        return MedicalDocument(content=full_text, metadata=metadata)
    
    def load_text(self, file_path: str) -> MedicalDocument:
        """
        Load a plain text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            MedicalDocument object
        """
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            'doc_id': file_path.stem,
            'filename': file_path.name,
            'file_path': str(file_path),
            'extracted_date': datetime.now().isoformat(),
            'doc_type': self._infer_document_type(content)
        }
        
        dates = self._extract_dates(content)
        if dates:
            metadata['document_dates'] = dates
        
        return MedicalDocument(content=content, metadata=metadata)
    
    def chunk_document(self, document: MedicalDocument) -> List[Dict]:
        """
        Chunk a medical document into smaller pieces for embedding.
        Uses semantic chunking based on paragraphs when possible.
        
        Args:
            document: MedicalDocument to chunk
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        chunks = []
        content = document.content
        
        # Split by paragraphs first
        paragraphs = self._split_into_paragraphs(content)
        
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        for para in paragraphs:
            para_tokens = len(self.tokenizer.encode(para))
            
            # If adding this paragraph would exceed chunk size
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(self._create_chunk(
                    content=current_chunk,
                    index=chunk_index,
                    document=document
                ))
                chunk_index += 1
                
                # Start new chunk with overlap
                current_chunk = self._get_overlap_text(current_chunk) + para
                current_tokens = len(self.tokenizer.encode(current_chunk))
            else:
                # Add paragraph to current chunk
                current_chunk += "\n\n" + para if current_chunk else para
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                content=current_chunk,
                index=chunk_index,
                document=document
            ))
        
        document.chunks = chunks
        return chunks
    
    def _create_chunk(self, content: str, index: int, document: MedicalDocument) -> Dict:
        """Create a chunk dictionary with metadata"""
        return {
            'content': content.strip(),
            'chunk_index': index,
            'doc_id': document.metadata['doc_id'],
            'filename': document.metadata['filename'],
            'doc_type': document.metadata.get('doc_type', 'unknown'),
            'chunk_id': f"{document.metadata['doc_id']}_chunk_{index}"
        }
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines or section headers
        paragraphs = re.split(r'\n\s*\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get the last N tokens for overlap"""
        tokens = self.tokenizer.encode(text)
        overlap_tokens = tokens[-self.chunk_overlap:] if len(tokens) > self.chunk_overlap else tokens
        return self.tokenizer.decode(overlap_tokens)
    
    def _infer_document_type(self, content: str) -> str:
        """
        Infer the type of medical document from content.
        Simple keyword-based approach.
        """
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['discharge summary', 'hospital discharge']):
            return 'discharge_summary'
        elif any(word in content_lower for word in ['progress note', 'clinical note']):
            return 'progress_note'
        elif any(word in content_lower for word in ['lab result', 'laboratory']):
            return 'lab_results'
        elif any(word in content_lower for word in ['radiology', 'imaging', 'x-ray', 'mri', 'ct scan']):
            return 'radiology_report'
        elif any(word in content_lower for word in ['prescription', 'medication list']):
            return 'prescription'
        else:
            return 'medical_record'
    
    def _extract_dates(self, content: str) -> List[str]:
        """Extract dates from document content"""
        # Simple date pattern matching
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, content)
            dates.extend(matches)
        
        return list(set(dates))[:5]  # Return up to 5 unique dates
