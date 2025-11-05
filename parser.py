# parser.py
# Document parsing using PyMuPDF - extracts text from PDF, DOCX, etc.

import fitz  # PyMuPDF
from docx import Document
import os
from pathlib import Path


class DocumentParser:
    """
    Parser class using PyMuPDF to extract text from various document formats
    Supports: PDF, DOCX
    """
    
    def __init__(self, use_ocr=False, use_table_structure=True):
        """
        Initialize the DocumentParser
        
        Args:
            use_ocr (bool): Not used with PyMuPDF (kept for compatibility)
            use_table_structure (bool): Not used with PyMuPDF (kept for compatibility)
        """
        print(f"✅ DocumentParser initialized (using PyMuPDF)")
    
    
    def parse_document(self, file_path):
        """
        Parse a document and extract text
        
        Args:
            file_path (str or Path): Path to the document file
            
        Returns:
            dict: {
                'text': extracted text as string,
                'filename': original filename,
                'format': document format,
                'success': True/False,
                'error': error message if failed
            }
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                return {
                    'text': None,
                    'filename': str(file_path),
                    'format': None,
                    'success': False,
                    'error': f"File not found: {file_path}"
                }
            
            file_ext = Path(file_path).suffix.lower()
            filename = Path(file_path).name
            
            # Parse based on file type
            if file_ext == '.pdf':
                extracted_text = self._parse_pdf(file_path)
                doc_format = 'PDF'
            elif file_ext in ['.docx', '.doc']:
                extracted_text = self._parse_docx(file_path)
                doc_format = 'DOCX'
            else:
                return {
                    'text': None,
                    'filename': filename,
                    'format': None,
                    'success': False,
                    'error': f"Unsupported format: {file_ext}"
                }
            
            return {
                'text': extracted_text,
                'filename': filename,
                'format': doc_format,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'text': None,
                'filename': Path(file_path).name if file_path else 'unknown',
                'format': None,
                'success': False,
                'error': str(e)
            }
    
    
    def _parse_pdf(self, pdf_path):
        """Extract text from PDF using PyMuPDF"""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    
    
    def _parse_docx(self, docx_path):
        """Extract text from DOCX using python-docx"""
        doc = Document(docx_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    
    
    def parse_resume(self, resume_path):
        """
        Parse a resume document
        
        Args:
            resume_path: Path to resume file
            
        Returns:
            dict: Parsed result with text
        """
        result = self.parse_document(resume_path)
        
        if result['success']:
            print(f"✅ Resume parsed: {result['filename']}")
        else:
            print(f"❌ Resume parsing failed: {result['error']}")
        
        return result
    
    
    def parse_job_description(self, jd_path):
        """
        Parse a job description document
        
        Args:
            jd_path: Path to JD file
            
        Returns:
            dict: Parsed result with text
        """
        result = self.parse_document(jd_path)
        
        if result['success']:
            print(f"✅ JD parsed: {result['filename']}")
        else:
            print(f"❌ JD parsing failed: {result['error']}")
        
        return result
    
    
    def parse_multiple_resumes(self, resume_paths):
        """
        Parse multiple resume documents
        
        Args:
            resume_paths (list): List of paths to resume files
            
        Returns:
            list: List of parsed results
        """
        results = []
        
        print(f"\n📄 Parsing {len(resume_paths)} resumes...")
        
        for i, resume_path in enumerate(resume_paths, 1):
            print(f"[{i}/{len(resume_paths)}] Processing: {Path(resume_path).name}")
            result = self.parse_document(resume_path)
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        print(f"\n✅ Successfully parsed: {successful}/{len(resume_paths)}")
        if failed > 0:
            print(f"❌ Failed: {failed}")
        
        return results