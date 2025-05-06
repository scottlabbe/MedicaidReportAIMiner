import os
import io
import hashlib
import logging
import PyPDF2
from werkzeug.utils import secure_filename

def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text content
    """
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n---- Page Break ----\n\n"
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        raise ValueError(f"Failed to extract text from PDF: {e}")

def get_file_hash(file_path):
    """
    Calculate SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: SHA-256 hash of the file
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def extract_text_from_pdf_memory(pdf_io):
    """
    Extract text content from a PDF file in memory.
    
    Args:
        pdf_io: BytesIO object containing the PDF file
        
    Returns:
        str: Extracted text content
    """
    try:
        text = ""
        pdf_io.seek(0)  # Reset pointer to beginning of file
        pdf_reader = PyPDF2.PdfReader(pdf_io)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n---- Page Break ----\n\n"
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF in memory: {e}")
        raise ValueError(f"Failed to extract text from PDF in memory: {e}")

def extract_keywords_from_pdf_metadata(pdf_path):
    """
    Extract keywords from PDF metadata.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        list: List of keywords found in PDF metadata
    """
    keywords = []
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if pdf_reader.metadata and hasattr(pdf_reader.metadata, 'get'):
                # Get keywords from metadata - they can be stored under different names
                for key in ['/Keywords', '/Subject']:
                    keyword_str = pdf_reader.metadata.get(key, '')
                    if keyword_str:
                        # Keywords can be comma or semicolon-separated
                        for sep in [',', ';']:
                            if sep in keyword_str:
                                # Split, strip and add non-empty keywords to the list
                                for kw in keyword_str.split(sep):
                                    kw = kw.strip()
                                    if kw:
                                        keywords.append(kw)
                                break
                        else:
                            # If no separator found, add the entire string as a single keyword
                            keywords.append(keyword_str.strip())
        
        return [kw for kw in keywords if kw]  # Return non-empty keywords
    except Exception as e:
        logging.error(f"Error extracting keywords from PDF metadata: {e}")
        return []  # Return empty list on error
        
def extract_keywords_from_pdf_metadata_memory(pdf_io):
    """
    Extract keywords from PDF metadata in memory.
    
    Args:
        pdf_io: BytesIO object containing the PDF file
        
    Returns:
        list: List of keywords found in PDF metadata
    """
    keywords = []
    try:
        pdf_io.seek(0)  # Reset pointer to beginning of file
        pdf_reader = PyPDF2.PdfReader(pdf_io)
        if pdf_reader.metadata and hasattr(pdf_reader.metadata, 'get'):
            # Get keywords from metadata - they can be stored under different names
            for key in ['/Keywords', '/Subject']:
                keyword_str = pdf_reader.metadata.get(key, '')
                if keyword_str:
                    # Keywords can be comma or semicolon-separated
                    for sep in [',', ';']:
                        if sep in keyword_str:
                            # Split, strip and add non-empty keywords to the list
                            for kw in keyword_str.split(sep):
                                kw = kw.strip()
                                if kw:
                                    keywords.append(kw)
                            break
                    else:
                        # If no separator found, add the entire string as a single keyword
                        keywords.append(keyword_str.strip())
    
        return [kw for kw in keywords if kw]  # Return non-empty keywords
    except Exception as e:
        logging.error(f"Error extracting keywords from PDF metadata in memory: {e}")
        return []  # Return empty list on error

def process_keywords(pdf_keywords, ai_keywords):
    """
    Process and combine keywords from PDF metadata and AI extraction.
    
    Args:
        pdf_keywords: List of keywords from PDF metadata
        ai_keywords: List of keywords from AI extraction
        
    Returns:
        list: Combined, deduplicated list of keywords
    """
    # Combine keywords
    combined_keywords = pdf_keywords + ai_keywords
    
    # Normalize case and strip whitespace
    normalized_keywords = [kw.lower().strip() for kw in combined_keywords if kw]
    
    # Remove duplicates while preserving order
    unique_keywords = []
    for kw in normalized_keywords:
        if kw and kw not in unique_keywords:
            unique_keywords.append(kw)
    
    return unique_keywords

def process_uploaded_file_memory(file):
    """
    Process an uploaded file in memory without saving to disk.
    
    Args:
        file: FileStorage object from Flask's request.files
        
    Returns:
        tuple: (secure filename, file size in bytes, file hash, file content)
    """
    try:
        # Create secure filename
        filename = secure_filename(file.filename)
        
        # Read file content
        file_content = file.read()
        
        # Get file size in bytes
        file_size = len(file_content)
        
        # Calculate file hash
        sha256_hash = hashlib.sha256()
        sha256_hash.update(file_content)
        file_hash = sha256_hash.hexdigest()
        
        return (filename, file_size, file_hash, file_content)
    except Exception as e:
        logging.error(f"Error processing uploaded file: {e}")
        raise ValueError(f"Failed to process uploaded file: {e}")
