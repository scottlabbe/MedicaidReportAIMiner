import os
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

def save_uploaded_file(file, upload_folder):
    """
    Save an uploaded file to the specified folder.
    
    Args:
        file: FileStorage object from Flask's request.files
        upload_folder: Folder to save the file to
        
    Returns:
        tuple: (secure filename, file path, file size in bytes, file hash)
    """
    try:
        # Create secure filename
        filename = secure_filename(file.filename)
        
        # Ensure upload folder exists
        os.makedirs(upload_folder, exist_ok=True)
        
        # Create complete file path
        file_path = os.path.join(upload_folder, filename)
        
        # Save the file
        file.save(file_path)
        
        # Get file size in bytes
        file_size = os.path.getsize(file_path)
        
        # Calculate file hash
        file_hash = get_file_hash(file_path)
        
        return (filename, file_path, file_size, file_hash)
    except Exception as e:
        logging.error(f"Error saving uploaded file: {e}")
        raise ValueError(f"Failed to save uploaded file: {e}")
