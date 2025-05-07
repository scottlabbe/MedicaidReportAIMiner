import io
import logging
from enum import Enum
import fitz  # PyMuPDF

# Try to import unstructured if available
try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logging.warning("unstructured library not available. UNSTRUCTURED_FAST parser will not be available.")

class ParsingStrategy(Enum):
    CURRENT_METHOD = "Current Method"
    PYMUPDF_SIMPLE_TEXT = "PyMuPDF (Simple Text)"
    PYMUPDF_BLOCKS_SORTED = "PyMuPDF (Blocks Sorted)"
    PYMUPDF_WORDS_RECONSTRUCTED = "PyMuPDF (Words Reconstructed)"
    UNSTRUCTURED_FAST = "Unstructured.io (Fast)"
    # PDFPLUMBER_DEFAULT = "PDFPlumber (Default)"  # Add if implementing

    @classmethod
    def choices(cls):
        # Return list of tuples (enum_name, human_readable_name) for UI dropdowns
        choices = [(member.name, member.value) for member in cls]
        
        # Remove UNSTRUCTURED_FAST if library is not available
        if not UNSTRUCTURED_AVAILABLE and cls.UNSTRUCTURED_FAST.name in [choice[0] for choice in choices]:
            choices = [choice for choice in choices if choice[0] != cls.UNSTRUCTURED_FAST.name]
            
        return choices


def current_method_parser(pdf_io):
    """
    The current default method used in the application.
    This is a wrapper around our existing extract_text_from_pdf_memory function.
    
    Args:
        pdf_io: BytesIO object containing the PDF file
        
    Returns:
        str: Extracted text content
    """
    from utils.pdf_utils import extract_text_from_pdf_memory
    return extract_text_from_pdf_memory(pdf_io)


def pymupdf_simple_text_parser(pdf_io):
    """
    Extract text using PyMuPDF's simple text extraction.
    
    Args:
        pdf_io: BytesIO object containing the PDF file
        
    Returns:
        str: Extracted text content
    """
    try:
        text = ""
        pdf_io.seek(0)  # Reset pointer to beginning of file
        doc = fitz.open(stream=pdf_io.read(), filetype="pdf")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Use "text" mode to preserve reading order and line breaks
            page_text = page.get_text("text")
            text += f"PAGE {page_num + 1}\n{'-' * 40}\n{page_text}\n\n"
        
        doc.close()
        return text
    except Exception as e:
        logging.error(f"Error in pymupdf_simple_text_parser: {e}")
        raise ValueError(f"Failed to extract text with PyMuPDF simple text parser: {e}")


def pymupdf_blocks_sorted_parser(pdf_io):
    """
    Extract text using PyMuPDF's blocks extraction, sorted by position.
    
    Args:
        pdf_io: BytesIO object containing the PDF file
        
    Returns:
        str: Extracted text content
    """
    try:
        text = ""
        pdf_io.seek(0)  # Reset pointer to beginning of file
        doc = fitz.open(stream=pdf_io.read(), filetype="pdf")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get blocks - each block is a tuple: (x0, y0, x1, y1, "text", block_no, block_type)
            blocks = page.get_text("blocks")
            
            # Sort blocks first by y0 (top coordinate), then by x0 (left coordinate)
            # This attempts to reconstruct reading order
            sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
            
            page_text = ""
            for block in sorted_blocks:
                # Only add text blocks (block_type == 0), skip image blocks
                if block[6] == 0:  # Check if text block
                    page_text += block[4] + "\n"
            
            text += f"PAGE {page_num + 1}\n{'-' * 40}\n{page_text}\n\n"
        
        doc.close()
        return text
    except Exception as e:
        logging.error(f"Error in pymupdf_blocks_sorted_parser: {e}")
        raise ValueError(f"Failed to extract text with PyMuPDF blocks parser: {e}")


def pymupdf_words_reconstructed_parser(pdf_io):
    """
    Extract text using PyMuPDF's words extraction, reconstructing lines and paragraphs.
    
    Args:
        pdf_io: BytesIO object containing the PDF file
        
    Returns:
        str: Extracted text content
    """
    try:
        text = ""
        pdf_io.seek(0)  # Reset pointer to beginning of file
        doc = fitz.open(stream=pdf_io.read(), filetype="pdf")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get words - each word is a tuple: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            words = page.get_text("words")
            
            # Sort words by block, then by line, then by word number
            sorted_words = sorted(words, key=lambda w: (w[5], w[6], w[7]))
            
            # Reconstruct text with proper spacing
            page_text = ""
            last_block = -1
            last_line = -1
            
            for i, word in enumerate(sorted_words):
                block_no = word[5]
                line_no = word[6]
                
                # Handle block transitions (add extra newline)
                if block_no != last_block:
                    if last_block != -1:  # Not the first block
                        page_text += "\n\n"
                    last_block = block_no
                    last_line = line_no
                
                # Handle line transitions within a block (add newline)
                elif line_no != last_line:
                    page_text += "\n"
                    last_line = line_no
                
                # Add a space between words on the same line
                elif i > 0:
                    page_text += " "
                
                # Add the word
                page_text += word[4]
            
            text += f"PAGE {page_num + 1}\n{'-' * 40}\n{page_text}\n\n"
        
        doc.close()
        return text
    except Exception as e:
        logging.error(f"Error in pymupdf_words_reconstructed_parser: {e}")
        raise ValueError(f"Failed to extract text with PyMuPDF words reconstructed parser: {e}")


def unstructured_fast_parser(pdf_io):
    """
    Extract text using the unstructured library with fast strategy.
    
    Args:
        pdf_io: BytesIO object containing the PDF file
        
    Returns:
        str: Extracted text content
    """
    if not UNSTRUCTURED_AVAILABLE:
        raise ValueError("unstructured library is not available. Please install it to use this parser.")
    
    try:
        pdf_io.seek(0)  # Reset pointer to beginning of file
        elements = partition_pdf(file=pdf_io, strategy="fast")
        
        # Extract text from elements
        text = "\n\n".join([str(element) for element in elements])
        return text
    except Exception as e:
        logging.error(f"Error in unstructured_fast_parser: {e}")
        raise ValueError(f"Failed to extract text with unstructured.io fast parser: {e}")


# Mapping of parsing strategy names to parser functions
PARSER_FUNCTIONS = {
    ParsingStrategy.CURRENT_METHOD.name: current_method_parser,
    ParsingStrategy.PYMUPDF_SIMPLE_TEXT.name: pymupdf_simple_text_parser,
    ParsingStrategy.PYMUPDF_BLOCKS_SORTED.name: pymupdf_blocks_sorted_parser,
    ParsingStrategy.PYMUPDF_WORDS_RECONSTRUCTED.name: pymupdf_words_reconstructed_parser,
    ParsingStrategy.UNSTRUCTURED_FAST.name: unstructured_fast_parser,
}


def get_parser_function(parser_key):
    """
    Get the parser function for a given parser key.
    
    Args:
        parser_key: Name of the parser strategy (string from ParsingStrategy enum)
        
    Returns:
        function: The parser function
    """
    if parser_key not in PARSER_FUNCTIONS:
        raise ValueError(f"Unknown parser strategy: {parser_key}")
    
    # If it's the unstructured parser, check if it's available
    if parser_key == ParsingStrategy.UNSTRUCTURED_FAST.name and not UNSTRUCTURED_AVAILABLE:
        raise ValueError("unstructured library is not available. Please install it to use this parser.")
    
    return PARSER_FUNCTIONS[parser_key]