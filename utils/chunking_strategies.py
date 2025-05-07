import logging
import re
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal, Callable
import tiktoken
from pydantic import BaseModel, Field

# Try importing LlamaIndex components
try:
    from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter
    from llama_index.core.node_parser import MarkdownHeaderTextSplitter
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from llama_index.core.schema import Document
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    logging.warning("LlamaIndex not available. Some chunking strategies will not be available.")
    LLAMA_INDEX_AVAILABLE = False

# Define the Chunk model
class Chunk(BaseModel):
    """
    Represents a chunk of text with metadata and statistics.
    """
    chunk_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    char_count: int
    token_count: int
    chunk_id: Optional[str] = None

# Parameter models for different chunking strategies
class SimpleSplitterParams(BaseModel):
    """Parameters for simple recursive text splitter."""
    strategy_type: Literal["SIMPLE_RECURSIVE_SPLITTER"] = "SIMPLE_RECURSIVE_SPLITTER"
    chunk_size: int = Field(default=512, gt=0, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=50, ge=0, description="Number of overlapping tokens between chunks")
    split_method: Literal["token", "sentence"] = Field(
        default="token", 
        description="Split by token count or by sentences"
    )

class SemanticSplitterParams(BaseModel):
    """Parameters for semantic text splitter."""
    strategy_type: Literal["SEMANTIC_CHUNKING_LLAMAINDEX"] = "SEMANTIC_CHUNKING_LLAMAINDEX"
    chunk_size: int = Field(default=512, gt=0, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=50, ge=0, description="Number of overlapping tokens between chunks")
    breakpoint_percentile_threshold: int = Field(
        default=95, ge=0, le=100,
        description="Percentile threshold for determining breakpoints"
    )

class MarkdownSplitterParams(BaseModel):
    """Parameters for Markdown header text splitter."""
    strategy_type: Literal["MARKDOWN_HEADER_SPLITTER"] = "MARKDOWN_HEADER_SPLITTER"
    chunk_size: int = Field(default=512, gt=0, description="Target chunk size in tokens")
    headers_to_split_on: List[tuple] = Field(
        default=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ],
        description="List of header tuples (markdown_header, display_name) to split on"
    )

# Chunking Strategy Enum
class ChunkingStrategy(Enum):
    """
    Enum of available chunking strategies.
    Each strategy has a display name and associated parameter model.
    """
    SIMPLE_RECURSIVE_SPLITTER = ("Simple Recursive Splitter (LlamaIndex)", SimpleSplitterParams)
    SEMANTIC_CHUNKING_LLAMAINDEX = ("Semantic Chunking (LlamaIndex)", SemanticSplitterParams)
    MARKDOWN_HEADER_SPLITTER = ("Markdown Header Splitter", MarkdownSplitterParams)

    def __init__(self, display_name, param_model_cls):
        self._display_name = display_name
        self._param_model_cls = param_model_cls

    @property
    def display_name(self):
        return self._display_name

    @property
    def param_model(self):
        return self._param_model_cls

    @classmethod
    def choices(cls):
        """Return list of (enum_name, display_name) tuples for UI dropdowns."""
        choices = [(member.name, member._display_name) for member in cls]
        
        # If LlamaIndex is not available, remove strategies that require it
        if not LLAMA_INDEX_AVAILABLE:
            choices = [choice for choice in choices if not choice[0].endswith("LLAMAINDEX")]
            
        return choices
            
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model: The model to use for token counting
        
    Returns:
        int: The number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logging.warning(f"Failed to count tokens with tiktoken: {e}")
        # Fallback to approximate counting (very crude estimation)
        return len(text.split())

def chunk_with_simple_recursive(text: str, params: SimpleSplitterParams) -> List[Chunk]:
    """
    Chunk text using LlamaIndex's recursive text splitter.
    
    Args:
        text: Text to chunk
        params: Parameters for the chunking strategy
        
    Returns:
        List[Chunk]: List of generated chunks
    """
    if not LLAMA_INDEX_AVAILABLE:
        raise ValueError("LlamaIndex is not available. Cannot use this chunking strategy.")
    
    chunks = []
    
    try:
        # Create a Document from the text
        document = Document(text=text)
        
        # Choose the appropriate splitter based on the split method
        if params.split_method == "token":
            splitter = TokenTextSplitter(
                chunk_size=params.chunk_size,
                chunk_overlap=params.chunk_overlap
            )
        else:  # sentence
            splitter = SentenceSplitter(
                chunk_size=params.chunk_size,
                chunk_overlap=params.chunk_overlap
            )
        
        # Split the document into nodes
        nodes = splitter.get_nodes_from_documents([document])
        
        # Convert nodes to Chunk objects
        for i, node in enumerate(nodes):
            chunk_text = node.text
            
            # Extract existing metadata from the node
            metadata = dict(node.metadata) if hasattr(node, 'metadata') else {}
            
            # Add additional metadata
            metadata.update({
                "chunk_index": i,
                "splitter": "simple_recursive",
                "split_method": params.split_method
            })
            
            # Count tokens and characters
            token_count = count_tokens(chunk_text)
            char_count = len(chunk_text)
            
            # Create Chunk object
            chunk = Chunk(
                chunk_text=chunk_text,
                metadata=metadata,
                char_count=char_count,
                token_count=token_count,
                chunk_id=f"simple-{i}"
            )
            
            chunks.append(chunk)
            
    except Exception as e:
        logging.error(f"Error in simple recursive chunking: {e}")
        raise ValueError(f"Failed to chunk text with simple recursive splitter: {e}")
    
    return chunks

def chunk_with_semantic(text: str, params: SemanticSplitterParams) -> List[Chunk]:
    """
    Chunk text using LlamaIndex's semantic splitter.
    
    Args:
        text: Text to chunk
        params: Parameters for the chunking strategy
        
    Returns:
        List[Chunk]: List of generated chunks
    """
    if not LLAMA_INDEX_AVAILABLE:
        raise ValueError("LlamaIndex is not available. Cannot use this chunking strategy.")
    
    chunks = []
    
    try:
        # Create a Document from the text
        document = Document(text=text)
        
        # Create semantic splitter
        # Note: This will use OpenAI embeddings by default
        splitter = SemanticSplitterNodeParser(
            buffer_size=params.chunk_size,
            breakpoint_percentile_threshold=params.breakpoint_percentile_threshold / 100.0,
            # Use text-embedding-ada-002 or similar model
        )
        
        # Split the document into nodes
        nodes = splitter.get_nodes_from_documents([document])
        
        # Convert nodes to Chunk objects
        for i, node in enumerate(nodes):
            chunk_text = node.text
            
            # Extract existing metadata from the node
            metadata = dict(node.metadata) if hasattr(node, 'metadata') else {}
            
            # Add additional metadata
            metadata.update({
                "chunk_index": i,
                "splitter": "semantic",
                "breakpoint_percentile": params.breakpoint_percentile_threshold
            })
            
            # Count tokens and characters
            token_count = count_tokens(chunk_text)
            char_count = len(chunk_text)
            
            # Create Chunk object
            chunk = Chunk(
                chunk_text=chunk_text,
                metadata=metadata,
                char_count=char_count,
                token_count=token_count,
                chunk_id=f"semantic-{i}"
            )
            
            chunks.append(chunk)
            
    except Exception as e:
        logging.error(f"Error in semantic chunking: {e}")
        raise ValueError(f"Failed to chunk text with semantic splitter: {e}")
    
    return chunks

def chunk_with_markdown_header(text: str, params: MarkdownSplitterParams) -> List[Chunk]:
    """
    Chunk text using LlamaIndex's Markdown header splitter.
    
    Args:
        text: Text to chunk
        params: Parameters for the chunking strategy
        
    Returns:
        List[Chunk]: List of generated chunks
    """
    if not LLAMA_INDEX_AVAILABLE:
        raise ValueError("LlamaIndex is not available. Cannot use this chunking strategy.")
    
    chunks = []
    
    try:
        # Convert headers_to_split_on to the format expected by MarkdownHeaderTextSplitter
        # (list of tuples like [(#, "Header 1"), (##, "Header 2")])
        headers_to_split_on = params.headers_to_split_on
        
        # Create a splitter
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        
        # Create a Document from the text (trying to guess if it's markdown)
        # If it doesn't look like markdown, we'll attempt to convert some common
        # header patterns to markdown format
        if not re.search(r'^#+\s', text, re.MULTILINE):
            # This doesn't look like markdown, try to convert common header patterns
            # Like all-caps lines followed by newlines, or numbered sections
            processed_text = text
            
            # Convert ALL CAPS LINES to markdown headers
            processed_text = re.sub(
                r'^([A-Z][A-Z\s]+[A-Z])$', 
                r'# \1', 
                processed_text, 
                flags=re.MULTILINE
            )
            
            # Convert numbered sections like "1. Section Title" to markdown headers
            processed_text = re.sub(
                r'^(\d+\.)\s+([A-Z][^\n]+)$', 
                r'## \2', 
                processed_text, 
                flags=re.MULTILINE
            )
            
            # Convert subsections like "1.1 Subsection Title" to markdown headers
            processed_text = re.sub(
                r'^(\d+\.\d+)\s+([A-Z][^\n]+)$', 
                r'### \2', 
                processed_text, 
                flags=re.MULTILINE
            )
            
            document = Document(text=processed_text)
        else:
            document = Document(text=text)
        
        # Split the document - this returns documents, not nodes
        documents = splitter.split_nodes([document])
        
        # Convert documents to Chunk objects
        for i, doc in enumerate(documents):
            chunk_text = doc.text
            
            # Extract header info from metadata
            metadata = dict(doc.metadata) if hasattr(doc, 'metadata') else {}
            
            # Count tokens and characters
            token_count = count_tokens(chunk_text)
            char_count = len(chunk_text)
            
            # Create Chunk object
            chunk = Chunk(
                chunk_text=chunk_text,
                metadata=metadata,
                char_count=char_count,
                token_count=token_count,
                chunk_id=f"markdown-{i}"
            )
            
            chunks.append(chunk)
            
    except Exception as e:
        logging.error(f"Error in markdown header chunking: {e}")
        raise ValueError(f"Failed to chunk text with markdown header splitter: {e}")
    
    return chunks

# Mapping of chunking strategy names to chunker functions
CHUNKER_FUNCTIONS = {
    ChunkingStrategy.SIMPLE_RECURSIVE_SPLITTER.name: chunk_with_simple_recursive,
    ChunkingStrategy.SEMANTIC_CHUNKING_LLAMAINDEX.name: chunk_with_semantic,
    ChunkingStrategy.MARKDOWN_HEADER_SPLITTER.name: chunk_with_markdown_header,
}

def get_chunker_function(strategy_key: str) -> Callable:
    """
    Get the chunker function for a given strategy key.
    
    Args:
        strategy_key: Name of the chunking strategy
        
    Returns:
        function: The chunker function
    """
    if strategy_key not in CHUNKER_FUNCTIONS:
        raise ValueError(f"Unknown chunking strategy: {strategy_key}")
    
    return CHUNKER_FUNCTIONS[strategy_key]

def calculate_chunk_statistics(chunks: List[Chunk]) -> Dict[str, Any]:
    """
    Calculate statistics for a list of chunks.
    
    Args:
        chunks: List of Chunk objects
        
    Returns:
        dict: Statistics about the chunks
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_length_chars": 0,
            "avg_chunk_length_tokens": 0,
            "min_chunk_length_tokens": 0,
            "max_chunk_length_tokens": 0,
            "total_chars": 0,
            "total_tokens": 0
        }
    
    total_chunks = len(chunks)
    total_chars = sum(chunk.char_count for chunk in chunks)
    total_tokens = sum(chunk.token_count for chunk in chunks)
    
    avg_chunk_length_chars = total_chars / total_chunks if total_chunks > 0 else 0
    avg_chunk_length_tokens = total_tokens / total_chunks if total_chunks > 0 else 0
    
    min_chunk_length_tokens = min(chunk.token_count for chunk in chunks) if chunks else 0
    max_chunk_length_tokens = max(chunk.token_count for chunk in chunks) if chunks else 0
    
    return {
        "total_chunks": total_chunks,
        "avg_chunk_length_chars": round(avg_chunk_length_chars, 1),
        "avg_chunk_length_tokens": round(avg_chunk_length_tokens, 1),
        "min_chunk_length_tokens": min_chunk_length_tokens,
        "max_chunk_length_tokens": max_chunk_length_tokens,
        "total_chars": total_chars,
        "total_tokens": total_tokens
    }