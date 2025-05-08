import logging
import re
import os
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal, Callable
import tiktoken
from pydantic import BaseModel, Field

# Define placeholders for types that might not be imported
# This helps with static analysis and prevents NameErrors if imports fail
Document, TokenTextSplitter, SentenceSplitter, MarkdownNodeParser, SemanticSplitterNodeParser, OpenAIEmbedding = (None,) * 6
BaseEmbedding = None

# Track availability of components
LLAMA_INDEX_AVAILABLE = False
OPENAI_EMBEDDING_AVAILABLE = False

# Try importing LlamaIndex core components
try:
    from llama_index.core.node_parser import (
        TokenTextSplitter as CoreTokenTextSplitter,
        SentenceSplitter as CoreSentenceSplitter,
        MarkdownNodeParser as CoreMarkdownNodeParser,
        SemanticSplitterNodeParser as CoreSemanticSplitterNodeParser
    )
    from llama_index.core.schema import Document as CoreDocument
    
    # Assign imported classes to the global names
    Document = CoreDocument
    TokenTextSplitter = CoreTokenTextSplitter
    SentenceSplitter = CoreSentenceSplitter
    MarkdownNodeParser = CoreMarkdownNodeParser
    SemanticSplitterNodeParser = CoreSemanticSplitterNodeParser
    
    LLAMA_INDEX_AVAILABLE = True
    logging.info("LlamaIndex core components successfully imported")
    
    # Try importing OpenAI embedding model for semantic chunking
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding as ImportedOpenAIEmbedding
        OpenAIEmbedding = ImportedOpenAIEmbedding
        OPENAI_EMBEDDING_AVAILABLE = True
        logging.info("OpenAIEmbedding successfully imported")
    except ImportError:
        logging.warning(
            "OpenAIEmbedding could not be imported from llama_index.embeddings.openai. "
            "Semantic chunking with OpenAI embeddings will not be available. "
            "Ensure 'llama-index-embeddings-openai' is installed."
        )
except ImportError as e:
    logging.warning(f"LlamaIndex not available: {str(e)}. Some chunking strategies will not be available.")
    # All component placeholders remain as None

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
    max_chunk_size: int = Field(default=512, gt=0, description="Maximum chunk size in tokens - chunks may be smaller based on semantic breakpoints")
    breakpoint_percentile_threshold: int = Field(
        default=95, ge=0, le=100,
        description="Percentile threshold for determining breakpoints (higher = fewer but more significant breakpoints)"
    )

class MarkdownSplitterParams(BaseModel):
    """Parameters for Markdown node parser."""
    strategy_type: Literal["MARKDOWN_NODE_PARSER"] = "MARKDOWN_NODE_PARSER"
    chunk_size: int = Field(default=512, gt=0, description="Target chunk size in tokens")

# Chunking Strategy Enum
class ChunkingStrategy(Enum):
    """
    Enum of available chunking strategies.
    Each strategy has a display name and associated parameter model.
    """
    SIMPLE_RECURSIVE_SPLITTER = ("Simple Recursive Splitter (LlamaIndex)", SimpleSplitterParams)
    SEMANTIC_CHUNKING_LLAMAINDEX = ("Semantic Chunking (LlamaIndex)", SemanticSplitterParams)
    MARKDOWN_NODE_PARSER = ("Markdown Parser", MarkdownSplitterParams)  # Renamed from MARKDOWN_HEADER_SPLITTER

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
        
        # Filter based on availability
        if not LLAMA_INDEX_AVAILABLE:
            # If LlamaIndex is not available, remove all LlamaIndex-based strategies
            choices = [choice for choice in choices if not choice[0].endswith("LLAMAINDEX")]
        elif not OPENAI_EMBEDDING_AVAILABLE:
            # If LlamaIndex is available but OpenAI embeddings are not, 
            # remove just the semantic chunker which requires embeddings
            choices = [choice for choice in choices if choice[0] != "SEMANTIC_CHUNKING_LLAMAINDEX"]
        
        # Ensure we always have at least one choice
        if not choices:
            logging.warning("No chunking strategies are available. Check your dependencies.")
        
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
    
    # Check if OpenAIEmbedding specifically is available (imported correctly)
    if not OPENAI_EMBEDDING_AVAILABLE or OpenAIEmbedding is None:
        raise ImportError(
            "OpenAIEmbedding is required for semantic chunking but is not available. "
            "Please ensure 'llama-index-embeddings-openai' is installed."
        )
    
    chunks = []
    
    try:
        # Create a Document from the text
        document = Document(text=text)
        
        # 1. Initialize OpenAI Embedding Model
        try:
            # Check for API key
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is not set. "
                    "This is required for semantic chunking with OpenAI embeddings."
                )
            
            embed_model = OpenAIEmbedding()
            logging.info("Successfully initialized OpenAI embedding model")
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI embedding model: {e}")
            raise ValueError(
                f"Failed to initialize OpenAI embedding model: {e}. "
                "Please check your API key and model configuration."
            )
        
        # 2. Create a sentence splitter for pre-segmenting text
        initial_sentence_splitter = SentenceSplitter(
            chunk_size=params.max_chunk_size
        )
        
        # 3. Create semantic splitter with proper configuration
        splitter = SemanticSplitterNodeParser(
            embed_model=embed_model,
            breakpoint_percentile_threshold=params.breakpoint_percentile_threshold,  # Pass as integer (e.g., 95)
            sentence_splitter=initial_sentence_splitter.split_text
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
                "breakpoint_percentile": params.breakpoint_percentile_threshold,
                "max_chunk_size": params.max_chunk_size
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

def chunk_with_markdown_parser(text: str, params: MarkdownSplitterParams) -> List[Chunk]:
    """
    Chunk text using LlamaIndex's Markdown node parser.
    
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
        
        # Create a MarkdownNodeParser
        parser = MarkdownNodeParser()
        
        # Parse the document into nodes
        nodes = parser.get_nodes_from_documents([document])
        
        # Convert nodes to Chunk objects
        for i, node in enumerate(nodes):
            chunk_text = node.text
            
            # Extract metadata from the node
            metadata = dict(node.metadata) if hasattr(node, 'metadata') else {}
            
            # Add additional metadata
            metadata.update({
                "chunk_index": i,
                "parser": "markdown"
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
                chunk_id=f"markdown-{i}"
            )
            
            chunks.append(chunk)
            
    except Exception as e:
        logging.error(f"Error in markdown parsing: {e}")
        raise ValueError(f"Failed to chunk text with markdown parser: {e}")
    
    return chunks

# Mapping of chunking strategy names to chunker functions
CHUNKER_FUNCTIONS = {
    ChunkingStrategy.SIMPLE_RECURSIVE_SPLITTER.name: chunk_with_simple_recursive,
    ChunkingStrategy.SEMANTIC_CHUNKING_LLAMAINDEX.name: chunk_with_semantic,
    ChunkingStrategy.MARKDOWN_NODE_PARSER.name: chunk_with_markdown_parser,
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