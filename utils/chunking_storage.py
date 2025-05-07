import logging
from typing import Dict, List, Any, Optional
from utils.comparison_storage import ComparisonStorage
from utils.chunking_strategies import Chunk

class ChunkingComparisonStorage(ComparisonStorage):
    """
    Extension of ComparisonStorage for chunking comparison results.
    """
    
    def __init__(self, app):
        """
        Initialize the chunking comparison storage.
        
        Args:
            app: Flask app instance
        """
        super().__init__(app)
        
    def store_chunking_comparison(self, comparison_data: Dict[str, Any]) -> str:
        """
        Store chunking comparison data with a unique ID and expiration time.
        
        Args:
            comparison_data: Dictionary containing chunking comparison results
            
        Returns:
            str: Unique ID for the stored comparison
        """
        # Convert Chunk objects to dictionaries for storage
        if 'chunks_1' in comparison_data and isinstance(comparison_data['chunks_1'], list):
            comparison_data['chunks_1'] = [
                chunk.dict() if isinstance(chunk, Chunk) else chunk 
                for chunk in comparison_data['chunks_1']
            ]
            
        if 'chunks_2' in comparison_data and isinstance(comparison_data['chunks_2'], list):
            comparison_data['chunks_2'] = [
                chunk.dict() if isinstance(chunk, Chunk) else chunk 
                for chunk in comparison_data['chunks_2']
            ]
        
        # Store with "chunk_" prefix to distinguish from other comparison types
        comparison_id = super().store_comparison(comparison_data)
        return f"chunk_{comparison_id}"
    
    def get_chunking_comparison(self, comparison_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve chunking comparison data by ID.
        
        Args:
            comparison_id: Unique ID for the stored comparison
            
        Returns:
            dict: Chunking comparison data, or None if not found or expired
        """
        # Remove "chunk_" prefix if present
        actual_id = comparison_id
        if comparison_id.startswith("chunk_"):
            actual_id = comparison_id[6:]
            
        data = super().get_comparison(actual_id)
        
        # If we found data, convert dictionaries back to Chunk objects if needed
        if data:
            # We might want to convert back to Chunk objects here,
            # but for now we'll just use the dictionaries directly in the frontend
            pass
            
        return data