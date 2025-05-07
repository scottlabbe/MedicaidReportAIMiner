import time
import uuid
import logging
from datetime import datetime, timedelta

# Default expiration time for comparison data (30 minutes)
DEFAULT_EXPIRATION_SECONDS = 30 * 60

class ComparisonStorage:
    """
    Temporary storage for PDF parser comparison results.
    This class manages the storage and retrieval of comparison data,
    with automatic expiration of old entries.
    """
    
    def __init__(self, app):
        """
        Initialize the comparison storage.
        
        Args:
            app: Flask app instance
        """
        self.app = app
        
        # Create a storage key in the app config if it doesn't exist
        if 'comparison_data' not in app.config:
            app.config['comparison_data'] = {}
    
    def store_comparison(self, comparison_data):
        """
        Store comparison data with a unique ID and expiration time.
        
        Args:
            comparison_data: Dictionary containing comparison results
            
        Returns:
            str: Unique ID for the stored comparison
        """
        # Generate a unique ID
        comparison_id = str(uuid.uuid4())
        
        # Set expiration time
        expiration_time = time.time() + DEFAULT_EXPIRATION_SECONDS
        
        # Store the data with expiration time
        self.app.config['comparison_data'][comparison_id] = {
            'data': comparison_data,
            'expires_at': expiration_time,
            'created_at': datetime.now().isoformat()
        }
        
        # Clean up expired entries
        self._cleanup_expired()
        
        return comparison_id
    
    def get_comparison(self, comparison_id):
        """
        Retrieve comparison data by ID.
        
        Args:
            comparison_id: Unique ID for the stored comparison
            
        Returns:
            dict: Comparison data, or None if not found or expired
        """
        # Clean up expired entries first
        self._cleanup_expired()
        
        # Check if the ID exists
        if comparison_id not in self.app.config['comparison_data']:
            return None
        
        # Return the comparison data
        return self.app.config['comparison_data'][comparison_id]['data']
    
    def _cleanup_expired(self):
        """
        Remove expired comparison data entries.
        """
        current_time = time.time()
        expired_ids = []
        
        # Find expired entries
        for comparison_id, entry in self.app.config['comparison_data'].items():
            if entry['expires_at'] < current_time:
                expired_ids.append(comparison_id)
        
        # Remove expired entries
        for comparison_id in expired_ids:
            del self.app.config['comparison_data'][comparison_id]
            logging.info(f"Removed expired comparison data: {comparison_id}")