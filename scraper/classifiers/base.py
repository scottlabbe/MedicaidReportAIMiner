# scraper/classifiers/base.py
"""
Base interface for AI classifiers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ClassificationResult:
    """Result of AI classification with error handling."""
    is_medicaid_audit: bool
    confidence: float
    document_type: str
    reasoning: str
    success: bool = True
    error: Optional[str] = None
    provider: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "is_medicaid_audit": self.is_medicaid_audit,
            "confidence": self.confidence,
            "document_type": self.document_type,
            "reasoning": self.reasoning,
            "success": self.success,
            "error": self.error,
            "provider": self.provider
        }


class ClassifierInterface(ABC):
    """Abstract base class for AI document classifiers."""
    
    @abstractmethod
    def classify_document(self, title: str, snippet: str = "", url: str = "") -> ClassificationResult:
        """
        Classify a document as a Medicaid audit or not.
        
        Args:
            title: Document title
            snippet: Document snippet/description
            url: Document URL
            
        Returns:
            ClassificationResult with classification details
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of the AI provider (e.g., 'OpenAI', 'Gemini')."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the classifier is available (API keys, etc.)."""
        pass