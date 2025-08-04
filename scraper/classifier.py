# scraper/classifier.py
"""
AI-powered classification of search results to identify legitimate Medicaid audit documents.
"""
import yaml
from typing import Dict, List, Any
from rich.console import Console
from dotenv import load_dotenv

from .classifiers import ClassifierInterface, ClassificationResult, OpenAIClassifier, GeminiClassifier

# Load environment variables
load_dotenv()

console = Console()


class MedicaidAuditClassifier:
    """AI classifier to identify legitimate Medicaid audit documents from search results."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the classifier with configuration."""
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        classifier_config = self.config.get('classifier', {})
        self.provider = classifier_config.get('provider', 'openai')
        self.model = classifier_config.get('model', 'gpt-4.1-nano')
        self.show_errors = classifier_config.get('show_errors', True)
        self.retry_attempts = classifier_config.get('retry_attempts', 2)
        
        # Initialize the selected classifier
        self.classifier = self._create_classifier()
        
        if not self.classifier.is_available():
            console.print(f"[yellow]Warning: {self.provider} classifier not available, attempting fallback[/yellow]")
            # Try the other provider as fallback
            fallback_provider = "gemini" if self.provider == "openai" else "openai"
            fallback_classifier = self._create_classifier(fallback_provider)
            if fallback_classifier.is_available():
                console.print(f"[green]Using {fallback_provider} as fallback classifier[/green]")
                self.classifier = fallback_classifier
                self.provider = fallback_provider
            else:
                console.print("[red]No AI classifiers available![/red]")
    
    def _create_classifier(self, provider: str = None) -> ClassifierInterface:
        """Create a classifier instance based on provider."""
        provider = provider or self.provider
        
        if provider.lower() == "openai":
            model = self.model if self.model.startswith("gpt") else "gpt-4.1-nano"
            return OpenAIClassifier(model)
        elif provider.lower() == "gemini":
            model = self.model if self.model.startswith("gemini") else "gemini-1.5-flash"
            return GeminiClassifier(model)
        else:
            raise ValueError(f"Unknown classifier provider: {provider}")
    
    def classify_document(self, title: str, snippet: str = "", url: str = "") -> dict:
        """
        Classify a document as a Medicaid audit or not using AI.
        
        Args:
            title: Document title
            snippet: Document snippet/description
            url: Document URL
            
        Returns:
            dict: Classification result with confidence score and error info
        """
        result = self._classify_with_retry(title, snippet, url)
        
        # Convert to legacy dict format for backward compatibility
        return result.to_dict()
    
    def _classify_with_retry(self, title: str, snippet: str = "", url: str = "") -> ClassificationResult:
        """Classify with retry logic."""
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                result = self.classifier.classify_document(title, snippet, url)
                
                if result.success:
                    return result
                else:
                    last_error = result.error
                    if self.show_errors:
                        console.print(f"[yellow]Classification attempt {attempt + 1} failed: {result.error}[/yellow]")
                    
            except Exception as e:
                last_error = str(e)
                if self.show_errors:
                    console.print(f"[red]Classification attempt {attempt + 1} error: {e}[/red]")
        
        # All attempts failed
        return ClassificationResult(
            is_medicaid_audit=False,
            confidence=0.0,
            document_type="unknown",
            reasoning=f"All {self.retry_attempts} attempts failed. Last error: {last_error}",
            success=False,
            error=f"Failed after {self.retry_attempts} attempts: {last_error}",
            provider=self.classifier.get_provider_name()
        )
    
    def get_status(self) -> dict:
        """Get current classifier status."""
        return {
            "provider": self.provider,
            "model": self.model,
            "available": self.classifier.is_available(),
            "provider_name": self.classifier.get_provider_name()
        }
    
    def classify_from_summary(self, title: str, snippet: str, url: str, source: str, 
                         metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility - uses the new classifier system.
        """
        # Use the new classification method
        return self.classify_document(title, snippet, url)

    def classify_batch(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify multiple search results using summary data.
        
        Args:
            search_results: List of search result dicts
            
        Returns:
            Same list with 'ai_classification' field added to each result
        """
        classified_results = []
        
        console.print(f"\n[bold cyan]Classifying {len(search_results)} results with Gemini AI...[/bold cyan]")
        
        for idx, result in enumerate(search_results):
            console.print(f"  Analyzing [{idx + 1}/{len(search_results)}]: {result['title'][:50]}...")
            
            # Use the new classification method
            classification = self.classify_document(
                title=result.get('title', ''),
                snippet=result.get('snippet', ''),
                url=result.get('url', '')
            )
            
            # Add classification with specific field name
            result_copy = result.copy()
            result_copy['ai_classification'] = classification
            classified_results.append(result_copy)
        
        return classified_results