# scraper/classifier.py
"""
AI-powered classification for identifying Medicaid audit documents using Google Gemini.
"""
import os
from typing import Dict, List, Any
import google.generativeai as genai
from dotenv import load_dotenv
import json
from rich.console import Console

load_dotenv()
console = Console()


class MedicaidAuditClassifier:
    """Classify documents as likely Medicaid audits using AI."""
    
    def __init__(self):
        """Initialize with Gemini client."""
        api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        if not api_key:
            # Try alternate env var name
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_GEMINI_API_KEY or GEMINI_API_KEY in .env file")
        
        # Configure the API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def classify_from_summary(self, title: str, snippet: str, url: str, source: str, 
                         metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Classify based on summary information including metadata.
        Returns:
            Dict containing classification results
        """ 
        # Extract metadata fields
        author = ""
        creation_date = ""
        subject = ""
        
        if metadata:
            author = metadata.get("author", "")
            creation_date = metadata.get("creation_date", "")
            subject = metadata.get("subject", "")
        
        prompt = f"""Analyze if this is a Medicaid audit report based on the available information.

    Title: {title}
    Snippet: {snippet}
    URL: {url}
    Source Domain: {source}
    Document Author: {author if author else "Not available"}
    Creation Date: {creation_date if creation_date else "Not available"}
    Document Subject: {subject if subject else "Not available"}

    Determine if this is specifically a Medicaid audit report (not a provider manual, guide, form, or general policy document).

    Consider these factors:
    1. Does the title/snippet indicate this is an audit, review, or investigation?
    2. Is Medicaid the primary focus (not just mentioned)?
    3. Does it appear to be from an authoritative audit source (auditor general, OIG, GAO)?
    4. Are there indicators of audit content (findings, recommendations, compliance review)?
    5. Does the author field indicate an audit office or inspector general?
    6. Does the document subject suggest audit-related content?

    Respond with ONLY valid JSON in this exact format:
    {{
        "is_medicaid_audit": true or false,
        "confidence": 0.0 to 1.0,
        "document_type": "audit_report" or "manual" or "guide" or "form" or "policy" or "other",
        "reasoning": "Brief explanation of your determination"
    }}"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 200,
                }
            )
            
            # Parse the JSON response
            result = json.loads(response.text)
            return result
            
        except Exception as e:
            console.print(f"[red]AI classification error: {e}[/red]")
            return {
                "is_medicaid_audit": False,
                "confidence": 0.0,
                "document_type": "unknown",
                "reasoning": f"Classification failed: {str(e)}"
            }

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
            
            # Pass metadata to classification
            classification = self.classify_from_summary(
                title=result.get('title', ''),
                snippet=result.get('snippet', ''),
                url=result.get('url', ''),
                source=result.get('source', ''),
                metadata=result.get('metadata', {})  # Pass the metadata
            )
            
            # Add classification with specific field name
            result_copy = result.copy()
            result_copy['ai_classification'] = classification
            classified_results.append(result_copy)
        
        return classified_results