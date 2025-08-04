# scraper/classifiers/openai_classifier.py
"""
OpenAI-based document classifier for Medicaid audit detection.
"""

import os
import json
import httpx
from typing import Optional
from openai import OpenAI
from rich.console import Console

from .base import ClassifierInterface, ClassificationResult

console = Console()


class OpenAIClassifier(ClassifierInterface):
    """OpenAI-based classifier for Medicaid audit documents."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize OpenAI classifier with specified model."""
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            console.print("[red]Warning: OPENAI_API_KEY not found in environment[/red]")
            self.client = None
        else:
            try:
                # Configure timeout to prevent SSL read timeouts
                timeout = httpx.Timeout(
                    120.0,        # Total timeout
                    connect=10.0, # Connection timeout
                    read=60.0,    # Read timeout (prevents SSL timeout)
                    write=5.0     # Write timeout
                )
                self.client = OpenAI(api_key=self.api_key, timeout=timeout)
            except Exception as e:
                console.print(f"[red]Failed to initialize OpenAI client: {e}[/red]")
                self.client = None
    
    def is_available(self) -> bool:
        """Check if OpenAI classifier is available."""
        return self.client is not None and self.api_key is not None
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return "OpenAI"
    
    def classify_document(self, title: str, snippet: str = "", url: str = "") -> ClassificationResult:
        """
        Classify document using OpenAI API.
        
        Args:
            title: Document title
            snippet: Document snippet/description  
            url: Document URL
            
        Returns:
            ClassificationResult with classification details
        """
        if not self.is_available():
            return ClassificationResult(
                is_medicaid_audit=False,
                confidence=0.0,
                document_type="unknown",
                reasoning="OpenAI API not available",
                success=False,
                error="Missing OPENAI_API_KEY",
                provider="OpenAI"
            )
        
        # Build prompt
        prompt = f"""Analyze this document and determine if it's a legitimate Medicaid audit report.

Document Information:
- Title: {title}
- Snippet: {snippet or "No snippet available"}
- URL: {url or "No URL available"}

Classification Criteria:
- A Medicaid audit report contains findings, recommendations, or analysis of Medicaid program operations
- It should NOT be: manuals, guides, forms, policies, newsletters, or general healthcare documents
- Look for audit-specific language like "findings", "recommendations", "deficiencies", "compliance"

Respond with JSON in this exact format:
{{
    "is_medicaid_audit": true/false,
    "confidence": 0.0-1.0,
    "document_type": "audit_report" or "manual" or "guide" or "form" or "policy" or "other", 
    "reasoning": "Brief explanation of your determination"
}}"""

        try:
            if not self.client:
                return ClassificationResult(
                    is_medicaid_audit=False,
                    confidence=0.0,
                    document_type="unknown",
                    reasoning="OpenAI client not initialized",
                    success=False,
                    error="Client not available",
                    provider="OpenAI"
                )
                
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a document classification expert. Analyze documents to determine if they are Medicaid audit reports. Respond only with valid JSON."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=200
            )
            
            if not response.choices or not response.choices[0].message.content:
                return ClassificationResult(
                    is_medicaid_audit=False,
                    confidence=0.0,
                    document_type="unknown",
                    reasoning="Empty response from OpenAI",
                    success=False,
                    error="Empty response",
                    provider="OpenAI"
                )
            
            # Parse JSON response
            result_text = response.choices[0].message.content.strip()
            result_data = json.loads(result_text)
            
            return ClassificationResult(
                is_medicaid_audit=result_data.get("is_medicaid_audit", False),
                confidence=float(result_data.get("confidence", 0.0)),
                document_type=result_data.get("document_type", "unknown"),
                reasoning=result_data.get("reasoning", "No reasoning provided"),
                success=True,
                error=None,
                provider="OpenAI"
            )
            
        except json.JSONDecodeError as e:
            console.print(f"[red]OpenAI JSON decode error: {e}[/red]")
            return ClassificationResult(
                is_medicaid_audit=False,
                confidence=0.0,
                document_type="unknown",
                reasoning=f"JSON parse error: {str(e)}",
                success=False,
                error=f"JSON decode error: {str(e)}",
                provider="OpenAI"
            )
            
        except Exception as e:
            console.print(f"[red]OpenAI classification error: {e}[/red]")
            return ClassificationResult(
                is_medicaid_audit=False,
                confidence=0.0,
                document_type="unknown",
                reasoning=f"Classification failed: {str(e)}",
                success=False,
                error=str(e),
                provider="OpenAI"
            )