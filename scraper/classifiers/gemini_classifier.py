# scraper/classifiers/gemini_classifier.py
"""
Gemini-based document classifier for Medicaid audit detection.
"""

import os
import json
from typing import Optional
import google.generativeai as genai
from rich.console import Console

from .base import ClassifierInterface, ClassificationResult

console = Console()


class GeminiClassifier(ClassifierInterface):
    """Gemini-based classifier for Medicaid audit documents."""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        """Initialize Gemini classifier with specified model."""
        self.model_name = model
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            console.print("[red]Warning: GOOGLE_API_KEY not found in environment[/red]")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(model)
            except Exception as e:
                console.print(f"[red]Failed to initialize Gemini client: {e}[/red]")
                self.model = None
    
    def is_available(self) -> bool:
        """Check if Gemini classifier is available."""
        return self.model is not None and self.api_key is not None
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return "Gemini"
    
    def classify_document(self, title: str, snippet: str = "", url: str = "") -> ClassificationResult:
        """
        Classify document using Gemini API.
        
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
                reasoning="Gemini API not available",
                success=False,
                error="Missing GOOGLE_API_KEY",
                provider="Gemini"
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
            if not self.model:
                return ClassificationResult(
                    is_medicaid_audit=False,
                    confidence=0.0,
                    document_type="unknown",
                    reasoning="Gemini model not initialized",
                    success=False,
                    error="Model not available",
                    provider="Gemini"
                )
                
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 200,
                }
            )
            
            # Debug the response
            if not response or not hasattr(response, 'text'):
                return ClassificationResult(
                    is_medicaid_audit=False,
                    confidence=0.0,
                    document_type="unknown",
                    reasoning="Empty response from Gemini",
                    success=False,
                    error="Empty response",
                    provider="Gemini"
                )
            
            response_text = response.text
            if not response_text or response_text.strip() == "":
                return ClassificationResult(
                    is_medicaid_audit=False,
                    confidence=0.0,
                    document_type="unknown",
                    reasoning="Empty response text from Gemini",
                    success=False,
                    error="Empty response text",
                    provider="Gemini"
                )
            
            # Clean the response - sometimes AI adds extra text before/after JSON
            response_text = response_text.strip()
            
            # Try to find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return ClassificationResult(
                    is_medicaid_audit=False,
                    confidence=0.0,
                    document_type="unknown",
                    reasoning=f"No JSON in response: {response_text[:50]}",
                    success=False,
                    error=f"No JSON found in response: {response_text[:100]}",
                    provider="Gemini"
                )
            
            json_text = response_text[json_start:json_end]
            
            # Parse the JSON response
            result_data = json.loads(json_text)
            
            return ClassificationResult(
                is_medicaid_audit=result_data.get("is_medicaid_audit", False),
                confidence=float(result_data.get("confidence", 0.0)),
                document_type=result_data.get("document_type", "unknown"),
                reasoning=result_data.get("reasoning", "No reasoning provided"),
                success=True,
                error=None,
                provider="Gemini"
            )
            
        except json.JSONDecodeError as e:
            console.print(f"[red]Gemini JSON decode error: {e}[/red]")
            if hasattr(response, 'text'):
                console.print(f"[red]Response was: {response.text[:200]}[/red]")
            return ClassificationResult(
                is_medicaid_audit=False,
                confidence=0.0,
                document_type="unknown",
                reasoning=f"JSON parse error: {str(e)}",
                success=False,
                error=f"JSON decode error: {str(e)}",
                provider="Gemini"
            )
            
        except Exception as e:
            console.print(f"[red]Gemini classification error: {e}[/red]")
            return ClassificationResult(
                is_medicaid_audit=False,
                confidence=0.0,
                document_type="unknown",
                reasoning=f"Classification failed: {str(e)}",
                success=False,
                error=str(e),
                provider="Gemini"
            )