import os
import time
import logging
import json
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from utils.token_usage_logger import TokenUsageLogger

# Using Gemini 2.5 Flash model for structured output
GEMINI_MODEL = "gemini-2.5-flash"

class ReportData(BaseModel):
    """Pydantic model for structured audit report data extraction - compatible with existing database schema"""
    report_title: str = Field(..., description="The full title of the audit report, converted to standard title case (e.g., 'Annual Audit Report') even if it appears in all caps in the source.")
    audit_organization: str = Field(..., description="The organization that conducted the audit")
    publication_year: int = Field(..., description="The year the report was published (4-digit)")
    publication_month: int = Field(..., description="The month the report was published (1-12)")
    publication_day: Optional[int] = Field(None, description="The day the report was published (1-31), if available")
    objectives: List[str] = Field(default_factory=list, description="List of distinct audit objective texts. Each objective should be a separate string in the list.")
    findings: List[str] = Field(default_factory=list, description="List of distinct audit finding texts. Each finding should be a separate string in the list.")
    recommendations: List[str] = Field(default_factory=list, description="List of distinct audit recommendation texts. Each recommendation should be a separate string in the list.")
    overall_conclusion: Optional[str] = Field(None, description="The overall conclusion of the audit report")
    llm_insight: str = Field(..., description="An AI-generated summary/insight about the report")
    potential_objective_summary: Optional[str] = Field(None, description="An AI-generated summary of the objectives")
    original_report_source_url: Optional[str] = Field(None, description="URL to the original report, if available")
    state: str = Field(..., description="The US state code related to the report (e.g., 'NY', 'CA'). Use 'US' for federal agencies and nationwide reports.")
    audit_scope: str = Field(..., description="The scope of the audit, including only the time period.")
    extracted_keywords: List[str] = Field(default_factory=list, description="Relevant keywords extracted from the report content")

class AIExtractionLog(BaseModel):
    """Log entry for AI extraction operations - compatible with existing database schema"""
    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    processing_time_ms: int
    extraction_status: str
    error_details: Optional[str] = None

def extract_data_with_gemini(pdf_text: str, api_key: str) -> tuple[ReportData, AIExtractionLog]:
    """
    Extract structured data from PDF text using Gemini's native structured output.
    
    Args:
        pdf_text: Text content of the PDF
        api_key: Gemini API key
        
    Returns:
        tuple: (ReportData object, AIExtractionLog object)
    """
    start_time = time.time()
    
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        
        # Prepare the prompt for audit report extraction
        prompt = f"""
        You are an AI assistant specialized in extracting structured information from Medicaid audit reports.
        Your task is to extract specific data points from the provided report text and format them according to the specified schema.
        Focus on accuracy and be as detailed as possible. If some information is not present in the text, leave those fields empty or null.
        
        Important guidelines:
        - For audit_scope, provide a comprehensive text description of the audit's scope including time periods, programs examined, departments audited, or any other contextual information.
        - Always generate a potential_objective_summary that summarizes the main objectives of the audit in a concise paragraph.
        - Convert report titles to proper title case even if they appear in all caps.
        - Extract all distinct findings, recommendations, and objectives as separate list items.
        - Identify the US state code or use 'US' for federal reports.
        - Generate relevant keywords that would help with searching and categorization.
        
        Here is the audit report text to analyze:
        
        {pdf_text}
        """
        
        # Generate structured content using Gemini's native Pydantic support
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": ReportData,
            }
        )
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        # Extract the parsed Pydantic object
        # Note: response.parsed may return BaseModel, need to ensure it's ReportData type
        if response.parsed and isinstance(response.parsed, ReportData):
            report_data = response.parsed
        else:
            # Fallback: parse from JSON text if parsed object is not available
            import json
            data = json.loads(response.text) if response.text else {}
            report_data = ReportData(**data)
        
        # Add fallback validation for required fields to prevent data loss
        if not report_data.state:
            report_data.state = "US"  # Default fallback
            logging.warning("State field was empty, using fallback value 'US'")
        
        if not report_data.audit_scope:
            report_data.audit_scope = "Audit scope not specified in source document"
            logging.warning("Audit scope field was empty, using fallback value")
        
        # Get actual token usage from Gemini response
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            # Use actual token counts from Gemini
            usage = response.usage_metadata
            input_tokens = usage.prompt_token_count if usage.prompt_token_count else 0
            output_tokens = usage.candidates_token_count if usage.candidates_token_count else 0
            total_tokens = usage.total_token_count if usage.total_token_count else (input_tokens + output_tokens)
            
            # Log initial token counts
            TokenUsageLogger.log_extraction(
                provider="Gemini",
                model=GEMINI_MODEL,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                input_cost=0,  # Will be calculated below
                output_cost=0,  # Will be calculated below
                total_cost=0,  # Will be calculated below
                processing_time_ms=0,  # Will be set below
                status="IN_PROGRESS",
                report_title=report_data.report_title
            )
            
            # Log cached content tokens if present
            if usage.cached_content_token_count:
                logging.info(f"Cached content tokens: {usage.cached_content_token_count}")
        else:
            # Fallback to estimates only if usage metadata is not available
            logging.warning("Could not get actual token usage from Gemini, using estimates")
            input_tokens = estimate_tokens(pdf_text)
            output_tokens = estimate_tokens(response.text if response.text else "")
            total_tokens = input_tokens + output_tokens
        
        # Calculate costs based on Gemini pricing
        # Gemini 2.5 Flash pricing: $0.075 per 1M input tokens, $0.30 per 1M output tokens
        # Updated pricing as of 2024/2025
        input_cost = input_tokens * 0.000000075  # $0.075 per 1M input tokens
        output_cost = output_tokens * 0.0000003   # $0.30 per 1M output tokens
        total_cost = input_cost + output_cost
        
        # Create extraction log
        extraction_log = AIExtractionLog(
            model_name=GEMINI_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            processing_time_ms=processing_time,
            extraction_status="success",
            error_details=None
        )
        
        # Log successful extraction with full details
        TokenUsageLogger.log_extraction(
            provider="Gemini",
            model=GEMINI_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            processing_time_ms=processing_time,
            status="success",
            report_title=report_data.report_title
        )
        
        # Log estimation accuracy if we have actual usage
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            estimated_input = len(pdf_text) // 4
            TokenUsageLogger.log_token_estimation_accuracy(estimated_input, input_tokens, "input")
        
        return report_data, extraction_log
        
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        error_message = str(e)
        
        logging.error(f"Error extracting data with Gemini: {error_message}")
        
        # Log the failure
        TokenUsageLogger.log_extraction(
            provider="Gemini",
            model=GEMINI_MODEL,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
            processing_time_ms=processing_time,
            status="failed",
            error=error_message
        )
        
        # Create error log
        extraction_log = AIExtractionLog(
            model_name=GEMINI_MODEL,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
            processing_time_ms=processing_time,
            extraction_status="failed",
            error_details=error_message
        )
        
        # Return minimal report data with fallbacks to prevent data loss
        fallback_report = ReportData(
            report_title="Failed to extract title",
            audit_organization="Failed to extract organization", 
            publication_year=datetime.now().year,
            publication_month=1,
            publication_day=None,
            overall_conclusion=None,
            llm_insight="Extraction failed - please review manually",
            potential_objective_summary=None,
            original_report_source_url=None,
            state="US",  # Required field fallback
            audit_scope="Extraction failed - please review manually",  # Required field fallback
            extracted_keywords=[]
        )
        
        return fallback_report, extraction_log

def estimate_tokens(text: str) -> int:
    """
    Fallback token estimation for Gemini models when actual usage is not available.
    This is only used as a last resort when response.usage_metadata is not accessible.
    """
    if not text:
        return 0
    # Rough approximation: 1 token ≈ 4 characters for English text
    return len(text) // 4

def test_gemini_extraction():
    """Test function to verify Gemini structured output works correctly"""
    test_text = """
    MEDICAID ELIGIBILITY AUDIT REPORT
    
    State of California Department of Health Services
    Audit Period: January 2023 - December 2023
    
    OBJECTIVES:
    1. Review Medicaid eligibility determination processes
    2. Assess compliance with federal regulations
    
    FINDINGS:
    1. 15% of applications processed beyond regulatory timeframe
    2. Documentation deficiencies identified in 230 cases
    
    RECOMMENDATIONS:
    1. Implement automated processing system
    2. Enhance staff training programs
    
    CONCLUSION:
    The audit identified significant areas for improvement in processing efficiency.
    """
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in environment")
        return False
    
    try:
        report_data, log = extract_data_with_gemini(test_text, api_key)
        print("✓ Gemini extraction test successful")
        print(f"  Title: {report_data.report_title}")
        print(f"  State: {report_data.state}")
        print(f"  Objectives count: {len(report_data.objectives)}")
        print(f"  Processing time: {log.processing_time_ms}ms")
        print(f"  Cost: ${log.total_cost:.4f}")
        return True
    except Exception as e:
        print(f"✗ Gemini extraction test failed: {e}")
        return False

if __name__ == "__main__":
    test_gemini_extraction()